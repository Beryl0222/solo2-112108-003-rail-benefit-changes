"""读通告、生成并办理改乘方案。

方案只使用通告允许的班次与上下车区间。已确认的同行条件、本人接受的上车
站、体验券规则和余位是硬约束；座位偏好、少换站等只用于排序。

一次确认对应当时展示的整套方案。确认前如果条件（转让、余位、对话）变了，
原方案失效并说明哪些条件已变；客服可重新核对并发新方案。办理成功后，同一
回复号反复查询得到同一结果。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .conversation import WITHDRAWN, Claim, Conversation
from .model import Benefit, DataError, Message, Order, Train, Workbook


# ----------------------------------------------------------------------
# 通告


@dataclass(frozen=True)
class NoticeRules:
    """从原始交通通告读到的改乘范围。"""

    suspended: frozenset[tuple[str, str]]
    alternative_trains: frozenset[str]
    added_boardings: frozenset[str]
    backup_trains: frozenset[str]
    cites: tuple[str, ...] = ()

    def permitted(self, train: Train, boarding: str, alighting: str) -> bool:
        """班次与区间都必须在通告公开的范围内。

        备用班次只按原票上车站搭乘；临时补进的站由调用方按本人意愿另行限定。
        """
        if train.id not in self.candidate_trains() or not train.accepted:
            return False
        return train.serves(boarding, alighting)

    def service_affected(self, order: Order) -> bool:
        return (order.service, order.boarding) in self.suspended

    def candidate_trains(self) -> frozenset[str]:
        return self.alternative_trains | self.backup_trains

    def describe(self) -> dict[str, Any]:
        return {
            "suspended": sorted(self.suspended),
            "alternative_trains": sorted(self.alternative_trains),
            "added_boardings": sorted(self.added_boardings),
            "backup_trains": sorted(self.backup_trains),
            "cites": list(self.cites),
        }


def read_notices(workbook: Workbook) -> NoticeRules:
    suspended: set[tuple[str, str]] = set()
    alternatives: set[str] = set()
    added: set[str] = set()
    backups: set[str] = set()
    cites: list[str] = []
    stations = workbook.stations()
    last_station: str | None = None
    for notice in sorted(workbook.notices.values(), key=lambda row: (row.at, row.id)):
        text = notice.text
        for segment in re.split(r"[。；;，,]", text):
            here = [stop for stop in stations if stop in segment]
            if here:
                last_station = max(here, key=len)
            if re.search(r"取消办客|停止办客|停办|不停靠", segment):
                services = re.findall(r"R\d+", segment)
                station = max(here, key=len) if here else last_station
                for service in services:
                    if station:
                        suspended.add((service, station))
            if "可改" in segment:
                for group in re.findall(r"标为([^。；]+?)的班次可改", segment):
                    alternatives.update(re.findall(r"R\d+", group))
                for match in re.finditer(r"备用\s*(R\d+)", segment):
                    backups.add(match.group(1))
            for match in re.finditer(r"([一-鿿]{2,4})可以(?:补进|上车)", segment):
                added.add(match.group(1))
        if alternatives or backups or added or suspended:
            cites.append(notice.id)
    return NoticeRules(
        suspended=frozenset(suspended),
        alternative_trains=frozenset(alternatives),
        added_boardings=frozenset(added),
        backup_trains=frozenset(backups),
        cites=tuple(cites),
    )


# ----------------------------------------------------------------------
# 结果对象


@dataclass(frozen=True)
class Leg:
    order: str
    person: str
    train: str
    boarding: str
    alighting: str
    changed: bool

    def key(self) -> tuple[str, str, str, str]:
        return (self.order, self.train, self.boarding, self.alighting)

    def describe(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "person": self.person,
            "train": self.train,
            "boarding": self.boarding,
            "alighting": self.alighting,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class BenefitUse:
    benefit: str
    holder: str
    rider: str
    train: str
    transfer: bool
    authorized_by: str | None = None
    note: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "benefit": self.benefit,
            "holder": self.holder,
            "rider": self.rider,
            "train": self.train,
            "transfer": self.transfer,
            "authorized_by": self.authorized_by,
            "note": self.note,
        }


@dataclass(frozen=True)
class Tradeoff:
    code: str
    summary: str
    question: str

    def describe(self) -> dict[str, Any]:
        return {"code": self.code, "summary": self.summary, "question": self.question}


@dataclass(frozen=True)
class Plan:
    id: str
    legs: tuple[Leg, ...]
    benefits: tuple[BenefitUse, ...]
    unused_benefits: tuple[str, ...]
    tradeoffs: tuple[Tradeoff, ...]
    notes: tuple[str, ...]
    seats_used: Mapping[str, int]
    fingerprint: str
    condition_fingerprint: str = ""
    seat_fingerprint: str = ""
    condition_snapshot: Mapping[str, str] = field(default_factory=dict)
    changes: tuple[str, ...] = ()

    def leg(self, order: str) -> Leg:
        for leg in self.legs:
            if leg.order == order:
                return leg
        raise DataError(f"方案 {self.id} 没有订单 {order}")

    def person_leg(self, person: str) -> Leg | None:
        for leg in self.legs:
            if leg.person == person:
                return leg
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "legs": [leg.describe() for leg in self.legs],
            "benefits": [use.describe() for use in self.benefits],
            "unused_benefits": list(self.unused_benefits),
            "tradeoffs": [item.describe() for item in self.tradeoffs],
            "notes": list(self.notes),
            "seats_used": dict(self.seats_used),
            "changes": list(self.changes),
            "fingerprint": self.fingerprint,
            "conditions": self.condition_fingerprint,
            "seats": self.seat_fingerprint,
        }


@dataclass(frozen=True)
class Rejected:
    plan_id: str | None
    reasons: tuple[str, ...]
    changed: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "rejected": list(self.reasons),
            "changed": list(self.changed),
            "questions": list(self.questions),
            "kept": list(self.kept),
        }


@dataclass(frozen=True)
class Outcome:
    """一次成功办理的结果。同一 request 查询须返回同一份。"""

    request: str
    plan_id: str
    actors: tuple[str, ...]
    legs: tuple[Leg, ...]
    benefits: tuple[BenefitUse, ...]
    seats_used: Mapping[str, int]
    fingerprint: str

    def describe(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "plan": self.plan_id,
            "actors": list(self.actors),
            "legs": [leg.describe() for leg in self.legs],
            "benefits": [use.describe() for use in self.benefits],
            "seats_used": dict(self.seats_used),
            "fingerprint": self.fingerprint,
        }


# ----------------------------------------------------------------------
# 值班台


class Desk:
    """在一份资料和一个会话状态上生成、校验、确认改乘。"""

    def __init__(self, workbook: Workbook, conversation: Conversation | None = None):
        self.workbook = workbook
        self.conversation = conversation or Conversation(workbook)
        if not self.conversation.seen:
            self.conversation.read_all()
        self.rules = read_notices(workbook)
        self._base_seats = {
            train_id: train.seats for train_id, train in workbook.trains.items()
        }
        for row in workbook.counter:
            self._base_seats[row.service] = (
                self._base_seats.get(row.service, 0) + row.seat_delta
            )
        self._sold: dict[str, int] = {}
        self._outcomes: dict[str, Outcome] = {}
        self._plans: dict[str, Plan] = {}
        self._expired: dict[str, Rejected] = {}  # 按条件指纹记，避免误伤新方案
        self._confirmed: dict[str, str] = {}
        self._ledger: list[dict[str, Any]] = []
        self._committed_orders: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 会话与座位的变化

    def read(self, message: Message) -> list[Claim]:
        """读入一条新往来；此前生成的方案会因条件指纹变化而失效。"""
        claims = self.conversation.read(message)
        return claims

    def free_seats(self, train_id: str) -> int:
        return max(0, self._base_seats.get(train_id, 0) - self._sold.get(train_id, 0))

    def record_sale(self, service: str, count: int, request: str) -> None:
        """另一订单先成交，占用余位。"""
        if service not in self.workbook.trains:
            raise DataError(f"没有班次 {service}")
        if count < 0:
            raise DataError("成交数量不能为负")
        if self.free_seats(service) < count:
            raise DataError(f"{service} 余位不足")
        self._sold[service] = self._sold.get(service, 0) + count
        self._ledger.append(
            {
                "at": "",
                "service": service,
                "seat_delta": -count,
                "request": request,
            }
        )

    # ------------------------------------------------------------------
    # 硬约束

    def affected(self) -> tuple[Order, ...]:
        return tuple(
            order
            for order in sorted(self.workbook.orders.values(), key=lambda row: row.id)
            if self.rules.service_affected(order)
        )

    def allowed_boardings(self, order: Order) -> frozenset[str]:
        """本人明确接受的上车站是硬约束；未表态时保留原站与通告补进的站。"""
        rule = self.conversation.boarding_rule(order.person)
        if rule is not None and rule.exclusive:
            return rule.stations
        allowed = {order.boarding}
        if rule is not None:
            allowed |= set(rule.stations)
        # 本人明确接受过的临时补进站才可用；通告只是“可以补进”，不是本人同意。
        if self.rules.service_affected(order):
            allowed |= set(self.rules.added_boardings) & self._accepted_stations(order)
        return frozenset(allowed)

    def _accepted_stations(self, order: Order) -> set[str]:
        rule = self.conversation.boarding_rule(order.person)
        return set(rule.stations) if rule is not None else set()

    def candidates(self, order: Order) -> list[tuple[Train, str]]:
        if not self.rules.service_affected(order):
            train = self.workbook.trains.get(order.service)
            return [(train, order.boarding)] if train is not None else []
        result: list[tuple[Train, str]] = []
        for train_id in sorted(self.rules.candidate_trains()):
            train = self.workbook.trains.get(train_id)
            if train is None or not train.accepted:
                continue
            for boarding in sorted(self.allowed_boardings(order)):
                if self.rules.permitted(train, boarding, order.alighting):
                    result.append((train, boarding))
        return result

    # ------------------------------------------------------------------
    # 方案生成

    def _joint_members(self, person: str) -> frozenset[str]:
        return self.conversation.joint_of(person)

    def _groups(self) -> list[tuple[str, ...]]:
        """把有同行约束的人合成组；组内所有人的全部订单必须同车。"""
        people = sorted({order.person for order in self.workbook.orders.values()})
        groups: list[tuple[str, ...]] = []
        placed: set[str] = set()
        for person in people:
            if person in placed:
                continue
            group = tuple(
                sorted(member for member in self._joint_members(person) if member in people)
            )
            if not group:
                group = (person,)
            groups.append(group)
            placed.update(group)
        return groups

    def _group_options(self, members: tuple[str, ...]) -> list[tuple[Leg, ...]]:
        """每个订单的候选；未受影响的订单只有“原地不动”一种。"""
        orders = [
            order
            for person in members
            for order in self.workbook.orders_of(person)
        ]
        if not orders:
            return []
        combined = len(members) > 1
        per_order: list[list[tuple[str, str, bool]]] = []
        for order in orders:
            if not self.rules.service_affected(order):
                per_order.append([(order.service, order.boarding, False)])
                continue
            options = [
                (train.id, boarding, True)
                for train, boarding in self.candidates(order)
            ]
            if not options:
                return []
            per_order.append(options)
        result: list[tuple[Leg, ...]] = []
        for choice in itertools.product(*per_order):
            trains = {train_id for train_id, _, _ in choice}
            if combined and len(trains) > 1:
                continue
            result.append(
                tuple(
                    Leg(
                        order=order.id,
                        person=order.person,
                        train=train_id,
                        boarding=boarding,
                        alighting=order.alighting,
                        changed=changed,
                    )
                    for order, (train_id, boarding, changed) in zip(orders, choice)
                )
            )
        return result

    def draft(self, limit: int = 5) -> list[Plan]:
        """列出可办理的整套方案，按体验兑现、换站少、发车早排序。

        已经按某回复办理过的订单不再出现在候选里，避免重复占用席位。
        """
        if self._committed_orders:
            return []
        options = [self._group_options(members) for members in self._groups()]
        if not options or any(not item for item in options):
            return []
        plans: list[Plan] = []
        for combination in itertools.product(*options):
            legs = []
            seats_used: dict[str, int] = {}
            for part in combination:
                for leg in part:
                    # 只有替代班次需要占余位；未受影响的订单沿用原票。
                    if leg.changed:
                        seats_used[leg.train] = seats_used.get(leg.train, 0) + 1
            if any(self.free_seats(train) < count for train, count in seats_used.items()):
                continue
            for part in combination:
                legs.extend(part)
            legs.sort(key=lambda row: row.order)
            plans.extend(self._build_plans(tuple(legs), seats_used))
        plans.sort(
            key=lambda plan: (
                len(plan.tradeoffs),
                len(plan.unused_benefits),
                sum(1 for leg in plan.legs if leg.changed),
                sum(
                    1
                    for leg in plan.legs
                    if leg.changed
                    and leg.boarding
                    != self.workbook.order(leg.order).boarding
                ),
                max(
                    (
                        self.workbook.train(leg.train).departures
                        for leg in plan.legs
                        if leg.train in self.workbook.trains
                    ),
                    default="",
                ),
                plan.id,
            )
        )
        unique: list[Plan] = []
        seen: set[str] = set()
        for plan in plans:
            if plan.id in seen:
                continue
            seen.add(plan.id)
            unique.append(plan)
            self._plans[plan.id] = plan
            if len(unique) >= limit:
                break
        return unique

    def _build_plans(
        self, legs: tuple[Leg, ...], seats_used: Mapping[str, int]
    ) -> list[Plan]:
        orders = {order.id: order for order in self.workbook.orders.values()}
        by_order = {leg.order: leg for leg in legs}
        condition_fingerprint = self._condition_digest()
        seat_fingerprint = self._seat_digest()
        condition_snapshot = {
            claim.id: f"{claim.kind}:{claim.status}"
            for claim in self.conversation.claims
        }
        changes = tuple(
            f"{leg.order} {orders[leg.order].service}/{orders[leg.order].boarding}"
            f" → {leg.train}/{leg.boarding}"
            for leg in legs
            if leg.changed
        )
        built: list[Plan] = []
        for uses, tradeoffs, notes in self._benefit_variants(by_order, orders):
            notes = list(notes) + [item.summary for item in self.conversation.conflicts]
            used = {use.benefit for use in uses}
            unused = tuple(
                benefit.id
                for benefit in sorted(
                    self.workbook.benefits.values(), key=lambda row: row.id
                )
                if benefit.id not in used
            )
            payload = {
                "legs": sorted(leg.key() for leg in legs),
                "benefits": sorted((use.benefit, use.rider) for use in uses),
                "conditions": condition_fingerprint,
                "seats": seat_fingerprint,
            }
            fingerprint = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()[:16]
            plan_id = "P-" + hashlib.sha256(
                json.dumps(
                    {
                        "legs": sorted(
                            (leg.order, leg.train, leg.boarding) for leg in legs
                        ),
                        "benefits": sorted(
                            (use.benefit, use.rider) for use in uses
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:8]
            built.append(
                Plan(
                    id=plan_id,
                    legs=legs,
                    benefits=uses,
                    unused_benefits=unused,
                    tradeoffs=tuple(tradeoffs),
                    notes=tuple(notes),
                    seats_used=dict(seats_used),
                    fingerprint=fingerprint,
                    condition_fingerprint=condition_fingerprint,
                    seat_fingerprint=seat_fingerprint,
                    condition_snapshot=condition_snapshot,
                    changes=changes,
                )
            )
        built.sort(
            key=lambda plan: (
                len(plan.tradeoffs),
                len(plan.unused_benefits),
                -len(plan.benefits),
            )
        )
        return built

    def _benefit_variants(
        self,
        by_order: Mapping[str, Leg],
        orders: Mapping[str, Order],
    ) -> list[tuple[tuple[BenefitUse, ...], list[Tradeoff], list[str]]]:
        """按券规则列本方案可兑现的体验组合。

        转让只有在会话中有仍然有效的授权、且受让人实际乘坐符合券区间的班次
        时才使用；持券人本人自用始终是并列的备选。无法兑现的券保持未使用，并
        把取舍和问题留给旅客，绝不替别人先核销。
        """
        variants: list[
            tuple[tuple[BenefitUse, ...], list[Tradeoff], list[str]]
        ] = [((), [], [])]
        for benefit in sorted(self.workbook.benefits.values(), key=lambda row: row.id):
            if benefit.holder is None:
                continue
            holder_leg = self._leg_of(benefit.holder, by_order, orders)
            holder_fits = (
                holder_leg is not None
                and self._benefit_fits(benefit, holder_leg, orders)
            )
            state = self.conversation.transfers.get(benefit.id)
            choices: list[
                tuple[BenefitUse | None, list[Tradeoff], list[str]]
            ] = []
            if holder_fits:
                tradeoffs: list[Tradeoff] = []
                notes: list[str] = []
                if state is not None and state.status == WITHDRAWN:
                    tradeoffs.append(
                        Tradeoff(
                            code="benefit_withdrawn",
                            summary=(
                                f"{benefit.id} 的转让已由持券人 {benefit.holder} 收回；"
                                "同行人无法再使用，只能由本人自用。"
                            ),
                            question=(
                                f"是否请 {benefit.holder} 重新确认转出 {benefit.id}，"
                                "还是保留车票方案、由本人使用？"
                            ),
                        )
                    )
                elif state is not None and state.active and state.recipient:
                    notes.append(
                        f"{benefit.id} 的受让人 {state.recipient} 本方案无法满足券区间，"
                        f"按 {benefit.holder} 本人使用。"
                    )
                choices.append(
                    (
                        BenefitUse(
                            benefit=benefit.id,
                            holder=benefit.holder,
                            rider=benefit.holder,
                            train=holder_leg.train if holder_leg else "",
                            transfer=False,
                            authorized_by=benefit.holder,
                        ),
                        tradeoffs,
                        notes,
                    )
                )
            if state is not None and state.active and state.recipient is not None:
                recipient_leg = self._leg_of(state.recipient, by_order, orders)
                if (
                    recipient_leg is not None
                    and self._scope_allows(benefit, state.recipient)
                    and self._transfer_condition_met(state, recipient_leg)
                    and self._benefit_fits(benefit, recipient_leg, orders)
                ):
                    tradeoffs = []
                    question = (
                        f"{benefit.holder} 是否确认由 {state.recipient} 使用 {benefit.id}？"
                    )
                    if holder_fits:
                        tradeoffs.append(
                            Tradeoff(
                                code="benefit_moved",
                                summary=(
                                    f"{benefit.id} 转给 {state.recipient} 后，"
                                    f"{benefit.holder} 本人不再享受该体验。"
                                ),
                                question=question,
                            )
                        )
                    choices.append(
                        (
                            BenefitUse(
                                benefit=benefit.id,
                                holder=benefit.holder,
                                rider=state.recipient,
                                train=recipient_leg.train,
                                transfer=True,
                                authorized_by=state.holder,
                                note="按持券人有效授权转给同行人。",
                            ),
                            tradeoffs,
                            [],
                        )
                    )
            if not choices:
                tradeoffs = []
                notes = []
                self._benefit_tradeoff(
                    benefit, state, by_order, orders, tradeoffs, notes
                )
                choices.append((None, tradeoffs, notes))
            expanded = []
            for uses_, tradeoffs_, notes_ in variants:
                for use, extra_tradeoffs, extra_notes in choices:
                    expanded.append(
                        (
                            uses_ + ((use,) if use is not None else ()),
                            tradeoffs_ + list(extra_tradeoffs),
                            notes_ + list(extra_notes),
                        )
                    )
            variants = expanded
        return variants

    def _benefit_tradeoff(
        self,
        benefit: Benefit,
        state: Any,
        by_order: Mapping[str, Leg],
        orders: Mapping[str, Order],
        tradeoffs: list[Tradeoff],
        notes: list[str],
    ) -> None:
        holder_leg = self._leg_of(benefit.holder or "", by_order, orders)
        reason = self._why_not(benefit, holder_leg, orders)
        if state is not None and state.status == WITHDRAWN:
            tradeoffs.append(
                Tradeoff(
                    code="benefit_withdrawn",
                    summary=(
                        f"{benefit.id} 的转让已由持券人 {benefit.holder} 收回，"
                        "本方案无法兑现团队体验。"
                    ),
                    question=(
                        f"是否请 {benefit.holder} 重新确认转出 {benefit.id}，"
                        "还是保留车票方案、放弃该体验？"
                    ),
                )
            )
            return
        if state is not None and state.active and state.recipient:
            tradeoffs.append(
                Tradeoff(
                    code="benefit_unavailable",
                    summary=(
                        f"{benefit.id} 已授权转给 {state.recipient}，"
                        f"但本方案中该同行人不符合券的区间或班次要求。"
                    ),
                    question=(
                        f"是在 {state.recipient} 符合条件的班次上重排，"
                        "还是改由持券人自用/放弃该体验？"
                    ),
                )
            )
            return
        if holder_leg is not None:
            tradeoffs.append(
                Tradeoff(
                    code="benefit_unavailable",
                    summary=f"{benefit.id} 在本方案无法兑现：{reason}",
                    question=(
                        f"是否优先选择能兑现 {benefit.id} 的班次，"
                        "还是按当前车票方案办理并放弃该体验？"
                    ),
                )
            )
        else:
            notes.append(f"{benefit.id} 的持券人不在本方案中。")

    def _why_not(
        self, benefit: Benefit, leg: Leg | None, orders: Mapping[str, Order]
    ) -> str:
        if leg is None:
            return "持券人不在本方案中"
        train = self.workbook.train(leg.train)
        if benefit.feature not in train.features:
            return f"{leg.train} 不提供 {benefit.feature}"
        if not train.serves(leg.boarding, leg.alighting):
            return f"{leg.train} 不办理 {leg.boarding}→{leg.alighting}"
        if benefit.required_stop not in train.interval(leg.boarding, leg.alighting):
            return f"乘坐区间不经过 {benefit.required_stop}"
        if benefit.holder_only and orders[leg.order].person != benefit.holder:
            return "券限本人使用"
        return "不符合券的使用规则"

    def _leg_of(
        self, person: str, by_order: Mapping[str, Leg], orders: Mapping[str, Order]
    ) -> Leg | None:
        for order_id, leg in by_order.items():
            if orders[order_id].person == person:
                return leg
        return None

    def _benefit_fits(
        self, benefit: Benefit, leg: Leg, orders: Mapping[str, Order]
    ) -> bool:
        train = self.workbook.train(leg.train)
        if benefit.feature not in train.features:
            return False
        if not train.serves(leg.boarding, leg.alighting):
            return False
        if benefit.required_stop not in train.interval(leg.boarding, leg.alighting):
            return False
        if benefit.holder_only:
            return orders[leg.order].person == benefit.holder
        return self._scope_allows(benefit, orders[leg.order].person)

    def _scope_allows(self, benefit: Benefit, person: str) -> bool:
        """受让人范围：本人、同一改乘会话中的儿童或同行人。"""
        if person == benefit.holder:
            return True
        scope = benefit.recipient_scope
        if "儿童" in scope or "孩子" in scope:
            if person not in self._session_people():
                return False
            aliases = self.workbook.person(person).aliases
            return any(
                alias.startswith("小") or "孩子" in alias or "儿童" in alias
                for alias in aliases
            )
        if "同行" in scope:
            return person in self._session_people()
        if scope == "本人":
            return False
        return person in self._session_people()

    def _session_people(self) -> set[str]:
        people = self.conversation.joint_people()
        return people | {order.person for order in self.affected()}

    def _transfer_condition_met(self, state: Any, leg: Leg) -> bool:
        """附条件转让：受让人实际乘坐的区间必须经过当时约定的站。"""
        station = getattr(state, "condition_stop", None)
        if not station:
            return True
        train = self.workbook.trains.get(leg.train)
        if train is None or station not in train.stops:
            return False
        if not train.serves(leg.boarding, leg.alighting):
            return False
        return station in train.interval(leg.boarding, leg.alighting)

    # ------------------------------------------------------------------
    # 指纹

    def _condition_digest(self) -> str:
        rows = [
            (claim.id, claim.kind, claim.status, sorted(claim.value.items(), key=str))
            for claim in self.conversation.claims
            if claim.kind != "narrative"
        ]
        return hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def _seat_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(sorted(self._sold.items())).encode()
        ).hexdigest()[:16]

    # ------------------------------------------------------------------
    # 校验与确认

    def check(self, plan: Plan) -> list[str]:
        """逐项复核方案现在是否仍成立。"""
        problems: list[str] = []
        if self._condition_digest() != plan.condition_fingerprint:
            current = {
                claim.id: f"{claim.kind}:{claim.status}"
                for claim in self.conversation.claims
            }
            changed = [
                f"{key} {before} → {current.get(key, '已撤销')}"
                for key, before in plan.condition_snapshot.items()
                if current.get(key) != before
            ]
            changed.extend(
                f"{key} 新增" for key in current if key not in plan.condition_snapshot
            )
            problems.append(
                "会话中的条件在方案生成后发生变化"
                + ("：" + "，".join(changed) if changed else "。")
            )
        for train_id, count in plan.seats_used.items():
            if self.free_seats(train_id) < count:
                problems.append(
                    f"{train_id} 余位不足：需要 {count}，现有 {self.free_seats(train_id)}。"
                )
        candidate_trains = self.rules.candidate_trains()
        for leg in plan.legs:
            if not leg.changed:
                continue
            train = self.workbook.trains.get(leg.train)
            order = self.workbook.order(leg.order)
            if train is None or not train.accepted:
                problems.append(f"{leg.train} 未公开办理改乘。")
                continue
            if leg.train not in candidate_trains:
                problems.append(f"{leg.train} 不在通告允许的改乘班次内。")
            if not self.rules.permitted(train, leg.boarding, leg.alighting):
                problems.append(
                    f"{leg.train} 不按规定办理 {leg.boarding}→{leg.alighting}。"
                )
            if leg.boarding not in self.allowed_boardings(order):
                problems.append(
                    f"{order.person} 不接受从 {leg.boarding} 上车。"
                )

            if leg.alighting != order.alighting:
                problems.append(
                    f"订单 {order.id} 的下车站被改动为 {leg.alighting}。"
                )
        by_order = {leg.order: leg for leg in plan.legs}
        for group in self.conversation.joint_groups:
            trains = {
                by_order[order.id].train
                for person in group
                for order in self.workbook.orders_of(person)
                if order.id in by_order
            }
            if len(trains) > 1:
                problems.append(
                    "已确认同车的 "
                    + "、".join(sorted(group))
                    + " 被分到不同班次。"
                )
        orders = {order.id: order for order in self.workbook.orders.values()}
        for order_id, claim in self.conversation.protected.items():
            leg = by_order.get(order_id)
            if leg is None or not leg.changed:
                continue
            original = self.workbook.order(order_id)
            if not self.rules.service_affected(original):
                problems.append(
                    f"订单 {order_id} 的本人要求保留原座位，但方案改动了它。"
                )
        for use in plan.benefits:
            benefit = self.workbook.benefit(use.benefit)
            if not use.transfer:
                leg = self._leg_for_person(plan, use.rider)
                if leg is None or not self._benefit_fits(benefit, leg, orders):
                    problems.append(
                        f"{benefit.id} 已不符合本人使用的券规则。"
                    )
                continue
            state = self.conversation.transfers.get(benefit.id)
            if state is not None and state.status == WITHDRAWN:
                problems.append(
                    f"{benefit.id} 已由持券人 {benefit.holder} 收回转让；"
                    f"{use.rider} 不能再使用。"
                )
                continue
            if state is None or state.status != "granted" or state.recipient != use.rider:
                problems.append(
                    f"{benefit.id} 转给 {use.rider} 的授权已不成立。"
                )
                continue
            leg = self._leg_for_person(plan, use.rider)
            if (
                leg is None
                or not self._benefit_fits(benefit, leg, orders)
                or not self._transfer_condition_met(state, leg)
            ):
                problems.append(
                    f"{benefit.id} 的乘车区间、班次或附条件已不符合券规则。"
                )
        return list(dict.fromkeys(problems))

    def _leg_for_person(self, plan: Plan, person: str) -> Leg | None:
        for leg in plan.legs:
            if leg.person == person:
                return leg
        return None

    # -- 权限 ---------------------------------------------------------

    def confirmers_for(self, person: str) -> frozenset[str]:
        """本人与订单账户联系人；本人若声明只能代选，则账户不能代确认。"""
        allowed = {person}
        if self.conversation.authority_limits.get(person, {}).get("may_confirm") is False:
            return frozenset(allowed)
        for order in self.workbook.orders_of(person):
            if order.account:
                allowed.add(order.account)
        return frozenset(allowed)

    def _needed_people(self, plan: Plan) -> set[str]:
        needed = {leg.person for leg in plan.legs if leg.changed}
        for use in plan.benefits:
            if use.transfer:
                needed.add(use.holder)
                needed.add(use.rider)
        return needed

    def missing_confirmers(self, plan: Plan, actors: Iterable[str]) -> list[str]:
        actors = set(actors)
        return sorted(
            person
            for person in self._needed_people(plan)
            if not actors & set(self.confirmers_for(person))
        )

    # -- 确认 ---------------------------------------------------------

    def confirm(
        self,
        plan: Plan,
        actor: str | Iterable[str],
        request: str,
    ) -> Outcome | Rejected:
        actors = tuple(sorted({actor} if isinstance(actor, str) else set(actor)))
        if not actors:
            return Rejected(plan.id, ("没有记录任何确认人。",))
        if request in self._outcomes:
            existing = self._outcomes[request]
            if existing.plan_id != plan.id:
                return Rejected(
                    plan.id,
                    (f"回复 {request} 已办理过另一份方案 {existing.plan_id}。",),
                )
            return existing
        if request in self._confirmed:
            return Rejected(
                plan.id,
                (f"回复 {request} 已用于确认方案 {self._confirmed[request]}。",),
            )
        already = sorted(
            leg.order for leg in plan.legs if leg.order in self._committed_orders
        )
        if already:
            return Rejected(
                plan.id,
                tuple(
                    f"订单 {order} 已按 {self._committed_orders[order]} 办理，不能重复确认。"
                    for order in already
                ),
            )
        problems = self.check(plan)
        missing = self.missing_confirmers(plan, actors)
        for person in missing:
            problems.append(
                f"{'、'.join(actors)} 不能替 {person} 确认；"
                "需本人或订单账户联系人确认。"
            )
        if plan.fingerprint in self._expired:
            problems.extend(self._expired[plan.fingerprint].reasons)
        if problems:
            changed = tuple(
                item
                for item in dict.fromkeys(problems)
                if any(word in item for word in ("变化", "收回", "授权", "余位"))
            )
            rejected = Rejected(
                plan.id,
                tuple(dict.fromkeys(problems)),
                changed=changed,
                questions=tuple(self.conversation.questions),
                kept=self._kept(),
            )
            # 只是缺确认人的方案不该被永久作废；条件或余位变了才算失效。
            if changed or any("余位" in item for item in problems):
                self._expired[plan.fingerprint] = rejected
            return rejected
        for train_id, count in plan.seats_used.items():
            self._sold[train_id] = self._sold.get(train_id, 0) + count
            self._ledger.append(
                {
                    "at": "",
                    "service": train_id,
                    "seat_delta": -count,
                    "request": request,
                }
            )
        outcome = Outcome(
            request=request,
            plan_id=plan.id,
            actors=actors,
            legs=plan.legs,
            benefits=plan.benefits,
            seats_used=plan.seats_used,
            fingerprint=plan.fingerprint,
        )
        self._outcomes[request] = outcome
        self._confirmed[request] = plan.id
        for leg in plan.legs:
            self._committed_orders[leg.order] = plan.id
        return outcome

    def _kept(self) -> tuple[str, ...]:
        kept = []
        if self.conversation.joint_groups:
            kept.append("已确认同车：" + "；".join(
                "、".join(sorted(group)) for group in self.conversation.joint_groups
            ))
        for person, rule in sorted(self.conversation.boarding_claims.items()):
            kept.append(
                f"{person} 接受的上车站：" + "、".join(sorted(rule.stations))
            )
        for benefit_id, state in sorted(self.conversation.transfers.items()):
            if state.status == WITHDRAWN:
                kept.append(f"{benefit_id} 已收回，不得转给他人")
        return tuple(kept)

    def lookup(self, request: str) -> Outcome | None:
        return self._outcomes.get(request)

    def outcomes(self) -> tuple[Outcome, ...]:
        return tuple(self._outcomes[key] for key in sorted(self._outcomes))

    def restore(self, outcome: Outcome) -> None:
        """载入已办理结果并占用对应座位，用于跨进程查询。"""
        if outcome.request in self._outcomes:
            return
        self._outcomes[outcome.request] = outcome
        self._confirmed[outcome.request] = outcome.plan_id
        for train_id, count in outcome.seats_used.items():
            self._sold[train_id] = self._sold.get(train_id, 0) + count
        for leg in outcome.legs:
            self._committed_orders[leg.order] = outcome.plan_id

    def recheck(
        self, plan_id: str
    ) -> tuple[Plan | None, Rejected | None, list[Plan]]:
        """重新核对旧方案；失效则说明变化并给出可重发的新方案。"""
        try:
            prior = self._plans[plan_id]
        except KeyError as error:
            raise DataError(f"没有方案 {plan_id}") from error
        problems = self.check(prior)
        if not problems:
            return prior, None, []
        rejected = Rejected(
            plan_id,
            tuple(dict.fromkeys(problems)),
            changed=tuple(
                item
                for item in dict.fromkeys(problems)
                if any(word in item for word in ("变化", "收回", "余位", "授权"))
            ),
            questions=tuple(self.conversation.questions),
            kept=self._kept(),
        )
        self._expired[prior.fingerprint] = rejected
        return None, rejected, self.draft()

    def questions(self) -> list[str]:
        return list(dict.fromkeys(self.conversation.questions))

    def explain(self, plan: Plan, other: Plan | None = None) -> list[str]:
        lines: list[str] = []
        for leg in plan.legs:
            order = self.workbook.order(leg.order)
            if leg.changed:
                lines.append(
                    f"{leg.person}：{order.service}/{order.boarding} → "
                    f"{leg.train}/{leg.boarding}（{leg.alighting} 下车）。"
                )
            else:
                lines.append(f"{leg.person}：维持 {leg.train}/{leg.boarding}。")
        for use in plan.benefits:
            lines.append(
                f"{use.benefit}：{use.rider} 在 {use.train} 使用"
                + ("（按转让授权）" if use.transfer else "（本人）")
                + "。"
            )
        for item in plan.tradeoffs:
            lines.append(f"取舍 {item.code}：{item.summary} 需确认：{item.question}")
        for note in plan.notes:
            lines.append(note)
        if other is not None:
            changed = sorted(
                {(leg.order, leg.train, leg.boarding) for leg in plan.legs}
                ^ {(leg.order, leg.train, leg.boarding) for leg in other.legs}
            )
            lines.append(
                "与另一方案的差异："
                + ("；".join(f"{order} {train}/{boarding}" for order, train, boarding in changed)
                   if changed else "无")
            )
        return lines
