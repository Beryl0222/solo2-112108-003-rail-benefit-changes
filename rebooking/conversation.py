"""把客服往来读成可追溯的表态。

只有本人，或订单上登记的账户联系人，才能确认变更。字面上的“他说”“替我”
“可以不等”只是转述或提议，必须由有权限的人再确认。收回表态优先于此前
的授权；同名的两个人的候选不会被静默合并。

每条表态都记录来源消息与原文，方便客服核对；指代不明时保留候选并生成
具体问题，而不是按姓名擅自合并。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .entities import Mention, Resolver
from .model import DataError, Message, Order, Workbook

CONFIRMED = "confirmed"
RELAYED = "relayed"
PROPOSED = "proposed"
ANSWERED = "answered"
WITHDRAWN = "withdrawn"
NARRATIVE = "narrative"
UNRESOLVED = "unresolved"

CONSENT = re.compile(r"也成|可以|同意|没问题|确认")
REFUSAL = re.compile(r"不行|不能|不同意|不要|先别|没法|拒绝")
JOINT = re.compile(r"同车|同一班|同一趟|一起走|一定一起|同行|一起")
BINDING = re.compile(r"一定|必须|要|得|就")


@dataclass(frozen=True)
class Claim:
    id: str
    message: str
    actor: str
    kind: str
    quote: str
    subjects: tuple[str, ...]
    value: Mapping[str, Any]
    status: str
    authorized: bool
    note: str = ""

    @property
    def binding(self) -> bool:
        return self.status == CONFIRMED and self.authorized


@dataclass
class Proposal:
    id: str
    message: str
    subject: str
    kind: str
    value: Mapping[str, Any]
    quote: str
    status: str = "pending"
    answered_by: str | None = None


@dataclass
class TransferState:
    benefit: str
    holder: str
    recipient: str | None
    condition_stop: str | None
    grants: list[str] = field(default_factory=list)
    withdrawals: list[str] = field(default_factory=list)
    status: str = "none"

    @property
    def active(self) -> bool:
        return self.status == "granted"


@dataclass(frozen=True)
class BoardingRule:
    stations: frozenset[str]
    exclusive: bool
    claim: str
    status: str


@dataclass
class Conflict:
    topic: str
    message: str
    summary: str
    kept: tuple[str, ...]
    question: str

    def describe(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "message": self.message,
            "summary": self.summary,
            "kept": list(self.kept),
            "question": self.question,
        }


@dataclass(frozen=True)
class Clause:
    text: str
    start: int
    end: int
    stations: tuple[str, ...]

    def mentions_of(self, mentions: Iterable[Mention]) -> tuple[Mention, ...]:
        return tuple(
            item for item in mentions if self.start <= item.start < self.end
        )


@dataclass(frozen=True)
class Sentence:
    text: str
    mentions: tuple[Mention, ...]
    clauses: tuple[Clause, ...]
    trains: tuple[str, ...]


class Conversation:
    """按资料和消息顺序累积的会话状态。"""

    def __init__(self, workbook: Workbook):
        self.workbook = workbook
        self.resolver = Resolver(workbook)
        self.claims: list[Claim] = []
        self.proposals: dict[str, Proposal] = {}
        self.transfers: dict[str, TransferState] = {}
        self.joint_groups: list[frozenset[str]] = []
        self.boardings: dict[str, set[str]] = {}
        self.boarding_claims: dict[str, BoardingRule] = {}
        self.protected: dict[str, str] = {}
        self.trusted_selectors: dict[str, set[str]] = {}
        self.authority_limits: dict[str, dict[str, Any]] = {}
        self.questions: list[str] = []
        self.conflicts: list[Conflict] = []
        self.identity_notes: list[str] = []
        self.unresolved: list[Claim] = []
        self._seq = 0
        self._proposal_seq = 0
        self.seen: list[str] = []

    # ------------------------------------------------------------------
    # 读入

    def read_all(self) -> None:
        for message in self.workbook.ordered_messages():
            self.read(message)

    def read(self, message: Message) -> list[Claim]:
        if message.id in self.seen:
            raise DataError(f"消息 {message.id} 已经读过")
        self.seen.append(message.id)
        context = self._person_context(message)
        new: list[Claim] = []
        for sentence in self._sentences(message, context):
            for claim in self._interpret(message, sentence):
                new.append(claim)
                self._apply(claim)
        self.claims.extend(new)
        self._refresh_questions()
        return new

    def _sentences(self, message: Message, context: frozenset[str]) -> list[Sentence]:
        stations = self.workbook.stations()
        parts = [
            part.strip()
            for part in re.split(r"[。！？!?\n]+", message.text)
            if part.strip()
        ]
        known: dict[str, str] = {}
        result: list[Sentence] = []
        for text in parts:
            clauses = []
            cursor = 0
            for chunk in re.split(r"([，,；;])", text):
                if chunk in {"，", ",", "；", ";"}:
                    cursor += len(chunk)
                    continue
                if chunk.strip():
                    start = cursor
                    end = cursor + len(chunk)
                    clauses.append(
                        Clause(
                            text=chunk.strip(),
                            start=start,
                            end=end,
                            stations=tuple(
                                sorted(
                                    (station for station in stations if station in chunk),
                                    key=lambda item: (chunk.find(item), -len(item)),
                                )
                            ),
                        )
                    )
                cursor += len(chunk)
            mentions = self.resolver.scan(text, message.actor, tuple(context))
            narrowed: list[Mention] = []
            for clause in clauses:
                local = [
                    item
                    for item in mentions
                    if clause.start <= item.start < clause.end
                    and item not in narrowed
                ]
                narrowed.extend(
                    self.resolver.narrow_by_station(
                        local, clause.text, clause.stations
                    )
                )
            mentions = [
                item for item in narrowed
            ]
            mentions = self.resolver.cross_sentence(mentions, known)
            mentions = self._same_sentence_coreference(mentions, clauses)
            for item in mentions:
                surname = _surname(item.token)
                if item.person and surname and not item.pronoun:
                    known.setdefault(surname, item.person)
            result.append(
                Sentence(
                    text=text,
                    mentions=tuple(sorted(mentions, key=lambda row: row.start)),
                    clauses=tuple(clauses),
                    trains=tuple(sorted(set(re.findall(r"R\d+", text)))),
                )
            )
        return result

    def _same_sentence_coreference(
        self, mentions: list[Mention], clauses: tuple[Clause, ...]
    ) -> list[Mention]:
        """同一句里前一处的“那个”可由后一分句的车站限定补全。

        例如“叫陈师傅的那个从东城上……问下石影下车的老陈”：两处指同一位
        person，但只有后一处带车站证据；前一处仍保留“同指”的依据说明。
        """
        resolved = [item for item in mentions if item.person is not None]
        result: list[Mention] = []
        for item in mentions:
            if item.person is not None:
                result.append(item)
                continue
            surname = _surname(item.token)
            same = [
                other
                for other in resolved
                if other.person
                and surname
                and _surname(other.token) == surname
                and other.basis.startswith("句内车站限定")
            ]
            if len({other.person for other in same}) == 1:
                person = same[0].person
                result.append(
                    Mention(
                        token=item.token,
                        start=item.start,
                        end=item.end,
                        candidates=frozenset({person}),
                        person=person,
                        basis="同句后文车站限定，指同一人",
                        pronoun=item.pronoun,
                    )
                )
            else:
                result.append(item)
        return result

    def _person_context(self, message: Message) -> frozenset[str]:
        """同一会话链上已明确出现的旅客；只用于代词消解。"""
        found: set[str] = set()
        try:
            chain = self.workbook.message_context(message.id)
        except DataError:
            chain = (message,)
        for row in chain:
            found.add(row.actor)
        return frozenset(found - {""})

    # ------------------------------------------------------------------
    # 语义

    def _interpret(self, message: Message, sentence: Sentence) -> list[Claim]:
        actor = message.actor
        claims: list[Claim] = []
        previous: set[str] | None = None
        if self._withdraws_benefit(sentence.text):
            for benefit in self._benefits_of(actor):
                claims.append(
                    self._claim(
                        message,
                        "benefit_withdrawn",
                        sentence.text,
                        (actor,),
                        {"benefit": benefit},
                        status=CONFIRMED,
                        authorized=True,
                        note="持券人本人收回转让。",
                    )
                )
                break
        for clause in sentence.clauses:
            local = clause.mentions_of(sentence.mentions)
            subjects = self._subjects(local, previous, actor)
            previous = subjects or previous
            claims.extend(
                self._clause_claims(message, sentence, clause, local, subjects)
            )
        if not claims:
            claims.append(
                self._claim(
                    message,
                    NARRATIVE,
                    sentence.text,
                    tuple(sorted({item.person for item in sentence.mentions if item.person})),
                    {},
                    status=NARRATIVE,
                    authorized=False,
                )
            )
        return claims

    def _subjects(
        self,
        mentions: tuple[Mention, ...],
        previous: set[str] | None,
        actor: str,
    ) -> set[str]:
        named = {item.person for item in mentions if item.person and not item.pronoun}
        first = any(item.person == actor for item in mentions if item.pronoun)
        if named and first:
            return named | {actor}
        if named:
            return named
        if first:
            return {actor}
        # “他/她”若仍有多位候选，就不猜；返回空集，让该分句保留为待澄清。
        third = [item for item in mentions if item.pronoun and item.person is None]
        if third and not previous:
            return set()
        return set(previous) if previous else {actor}

    def _clause_claims(
        self,
        message: Message,
        sentence: Sentence,
        clause: Clause,
        mentions: tuple[Mention, ...],
        subjects: set[str],
    ) -> list[Claim]:
        actor = message.actor
        text = clause.text
        stations = clause.stations
        if not subjects:
            return [
                self._claim(
                    message,
                    "pronoun_unclear",
                    text,
                    tuple(),
                    {"candidates": sorted(
                        {person for item in mentions for person in item.candidates}
                    )},
                    status=UNRESOLVED,
                    authorized=False,
                    note="“他/她”仍有多位候选，需客服先问清指谁。",
                )
            ]
        claimant = next(iter(subjects)) if len(subjects) == 1 else actor
        claims: list[Claim] = []

        if JOINT.search(text) and BINDING.search(text) and len(subjects) >= 2:
            group = subjects
            authorized = all(self._may_confirm(actor, person) for person in group)
            claims.append(
                self._claim(
                    message,
                    "joint",
                    text,
                    tuple(sorted(group)),
                    {"group": sorted(group), "same_train": True},
                    status=CONFIRMED if authorized else RELAYED,
                    authorized=authorized,
                    note=(
                        "同车是已确认的硬约束。"
                        if authorized
                        else "替同行人转述同车要求，须本人确认。"
                    ),
                )
            )

        if re.search(r"两张|两张票|张票", text) and JOINT.search(text):
            orders = self._orders_together(actor)
            if len(orders) >= 2:
                persons = tuple(sorted({order.person for order in orders}))
                authorized = all(self._may_confirm(actor, person) for person in persons)
                claims.append(
                    self._claim(
                        message,
                        "joint",
                        text,
                        persons,
                        {
                            "group": list(persons),
                            "orders": sorted(order.id for order in orders),
                            "same_train": True,
                        },
                        status=CONFIRMED if authorized else RELAYED,
                        authorized=authorized,
                        note="账户联系人逐单确认同车；不是对整套新方案的确认。",
                    )
                )

        if re.search(r"座位|位子", text) and re.search(r"别|不要|不能让|保留", text):
            claims.append(
                self._claim(
                    message,
                    "seat_protected",
                    text,
                    (claimant,),
                    {"orders": [order.id for order in self.workbook.orders_of(claimant)]},
                    status=CONFIRMED if claimant == actor else RELAYED,
                    authorized=claimant == actor,
                    note="本人要求保留现有座位。",
                )
            )
        elif re.search(r"座位|位子", text) and CONSENT.search(text):
            claims.append(
                self._claim(
                    message,
                    "seating_confirmed",
                    text,
                    (claimant,),
                    {"orders": [order.id for order in self.workbook.orders_of(claimant)]},
                    status=CONFIRMED if claimant == actor else RELAYED,
                    authorized=claimant == actor,
                    note="维持此前说过的选座意见；不等同于恢复券的转让。",
                )
            )

        if re.search(r"替我答复|代我答复|替我确认|代我确认", text):
            allowed = self._selectors_from(text, actor)
            claims.append(
                self._claim(
                    message,
                    "authority_limit",
                    text,
                    (actor,),
                    {
                        "selectors": sorted(allowed),
                        "may_confirm": False,
                    },
                    status=CONFIRMED,
                    authorized=True,
                    note="可以请人代选，但必须本人确认。",
                )
            )

        exclusive = re.search(r"只(?:走|到|能到|接受|坐)([^，。；]*)", text)
        if exclusive and stations:
            allowed = {station for station in stations if station in exclusive.group(1)}
            if allowed:
                claims.append(
                    self._claim(
                        message,
                        "boarding_allowed",
                        text,
                        (claimant,),
                        {"stations": sorted(allowed), "exclusive": True},
                        status=CONFIRMED if claimant == actor else RELAYED,
                        authorized=claimant == actor,
                        note="本人只接受这些上车站。" if claimant == actor else "转述，须本人确认。",
                    )
                )

        accepted = re.search(r"([一-鿿]{2,4})也成", text)
        if accepted and stations and not self._is_separation(text):
            allowed = {station for station in stations if station in accepted.group(1)}
            if allowed:
                claims.append(
                    self._claim(
                        message,
                        "boarding_allowed",
                        text,
                        (claimant,),
                        {"stations": sorted(allowed), "exclusive": False},
                        status=CONFIRMED if claimant == actor else RELAYED,
                        authorized=claimant == actor,
                        note="本人接受该上车站。" if claimant == actor else "转述，须本人确认。",
                    )
                )
            answer = self._answer(message, text, allowed)
            if answer is not None:
                claims.append(answer)

        if re.search(r"上的是我|上车的是我", text):
            claims.append(
                self._claim(
                    message,
                    "identity",
                    text,
                    (actor,),
                    {"stations": sorted(stations)},
                    status=CONFIRMED,
                    authorized=True,
                    note="来信人自报为在该站上车的同名旅客；同名者不合并。",
                )
            )

        if re.search(r"我到", text) and stations:
            claims.append(
                self._claim(
                    message,
                    "alighting_confirmed",
                    text,
                    (claimant,),
                    {"stations": sorted(stations)},
                    status=CONFIRMED if claimant == actor else RELAYED,
                    authorized=claimant == actor,
                    note="本人确认下车站。",
                )
            )

        if self._is_separation(text):
            others = subjects - {actor}
            station = None
            for candidate in stations:
                if re.search(re.escape(candidate) + r".{0,4}上", text):
                    station = candidate
                    break
            ambiguous = [
                item
                for item in mentions
                if item.ambiguous and item.token.endswith("师傅")
            ]
            if station and len(others) > 1:
                claims.append(
                    self._claim(
                        message,
                        "proposal",
                        text,
                        tuple(sorted(others)),
                        {"kind": "board_and_separate", "station": station},
                        status=UNRESOLVED,
                        authorized=False,
                        note="提议对象有两位同名候选，需先说明是哪一位。",
                    )
                )
            elif station and len(others) == 1:
                subject = next(iter(others))
                if self._order_of(subject).boarding != station:
                    claims.append(
                        self._claim(
                            message,
                            "proposal",
                            text,
                            (subject,),
                            {"kind": "board_and_separate", "station": station},
                            status=PROPOSED,
                            authorized=False,
                            note="只是备选提议，本人答复前不生效。",
                        )
                    )
            elif station is None and ambiguous:
                pass

        if re.search(r"券", text) and not self._withdraws_benefit(sentence.text):
            transfer = self._transfer(message, text, mentions, stations)
            if transfer is not None:
                claims.append(transfer)
        return claims

    # ------------------------------------------------------------------
    # 判定辅助

    def _withdraws_benefit(self, text: str) -> bool:
        return bool(
            re.search(r"券", text)
            and re.search(r"先别转|别转|不要转|不转了|收回|作废", text)
            and not re.search(r"若|如果|要是", text)
        )

    @staticmethod
    def _is_separation(text: str) -> bool:
        return bool(re.search(r"可不等|不用等|可以先走|先走|不跟|分开", text))

    def _answer(
        self, message: Message, text: str, stations: set[str]
    ) -> Claim | None:
        if not CONSENT.search(text) or REFUSAL.search(text):
            return None
        for proposal in self.proposals.values():
            if proposal.status != "pending" or proposal.subject != message.actor:
                continue
            station = proposal.value.get("station")
            if station and station in stations:
                return self._claim(
                    message,
                    "proposal_answer",
                    text,
                    (message.actor,),
                    {"proposal": proposal.id, "accepted": True, "station": station},
                    status=CONFIRMED,
                    authorized=True,
                    note="提议人本人接受该备选。",
                )
        return None

    def _transfer(
        self,
        message: Message,
        text: str,
        mentions: tuple[Mention, ...],
        stations: tuple[str, ...],
    ) -> Claim | None:
        if not re.search(r"给|转|让给|送", text):
            return None
        benefit = self._benefit_of(message.actor)
        if benefit is None:
            return None
        marker = re.search(r"(?:就?给|转给|让给|送给)", text)
        recipient = None
        if marker:
            tail = text[marker.end() : marker.end() + 10]
            for item in mentions:
                if item.person and item.person != message.actor and item.token in tail:
                    recipient = item.person
                    break
        if recipient is None:
            recipient = next(
                (
                    item.person
                    for item in mentions
                    if item.person and item.person != message.actor
                ),
                None,
            )
        if recipient is None:
            return self._claim(
                message,
                "benefit_transfer",
                text,
                (message.actor,),
                {"benefit": benefit},
                status=UNRESOLVED,
                authorized=False,
                note="受让人不唯一，无法建立转让。",
            )
        condition = None
        conditional = re.search(
            r"(?:若|如果|要是)([^，。；]{0,16}?)(?:能到|可到|坐到|赶上)([一-鿿]{2,4})",
            text,
        )
        if conditional:
            for station in stations:
                if station in conditional.group(0):
                    condition = station
                    break
        return self._claim(
            message,
            "benefit_transfer",
            text,
            (message.actor, recipient),
            {
                "benefit": benefit,
                "holder": message.actor,
                "recipient": recipient,
                "condition_stop": condition,
            },
            status=PROPOSED,
            authorized=True,
            note=(
                "附条件转让，条件未满足前不得使用。"
                if condition
                else "转让表态，尚待确认。"
            ),
        )

    # ------------------------------------------------------------------
    # 状态落账

    def _apply(self, claim: Claim) -> None:
        if claim.kind == "joint" and claim.status == CONFIRMED:
            group = frozenset(claim.value["group"])
            if group not in self.joint_groups:
                self.joint_groups.append(group)
        elif claim.kind == "boarding_allowed" and claim.status == CONFIRMED:
            person = claim.subjects[0]
            rule = BoardingRule(
                stations=frozenset(claim.value["stations"]),
                exclusive=bool(claim.value.get("exclusive")),
                claim=claim.id,
                status=claim.status,
            )
            prior = self.boarding_claims.get(person)
            if prior is None or prior.exclusive or not rule.exclusive:
                self.boarding_claims[person] = rule
            self.boardings.setdefault(person, set()).update(rule.stations)
        elif claim.kind == "seat_protected" and claim.status == CONFIRMED:
            for order in claim.value["orders"]:
                self.protected[order] = claim.id
        elif claim.kind == "authority_limit" and claim.status == CONFIRMED:
            person = claim.subjects[0]
            self.trusted_selectors.setdefault(person, set()).update(
                claim.value.get("selectors", [])
            )
            prior = self.authority_limits.get(
                person, {"selectors": set(), "may_confirm": True}
            )
            prior["selectors"] = prior["selectors"] | set(
                claim.value.get("selectors", [])
            )
            prior["may_confirm"] = prior["may_confirm"] and bool(
                claim.value.get("may_confirm", True)
            )
            self.authority_limits[person] = prior
        elif claim.kind == "proposal" and claim.status == PROPOSED:
            self._proposal_seq += 1
            proposal = Proposal(
                id=f"q{self._proposal_seq}",
                message=claim.message,
                subject=claim.subjects[0],
                kind=claim.value.get("kind", "proposal"),
                value=dict(claim.value),
                quote=claim.quote,
            )
            self.proposals[proposal.id] = proposal
        elif claim.kind == "proposal_answer" and claim.status == CONFIRMED:
            proposal = self.proposals.get(claim.value["proposal"])
            if proposal is not None and proposal.status == "pending":
                proposal.status = ANSWERED
                proposal.answered_by = claim.message
                station = proposal.value.get("station")
                if station:
                    self.boardings.setdefault(proposal.subject, set()).add(station)
        elif claim.kind == "benefit_transfer" and claim.status == PROPOSED:
            self._grant(claim)
        elif claim.kind == "benefit_withdrawn" and claim.status == CONFIRMED:
            self._withdraw(claim)
        elif claim.kind == "identity":
            self.identity_notes.append(claim.note)
        elif claim.status == UNRESOLVED:
            self.unresolved.append(claim)

    def _grant(self, claim: Claim) -> None:
        benefit = claim.value["benefit"]
        row = self.transfers.get(benefit)
        if row is None:
            row = TransferState(
                benefit=benefit,
                holder=claim.value["holder"],
                recipient=claim.value["recipient"],
                condition_stop=claim.value.get("condition_stop"),
            )
            self.transfers[benefit] = row
        if row.status == "withdrawn":
            self.conflicts.append(
                Conflict(
                    topic="benefit_transfer",
                    message=claim.message,
                    summary=f"{benefit} 曾由持券人收回，现又提出新的转让。",
                    kept=("alighting_confirmed", "boarding_allowed"),
                    question=(
                        f"{row.holder} 是否确认将 {benefit} 转给 "
                        f"{claim.value['recipient']}？"
                    ),
                )
            )
            return
        row.grants.append(claim.id)
        row.recipient = claim.value["recipient"]
        row.condition_stop = claim.value.get("condition_stop")
        row.status = "granted"

    def _withdraw(self, claim: Claim) -> None:
        benefit = claim.value["benefit"]
        row = self.transfers.get(benefit)
        if row is None:
            row = TransferState(
                benefit=benefit,
                holder=claim.subjects[0],
                recipient=None,
                condition_stop=None,
            )
            self.transfers[benefit] = row
        row.withdrawals.append(claim.id)
        row.status = "withdrawn"
        if row.grants:
            self.conflicts.append(
                Conflict(
                    topic="benefit_transfer",
                    message=claim.message,
                    summary=(
                        f"{benefit} 的转让先被提出，后由持券人收回；只保留收回后的"
                        "状态，已确认的座位与下车站条件不受影响。"
                    ),
                    kept=("boarding_allowed", "alighting_confirmed", "seat_protected"),
                    question=(
                        f"{row.holder} 是否同意重新转出 {benefit}，"
                        "或改为本人持券乘车？"
                    ),
                )
            )

    def _refresh_questions(self) -> None:
        questions: list[str] = []
        for row in self.proposals.values():
            if row.status == "pending":
                questions.append(
                    f"向 {row.subject} 确认：{row.quote} 中的备选是否接受？"
                )
        for row in self.transfers.values():
            if row.status == "withdrawn":
                questions.append(
                    f"{row.holder} 已收回 {row.benefit} 的转让；"
                    "任何使用该券的方案都必须重新取得本人同意。"
                )
        for claim in self.claims:
            if claim.kind == "authority_limit" and claim.status == CONFIRMED:
                if claim.value.get("may_confirm") is False:
                    questions.append(
                        f"{claim.subjects[0]} 声明不接受他人代答；"
                        "办理前须由本人确认。"
                    )
        for claim in self.unresolved:
            questions.append(f"需澄清（{claim.message}）：{claim.quote}")
        self.questions = list(dict.fromkeys(questions))

    # ------------------------------------------------------------------
    # 权限与辅助

    def _may_confirm(self, actor: str, person: str) -> bool:
        if actor == person:
            return True
        if self.authority_limits.get(person, {}).get("may_confirm") is False:
            return False
        return any(
            order.account == actor for order in self.workbook.orders_of(person)
        )

    def _orders_together(self, actor: str) -> tuple[Order, ...]:
        orders = list(self.workbook.orders_of(actor))
        orders.extend(
            order
            for order in self.workbook.orders.values()
            if order.account == actor and order.person != actor
        )
        if not orders:
            return ()
        service = orders[0].service
        return tuple(
            sorted(
                {order.id: order for order in orders if order.service == service}.values(),
                key=lambda row: row.id,
            )
        )

    def _benefit_of(self, actor: str) -> str | None:
        benefits = self.workbook.benefits_of(actor)
        return benefits[0].id if len(benefits) == 1 else None

    def _benefits_of(self, actor: str) -> tuple[str, ...]:
        return tuple(benefit.id for benefit in self.workbook.benefits_of(actor))

    def _order_of(self, person: str) -> Order:
        return self.workbook.orders[self.workbook.person(person).orders[0]]

    def _selectors_from(self, text: str, actor: str) -> set[str]:
        match = re.search(r"等([^，。；]{1,12}?)(?:挑|选|定|决定)", text)
        if not match:
            return set()
        chunk = match.group(1)
        result: set[str] = set()
        for token, persons in self.workbook.alias_map().items():
            if token in chunk:
                result.update(persons)
        if not result:
            for surname, persons in self.workbook.surname_map().items():
                if surname in chunk:
                    result.update(persons)
        result.discard(actor)
        return result

    def _claim(
        self,
        message: Message,
        kind: str,
        quote: str,
        subjects: Iterable[str],
        value: Mapping[str, Any],
        *,
        status: str,
        authorized: bool,
        note: str = "",
    ) -> Claim:
        self._seq += 1
        return Claim(
            id=f"c{self._seq}",
            message=message.id,
            actor=message.actor,
            kind=kind,
            quote=quote,
            subjects=tuple(subjects),
            value=dict(value),
            status=status,
            authorized=authorized,
            note=note,
        )

    # ------------------------------------------------------------------
    # 查询

    def binding(self, kind: str) -> list[Claim]:
        return [
            claim for claim in self.claims if claim.binding and claim.kind == kind
        ]

    def joint_of(self, person: str) -> frozenset[str]:
        for group in self.joint_groups:
            if person in group:
                return group
        return frozenset({person})

    def joint_people(self) -> set[str]:
        return {person for group in self.joint_groups for person in group}

    def boarding_rule(self, person: str) -> BoardingRule | None:
        return self.boarding_claims.get(person)


def _surname(token: str) -> str | None:
    match = re.match(r"(?:老|小)?([一-鿿])", token)
    return match.group(1) if match else None
