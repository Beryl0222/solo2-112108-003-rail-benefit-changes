"""票务资料模型与读取。

六份资料都是合成数据。读取时只校验引用和明显矛盾，不改动记录；办改乘产生
的座位占用、订单变化与券核销由 :mod:`rebooking.desk` 单独保存。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


class DataError(ValueError):
    """资料缺失、重复或互相矛盾。"""


@dataclass(frozen=True)
class Order:
    """票务原始订单。account 是变更权限，不代表乘车人本人。"""

    id: str
    person: str
    account: str | None
    aliases: tuple[str, ...]
    boarding: str
    alighting: str
    service: str
    benefit: str | None


@dataclass(frozen=True)
class Benefit:
    """体验券规则。recipient_scope 说明可改到哪位同行人。"""

    id: str
    holder: str | None
    recipient_scope: str
    required_stop: str
    feature: str

    @property
    def holder_only(self) -> bool:
        return self.recipient_scope == "本人"


@dataclass(frozen=True)
class Train:
    """替代班次。accepted 为 False 的班次不得用于改乘。"""

    id: str
    departures: str
    stops: tuple[str, ...]
    seats: int
    features: tuple[str, ...]
    accepted: bool

    def stop_index(self, stop: str) -> int:
        if stop not in self.stops:
            raise DataError(f"{self.id} 不停靠 {stop}")
        return self.stops.index(stop)

    def serves(self, boarding: str, alighting: str) -> bool:
        if boarding not in self.stops or alighting not in self.stops:
            return False
        return self.stops.index(boarding) < self.stops.index(alighting)

    def interval(self, boarding: str, alighting: str) -> tuple[str, ...]:
        start = self.stop_index(boarding)
        end = self.stop_index(alighting)
        if start >= end:
            raise DataError(f"{self.id} 的 {boarding}→{alighting} 方向不成立")
        return self.stops[start : end + 1]


@dataclass(frozen=True)
class Notice:
    id: str
    author: str
    at: str
    reply: str | None
    text: str


@dataclass(frozen=True)
class Message:
    id: str
    actor: str
    reply: str | None
    at: str
    text: str


@dataclass(frozen=True)
class Counter:
    """票务系统独立成交；seat_delta 计入余位。"""

    at: str
    service: str
    seat_delta: int
    request: str


@dataclass(frozen=True)
class Person:
    """订单中的一位旅客。

    同名的两位旅客是不同的 person；只有相同 person 的订单才属于同一人。
    客服名片（aliases）只用于指代和消歧，不能把同名订单并成一位。
    """

    id: str
    aliases: tuple[str, ...]
    orders: tuple[str, ...]
    accounts: frozenset[str]
    surname: str

    def called(self, token: str) -> bool:
        return token == self.id or token in self.aliases

    @property
    def label(self) -> str:
        names = "、".join(self.aliases) if self.aliases else self.id
        return f"{self.id}（{names}）"


@dataclass(frozen=True)
class Workbook:
    orders: Mapping[str, Order]
    people: Mapping[str, Person]
    benefits: Mapping[str, Benefit]
    trains: Mapping[str, Train]
    notices: Mapping[str, Notice]
    messages: Mapping[str, Message]
    counter: tuple[Counter, ...]

    def order(self, order_id: str) -> Order:
        try:
            return self.orders[order_id]
        except KeyError as error:
            raise DataError(f"没有订单 {order_id}") from error

    def person(self, person_id: str) -> Person:
        try:
            return self.people[person_id]
        except KeyError as error:
            raise DataError(f"没有旅客 {person_id}") from error

    def train(self, train_id: str) -> Train:
        try:
            return self.trains[train_id]
        except KeyError as error:
            raise DataError(f"没有班次 {train_id}") from error

    def benefit(self, benefit_id: str) -> Benefit:
        try:
            return self.benefits[benefit_id]
        except KeyError as error:
            raise DataError(f"没有体验券 {benefit_id}") from error

    def orders_of(self, person_id: str) -> tuple[Order, ...]:
        return tuple(
            sorted(
                (
                    order
                    for order in self.orders.values()
                    if order.person == person_id
                ),
                key=lambda row: row.id,
            )
        )

    def orders_with(self, person_id: str) -> tuple[Order, ...]:
        """以本人为乘车人或订单联系人的全部订单。"""
        return tuple(
            order
            for order in self.orders.values()
            if order.person == person_id or order.account == person_id
        )

    def benefits_of(self, person_id: str) -> tuple[Benefit, ...]:
        return tuple(
            self.benefits[key]
            for key in sorted(self.benefits)
            if self.benefits[key].holder == person_id
        )

    def alias_map(self) -> dict[str, frozenset[str]]:
        index: dict[str, set[str]] = {}
        for person in self.people.values():
            index.setdefault(person.id, set()).add(person.id)
            for alias in person.aliases:
                index.setdefault(alias, set()).add(person.id)
        return {token: frozenset(persons) for token, persons in index.items()}

    def surname_map(self) -> dict[str, frozenset[str]]:
        index: dict[str, set[str]] = {}
        for person in self.people.values():
            if person.surname:
                index.setdefault(person.surname, set()).add(person.id)
        return {token: frozenset(persons) for token, persons in index.items()}

    def stations(self) -> frozenset[str]:
        found: set[str] = set()
        for order in self.orders.values():
            found.update({order.boarding, order.alighting})
        for train in self.trains.values():
            found.update(train.stops)
        for benefit in self.benefits.values():
            found.add(benefit.required_stop)
        return frozenset(found)

    def message_context(self, message_id: str) -> tuple[Message, ...]:
        """沿 reply 上溯，按最早到本条返回。"""
        try:
            current: Message | None = self.messages[message_id]
        except KeyError as error:
            raise DataError(f"没有消息 {message_id}") from error
        chain: list[Message] = []
        seen: set[str] = set()
        while current is not None:
            if current.id in seen:
                raise DataError(f"消息 {current.id} 的回复链成环")
            seen.add(current.id)
            chain.append(current)
            if current.reply is None:
                break
            try:
                current = self.messages[current.reply]
            except KeyError as error:
                raise DataError(
                    f"消息 {current.id} 指向不存在的 {current.reply}"
                ) from error
        chain.reverse()
        return tuple(chain)

    def ordered_messages(self) -> tuple[Message, ...]:
        """父消息先于回复，其次按时间与编号排序。"""
        pending = {key: message for key, message in self.messages.items()}
        ordered: list[Message] = []
        while pending:
            ready = [
                message
                for message in pending.values()
                if message.reply is None or message.reply in {row.id for row in ordered}
            ]
            if not ready:
                raise DataError("消息回复链无法排序")
            ready.sort(key=lambda row: (row.at, row.id))
            chosen = ready[0]
            ordered.append(chosen)
            del pending[chosen.id]
        return tuple(ordered)


def _read_rows(root: Path, name: str) -> list[dict]:
    path = root / f"{name}.jsonl"
    try:
        with path.open(encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]
    except OSError as error:
        raise DataError(f"读不到 {path}") from error
    except json.JSONDecodeError as error:
        raise DataError(f"{path} 含无效 JSON") from error


def _surname_of(aliases: Iterable[str]) -> str:
    for alias in aliases:
        if len(alias) >= 2 and alias[0] not in "老小":
            return alias[0]
    for alias in aliases:
        stripped = alias.lstrip("老小")
        if stripped:
            return stripped[0]
    return ""


def _build_people(orders: Mapping[str, Order]) -> dict[str, Person]:
    grouped: dict[str, list[Order]] = {}
    for order in orders.values():
        grouped.setdefault(order.person, []).append(order)
    people: dict[str, Person] = {}
    for person_id, rows in grouped.items():
        rows.sort(key=lambda row: row.id)
        aliases: list[str] = []
        for row in rows:
            for alias in row.aliases:
                if alias not in aliases:
                    aliases.append(alias)
        accounts = {row.account for row in rows if row.account is not None}
        people[person_id] = Person(
            id=person_id,
            aliases=tuple(aliases),
            orders=tuple(row.id for row in rows),
            accounts=frozenset(accounts),
            surname=_surname_of(aliases),
        )
    return people


def load_workbook(directory: str | Path = "data") -> Workbook:
    root = Path(directory)
    try:
        manifest = json.loads((root / "index.json").read_text(encoding="utf-8"))
    except OSError as error:
        raise DataError(f"读不到资料目录 {root}") from error
    raw = {name: _read_rows(root, name) for name in manifest}

    orders: dict[str, Order] = {}
    for row in raw["orders"]:
        if row["id"] in orders:
            raise DataError(f"订单 {row['id']} 重复")
        orders[row["id"]] = Order(
            id=row["id"],
            person=row["person"],
            account=row["account"],
            aliases=tuple(row["aliases"]),
            boarding=row["boarding"],
            alighting=row["alighting"],
            service=row["service"],
            benefit=row["benefit"],
        )
    people = _build_people(orders)

    benefits: dict[str, Benefit] = {}
    for row in raw["benefits"]:
        if row["id"] in benefits:
            raise DataError(f"体验券 {row['id']} 重复")
        if row["holder"] is not None and row["holder"] not in people:
            raise DataError(f"体验券 {row['id']} 的持有人不在订单中")
        benefits[row["id"]] = Benefit(
            id=row["id"],
            holder=row["holder"],
            recipient_scope=row["recipient_scope"],
            required_stop=row["required_stop"],
            feature=row["feature"],
        )
    for order in orders.values():
        if order.benefit is not None and order.benefit not in benefits:
            raise DataError(f"订单 {order.id} 的体验券 {order.benefit} 不存在")
        if order.benefit is not None:
            holder = benefits[order.benefit].holder
            if holder is not None and holder != order.person:
                raise DataError(f"订单 {order.id} 的体验券持有人与乘车人不一致")

    trains: dict[str, Train] = {}
    for row in raw["trains"]:
        if row["id"] in trains:
            raise DataError(f"班次 {row['id']} 重复")
        if row["seats"] < 0:
            raise DataError(f"班次 {row['id']} 的余位为负")
        if not row["stops"]:
            raise DataError(f"班次 {row['id']} 没有停站")
        trains[row["id"]] = Train(
            id=row["id"],
            departures=row["departures"],
            stops=tuple(row["stops"]),
            seats=row["seats"],
            features=tuple(row["features"]),
            accepted=bool(row["accepted"]),
        )
    for benefit in benefits.values():
        if benefit.required_stop not in {
            stop for train in trains.values() for stop in train.stops
        }:
            raise DataError(f"体验券 {benefit.id} 的要求站不存在")

    notices: dict[str, Notice] = {}
    for row in raw["notices"]:
        if row["id"] in notices:
            raise DataError(f"通告 {row['id']} 重复")
        notices[row["id"]] = Notice(
            id=row["id"],
            author=row["author"],
            at=row["at"],
            reply=row["reply"],
            text=row["text"],
        )
    for notice in notices.values():
        if notice.reply is not None and notice.reply not in notices:
            raise DataError(f"通告 {notice.id} 指向不存在的 {notice.reply}")

    messages: dict[str, Message] = {}
    for row in raw["messages"]:
        if row["id"] in messages:
            raise DataError(f"消息 {row['id']} 重复")
        if row["actor"] not in people:
            raise DataError(f"消息 {row['id']} 的 actor {row['actor']} 不在订单中")
        messages[row["id"]] = Message(
            id=row["id"],
            actor=row["actor"],
            reply=row["reply"],
            at=row["at"],
            text=row["text"],
        )
    for message in messages.values():
        if message.reply is not None and message.reply not in messages:
            raise DataError(f"消息 {message.id} 指向不存在的 {message.reply}")

    counter: list[Counter] = []
    for row in raw["counter"]:
        if row["service"] not in trains:
            raise DataError(f"票务操作引用了不存在的班次 {row['service']}")
        counter.append(
            Counter(
                at=row["at"],
                service=row["service"],
                seat_delta=int(row["seat_delta"]),
                request=row["request"],
            )
        )

    return Workbook(
        orders=orders,
        people=people,
        benefits=benefits,
        trains=trains,
        notices=notices,
        messages=messages,
        counter=tuple(counter),
    )
