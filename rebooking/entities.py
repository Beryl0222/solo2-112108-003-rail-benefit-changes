"""称呼与实体消歧。

昵称、姓氏加称呼和代词都先展开成候选集合。只有会话上下文能排除到一人时
才绑定到 person；两位同名“陈师傅”不会因为名字相同而被并成一位。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from .model import Order, Workbook

SURNAME_CALL = re.compile(r"老([一-鿿])|小([一-鿿])|([一-鿿])师傅")


@dataclass(frozen=True)
class Mention:
    token: str
    start: int
    end: int
    candidates: frozenset[str]
    person: str | None
    basis: str
    pronoun: bool = False

    @property
    def ambiguous(self) -> bool:
        return self.person is None and len(self.candidates) > 1


class Resolver:
    """按一份资料和一段上下文解析称呼。"""

    def __init__(self, workbook: Workbook):
        self.workbook = workbook
        self.aliases = workbook.alias_map()
        self.surnames = workbook.surname_map()

    def scan(self, text: str, speaker: str, context: tuple[str, ...] = ()) -> list[Mention]:
        """扫描一句话。

        context 是本段会话中已经明确出现的 person，用来消解“他/她”和未限定
        的姓氏称呼。说话人自己的“我/我们”直接绑定发言人。
        """
        taken: list[tuple[int, int]] = []
        found: list[Mention] = []
        for token in sorted(self.aliases, key=len, reverse=True):
            start = 0
            while True:
                at = text.find(token, start)
                if at < 0:
                    break
                end = at + len(token)
                if not any(at < right and left < end for left, right in taken):
                    people = self.aliases[token]
                    found.append(
                        Mention(
                            token=token,
                            start=at,
                            end=end,
                            candidates=people,
                            person=next(iter(people)) if len(people) == 1 else None,
                            basis="客服名片唯一" if len(people) == 1 else "同名，待限定",
                        )
                    )
                    taken.append((at, end))
                start = end
        for match in SURNAME_CALL.finditer(text):
            at, end = match.span()
            if any(at < right and left < end for left, right in taken):
                continue
            surname = next(group for group in match.groups() if group)
            people = self.surnames.get(surname, frozenset())
            if not people:
                continue
            found.append(
                Mention(
                    token=match.group(0),
                    start=at,
                    end=end,
                    candidates=people,
                    person=next(iter(people)) if len(people) == 1 else None,
                    basis="姓氏称呼" if len(people) == 1 else "同姓，待限定",
                )
            )
            taken.append((at, end))
        for token, person, pronoun in (
            ("我们", speaker, True),
            ("我", speaker, True),
            ("自己", speaker, True),
            ("他", None, True),
            ("她", None, True),
        ):
            start = 0
            while True:
                at = text.find(token, start)
                if at < 0:
                    break
                end = at + len(token)
                if not any(at < right and left < end for left, right in taken):
                    if pronoun and person is not None:
                        candidates = frozenset({person})
                    else:
                        candidates = frozenset(context)
                    found.append(
                        Mention(
                            token=token,
                            start=at,
                            end=end,
                            candidates=candidates,
                            person=person,
                            basis="发言人自称" if person else "代词，依上下文",
                            pronoun=pronoun,
                        )
                    )
                    taken.append((at, end))
                start = end
        return sorted(found, key=lambda item: (item.start, item.end))

    def narrow_by_station(
        self,
        mentions: list[Mention],
        clause_text: str,
        stations: Iterable[str],
    ) -> list[Mention]:
        """用同一个分句内的“站+下车/上车”限定同姓或同名候选。

        只在分句内限定，避免把“外婆只走西桥”套到同一句里另外提到的老陈。
        """
        text = clause_text
        cues: list[tuple[str, str]] = []
        for station in stations:
            for verb, field in (("下车", "alighting"), ("上车", "boarding")):
                if re.search(re.escape(station) + r".{0,4}" + verb, text):
                    cues.append((station, field))
                if re.search(verb + r".{0,4}" + re.escape(station), text):
                    cues.append((station, field))
        if not cues:
            return mentions
        result = []
        for mention in mentions:
            if mention.person is not None:
                result.append(mention)
                continue
            narrowed = set(mention.candidates)
            for station, field in cues:
                narrowed = {
                    person
                    for person in narrowed
                    if getattr(
                        self._order_of(person), field
                    )
                    == station
                }
            if len(narrowed) == 1:
                result.append(
                    Mention(
                        token=mention.token,
                        start=mention.start,
                        end=mention.end,
                        candidates=frozenset(narrowed),
                        person=next(iter(narrowed)),
                        basis="句内车站限定",
                        pronoun=mention.pronoun,
                    )
                )
            elif 0 < len(narrowed) < len(mention.candidates):
                result.append(
                    Mention(
                        token=mention.token,
                        start=mention.start,
                        end=mention.end,
                        candidates=frozenset(narrowed),
                        person=None,
                        basis="车站限定后仍不唯一",
                        pronoun=mention.pronoun,
                    )
                )
            else:
                result.append(mention)
        return result

    def cross_sentence(
        self, mentions: list[Mention], known: Mapping[str, str]
    ) -> list[Mention]:
        """同一封信里别处已用车站限定的姓氏，可延续到未限定的称呼。"""
        result = []
        for mention in mentions:
            if mention.person is not None or len(mention.candidates) <= 1:
                result.append(mention)
                continue
            surname = _surname(mention.token)
            answer = known.get(surname) if surname else None
            if answer is not None and answer in mention.candidates:
                result.append(
                    Mention(
                        token=mention.token,
                        start=mention.start,
                        end=mention.end,
                        candidates=frozenset({answer}),
                        person=answer,
                        basis="同封信前后文",
                        pronoun=mention.pronoun,
                    )
                )
            else:
                result.append(mention)
        return result

    def _order_of(self, person: str) -> Order:
        """取该人最早的订单，用于按上/下车站限定同姓候选。"""
        return self.workbook.orders_of(person)[0]


def _surname(token: str) -> str | None:
    match = re.match(r"(?:老|小)?([一-鿿])", token)
    return match.group(1) if match else None
