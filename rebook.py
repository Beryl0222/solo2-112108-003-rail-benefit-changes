"""命令行：核对资料、恢复会话约定、列出方案并办理改乘。

用法示例：
    python3 rebook.py status
    python3 rebook.py plans
    python3 rebook.py confirm P-b99ccd49 --actor p1 --actor p3 --actor p5 --request B90
    python3 rebook.py lookup B90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rebooking import Conversation, Desk, Rejected, load_workbook
from rebooking.desk import BenefitUse, Leg, Outcome


def _desk(args: argparse.Namespace) -> Desk:
    workbook = load_workbook(args.data)
    conversation = Conversation(workbook)
    conversation.read_all()
    desk = Desk(workbook, conversation)
    _load(desk, Path(args.state))
    return desk


def _load(desk: Desk, path: Path) -> None:
    """读取上次办理留下的结果；命令行每次运行都是新进程。"""
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("outcomes", []):
        outcome = Outcome(
            request=row["request"],
            plan_id=row["plan"],
            actors=tuple(row["actors"]),
            legs=tuple(
                Leg(
                    order=item["order"],
                    person=item["person"],
                    train=item["train"],
                    boarding=item["boarding"],
                    alighting=item["alighting"],
                    changed=item["changed"],
                )
                for item in row["legs"]
            ),
            benefits=tuple(
                BenefitUse(
                    benefit=item["benefit"],
                    holder=item["holder"],
                    rider=item["rider"],
                    train=item["train"],
                    transfer=item["transfer"],
                    authorized_by=item.get("authorized_by"),
                    note=item.get("note", ""),
                )
                for item in row["benefits"]
            ),
            seats_used=row["seats_used"],
            fingerprint=row["fingerprint"],
        )
        desk.restore(outcome)


def _save(desk: Desk, path: Path) -> None:
    payload = {
        "outcomes": [outcome.describe() for outcome in desk.outcomes()],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _print(payload) -> None:
    if hasattr(payload, "describe"):
        payload = payload.describe()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def cmd_status(args: argparse.Namespace) -> int:
    desk = _desk(args)
    _print(
        {
            "notice": desk.rules.describe(),
            "affected": [order.id for order in desk.affected()],
            "joint": [sorted(group) for group in desk.conversation.joint_groups],
            "boardings": {
                person: sorted(rule.stations)
                for person, rule in sorted(desk.conversation.boarding_claims.items())
            },
            "transfers": {
                key: {"status": state.status, "recipient": state.recipient}
                for key, state in sorted(desk.conversation.transfers.items())
            },
            "proposals": {
                key: {"subject": row.subject, "status": row.status}
                for key, row in sorted(desk.conversation.proposals.items())
            },
            "conflicts": [item.describe() for item in desk.conversation.conflicts],
            "questions": desk.questions(),
        }
    )
    return 0


def cmd_plans(args: argparse.Namespace) -> int:
    desk = _desk(args)
    _print([plan.describe() for plan in desk.draft(limit=args.limit)])
    return 0


def cmd_confirm(args: argparse.Namespace) -> int:
    desk = _desk(args)
    plan = next(
        (row for row in desk.draft(limit=200) if row.id == args.plan),
        None,
    )
    if plan is None:
        print(f"没有方案 {args.plan}", file=sys.stderr)
        return 2
    result = desk.confirm(plan, args.actor, args.request)
    if not isinstance(result, Rejected):
        _save(desk, Path(args.state))
    _print(result)
    return 0 if not isinstance(result, Rejected) else 1


def cmd_lookup(args: argparse.Namespace) -> int:
    desk = _desk(args)
    outcome = desk.lookup(args.request)
    if outcome is None:
        print(f"回复 {args.request} 没有办理记录", file=sys.stderr)
        return 2
    _print(outcome)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="主题列车改乘值班台")
    parser.add_argument("--data", default="data", help="含索引的资料目录")
    parser.add_argument(
        "--state", default="rebook-state.json", help="保存已办理结果的 JSON 文件"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="显示通告范围、约定与待答问题")
    status.set_defaults(func=cmd_status)

    plans = sub.add_parser("plans", help="列出候选的整套改乘方案")
    plans.add_argument("--limit", type=int, default=5)
    plans.set_defaults(func=cmd_plans)

    confirm = sub.add_parser("confirm", help="按回复确认整套方案")
    confirm.add_argument("plan")
    confirm.add_argument("--actor", action="append", default=[])
    confirm.add_argument("--request", required=True)
    confirm.set_defaults(func=cmd_confirm)

    lookup = sub.add_parser("lookup", help="按回复号查询办理结果")
    lookup.add_argument("request")
    lookup.set_defaults(func=cmd_lookup)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
