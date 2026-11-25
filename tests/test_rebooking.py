"""改乘助手的行为测试：同行、转述、提议、收回、重核与一致查询。"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from rebooking import Desk, Conversation, load_workbook
from rebooking.desk import BenefitUse, Plan, Rejected

ROOT = Path(__file__).resolve().parents[1] / "data"


def build(workbook=None):
    wb = workbook or load_workbook(ROOT)
    conversation = Conversation(wb)
    conversation.read_all()
    return wb, conversation, Desk(wb, conversation)


def without(workbook, *message_ids):
    remaining = {
        key: value
        for key, value in workbook.messages.items()
        if key not in message_ids
    }
    return replace(workbook, messages=remaining)


class IdentityTests(unittest.TestCase):
    def test_same_nickname_is_not_merged(self):
        wb, conv, _ = build()
        self.assertNotEqual(conv.resolver.aliases["陈师傅"], frozenset({"p3"}))
        self.assertEqual(
            conv.resolver.aliases["陈师傅"], frozenset({"p3", "p4"})
        )

    def test_station_cue_separates_two_chen(self):
        wb, conv, _ = build()
        m3 = [claim for claim in conv.claims if claim.message == "m3"]
        # m3 自报石湾上车的是 p4，不并入 p3。
        self.assertEqual({claim.subjects for claim in m3}, {("p4",)})
        self.assertIn("identity", {claim.kind for claim in m3})

    def test_person_can_have_several_orders(self):
        wb, conv, _ = build()
        # p1 是 b1 的乘车人、b2 的账户联系人；不能把 p2 并进 p1。
        self.assertNotEqual(wb.order("b1").person, wb.order("b2").person)
        self.assertEqual(wb.order("b2").account, "p1")


class ConversationTests(unittest.TestCase):
    def test_joint_is_confirmed_hard_constraint(self):
        wb, conv, _ = build()
        groups = {tuple(sorted(group)) for group in conv.joint_groups}
        self.assertIn(("p1", "p2"), groups)

    def test_relayed_boarding_is_not_binding(self):
        wb, conv, _ = build()
        relayed = [
            claim
            for claim in conv.claims
            if claim.kind == "boarding_allowed" and claim.status == "relayed"
        ]
        self.assertTrue(relayed)
        self.assertTrue(all(not claim.binding for claim in relayed))
        # 外婆本人随后确认只接受西桥或石湾。
        rule = conv.boarding_rule("p5")
        self.assertEqual(rule.stations, frozenset({"西桥", "石湾"}))
        self.assertTrue(rule.exclusive)

    def test_proposal_needs_answer_then_is_accepted(self):
        wb, conv, _ = build()
        (proposal,) = conv.proposals.values()
        self.assertEqual(proposal.subject, "p3")
        self.assertEqual(proposal.status, "answered")
        self.assertEqual(proposal.value["station"], "东城")

    def test_unanswered_proposal_stays_pending(self):
        wb = load_workbook(ROOT)
        wb = without(wb, "m2", "m5", "m6")
        conversation = Conversation(wb)
        conversation.read_all()
        (proposal,) = conversation.proposals.values()
        self.assertEqual(proposal.status, "pending")
        self.assertTrue(
            any("p3" in question for question in conversation.questions)
        )
        # 未回答的提议不能变成上车站约束。
        self.assertIsNone(conversation.boarding_rule("p3"))

    def test_benefit_withdrawal_overrides_transfer(self):
        wb, conv, _ = build()
        state = conv.transfers["e3"]
        self.assertEqual(state.status, "withdrawn")
        self.assertFalse(state.active)
        self.assertTrue(conv.conflicts)
        self.assertTrue(
            any("e3" in question and "收回" in question for question in conv.questions)
        )

    def test_withdrawal_keeps_local_conditions(self):
        wb, conv, _ = build()
        conflict = conv.conflicts[0]
        self.assertIn("boarding_allowed", conflict.kept)
        self.assertIn("alighting_confirmed", conflict.kept)
        self.assertTrue(conflict.question)

    def test_account_holder_can_confirm_joint_but_not_replace_self(self):
        wb, conv, _ = build()
        self.assertTrue(conv._may_confirm("p1", "p2"))
        self.assertFalse(conv._may_confirm("p1", "p5"))
        self.assertFalse(conv._may_confirm("p1", "p3"))

    def test_authority_limit_blocks_relay_confirmation(self):
        wb = load_workbook(ROOT)
        wb = without(wb, "m4", "m6")
        conversation = Conversation(wb)
        conversation.read_all()
        # 没有 m4 时 p5 没有表态，p1 不是 p5 的账户联系人。
        self.assertFalse(conversation._may_confirm("p1", "p5"))


class PlanTests(unittest.TestCase):
    def test_only_published_trains_and_allowed_intervals(self):
        wb, conv, desk = build()
        allowed = desk.rules.candidate_trains()
        self.assertEqual(allowed, frozenset({"R74", "R75", "R81"}))
        for plan in desk.draft(limit=20):
            for leg in plan.legs:
                if not leg.changed:
                    continue
                train = wb.train(leg.train)
                self.assertTrue(train.accepted)
                self.assertIn(leg.train, allowed)
                self.assertTrue(train.serves(leg.boarding, leg.alighting))
                self.assertNotEqual(leg.train, "R88")

    def test_joint_group_stays_on_one_train(self):
        wb, conv, desk = build()
        for plan in desk.draft(limit=20):
            self.assertEqual(plan.leg("b1").train, plan.leg("b2").train)

    def test_elder_boarding_is_hard(self):
        wb, conv, desk = build()
        for plan in desk.draft(limit=20):
            self.assertIn(plan.leg("b5").boarding, {"西桥", "石湾"})

    def test_changed_order_keeps_alighting(self):
        wb, conv, desk = build()
        for plan in desk.draft(limit=20):
            for leg in plan.legs:
                self.assertEqual(leg.alighting, wb.order(leg.order).alighting)

    def test_no_draft_uses_withdrawn_benefit_for_other_person(self):
        wb, conv, desk = build()
        self.assertEqual(conv.transfers["e3"].status, "withdrawn")
        for plan in desk.draft(limit=20):
            for use in plan.benefits:
                self.assertFalse(use.transfer)
                self.assertEqual(use.rider, use.holder)

    def test_conditional_transfer_used_when_valid(self):
        wb = load_workbook(ROOT)
        wb = without(wb, "m5")
        conversation = Conversation(wb)
        conversation.read_all()
        desk = Desk(wb, conversation)
        transferred = [
            (plan, use)
            for plan in desk.draft(limit=50)
            for use in plan.benefits
            if use.transfer
        ]
        self.assertTrue(transferred)
        for plan, use in transferred:
            self.assertEqual(use.rider, "p2")
            self.assertEqual(use.benefit, "e3")
            leg = plan.person_leg("p2")
            self.assertEqual(use.train, leg.train)
            # 同行同车依旧成立，且券的区间/场次符合。
            self.assertEqual(plan.leg("b1").train, leg.train)
            train = wb.train(leg.train)
            self.assertIn("演出", train.features)
            self.assertIn("石影", train.interval(leg.boarding, leg.alighting))

    def test_tradeoff_is_offered_not_taken(self):
        wb, conv, desk = build()
        plan = desk.draft()[0]
        self.assertTrue(any(item.code == "benefit_withdrawn" for item in plan.tradeoffs))
        self.assertFalse(any(use.transfer for use in plan.benefits))
        self.assertTrue(plan.tradeoffs[0].question)

    def test_benefit_needs_matching_feature_and_interval(self):
        wb, conv, desk = build()
        plan = desk.draft()[0]
        for use in plan.benefits:
            leg = plan.person_leg(use.rider)
            train = wb.train(use.train)
            benefit = wb.benefit(use.benefit)
            self.assertEqual(leg.train, use.train)
            self.assertIn(benefit.feature, train.features)
            self.assertIn(benefit.required_stop, train.interval(leg.boarding, leg.alighting))


class ConfirmationTests(unittest.TestCase):
    def test_full_confirmation_requires_each_traveller(self):
        wb, conv, desk = build()
        plan = desk.draft()[0]
        rejected = desk.confirm(plan, "p1", "B90")
        self.assertIsInstance(rejected, Rejected)
        self.assertTrue(any("p5" in reason for reason in rejected.reasons))
        self.assertIsNone(desk.lookup("B90"))

    def test_account_contact_confirms_their_group(self):
        wb, conv, desk = build()
        plan = desk.draft()[0]
        rejected = desk.confirm(plan, ["p1", "p3"], "B91")
        self.assertIsInstance(rejected, Rejected)
        self.assertFalse(any("'p2'" in reason for reason in rejected.reasons))
        self.assertTrue(any("p5" in reason for reason in rejected.reasons))

    def test_successful_confirmation_is_idempotent(self):
        wb, conv, desk = build()
        plan = desk.draft()[0]
        outcome = desk.confirm(plan, ["p1", "p3", "p5"], "B92")
        self.assertNotIsInstance(outcome, Rejected)
        again = desk.confirm(plan, ["p1", "p3", "p5"], "B92")
        self.assertIs(outcome, again)
        self.assertEqual(desk.lookup("B92").describe(), outcome.describe())

    def test_confirming_one_plan_with_another_request_fails(self):
        wb, conv, desk = build()
        first, second = desk.draft()[:2]
        desk.confirm(first, ["p1", "p3", "p5"], "B93")
        clash = desk.confirm(second, ["p1", "p3", "p5"], "B93")
        self.assertIsInstance(clash, Rejected)

    def test_seat_shortage_rejects_before_side_effects(self):
        wb, conv, desk = build()
        plan = desk.draft()[0]
        train_id = next(iter(plan.seats_used))
        desk.record_sale(train_id, desk.free_seats(train_id), "B60")
        rejected = desk.confirm(plan, ["p1", "p3", "p5"], "B94")
        self.assertIsInstance(rejected, Rejected)
        self.assertTrue(
            any(train_id in reason for reason in rejected.reasons), rejected.reasons
        )
        self.assertIsNone(desk.lookup("B94"))

    def test_transfer_withdrawal_expires_old_plan(self):
        wb = load_workbook(ROOT)
        wb = without(wb, "m5")
        conversation = Conversation(wb)
        conversation.read_all()
        desk = Desk(wb, conversation)
        plan = next(
            plan for plan in desk.draft(limit=20) if any(use.transfer for use in plan.benefits)
        )
        self.assertFalse(desk.check(plan))
        # 持券人收回。
        from rebooking.model import Message

        desk.read(
            Message(
                id="m9",
                actor="p3",
                reply="m2",
                at="2026-11-23T11:31:00+08:00",
                text="券先别转了，座位可以照刚说的选。",
            )
        )
        old, rejected, fresh = desk.recheck(plan.id)
        self.assertIsNone(old)
        self.assertIsNotNone(rejected)
        self.assertTrue(any("收回" in item for item in rejected.changed))
        self.assertTrue(rejected.questions)
        self.assertTrue(rejected.kept)
        for candidate in fresh:
            self.assertFalse(
                any(use.transfer for use in candidate.benefits if use.benefit == "e3")
            )

    def test_recheck_without_change_keeps_plan(self):
        wb, conv, desk = build()
        plan = desk.draft()[0]
        old, rejected, fresh = desk.recheck(plan.id)
        self.assertEqual(old, plan)
        self.assertIsNone(rejected)
        self.assertEqual(fresh, [])


if __name__ == "__main__":
    unittest.main()


class EdgeCaseTests(unittest.TestCase):
    def test_same_request_cannot_be_reused_after_commit(self):
        wb, conv, desk = build()
        plan = desk.draft()[0]
        desk.confirm(plan, ["p1", "p3", "p5"], "B96")
        rejected = desk.confirm(plan, ["p1", "p3", "p5"], "B97")
        self.assertIsInstance(rejected, Rejected)
        self.assertTrue(any("重复" in reason for reason in rejected.reasons))

    def test_other_order_taking_last_seat_expires_plan(self):
        wb, conv, desk = build()
        plan = desk.draft()[0]
        train_id, count = next(iter(plan.seats_used.items()))
        desk.record_sale(train_id, desk.free_seats(train_id), "B70")
        old, rejected, fresh = desk.recheck(plan.id)
        self.assertIsNone(old)
        self.assertTrue(any("余位" in item for item in rejected.changed))
        for candidate in fresh:
            self.assertNotIn(train_id, candidate.seats_used)

    def test_declared_authority_limit_blocks_account_holder(self):
        wb, conv, desk = build()
        # p5 明说“等她替我答复不行”，即便 p1 想代答也不够。
        self.assertEqual(desk.confirmers_for("p5"), frozenset({"p5"}))
        plan = desk.draft()[0]
        rejected = desk.confirm(plan, ["p1", "p3"], "B98")
        self.assertIsInstance(rejected, Rejected)
        self.assertIn("p5", " ".join(rejected.reasons))

    def test_draft_is_stable_and_readonly(self):
        wb, conv, desk = build()
        first = [plan.describe() for plan in desk.draft(limit=3)]
        second = [plan.describe() for plan in desk.draft(limit=3)]
        self.assertEqual(first, second)
        self.assertTrue(all(desk.lookup("unused") is None for _ in [0]))

    def test_unanswered_proposal_does_not_force_boarding(self):
        wb = load_workbook(ROOT)
        wb = without(wb, "m2", "m5", "m6")
        conversation = Conversation(wb)
        conversation.read_all()
        desk = Desk(wb, conversation)
        # p3 未答复，东城不能当成他接受的上车站；通告补进不等于本人同意。
        allowed = desk.allowed_boardings(wb.order("b3"))
        self.assertEqual(allowed, frozenset({"西桥"}))
        self.assertIsNone(conversation.boarding_rule("p3"))

    def test_excluded_train_never_used(self):
        wb, conv, desk = build()
        for plan in desk.draft(limit=50):
            self.assertNotIn("R88", {leg.train for leg in plan.legs})


class AmbiguityTests(unittest.TestCase):
    def test_ambiguous_pronoun_is_not_guessed(self):
        wb = load_workbook(ROOT)
        from rebooking.model import Message

        wb = replace(
            wb,
            messages={
                **wb.messages,
                "m9": Message(
                    id="m9",
                    actor="p1",
                    reply="m1",
                    at="2026-11-23T11:59:00+08:00",
                    text="她的座位不要动。",
                ),
            },
        )
        conversation = Conversation(wb)
        conversation.read_all()
        unclear = [
            claim for claim in conversation.claims if claim.kind == "pronoun_unclear"
        ]
        self.assertTrue(unclear)
        self.assertTrue(any("代词" in claim.note or "他/她" in claim.note for claim in unclear))
        self.assertIn(unclear[0].id, {claim.id for claim in conversation.unresolved})


class TransferRuleTests(unittest.TestCase):
    def test_transfer_condition_stop_is_enforced(self):
        wb = load_workbook(ROOT)
        wb = without(wb, "m5")
        conversation = Conversation(wb)
        conversation.read_all()
        desk = Desk(wb, conversation)
        transfers = [
            (plan, use)
            for plan in desk.draft(limit=60)
            for use in plan.benefits
            if use.transfer
        ]
        self.assertTrue(transfers)
        for plan, use in transfers:
            leg = plan.person_leg(use.rider)
            train = wb.train(leg.train)
            self.assertIn("石影", train.interval(leg.boarding, leg.alighting))

    def test_scope_rejects_unrelated_person(self):
        wb, conv, desk = build()
        benefit = wb.benefit("e3")
        self.assertTrue(desk._scope_allows(benefit, "p2"))
        self.assertFalse(desk._scope_allows(benefit, "p5"))
        self.assertFalse(desk._scope_allows(wb.benefit("e1"), "p2"))


class MultiOrderTests(unittest.TestCase):
    def test_two_orders_for_same_person_move_together(self):
        wb = load_workbook(ROOT)
        from rebooking.model import Order

        extra = Order(
            id="b6",
            person="p1",
            account="p1",
            aliases=("林姐", "林岚"),
            boarding="石湾",
            alighting="北林",
            service="R63",
            benefit=None,
        )
        wb = replace(wb, orders={**wb.orders, "b6": extra})
        conversation = Conversation(wb)
        conversation.read_all()
        desk = Desk(wb, conversation)
        self.assertEqual(
            [order.id for order in wb.orders_of("p1")], ["b1", "b6"]
        )
        # 同车约束覆盖受影响的两张；未受影响的 b6 保持原票，不擅自改动。
        for plan in desk.draft(limit=10):
            self.assertEqual(plan.leg("b1").train, plan.leg("b2").train)
            self.assertEqual(plan.leg("b6").train, "R63")
            self.assertFalse(plan.leg("b6").changed)
