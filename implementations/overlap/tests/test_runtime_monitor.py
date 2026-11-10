import unittest

from S_utils.certificate_store import CertificateStore
from S_utils.evaluators import StateRegionEvaluator
from S_utils.guard_terms import GuardTerm
from S_utils.runtime_monitor import (
    RuntimeMonitor,
    RuntimeMonitorConfig,
    can_advance_to_waypoint,
)
from S_utils.semantic_types import SemanticCarrier


class _BuchiRuntime:
    def branch_reaches(self, q_start, label_sequence, q_target):
        labels = [set(x) for x in label_sequence]
        if q_start == "q0" and q_target == "q1":
            return any("a" in label for label in labels)
        return False


class RuntimeMonitorTests(unittest.TestCase):
    def setUp(self):
        evaluator = StateRegionEvaluator(
            [
                {"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)},
                {"id": "bad", "kind": "rect", "bounds": (4.0, 5.0, 4.0, 5.0)},
            ]
        )
        self.store = CertificateStore(evaluator, ["a", "bad"])
        self.monitor = RuntimeMonitor(self.store)

    def test_ordinary_waypoint_can_be_skipped(self):
        self.assertTrue(can_advance_to_waypoint(0, 2, semantic_checkpoint_indices={3}))

    def test_semantic_checkpoint_cannot_be_skipped_until_verified(self):
        self.assertTrue(can_advance_to_waypoint(0, 3, semantic_checkpoint_indices={3}))
        self.assertFalse(can_advance_to_waypoint(0, 4, semantic_checkpoint_indices={3}))
        self.assertTrue(
            can_advance_to_waypoint(
                0,
                4,
                semantic_checkpoint_indices={3},
                verified_checkpoint_indices={3},
            )
        )

    def test_positive_guard_requires_runtime_certificate_at_checkpoint(self):
        guard = GuardTerm("g_a", pos=frozenset({"a"}))
        miss = self.monitor.monitor_transition_segment(
            guard=guard,
            carriers=[SemanticCarrier(observation=(2.0, 2.0), index=0)],
            waypoint_reached=True,
        )
        self.assertFalse(miss.realized)
        self.assertEqual(miss.reason, "missing_positive_certificate")

        hit = self.monitor.monitor_transition_segment(
            guard=guard,
            carriers=[SemanticCarrier(observation=(0.5, 0.5), index=1)],
            waypoint_reached=True,
        )
        self.assertTrue(hit.realized)

    def test_negative_guard_is_checked_over_segment(self):
        guard = GuardTerm("g_safe", neg=frozenset({"bad"}))
        result = self.monitor.monitor_transition_segment(
            guard=guard,
            carriers=[
                SemanticCarrier(observation=(2.0, 2.0), index=0),
                SemanticCarrier(observation=(4.5, 4.5), index=1),
            ],
            waypoint_reached=True,
        )
        self.assertFalse(result.realized)
        self.assertEqual(result.reason, "missing_negative_certificate")
        self.assertEqual(result.details["step"], 1)

    def test_timeout_reports_runtime_unrealized(self):
        guard = GuardTerm("g_free")
        monitor = RuntimeMonitor(self.store, RuntimeMonitorConfig(local_step_budget=2))
        result = monitor.monitor_transition_segment(
            guard=guard,
            carriers=[
                SemanticCarrier(observation=(2.0, 2.0), index=0),
                SemanticCarrier(observation=(2.0, 2.1), index=1),
            ],
            waypoint_reached=False,
        )
        self.assertFalse(result.realized)
        self.assertEqual(result.reason, "runtime_unrealized")

    def test_branch_feasibility_can_be_checked_against_buchi_runtime(self):
        guard = GuardTerm("g_a", pos=frozenset({"a"}))
        result = self.monitor.monitor_transition_segment(
            guard=guard,
            carriers=[SemanticCarrier(observation=(0.5, 0.5), index=0)],
            waypoint_reached=True,
            buchi_runtime=_BuchiRuntime(),
            q_start="q0",
            q_target="q1",
        )
        self.assertTrue(result.realized)


if __name__ == "__main__":
    unittest.main()
