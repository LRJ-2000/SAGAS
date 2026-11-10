import unittest

from S_utils.certificate_store import CertificateStore
from S_utils.evaluators import StateRegionEvaluator
from S_utils.execution import (
    SegmentExecutionResult,
    execute_finite_product_plan,
)
from S_utils.guard_terms import GuardTerm
from S_utils.product_planner import FiniteProductExecutionPlan
from S_utils.runtime_monitor import RuntimeMonitor, RuntimeMonitorConfig
from S_utils.semantic_types import SemanticCarrier


class _BuchiRuntime:
    def branch_reaches(self, q_start, label_sequence, q_target):
        labels = [set(x) for x in label_sequence]
        if q_start == "q0" and q_target == "q1":
            return any("a" in label for label in labels)
        if q_start == "q0" and q_target == "q2":
            return any("a" in label for label in labels)
        if q_start == "q1" and q_target == "q1":
            return True
        return False


class ExecutionProtocolTests(unittest.TestCase):
    def setUp(self):
        evaluator = StateRegionEvaluator(
            [
                {"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)},
                {"id": "bad", "kind": "rect", "bounds": (4.0, 5.0, 4.0, 5.0)},
            ]
        )
        self.store = CertificateStore(evaluator, ["a", "bad"])
        self.monitor = RuntimeMonitor(self.store)
        self.plan = FiniteProductExecutionPlan(
            states=[("s", "q0"), ("a_anchor", "q1"), ("a_anchor", "q1"), ("a_anchor", "q1")],
            guards=[
                GuardTerm("g_a", source_state="q0", target_state="q1", pos=frozenset({"a"})),
                GuardTerm("g_stay", source_state="q1", target_state="q1"),
                GuardTerm("g_stay", source_state="q1", target_state="q1"),
            ],
            prefix_end_index=1,
            suffix_cycles_required=2,
            suffix_cycle_num_edges=1,
        )

    def test_successful_finite_execution_records_trace_and_suffix_cycles(self):
        def executor(req):
            if req.guard.guard_id == "g_a":
                return SegmentExecutionResult(
                    carriers=[SemanticCarrier(observation=(0.5, 0.5), state=(0.5, 0.5), index=req.transition_index)],
                    waypoint_reached=True,
                    actions=[req.transition_index],
                )
            return SegmentExecutionResult(
                carriers=[SemanticCarrier(observation=(2.0, 2.0), state=(2.0, 2.0), index=req.transition_index)],
                waypoint_reached=True,
                actions=[req.transition_index],
            )

        result = execute_finite_product_plan(
            execution_plan=self.plan,
            segment_executor=executor,
            runtime_monitor=self.monitor,
            buchi_runtime=_BuchiRuntime(),
        )

        self.assertTrue(result.finite_lasso_success)
        self.assertTrue(result.prefix_completed)
        self.assertEqual(result.suffix_cycles_completed, 2)
        self.assertEqual(len(result.trace.observations), 3)
        self.assertEqual(result.trace.actions, [0, 1, 2])
        self.assertIsNone(result.failure_reason)

    def test_missing_anchor_label_stops_execution(self):
        def executor(_req):
            return SegmentExecutionResult(
                carriers=[SemanticCarrier(observation=(2.0, 2.0), index=0)],
                waypoint_reached=True,
            )

        result = execute_finite_product_plan(
            execution_plan=self.plan,
            segment_executor=executor,
            runtime_monitor=self.monitor,
            buchi_runtime=_BuchiRuntime(),
        )

        self.assertFalse(result.finite_lasso_success)
        self.assertEqual(result.failure_reason, "missing_positive_certificate")
        self.assertFalse(result.prefix_completed)

    def test_timeout_is_reported_as_runtime_unrealized(self):
        monitor = RuntimeMonitor(self.store, RuntimeMonitorConfig(local_step_budget=1))

        def executor(_req):
            return SegmentExecutionResult(
                carriers=[SemanticCarrier(observation=(2.0, 2.0), index=0)],
                waypoint_reached=False,
            )

        result = execute_finite_product_plan(
            execution_plan=self.plan,
            segment_executor=executor,
            runtime_monitor=monitor,
        )

        self.assertFalse(result.finite_lasso_success)
        self.assertEqual(result.failure_reason, "runtime_unrealized")
        self.assertEqual(result.metadata["transitions_completed"], 0)

    def test_executor_can_advance_over_ordinary_waypoint_span(self):
        plan = FiniteProductExecutionPlan(
            states=[("s", "q0"), ("mid", "q0"), ("a_anchor", "q2")],
            guards=[
                GuardTerm("g_safe", source_state="q0", target_state="q0", neg=frozenset({"bad"})),
                GuardTerm("g_a", source_state="q0", target_state="q2", pos=frozenset({"a"})),
            ],
            prefix_end_index=2,
            suffix_cycles_required=0,
            suffix_cycle_num_edges=1,
        )

        def executor(req):
            self.assertEqual(req.transition_index, 0)
            return SegmentExecutionResult(
                carriers=[SemanticCarrier(observation=(0.5, 0.5), state=(0.5, 0.5), index=0)],
                waypoint_reached=True,
                advance_to_transition_index=2,
            )

        result = execute_finite_product_plan(
            execution_plan=plan,
            segment_executor=executor,
            runtime_monitor=self.monitor,
            buchi_runtime=_BuchiRuntime(),
        )

        self.assertTrue(result.finite_lasso_success)
        self.assertEqual(result.metadata["transitions_completed"], 2)
        self.assertEqual(len(result.transition_results), 1)
        self.assertEqual(
            result.trace.product_progress[0]["spanned_guard_ids"],
            ["g_safe", "g_a"],
        )


if __name__ == "__main__":
    unittest.main()
