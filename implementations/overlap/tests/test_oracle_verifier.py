import unittest

from S_utils.oracle_verifier import (
    buchi_trace_summary,
    extract_oracle_label_trace,
    finite_lasso_summary,
    oracle_finite_lasso_verdict,
    oracle_plan_realization_summary,
)
from S_utils.guard_terms import GuardTerm
from S_utils.product_planner import FiniteProductExecutionPlan


class _BuchiRuntime:
    init_states = ["q0"]

    def initial_state_set_after_label(self, label_set):
        return self.step_set(self.init_states, label_set)

    def step_set(self, q_set, label_set):
        out = set()
        labels = set(label_set)
        for q in q_set:
            if q == "q0" and "a" in labels:
                out.add("q1")
            elif q == "q0":
                out.add("q0")
            elif q == "q1":
                out.add("q1")
        return out

    def is_accepting(self, q):
        return q == "q1"


class OracleVerifierTests(unittest.TestCase):
    def test_extract_oracle_labels(self):
        labels = extract_oracle_label_trace([0, 1, 2], lambda s: {"even"} if s % 2 == 0 else set())
        self.assertEqual(labels, [{"even"}, set(), {"even"}])

    def test_finite_lasso_summary(self):
        summary = finite_lasso_summary([{"a"}], True, 2, 2)
        self.assertTrue(summary["finite_lasso_success"])
        summary = finite_lasso_summary([{"a"}], True, 1, 2)
        self.assertFalse(summary["finite_lasso_success"])

    def test_buchi_trace_summary_consumes_initial_label(self):
        summary = buchi_trace_summary([{"a"}], _BuchiRuntime())
        self.assertTrue(summary["buchi_accepting_seen"])
        self.assertFalse(summary["buchi_dead"])
        self.assertEqual(summary["final_state_set"], ["q1"])

    def test_oracle_finite_lasso_verdict_separates_lasso_and_buchi(self):
        verdict = oracle_finite_lasso_verdict(
            states=[0, 1, 2],
            oracle_labeler=lambda s: {"a"} if s == 1 else set(),
            buchi_runtime=_BuchiRuntime(),
            prefix_completed=True,
            suffix_cycles_completed=2,
            suffix_cycles_required=2,
        )
        self.assertTrue(verdict["oracle_success"])
        self.assertIsNone(verdict["failure_reason"])

        verdict = oracle_finite_lasso_verdict(
            states=[0, 1],
            oracle_labeler=lambda s: {"a"} if s == 1 else set(),
            buchi_runtime=_BuchiRuntime(),
            prefix_completed=True,
            suffix_cycles_completed=0,
            suffix_cycles_required=1,
        )
        self.assertFalse(verdict["oracle_success"])
        self.assertEqual(verdict["failure_reason"], "horizon_exhausted")

    def test_oracle_plan_realization_checks_actual_guard_labels(self):
        plan = FiniteProductExecutionPlan(
            states=[("s", "q0"), ("a", "q1")],
            guards=[GuardTerm("g_a", "q0", "q1", pos=frozenset({"a"}))],
            prefix_end_index=1,
            suffix_cycles_required=0,
            suffix_cycle_num_edges=1,
        )
        progress = [
            {
                "transition_index": 0,
                "advance_to_transition_index": 1,
                "source": ("s", "q0"),
                "target": ("a", "q1"),
            }
        ]
        summary = oracle_plan_realization_summary(
            [{"a"}],
            product_progress=progress,
            execution_plan=plan,
            buchi_runtime=_BuchiRuntime(),
        )
        self.assertTrue(summary["oracle_plan_realized"])
        self.assertTrue(summary["oracle_prefix_completed"])

        summary = oracle_plan_realization_summary(
            [set()],
            product_progress=progress,
            execution_plan=plan,
            buchi_runtime=_BuchiRuntime(),
        )
        self.assertFalse(summary["oracle_plan_realized"])
        self.assertEqual(summary["oracle_plan_failure_reason"], "oracle_missing_positive")

    def test_oracle_finite_lasso_uses_plan_realization_over_runtime_counts(self):
        plan = FiniteProductExecutionPlan(
            states=[("s", "q0"), ("a", "q1")],
            guards=[GuardTerm("g_a", "q0", "q1", pos=frozenset({"a"}))],
            prefix_end_index=1,
            suffix_cycles_required=0,
            suffix_cycle_num_edges=1,
        )
        progress = [{"transition_index": 0, "advance_to_transition_index": 1}]
        verdict = oracle_finite_lasso_verdict(
            states=[0],
            oracle_labeler=lambda _s: set(),
            buchi_runtime=_BuchiRuntime(),
            prefix_completed=True,
            suffix_cycles_completed=0,
            suffix_cycles_required=0,
            execution_plan=plan,
            product_progress=progress,
        )
        self.assertFalse(verdict["oracle_success"])
        self.assertTrue(verdict["oracle_plan_checked"])
        self.assertEqual(verdict["failure_reason"], "oracle_missing_positive")


if __name__ == "__main__":
    unittest.main()
