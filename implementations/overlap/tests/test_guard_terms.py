import unittest

from S_utils.guard_terms import GuardTerm

try:
    from L_utils.buchi_parse import Buchi
except ModuleNotFoundError:
    Buchi = None


class _Task:
    formula = ""
    subformula = {}
    number_of_robots = 1


class GuardTermTests(unittest.TestCase):
    def setUp(self):
        if Buchi is None:
            self.skipTest("networkx/sympy not available in this Python environment")
        self.buchi = Buchi(_Task())

    def test_multi_positive_clause_is_kept(self):
        truths = self.buchi._get_truth_assignments("(a && b)")
        self.assertEqual(truths, [{"a": True, "b": True}])

    def test_disjunction_becomes_multiple_truths(self):
        truths = self.buchi._get_truth_assignments("(a || b)")
        self.assertEqual(
            {tuple(sorted(t.items())) for t in truths},
            {
                (("a", True),),
                (("b", True),),
            },
        )

    def test_negative_clause(self):
        truths = self.buchi._get_truth_assignments("(a && !c)")
        self.assertEqual(truths, [{"a": True, "c": False}])

    def test_unconditional_guard_is_empty_dict(self):
        self.assertEqual(self.buchi._get_truth_assignments("(1)"), [{}])

    def test_guard_term_from_truth(self):
        guard = GuardTerm.from_truth_dict(
            {"a": True, "c": False},
            guard_id="g0",
            source_state="q0",
            target_state="q1",
        )
        self.assertEqual(guard.pos, frozenset({"a"}))
        self.assertEqual(guard.neg, frozenset({"c"}))
        self.assertTrue(guard.satisfied_by_sets({"a", "b"}, {"c"}))
        self.assertFalse(guard.satisfied_by_sets({"a"}, set()))


if __name__ == "__main__":
    unittest.main()
