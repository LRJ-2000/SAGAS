import unittest

from S_utils.anchor_manager import AnchorManager
from S_utils.certificate_store import CertificateStore
from S_utils.evaluators import SemanticEvaluator
from S_utils.evaluators import StateRegionEvaluator
from S_utils.guard_terms import GuardTerm
from S_utils.semantic_types import PredicateCertificate, SemanticCarrier


class _UncertainEvaluator(SemanticEvaluator):
    def evaluate(self, carrier, proposition):
        return PredicateCertificate.uncertain_value(
            proposition,
            score=0.5,
            lower=0.0,
            upper=1.0,
        )


class AnchorManagerTests(unittest.TestCase):
    def test_guard_anchor_requires_full_guard(self):
        evaluator = StateRegionEvaluator(
            [
                {"id": "a", "kind": "rect", "bounds": (0.0, 2.0, 0.0, 2.0)},
                {"id": "b", "kind": "rect", "bounds": (1.0, 3.0, 1.0, 3.0)},
                {"id": "c", "kind": "rect", "bounds": (5.0, 6.0, 5.0, 6.0)},
            ]
        )
        store = CertificateStore(evaluator, ["a", "b", "c"])
        carriers = [
            SemanticCarrier(observation=(0.5, 0.5), index=0),
            SemanticCarrier(observation=(1.5, 1.5), index=1),
        ]
        manager = AnchorManager(store)
        guard = GuardTerm("g0", pos=frozenset({"a", "b"}), neg=frozenset({"c"}))
        manager.build_guard_anchors([guard], carriers)
        self.assertEqual(manager.guard_reports["g0"].status, "supported")
        self.assertEqual(len(manager.anchors_by_guard["g0"]), 1)
        anchor_id = manager.anchors_by_guard["g0"][0]
        self.assertEqual(manager.anchors[anchor_id].cert_pos, {"a", "b"})

    def test_no_connected_anchor_report(self):
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 2.0, 0.0, 2.0)}]
        )
        store = CertificateStore(evaluator, ["a"])
        carriers = [SemanticCarrier(observation=(0.5, 0.5), index=0)]
        manager = AnchorManager(store, connectivity_fn=lambda _carrier: False)
        guard = GuardTerm("g0", pos=frozenset({"a"}))
        manager.build_guard_anchors([guard], carriers)
        self.assertEqual(manager.guard_reports["g0"].status, "no_connected_anchor")

    def test_proposition_anchor_reports_all_supported_aps(self):
        evaluator = StateRegionEvaluator(
            [
                {"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)},
                {"id": "b", "kind": "rect", "bounds": (5.0, 6.0, 5.0, 6.0)},
            ]
        )
        store = CertificateStore(evaluator, ["a", "b"])
        carriers = [SemanticCarrier(observation=(0.5, 0.5), index=0)]
        manager = AnchorManager(store)

        manager.build_proposition_anchors(carriers)

        self.assertEqual(manager.prop_reports["a"].status, "supported")
        self.assertEqual(manager.prop_reports["b"].status, "no_candidate")
        self.assertEqual(manager.prop_reports["a"].num_connected_anchors, 1)
        self.assertFalse(manager.prop_reports["b"].num_connected_anchors)

    def test_guard_report_records_uncertainty_blocks(self):
        store = CertificateStore(_UncertainEvaluator(), ["a"])
        manager = AnchorManager(store)
        guard = GuardTerm("g0", pos=frozenset({"a"}))

        manager.build_guard_anchors([guard], [SemanticCarrier(observation=(0.0, 0.0), index=0)])

        report = manager.guard_reports["g0"]
        self.assertEqual(report.status, "blocked_by_uncertainty")
        self.assertEqual(report.example_failures[0]["reason"], "blocked_by_uncertainty")


if __name__ == "__main__":
    unittest.main()
