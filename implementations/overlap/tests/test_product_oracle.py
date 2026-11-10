import unittest

from S_utils.anchor_manager import AnchorManager
from S_utils.certificate_store import CertificateStore
from S_utils.evaluators import StateRegionEvaluator
from S_utils.guard_terms import GuardTerm
from S_utils.product_oracle import ProductOracle
from S_utils.semantic_types import SemanticCarrier


class ProductOracleTests(unittest.TestCase):
    def test_positive_guard_requires_guard_supported_anchor(self):
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["a"])
        carrier = SemanticCarrier(observation=(0.5, 0.5), index=0)
        manager = AnchorManager(store)
        guard = GuardTerm("g0", pos=frozenset({"a"}))
        manager.build_guard_anchors([guard], [carrier])
        oracle = ProductOracle(store, manager)
        anchor_id = manager.anchors_by_guard["g0"][0]
        self.assertTrue(oracle.enable(anchor_id, guard))
        self.assertFalse(oracle.enable("ordinary", guard))
        self.assertEqual(
            oracle.diagnostics_summary()["block_counts"]["missing_positive_anchor"],
            1,
        )

    def test_positive_guard_requires_anchor_positive_certificate(self):
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["a"])
        manager = AnchorManager(store)
        guard = GuardTerm("g0", pos=frozenset({"a"}))
        record = manager._new_anchor(
            SemanticCarrier(observation=(2.0, 2.0), index=0),
            cert_pos=set(),
            cert_neg={"a"},
            guard_ids={"g0"},
        )
        oracle = ProductOracle(store, manager)

        self.assertFalse(oracle.enable(record.anchor_id, guard))
        self.assertEqual(
            oracle.diagnostics_summary()["block_counts"]["positive_anchor_missing_certificate"],
            1,
        )

    def test_negative_only_guard_allows_negative_certified_node(self):
        evaluator = StateRegionEvaluator(
            [{"id": "bad", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["bad"], tau_neg=0.01)
        support = {"v0": [SemanticCarrier(observation=(2.0, 2.0), index=0)]}
        oracle = ProductOracle(
            store,
            AnchorManager(store),
            node_support_lookup=lambda node: support[node],
        )
        guard = GuardTerm("g0", neg=frozenset({"bad"}))
        self.assertTrue(oracle.enable("v0", guard))

    def test_uncertain_negative_records_block_diagnostics(self):
        evaluator = StateRegionEvaluator(
            [{"id": "bad", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["bad"], tau_neg=0.01)
        support = {
            "mixed": [
                SemanticCarrier(observation=(0.5, 0.5), index=0),
                SemanticCarrier(observation=(2.0, 2.0), index=1),
            ]
        }
        oracle = ProductOracle(
            store,
            AnchorManager(store),
            node_support_lookup=lambda node: support[node],
        )
        guard = GuardTerm("g_safe", neg=frozenset({"bad"}))
        self.assertFalse(oracle.enable("mixed", guard))
        summary = oracle.diagnostics_summary()
        self.assertEqual(summary["negative_uncertainty_blocks"], 1)
        self.assertEqual(summary["blocked_guard_terms"], ["g_safe"])


if __name__ == "__main__":
    unittest.main()
