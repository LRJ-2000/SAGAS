import unittest

from S_utils.certificate_store import CertificateStore
from S_utils.evaluators import StateRegionEvaluator
from S_utils.semantic_types import SemanticCarrier


class CertificateStoreTests(unittest.TestCase):
    def test_constant_kernel_frequency_cert_neg(self):
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["a"], tau_neg=0.30, tau_pos=0.99, min_support=1)
        support = [
            SemanticCarrier(observation=(2.0, 2.0), index=0),
            SemanticCarrier(observation=(3.0, 3.0), index=1),
            SemanticCarrier(observation=(0.5, 0.5), index=2),
        ]
        cert = store.aggregate_node("v0", support, "a")
        self.assertAlmostEqual(cert.score, 1.0 / 3.0)
        self.assertFalse(cert.cert_neg)
        self.assertFalse(cert.cert_pos)

    def test_low_support_is_uncertain(self):
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["a"], tau_neg=0.01, tau_pos=0.99, min_support=2)
        support = [SemanticCarrier(observation=(2.0, 2.0), index=0)]
        cert = store.aggregate_node("v0", support, "a")
        self.assertTrue(cert.uncertain)
        self.assertFalse(cert.cert_neg)

    def test_independent_overlapping_props(self):
        evaluator = StateRegionEvaluator(
            [
                {"id": "a", "kind": "rect", "bounds": (0.0, 2.0, 0.0, 2.0)},
                {"id": "b", "kind": "rect", "bounds": (1.0, 3.0, 1.0, 3.0)},
            ]
        )
        store = CertificateStore(evaluator, ["a", "b"], tau_pos=0.99, tau_neg=0.01, min_support=1)
        support = [SemanticCarrier(observation=(1.5, 1.5), index=0)]
        certs = store.aggregate_all_for_node("v0", support)
        self.assertTrue(certs["a"].cert_pos)
        self.assertTrue(certs["b"].cert_pos)

    def test_same_index_from_different_sources_does_not_share_cache(self):
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["a"])
        offline = SemanticCarrier(
            observation=(0.5, 0.5),
            index=7,
            source="offline_support",
        )
        runtime = SemanticCarrier(
            observation=(2.0, 2.0),
            index=7,
            source="runtime_rollout",
        )
        self.assertEqual(store.cert_pos_set(offline), {"a"})
        self.assertEqual(store.cert_pos_set(runtime), set())
        self.assertEqual(store.cert_neg_set(runtime), {"a"})


if __name__ == "__main__":
    unittest.main()
