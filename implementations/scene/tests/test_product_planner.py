import unittest

from S_utils.anchor_manager import AnchorManager
from S_utils.certificate_store import CertificateStore
from S_utils.evaluators import StateRegionEvaluator
from S_utils.guard_terms import GuardTerm
from S_utils.product_oracle import ProductOracle
from S_utils.product_planner import (
    initial_product_states,
    materialize_finite_lasso,
    plan_prefix_suffix,
    search_prefix_candidates,
)
from S_utils.semantic_types import PredicateCertificate, SemanticCarrier


class _Graph:
    def __init__(self):
        self.edges = {}

    def add_edge(self, u, v, weight=1.0):
        self.edges.setdefault(u, {})[v] = {"weight": float(weight)}

    def successors(self, u):
        return list(self.edges.get(u, {}))

    def __getitem__(self, u):
        return self.edges[u]


class _BuchiRuntime:
    def __init__(self, edges, accepts, init_after=None):
        self.edges = edges
        self.accepts = set(accepts)
        self.init_after = init_after or {}

    def enabled_guard_successors(self, q):
        return list(self.edges.get(q, []))

    def is_accepting(self, q):
        return q in self.accepts

    def initial_state_set_after_label(self, labels):
        return set(self.init_after.get(frozenset(labels), {"q0"}))


class ProductPlannerTests(unittest.TestCase):
    def test_initial_product_states_consume_start_label(self):
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["a"])
        start = SemanticCarrier(observation=(0.5, 0.5), index=0)
        buchi = _BuchiRuntime({}, accepts=set(), init_after={frozenset({"a"}): {"q1"}})

        self.assertEqual(initial_product_states("s", start, store, buchi), [("s", "q1")])

    def test_plan_records_guard_terms_and_logical_suffix(self):
        graph = _Graph()
        graph.add_edge("s", "a_anchor", weight=2.0)
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["a"])
        manager = AnchorManager(store)
        guard = GuardTerm("g_a", source_state="q0", target_state="q1", pos=frozenset({"a"}))
        stay = GuardTerm("g_stay", source_state="q1", target_state="q1")
        carrier = SemanticCarrier(observation=(0.5, 0.5), index=0)
        manager.build_guard_anchors([guard], [carrier])
        anchor_id = manager.anchors_by_guard["g_a"][0]
        manager.attach_graph_node(anchor_id, "a_anchor")
        oracle = ProductOracle(store, manager)
        buchi = _BuchiRuntime(
            {
                "q0": [("q1", guard)],
                "q1": [("q1", stay)],
            },
            accepts={"q1"},
        )

        plan = plan_prefix_suffix(graph, buchi, oracle, [("s", "q0")])

        self.assertIsNotNone(plan)
        self.assertEqual(plan.prefix.states, [("s", "q0"), ("a_anchor", "q1")])
        self.assertEqual([g.guard_id for g in plan.guard_terms], ["g_a", "g_stay"])
        self.assertEqual(plan.suffix.cost, 0.0)
        self.assertEqual(plan.total_cost, 2.0)

    def test_negative_guard_chooses_certified_safe_node(self):
        graph = _Graph()
        graph.add_edge("s", "bad_node", weight=1.0)
        graph.add_edge("s", "safe_node", weight=3.0)
        evaluator = StateRegionEvaluator(
            [{"id": "bad", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["bad"], tau_neg=0.01)
        support = {
            "bad_node": [SemanticCarrier(observation=(0.5, 0.5), index=0)],
            "safe_node": [SemanticCarrier(observation=(2.0, 2.0), index=1)],
        }
        oracle = ProductOracle(
            store,
            AnchorManager(store),
            node_support_lookup=lambda node: support[node],
        )
        avoid = GuardTerm("g_avoid", source_state="q0", target_state="q1", neg=frozenset({"bad"}))
        stay = GuardTerm("g_stay", source_state="q1", target_state="q1")
        buchi = _BuchiRuntime(
            {
                "q0": [("q1", avoid)],
                "q1": [("q1", stay)],
            },
            accepts={"q1"},
        )

        prefixes = search_prefix_candidates(graph, buchi, oracle, [("s", "q0")])

        self.assertEqual(len(prefixes), 1)
        self.assertEqual(prefixes[0].states[-1], ("safe_node", "q1"))
        self.assertEqual(prefixes[0].guards[0].guard_id, "g_avoid")

    def test_top_k_prefixes_are_retained_before_candidate_selection(self):
        graph = _Graph()
        graph.add_edge("s", "a1", weight=1.0)
        graph.add_edge("s", "a2", weight=2.0)
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 3.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["a"])
        manager = AnchorManager(store)
        guard = GuardTerm("g_a", pos=frozenset({"a"}))
        carriers = [
            SemanticCarrier(observation=(0.5, 0.5), index=0),
            SemanticCarrier(observation=(2.0, 0.5), index=1),
        ]
        manager.build_guard_anchors([guard], carriers)
        for anchor_id, node_id in zip(manager.anchors_by_guard["g_a"], ["a1", "a2"]):
            manager.attach_graph_node(anchor_id, node_id)
        oracle = ProductOracle(store, manager)
        buchi = _BuchiRuntime({"q0": [("q1", guard)]}, accepts={"q1"})

        prefixes = search_prefix_candidates(
            graph,
            buchi,
            oracle,
            [("s", "q0")],
            max_candidates=2,
        )

        self.assertEqual([p.states[-1][0] for p in prefixes], ["a1", "a2"])

    def test_materialize_finite_lasso_repeats_suffix_without_duplicate_start(self):
        graph = _Graph()
        graph.add_edge("s", "a_anchor", weight=2.0)
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["a"])
        manager = AnchorManager(store)
        guard = GuardTerm("g_a", source_state="q0", target_state="q1", pos=frozenset({"a"}))
        stay = GuardTerm("g_stay", source_state="q1", target_state="q1")
        carrier = SemanticCarrier(observation=(0.5, 0.5), index=0)
        manager.build_guard_anchors([guard], [carrier])
        manager.attach_graph_node(manager.anchors_by_guard["g_a"][0], "a_anchor")
        oracle = ProductOracle(store, manager)
        buchi = _BuchiRuntime(
            {"q0": [("q1", guard)], "q1": [("q1", stay)]},
            accepts={"q1"},
        )
        plan = plan_prefix_suffix(graph, buchi, oracle, [("s", "q0")])

        execution = materialize_finite_lasso(plan, suffix_cycles=2)

        self.assertEqual(execution.graph_nodes, ["s", "a_anchor", "a_anchor", "a_anchor"])
        self.assertEqual([g.guard_id for g in execution.guards], ["g_a", "g_stay", "g_stay"])
        self.assertEqual(execution.prefix_end_index, 1)
        self.assertEqual(execution.suffix_cycles_required, 2)

    def test_suffix_cycle_reconstructs_non_self_loop_return_path(self):
        graph = _Graph()
        graph.add_edge("s", "a_anchor", weight=1.0)
        graph.add_edge("a_anchor", "mid", weight=2.0)
        graph.add_edge("mid", "a_anchor", weight=3.0)
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 1.0, 0.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["a"])
        manager = AnchorManager(store)
        guard = GuardTerm("g_a", source_state="q0", target_state="q1", pos=frozenset({"a"}))
        reset = GuardTerm("g_reset", source_state="q1", target_state="q0")
        carrier = SemanticCarrier(observation=(0.5, 0.5), index=0)
        manager.build_guard_anchors([guard], [carrier])
        manager.attach_graph_node(manager.anchors_by_guard["g_a"][0], "a_anchor")
        oracle = ProductOracle(store, manager)
        buchi = _BuchiRuntime(
            {
                "q0": [("q1", guard)],
                "q1": [("q0", reset)],
            },
            accepts={"q1"},
        )

        plan = plan_prefix_suffix(graph, buchi, oracle, [("s", "q0")])

        self.assertIsNotNone(plan)
        self.assertEqual(plan.suffix.states[0], ("a_anchor", "q1"))
        self.assertEqual(plan.suffix.states[-1], ("a_anchor", "q1"))
        self.assertGreater(len(plan.suffix.states), 2)
        self.assertEqual([g.guard_id for g in plan.suffix.guards], ["g_reset", "g_a"])


if __name__ == "__main__":
    unittest.main()
