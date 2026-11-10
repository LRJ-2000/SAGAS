import unittest

from S_utils.anchor_manager import AnchorManager
from S_utils.certificate_store import CertificateStore
from S_utils.evaluators import StateRegionEvaluator
from S_utils.guard_terms import GuardTerm
from S_utils.product_oracle import ProductOracle
from S_utils.semantic_graph import KeyGraphSemanticAugmentor
from S_utils.semantic_types import SemanticCarrier


class _NodeView:
    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, node_id):
        return self._graph._nodes[node_id]

    def __iter__(self):
        return iter(self._graph._nodes)


class _Graph:
    def __init__(self):
        self._nodes = {}
        self._edges = {}
        self.nodes = _NodeView(self)

    def add_node(self, node_id, **attrs):
        self._nodes.setdefault(node_id, {}).update(attrs)

    def add_edge(self, u, v, **attrs):
        self._edges[(u, v)] = dict(attrs)

    def has_edge(self, u, v):
        return (u, v) in self._edges

    def number_of_nodes(self):
        return len(self._nodes)


class _KeyGraph:
    def __init__(self, nodes, way_steps=1.0):
        self.nodes = [list(n) for n in nodes]
        self.way_steps = way_steps
        self.base_node_cnt = len(nodes)
        self.graph = _Graph()
        self.label_node_idx_dict = {}
        self.dist_precomputed = False
        for idx, pos in enumerate(self.nodes):
            self.graph.add_node(idx, pos=pos)

    def precompute_dist_to_label(self):
        self.dist_precomputed = True


class SemanticGraphTests(unittest.TestCase):
    def test_guard_anchor_inserted_into_keygraph_and_product_oracle(self):
        key_graph = _KeyGraph([(0.0, 0.0), (4.0, 0.0)], way_steps=1.0)
        augmentor = KeyGraphSemanticAugmentor(
            key_graph,
            get_phi=lambda obs: obs,
        )
        evaluator = StateRegionEvaluator(
            [{"id": "a", "kind": "rect", "bounds": (0.0, 1.0, -0.5, 0.5)}]
        )
        store = CertificateStore(evaluator, ["a"])
        manager = AnchorManager(
            store,
            connectivity_fn=lambda carrier: augmentor.has_local_neighbor(carrier),
        )
        guard = GuardTerm("g0", pos=frozenset({"a"}))
        carriers = [SemanticCarrier(observation=(0.5, 0.0), index=0)]

        manager.build_guard_anchors([guard], carriers)
        inserted = augmentor.insert_anchor_records(
            manager.anchors.values(),
            manager=manager,
        )

        anchor_id = manager.anchors_by_guard["g0"][0]
        node_id = inserted[anchor_id].node_id
        self.assertEqual(inserted[anchor_id].connection_status, "local_connected")
        self.assertTrue(key_graph.graph.has_edge(0, node_id))
        self.assertTrue(key_graph.graph.has_edge(node_id, 0))
        self.assertEqual(key_graph.graph.nodes[node_id]["semantic_guard_ids"], ["g0"])
        self.assertEqual(key_graph.label_node_idx_dict["a"], [node_id])
        self.assertTrue(ProductOracle(store, manager).enable(node_id, guard))

    def test_start_node_uses_nearest_fallback_without_marking_anchor(self):
        key_graph = _KeyGraph([(0.0, 0.0), (4.0, 0.0)], way_steps=0.5)
        augmentor = KeyGraphSemanticAugmentor(
            key_graph,
            get_phi=lambda obs: obs,
        )
        start = SemanticCarrier(observation=(3.0, 0.0), index=7, source="start")

        inserted = augmentor.insert_start_node(start)

        self.assertEqual(inserted.connection_status, "nearest_fallback")
        self.assertEqual(inserted.neighbors, [1])
        self.assertTrue(key_graph.graph.has_edge(1, inserted.node_id))
        self.assertFalse(key_graph.graph.nodes[inserted.node_id].get("is_anchor", False))
        self.assertTrue(key_graph.graph.nodes[inserted.node_id].get("is_start", False))

    def test_anchor_without_local_neighbor_is_not_inserted(self):
        key_graph = _KeyGraph([(0.0, 0.0)], way_steps=0.5)
        augmentor = KeyGraphSemanticAugmentor(
            key_graph,
            get_phi=lambda obs: obs,
        )
        evaluator = StateRegionEvaluator(
            [{"id": "far", "kind": "rect", "bounds": (9.0, 11.0, -1.0, 1.0)}]
        )
        store = CertificateStore(evaluator, ["far"])
        manager = AnchorManager(store)
        guard = GuardTerm("g0", pos=frozenset({"far"}))
        carrier = SemanticCarrier(observation=(10.0, 0.0), index=0)

        manager.build_guard_anchors([guard], [carrier])
        anchor_id = manager.anchors_by_guard["g0"][0]
        inserted = augmentor.insert_anchor_record(
            manager.anchors[anchor_id],
            manager=manager,
        )

        self.assertEqual(inserted.connection_status, "no_local_neighbor")
        self.assertIsNone(inserted.node_id)
        self.assertEqual(key_graph.graph.number_of_nodes(), 1)
        self.assertFalse(ProductOracle(store, manager).enable("v_missing", guard))


if __name__ == "__main__":
    unittest.main()
