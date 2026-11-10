import unittest

try:
    import networkx as nx
    from L_utils.ltl_runtime import BuchiRuntime
except ModuleNotFoundError:
    nx = None
    BuchiRuntime = None

from S_utils.guard_terms import GuardTerm


class _Buchi:
    def __init__(self):
        self.buchi_graph = nx.MultiDiGraph(type="buchi", init=["q0"], accept=["q1"])
        self.buchi_graph.add_node("q0")
        self.buchi_graph.add_node("q1")
        self.buchi_graph.add_edge(
            "q0",
            "q1",
            truth={"a": True},
            guard_term=GuardTerm("g0", source_state="q0", target_state="q1", pos=frozenset({"a"})),
        )
        self.buchi_graph.add_edge(
            "q0",
            "q0",
            truth={"a": False},
            guard_term=GuardTerm("g1", source_state="q0", target_state="q0", neg=frozenset({"a"})),
        )


class BuchiRuntimeStateSetTests(unittest.TestCase):
    def test_initial_label_consumption(self):
        if nx is None or BuchiRuntime is None:
            self.skipTest("networkx not available in this Python environment")
        rt = BuchiRuntime(_Buchi(), label_to_literal={"a": "a"})
        self.assertEqual(rt.initial_state_set_after_label({"a"}), {"q1"})
        self.assertEqual(rt.initial_state_set_after_label(set()), {"q0"})


if __name__ == "__main__":
    unittest.main()
