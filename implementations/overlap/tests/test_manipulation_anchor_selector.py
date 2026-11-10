import unittest

from S_utils.anchor_manager import AnchorManager
from S_utils.certificate_store import CertificateStore
from S_utils.guard_terms import GuardTerm
from S_utils.manipulation_anchor_selection import ManipulationAnchorSelector
from S_utils.manipulation_predicates import ObjectStateEvaluator
from S_utils.semantic_types import SemanticCarrier


def _scene_carrier(
    index,
    *,
    window_pos,
    cube_xyz=(0.425, 0.0, 0.02),
    button_states=(0, 0),
):
    return SemanticCarrier(
        observation=None,
        state={
            "window_pos": float(window_pos),
            "drawer_pos": 0.0,
            "button_states": list(button_states),
            "block_positions": {0: list(cube_xyz)},
        },
        index=index,
        source="scene_support",
    )


class ManipulationAnchorSelectorTests(unittest.TestCase):
    def _store(self):
        evaluator = ObjectStateEvaluator(
            {
                "window_open": {"evaluator": "object_state", "property": "window_open"},
                "button0_on": {"evaluator": "object_state", "property": "button0_on"},
            }
        )
        return CertificateStore(evaluator, ["window_open", "button0_on"])

    def test_proposition_selection_prioritizes_event_endpoint(self):
        store = self._store()
        selector = ManipulationAnchorSelector()
        carriers = [
            _scene_carrier(0, window_pos=0.0),
            _scene_carrier(1, window_pos=0.2),
            _scene_carrier(2, window_pos=0.2, cube_xyz=(0.20, 0.45, 0.02)),
        ]

        report = selector.select_proposition(
            proposition="window_open",
            carriers=carriers,
            max_count=2,
            store=store,
            connectivity_fn=lambda _carrier: True,
        )

        self.assertEqual(report.selected[0].carrier.index, 1)
        self.assertEqual(
            report.selected[0].metadata["candidate_priority"],
            "event_endpoint",
        )
        self.assertEqual(report.metadata["num_event_endpoint_candidates"], 1)

    def test_anchor_manager_uses_context_diverse_scene_candidates(self):
        store = self._store()
        manager = AnchorManager(
            store,
            connectivity_fn=lambda _carrier: True,
            candidate_selector=ManipulationAnchorSelector(),
        )
        carriers = [
            _scene_carrier(0, window_pos=0.0),
            _scene_carrier(1, window_pos=0.2, cube_xyz=(0.425, 0.0, 0.02)),
            _scene_carrier(2, window_pos=0.2, cube_xyz=(0.425, 0.0, 0.02)),
            _scene_carrier(3, window_pos=0.2, cube_xyz=(0.20, 0.45, 0.02)),
            _scene_carrier(4, window_pos=0.2, cube_xyz=(0.62, -0.45, 0.02)),
        ]

        manager.build_proposition_anchors(carriers, max_per_prop=3)
        selected = manager.anchors_by_prop["window_open"]

        self.assertEqual(len(selected), 3)
        self.assertEqual(manager.prop_reports["window_open"].status, "supported")
        selected_indices = [manager.anchors[anchor_id].carrier.index for anchor_id in selected]
        self.assertIn(1, selected_indices)
        self.assertNotEqual(selected_indices[:2], [1, 2])

    def test_guard_selection_records_selector_metadata(self):
        store = self._store()
        manager = AnchorManager(
            store,
            connectivity_fn=lambda _carrier: True,
            candidate_selector=ManipulationAnchorSelector(),
        )
        carriers = [
            _scene_carrier(0, window_pos=0.0, button_states=(0, 0)),
            _scene_carrier(1, window_pos=0.2, button_states=(0, 0)),
        ]
        guard = GuardTerm(
            "g_window_without_button",
            pos=frozenset({"window_open"}),
            neg=frozenset({"button0_on"}),
        )

        manager.build_guard_anchors([guard], carriers, max_per_guard=1)

        anchor_id = manager.anchors_by_guard[guard.guard_id][0]
        self.assertEqual(manager.guard_reports[guard.guard_id].status, "supported")
        self.assertEqual(
            manager.anchors[anchor_id].metadata["candidate_priority"],
            "event_endpoint",
        )
        self.assertIn(f"guard:{guard.guard_id}", manager.selector_reports)


if __name__ == "__main__":
    unittest.main()
