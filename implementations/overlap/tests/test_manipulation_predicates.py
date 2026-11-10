import unittest

import numpy as np

from evaluate_esagas import make_semantic_evaluator
from S_utils.evaluators import ObjectStateEvaluator
from S_utils.manipulation_predicates import (
    scene_object_label_set,
    scene_state_from_task_info,
)
from S_utils.semantic_types import SemanticCarrier


def scene_mapping():
    qpos = np.zeros(25, dtype=np.float64)
    qpos[14:17] = np.array([0.33, -0.34, 0.066])
    qpos[23] = -0.16
    qpos[24] = 0.2
    return {"qpos": qpos, "button_states": np.array([1, 0])}


class ManipulationPredicateTests(unittest.TestCase):
    def test_object_state_evaluator_reads_scene_mapping(self):
        specs = {
            "button0_on": {"evaluator": "object_state", "property": "button0_on"},
            "drawer_open": {"evaluator": "object_state", "property": "drawer_open"},
            "window_closed": {"evaluator": "object_state", "property": "window_closed"},
            "cube_in_drawer": {"evaluator": "object_state", "property": "cube_in_drawer"},
        }
        evaluator = ObjectStateEvaluator(specs)
        carrier = SemanticCarrier(observation=np.zeros(1), state=scene_mapping())

        self.assertTrue(evaluator.evaluate(carrier, "button0_on").cert_pos)
        self.assertTrue(evaluator.evaluate(carrier, "drawer_open").cert_pos)
        self.assertTrue(evaluator.evaluate(carrier, "window_closed").cert_neg)
        self.assertTrue(evaluator.evaluate(carrier, "cube_in_drawer").cert_pos)

    def test_cube_in_region_supports_task_time_sphere(self):
        specs = {
            "cube_zone": {
                "evaluator": "object_state",
                "object": "cube",
                "property": "in_region",
                "region": {"type": "sphere", "center": [0.33, -0.356, 0.066], "radius": 0.02},
            }
        }
        labels = scene_object_label_set(scene_mapping(), specs)
        self.assertEqual(labels, {"cube_zone"})

    def test_scene_observation_parser_handles_state_observation(self):
        obs = np.zeros(40, dtype=np.float64)
        obs[19:22] = (np.array([0.4, 0.1, 0.02]) - np.array([0.425, 0.0, 0.0])) * 10.0
        obs[28:30] = [0, 1]
        obs[32:34] = [1, 0]
        obs[36] = -0.16 * 18.0
        obs[38] = 0.0
        specs = {
            "button0_on": {"evaluator": "object_state", "property": "button0_on"},
            "button1_on": {"evaluator": "object_state", "property": "button1_on"},
            "drawer_open": {"evaluator": "object_state", "property": "drawer_open"},
            "window_closed": {"evaluator": "object_state", "property": "window_closed"},
        }
        evaluator = ObjectStateEvaluator(specs)
        carrier = SemanticCarrier(observation=obs)

        self.assertTrue(evaluator.evaluate(carrier, "button0_on").cert_pos)
        self.assertTrue(evaluator.evaluate(carrier, "button1_on").cert_neg)
        self.assertTrue(evaluator.evaluate(carrier, "drawer_open").cert_pos)
        self.assertTrue(evaluator.evaluate(carrier, "window_closed").cert_pos)

    def test_off_and_closed_predicates_are_mutually_consistent(self):
        specs = {
            "button0_off": {"evaluator": "object_state", "property": "button0_off"},
            "button1_on": {"evaluator": "object_state", "property": "button1_on"},
            "drawer_closed": {"evaluator": "object_state", "property": "drawer_closed"},
            "window_open": {"evaluator": "object_state", "property": "window_open"},
        }
        state = {
            "button_states": np.array([0, 1]),
            "drawer_pos": 0.0,
            "window_pos": 0.2,
        }

        labels = scene_object_label_set(state, specs)

        self.assertEqual(labels, {"button0_off", "button1_on", "drawer_closed", "window_open"})

    def test_scene_state_from_task_info_supports_generator_validation(self):
        task_info = {
            "init": {
                "block_xyzs": [[0.35, -0.05, 0.02]],
                "button_states": [0, 1],
                "drawer_pos": 0.0,
                "window_pos": 0.2,
            },
            "goal": {
                "block_xyzs": [[0.35, 0.05, 0.02]],
                "button_states": [1, 0],
                "drawer_pos": -0.16,
                "window_pos": 0.0,
            },
        }
        specs = {
            "button0_off": {"evaluator": "object_state", "property": "button0_off"},
            "button1_on": {"evaluator": "object_state", "property": "button1_on"},
            "drawer_closed": {"evaluator": "object_state", "property": "drawer_closed"},
            "window_open": {"evaluator": "object_state", "property": "window_open"},
            "cube_zone": {
                "evaluator": "object_state",
                "object": "cube",
                "property": "in_region",
                "region": {"type": "sphere", "center": [0.35, -0.05, 0.02], "radius": 0.03},
            },
        }

        labels = scene_object_label_set(scene_state_from_task_info(task_info), specs)

        self.assertEqual(
            labels,
            {"button0_off", "button1_on", "drawer_closed", "window_open", "cube_zone"},
        )

    def test_composite_object_state_predicates_support_configuration_labels(self):
        specs = {
            "cfg_ready": {
                "evaluator": "object_state",
                "all_of": [
                    {"property": "button0_on"},
                    {"property": "drawer_open"},
                    {"property": "cube_in_drawer"},
                ],
            },
            "cfg_alt": {
                "evaluator": "object_state",
                "any_of": [
                    {"property": "window_open"},
                    {"property": "button1_on"},
                ],
            },
        }
        evaluator = ObjectStateEvaluator(specs)
        carrier = SemanticCarrier(observation=np.zeros(1), state=scene_mapping())

        self.assertTrue(evaluator.evaluate(carrier, "cfg_ready").cert_pos)
        self.assertTrue(evaluator.evaluate(carrier, "cfg_alt").cert_pos)
        self.assertEqual(scene_object_label_set(scene_mapping(), specs), {"cfg_ready", "cfg_alt"})

    def test_task_info_cube_in_drawer_uses_runtime_drawer_bounds(self):
        specs = {
            "cube_in_drawer": {"evaluator": "object_state", "property": "cube_in_drawer"}
        }
        near_closed_handle_but_not_inside = {
            "init": {
                "block_xyzs": [[0.3979, -0.2486, 0.02]],
                "button_states": [0, 0],
                "drawer_pos": 0.0,
                "window_pos": 0.0,
            }
        }
        inside_closed_drawer = {
            "init": {
                "block_xyzs": [[0.35, -0.36, 0.02]],
                "button_states": [0, 0],
                "drawer_pos": 0.0,
                "window_pos": 0.0,
            }
        }

        self.assertNotIn(
            "cube_in_drawer",
            scene_object_label_set(
                scene_state_from_task_info(near_closed_handle_but_not_inside),
                specs,
            ),
        )
        self.assertIn(
            "cube_in_drawer",
            scene_object_label_set(scene_state_from_task_info(inside_closed_drawer), specs),
        )

    def test_object_state_backend_selects_declared_propositions(self):
        evaluator = make_semantic_evaluator(
            "object_state",
            [],
            predicate_specs={
                "drawer_open": {"evaluator": "object_state", "property": "drawer_open"},
                "drawer_closed": {"evaluator": "object_state", "property": "drawer_closed"},
            },
            propositions=["drawer_open"],
        )
        carrier = SemanticCarrier(observation=np.zeros(1), state=scene_mapping())
        self.assertEqual(evaluator.propositions, ["drawer_open"])
        cert = evaluator.evaluate(carrier, "drawer_open")
        self.assertTrue(cert.cert_pos)
        self.assertTrue(evaluator.evaluate(carrier, "drawer_closed").uncertain)

    def test_object_state_backend_rejects_missing_proposition(self):
        with self.assertRaisesRegex(ValueError, "Object-state predicates required"):
            make_semantic_evaluator("object_state", [], propositions=["a"])


if __name__ == "__main__":
    unittest.main()
