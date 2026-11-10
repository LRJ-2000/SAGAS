import unittest

from generate_semantic_ltl_tasks import (
    check_relations,
    generate_payload,
    templates_for,
)


def _grid_points():
    return [(float(x), float(y)) for x in range(-8, 9) for y in range(-8, 9)]


class GenerateSemanticLTLTasksTests(unittest.TestCase):
    def test_medium_generation_keeps_real_task_file_schema_and_metadata(self):
        payload = generate_payload(
            env_name="toy-maze-v0",
            complexity="medium",
            num_tasks=5,
            seed=7,
            free_points=_grid_points(),
            max_task_attempts=200,
            shape_mode="mixed",
            nonconvex_prob=0.4,
            region_size_scale=0.8,
        )
        self.assertEqual(payload["num_tasks"], 5)
        self.assertEqual(payload["semantic_task_schema"], "state_region_with_predicate_specs_v1")
        for task in payload["tasks"]:
            self.assertIn("formula_e", task)
            self.assertIn("labels", task)
            self.assertIn("regions", task)
            self.assertIn("start_xy", task)
            self.assertIn("metadata", task)
            self.assertIn("predicate_specs", task["metadata"])
            self.assertTrue(all(task["metadata"]["relation_checks"].values()))

    def test_hard_generation_covers_overlap_negative_and_recurrence_features(self):
        payload = generate_payload(
            env_name="toy-maze-v0",
            complexity="hard",
            num_tasks=5,
            seed=11,
            free_points=_grid_points(),
            max_task_attempts=300,
            shape_mode="mixed",
            nonconvex_prob=0.35,
            region_size_scale=0.8,
        )
        feature_union = set()
        for task in payload["tasks"]:
            feature_union.update(task["metadata"]["semantic_features"])
            self.assertTrue(all(task["metadata"]["relation_checks"].values()))
        self.assertIn("overlap", feature_union)
        self.assertIn("multi_positive", feature_union)
        self.assertIn("negative_literal", feature_union)
        self.assertIn("suffix", feature_union)

    def test_nonconvex_shape_mode_records_future_compatible_predicate_specs(self):
        payload = generate_payload(
            env_name="toy-maze-v0",
            complexity="medium",
            num_tasks=2,
            seed=13,
            free_points=_grid_points(),
            max_task_attempts=200,
            shape_mode="nonconvex_polygon",
            nonconvex_prob=1.0,
            region_size_scale=0.8,
        )
        for task in payload["tasks"]:
            for label in task["labels"]:
                pred = task["metadata"]["predicate_specs"][label]
                self.assertEqual(pred["evaluator"], "state_region")
                self.assertEqual(pred["metadata"]["shape_kind"], "nonconvex_polygon")
                self.assertEqual(pred["geometry"]["type"], "polygon")
                self.assertTrue(pred["geometry"].get("nonconvex"))

    def test_relation_checker_reports_failed_overlap(self):
        template = templates_for("medium")[0]
        payload = generate_payload(
            env_name="toy-maze-v0",
            complexity="medium",
            num_tasks=1,
            seed=17,
            free_points=_grid_points(),
            max_task_attempts=200,
            shape_mode="circle",
            region_size_scale=0.8,
        )
        task = payload["tasks"][0]
        predicates = {
            label: type(
                "_Pred",
                (),
                {"geometry": spec["geometry"]},
            )()
            for label, spec in task["metadata"]["predicate_specs"].items()
        }
        relations = check_relations(
            predicates,
            template.relations,
            start_xy=tuple(task["start_xy"]),
            free_points=[],
        )
        self.assertIn("overlap:a,b", relations)


if __name__ == "__main__":
    unittest.main()
