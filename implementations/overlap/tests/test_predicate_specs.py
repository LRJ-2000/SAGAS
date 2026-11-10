import json
import unittest

from S_utils.predicate_specs import (
    extract_predicate_specs,
    normalize_evaluator_backend,
    task_regions_to_region_specs,
    task_to_region_specs,
    validate_predicate_specs,
)


class PredicateSpecTests(unittest.TestCase):
    def test_backend_alias_normalization(self):
        self.assertEqual(normalize_evaluator_backend("state"), "state_region")
        self.assertEqual(normalize_evaluator_backend("object-state"), "object_state")
        self.assertEqual(normalize_evaluator_backend("oracle"), "oracle_state_lift")
        self.assertEqual(normalize_evaluator_backend("custom_backend"), "custom_backend")

    def test_top_level_specs_override_metadata_and_regions(self):
        task = {
            "regions": {
                "legacy": {"type": "circle", "center": [0.0, 0.0], "radius": 1.0}
            },
            "metadata": {
                "predicate_specs": {
                    "meta": {
                        "evaluator": "oracle",
                        "geometry": {"type": "circle", "center": [1.0, 1.0], "radius": 1.0},
                    }
                }
            },
            "predicate_specs": {
                "top": {
                    "evaluator": "state",
                    "geometry": {
                        "type": "rect",
                        "xmin": 0.0,
                        "xmax": 2.0,
                        "ymin": 3.0,
                        "ymax": 4.0,
                    },
                }
            },
        }
        specs, source = extract_predicate_specs(task)
        self.assertEqual(source, "task.predicate_specs")
        self.assertEqual(set(specs), {"top"})
        self.assertEqual(specs["top"]["evaluator"], "state_region")
        self.assertEqual(specs["top"]["geometry"]["kind"], "rect")

        region_specs = task_to_region_specs(task)
        by_id = {spec["id"]: spec for spec in region_specs}
        self.assertEqual(set(by_id), {"legacy", "top"})
        self.assertEqual(by_id["top"]["bounds"], (0.0, 2.0, 3.0, 4.0))

    def test_metadata_specs_fallback(self):
        task = {
            "metadata": {
                "predicate_specs": {
                    "b": {
                        "evaluator": "object",
                        "property": "drawer_open",
                    }
                },
            }
        }
        specs, source = extract_predicate_specs(task)
        self.assertEqual(source, "task.metadata.predicate_specs")
        self.assertEqual(specs["b"]["evaluator"], "object_state")
        self.assertEqual(specs["b"]["property"], "drawer_open")

    def test_legacy_regions_fallback(self):
        specs, source = extract_predicate_specs(
            {
                "regions": [
                    {"id": "a", "type": "circle", "center": [0.0, 0.0], "radius": 1.0},
                    {
                        "id": "b",
                        "type": "rect",
                        "xmin": 1.0,
                        "xmax": 2.0,
                        "ymin": 3.0,
                        "ymax": 4.0,
                    },
                ]
            }
        )
        self.assertEqual(source, "task.regions")
        self.assertEqual(set(specs), {"a", "b"})
        self.assertEqual(specs["a"]["evaluator"], "state_region")
        self.assertEqual(specs["b"]["geometry"]["bounds"], (1.0, 2.0, 3.0, 4.0))

    def test_region_normalization_accepts_dict_and_list(self):
        as_dict = task_regions_to_region_specs(
            {"a": {"type": "polygon", "vertices": [[0, 0], [1, 0], [0, 1]]}}
        )
        as_list = task_regions_to_region_specs(
            [{"id": "a", "type": "polygon", "vertices": [[0, 0], [1, 0], [0, 1]]}]
        )
        self.assertEqual(as_dict, as_list)
        self.assertEqual(as_dict[0]["kind"], "polygon")

    def test_validation_report_unknown_backend_and_missing_label(self):
        report = validate_predicate_specs(
            {
                "a": {"evaluator": "unknown_backend"},
                "b": {"evaluator": "oracle_state_lift"},
            },
            fallback_used="task.predicate_specs",
            labels=["a", "b", "c"],
        )
        payload = report.to_dict()
        self.assertFalse(payload["ok"])
        warning_codes = {item["code"] for item in payload["warnings"]}
        error_codes = {item["code"] for item in payload["errors"]}
        self.assertIn("unknown_evaluator_backend", warning_codes)
        self.assertIn("missing_predicate_spec_for_label", warning_codes)
        self.assertIn("missing_geometry", error_codes)
        json.dumps(payload)

    def test_object_state_specs_validate_without_geometry(self):
        task = {
            "predicate_specs": {
                "drawer_open": {
                    "evaluator": "object",
                    "property": "drawer_open",
                },
                "cube_zone": {
                    "evaluator": "object_state",
                    "object": "cube",
                    "property": "in_region",
                    "region": {
                        "type": "sphere",
                        "center": [0.33, -0.35, 0.06],
                        "radius": 0.04,
                    },
                },
            }
        }
        specs, source = extract_predicate_specs(task)
        self.assertEqual(source, "task.predicate_specs")
        self.assertEqual(specs["drawer_open"]["evaluator"], "object_state")
        self.assertEqual(task_to_region_specs(task), [])

        report = validate_predicate_specs(
            specs,
            fallback_used=source,
            labels=["drawer_open", "cube_zone"],
        ).to_dict()
        self.assertTrue(report["ok"])
        self.assertEqual(report["backends"]["drawer_open"], "object_state")

    def test_object_state_cube_region_requires_region(self):
        report = validate_predicate_specs(
            {
                "cube_zone": {
                    "evaluator": "object_state",
                    "object": "cube",
                    "property": "in_region",
                }
            },
            fallback_used="task.predicate_specs",
        ).to_dict()
        self.assertFalse(report["ok"])
        self.assertIn(
            "missing_object_region",
            {item["code"] for item in report["errors"]},
        )

    def test_object_state_composite_specs_validate_children(self):
        report = validate_predicate_specs(
            {
                "cfg_ready": {
                    "evaluator": "object_state",
                    "all_of": [
                        {"property": "button0_on"},
                        {"object": "cube", "property": "in_region"},
                    ],
                }
            },
            fallback_used="task.predicate_specs",
        ).to_dict()
        self.assertFalse(report["ok"])
        self.assertIn(
            "missing_composite_child_region",
            {item["code"] for item in report["errors"]},
        )

        ok_report = validate_predicate_specs(
            {
                "cfg_ready": {
                    "evaluator": "object_state",
                    "all_of": [
                        {"property": "button0_on"},
                        {"property": "drawer_closed"},
                    ],
                }
            },
            fallback_used="task.predicate_specs",
        ).to_dict()
        self.assertTrue(ok_report["ok"])


if __name__ == "__main__":
    unittest.main()
