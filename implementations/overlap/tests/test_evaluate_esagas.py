import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate_esagas import (
    _MiniKeyGraph,
    _oracle_labeler_for_task,
    build_real_task_batch_manifest,
    build_real_task_schedule,
    collect_node_support_from_keygraph,
    find_keygraph_dir_from_manifest,
    guard_span_obligations,
    oracle_failure_reason_with_grounding,
    jsonable,
    load_schedule_manifests,
    load_saved_task_payload,
    load_completed_real_task_diagnostics,
    load_keygraph_from_dir,
    persist_real_task_outputs,
    resolve_task_file_paths,
    run_execution_smoke,
    run_m1_acceptance_smoke,
    run_real_graph_smoke,
    run_semantic_smoke,
    select_task_indices,
    scheduled_task_keys_from_manifests,
    summarize_real_task_aggregates,
    summarize_real_task_results_dir,
    task_result_key,
    task_predicate_specs,
    task_regions_to_region_specs,
    task_to_region_specs,
    task_file_breakdown,
    write_real_task_batch_manifest,
)
from S_utils.guard_terms import GuardTerm
from S_utils.product_planner import FiniteProductExecutionPlan


class EvaluateESAGASTests(unittest.TestCase):
    def test_semantic_smoke_pipeline_finds_lasso(self):
        diagnostics = run_semantic_smoke()
        self.assertTrue(diagnostics["planner_found_plan"])
        self.assertEqual(diagnostics["start_connection_status"], "local_connected")
        self.assertEqual(diagnostics["guard_sequence"], ["g_not_a", "g_not_a", "g_a", "g_stay"])
        self.assertEqual(diagnostics["guard_report_status_counts"]["supported"], 1)
        self.assertTrue(diagnostics["prop_anchor_supported"]["a"])

    def test_cli_semantic_smoke_json(self):
        proc = subprocess.run(
            [sys.executable, "evaluate_esagas.py", "--mode", "semantic_smoke", "--json"],
            check=True,
            text=True,
            capture_output=True,
        )
        diagnostics = json.loads(proc.stdout)
        self.assertTrue(diagnostics["planner_found_plan"])
        self.assertGreaterEqual(diagnostics["num_inserted_anchors"], 1)

    def test_m1_acceptance_smoke_matrix(self):
        diagnostics = run_m1_acceptance_smoke()
        self.assertTrue(diagnostics["all_passed"])
        self.assertEqual(diagnostics["num_cases"], 5)
        names = {case["case"] for case in diagnostics["cases"]}
        self.assertEqual(
            names,
            {
                "conference_single_positive",
                "overlap_multi_positive",
                "dnf_disjunction_parallel_edges",
                "negative_only",
                "recurrent_suffix",
            },
        )

    def test_cli_m1_acceptance_smoke_json(self):
        proc = subprocess.run(
            [sys.executable, "evaluate_esagas.py", "--mode", "m1_acceptance_smoke", "--json"],
            check=True,
            text=True,
            capture_output=True,
        )
        diagnostics = json.loads(proc.stdout)
        self.assertTrue(diagnostics["all_passed"])
        self.assertEqual(diagnostics["num_passed"], 5)

    def test_execution_smoke_runs_runtime_and_oracle_layers(self):
        diagnostics = run_execution_smoke()
        self.assertTrue(diagnostics["runtime_finite_lasso_success"])
        self.assertTrue(diagnostics["oracle_success"])
        self.assertFalse(diagnostics["semantic_grounding_error"])
        self.assertEqual(diagnostics["suffix_cycles_completed"], 2)

    def test_oracle_labeler_for_task_combines_region_and_object_state(self):
        qpos = [0.0] * 25
        qpos[14:17] = [0.33, -0.35, 0.06]
        qpos[23] = -0.16
        qpos[24] = 0.0
        state = {
            "qpos": qpos,
            "button_states": [1, 0],
            "xy": [0.0, 0.0],
        }
        labeler = _oracle_labeler_for_task(
            [{"id": "at_origin", "kind": "circle", "center": [0.0, 0.0], "radius": 0.2}],
            {
                "button0_on": {"evaluator": "object_state", "property": "button0_on"},
                "drawer_open": {"evaluator": "object_state", "property": "drawer_open"},
            },
        )
        labels = labeler(state)
        self.assertIn("button0_on", labels)
        self.assertIn("drawer_open", labels)

    def test_semantic_grounding_error_classification(self):
        self.assertEqual(
            oracle_failure_reason_with_grounding(
                True,
                {"oracle_success": False, "failure_reason": "oracle_ltl_violation"},
            ),
            "semantic_grounding_error",
        )
        self.assertEqual(
            oracle_failure_reason_with_grounding(
                False,
                {"oracle_success": False, "failure_reason": "horizon_exhausted"},
            ),
            "horizon_exhausted",
        )

    def test_guard_span_obligations_union_skipped_edges(self):
        plan = FiniteProductExecutionPlan(
            states=[("s", "q0"), ("m", "q0"), ("a", "q1")],
            guards=[
                GuardTerm("g_safe", neg=frozenset({"bad"})),
                GuardTerm("g_a", pos=frozenset({"a"})),
            ],
            prefix_end_index=2,
            suffix_cycles_required=0,
            suffix_cycle_num_edges=1,
        )
        pos, neg = guard_span_obligations(plan, 0, 2)
        self.assertEqual(pos, {"a"})
        self.assertEqual(neg, {"bad"})

    def test_cli_execution_smoke_json(self):
        proc = subprocess.run(
            [sys.executable, "evaluate_esagas.py", "--mode", "execution_smoke", "--json"],
            check=True,
            text=True,
            capture_output=True,
        )
        diagnostics = json.loads(proc.stdout)
        self.assertTrue(diagnostics["runtime_finite_lasso_success"])
        self.assertTrue(diagnostics["oracle_success"])

    def test_load_keygraph_from_saved_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_dir = Path(tmp)
            key_graph = _MiniKeyGraph(nodes=[(0.0, 0.0), (1.0, 0.0)], way_steps=1.1)
            key_graph.add_base_edge(0, 1, weight=1.0)
            with (graph_dir / "keygraph.pkl").open("wb") as f:
                pickle.dump(key_graph.__dict__, f)

            loaded = load_keygraph_from_dir(graph_dir)
            self.assertEqual(loaded.base_node_cnt, 2)
            self.assertEqual(len(loaded.nodes), 2)

    def test_collect_node_support_decodes_observation_state_pairs(self):
        class _KeyNodes:
            cluster_members_raw = {
                0: [
                    {
                        "observation": [[[1], [2]], [[3], [4]]],
                        "state": [5.0, 6.0, 7.0],
                        "state_source": "qpos",
                    }
                ]
            }
            cluster_members_indices = {0: [42]}

        class _Graph:
            nodes = [[0.0, 0.0]]
            base_node_cnt = 1
            key_nodes = _KeyNodes()

        support = collect_node_support_from_keygraph(_Graph())
        carrier = support[0][0]
        self.assertEqual(carrier.observation, [[[1], [2]], [[3], [4]]])
        self.assertEqual(carrier.state, [5.0, 6.0, 7.0])
        self.assertEqual(carrier.info["state_source"], "qpos")
        self.assertEqual(carrier.index, 42)
        self.assertEqual(carrier.info["original_index"], 42)

    def test_find_keygraph_dir_from_manifest_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_name = "antmaze-giant-stitch-v0"
            manifest = root / "assets" / "manifests" / "antmaze-giant-stitch_assets.json"
            record = root / "assets" / "manifests" / "antmaze-giant-stitch_keygraph_rebuild_command.json"
            graph_dir = (
                root
                / "assets"
                / "rebuilt_keygraphs"
                / "JournalKeyGraph_antmaze-giant-stitch"
                / f"{env_name}_sd000__2026-05-28"
            )
            graph_dir.mkdir(parents=True)
            (graph_dir / "keygraph.pkl").write_bytes(b"placeholder")
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps({"env_name": env_name}))
            record.write_text(
                json.dumps(
                    {
                        "output_root": str(root / "assets" / "rebuilt_keygraphs"),
                        "run_group": "JournalKeyGraph_antmaze-giant-stitch",
                    }
                )
            )

            self.assertEqual(find_keygraph_dir_from_manifest(manifest, root), graph_dir)

    def test_real_graph_smoke_on_saved_mini_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_dir = Path(tmp)
            key_graph = _MiniKeyGraph(
                nodes=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
                way_steps=1.1,
            )
            key_graph.add_base_edge(0, 1, weight=1.0)
            key_graph.add_base_edge(1, 2, weight=1.0)
            with (graph_dir / "keygraph.pkl").open("wb") as f:
                pickle.dump(key_graph.__dict__, f)

            diagnostics = run_real_graph_smoke(keygraph_dir=graph_dir)
            self.assertTrue(diagnostics["planner_found_plan"])
            self.assertEqual(diagnostics["mode"], "real_graph_smoke")
            self.assertEqual(diagnostics["num_graph_nodes_before_semantic_insert"], 3)

    def test_jsonable_converts_scalar_like_objects(self):
        class ScalarLike:
            def item(self):
                return 7

        converted = jsonable({"x": [ScalarLike()]})
        self.assertEqual(converted, {"x": [7]})
        json.dumps(converted)

    def test_task_regions_to_region_specs_accepts_saved_task_format(self):
        specs = task_regions_to_region_specs(
            {
                "e0": {"type": "circle", "center": [1.0, 2.0], "radius": 0.5},
                "e1": {"type": "rect", "xmin": 0.0, "xmax": 1.0, "ymin": 2.0, "ymax": 3.0},
                "e2": {"type": "polygon", "vertices": [[0, 0], [1, 0], [0, 1]]},
            }
        )
        by_id = {spec["id"]: spec for spec in specs}
        self.assertEqual(by_id["e0"]["kind"], "circle")
        self.assertEqual(by_id["e1"]["bounds"], (0.0, 1.0, 2.0, 3.0))
        self.assertEqual(by_id["e2"]["kind"], "polygon")

    def test_task_predicate_specs_prefers_top_level_and_normalizes_geometry(self):
        task = {
            "regions": {
                "a": {"type": "circle", "center": [0.0, 0.0], "radius": 1.0},
            },
            "metadata": {
                "predicate_specs": {
                    "a": {
                        "evaluator": "state_region",
                        "geometry": {"type": "circle", "center": [1.0, 1.0], "radius": 1.0},
                    }
                }
            },
            "predicate_specs": {
                "b": {
                    "evaluator": "state",
                    "geometry": {"type": "rect", "xmin": 0, "xmax": 1, "ymin": 2, "ymax": 3},
                }
            },
        }
        specs, source = task_predicate_specs(task)
        self.assertEqual(source, "task.predicate_specs")
        self.assertEqual(set(specs), {"b"})
        self.assertEqual(specs["b"]["evaluator"], "state_region")
        self.assertEqual(specs["b"]["geometry"]["kind"], "rect")

        region_specs = task_to_region_specs(task)
        by_id = {spec["id"]: spec for spec in region_specs}
        self.assertEqual(set(by_id), {"a", "b"})
        self.assertEqual(by_id["b"]["bounds"], (0.0, 1.0, 2.0, 3.0))

    def test_persist_real_task_outputs_writes_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = persist_real_task_outputs(
                result_dir=Path(tmp),
                diagnostics={"oracle_success": True, "num_actions": 3},
            )
            self.assertIsNotNone(paths)
            diagnostics_path = Path(paths["diagnostics_path"])
            self.assertTrue(diagnostics_path.is_file())
            payload = json.loads(diagnostics_path.read_text())
            self.assertTrue(payload["oracle_success"])
            self.assertEqual(payload["num_actions"], 3)

    def test_resolve_task_file_paths_accepts_file_or_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_a = root / "a.json"
            task_b = root / "b.json"
            task_a.write_text("{}", encoding="utf-8")
            task_b.write_text("{}", encoding="utf-8")
            (root / "summary.json").write_text("{}", encoding="utf-8")

            self.assertEqual(resolve_task_file_paths(task_a), [task_a])
            self.assertEqual(resolve_task_file_paths(root), [task_a, task_b])

    def test_select_task_indices_supports_all_tasks_and_sharding(self):
        self.assertEqual(
            select_task_indices(num_tasks=5, task_id=0, task_limit=0),
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            select_task_indices(
                num_tasks=5,
                task_id=0,
                task_limit=5,
                task_shard_index=0,
                task_num_shards=2,
            ),
            [0, 2, 4],
        )
        self.assertEqual(
            select_task_indices(
                num_tasks=5,
                task_id=0,
                task_limit=5,
                task_shard_index=1,
                task_num_shards=2,
            ),
            [1, 3],
        )

    def test_build_real_task_schedule_loads_task_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "tasks.json"
            task_path.write_text(
                json.dumps({"tasks": [{"task_id": 0}, {"task_id": 1}, {"task_id": 2}]}),
                encoding="utf-8",
            )
            schedule = build_real_task_schedule(
                task_path,
                task_id=0,
                task_limit=3,
                task_shard_index=1,
                task_num_shards=2,
            )
            self.assertEqual(len(schedule), 1)
            self.assertEqual(schedule[0].task_path, task_path)
            self.assertEqual(schedule[0].task_indices, (1,))
            self.assertEqual(schedule[0].num_tasks_in_file, 3)

    def test_load_saved_task_payload_has_no_numpy_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "tasks.json"
            task_path.write_text(json.dumps({"tasks": [{"task_id": 1}]}), encoding="utf-8")
            self.assertEqual(load_saved_task_payload(task_path)["tasks"][0]["task_id"], 1)

    def test_load_completed_real_task_diagnostics_requires_terminal_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            diagnostics_path = result_dir / "diagnostics.json"
            diagnostics_path.write_text(json.dumps({"planner_found_plan": True}), encoding="utf-8")
            self.assertIsNone(load_completed_real_task_diagnostics(result_dir))

            diagnostics_path.write_text(
                json.dumps({"oracle_success": True, "num_actions": 12}),
                encoding="utf-8",
            )
            payload = load_completed_real_task_diagnostics(result_dir)
            self.assertIsNotNone(payload)
            self.assertTrue(payload["resume_skipped"])
            self.assertEqual(payload["num_actions"], 12)

    def test_build_real_task_batch_manifest_reports_resume_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "tasks.json"
            task_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"task_id": 5, "regions": {}, "formula_e": "F a"},
                            {"task_id": 6, "regions": {}, "formula_e": "F a"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            completed_dir = root / "results" / "tasks" / "task_0005"
            completed_dir.mkdir(parents=True)
            (completed_dir / "diagnostics.json").write_text(
                json.dumps({"oracle_success": True}),
                encoding="utf-8",
            )
            manifest = build_real_task_batch_manifest(
                task_path=task_path,
                result_dir=root / "results",
                task_id=0,
                task_limit=2,
                resume=True,
            )
            self.assertEqual(manifest["num_scheduled_tasks"], 2)
            self.assertEqual(manifest["num_completed_for_resume"], 1)
            self.assertEqual(manifest["num_pending_if_resume"], 1)
            self.assertTrue(manifest["schedule"][0]["tasks"][0]["completed_for_resume"])

    def test_write_real_task_batch_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "run_manifest.json"
            payload = {"mode": "real_task_schedule", "num_scheduled_tasks": 0}
            written = write_real_task_batch_manifest(output_path, payload)
            self.assertTrue(output_path.is_file())
            self.assertEqual(written["schedule_manifest_path"], str(output_path))
            self.assertEqual(json.loads(output_path.read_text())["num_scheduled_tasks"], 0)

    def test_summarize_real_task_aggregates_counts_success_and_failures(self):
        summary = summarize_real_task_aggregates(
            [
                {
                    "results": [
                        {
                            "ltl_task_path": "tasks.json",
                            "task_index": 0,
                            "task_id": 10,
                            "oracle_success": True,
                            "runtime_finite_lasso_success": True,
                            "unsupported_guard_terms": [],
                            "semantic_grounding_error": False,
                            "num_actions": 5,
                            "prefix_completed": True,
                            "suffix_cycles_completed": 1,
                            "suffix_cycles_required": 1,
                        },
                        {
                            "ltl_task_path": "tasks.json",
                            "task_index": 1,
                            "task_id": 11,
                            "oracle_success": False,
                            "runtime_finite_lasso_success": False,
                            "oracle_failure_reason": "oracle_ltl_violation",
                            "runtime_failure_reason": "runtime_unrealized",
                            "unsupported_guard_terms": ["q0->q1:0"],
                            "semantic_grounding_error": True,
                            "num_actions": 9,
                        },
                    ]
                }
            ]
        )
        self.assertEqual(summary["num_tasks"], 2)
        self.assertEqual(summary["oracle_success_count"], 1)
        self.assertEqual(summary["runtime_success_count"], 1)
        self.assertEqual(summary["mean_num_actions"], 7.0)
        self.assertEqual(summary["mean_success_num_actions"], 5.0)
        self.assertEqual(summary["oracle_failure_reason_counts"], {"oracle_ltl_violation": 1})
        self.assertEqual(summary["runtime_failure_reason_counts"], {"runtime_unrealized": 1})
        self.assertEqual(summary["tasks"][0]["task_id"], 10)
        self.assertTrue(summary["tasks"][0]["runtime_success"])
        self.assertEqual(summary["tasks"][1]["unsupported_guard_terms"], ["q0->q1:0"])
        self.assertTrue(summary["tasks"][1]["semantic_grounding_error"])
        self.assertEqual(summary["unsupported_guard_terms"], ["q0->q1:0"])
        self.assertEqual(summary["semantic_grounding_error_count"], 1)
        self.assertEqual(summary["num_unique_task_results"], 2)
        self.assertEqual(summary["num_duplicate_task_results"], 0)
        self.assertEqual(summary["by_task_file"]["tasks.json"]["num_tasks"], 2)

    def test_summarize_real_task_aggregates_reports_duplicate_results(self):
        duplicate = {
            "ltl_task_path": "tasks.json",
            "task_index": 0,
            "task_id": 0,
            "oracle_success": True,
            "runtime_finite_lasso_success": True,
        }
        summary = summarize_real_task_aggregates([{"results": [duplicate, duplicate]}])
        self.assertEqual(summary["num_tasks"], 2)
        self.assertEqual(summary["num_unique_task_results"], 1)
        self.assertEqual(summary["num_duplicate_task_results"], 1)
        self.assertEqual(summary["duplicate_task_keys"], ["tasks.json::0"])

    def test_task_result_key_and_task_file_breakdown(self):
        self.assertEqual(task_result_key({"ltl_task_path": "a.json", "task_index": 3}), "a.json::3")
        breakdown = task_file_breakdown(
            [
                {
                    "ltl_task_path": "a.json",
                    "oracle_success": True,
                    "runtime_finite_lasso_success": True,
                    "num_actions": 4,
                },
                {
                    "ltl_task_path": "a.json",
                    "oracle_success": False,
                    "runtime_finite_lasso_success": False,
                    "num_actions": 8,
                },
            ]
        )
        self.assertEqual(breakdown["a.json"]["num_tasks"], 2)
        self.assertEqual(breakdown["a.json"]["oracle_success_rate"], 0.5)
        self.assertEqual(breakdown["a.json"]["mean_num_actions"], 6.0)

    def test_summarize_real_task_results_dir_loads_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "aggregate_diagnostics.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "oracle_success": True,
                                "runtime_finite_lasso_success": True,
                                "num_actions": 4,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_real_task_results_dir(root, output_path=root / "summary.json")
            self.assertEqual(summary["num_tasks"], 1)
            self.assertEqual(summary["oracle_success_rate"], 1.0)
            self.assertTrue((root / "summary.json").is_file())

    def test_summarize_real_task_results_dir_uses_schedule_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard0 = root / "shard0"
            shard1 = root / "shard1"
            shard0.mkdir()
            shard1.mkdir()
            (shard0 / "aggregate_diagnostics.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "ltl_task_path": "tasks.json",
                                "task_index": 0,
                                "task_id": 0,
                                "oracle_success": True,
                                "runtime_finite_lasso_success": True,
                                "num_actions": 4,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (shard0 / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schedule": [
                            {
                                "task_path": "tasks.json",
                                "tasks": [{"task_index": 0, "task_id": 0}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (shard1 / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schedule": [
                            {
                                "task_path": "tasks.json",
                                "tasks": [{"task_index": 1, "task_id": 1}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_real_task_results_dir(root)
            self.assertEqual(summary["num_schedule_manifests"], 2)
            self.assertEqual(summary["num_scheduled_tasks_from_manifests"], 2)
            self.assertEqual(summary["num_missing_scheduled_tasks"], 1)
            self.assertEqual(summary["missing_scheduled_task_keys"], ["tasks.json::1"])

    def test_load_schedule_manifests_and_scheduled_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "run_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schedule": [
                            {
                                "task_path": "tasks.json",
                                "tasks": [{"task_index": 2, "task_id": 7}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifests = load_schedule_manifests(root)
            self.assertEqual(len(manifests), 1)
            self.assertEqual(scheduled_task_keys_from_manifests(manifests), ["tasks.json::2"])


if __name__ == "__main__":
    unittest.main()
