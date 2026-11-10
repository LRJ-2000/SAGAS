"""Check public dispatch and asset binding without loading an environment."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from release_io import ROOT, check_file, read_json, sha256, verify_assets
from release_reports import summarize
from sagas import make_parser, main, select, suite_job


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read_json(ROOT / "assets/manifest.json")
        cls.files = cls.manifest["files"]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.out = self.root / "out"

    def command(self, suite, *flags):
        return ["evaluate", suite, *flags, "--output-dir", str(self.out),
                "--assets-dir", str(self.root / "assets"),
                "--data-dir", str(self.root / "data"), "--dry-run"]

    def job(self, suite, *flags):
        args = make_parser().parse_args(self.command(suite, *flags))
        record, = select(suite, args.env, args.difficulty, args.method, args.shard)
        return suite_job(args, record, self.files), record

    def test_every_navigation_cell_keeps_recorded_tasks_and_parameters(self):
        rows = select("navigation", all_cells=True)
        self.assertEqual(len(rows), 42)
        tasks_by_cell = {}
        for row in rows:
            (suite, script, argv, _, generated, note), _ = self.job(
                "navigation", "--env", row["env"], "--difficulty", row["difficulty"],
                "--method", "sagas")
            self.assertEqual(row["method"], "sagas")
            self.assertNotIn("reference", row)
            self.assertEqual((suite, script), ("navigation", "evaluate_ltl.py"))
            self.assertEqual(generated, {})
            for key, value in row["flags"].items():
                self.assertEqual(note["flags"][key], value)
            for role, key in (("policy", "policy_path"), ("graph", "keygraph_path")):
                self.assertEqual(note["flags"][key],
                                 self.root / "assets" / self.files[row["assets"][role]]["path"])
            task = read_json(note["flags"]["ltl_task_path"])
            self.assertEqual([r["task_id"] for r in task["tasks"]], list(range(100)))
            key = row["env"], row["difficulty"]
            self.assertEqual(tasks_by_cell.setdefault(key, task["tasks"]), task["tasks"])
        self.assertEqual(sum(map(len, tasks_by_cell.values())), 4200)

    def test_seeded_subset_keeps_full_task_object(self):
        (_, _, _, _, generated, note), record = self.job(
            "navigation", "--env", "antmaze-giant-stitch-v0",
            "--difficulty", "easy", "--task-id", "23")
        payload, = generated.values()
        self.assertEqual(payload["num_tasks"], 1)
        source = read_json(ROOT / record["task_file"])
        self.assertEqual(payload["tasks"], [source["tasks"][23]])
        self.assertTrue(note["flags"]["ltl_use_task_seed"])

    def test_overlap_dispatch_and_binding(self):
        records = select("overlap", all_cells=True)
        self.assertEqual(sum(r["task_count"] for r in records), 320)
        for rec in records:
            (suite, script, argv, _, inputs, note), _ = self.job(
                "overlap", "--env", rec["env"], "--shard", rec["shard"], "--task-id", "0")
            self.assertEqual((suite, script), ("overlap", "evaluate_esagas.py"))
            manifest, = inputs.values()
            self.assertEqual(manifest["tdr_path"], self.files[rec["assets"]["tdr"]]["path"])
            self.assertEqual(manifest["policy_path"], self.files[rec["assets"]["policy"]]["path"])
            self.assertEqual(note["flags"]["mode"], "real_task_batch")
            self.assertEqual(note["flags"]["suffix-cycles"], 2)
            self.assertEqual(note["flags"]["local-step-budget"], 8000)
            self.assertEqual(note["flags"]["anchor-reached-threshold"], 8)
            self.assertEqual(argv[argv.index("--task-id") + 1], "0")

    def test_scene_keeps_own_execution_protocol(self):
        (suite, script, argv, _, _, note), rec = self.job(
            "scene", "--task-id", "0", "--render", "--render-every", "10")
        self.assertEqual((suite, script), ("scene", "tools/run_scene_object_state_stress_suite.py"))
        self.assertEqual(rec["task_count"], 120)
        self.assertEqual(note["flags"]["suffix-cycles"], 1)
        self.assertEqual(note["flags"]["semantic-repulsion"], "auto")
        self.assertEqual(note["flags"]["env-max-episode-steps"], 8000)
        self.assertNotIn("--no-render-video", argv)
        self.assertEqual(argv[argv.index("--render-every") + 1], "10")
        self.assertNotIn("--seed", argv)

    def test_override_priority_and_fixed_variants(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"flags": {"gpu": 2, "seed": 321, "suffix-cycles": 3}}))
        (_, _, _, _, _, note), _ = self.job(
            "overlap", "--env", "antmaze-giant-stitch-v0", "--shard", "events",
            "--config", str(config), "--gpu", "1")
        self.assertEqual(note["flags"]["gpu"], 1)
        self.assertEqual(note["flags"]["seed"], 321)
        self.assertEqual(note["flags"]["suffix-cycles"], 3)
        config.write_text(json.dumps({"flags": {"semantic-evaluator": "reference"}}))
        with self.assertRaisesRegex(ValueError, "Unsupported or fixed"):
            self.job("overlap", "--env", "antmaze-giant-stitch-v0", "--shard", "events",
                     "--config", str(config))

    def test_unsupported_combinations(self):
        with self.assertRaises(ValueError):
            select("overlap", "antmaze-giant-stitch-v0", difficulty="easy")
        for method in ("lf-gas", "ga-gas", "ap-hiql"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as fail:
                make_parser().parse_args(self.command("navigation", "--method", method))
            self.assertEqual(fail.exception.code, 2)
            with self.assertRaisesRegex(ValueError, "only supports SAGAS"):
                select("navigation", method=method, all_cells=True)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            make_parser().parse_args(self.command("navigation", "--hiql-impls", str(self.root)))
        with self.assertRaises(ValueError):
            self.job("scene", "--task-id", "10000")

    def test_dry_run_never_writes_or_requires_assets(self):
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(self.command("scene", "--task-id", "0")), 0)
        record = json.loads(out.getvalue())
        self.assertEqual(record["task_id"], 0)
        self.assertFalse(self.out.exists())
        with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as fail:
            main(["assets", "scene", "--assets-dir", str(self.root / "assets"),
                  "--data-dir", str(self.root / "data")])
        self.assertEqual(fail.exception.code, 2)
        self.assertIn("Missing", err.getvalue())
        self.assertFalse(self.out.exists())

    def test_hash_check_detects_wrong_bytes(self):
        asset = self.root / "weights.pkl"
        asset.write_bytes(b"expected")
        rec = {"bytes": 8, "sha256": sha256(asset)}
        check_file(asset, rec)
        asset.write_bytes(b"wrong!!!")
        with self.assertRaises(ValueError):
            check_file(asset, rec)

    def test_pretrained_whitelist_keeps_exact_bundles(self):
        selected = set(self.manifest["hub"]["assets"])
        self.assertEqual(len(selected), 17)
        self.assertEqual(sum(self.files[k]["bytes"] for k in selected), 933_767_906)
        self.assertFalse(any(item["kind"] == "hiql" for item in self.files.values()))
        nav = [r for r in select("navigation", all_cells=True)
               if set(r["assets"].values()) <= selected]
        self.assertEqual(len(nav), 15)
        self.assertEqual(len({r["env"] for r in nav}), 5)
        scene, = select("scene")
        self.assertTrue(set(scene["assets"].values()) <= selected)
        overlap = [r for r in select("overlap", all_cells=True)
                   if set(r["assets"].values()) <= selected]
        self.assertEqual({r["env"] for r in overlap}, {"antmaze-large-explore-v0"})

    def test_hub_download_requires_a_pin_and_checks_bytes(self):
        blob = self.root / "cache"
        blob.write_bytes(b"expected")
        item = {"path": "graph.pkl", "storage": "assets", "bytes": 8, "sha256": sha256(blob)}
        hub = {"repo_id": "lrj-2000/SAGAS", "repo_type": "model",
               "revision": None, "assets": ["graph"]}
        assets = self.root / "assets"
        with self.assertRaisesRegex(FileNotFoundError, "revision not pinned"):
            verify_assets(["graph"], {"graph": item}, assets, self.root, fetch=True, hub=hub)
        self.assertFalse(assets.exists())
        hub["revision"] = "a" * 40
        with patch.dict("sys.modules", {"huggingface_hub": types.SimpleNamespace(
            hf_hub_download=lambda **kwargs: str(blob) if kwargs == {
                "repo_id": "lrj-2000/SAGAS", "repo_type": "model",
                "revision": "a" * 40, "filename": "graph.pkl",
            } else self.fail("Unexpected Hub download arguments")
        )}):
            verify_assets(["graph"], {"graph": item}, assets, self.root, fetch=True, hub=hub)
        self.assertEqual((assets / "graph.pkl").read_bytes(), blob.read_bytes())

    def test_summary_requires_user_results_and_preserves_metrics(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            make_parser().parse_args(["summarize", "scene", "--output-dir", str(self.out)])
        preset = {"env": "test-env", "method": "sagas", "difficulty": "easy"}
        task = self.root / "task.json"
        task.write_text(json.dumps({"tasks": [{"formula_e": "<>a"}, {"formula_e": "<>b"}]}))
        manifest = self.root / "assets/manifest.json"
        manifest.parent.mkdir()
        manifest.write_text("{}")
        record = {**preset, "task_file": task.name, "task_sha256": sha256(task),
                  "assets": {"graph": "graph-id", "policy": "policy-id"},
                  "datasets": ["dataset-id"],
                  "flags": {"max_steps": 10, "gpu": 0}}
        results_dir = self.root / "runs"
        leaf = results_dir / "test-env/easy/sagas"
        (leaf / "results").mkdir(parents=True)
        run = {
            "status": "complete", "task_id": None, "input_task_sha256": record["task_sha256"],
            "preset": preset, "flags": {"max_steps": 10, "gpu": 2},
            "asset_ids": record["assets"], "dataset_ids": record["datasets"],
            "asset_manifest_sha256": sha256(manifest),
        }
        (leaf / "run.json").write_text(json.dumps(run))
        with (leaf / "results/overall_ltl_results.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("ltl_formula", "success_rate", "suffix_one_cycle_success",
                             "prefix_steps", "suffix_one_cycle_steps"))
            writer.writerow(("<>a", 1, "[True]", "[2]", "[3]"))
            writer.writerow(("<>b", 0, "[False]", "[8]", "[10]"))
        with patch("release_reports.ROOT", self.root):
            summarize("navigation", [record], results_dir, self.out)
        with (self.out / "navigation.csv").open(newline="") as f:
            row, = csv.DictReader(f)
        self.assertEqual(int(row["tasks"]), 2)
        self.assertEqual(float(row["success_rate"]), 0.5)
        self.assertEqual(float(row["avg_normalized_capped_cost"]), 0.625)
        fixed = read_json(self.out / "summary_protocol.json")
        self.assertEqual(fixed["protocol"], "fixed_preset")
        self.assertEqual(fixed["presets"], [{"preset": preset, "flag_overrides": {}}])

        run["flags"]["max_steps"] = 5
        (leaf / "run.json").write_text(json.dumps(run))
        custom = self.root / "custom"
        with patch("release_reports.ROOT", self.root), redirect_stderr(io.StringIO()) as stderr:
            summarize("navigation", [record], results_dir, custom)
        modified = read_json(custom / "summary_protocol.json")
        self.assertIn("modified evaluation settings", stderr.getvalue())
        self.assertEqual(modified["protocol"], "modified_preset")
        self.assertEqual(modified["presets"][0]["flag_overrides"], {
            "max_steps": {"preset": 10, "effective": 5},
        })
        with (custom / "navigation.csv").open(newline="") as f:
            row, = csv.DictReader(f)
        self.assertEqual(float(row["avg_normalized_capped_cost"]), 0.75)

    def test_summary_rejects_unverified_assets_before_writing_output(self):
        (_, _, _, leaf, _, note), record = self.job("scene")
        leaf.mkdir(parents=True)
        run = {**note, "status": "complete"}
        mismatches = {
            "asset_ids": {**note["asset_ids"], "graph": "different-graph"},
            "dataset_ids": ["different-dataset"],
            "asset_manifest_sha256": "0" * 64,
        }
        for field, value in mismatches.items():
            for change in ("missing", "mismatched"):
                with self.subTest(field=field, change=change):
                    saved = {**run, field: value}
                    if change == "missing":
                        saved.pop(field)
                    (leaf / "run.json").write_text(json.dumps(saved, default=str))
                    report = self.root / f"report-{field}-{change}"
                    with self.assertRaisesRegex(ValueError, f"Asset provenance.*{field}"):
                        summarize("scene", [record], self.out, report)
                    self.assertFalse(report.exists())

    def test_scene_summary_counts_joint_success_per_task(self):
        (_, _, _, leaf, _, note), record = self.job("scene")
        tasks = read_json(ROOT / record["task_file"])["tasks"]
        (leaf / "results").mkdir(parents=True)
        (leaf / "run.json").write_text(json.dumps({**note, "status": "complete"}, default=str))
        (leaf / "results/stress_suite_summary.json").write_text(json.dumps({"tasks": [
            {"task_id": task["task_id"], "formula": task["formula_e"], "status": "ok",
             "runtime_success": index in {0, 1}, "oracle_success": index in {0, 2},
             "planning_timing_seconds": {"total": 1.0}}
            for index, task in enumerate(tasks)
        ]}))
        report = self.root / "scene-summary"
        summarize("scene", [record], self.out, report)
        overall = read_json(report / "structured_manipulation_summary.json")["overall"]
        self.assertEqual(overall["runtime_successes"], 2)
        self.assertEqual(overall["oracle_successes"], 2)
        self.assertEqual(overall["finite_lasso_successes"], 1)
        self.assertAlmostEqual(overall["finite_lasso_success_rate"], 1 / len(tasks))
        self.assertEqual(read_json(report / "summary_protocol.json")["protocol"], "fixed_preset")


if __name__ == "__main__":
    unittest.main()
