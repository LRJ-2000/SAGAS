"""Group saved task outcomes, then call the original metrics implementations."""

from collections import defaultdict
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

from release_io import IMPLEMENTATIONS, ROOT, read_json, sha256, write_json


def load_report(suite, path):
    spec = importlib.util.spec_from_file_location(
        f"_sagas_{suite}_report", IMPLEMENTATIONS / suite / path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_path(suite, record, results_dir, manifest_hash):
    folder = results_dir / record["env"]
    if suite == "navigation":
        folder = folder / record["difficulty"] / record["method"]
        paths = list((folder / "results").rglob("overall_ltl_results.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one navigation result CSV under {folder}")
        path = paths[0]
    elif suite == "overlap":
        folder = folder / record["shard"]
        path = folder / "results/batch_summary.json"
    else:
        path = folder / "results/stress_suite_summary.json"
    run_path = folder / "run.json"
    run = read_json(run_path)
    if (run["status"] != "complete" or run["task_id"] is not None
            or run["input_task_sha256"] != record["task_sha256"]
            or any(run["preset"][k] != record[k] for k in run["preset"])):
        raise ValueError(f"Expected a completed run on the full fixed task file: {folder}")
    expected = {
        "asset_ids": record["assets"],
        "dataset_ids": record["datasets"],
        "asset_manifest_sha256": manifest_hash,
    }
    mismatches = [key for key, value in expected.items() if run.get(key) != value]
    if mismatches:
        raise ValueError(
            f"Asset provenance missing or mismatched in {run_path}: {', '.join(mismatches)}. "
            "Use results from the matching release."
        )
    return path, run["flags"]


def check_task_rows(tasks, rows, formula_key):
    if len(rows) != len(tasks) or any(
        str(row[formula_key]) != str(task["formula_e"])
        or ("task_id" in row and int(row["task_id"]) != int(task["task_id"]))
        or row.get("status", "ok") != "ok"
        for task, row in zip(tasks, rows)
    ):
        raise ValueError("Expected a complete, ordered result for the selected fixed task file.")


def preset_protocol(record, flags):
    preset = {key: record[key] for key in ("env", "difficulty", "method", "shard") if key in record}
    defaults = record["flags"]
    # Only compare preset settings and explicit seeds, not launcher paths or GPU IDs.
    overrides = {
        key: {"preset": defaults.get(key), "effective": flags.get(key)}
        for key in sorted((defaults.keys() | {"seed"}) - {"gpu"})
        if defaults.get(key) != flags.get(key)
    }
    return {"preset": preset, "flag_overrides": overrides}


def summarize(suite, records, results_dir, output):
    manifest_hash = sha256(ROOT / "assets/manifest.json")
    inputs = [(r, *result_path(suite, r, results_dir, manifest_hash)) for r in records]
    presets = [preset_protocol(r, flags) for r, _, flags in inputs]
    modified = any(preset["flag_overrides"] for preset in presets)
    output.mkdir(parents=True, exist_ok=False)
    if suite == "navigation":
        report = load_report(suite, "summarize_navigation.py")
        rows = []
        for r, source, flags in inputs:
            tasks = read_json(ROOT / r["task_file"])["tasks"]
            with source.open(newline="", encoding="utf-8") as f:
                check_task_rows(tasks, list(csv.DictReader(f)), "ltl_formula")
            metrics = report.summarize_overall_csv(source, max_steps=flags["max_steps"])
            rows.append({"environment": r["env"], "difficulty": r["difficulty"], **metrics})
        report.write_csv_rows(output / "navigation.csv", rows)
    elif suite == "overlap":
        report = load_report(suite, "rebuttal_e5/summarize_e5_overlap.py")
        groups = defaultdict(list)
        for r, source, _ in inputs:
            payload = read_json(source)
            rows = payload["file_aggregates"][0]["results"]
            tasks = read_json(ROOT / r["task_file"])["tasks"]
            check_task_rows(tasks, rows, "formula")
            for task, row in zip(tasks, rows):
                family = task["metadata"]["e5_group"]
                for group in ((r["env"], family), (r["env"], "all"),
                              ("all", family), ("all", "all")):
                    groups[group].append(row)
        summary = []
        for (env, family), items in sorted(groups.items()):
            metrics = report.aggregate(family, items)
            summary.append({"environment": env, "family": family, **{
                key: metrics[key] for key in (
                    "tasks", "plans_found", "plan_rate", "oracle_successes",
                    "oracle_success_rate", "wilson_lo", "wilson_hi"
                )
            }})
        report.write_csv(output / "overlap.csv", summary)
    else:
        (r, result, _), = inputs
        task = ROOT / r["task_file"]
        check_task_rows(read_json(task)["tasks"], read_json(result)["tasks"], "formula")
        subprocess.run([
            sys.executable, "-B",
            str(IMPLEMENTATIONS / "scene/rebuttal_e5/summarize_e6_structured_manipulation.py"),
            "--task-path", str(task), "--result-summary", str(result),
            "--output-dir", str(output),
        ], check=True)
    protocol_path = output / "summary_protocol.json"
    write_json(protocol_path, {
        "suite": suite,
        "protocol": "modified_preset" if modified else "fixed_preset",
        "presets": presets,
    })
    if modified:
        print(f"Warning: this summary includes modified evaluation settings. See {protocol_path}.",
              file=sys.stderr)
