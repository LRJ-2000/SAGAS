#!/usr/bin/env python3
"""Run the recorded SAGAS benchmarks using their original entrypoints."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from release_io import (
    IMPLEMENTATIONS, ROOT, asset_path, read_json, sha256, verify_assets, write_json,
)
from release_reports import summarize


SUITES = ("navigation", "overlap", "scene")
ENTRIES = {
    "navigation": "evaluate_ltl.py",
    "overlap": "evaluate_esagas.py",
    "scene": "tools/run_scene_object_state_stress_suite.py",
}
FIXED_FLAGS = {
    "mode", "semantic-evaluator", "task-ids", "task-limit", "no-render-video",
    "resume", "json",
}


def select(suite, env=None, difficulty=None, method=None, shard=None, all_cells=False):
    if method not in (None, "sagas"):
        raise ValueError("This release only supports SAGAS.")
    if suite != "navigation" and difficulty:
        raise ValueError(f"{suite} has no difficulty selector.")
    if shard and suite != "overlap":
        raise ValueError("--shard is specific to overlap.")
    if suite == "navigation" and not all_cells and (not env or not difficulty):
        raise ValueError("Choose --env and --difficulty for navigation, or explicitly use --all.")
    if suite == "overlap" and not env and not all_cells:
        raise ValueError("Choose --env for overlap, or explicitly use --all.")
    rows = [
        r for r in read_json(ROOT / "configs" / (suite + ".json"))
        if (env is None or r["env"] == env)
        and (difficulty is None or r.get("difficulty") == difficulty)
        and r["method"] == "sagas"
        and (shard is None or r.get("shard") == shard)
    ]
    if not rows:
        raise ValueError("No recorded preset matches these selectors.")
    return rows


def resolved_flags(preset, config, **overrides):
    flags = dict(preset)
    if config:
        payload = read_json(config)
        if not isinstance(payload, dict) or set(payload) != {"flags"} or not isinstance(payload["flags"], dict):
            raise ValueError('Config must have the form {"flags": {...}}.')
        for key, value in payload["flags"].items():
            if key in FIXED_FLAGS or key not in flags or value is None:
                raise ValueError(f"Unsupported or fixed config flag: {key}")
            if key == "agent_config":
                if not isinstance(value, dict) or not set(value) <= set(flags[key]):
                    raise ValueError("Unknown agent_config field.")
                flags[key] = {**flags[key], **value}
            else:
                flags[key] = value
    flags.update({k: v for k, v in overrides.items() if v is not None})
    return flags


def absl_args(flags):
    argv = []
    for key, value in flags.items():
        if value is None:
            continue
        if key == "agent_config":
            argv.append("--agent_config=M_utils/agents/gas.py")
            for name, val in value.items():
                if val is not None:
                    if isinstance(val, list):
                        val = repr(tuple(val))
                    elif isinstance(val, bool):
                        val = str(val).lower()
                    argv.append(f"--agent_config.{name}={val}")
        elif key == "gpu":
            # The original launcher reads this flag from argv before absl starts.
            argv.extend(("--gpu", str(value)))
        else:
            argv.append(f"--{key}={str(value).lower() if isinstance(value, bool) else value}")
    return argv


def argparse_args(flags):
    argv = []
    for key, value in flags.items():
        if value is True:
            argv.append(f"--{key}")
        elif value is not None and value is not False:
            argv.extend((f"--{key}", str(value)))
    return argv


def suite_job(args, record, files):
    suite = args.suite
    leaf = args.output_dir / record["env"]
    task = ROOT / record["task_file"] if args.task_file is None else args.task_file
    if args.task_file is None and sha256(task) != record["task_sha256"]:
        raise ValueError(f"Fixed task file changed: {task}")
    payload = read_json(task)
    tasks = payload["tasks"]
    if not tasks or payload.get("env_name", record["env"]) != record["env"]:
        raise ValueError(f"Task file must contain tasks for {record['env']}.")
    index = None
    if args.task_id is not None:
        matches = [i for i, row in enumerate(tasks) if int(row["task_id"]) == args.task_id]
        if len(matches) != 1:
            raise ValueError("--task-id must identify one saved task in this file.")
        index, = matches
    paths = {k: asset_path(files[v], args.assets_dir, args.data_dir)
             for k, v in record["assets"].items()}
    flags = resolved_flags(record["flags"], args.config, seed=args.seed, gpu=args.gpu)
    generated = {}
    if suite == "navigation":
        leaf = leaf / record["difficulty"] / record["method"]
        if args.render or args.render_every is not None:
            raise ValueError("Use the original navigation renderer or render an overlap/Scene task.")
        if index is not None:
            if not flags.get("ltl_use_task_seed") or "seed" not in tasks[index]:
                raise ValueError("A navigation subset requires the original per-task seed.")
            payload = {**payload, "num_tasks": 1, "tasks": [tasks[index]]}
            task = leaf / "selected_task.json"
            generated[task] = payload
        flags.update(env_name=record["env"], run_group="navigation",
                     ltl_task_path=task, keygraph_path=paths["graph"],
                     policy_path=paths["policy"], save_eval_dir=leaf / "results")
        argv = absl_args(flags)
    else:
        if suite == "overlap":
            leaf = leaf / record["shard"]
        manifest = leaf / "asset_config.json"
        generated[manifest] = {
            "env_name": record["env"],
            "tdr_path": str(paths["tdr"].relative_to(args.assets_dir)),
            "policy_path": str(paths["policy"].relative_to(args.assets_dir)),
            "agent_config_overrides": record["agent_config_overrides"],
        }
        flags.update({
            "manifest": manifest, "target-root": args.assets_dir,
            "ltl-task-path": task, "keygraph-dir": paths["graph"].parent,
        })
        if suite == "overlap":
            flags["result-dir"] = leaf / "results"
            if index is not None:
                flags.update({"task-id": index, "task-limit": 1})
            if args.render:
                flags["render-video"] = leaf / "rollout.mp4"
        else:
            flags["result-root"] = leaf / "results"
            if index is not None:
                flags["task-ids"] = str(index)
            if args.render:
                flags["no-render-video"] = False
        if args.render_every is not None:
            flags["render-every"] = args.render_every
        argv = argparse_args(flags)
    note = {"suite": suite, "preset": {k: record[k] for k in ("env", "difficulty", "method", "shard") if k in record},
            "input_task_sha256": sha256(ROOT / record["task_file"] if args.task_file is None else args.task_file),
            "task_id": args.task_id, "asset_ids": record["assets"], "dataset_ids": record["datasets"],
            "asset_manifest_sha256": sha256(ROOT / "assets/manifest.json"), "flags": flags}
    return (suite, ENTRIES[suite], argv, leaf, generated, note)


def entry_job(args, files):
    action = args.command
    suite = "scene" if action == "build" and args.suite == "scene" else "navigation"
    env = args.env or ("scene-play-v0" if suite == "scene" else None)
    if env is None:
        raise ValueError("--env is required.")
    asset_ids = []
    out = args.output_dir
    if action == "generate":
        if args.count is not None and args.count < 1:
            raise ValueError("--count must be positive.")
        if args.suite == "overlap":
            if env != "antmaze-giant-stitch-v0":
                raise ValueError("The original overlap generator preset is for AntMaze Giant-Stitch.")
            if args.difficulty is not None:
                raise ValueError("Overlap generation does not use --difficulty.")
            return ("overlap", "rebuttal_e5/build_e5_overlap_suite.py",
                    ["--output", str(out / "tasks.json"), "--manifest-output", str(out / "generation.json"),
                     "--seed", str(args.seed if args.seed is not None else 20260724),
                     "--tasks-per-group", str(args.count if args.count is not None else 8)],
                    out, {}, {"environment": env}), []
        flags = {"env_name": env, "complexity": args.difficulty or "easy", "save_path": out / "tasks.json"}
        if args.seed is not None:
            flags["seed"] = args.seed
        if args.count is not None:
            flags["num_tasks"] = args.count
        return (suite, "generate_ltl_tasks.py", absl_args(flags), out, {}, {"flags": flags}), []
    if suite == "scene":
        row = read_json(ROOT / "configs/scene_build.json")
        if env != row["env"]:
            raise ValueError("Scene build supports scene-play-v0.")
    else:
        stage = "graph" if action == "build" else args.stage
        presets = read_json(ROOT / "configs/training.json")
        if env not in presets:
            raise ValueError(f"No recorded training preset for {env}.")
        row = presets[env][stage]
    asset_ids.extend(row["datasets"])
    flags = resolved_flags(row["flags"], args.config, gpu=args.gpu, seed=args.seed)
    flags.update(env_name=env, run_group="sagas")
    if action == "build" or args.stage == "policy":
        tdr = args.tdr_checkpoint or asset_path(files[row["tdr"]], args.assets_dir, args.data_dir)
        if args.tdr_checkpoint is None:
            asset_ids.append(row["tdr"])
        elif not args.dry_run and not tdr.is_file():
            raise FileNotFoundError(tdr)
        flags["tdr_path"] = tdr
    elif args.tdr_checkpoint is not None:
        raise ValueError("--tdr-checkpoint is for policy training or graph construction.")
    if action == "build":
        script, output_flag = "construct_graph.py", "save_graph_dir"
    elif args.stage == "policy":
        script, output_flag = "train_policy.py", "save_policy_dir"
    else:
        script, output_flag = "pretrain_tdr.py", "save_tdr_dir"
    flags[output_flag] = out / "results"
    note = {"flags": flags, "asset_ids": asset_ids,
            "asset_manifest_sha256": sha256(ROOT / "assets/manifest.json")}
    return (suite, script, absl_args(flags), out, {}, note), asset_ids


def run_job(job, env, dry_run):
    suite, script, argv, output, generated, note = job
    cwd = IMPLEMENTATIONS / suite
    command = [sys.executable, "-B", str(cwd / script), *argv]
    note = {**note, "cwd": str(cwd), "command": command,
            "data_dir": env.get("SAGAS_DATA_DIR"), "ltl2ba": env.get("SAGAS_LTL2BA")}
    note = json.loads(json.dumps(note, default=str))
    if dry_run:
        print(json.dumps(note, indent=2, sort_keys=True))
        return
    output.mkdir(parents=True, exist_ok=True)
    for path, value in generated.items():
        write_json(path, value)
    if generated:
        note["generated_inputs"] = {str(p.name): sha256(p) for p in generated}
    record = output / "run.json"
    write_json(record, {**note, "status": "started"})
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        if suite == "scene" and script == ENTRIES["scene"]:
            summary = read_json(output / "results/stress_suite_summary.json")
            if any(row.get("status") != "ok" for row in summary["tasks"]):
                raise RuntimeError("Scene runner recorded an execution error; inspect its summary.")
    except BaseException as exc:
        write_json(record, {**note, "status": "error", "error": str(exc)})
        raise
    write_json(record, {**note, "status": "complete"})


def add_selection(parser):
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument("--env")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    parser.add_argument("--method", choices=("sagas",), help="Optional; this release runs SAGAS.")
    parser.add_argument("--shard")
    parser.add_argument("--all", dest="all_cells", action="store_true")


def abs_path(value):
    return Path(value).expanduser().resolve()


def add_assets(parser):
    parser.add_argument("--assets-dir", type=abs_path, default=ROOT / "assets")
    parser.add_argument("--data-dir", type=abs_path, default=Path.home() / ".ogbench/data")


def add_run(parser):
    add_assets(parser)
    parser.add_argument("--output-dir", required=True, type=abs_path)
    parser.add_argument("--config", type=abs_path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--ltl2ba", type=abs_path)
    parser.add_argument("--dry-run", action="store_true")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    evaluate = subs.add_parser("evaluate", help="Evaluate a fixed benchmark or one saved task.")
    add_selection(evaluate)
    add_run(evaluate)
    evaluate.add_argument("--task-id", type=int)
    evaluate.add_argument("--task-file", type=abs_path)
    evaluate.add_argument("--render", action="store_true")
    evaluate.add_argument("--render-every", type=int)
    assets = subs.add_parser("assets", help="Verify the paired assets; optionally fetch public files.")
    add_selection(assets)
    add_assets(assets)
    assets.add_argument("--fetch", action="store_true")
    report = subs.add_parser("summarize", help="Summarize your completed SAGAS runs.")
    add_selection(report)
    report.add_argument("--results-dir", required=True, type=abs_path)
    report.add_argument("--output-dir", required=True, type=abs_path)
    train = subs.add_parser("train", help="Invoke the original navigation training script.")
    train.add_argument("stage", choices=("tdr", "policy"))
    train.add_argument("--env", required=True)
    train.add_argument("--tdr-checkpoint", type=abs_path)
    add_run(train)
    build = subs.add_parser("build", help="Invoke the original graph builder.")
    build.add_argument("suite", choices=("navigation", "scene"))
    build.add_argument("--env")
    build.add_argument("--tdr-checkpoint", type=abs_path)
    add_run(build)
    gen = subs.add_parser("generate", help="Generate new tasks; never replace the fixed benchmark.")
    gen.add_argument("suite", choices=("navigation", "overlap"))
    gen.add_argument("--env", required=True)
    gen.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    gen.add_argument("--count", type=int)
    add_assets(gen)
    gen.add_argument("--output-dir", required=True, type=abs_path)
    gen.add_argument("--seed", type=int)
    gen.add_argument("--ltl2ba", type=abs_path)
    gen.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        manifest = read_json(ROOT / "assets/manifest.json")
        files = manifest["files"]
        if args.command in {"evaluate", "summarize", "assets"}:
            records = select(args.suite, args.env, args.difficulty, args.method, args.shard, args.all_cells)
            if args.command == "summarize":
                summarize(args.suite, records, args.results_dir, args.output_dir)
                return 0
            ids = [v for r in records for v in [*r["assets"].values(), *r["datasets"]]]
            if args.command == "assets":
                verify_assets(ids, files, args.assets_dir, args.data_dir,
                              fetch=args.fetch, hub=manifest["hub"])
                print(f"Verified {len(set(ids))} files.")
                return 0
            if len(records) != 1 and (args.task_id is not None or args.task_file or args.config or args.render):
                raise ValueError("Select a single preset for a custom task, config or render.")
            if args.render and args.task_id is None:
                raise ValueError("Render a single task with --task-id.")
            if args.render_every is not None and (not args.render or args.render_every < 1):
                raise ValueError("--render-every requires --render and a positive interval.")
            jobs = [suite_job(args, r, files) for r in records]
        else:
            job, ids = entry_job(args, files)
            jobs = [job]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.setdefault("WANDB_MODE", "disabled")
        environment.setdefault("MUJOCO_GL", "egl")
        environment["SAGAS_DATA_DIR"] = str(args.data_dir)
        binary = args.ltl2ba or environment.get("SAGAS_LTL2BA") or shutil.which("ltl2ba")
        if binary:
            environment["SAGAS_LTL2BA"] = str(abs_path(binary))
        if not args.dry_run:
            if args.command == "evaluate" and (not binary or not os.access(binary, os.X_OK)):
                raise ValueError("Install ltl2ba or pass an executable with --ltl2ba.")
            verify_assets(ids, files, args.assets_dir, args.data_dir)
            args.output_dir.mkdir(parents=True, exist_ok=False)
        for job in jobs:
            run_job(job, environment, args.dry_run)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
