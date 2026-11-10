"""Rebuild a KeyGraph in a separate output directory.

This helper consumes the asset manifest produced by `prepare_esagas_assets.py`
and invokes the existing GAS-style `construct_graph.py` entrypoint with copied
TDR checkpoints.  It is intentionally conservative: graph outputs are written
only under the requested target root, and an existing run-group output is not
reused unless explicitly allowed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from M_utils.agent_config_utils import (
    default_agent_config_overrides,
    default_te_threshold,
)
from prepare_esagas_assets import env_stem


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_manifest(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Asset manifest not found: {path}")
    return json.loads(path.read_text())


def resolve_manifest_path(raw_path: str, manifest_target_root: Path, target_root: Path) -> Path:
    """Resolve a manifest path after copying the workspace to another machine."""
    raw = Path(raw_path)
    if raw.is_absolute():
        try:
            rel = raw.relative_to(manifest_target_root)
            relocated = target_root / rel
            if relocated.exists():
                return relocated.resolve()
        except ValueError:
            pass
        return raw.resolve()
    return (target_root / raw).resolve()


def resolve_tdr_path(manifest: Dict, target_root: Path) -> Path:
    tdr_path = manifest.get("tdr_path")
    if not tdr_path:
        raise KeyError("Asset manifest does not contain `tdr_path`.")
    manifest_target_root = Path(manifest.get("target_root") or target_root)
    resolved = resolve_manifest_path(str(tdr_path), manifest_target_root, target_root)
    if not resolved.is_file():
        raise FileNotFoundError(f"Copied TDR checkpoint is missing: {resolved}")
    if not _is_relative_to(resolved, target_root):
        raise ValueError(
            "Refusing to use a TDR checkpoint outside the target root. "
            f"tdr_path={resolved}, target_root={target_root.resolve()}"
        )
    return resolved


def default_run_group(env_name: str) -> str:
    return f"JournalKeyGraph_{_safe_name(env_stem(env_name))}"


def existing_graph_dirs(output_root: Path, run_group: str, env_name: str) -> List[Path]:
    run_group_dir = output_root / run_group
    if not run_group_dir.exists():
        return []
    return sorted(
        p for p in run_group_dir.iterdir()
        if p.is_dir() and p.name.startswith(f"{env_name}_sd")
    )


def build_rebuild_command(
    *,
    manifest_path: Path,
    target_root: Path,
    env_name: Optional[str] = None,
    output_root: Optional[Path] = None,
    run_group: Optional[str] = None,
    seed: int = 0,
    gpu: int = 0,
    te_threshold: Optional[float] = None,
    component_merge_strategy: str = "exact",
    python_executable: Optional[str] = None,
) -> Dict:
    manifest_path = manifest_path.resolve()
    target_root = target_root.resolve()
    manifest = load_manifest(manifest_path)
    env_name = env_name or manifest.get("env_name")
    if not env_name:
        raise KeyError("Environment name must be provided or stored in the asset manifest.")
    resolved_te_threshold = (
        default_te_threshold(env_name) if te_threshold is None else float(te_threshold)
    )
    agent_config_overrides = dict(default_agent_config_overrides(env_name))
    agent_config_overrides.update(dict(manifest.get("agent_config_overrides") or {}))
    run_group = run_group or default_run_group(env_name)
    output_root = (output_root or (target_root / "assets" / "rebuilt_keygraphs")).resolve()
    if not _is_relative_to(output_root, target_root):
        raise ValueError(
            f"Refusing to write graph output outside target root: {output_root}"
        )
    tdr_path = resolve_tdr_path(manifest, target_root)
    construct_graph = (target_root / "construct_graph.py").resolve()
    if not construct_graph.is_file():
        raise FileNotFoundError(f"construct_graph.py not found under target root: {construct_graph}")
    python_executable = python_executable or sys.executable
    command = [
        python_executable,
        str(construct_graph),
        "--env_name",
        str(env_name),
        "--tdr_path",
        str(tdr_path),
        "--save_graph_dir",
        str(output_root),
        "--run_group",
        str(run_group),
        "--seed",
        str(int(seed)),
        "--gpu",
        str(int(gpu)),
        "--te_threshold",
        str(float(resolved_te_threshold)),
        "--component_merge_strategy",
        str(component_merge_strategy),
    ]
    for key, value in sorted(agent_config_overrides.items()):
        command.extend([f"--agent_config.{key}", str(value)])
    return {
        "manifest_path": str(manifest_path),
        "target_root": str(target_root),
        "env_name": str(env_name),
        "tdr_path": str(tdr_path),
        "output_root": str(output_root),
        "run_group": str(run_group),
        "agent_config_overrides": agent_config_overrides,
        "te_threshold": float(resolved_te_threshold),
        "component_merge_strategy": str(component_merge_strategy),
        "existing_graph_dirs": [str(p) for p in existing_graph_dirs(output_root, run_group, env_name)],
        "command": command,
        "cwd": str(target_root),
    }


def write_rebuild_record(target_root: Path, env_name: str, record: Dict) -> Path:
    manifest_dir = target_root / "assets" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{env_stem(env_name)}_keygraph_rebuild_command.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def rebuild_keygraph(
    *,
    manifest_path: Path,
    target_root: Path,
    env_name: Optional[str] = None,
    output_root: Optional[Path] = None,
    run_group: Optional[str] = None,
    seed: int = 0,
    gpu: int = 0,
    te_threshold: Optional[float] = None,
    component_merge_strategy: str = "exact",
    python_executable: Optional[str] = None,
    allow_existing: bool = False,
    execute: bool = False,
) -> Dict:
    plan = build_rebuild_command(
        manifest_path=manifest_path,
        target_root=target_root,
        env_name=env_name,
        output_root=output_root,
        run_group=run_group,
        seed=seed,
        gpu=gpu,
        te_threshold=te_threshold,
        component_merge_strategy=component_merge_strategy,
        python_executable=python_executable,
    )
    if plan["existing_graph_dirs"] and not allow_existing:
        raise FileExistsError(
            "Existing KeyGraph output found for this run group; refusing to append. "
            "Use a new --run-group or pass --allow-existing."
        )

    Path(plan["output_root"]).mkdir(parents=True, exist_ok=True)
    record = dict(plan)
    record["executed"] = bool(execute)
    record["status"] = "planned"
    if execute:
        completed = subprocess.run(
            plan["command"],
            cwd=plan["cwd"],
            check=False,
        )
        record["returncode"] = int(completed.returncode)
        record["status"] = "completed" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            write_rebuild_record(target_root.resolve(), plan["env_name"], record)
            raise subprocess.CalledProcessError(completed.returncode, plan["command"])
    record_path = write_rebuild_record(target_root.resolve(), plan["env_name"], record)
    record["record_path"] = str(record_path)
    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-root", default=".")
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-group", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--te-threshold", type=float, default=None)
    parser.add_argument(
        "--component-merge-strategy",
        choices=["exact", "fast"],
        default="exact",
        help="Use `fast` for smoke-test graph rebuilds; default preserves exact GAS-style merging.",
    )
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--allow-existing", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    record = rebuild_keygraph(
        manifest_path=Path(args.manifest),
        target_root=Path(args.target_root),
        env_name=args.env_name,
        output_root=Path(args.output_root) if args.output_root else None,
        run_group=args.run_group,
        seed=args.seed,
        gpu=args.gpu,
        te_threshold=args.te_threshold,
        component_merge_strategy=args.component_merge_strategy,
        python_executable=args.python_executable,
        allow_existing=args.allow_existing,
        execute=args.execute,
    )
    if args.json:
        print(json.dumps(record, sort_keys=True))
    else:
        verb = "Executed" if args.execute else "Prepared"
        print(f"{verb} KeyGraph rebuild command for {record['env_name']}")
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
