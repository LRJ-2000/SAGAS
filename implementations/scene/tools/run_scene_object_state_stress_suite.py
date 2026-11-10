"""Run the fixed M6.2 scene object-state stress suite.

The suite is a small diagnostic validation set for state-based manipulation
tasks.  It reuses the M6.1 real-rollout wrapper and aggregates per-task
planning, runtime, oracle, and grounding diagnostics without changing the
planner or executor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_esagas import jsonable
from tools.run_scene_object_state_rollout_smoke import (
    DEFAULT_MANIFEST,
    run_scene_object_state_rollout_smoke,
)


DEFAULT_TASK_FILE = Path(
    "ltl_tasks/validation/scene-play_m6_2_object_state_stress_suite.json"
)
DEFAULT_RESULT_ROOT = Path("artifacts/m6_2_scene_object_state_stress_suite")
DEFAULT_ENV_MAX_EPISODE_STEPS = 8000


def _load_task_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = list(payload.get("tasks", []))
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return payload


def _parse_task_ids(spec: str, *, num_tasks: int) -> List[int]:
    if spec.strip().lower() in {"all", "*"}:
        return list(range(num_tasks))
    ids: List[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending task-id range: {part}")
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))
    unique_ids = []
    seen = set()
    for task_id in ids:
        if task_id < 0 or task_id >= num_tasks:
            raise ValueError(
                f"Task id {task_id} is outside valid range [0, {num_tasks - 1}]"
            )
        if task_id not in seen:
            unique_ids.append(task_id)
            seen.add(task_id)
    return unique_ids


def _failure_class(summary: Dict[str, Any]) -> str:
    if summary.get("status") == "error":
        return "runner_error"
    if summary.get("runtime_success") and summary.get("oracle_success"):
        return "success"
    if summary.get("semantic_grounding_error"):
        return "semantic_grounding_error"
    if not summary.get("planner_found_plan"):
        unsupported = summary.get("unsupported_guard_terms") or []
        if unsupported:
            return "unsupported_guard_terms"
        return "planner_no_plan"
    if not summary.get("runtime_success"):
        return str(summary.get("runtime_failure_reason") or "runtime_failure")
    if not summary.get("oracle_success"):
        return str(summary.get("oracle_failure_reason") or "oracle_failure")
    return "success"


def _compact_task_summary(
    *,
    task: Dict[str, Any],
    task_id: int,
    seed: int,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    aggregate = report.get("aggregate", {})
    result = {}
    results = aggregate.get("results", []) if isinstance(aggregate, dict) else []
    if results:
        result = dict(results[0])
    mismatch = result.get("evaluator_oracle_mismatch")
    if not isinstance(mismatch, dict):
        mismatch = {}
    return {
        "task_id": int(task_id),
        "seed": int(seed),
        "formula": task.get("formula_e", task.get("formula")),
        "template": task.get("metadata", {}).get("template"),
        "semantic_features": list(task.get("metadata", {}).get("semantic_features", [])),
        "status": report.get("status", "ok"),
        "failure_class": _failure_class(report),
        "planner_found_plan": bool(report.get("planner_found_plan")),
        "runtime_success": bool(report.get("runtime_success")),
        "runtime_failure_reason": report.get("runtime_failure_reason"),
        "oracle_success": bool(report.get("oracle_success")),
        "oracle_failure_reason": report.get("oracle_failure_reason"),
        "unsupported_guard_terms": list(report.get("unsupported_guard_terms", [])),
        "semantic_grounding_error": bool(report.get("semantic_grounding_error")),
        "reset_task_info_used": bool(report.get("reset_task_info_used")),
        "actual_initial_oracle_labels": list(
            report.get("actual_initial_oracle_labels", [])
        ),
        "generation_initial_oracle_labels": report.get("generation_initial_oracle_labels"),
        "initial_oracle_label_mismatch": report.get("initial_oracle_label_mismatch"),
        "negative_uncertainty_blocks": int(
            result.get("negative_uncertainty_blocks") or 0
        ),
        "planning_timing_seconds": dict(
            report.get("planning_timing_seconds", {})
        ),
        "planning_timing_contract": report.get("planning_timing_contract"),
        "evaluator_oracle_mismatch_rate": mismatch.get("mismatch_rate"),
        "num_actions": report.get("num_actions"),
        "num_trace_states": report.get("num_trace_states"),
        "prefix_completed": report.get("prefix_completed"),
        "suffix_cycles_completed": report.get("suffix_cycles_completed"),
        "suffix_cycles_required": report.get("suffix_cycles_required"),
        "summary_path": report.get("summary_path"),
        "diagnostics_path": report.get("diagnostics_path"),
        "trace_path": report.get("trace_path"),
        "transition_results_path": report.get("transition_results_path"),
        "oracle_verdict_path": report.get("oracle_verdict_path"),
        "render_video_path": report.get("render_video_path"),
    }


def _mean(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def _percentile(values: Iterable[Any], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return None
    position = (len(xs) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return xs[lower]
    weight = position - lower
    return xs[lower] * (1.0 - weight) + xs[upper] * weight


def _timing_summary(
    task_reports: Sequence[Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    values = [
        item.get("planning_timing_seconds", {}).get(key)
        for item in task_reports
    ]
    finite_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return {
        "count": len(finite_values),
        "median": _percentile(finite_values, 0.5),
        "p95": _percentile(finite_values, 0.95),
        "max": max(finite_values) if finite_values else None,
    }


def run_stress_suite(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    target_root: Path = Path("."),
    task_path: Path = DEFAULT_TASK_FILE,
    keygraph_dir: Optional[Path] = None,
    task_ids: Sequence[int],
    seed_override: Optional[int] = None,
    gpu: int = 0,
    max_members_per_node: int = 4,
    max_anchor_support: int = 5000,
    suffix_cycles: int = 1,
    local_step_budget: int = 8000,
    reach_threshold: Optional[float] = None,
    action_clip: float = 1.0,
    render_video: bool = True,
    render_every: int = 5,
    render_fps: int = 30,
    env_max_episode_steps: Optional[int] = DEFAULT_ENV_MAX_EPISODE_STEPS,
    semantic_repulsion: Optional[bool] = None,
    result_root: Path = DEFAULT_RESULT_ROOT,
    resume: bool = False,
    stop_after_successes: Optional[int] = None,
) -> Dict[str, Any]:
    payload = _load_task_payload(task_path)
    tasks = list(payload["tasks"])
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)

    if stop_after_successes is not None and int(stop_after_successes) <= 0:
        raise ValueError("stop_after_successes must be positive when provided")

    requested_task_ids = [int(task_id) for task_id in task_ids]
    executed_task_ids: List[int] = []
    task_reports: List[Dict[str, Any]] = []
    for task_id in task_ids:
        task = dict(tasks[int(task_id)])
        seed = int(seed_override) if seed_override is not None else int(task.get("seed", 0))
        try:
            report = run_scene_object_state_rollout_smoke(
                manifest_path=Path(manifest_path),
                target_root=Path(target_root),
                task_path=Path(task_path),
                keygraph_dir=Path(keygraph_dir) if keygraph_dir is not None else None,
                task_id=int(task_id),
                seed=seed,
                gpu=int(gpu),
                max_members_per_node=int(max_members_per_node),
                max_anchor_support=int(max_anchor_support),
                suffix_cycles=int(suffix_cycles),
                local_step_budget=int(local_step_budget),
                reach_threshold=reach_threshold,
                action_clip=float(action_clip),
                render_video=bool(render_video),
                render_every=int(render_every),
                render_fps=int(render_fps),
                env_max_episode_steps=env_max_episode_steps,
                semantic_repulsion=semantic_repulsion,
                result_root=result_root,
                resume=bool(resume),
            )
            report["status"] = "ok"
        except Exception as exc:  # pragma: no cover - exercised in remote runs.
            report = {
                "status": "error",
                "task_id": int(task_id),
                "formula": task.get("formula_e", task.get("formula")),
                "planner_found_plan": False,
                "runtime_success": False,
                "oracle_success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        compact = _compact_task_summary(
            task=task,
            task_id=int(task_id),
            seed=seed,
            report=report,
        )
        task_reports.append(compact)
        executed_task_ids.append(int(task_id))
        if stop_after_successes is not None:
            successful = sum(
                bool(item["runtime_success"]) and bool(item["oracle_success"])
                for item in task_reports
            )
            if successful >= int(stop_after_successes):
                break

    num_tasks = len(task_reports)
    runtime_success_count = sum(1 for item in task_reports if item["runtime_success"])
    oracle_success_count = sum(1 for item in task_reports if item["oracle_success"])
    unsupported_guard_terms = sorted(
        {
            str(term)
            for item in task_reports
            for term in item.get("unsupported_guard_terms", [])
        }
    )
    failure_class_counts = Counter(item["failure_class"] for item in task_reports)
    summary = {
        "mode": "m6_2_scene_object_state_stress_suite",
        "manifest_path": str(manifest_path),
        "task_path": str(task_path),
        "result_root": str(result_root),
        "task_ids": executed_task_ids,
        "requested_task_ids": requested_task_ids,
        "stopping_rule": (
            {
                "type": "fixed_order_success_quota",
                "success_quota": int(stop_after_successes),
                "success_definition": (
                    "runtime_success AND independent_oracle_success"
                ),
            }
            if stop_after_successes is not None
            else {"type": "complete_requested_task_list"}
        ),
        "num_tasks": int(num_tasks),
        "runtime_success_count": int(runtime_success_count),
        "runtime_success_rate": (
            runtime_success_count / float(num_tasks) if num_tasks else 0.0
        ),
        "oracle_success_count": int(oracle_success_count),
        "oracle_success_rate": (
            oracle_success_count / float(num_tasks) if num_tasks else 0.0
        ),
        "failure_class_counts": dict(sorted(failure_class_counts.items())),
        "unsupported_guard_terms": unsupported_guard_terms,
        "semantic_grounding_error_count": int(
            sum(1 for item in task_reports if item["semantic_grounding_error"])
        ),
        "total_negative_uncertainty_blocks": int(
            sum(int(item.get("negative_uncertainty_blocks") or 0) for item in task_reports)
        ),
        "mean_num_actions": _mean(item.get("num_actions") for item in task_reports),
        "mean_success_num_actions": _mean(
            item.get("num_actions") for item in task_reports if item["oracle_success"]
        ),
        "mean_evaluator_oracle_mismatch_rate": _mean(
            item.get("evaluator_oracle_mismatch_rate") for item in task_reports
        ),
        "planning_timing_seconds": {
            key: _timing_summary(task_reports, key)
            for key in (
                "evaluator_setup",
                "semantic_augmentation",
                "product_search",
                "total",
            )
        },
        "planning_timing_contract": (
            next(
                (
                    item.get("planning_timing_contract")
                    for item in task_reports
                    if item.get("planning_timing_contract")
                ),
                None,
            )
        ),
        "tasks": task_reports,
    }
    summary_path = result_root / "stress_suite_summary.json"
    legacy_summary_path = result_root / "m6_2_stress_suite_summary.json"
    summary["summary_path"] = str(summary_path)
    summary["legacy_summary_path"] = str(legacy_summary_path)
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy_summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--target-root", default=".")
    parser.add_argument("--ltl-task-path", default=str(DEFAULT_TASK_FILE))
    parser.add_argument("--keygraph-dir", default=None)
    parser.add_argument(
        "--task-ids",
        default="all",
        help="Task ids to run, e.g. 'all', '0,2,5', or '0-4'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override. By default each fixed task uses its saved seed.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-members-per-node", type=int, default=4)
    parser.add_argument("--max-anchor-support", type=int, default=5000)
    parser.add_argument("--suffix-cycles", type=int, default=1)
    parser.add_argument("--local-step-budget", type=int, default=8000)
    parser.add_argument("--reach-threshold", type=float, default=None)
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument(
        "--env-max-episode-steps",
        type=int,
        default=DEFAULT_ENV_MAX_EPISODE_STEPS,
    )
    parser.add_argument(
        "--semantic-repulsion",
        choices=["auto", "on", "off"],
        default="auto",
        help="Guard-aware steering mode. `auto` disables it for scene manipulation.",
    )
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--no-render-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after-successes",
        type=int,
        default=None,
        help=(
            "Run the fixed task order until this many runtime-and-oracle "
            "successes are observed. All attempted tasks remain in the summary."
        ),
    )
    parser.add_argument("--fail-on-task-failure", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = _load_task_payload(Path(args.ltl_task_path))
    task_ids = _parse_task_ids(args.task_ids, num_tasks=len(payload["tasks"]))
    report = run_stress_suite(
        manifest_path=Path(args.manifest),
        target_root=Path(args.target_root),
        task_path=Path(args.ltl_task_path),
        keygraph_dir=Path(args.keygraph_dir) if args.keygraph_dir else None,
        task_ids=task_ids,
        seed_override=args.seed,
        gpu=args.gpu,
        max_members_per_node=args.max_members_per_node,
        max_anchor_support=args.max_anchor_support,
        suffix_cycles=args.suffix_cycles,
        local_step_budget=args.local_step_budget,
        reach_threshold=args.reach_threshold,
        action_clip=args.action_clip,
        render_video=not args.no_render_video,
        render_every=args.render_every,
        render_fps=args.render_fps,
        env_max_episode_steps=args.env_max_episode_steps,
        semantic_repulsion=(
            None
            if args.semantic_repulsion == "auto"
            else args.semantic_repulsion == "on"
        ),
        result_root=Path(args.result_root),
        resume=args.resume,
        stop_after_successes=args.stop_after_successes,
    )
    if args.json:
        print(json.dumps(jsonable(report), sort_keys=True))
    else:
        print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    if args.fail_on_task_failure and report.get("oracle_success_count") != report.get("num_tasks"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
