"""Run the M6.1 scene object-state real rollout smoke.

This is a thin, scene-specific wrapper around ``evaluate_esagas.run_real_task_file``.
It fixes the M6.1 defaults without changing the planner, executor, verifier,
or benchmark runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_esagas import jsonable, run_real_task_file


DEFAULT_MANIFEST = Path("assets/manifests/scene-play-v0_assets.json")
DEFAULT_TASK_FILE = Path(
    "ltl_tasks/validation/scene-play_m6_object_state_planning_smoke.json"
)
DEFAULT_RESULT_ROOT = Path("artifacts/m6_1_scene_object_state_rollout_smoke")
DEFAULT_SCENE_REACH_THRESHOLD = 48.0


def _resolve_reach_threshold(manifest_path: Path, reach_threshold: Optional[float]) -> float:
    if reach_threshold is not None:
        return float(reach_threshold)
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return float(
            payload.get("agent_config_overrides", {}).get(
                "way_steps", DEFAULT_SCENE_REACH_THRESHOLD
            )
        )
    except (OSError, TypeError, ValueError):
        return DEFAULT_SCENE_REACH_THRESHOLD


def _first_result(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    results = aggregate.get("results", [])
    if not results:
        return {}
    return dict(results[0])


def run_scene_object_state_rollout_smoke(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    target_root: Path = Path("."),
    task_path: Path = DEFAULT_TASK_FILE,
    keygraph_dir: Optional[Path] = None,
    task_id: int = 0,
    seed: int = 0,
    gpu: int = 0,
    max_members_per_node: int = 4,
    max_anchor_support: int = 5000,
    suffix_cycles: int = 1,
    local_step_budget: int = 8000,
    reach_threshold: Optional[float] = None,
    anchor_reached_threshold: Optional[float] = None,
    action_clip: float = 1.0,
    render_video: bool = True,
    render_every: int = 5,
    render_fps: int = 30,
    env_max_episode_steps: Optional[int] = None,
    semantic_repulsion: Optional[bool] = None,
    result_root: Path = DEFAULT_RESULT_ROOT,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run one fixed scene object-state task and return compact diagnostics."""

    task_id = int(task_id)
    task_result_root = Path(result_root) / f"task{task_id:02d}"
    video_path = task_result_root / "rollout.mp4" if render_video else None
    resolved_reach_threshold = _resolve_reach_threshold(Path(manifest_path), reach_threshold)
    aggregate = run_real_task_file(
        manifest_path=Path(manifest_path),
        target_root=Path(target_root),
        task_path=Path(task_path),
        keygraph_dir=Path(keygraph_dir) if keygraph_dir is not None else None,
        env_name="scene-play-v0",
        task_id=task_id,
        task_limit=1,
        seed=int(seed),
        gpu=int(gpu),
        action_dim=5,
        max_members_per_node=int(max_members_per_node),
        max_anchor_support=int(max_anchor_support),
        suffix_cycles=int(suffix_cycles),
        local_step_budget=int(local_step_budget),
        reach_threshold=float(resolved_reach_threshold),
        anchor_reached_threshold=(
            None
            if anchor_reached_threshold is None
            else float(anchor_reached_threshold)
        ),
        action_clip=float(action_clip),
        render_video=video_path,
        render_every=int(render_every),
        render_fps=int(render_fps),
        env_size=None,
        env_max_episode_steps=(
            None if env_max_episode_steps is None else int(env_max_episode_steps)
        ),
        semantic_repulsion_enable=semantic_repulsion,
        result_dir=task_result_root,
        resume=bool(resume),
        semantic_evaluator="object_state",
    )
    result = _first_result(aggregate)
    summary = {
        "mode": "scene_object_state_rollout_smoke",
        "manifest_path": str(manifest_path),
        "task_path": str(task_path),
        "task_id": task_id,
        "task_index": result.get("task_index"),
        "formula": result.get("formula"),
        "result_root": str(task_result_root),
        "aggregate_diagnostics_path": aggregate.get("aggregate_diagnostics_path"),
        "diagnostics_path": result.get("diagnostics_path"),
        "trace_path": result.get("trace_path"),
        "transition_results_path": result.get("transition_results_path"),
        "oracle_verdict_path": result.get("oracle_verdict_path"),
        "render_video_path": result.get("render_video_path"),
        "planner_found_plan": bool(result.get("planner_found_plan")),
        "runtime_success": bool(result.get("runtime_finite_lasso_success")),
        "runtime_failure_reason": result.get("runtime_failure_reason"),
        "semantic_repulsion_enable": result.get("semantic_repulsion_enable"),
        "oracle_success": bool(result.get("oracle_success")),
        "oracle_failure_reason": result.get("oracle_failure_reason"),
        "unsupported_guard_terms": list(result.get("unsupported_guard_terms", [])),
        "num_actions": result.get("num_actions"),
        "num_trace_states": result.get("num_trace_states"),
        "prefix_completed": result.get("prefix_completed"),
        "suffix_cycles_completed": result.get("suffix_cycles_completed"),
        "suffix_cycles_required": result.get("suffix_cycles_required"),
        "reset_task_info_used": bool(result.get("reset_task_info_used")),
        "actual_initial_oracle_labels": list(
            result.get("actual_initial_oracle_labels", [])
        ),
        "generation_initial_oracle_labels": result.get("generation_initial_oracle_labels"),
        "initial_oracle_label_mismatch": result.get("initial_oracle_label_mismatch"),
        "semantic_grounding_error": bool(result.get("semantic_grounding_error")),
        "planning_timing_seconds": dict(
            result.get("planning_timing_seconds", {})
        ),
        "planning_timing_contract": result.get("planning_timing_contract"),
        "carrier_state_source_counts": result.get("carrier_state_source_counts", {}),
        "aggregate": aggregate,
    }
    summary_path = task_result_root / "m6_1_smoke_summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--target-root", default=".")
    parser.add_argument("--ltl-task-path", default=str(DEFAULT_TASK_FILE))
    parser.add_argument("--keygraph-dir", default=None)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-members-per-node", type=int, default=4)
    parser.add_argument("--max-anchor-support", type=int, default=5000)
    parser.add_argument("--suffix-cycles", type=int, default=1)
    parser.add_argument("--local-step-budget", type=int, default=8000)
    parser.add_argument(
        "--reach-threshold",
        type=float,
        default=None,
        help=(
            "Latent waypoint reach threshold. Defaults to manifest "
            "agent_config_overrides.way_steps, matching GAS evaluation."
        ),
    )
    parser.add_argument(
        "--anchor-reached-threshold",
        type=float,
        default=None,
        help=(
            "Positive semantic checkpoint reach radius in latent space. "
            "Defaults to the resolved waypoint reach threshold."
        ),
    )
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--env-max-episode-steps", type=int, default=None)
    parser.add_argument(
        "--semantic-repulsion",
        choices=["auto", "on", "off"],
        default="auto",
        help="Guard-aware steering mode. `auto` disables it for scene manipulation.",
    )
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--no-render-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = run_scene_object_state_rollout_smoke(
        manifest_path=Path(args.manifest),
        target_root=Path(args.target_root),
        task_path=Path(args.ltl_task_path),
        keygraph_dir=Path(args.keygraph_dir) if args.keygraph_dir else None,
        task_id=args.task_id,
        seed=args.seed,
        gpu=args.gpu,
        max_members_per_node=args.max_members_per_node,
        max_anchor_support=args.max_anchor_support,
        suffix_cycles=args.suffix_cycles,
        local_step_budget=args.local_step_budget,
        reach_threshold=args.reach_threshold,
        anchor_reached_threshold=args.anchor_reached_threshold,
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
    )
    if args.json:
        print(json.dumps(jsonable(report), sort_keys=True))
    else:
        print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
