#!/usr/bin/env python3
"""Build the fixed E5 state-region overlap suite before rollout evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Permit direct invocation from the isolated ``rebuttal_e5`` directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generate_semantic_ltl_tasks import (
    RelationCheck,
    SemanticTaskTemplate,
    load_env_free_points,
    task_from_template,
)


ENV_NAME = "antmaze-giant-stitch-v0"
BASE_SEED = 20260724
DEFAULT_TASKS_PER_GROUP = 8
Point = Tuple[float, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def e5_templates() -> Dict[str, SemanticTaskTemplate]:
    """Return the three fixed template families used for E5.

    The unused proposition ``d`` is a non-overlapping semantic distractor in
    every family.  It makes the task geometry more representative than an
    isolated two-region overlap without changing the intended LTL formula.
    """

    overlap = RelationCheck("overlap", labels=("a", "b"), min_witnesses=1)
    distractor = RelationCheck(
        "separated", labels=("a", "b"), other=("d",), min_distance=4.0
    )
    start_clear = RelationCheck("start_clear")
    return {
        "overlap_event": SemanticTaskTemplate(
            name="e5_overlap_event",
            complexity="medium",
            formula="<>(a && b)",
            labels=("a", "b", "d"),
            relations=(overlap, distractor, start_clear),
            feature_tags=("e5", "overlap", "multi_positive", "distractor"),
        ),
        "mixed_until": SemanticTaskTemplate(
            name="e5_mixed_until_overlap",
            complexity="hard",
            formula="(!c U (a && b))",
            labels=("a", "b", "c", "d"),
            relations=(
                overlap,
                RelationCheck(
                    "separated", labels=("a", "b"), other=("c",), min_distance=4.0
                ),
                distractor,
                start_clear,
            ),
            feature_tags=(
                "e5",
                "overlap",
                "multi_positive",
                "negative_literal",
                "until",
                "distractor",
            ),
        ),
        "recurrent_overlap": SemanticTaskTemplate(
            name="e5_recurrent_overlap",
            complexity="hard",
            formula="[]<>(a && b)",
            labels=("a", "b", "d"),
            relations=(overlap, distractor, start_clear),
            feature_tags=(
                "e5",
                "overlap",
                "multi_positive",
                "recurrence",
                "suffix",
                "distractor",
            ),
        ),
    }


def _bbox(geometry: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return the task-space bounding box of a supported region geometry."""

    kind = str(geometry["type"])
    if kind == "circle":
        x, y = geometry["center"]
        radius = float(geometry["radius"])
        return float(x) - radius, float(x) + radius, float(y) - radius, float(y) + radius
    if kind == "rect":
        return (
            float(geometry["xmin"]),
            float(geometry["xmax"]),
            float(geometry["ymin"]),
            float(geometry["ymax"]),
        )
    if kind == "polygon":
        xs = [float(vertex[0]) for vertex in geometry["vertices"]]
        ys = [float(vertex[1]) for vertex in geometry["vertices"]]
        return min(xs), max(xs), min(ys), max(ys)
    raise ValueError(f"Unsupported geometry type: {kind}")


def _bbox_disjoint(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    """Use disjoint bounding boxes as a conservative non-overlap certificate."""

    ax0, ax1, ay0, ay1 = _bbox(first)
    bx0, bx1, by0, by1 = _bbox(second)
    return ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0


def _distractors_are_disjoint(task: Dict[str, Any]) -> bool:
    """Ensure E5 distractors do not geometrically overlap the target pair."""

    regions = task["regions"]
    targets = (regions["a"], regions["b"])
    distractors = [regions[label] for label in ("c", "d") if label in regions]
    return all(
        _bbox_disjoint(target, distractor)
        for target in targets
        for distractor in distractors
    )


def _task_with_disjoint_distractors(
    *,
    template: SemanticTaskTemplate,
    task_id: int,
    seed: int,
    free_points: Sequence[Point],
    max_attempts: int,
    shape_mode: str,
    nonconvex_prob: float,
    region_size_scale: float,
) -> Dict[str, Any]:
    """Sample a task whose non-target regions are provably non-overlapping.

    The base generator validates overlap, free-space support, and clear starts.
    This wrapper adds the E5-specific distractor constraint without querying the
    graph or attempting a rollout.
    """

    for resample in range(max(1, int(max_attempts))):
        candidate = task_from_template(
            template,
            task_id=task_id,
            seed=int(seed) + resample * 1_000_003,
            free_points=free_points,
            max_attempts=max_attempts,
            shape_mode=shape_mode,
            nonconvex_prob=nonconvex_prob,
            size_scale=region_size_scale,
        )
        if _distractors_are_disjoint(candidate):
            candidate["metadata"]["e5_distractor_bbox_disjoint"] = True
            candidate["metadata"]["e5_distractor_resample"] = resample
            return candidate
    raise RuntimeError(
        f"Could not place bbox-disjoint E5 distractors for task {task_id} after "
        f"{max_attempts} generation attempts."
    )


def build_suite(
    *,
    seed: int,
    max_attempts: int,
    shape_mode: str,
    nonconvex_prob: float,
    region_size_scale: float,
    tasks_per_group: int,
) -> Dict[str, Any]:
    tasks_per_group = int(tasks_per_group)
    if tasks_per_group < 1:
        raise ValueError("tasks_per_group must be positive.")
    suite_version = f"e5_overlap{3 * tasks_per_group}_v1"
    free_points = load_env_free_points(
        env_name=ENV_NAME,
        seed=seed,
        render_size=800,
        env_region_type="mixed",
        max_regions=16,
        jitter=0.75,
        samples_per_cell=4,
    )
    templates = e5_templates()
    groups = [
        ("overlap_event", templates["overlap_event"]),
        ("mixed_until", templates["mixed_until"]),
        ("recurrent_overlap", templates["recurrent_overlap"]),
    ]
    tasks: List[Dict[str, Any]] = []
    for group_index, (group_name, template) in enumerate(groups):
        for offset in range(tasks_per_group):
            task_id = len(tasks)
            task = _task_with_disjoint_distractors(
                template=template,
                task_id=task_id,
                seed=seed,
                free_points=free_points,
                max_attempts=max_attempts,
                shape_mode=shape_mode,
                nonconvex_prob=nonconvex_prob,
                region_size_scale=region_size_scale,
            )
            task["metadata"].update(
                {
                    "e5_suite_version": suite_version,
                    "e5_group": group_name,
                    "e5_group_index": group_index,
                    "e5_group_offset": offset,
                    "e5_role": {
                        "a_b": "overlapping multi-positive target",
                        "c": "active forbidden proposition" if group_name == "mixed_until" else None,
                        "d": "non-overlapping distractor",
                    },
                }
            )
            tasks.append(task)
    return {
        "env_name": ENV_NAME,
        "generator": "rebuttal_e5/build_e5_overlap_suite.py",
        "e5_suite_version": suite_version,
        "seed": int(seed),
        "num_tasks": len(tasks),
        "group_counts": {name: tasks_per_group for name, _ in groups},
        "generation": {
            "max_task_attempts": int(max_attempts),
            "shape_mode": shape_mode,
            "nonconvex_prob": float(nonconvex_prob),
            "region_size_scale": float(region_size_scale),
            "render_size": 800,
            "env_region_type": "mixed",
            "free_point_jitter": 0.75,
            "samples_per_cell": 4,
        },
        "template_definitions": {
            name: asdict(template) for name, template in templates.items()
        },
        "semantic_task_schema": "state_region_with_predicate_specs_v1",
        "tasks": tasks,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--max-attempts", type=int, default=250)
    parser.add_argument("--shape-mode", default="mixed")
    parser.add_argument("--nonconvex-prob", type=float, default=0.35)
    parser.add_argument("--region-size-scale", type=float, default=1.0)
    parser.add_argument(
        "--tasks-per-group",
        type=int,
        default=DEFAULT_TASKS_PER_GROUP,
        help="Number of frozen tasks in each of the three template groups.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    manifest_output = Path(args.manifest_output)
    existing = [path for path in (output, manifest_output) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite an E5 task artifact: "
            + ", ".join(str(path) for path in existing)
        )
    payload = build_suite(
        seed=args.seed,
        max_attempts=args.max_attempts,
        shape_mode=args.shape_mode,
        nonconvex_prob=args.nonconvex_prob,
        region_size_scale=args.region_size_scale,
        tasks_per_group=args.tasks_per_group,
    )
    write_json(output, payload)
    manifest = {
        "suite_version": payload["e5_suite_version"],
        "task_payload": str(output),
        "task_payload_sha256": sha256_file(output),
        "env_name": ENV_NAME,
        "seed": int(args.seed),
        "num_tasks": int(payload["num_tasks"]),
        "group_counts": payload["group_counts"],
        "generation": payload["generation"],
        "frozen_before_rollout": True,
    }
    write_json(manifest_output, manifest)
    print(json.dumps({"task_output": str(output), "manifest": manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
