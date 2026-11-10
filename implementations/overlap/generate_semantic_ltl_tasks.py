# -*- coding: utf-8 -*-
"""Generate small evaluator-conditioned LTL task files.

This generator is separate from the conference-style task generator. It keeps
the saved-task schema consumed by ``evaluate_esagas.py --mode real_task_file``,
while producing tasks that exercise the semantic planning layer:
overlapping predicates, mixed geometric shapes, multi-positive guards,
disjunctions, negative literals, and recurrent suffix formulas.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class PredicateSpec:
    """Task-time predicate description used internally by this generator."""

    name: str
    evaluator: str
    geometry: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationCheck:
    """Lightweight geometric relation required by a validation template."""

    kind: str
    labels: Tuple[str, ...] = ()
    other: Tuple[str, ...] = ()
    min_distance: float = 0.0
    min_witnesses: int = 1


@dataclass(frozen=True)
class SemanticTaskTemplate:
    name: str
    complexity: str
    formula: str
    labels: Tuple[str, ...]
    relations: Tuple[RelationCheck, ...]
    feature_tags: Tuple[str, ...]


COLORS = [
    [0.15, 0.45, 0.95, 0.55],
    [0.05, 0.70, 0.35, 0.55],
    [0.95, 0.25, 0.20, 0.50],
    [0.90, 0.70, 0.10, 0.50],
    [0.55, 0.35, 0.95, 0.50],
    [0.10, 0.75, 0.85, 0.50],
]


TEMPLATES: Tuple[SemanticTaskTemplate, ...] = (
    SemanticTaskTemplate(
        name="overlap_multi_positive",
        complexity="medium",
        formula="<>(a && b)",
        labels=("a", "b"),
        relations=(
            RelationCheck("overlap", labels=("a", "b"), min_witnesses=1),
            RelationCheck("start_clear"),
            RelationCheck("not_all_same_place", labels=("a", "b"), min_distance=0.8),
        ),
        feature_tags=("overlap", "multi_positive"),
    ),
    SemanticTaskTemplate(
        name="disjunction_choice",
        complexity="medium",
        formula="<>(a || b)",
        labels=("a", "b"),
        relations=(
            RelationCheck("separated", labels=("a",), other=("b",), min_distance=4.0),
            RelationCheck("start_clear"),
        ),
        feature_tags=("disjunction",),
    ),
    SemanticTaskTemplate(
        name="negative_until",
        complexity="medium",
        formula="(!c U a)",
        labels=("a", "c"),
        relations=(
            RelationCheck("separated", labels=("a",), other=("c",), min_distance=4.0),
            RelationCheck("start_clear"),
        ),
        feature_tags=("negative_literal", "until"),
    ),
    SemanticTaskTemplate(
        name="sequence",
        complexity="medium",
        formula="<>(a && <> b)",
        labels=("a", "b"),
        relations=(
            RelationCheck("separated", labels=("a",), other=("b",), min_distance=4.0),
            RelationCheck("start_clear"),
        ),
        feature_tags=("sequence",),
    ),
    SemanticTaskTemplate(
        name="overlap_or_choice",
        complexity="medium",
        formula="<>((a && b) || c)",
        labels=("a", "b", "c"),
        relations=(
            RelationCheck("overlap", labels=("a", "b"), min_witnesses=1),
            RelationCheck("separated", labels=("a", "b"), other=("c",), min_distance=4.0),
            RelationCheck("start_clear"),
        ),
        feature_tags=("overlap", "multi_positive", "disjunction"),
    ),
    SemanticTaskTemplate(
        name="until_multi_positive",
        complexity="hard",
        formula="(!c U (a && b))",
        labels=("a", "b", "c"),
        relations=(
            RelationCheck("overlap", labels=("a", "b"), min_witnesses=1),
            RelationCheck("separated", labels=("a", "b"), other=("c",), min_distance=4.0),
            RelationCheck("start_clear"),
        ),
        feature_tags=("overlap", "multi_positive", "negative_literal", "until"),
    ),
    SemanticTaskTemplate(
        name="recurrent_two_region_patrol",
        complexity="hard",
        formula="[]<>(a) && []<>(b)",
        labels=("a", "b"),
        relations=(
            RelationCheck("separated", labels=("a",), other=("b",), min_distance=5.0),
            RelationCheck("start_clear"),
        ),
        feature_tags=("recurrence", "suffix"),
    ),
    SemanticTaskTemplate(
        name="recurrent_overlap_patrol",
        complexity="hard",
        formula="[]<>(a && b)",
        labels=("a", "b"),
        relations=(
            RelationCheck("overlap", labels=("a", "b"), min_witnesses=1),
            RelationCheck("start_clear"),
        ),
        feature_tags=("recurrence", "suffix", "overlap", "multi_positive"),
    ),
    SemanticTaskTemplate(
        name="sequence_overlap_with_unlock",
        complexity="hard",
        formula="<>(a && <> (b && c)) && (!d U a)",
        labels=("a", "b", "c", "d"),
        relations=(
            RelationCheck("overlap", labels=("b", "c"), min_witnesses=1),
            RelationCheck("separated", labels=("a",), other=("b", "c"), min_distance=4.0),
            RelationCheck("separated", labels=("d",), other=("a",), min_distance=4.0),
            RelationCheck("start_clear"),
        ),
        feature_tags=("sequence", "overlap", "multi_positive", "negative_literal", "until"),
    ),
    SemanticTaskTemplate(
        name="choice_with_until_avoid",
        complexity="hard",
        formula="(!d U ((a && b) || c))",
        labels=("a", "b", "c", "d"),
        relations=(
            RelationCheck("overlap", labels=("a", "b"), min_witnesses=1),
            RelationCheck("separated", labels=("a", "b"), other=("c",), min_distance=4.0),
            RelationCheck("separated", labels=("d",), other=("a", "b", "c"), min_distance=4.0),
            RelationCheck("start_clear"),
        ),
        feature_tags=("overlap", "multi_positive", "disjunction", "negative_literal", "until"),
    ),
)


def templates_for(complexity: str) -> List[SemanticTaskTemplate]:
    selected = [t for t in TEMPLATES if t.complexity == complexity]
    if not selected:
        raise ValueError(f"Unsupported complexity: {complexity}")
    return selected


def _distance(p: Point, q: Point) -> float:
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def _bbox(geometry: Dict[str, Any]) -> Tuple[float, float, float, float]:
    gtype = geometry["type"]
    if gtype == "circle":
        cx, cy = geometry["center"]
        r = float(geometry["radius"])
        return float(cx - r), float(cx + r), float(cy - r), float(cy + r)
    if gtype == "rect":
        return (
            float(geometry["xmin"]),
            float(geometry["xmax"]),
            float(geometry["ymin"]),
            float(geometry["ymax"]),
        )
    if gtype == "polygon":
        xs = [float(v[0]) for v in geometry["vertices"]]
        ys = [float(v[1]) for v in geometry["vertices"]]
        return min(xs), max(xs), min(ys), max(ys)
    raise ValueError(f"Unsupported geometry type: {gtype}")


def _center(geometry: Dict[str, Any]) -> Point:
    if geometry["type"] == "circle":
        return float(geometry["center"][0]), float(geometry["center"][1])
    xmin, xmax, ymin, ymax = _bbox(geometry)
    return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0


def _point_on_segment(px, py, ax, ay, bx, by, eps=1e-9) -> bool:
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= eps


def _point_in_polygon(point: Point, vertices: Sequence[Sequence[float]]) -> bool:
    x, y = point
    verts = [(float(v[0]), float(v[1])) for v in vertices]
    inside = False
    j = len(verts) - 1
    for i in range(len(verts)):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if _point_on_segment(x, y, xi, yi, xj, yj):
            return True
        intersects = ((yi > y) != (yj > y)) and (
            x <= (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def geometry_contains(geometry: Dict[str, Any], point: Point) -> bool:
    x, y = point
    gtype = geometry["type"]
    if gtype == "circle":
        cx, cy = geometry["center"]
        r = float(geometry["radius"])
        return (x - float(cx)) ** 2 + (y - float(cy)) ** 2 <= r**2
    if gtype == "rect":
        return (
            float(geometry["xmin"]) <= x <= float(geometry["xmax"])
            and float(geometry["ymin"]) <= y <= float(geometry["ymax"])
        )
    if gtype == "polygon":
        return _point_in_polygon(point, geometry["vertices"])
    raise ValueError(f"Unsupported geometry type: {gtype}")


def _grid_points_in_bbox(
    bbox: Tuple[float, float, float, float],
    *,
    grid: int = 8,
) -> Iterable[Point]:
    xmin, xmax, ymin, ymax = bbox
    if xmax < xmin or ymax < ymin:
        return []
    xs = [xmin + (xmax - xmin) * (i + 0.5) / float(grid) for i in range(grid)]
    ys = [ymin + (ymax - ymin) * (i + 0.5) / float(grid) for i in range(grid)]
    return [(x, y) for x in xs for y in ys]


def _witnesses_for(
    labels: Sequence[str],
    predicates: Dict[str, PredicateSpec],
    free_points: Sequence[Point],
) -> List[Point]:
    geometries = [predicates[label].geometry for label in labels]
    if not geometries:
        return []
    xmin = max(_bbox(g)[0] for g in geometries)
    xmax = min(_bbox(g)[1] for g in geometries)
    ymin = max(_bbox(g)[2] for g in geometries)
    ymax = min(_bbox(g)[3] for g in geometries)
    candidates = list(free_points)
    if xmin <= xmax and ymin <= ymax:
        candidates.extend(_grid_points_in_bbox((xmin, xmax, ymin, ymax), grid=10))
    return [p for p in candidates if all(geometry_contains(g, p) for g in geometries)]


def check_relations(
    predicates: Dict[str, PredicateSpec],
    relation_checks: Sequence[RelationCheck],
    *,
    start_xy: Point,
    free_points: Sequence[Point],
) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for rel in relation_checks:
        if rel.kind == "start_clear":
            key = "start_clear"
            results[key] = not any(
                geometry_contains(spec.geometry, start_xy) for spec in predicates.values()
            )
        elif rel.kind == "nontrivial_region":
            for label in rel.labels:
                key = f"nontrivial:{label}"
                witnesses = _witnesses_for((label,), predicates, free_points)
                results[key] = len(witnesses) >= int(rel.min_witnesses)
        elif rel.kind == "overlap":
            key = "overlap:" + ",".join(rel.labels)
            witnesses = _witnesses_for(rel.labels, predicates, free_points)
            results[key] = len(witnesses) >= int(rel.min_witnesses)
        elif rel.kind == "separated":
            labels_a = rel.labels
            labels_b = rel.other
            key = "separated:" + ",".join(labels_a) + "|" + ",".join(labels_b)
            centers_a = [_center(predicates[label].geometry) for label in labels_a]
            centers_b = [_center(predicates[label].geometry) for label in labels_b]
            min_dist = min(_distance(a, b) for a in centers_a for b in centers_b)
            results[key] = min_dist >= float(rel.min_distance)
        elif rel.kind == "not_all_same_place":
            key = "not_all_same_place:" + ",".join(rel.labels)
            centers = [_center(predicates[label].geometry) for label in rel.labels]
            max_dist = max(_distance(a, b) for a in centers for b in centers)
            results[key] = max_dist >= float(rel.min_distance)
        else:
            raise ValueError(f"Unsupported relation kind: {rel.kind}")

    for label in predicates:
        key = f"nontrivial:{label}"
        results.setdefault(
            key,
            len(_witnesses_for((label,), predicates, free_points)) >= 1,
        )
    return results


def _sample_free_point(rng: random.Random, free_points: Sequence[Point]) -> Point:
    if not free_points:
        raise ValueError("free_points must not be empty")
    return tuple(float(x) for x in rng.choice(list(free_points)))  # type: ignore[arg-type]


def _overlap_groups(template: SemanticTaskTemplate) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, Tuple[str, ...]] = {}
    for rel in template.relations:
        if rel.kind == "overlap":
            for label in rel.labels:
                out[label] = tuple(rel.labels)
    return out


def _shape_kind_for(
    rng: random.Random,
    label_index: int,
    *,
    shape_mode: str,
    nonconvex_prob: float,
) -> str:
    if shape_mode != "mixed":
        return shape_mode
    if rng.random() < float(nonconvex_prob):
        return "nonconvex_polygon"
    return rng.choice(["circle", "rect", "polygon"])


def _make_geometry(
    rng: random.Random,
    *,
    center: Point,
    kind: str,
    size_scale: float,
) -> Dict[str, Any]:
    cx, cy = center
    base = rng.uniform(1.1, 2.2) * float(size_scale)
    if kind == "circle":
        return {"type": "circle", "center": [cx, cy], "radius": base}
    if kind == "rect":
        sx = base * rng.uniform(0.8, 1.5)
        sy = base * rng.uniform(0.8, 1.5)
        return {
            "type": "rect",
            "xmin": cx - sx,
            "xmax": cx + sx,
            "ymin": cy - sy,
            "ymax": cy + sy,
        }
    if kind == "polygon":
        n = rng.choice([3, 4, 5])
        angles = sorted(rng.uniform(0.0, 2.0 * math.pi) for _ in range(n))
        vertices = []
        for angle in angles:
            r = base * rng.uniform(0.75, 1.45)
            vertices.append([cx + r * math.cos(angle), cy + r * math.sin(angle)])
        return {"type": "polygon", "vertices": vertices}
    if kind == "nonconvex_polygon":
        w = base * rng.uniform(1.6, 2.4)
        h = base * rng.uniform(1.6, 2.4)
        t = min(w, h) * rng.uniform(0.35, 0.55)
        return {
            "type": "polygon",
            "vertices": [
                [cx - w / 2, cy - h / 2],
                [cx + w / 2, cy - h / 2],
                [cx + w / 2, cy - h / 2 + t],
                [cx - w / 2 + t, cy - h / 2 + t],
                [cx - w / 2 + t, cy + h / 2],
                [cx - w / 2, cy + h / 2],
            ],
            "nonconvex": True,
        }
    raise ValueError(f"Unsupported shape kind: {kind}")


def generate_predicates_for_template(
    template: SemanticTaskTemplate,
    *,
    rng: random.Random,
    free_points: Sequence[Point],
    shape_mode: str = "mixed",
    nonconvex_prob: float = 0.2,
    size_scale: float = 1.0,
) -> Dict[str, PredicateSpec]:
    overlap_groups = _overlap_groups(template)
    group_centers: Dict[Tuple[str, ...], Point] = {}
    label_centers: Dict[str, Point] = {}
    for label in template.labels:
        group = overlap_groups.get(label)
        if group is not None:
            if group not in group_centers:
                group_centers[group] = _sample_free_point(rng, free_points)
            gx, gy = group_centers[group]
            jitter = 0.7 * float(size_scale)
            label_centers[label] = (
                gx + rng.uniform(-jitter, jitter),
                gy + rng.uniform(-jitter, jitter),
            )
        else:
            label_centers[label] = _sample_free_point(rng, free_points)

    predicates: Dict[str, PredicateSpec] = {}
    for idx, label in enumerate(template.labels):
        shape_kind = _shape_kind_for(
            rng,
            idx,
            shape_mode=shape_mode,
            nonconvex_prob=nonconvex_prob,
        )
        geometry = _make_geometry(
            rng,
            center=label_centers[label],
            kind=shape_kind,
            size_scale=size_scale,
        )
        geometry["color"] = COLORS[idx % len(COLORS)]
        predicates[label] = PredicateSpec(
            name=label,
            evaluator="state_region",
            geometry=geometry,
            metadata={
                "shape_kind": shape_kind,
                "source": "task_space_geometry",
            },
        )
    return predicates


def task_from_template(
    template: SemanticTaskTemplate,
    *,
    task_id: int,
    seed: int,
    free_points: Sequence[Point],
    max_attempts: int,
    shape_mode: str,
    nonconvex_prob: float,
    size_scale: float,
) -> Dict[str, Any]:
    last_relations: Dict[str, bool] = {}
    for attempt in range(max(1, int(max_attempts))):
        cur_seed = int(seed) + int(task_id) * int(max_attempts) + attempt
        rng = random.Random(cur_seed)
        predicates = generate_predicates_for_template(
            template,
            rng=rng,
            free_points=free_points,
            shape_mode=shape_mode,
            nonconvex_prob=nonconvex_prob,
            size_scale=size_scale,
        )
        start_xy = _sample_free_point(rng, free_points)
        relation_results = check_relations(
            predicates,
            template.relations,
            start_xy=start_xy,
            free_points=free_points,
        )
        last_relations = relation_results
        if all(relation_results.values()):
            regions = {label: spec.geometry for label, spec in predicates.items()}
            predicate_specs = {
                label: {
                    "evaluator": spec.evaluator,
                    "geometry": spec.geometry,
                    "metadata": dict(spec.metadata),
                }
                for label, spec in predicates.items()
            }
            return {
                "task_id": int(task_id),
                "seed": int(cur_seed),
                "generation_attempt": int(attempt),
                "formula_e": template.formula,
                "formula_p": None,
                "labels": list(template.labels),
                "regions": regions,
                "start_xy": [float(start_xy[0]), float(start_xy[1])],
                "metadata": {
                    "generator": "generate_semantic_ltl_tasks.py",
                    "template": template.name,
                    "complexity": template.complexity,
                    "semantic_features": list(template.feature_tags),
                    "relation_checks": relation_results,
                    "predicate_specs": predicate_specs,
                    "region_kinds": {
                        label: spec.metadata["shape_kind"] for label, spec in predicates.items()
                    },
                },
            }
    raise RuntimeError(
        f"Failed to generate task {task_id} from template {template.name} after "
        f"{max_attempts} attempts; last relation checks: {last_relations}"
    )


def generate_payload(
    *,
    env_name: str,
    complexity: str,
    num_tasks: int,
    seed: int,
    free_points: Sequence[Point],
    max_task_attempts: int = 100,
    shape_mode: str = "mixed",
    nonconvex_prob: float = 0.2,
    region_size_scale: float = 1.0,
    template_sample_mode: str = "sequential",
) -> Dict[str, Any]:
    rng = random.Random(seed)
    templates = templates_for(complexity)
    tasks = []
    for task_id in range(int(num_tasks)):
        if template_sample_mode == "random":
            template = rng.choice(templates)
        elif template_sample_mode == "sequential":
            template = templates[task_id % len(templates)]
        else:
            raise ValueError(f"Unsupported template_sample_mode: {template_sample_mode}")
        tasks.append(
            task_from_template(
                template,
                task_id=task_id,
                seed=seed,
                free_points=free_points,
                max_attempts=max_task_attempts,
                shape_mode=shape_mode,
                nonconvex_prob=nonconvex_prob,
                size_scale=region_size_scale,
            )
        )
    return {
        "env_name": env_name,
        "generator": "generate_semantic_ltl_tasks.py",
        "complexity": complexity,
        "seed": int(seed),
        "num_tasks": len(tasks),
        "template_sample_mode": template_sample_mode,
        "shape_mode": shape_mode,
        "nonconvex_prob": float(nonconvex_prob),
        "region_size_scale": float(region_size_scale),
        "semantic_task_schema": "state_region_with_predicate_specs_v1",
        "tasks": tasks,
    }


def load_env_free_points(
    *,
    env_name: str,
    seed: int,
    render_size: int,
    env_region_type: str,
    max_regions: int,
    jitter: float,
    samples_per_cell: int,
) -> List[Point]:
    from O_utils.env_utils import make_ltl_env_and_datasets

    env, _train_dataset, _val_dataset = make_ltl_env_and_datasets(
        env_name,
        seed,
        size=render_size,
        region_type=env_region_type,
        max_regions=max_regions,
    )
    maze_map = env.get_maze_map()
    rng = random.Random(seed)
    points: List[Point] = []
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] != 0:
                continue
            x, y = env.ij_to_xy(i, j)
            for _ in range(max(1, int(samples_per_cell))):
                points.append(
                    (
                        float(x + rng.uniform(-jitter, jitter)),
                        float(y + rng.uniform(-jitter, jitter)),
                    )
                )
    if not points:
        raise RuntimeError(f"No free points found for {env_name}")
    return points


def default_save_path(env_name: str, complexity: str, seed: int, num_tasks: int) -> str:
    env_key = env_name.replace("-v0", "")
    return (
        "ltl_tasks/validation/"
        f"{env_key}_{complexity}_semantic_seed{int(seed)}_n{int(num_tasks)}.json"
    )


def save_tasks_json(payload: Dict[str, Any], save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate small evaluator-conditioned LTL validation task files."
    )
    parser.add_argument("--env_name", default="antmaze-giant-stitch-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--complexity", choices=["medium", "hard"], default="medium")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--max_task_attempts", type=int, default=100)
    parser.add_argument("--template_sample_mode", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--shape_mode", choices=["mixed", "circle", "rect", "polygon", "nonconvex_polygon"], default="mixed")
    parser.add_argument("--nonconvex_prob", type=float, default=0.25)
    parser.add_argument("--region_size_scale", type=float, default=1.0)
    parser.add_argument("--render_size", type=int, default=800)
    parser.add_argument("--env_region_type", default="mixed")
    parser.add_argument("--max_regions", type=int, default=16)
    parser.add_argument("--free_point_jitter", type=float, default=0.75)
    parser.add_argument("--samples_per_cell", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    free_points = load_env_free_points(
        env_name=args.env_name,
        seed=args.seed,
        render_size=args.render_size,
        env_region_type=args.env_region_type,
        max_regions=args.max_regions,
        jitter=args.free_point_jitter,
        samples_per_cell=args.samples_per_cell,
    )
    payload = generate_payload(
        env_name=args.env_name,
        complexity=args.complexity,
        num_tasks=args.num_tasks,
        seed=args.seed,
        free_points=free_points,
        max_task_attempts=args.max_task_attempts,
        shape_mode=args.shape_mode,
        nonconvex_prob=args.nonconvex_prob,
        region_size_scale=args.region_size_scale,
        template_sample_mode=args.template_sample_mode,
    )
    save_path = args.save_path or default_save_path(
        args.env_name,
        args.complexity,
        args.seed,
        args.num_tasks,
    )
    save_tasks_json(payload, save_path)
    print(json.dumps({"saved": save_path, "num_tasks": len(payload["tasks"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
