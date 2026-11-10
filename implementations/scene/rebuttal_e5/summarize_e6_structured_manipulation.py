#!/usr/bin/env python3
"""Summarize the structured Scene manipulation grid and planning times."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


TIMING_KEYS = (
    "evaluator_setup",
    "semantic_augmentation",
    "product_search",
    "total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-path", required=True)
    parser.add_argument("--result-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a fixed-order prefix of the task manifest.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timing_values(
    items: Sequence[Mapping[str, Any]],
    key: str,
) -> List[float]:
    values = []
    for item in items:
        value = finite(item.get(f"{key}_seconds"))
        if value is None:
            value = finite(
                item.get("planning_timing_seconds", {}).get(key)
            )
        if value is not None:
            values.append(value)
    return values


def summarize_group(
    group_type: str,
    group: str,
    items: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    failures = Counter(str(item.get("failure_class")) for item in items)
    actions = [
        value
        for item in items
        if (value := finite(item.get("num_actions"))) is not None
    ]
    finite_lasso_successes = sum(item["finite_lasso_success"] for item in items)
    row: Dict[str, Any] = {
        "group_type": group_type,
        "group": group,
        "task_instances": len(items),
        "plans_found": sum(bool(item.get("planner_found_plan")) for item in items),
        "runtime_successes": sum(
            bool(item.get("runtime_success")) for item in items
        ),
        "oracle_successes": sum(
            bool(item.get("oracle_success")) for item in items
        ),
        "finite_lasso_successes": finite_lasso_successes,
        "finite_lasso_success_rate": finite_lasso_successes / len(items) if items else 0.0,
        "semantic_grounding_errors": sum(
            bool(item.get("semantic_grounding_error")) for item in items
        ),
        "mean_num_actions": (
            sum(actions) / len(actions) if actions else None
        ),
        "failure_classes": json.dumps(
            dict(sorted(failures.items())),
            sort_keys=True,
        ),
    }
    for timing_key in TIMING_KEYS:
        values = timing_values(items, timing_key)
        row[f"{timing_key}_seconds_count"] = len(values)
        row[f"{timing_key}_seconds_median"] = percentile(values, 0.5)
        row[f"{timing_key}_seconds_p95"] = percentile(values, 0.95)
        row[f"{timing_key}_seconds_max"] = max(values) if values else None
    return row


def main() -> int:
    args = parse_args()
    task_path = Path(args.task_path).resolve()
    result_path = Path(args.result_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_payload = load_json(task_path)
    result_payload = load_json(result_path)
    task_specs = {
        int(task["task_id"]): task for task in task_payload.get("tasks", [])
    }
    result_items = list(result_payload.get("tasks", []))
    result_ids = {int(item["task_id"]) for item in result_items}
    expected_ids = set(task_specs)
    if args.allow_partial:
        if not result_ids or not result_ids.issubset(expected_ids):
            raise ValueError(
                "Partial results contain no valid task ids or ids absent "
                "from the task manifest"
            )
    elif result_ids != expected_ids:
        raise ValueError(
            "Task manifest and result summary do not contain the same task ids"
        )

    per_task: List[Dict[str, Any]] = []
    for result in sorted(result_items, key=lambda item: int(item["task_id"])):
        task_id = int(result["task_id"])
        task = task_specs[task_id]
        metadata = dict(task.get("metadata", {}))
        runtime_success = bool(result.get("runtime_success"))
        oracle_success = bool(result.get("oracle_success"))
        row = {
            "task_id": task_id,
            "base_task_id": metadata.get("base_task_id"),
            "template": metadata.get("template"),
            "logic_family": metadata.get("logic_family"),
            "family_index": metadata.get("family_index"),
            "object_track": metadata.get("object_track"),
            "stage_count": metadata.get("stage_count"),
            "official_init_config": metadata.get(
                "official_init_config"
            ),
            "has_negative_literal": metadata.get(
                "has_negative_literal"
            ),
            "has_disjunction": metadata.get("has_disjunction"),
            "has_recurrence": metadata.get("has_recurrence"),
            "requires_cube_motion": metadata.get(
                "requires_cube_motion"
            ),
            "formula": result.get("formula"),
            "planner_found_plan": bool(result.get("planner_found_plan")),
            "runtime_success": runtime_success,
            "oracle_success": oracle_success,
            "finite_lasso_success": result.get("status") == "ok" and runtime_success and oracle_success,
            "semantic_grounding_error": bool(
                result.get("semantic_grounding_error")
            ),
            "failure_class": result.get("failure_class"),
            "num_actions": result.get("num_actions"),
        }
        for timing_key in TIMING_KEYS:
            row[f"{timing_key}_seconds"] = result.get(
                "planning_timing_seconds", {}
            ).get(timing_key)
        per_task.append(row)

    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in per_task:
        groups[("all", "all")].append(item)
        for key in (
            "logic_family",
            "stage_count",
            "object_track",
            "official_init_config",
            "requires_cube_motion",
        ):
            groups[(key, str(item.get(key)))].append(item)
    grouped_rows = [
        summarize_group(group_type, group, items)
        for (group_type, group), items in sorted(groups.items())
    ]

    generation = dict(task_payload.get("generation_summary", {}))
    candidate_count = int(
        generation.get("candidate_count", len(task_specs))
    )
    rejected = list(generation.get("rejected_candidates", []))
    overall = next(
        row
        for row in grouped_rows
        if row["group_type"] == "all" and row["group"] == "all"
    )
    summary = {
        "protocol": {
            "candidate_count": candidate_count,
            "accepted_after_initial_automaton_check": len(task_specs),
            "rejected_by_initial_automaton_check": len(rejected),
            "num_base_formulas": len(
                {
                    item.get("base_task_id")
                    for item in per_task
                    if item.get("base_task_id") is not None
                }
            ),
            "logic_families": sorted(
                {
                    str(item.get("logic_family"))
                    for item in per_task
                }
            ),
            "official_initial_configurations": sorted(
                {
                    str(item.get("official_init_config"))
                    for item in per_task
                }
            ),
            "selection_rule": generation.get("selection_rule"),
        },
        "planning_timing_contract": result_payload.get(
            "planning_timing_contract"
        ),
        "overall": overall,
        "groups": grouped_rows,
        "provenance": {
            "task_path": str(task_path),
            "task_sha256": sha256(task_path),
            "result_summary": str(result_path),
            "result_summary_sha256": sha256(result_path),
        },
    }

    write_csv(output_dir / "structured_manipulation_per_task.csv", per_task)
    write_csv(
        output_dir / "structured_manipulation_grouped.csv",
        grouped_rows,
    )
    (output_dir / "structured_manipulation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    family_rows = [
        row for row in grouped_rows if row["group_type"] == "logic_family"
    ]
    tex_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Logic family & Tasks & Plan & Oracle & Finite-lasso success & "
            r"$t_{\mathrm{plan}}$ med. & $t_{\mathrm{plan}}$ p95 \\"
        ),
        r"\midrule",
    ]
    for row in family_rows:
        n = int(row["task_instances"])
        median = row["total_seconds_median"]
        p95 = row["total_seconds_p95"]
        tex_lines.append(
            f"{row['group'].replace('_', ' ')} & {n}"
            f" & {row['plans_found']}/{n}"
            f" & {row['oracle_successes']}/{n}"
            f" & {row['finite_lasso_successes']}/{n}"
            f" & {median:.2f}"
            f" & {p95:.2f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (output_dir / "structured_manipulation_table.tex").write_text(
        "\n".join(tex_lines),
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
