#!/usr/bin/env python3
"""Summarize the fixed E5 overlapping-predicate capability suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Sequence, Tuple


GROUPS: Sequence[Tuple[str, range]] = (
    ("Overlap event", range(0, 40)),
    ("Negative-until overlap", range(40, 80)),
    ("Recurrent overlap", range(80, 120)),
)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - radius, center + radius


def failure_reason(row: Mapping[str, Any]) -> str:
    if row.get("oracle_success"):
        return ""
    if not row.get("planner_found_plan"):
        return str(row.get("runtime_failure_reason") or "planner_infeasible")
    return str(
        row.get("runtime_failure_reason")
        or row.get("oracle_failure_reason")
        or "execution_failure"
    )


def aggregate(name: str, selected: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(selected)
    successes = sum(bool(row.get("oracle_success")) for row in selected)
    plans = sum(bool(row.get("planner_found_plan")) for row in selected)
    lo, hi = wilson(successes, total)
    supported_props = sum(
        sum(bool(value) for value in row.get("prop_anchor_supported", {}).values())
        for row in selected
    )
    total_props = sum(
        len(row.get("prop_anchor_supported", {})) for row in selected
    )
    compared_steps = sum(
        int(row.get("evaluator_oracle_mismatch", {}).get("num_compared_steps", 0))
        for row in selected
    )
    mismatch_steps = sum(
        int(row.get("evaluator_oracle_mismatch", {}).get("num_mismatch_steps", 0))
        for row in selected
    )
    successful_actions = [
        int(row["num_actions"]) for row in selected if row.get("oracle_success")
    ]
    failures = Counter(filter(None, (failure_reason(row) for row in selected)))
    return {
        "group": name,
        "tasks": total,
        "plans_found": plans,
        "plan_rate": plans / total,
        "oracle_successes": successes,
        "oracle_success_rate": successes / total,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "success_given_plan": successes / plans if plans else math.nan,
        "connected_proposition_fraction": (
            supported_props / total_props if total_props else math.nan
        ),
        "evaluator_oracle_mismatch_rate": (
            mismatch_steps / compared_steps if compared_steps else math.nan
        ),
        "successful_actions_median": (
            median(successful_actions) if successful_actions else math.nan
        ),
        "successful_actions_mean": (
            mean(successful_actions) if successful_actions else math.nan
        ),
        "failure_counts": json.dumps(dict(sorted(failures.items()))),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex(rows: Sequence[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "{} & {} & {}/{} ({:.1f}\\%) & {}/{} ({:.1f}\\%) & "
            "[{:.1f}, {:.1f}] & {:.1f}\\% & {:.0f}".format(
                row["group"],
                row["tasks"],
                row["plans_found"],
                row["tasks"],
                100.0 * row["plan_rate"],
                row["oracle_successes"],
                row["tasks"],
                100.0 * row["oracle_success_rate"],
                100.0 * row["wilson_lo"],
                100.0 * row["wilson_hi"],
                100.0 * row["connected_proposition_fraction"],
                row["successful_actions_median"],
            )
        )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\caption{Fixed overlapping-predicate and multi-positive-guard "
            "capability suite on AntMaze Giant-Stitch. Success is determined "
            "by independent post-rollout LTL verification.}",
            "\\label{tab:rebuttal-overlap}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Task group & $N$ & Plan found & Success & 95\\% CI "
            "& Prop. support & Med. actions \\\\",
            "\\midrule",
            *[line + " \\\\" for line in body],
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.batch_summary.read_text())
    results = payload["file_aggregates"][0]["results"]
    by_id = {int(row["task_id"]): row for row in results}
    rows = [
        aggregate(name, [by_id[task_id] for task_id in task_ids])
        for name, task_ids in GROUPS
    ]
    rows.append(aggregate("All", list(results)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "e5_overlap_summary.csv", rows)
    (args.output_dir / "e5_overlap_table.tex").write_text(
        latex(rows), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
