"""Summarize SAGAS navigation outcomes with the original SR and NCC definitions."""

import ast
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence



def parse_literal(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    if text.lower() in {"nan", "none"}:
        return math.nan if text.lower() == "nan" else None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def as_list(value: Any) -> List[Any]:
    parsed = parse_literal(value)
    if parsed is None:
        return []
    if isinstance(parsed, float) and math.isnan(parsed):
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    return [parsed]


def as_float(value: Any) -> float:
    parsed = parse_literal(value)
    if parsed is None:
        return math.nan
    if isinstance(parsed, (int, float)):
        return float(parsed)
    try:
        return float(parsed)
    except Exception:
        return math.nan


def numeric_list(value: Any) -> List[float]:
    values = []
    for item in as_list(value):
        number = as_float(item)
        if not math.isnan(number):
            values.append(number)
    return values


def mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def sample_std(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if len(clean) < 2:
        return math.nan
    avg = sum(clean) / len(clean)
    var = sum((v - avg) ** 2 for v in clean) / (len(clean) - 1)
    return math.sqrt(var)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_overall_csv(csv_path: Path, max_steps: float = math.nan) -> Dict[str, Any]:
    rows = read_csv_rows(csv_path)
    successes: List[float] = []
    one_suffix: List[float] = []
    cycle_fractions: List[float] = []
    prefix_steps: List[float] = []
    suffix_steps: List[float] = []
    prefix_one_suffix_steps: List[float] = []
    success_prefix_one_suffix_steps: List[float] = []
    total_steps: List[float] = []
    success_total_steps: List[float] = []
    normalized_capped_costs: List[float] = []
    plan_times: List[float] = []
    expanded_product_states: List[float] = []
    expanded_prefix_states: List[float] = []
    expanded_suffix_states: List[float] = []
    unavailable_counts: List[float] = []
    planner_variants: List[str] = []
    executor_variants: List[str] = []
    failure_reasons: Counter[str] = Counter()

    for row in rows:
        success = as_float(row.get("success_rate"))
        if not math.isnan(success):
            successes.append(success)

        planner_variant = str(row.get("planner_variant", "") or "")
        executor_variant = str(row.get("executor_variant", "") or "")
        if planner_variant:
            planner_variants.append(planner_variant)
        if executor_variant:
            executor_variants.append(executor_variant)

        one_vals = numeric_list(row.get("suffix_one_cycle_success"))
        one_suffix_success = one_vals[0] if one_vals else math.nan
        if one_vals:
            one_suffix.append(one_suffix_success)

        requested_vals = numeric_list(row.get("suffix_cycles_requested"))
        completed_vals = numeric_list(row.get("suffix_cycles_completed"))
        if requested_vals and completed_vals and requested_vals[0] > 0:
            cycle_fractions.append(completed_vals[0] / requested_vals[0])

        prefix_vals = numeric_list(row.get("prefix_steps"))
        suffix_vals = numeric_list(row.get("suffix_steps"))
        prefix_steps.extend(prefix_vals)
        suffix_steps.extend(suffix_vals)
        one_suffix_vals = numeric_list(row.get("suffix_one_cycle_steps"))
        prefix_one_suffix_step = math.nan
        if prefix_vals and one_suffix_vals:
            prefix_one_suffix_step = prefix_vals[0] + one_suffix_vals[0]
            prefix_one_suffix_steps.append(prefix_one_suffix_step)
            if not math.isnan(success) and success >= 0.5:
                success_prefix_one_suffix_steps.append(prefix_one_suffix_step)
        raw_total = numeric_list(row.get("total_steps"))
        if raw_total:
            total_steps.extend(raw_total)
            if not math.isnan(success) and success >= 0.5:
                success_total_steps.extend(raw_total)
        elif prefix_vals and suffix_vals and not math.isnan(success) and success >= 0.5:
            success_total_steps.append(prefix_vals[0] + suffix_vals[0])
        if not math.isnan(max_steps) and max_steps > 0:
            cap = 2.0 * max_steps
            if (
                not math.isnan(one_suffix_success)
                and one_suffix_success >= 0.5
                and not math.isnan(prefix_one_suffix_step)
            ):
                normalized_capped_costs.append(
                    min(max(prefix_one_suffix_step, 0.0), cap) / cap
                )
            else:
                normalized_capped_costs.append(1.0)
        plan_times.extend(numeric_list(row.get("plan_times")))

        for column, sink in [
            ("num_expanded_product_states", expanded_product_states),
            ("num_expanded_prefix_states", expanded_prefix_states),
            ("num_expanded_suffix_states", expanded_suffix_states),
            ("anchor_unavailable_count", unavailable_counts),
        ]:
            value = as_float(row.get(column))
            if not math.isnan(value):
                sink.append(value)

        for reason in as_list(row.get("failure_reasons")):
            if reason not in (None, "", "None"):
                failure_reasons[str(reason)] += 1

    if not total_steps and prefix_steps and suffix_steps:
        total_steps = [p + s for p, s in zip(prefix_steps, suffix_steps)]

    return {
        "status": "done",
        "tasks": len(rows),
        "n_success_rate": len(successes),
        "success_rate": mean(successes),
        "success_count": sum(1 for v in successes if v >= 0.5),
        "n_one_suffix_success_rate": len(one_suffix),
        "one_suffix_success_rate": mean(one_suffix),
        "n_suffix_cycle_completion_rate": len(cycle_fractions),
        "suffix_cycle_completion_rate": mean(cycle_fractions),
        "n_prefix_steps": len(prefix_steps),
        "avg_prefix_steps": mean(prefix_steps),
        "n_suffix_steps": len(suffix_steps),
        "avg_suffix_steps": mean(suffix_steps),
        "n_prefix_one_suffix_steps": len(prefix_one_suffix_steps),
        "avg_prefix_one_suffix_steps": mean(prefix_one_suffix_steps),
        "n_success_prefix_one_suffix_steps": len(success_prefix_one_suffix_steps),
        "avg_success_prefix_one_suffix_steps": mean(success_prefix_one_suffix_steps),
        "n_total_steps": len(total_steps),
        "avg_total_steps": mean(total_steps),
        "n_success_total_steps": len(success_total_steps),
        "avg_success_total_steps": mean(success_total_steps),
        "n_normalized_capped_cost": len(normalized_capped_costs),
        "avg_normalized_capped_cost": mean(normalized_capped_costs),
        "n_plan_times": len(plan_times),
        "avg_plan_times": mean(plan_times),
        "std_prefix_steps": sample_std(prefix_steps),
        "std_suffix_steps": sample_std(suffix_steps),
        "std_prefix_one_suffix_steps": sample_std(prefix_one_suffix_steps),
        "std_success_prefix_one_suffix_steps": sample_std(
            success_prefix_one_suffix_steps
        ),
        "std_total_steps": sample_std(total_steps),
        "std_success_total_steps": sample_std(success_total_steps),
        "std_normalized_capped_cost": sample_std(normalized_capped_costs),
        "std_plan_times": sample_std(plan_times),
        "avg_num_expanded_product_states": mean(expanded_product_states),
        "avg_num_expanded_prefix_states": mean(expanded_prefix_states),
        "avg_num_expanded_suffix_states": mean(expanded_suffix_states),
        "avg_anchor_unavailable_count": mean(unavailable_counts),
        "std_num_expanded_product_states": sample_std(expanded_product_states),
        "std_num_expanded_prefix_states": sample_std(expanded_prefix_states),
        "std_num_expanded_suffix_states": sample_std(expanded_suffix_states),
        "std_anchor_unavailable_count": sample_std(unavailable_counts),
        "planner_variant": ";".join(sorted(set(planner_variants))),
        "executor_variant": ";".join(sorted(set(executor_variants))),
        "failure_reasons": ";".join(
            f"{key}:{value}" for key, value in failure_reasons.most_common()
        )
        or "none",
        "csv": str(csv_path),
    }
