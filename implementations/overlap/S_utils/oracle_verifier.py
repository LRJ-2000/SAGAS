"""Final oracle verification helpers for actual state trajectories.

These helpers are intentionally separate from the runtime monitor.  The runtime
monitor is method-internal and evaluator-induced; the oracle verifier consumes
the actual state trajectory recorded during rollout and applies a common
state-label standard for evaluation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from S_utils.guard_terms import GuardTerm


def extract_oracle_label_trace(
    states: Sequence[Any],
    oracle_labeler: Callable[[Any], Set[str]],
) -> List[Set[str]]:
    """Extract set-valued oracle labels from actual rollout states."""

    return [set(oracle_labeler(state)) for state in states]


def finite_lasso_summary(
    oracle_labels: Sequence[Set[str]],
    prefix_completed: bool,
    suffix_cycles_completed: int,
    suffix_cycles_required: int,
) -> dict:
    """Basic finite-lasso bookkeeping independent of a specific LTL backend."""

    success = bool(prefix_completed and suffix_cycles_completed >= suffix_cycles_required)
    return {
        "finite_lasso_success": success,
        "prefix_completed": bool(prefix_completed),
        "suffix_cycles_completed": int(suffix_cycles_completed),
        "suffix_cycles_required": int(suffix_cycles_required),
        "num_labeled_states": len(oracle_labels),
    }


def buchi_trace_summary(
    oracle_labels: Sequence[Set[str]],
    buchi_runtime,
    consume_initial: bool = True,
) -> dict:
    """Run a nondeterministic Buchi state-set monitor on a finite label trace."""

    labels = [set(x) for x in oracle_labels]
    if consume_initial:
        if not labels:
            q_set = set()
            processed_start = 0
        else:
            q_set = set(buchi_runtime.initial_state_set_after_label(labels[0]))
            processed_start = 1
    else:
        if hasattr(buchi_runtime, "init_states"):
            q_set = set(buchi_runtime.init_states)
        else:
            q_set = {buchi_runtime.initial_state()}
        processed_start = 0

    state_sets = [set(q_set)]
    accepting_seen = any(buchi_runtime.is_accepting(q) for q in q_set)
    dead = not q_set
    for label_set in labels[processed_start:]:
        if dead:
            state_sets.append(set())
            continue
        q_set = set(buchi_runtime.step_set(q_set, label_set))
        state_sets.append(set(q_set))
        accepting_seen = accepting_seen or any(buchi_runtime.is_accepting(q) for q in q_set)
        dead = not q_set

    return {
        "buchi_accepting_seen": bool(accepting_seen),
        "buchi_dead": bool(dead),
        "final_state_set": sorted(q_set),
        "num_label_steps": len(labels),
        "state_sets": [sorted(s) for s in state_sets],
    }


def _product_state(value: Any) -> Tuple[Any, Any]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise ValueError(f"expected product state pair, got {value!r}")


def _span_guard(execution_plan: Any, start_idx: int, next_idx: int) -> GuardTerm:
    guards = list(execution_plan.guards[start_idx:next_idx])
    if not guards:
        return GuardTerm(guard_id=f"empty_span:{start_idx}->{next_idx}")
    if len(guards) == 1:
        return guards[0]
    pos = set()
    neg = set()
    guard_ids = []
    for guard in guards:
        pos.update(guard.pos)
        neg.update(guard.neg)
        guard_ids.append(guard.guard_id)
    source = _product_state(execution_plan.states[start_idx])
    target = _product_state(execution_plan.states[next_idx])
    return GuardTerm(
        guard_id="span:" + "|".join(guard_ids),
        source_state=str(source[1]),
        target_state=str(target[1]),
        pos=frozenset(pos),
        neg=frozenset(neg),
        metadata={"spanned_guard_ids": guard_ids},
    )


def _progress_groups(product_progress: Sequence[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
    """Group contiguous rollout samples by planned transition span.

    Returns tuples `(label_start, label_end, transition_index, next_index)`,
    where `label_end` is exclusive.
    """

    groups: List[Tuple[int, int, int, int]] = []
    if not product_progress:
        return groups
    start = 0
    current = product_progress[0]
    cur_transition = int(current.get("transition_index", 0))
    cur_next = int(current.get("advance_to_transition_index", cur_transition + 1))
    for idx, item in enumerate(product_progress[1:], start=1):
        transition = int(item.get("transition_index", 0))
        next_idx = int(item.get("advance_to_transition_index", transition + 1))
        if transition != cur_transition or next_idx != cur_next:
            groups.append((start, idx, cur_transition, cur_next))
            start = idx
            cur_transition = transition
            cur_next = next_idx
    groups.append((start, len(product_progress), cur_transition, cur_next))
    return groups


def _branch_reaches(buchi_runtime: Any, q_start: Any, labels: Sequence[Set[str]], q_target: Any) -> bool:
    if hasattr(buchi_runtime, "branch_reaches"):
        return bool(buchi_runtime.branch_reaches(q_start, labels, q_target))
    q_set = {q_start}
    for label_set in labels:
        q_set = set(buchi_runtime.step_set(q_set, label_set))
        if not q_set:
            return False
    return q_target in q_set


def oracle_plan_realization_summary(
    oracle_labels: Sequence[Set[str]],
    *,
    product_progress: Optional[Sequence[Dict[str, Any]]],
    execution_plan: Any,
    buchi_runtime: Any = None,
) -> dict:
    """Verify planned product-transition obligations using oracle labels.

    This check is independent of the method-internal runtime monitor.  It uses
    the recorded product-progress segmentation only to align actual state labels
    with the planned finite product schedule, then rechecks positive/negative
    guard obligations and optional Buchi branch reachability with oracle labels.
    """

    labels = [set(x) for x in oracle_labels]
    progress = list(product_progress or [])
    num_guards = len(getattr(execution_plan, "guards", []))
    prefix_end_index = int(getattr(execution_plan, "prefix_end_index", 0))
    suffix_cycle_len = max(1, int(getattr(execution_plan, "suffix_cycle_num_edges", 1)))
    suffix_required = int(getattr(execution_plan, "suffix_cycles_required", 0))
    details: List[Dict[str, Any]] = []

    if not progress:
        return {
            "oracle_plan_checked": False,
            "oracle_plan_realized": False,
            "oracle_plan_failure_reason": "missing_product_progress",
            "oracle_transitions_verified": 0,
            "oracle_prefix_completed": False,
            "oracle_suffix_cycles_completed": 0,
            "oracle_suffix_cycles_required": suffix_required,
            "oracle_transition_checks": details,
        }
    if len(progress) != len(labels):
        return {
            "oracle_plan_checked": False,
            "oracle_plan_realized": False,
            "oracle_plan_failure_reason": "progress_label_length_mismatch",
            "oracle_transitions_verified": 0,
            "oracle_prefix_completed": False,
            "oracle_suffix_cycles_completed": 0,
            "oracle_suffix_cycles_required": suffix_required,
            "oracle_transition_checks": details,
            "num_product_progress": len(progress),
            "num_oracle_labels": len(labels),
        }

    transitions_verified = 0
    failure_reason = None
    for label_start, label_end, transition_idx, next_idx in _progress_groups(progress):
        if transition_idx != transitions_verified:
            failure_reason = "product_progress_misaligned"
            details.append(
                {
                    "transition_index": transition_idx,
                    "expected_transition_index": transitions_verified,
                    "next_index": next_idx,
                    "realized": False,
                    "reason": failure_reason,
                }
            )
            break
        if next_idx <= transition_idx or next_idx > num_guards:
            failure_reason = "invalid_product_progress_span"
            details.append(
                {
                    "transition_index": transition_idx,
                    "next_index": next_idx,
                    "realized": False,
                    "reason": failure_reason,
                }
            )
            break

        guard = _span_guard(execution_plan, transition_idx, next_idx)
        segment_labels = labels[label_start:label_end]
        neg_hit = sorted(set().union(*[label.intersection(guard.neg) for label in segment_labels]))
        pos_ok = True
        if guard.pos:
            pos_ok = any(set(guard.pos).issubset(label) for label in segment_labels)
        buchi_ok = True
        if buchi_runtime is not None:
            source = _product_state(execution_plan.states[transition_idx])
            target = _product_state(execution_plan.states[next_idx])
            buchi_ok = _branch_reaches(buchi_runtime, source[1], segment_labels, target[1])

        realized = bool(not neg_hit and pos_ok and buchi_ok)
        reason = "ok"
        if neg_hit:
            reason = "oracle_negative_violation"
        elif not pos_ok:
            reason = "oracle_missing_positive"
        elif not buchi_ok:
            reason = "oracle_buchi_branch_not_realized"

        details.append(
            {
                "transition_index": transition_idx,
                "next_index": next_idx,
                "guard_id": guard.guard_id,
                "pos": sorted(guard.pos),
                "neg": sorted(guard.neg),
                "label_start": label_start,
                "label_end": label_end,
                "realized": realized,
                "reason": reason,
                "neg_hit": neg_hit,
            }
        )
        if not realized:
            failure_reason = reason
            break
        transitions_verified = next_idx
        if transitions_verified >= num_guards:
            break

    prefix_completed = transitions_verified >= prefix_end_index
    suffix_edges_completed = max(0, transitions_verified - prefix_end_index)
    suffix_completed = suffix_edges_completed // suffix_cycle_len
    plan_realized = bool(
        failure_reason is None
        and transitions_verified >= num_guards
        and prefix_completed
        and suffix_completed >= suffix_required
    )
    if failure_reason is None and not plan_realized:
        failure_reason = "oracle_finite_lasso_incomplete"

    return {
        "oracle_plan_checked": True,
        "oracle_plan_realized": plan_realized,
        "oracle_plan_failure_reason": None if plan_realized else failure_reason,
        "oracle_transitions_verified": transitions_verified,
        "oracle_prefix_completed": prefix_completed,
        "oracle_suffix_cycles_completed": int(suffix_completed),
        "oracle_suffix_cycles_required": suffix_required,
        "oracle_transition_checks": details,
    }


def oracle_finite_lasso_verdict(
    states: Sequence[Any],
    oracle_labeler: Callable[[Any], Set[str]],
    buchi_runtime,
    *,
    prefix_completed: bool,
    suffix_cycles_completed: int,
    suffix_cycles_required: int,
    execution_plan: Any = None,
    product_progress: Optional[Sequence[Dict[str, Any]]] = None,
) -> dict:
    """Combine oracle state labels, Buchi monitoring, and finite-lasso metrics."""

    labels = extract_oracle_label_trace(states, oracle_labeler)
    plan_summary = None
    if execution_plan is not None:
        plan_summary = oracle_plan_realization_summary(
            labels,
            product_progress=product_progress,
            execution_plan=execution_plan,
            buchi_runtime=buchi_runtime,
        )
        lasso = finite_lasso_summary(
            labels,
            prefix_completed=bool(plan_summary["oracle_prefix_completed"]),
            suffix_cycles_completed=int(plan_summary["oracle_suffix_cycles_completed"]),
            suffix_cycles_required=int(plan_summary["oracle_suffix_cycles_required"]),
        )
    else:
        lasso = finite_lasso_summary(
            labels,
            prefix_completed=prefix_completed,
            suffix_cycles_completed=suffix_cycles_completed,
            suffix_cycles_required=suffix_cycles_required,
        )
    buchi = buchi_trace_summary(labels, buchi_runtime)
    if plan_summary is not None and plan_summary["oracle_plan_checked"]:
        success = bool(plan_summary["oracle_plan_realized"])
        failure_reason = plan_summary["oracle_plan_failure_reason"]
    else:
        success = (
            bool(lasso["finite_lasso_success"])
            and bool(buchi["buchi_accepting_seen"])
            and not bool(buchi["buchi_dead"])
        )
        failure_reason = None if success else "oracle_ltl_violation"
        if not lasso["finite_lasso_success"]:
            failure_reason = "horizon_exhausted"
        if buchi["buchi_dead"]:
            failure_reason = "oracle_ltl_violation"
    return {
        **lasso,
        **buchi,
        **(plan_summary or {}),
        "oracle_success": bool(success),
        "failure_reason": failure_reason,
        "oracle_labels": [sorted(x) for x in labels],
    }
