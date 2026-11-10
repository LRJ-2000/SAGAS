"""Execution protocol glue for evaluator-conditioned SAGAS.

This module does not implement a MuJoCo controller.  It defines the interface
between a low-level segment executor and the semantic runtime monitor, so the
same bookkeeping can be reused by pure tests and real rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from S_utils.guard_terms import GuardTerm
from S_utils.product_planner import FiniteProductExecutionPlan, ProductState
from S_utils.runtime_monitor import RuntimeMonitor, TransitionMonitorResult
from S_utils.semantic_types import RolloutTrace, SemanticCarrier


@dataclass
class SegmentExecutionRequest:
    """One planned product transition to be executed by the low-level layer."""

    transition_index: int
    source: ProductState
    target: ProductState
    guard: GuardTerm


@dataclass
class SegmentExecutionResult:
    """Observed low-level rollout segment for one planned transition."""

    carriers: List[SemanticCarrier]
    waypoint_reached: bool
    actions: List[Any] = field(default_factory=list)
    semantic_checkpoint: Optional[bool] = None
    advance_to_transition_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionProtocolResult:
    """Runtime-monitor result for a finite product execution schedule."""

    trace: RolloutTrace
    transition_results: List[TransitionMonitorResult]
    prefix_completed: bool
    suffix_cycles_completed: int
    suffix_cycles_required: int
    finite_lasso_success: bool
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


SegmentExecutor = Callable[[SegmentExecutionRequest], SegmentExecutionResult]


def default_semantic_checkpoint(guard: GuardTerm) -> bool:
    """Positive guard progress is the default semantic checkpoint."""

    return bool(guard.pos)


def _combined_span_guard(guards: Sequence[GuardTerm], source_q: Any, target_q: Any) -> GuardTerm:
    """Combine obligations for a skipped span of planned product transitions."""

    if len(guards) == 1:
        return guards[0]
    pos = set()
    neg = set()
    guard_ids = []
    for guard in guards:
        pos.update(guard.pos)
        neg.update(guard.neg)
        guard_ids.append(guard.guard_id)
    return GuardTerm(
        guard_id="span:" + "|".join(guard_ids),
        source_state=str(source_q),
        target_state=str(target_q),
        pos=frozenset(pos),
        neg=frozenset(neg),
        metadata={"spanned_guard_ids": guard_ids},
    )


def execute_finite_product_plan(
    *,
    execution_plan: FiniteProductExecutionPlan,
    segment_executor: SegmentExecutor,
    runtime_monitor: RuntimeMonitor,
    buchi_runtime: Any = None,
) -> ExecutionProtocolResult:
    """Execute a finite product schedule through a segment executor callback."""

    trace = RolloutTrace()
    transition_results: List[TransitionMonitorResult] = []
    transitions_completed = 0
    failure_reason: Optional[str] = None

    idx = 0
    while idx < len(execution_plan.guards):
        guard = execution_plan.guards[idx]
        request = SegmentExecutionRequest(
            transition_index=idx,
            source=execution_plan.states[idx],
            target=execution_plan.states[idx + 1],
            guard=guard,
        )
        segment = segment_executor(request)
        requested_next_idx = idx + 1
        next_idx = int(segment.advance_to_transition_index or requested_next_idx)
        next_idx = max(requested_next_idx, min(next_idx, len(execution_plan.states) - 1))
        span_guards = execution_plan.guards[idx:next_idx]
        monitor_guard = _combined_span_guard(
            span_guards,
            source_q=request.source[1],
            target_q=execution_plan.states[next_idx][1],
        )
        semantic_checkpoint = (
            any(default_semantic_checkpoint(g) for g in span_guards)
            if segment.semantic_checkpoint is None
            else bool(segment.semantic_checkpoint)
        )
        source_q = request.source[1]
        target_q = execution_plan.states[next_idx][1]
        monitor_result = runtime_monitor.monitor_transition_segment(
            guard=monitor_guard,
            carriers=segment.carriers,
            waypoint_reached=segment.waypoint_reached,
            semantic_checkpoint=semantic_checkpoint,
            buchi_runtime=buchi_runtime,
            q_start=source_q,
            q_target=target_q,
        )
        if (
            not monitor_result.realized
            and segment.metadata.get("refresh_stop_reason") == "refresh_path_unavailable"
        ):
            monitor_result = TransitionMonitorResult(
                realized=False,
                reason="refresh_path_unavailable",
                details=dict(segment.metadata),
            )
        transition_results.append(monitor_result)

        for carrier in segment.carriers:
            trace.observations.append(carrier.observation)
            if carrier.state is not None:
                trace.states.append(carrier.state)
            trace.evaluator_labels.append(runtime_monitor.label_set(carrier))
            trace.product_progress.append(
                {
                    "transition_index": idx,
                    "source": request.source,
                    "target": execution_plan.states[next_idx],
                    "guard_id": monitor_guard.guard_id,
                    "spanned_guard_ids": [g.guard_id for g in span_guards],
                    "advance_to_transition_index": next_idx,
                    "realized": monitor_result.realized,
                    "segment_metadata": dict(segment.metadata),
                }
            )
        trace.actions.extend(segment.actions)

        if not monitor_result.realized:
            failure_reason = monitor_result.reason
            trace.failure_reason = failure_reason
            break
        transitions_completed += next_idx - idx
        idx = next_idx

    prefix_completed = transitions_completed >= execution_plan.prefix_end_index
    suffix_edges_completed = max(0, transitions_completed - execution_plan.prefix_end_index)
    suffix_cycle_len = max(1, int(execution_plan.suffix_cycle_num_edges))
    suffix_cycles_completed = suffix_edges_completed // suffix_cycle_len
    finite_lasso_success = bool(
        prefix_completed
        and suffix_cycles_completed >= execution_plan.suffix_cycles_required
        and failure_reason is None
    )
    if failure_reason is None and not finite_lasso_success:
        failure_reason = "horizon_exhausted"
        trace.failure_reason = failure_reason

    return ExecutionProtocolResult(
        trace=trace,
        transition_results=transition_results,
        prefix_completed=prefix_completed,
        suffix_cycles_completed=suffix_cycles_completed,
        suffix_cycles_required=execution_plan.suffix_cycles_required,
        finite_lasso_success=finite_lasso_success,
        failure_reason=None if finite_lasso_success else failure_reason,
        metadata={
            "transitions_completed": transitions_completed,
            "transitions_planned": len(execution_plan.guards),
            "prefix_end_index": execution_plan.prefix_end_index,
            "suffix_cycle_num_edges": execution_plan.suffix_cycle_num_edges,
        },
    )
