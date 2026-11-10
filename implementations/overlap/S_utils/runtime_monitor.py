"""Runtime monitor for evaluator-conditioned execution.

The monitor is method-internal: it uses the same SemanticEvaluator certificates
as planning, not the oracle state labels used for final benchmark scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from S_utils.certificate_store import CertificateStore
from S_utils.guard_terms import GuardTerm
from S_utils.semantic_types import SemanticCarrier


@dataclass
class TransitionMonitorResult:
    realized: bool
    reason: str = "ok"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeMonitorConfig:
    """Configuration for filtered evaluator-induced monitoring."""

    positive_window: int = 1
    negative_window: int = 1
    hard_negative: bool = True
    local_step_budget: int = 100


def can_advance_to_waypoint(
    current_index: int,
    target_index: int,
    semantic_checkpoint_indices: Iterable[int],
    verified_checkpoint_indices: Optional[Iterable[int]] = None,
) -> bool:
    """Return whether a monotone tracker may jump to `target_index`.

    Ordinary waypoints may be skipped.  Unverified semantic checkpoints between
    the current index and target index may not be bypassed.
    """

    if int(target_index) <= int(current_index):
        return True
    verified = set(int(i) for i in (verified_checkpoint_indices or []))
    for checkpoint in semantic_checkpoint_indices:
        checkpoint = int(checkpoint)
        if int(current_index) < checkpoint < int(target_index) and checkpoint not in verified:
            return False
    return True


class RuntimeMonitor:
    """Evaluator-induced monitor for planned guard obligations."""

    def __init__(
        self,
        certificate_store: CertificateStore,
        config: Optional[RuntimeMonitorConfig] = None,
    ):
        self.store = certificate_store
        self.config = config or RuntimeMonitorConfig()

    def label_set(self, carrier: SemanticCarrier) -> Set[str]:
        """Return evaluator-induced positive labels for one runtime carrier."""

        return self.store.cert_pos_set(carrier)

    def check_carrier_for_guard(
        self,
        carrier: SemanticCarrier,
        guard: GuardTerm,
    ) -> TransitionMonitorResult:
        cert_pos = self.store.cert_pos_set(carrier)
        cert_neg = self.store.cert_neg_set(carrier)
        if not set(guard.pos).issubset(cert_pos):
            return TransitionMonitorResult(
                realized=False,
                reason="missing_positive_certificate",
                details={"required": sorted(guard.pos), "cert_pos": sorted(cert_pos)},
            )
        if not set(guard.neg).issubset(cert_neg):
            return TransitionMonitorResult(
                realized=False,
                reason="missing_negative_certificate",
                details={"required": sorted(guard.neg), "cert_neg": sorted(cert_neg)},
            )
        return TransitionMonitorResult(realized=True)

    def _filtered_pos_set(self, carriers: Sequence[SemanticCarrier]) -> Set[str]:
        window = max(1, int(self.config.positive_window))
        if len(carriers) < window:
            return set()
        recent = carriers[-window:]
        out = set(self.store.propositions)
        for carrier in recent:
            out.intersection_update(self.store.cert_pos_set(carrier))
        return out

    def _filtered_neg_set(self, carriers: Sequence[SemanticCarrier]) -> Set[str]:
        window = max(1, int(self.config.negative_window))
        if len(carriers) < window:
            return set()
        recent = carriers[-window:]
        out = set(self.store.propositions)
        for carrier in recent:
            out.intersection_update(self.store.cert_neg_set(carrier))
        return out

    def _first_negative_failure(
        self,
        carriers: Sequence[SemanticCarrier],
        required_neg: Iterable[str],
    ) -> Optional[Dict[str, Any]]:
        required = set(required_neg)
        if not required:
            return None
        if not carriers:
            return {
                "step": None,
                "required": sorted(required),
                "cert_neg": [],
                "reason": "no_monitor_samples",
            }
        if self.config.hard_negative:
            for step, carrier in enumerate(carriers):
                cert_neg = self.store.cert_neg_set(carrier)
                if not required.issubset(cert_neg):
                    return {
                        "step": step,
                        "required": sorted(required),
                        "cert_neg": sorted(cert_neg),
                    }
            return None
        cert_neg = self._filtered_neg_set(carriers)
        if not required.issubset(cert_neg):
            return {
                "step": len(carriers) - 1,
                "required": sorted(required),
                "cert_neg": sorted(cert_neg),
            }
        return None

    def monitor_transition_segment(
        self,
        *,
        guard: GuardTerm,
        carriers: Sequence[SemanticCarrier],
        waypoint_reached: bool,
        semantic_checkpoint: bool = True,
        local_step_budget: Optional[int] = None,
        buchi_runtime: Any = None,
        q_start: Any = None,
        q_target: Any = None,
    ) -> TransitionMonitorResult:
        """Validate one planned product transition against runtime carriers.

        `carriers` are the deployable observations/states recorded while the
        low-level executor attempted the transition.  The method checks active
        negative obligations over the segment, positive guard progress at the
        semantic checkpoint, optional Buchi branch feasibility, and local
        tracking budget exhaustion.
        """

        carriers = list(carriers)
        budget = int(local_step_budget or self.config.local_step_budget)
        if not waypoint_reached and len(carriers) >= budget:
            return TransitionMonitorResult(
                realized=False,
                reason="runtime_unrealized",
                details={"steps": len(carriers), "local_step_budget": budget},
            )

        neg_failure = self._first_negative_failure(carriers, guard.neg)
        if neg_failure is not None:
            return TransitionMonitorResult(
                realized=False,
                reason="missing_negative_certificate",
                details=neg_failure,
            )

        if not waypoint_reached:
            return TransitionMonitorResult(
                realized=False,
                reason="target_not_reached",
                details={"steps": len(carriers), "local_step_budget": budget},
            )

        if semantic_checkpoint and guard.pos:
            cert_pos = self._filtered_pos_set(carriers)
            if not set(guard.pos).issubset(cert_pos):
                return TransitionMonitorResult(
                    realized=False,
                    reason="missing_positive_certificate",
                    details={
                        "required": sorted(guard.pos),
                        "cert_pos": sorted(cert_pos),
                        "positive_window": int(self.config.positive_window),
                    },
                )

        if buchi_runtime is not None and q_start is not None and q_target is not None:
            labels = [self.label_set(carrier) for carrier in carriers]
            if not buchi_runtime.branch_reaches(q_start, labels, q_target):
                return TransitionMonitorResult(
                    realized=False,
                    reason="buchi_branch_not_realized",
                    details={
                        "q_start": q_start,
                        "q_target": q_target,
                        "num_labels": len(labels),
                    },
                )

        return TransitionMonitorResult(
            realized=True,
            details={"steps": len(carriers), "guard_id": guard.guard_id},
        )
