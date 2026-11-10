"""Manipulation-specific anchor candidate selection.

The default SAGAS anchor path treats every predicate-positive carrier as an
equally good semantic witness.  This is reasonable for navigation regions, but
too ambiguous for manipulation predicates such as ``window_open``: a positive
carrier fixes the whole scene state, including nuisance object configurations.

This module keeps the manipulation refinement isolated.  It prioritizes
predicate event endpoints when consecutive support carriers are available, then
selects a context-diverse subset so anchors for the same proposition cover
different object configurations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from S_utils.certificate_store import CertificateStore
from S_utils.guard_terms import GuardTerm
from S_utils.manipulation_predicates import XYZ_CENTER, scene_state_from_value
from S_utils.semantic_types import PredicateCertificate, SemanticCarrier


@dataclass(frozen=True)
class SelectedAnchorCandidate:
    """A support carrier selected as an anchor candidate with diagnostics."""

    carrier: SemanticCarrier
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnchorSelectionReport:
    """Diagnostics returned by a manipulation candidate selector."""

    selected: List[SelectedAnchorCandidate]
    scanned: int
    certified: int
    connected: int
    saw_uncertainty: bool = False
    example_failures: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _carrier_key(carrier: SemanticCarrier) -> Any:
    if carrier.index is not None:
        return (carrier.source, int(carrier.index))
    return ("object", id(carrier))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def scene_object_context_descriptor(value: Any) -> List[float]:
    """Return a compact object-context descriptor for scene manipulation.

    The descriptor intentionally excludes the full robot qpos.  The diversity
    objective is to cover different object configurations, while event endpoint
    selection already provides contact-relevant robot poses when available.
    """

    scene = scene_state_from_value(value)
    buttons = scene.get("button_states")
    if buttons is None:
        button0 = button1 = -1.0
    else:
        try:
            button0 = _safe_float(buttons[0], -1.0)
            button1 = _safe_float(buttons[1], -1.0)
        except Exception:
            button0 = button1 = -1.0

    drawer = _safe_float(scene.get("drawer_pos"), 0.0) / 0.16
    window = _safe_float(scene.get("window_pos"), 0.0) / 0.2

    blocks = scene.get("block_positions", {})
    block = None
    if isinstance(blocks, Mapping):
        block = blocks.get(0)
    if block is None:
        cube = [0.0, 0.0, 0.0]
    else:
        try:
            cube_arr = list(block)
            cube = [
                (_safe_float(cube_arr[0]) - float(XYZ_CENTER[0])) / 0.35,
                (_safe_float(cube_arr[1]) - float(XYZ_CENTER[1])) / 0.55,
                (_safe_float(cube_arr[2]) - 0.02) / 0.2,
            ]
        except Exception:
            cube = [0.0, 0.0, 0.0]

    return [button0, button1, drawer, window] + cube


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _is_event_endpoint(
    *,
    current: SemanticCarrier,
    predecessor: Optional[SemanticCarrier],
    propositions: Iterable[str],
    store: CertificateStore,
) -> bool:
    if predecessor is None:
        return False
    for prop in propositions:
        prev = store.carrier_cert(predecessor, prop)
        cur = store.carrier_cert(current, prop)
        if prev.cert_neg and cur.cert_pos:
            return True
    return False


class ManipulationAnchorSelector:
    """Select event-prioritized and context-diverse anchors for scene tasks."""

    def __init__(
        self,
        *,
        context_thresholds: Sequence[float] = (0.35, 0.2, 0.0),
    ):
        self.context_thresholds = tuple(float(x) for x in context_thresholds)

    def _predecessor_map(
        self,
        carriers: Sequence[SemanticCarrier],
    ) -> Dict[Any, Optional[SemanticCarrier]]:
        by_int_index: Dict[int, SemanticCarrier] = {}
        for carrier in carriers:
            if carrier.index is None:
                continue
            try:
                by_int_index[int(carrier.index)] = carrier
            except (TypeError, ValueError):
                continue
        out: Dict[Any, Optional[SemanticCarrier]] = {}
        for carrier in carriers:
            pred = None
            if carrier.index is not None:
                try:
                    pred = by_int_index.get(int(carrier.index) - 1)
                except (TypeError, ValueError):
                    pred = None
            out[_carrier_key(carrier)] = pred
        return out

    def _select_diverse(
        self,
        candidates: Sequence[SelectedAnchorCandidate],
        max_count: int,
    ) -> List[SelectedAnchorCandidate]:
        if max_count <= 0:
            return []
        selected: List[SelectedAnchorCandidate] = []
        selected_keys = set()
        descriptors = {
            _carrier_key(candidate.carrier): scene_object_context_descriptor(candidate.carrier)
            for candidate in candidates
        }
        for threshold in self.context_thresholds:
            for candidate in candidates:
                key = _carrier_key(candidate.carrier)
                if key in selected_keys:
                    continue
                desc = descriptors[key]
                if selected:
                    min_dist = min(_l2(desc, descriptors[_carrier_key(item.carrier)]) for item in selected)
                    if min_dist < threshold:
                        continue
                else:
                    min_dist = math.inf
                metadata = dict(candidate.metadata)
                metadata["context_min_distance"] = (
                    None if math.isinf(min_dist) else float(min_dist)
                )
                metadata["context_threshold"] = float(threshold)
                selected.append(SelectedAnchorCandidate(candidate.carrier, metadata))
                selected_keys.add(key)
                if len(selected) >= max_count:
                    return selected
        return selected

    def _report(
        self,
        *,
        candidates: List[SelectedAnchorCandidate],
        max_count: int,
        scanned: int,
        certified: int,
        connected: int,
        saw_uncertainty: bool,
        example_failures: List[Dict[str, Any]],
    ) -> AnchorSelectionReport:
        candidates = sorted(
            candidates,
            key=lambda item: (
                0 if item.metadata.get("candidate_priority") == "event_endpoint" else 1,
                int(item.carrier.index) if item.carrier.index is not None else 10**18,
            ),
        )
        selected = self._select_diverse(candidates, max_count)
        return AnchorSelectionReport(
            selected=selected,
            scanned=scanned,
            certified=certified,
            connected=connected,
            saw_uncertainty=saw_uncertainty,
            example_failures=example_failures,
            metadata={
                "selector": "manipulation_event_context_diverse",
                "context_thresholds": list(self.context_thresholds),
                "num_event_endpoint_candidates": sum(
                    1
                    for item in candidates
                    if item.metadata.get("candidate_priority") == "event_endpoint"
                ),
            },
        )

    def select_proposition(
        self,
        *,
        proposition: str,
        carriers: Sequence[SemanticCarrier],
        max_count: int,
        store: CertificateStore,
        connectivity_fn: Callable[[SemanticCarrier], bool],
    ) -> AnchorSelectionReport:
        predecessors = self._predecessor_map(carriers)
        candidates: List[SelectedAnchorCandidate] = []
        scanned = certified = connected = 0
        saw_uncertainty = False
        example_failures: List[Dict[str, Any]] = []
        for carrier in carriers:
            scanned += 1
            cert = store.carrier_cert(carrier, proposition)
            if not cert.cert_pos:
                saw_uncertainty = saw_uncertainty or bool(cert.uncertain)
                if len(example_failures) < 5:
                    example_failures.append(
                        {
                            "carrier_index": carrier.index,
                            "reason": "uncertain"
                            if cert.uncertain
                            else "missing_positive_certificate",
                        }
                    )
                continue
            certified += 1
            if not connectivity_fn(carrier):
                if len(example_failures) < 5:
                    example_failures.append(
                        {"carrier_index": carrier.index, "reason": "no_connected_anchor"}
                    )
                continue
            connected += 1
            priority = (
                "event_endpoint"
                if _is_event_endpoint(
                    current=carrier,
                    predecessor=predecessors.get(_carrier_key(carrier)),
                    propositions=[proposition],
                    store=store,
                )
                else "predicate_support"
            )
            candidates.append(
                SelectedAnchorCandidate(
                    carrier,
                    {
                        "candidate_priority": priority,
                        "selector_proposition": proposition,
                    },
                )
            )
        return self._report(
            candidates=candidates,
            max_count=max_count,
            scanned=scanned,
            certified=certified,
            connected=connected,
            saw_uncertainty=saw_uncertainty,
            example_failures=example_failures,
        )

    def _guard_candidate_status(
        self,
        carrier: SemanticCarrier,
        guard: GuardTerm,
        store: CertificateStore,
    ):
        cert_pos = set()
        cert_neg = set()
        uncertain = []
        missing_pos = []
        missing_neg = []
        for prop in sorted(guard.pos):
            cert = store.carrier_cert(carrier, prop)
            if cert.cert_pos:
                cert_pos.add(prop)
            elif cert.uncertain:
                uncertain.append(prop)
            else:
                missing_pos.append(prop)
        for prop in sorted(guard.neg):
            cert = store.carrier_cert(carrier, prop)
            if cert.cert_neg:
                cert_neg.add(prop)
            elif cert.uncertain:
                uncertain.append(prop)
            else:
                missing_neg.append(prop)
        if uncertain:
            return False, "blocked_by_uncertainty", cert_pos, cert_neg, uncertain
        if missing_pos:
            return False, "missing_positive_certificate", cert_pos, cert_neg, missing_pos
        if missing_neg:
            return False, "missing_negative_certificate", cert_pos, cert_neg, missing_neg
        return True, "supported", store.cert_pos_set(carrier), store.cert_neg_set(carrier), []

    def select_guard(
        self,
        *,
        guard: GuardTerm,
        carriers: Sequence[SemanticCarrier],
        max_count: int,
        store: CertificateStore,
        connectivity_fn: Callable[[SemanticCarrier], bool],
    ) -> AnchorSelectionReport:
        predecessors = self._predecessor_map(carriers)
        candidates: List[SelectedAnchorCandidate] = []
        scanned = certified = connected = 0
        saw_uncertainty = False
        example_failures: List[Dict[str, Any]] = []
        for carrier in carriers:
            scanned += 1
            ok, reason, cert_pos, cert_neg, props = self._guard_candidate_status(
                carrier,
                guard,
                store,
            )
            if not ok:
                saw_uncertainty = saw_uncertainty or reason == "blocked_by_uncertainty"
                if len(example_failures) < 5:
                    example_failures.append(
                        {
                            "carrier_index": carrier.index,
                            "reason": reason,
                            "propositions": sorted(props),
                        }
                    )
                continue
            certified += 1
            if not connectivity_fn(carrier):
                if len(example_failures) < 5:
                    example_failures.append(
                        {"carrier_index": carrier.index, "reason": "no_connected_anchor"}
                    )
                continue
            connected += 1
            priority = (
                "event_endpoint"
                if _is_event_endpoint(
                    current=carrier,
                    predecessor=predecessors.get(_carrier_key(carrier)),
                    propositions=guard.pos,
                    store=store,
                )
                else "predicate_support"
            )
            candidates.append(
                SelectedAnchorCandidate(
                    carrier,
                    {
                        "candidate_priority": priority,
                        "selector_guard_id": guard.guard_id,
                    },
                )
            )
        return self._report(
            candidates=candidates,
            max_count=max_count,
            scanned=scanned,
            certified=certified,
            connected=connected,
            saw_uncertainty=saw_uncertainty,
            example_failures=example_failures,
        )
