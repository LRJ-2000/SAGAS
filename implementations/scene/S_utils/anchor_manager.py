"""Anchor construction and support diagnostics."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from S_utils.certificate_store import CertificateStore
from S_utils.guard_terms import GuardTerm
from S_utils.semantic_types import (
    AnchorRecord,
    GuardSupportReport,
    PropositionSupportReport,
    SemanticCarrier,
)


class AnchorManager:
    """Construct proposition anchors and guard support maps.

    This first version is intentionally graph-agnostic.  A caller can provide
    `connectivity_fn(carrier) -> bool` to enforce support-backed graph
    connection.  Full KeyGraph insertion will be wired in a later integration
    pass.
    """

    def __init__(
        self,
        certificate_store: CertificateStore,
        connectivity_fn: Optional[Callable[[SemanticCarrier], bool]] = None,
        candidate_selector: Optional[Any] = None,
    ):
        self.store = certificate_store
        self.connectivity_fn = connectivity_fn or (lambda _carrier: True)
        self.candidate_selector = candidate_selector
        self.anchors: Dict[Any, AnchorRecord] = {}
        self.anchors_by_prop: Dict[str, List[Any]] = {}
        self.anchors_by_guard: Dict[str, List[Any]] = {}
        self.anchor_node_to_id: Dict[Any, Any] = {}
        self.prop_reports: Dict[str, PropositionSupportReport] = {}
        self.guard_reports: Dict[str, GuardSupportReport] = {}
        self.selector_reports: Dict[str, Any] = {}
        self._next_anchor_id = 0

    def _new_anchor(
        self,
        carrier: SemanticCarrier,
        cert_pos: set[str],
        cert_neg: set[str],
        prop_ids: Optional[Iterable[str]] = None,
        guard_ids: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AnchorRecord:
        anchor_id = f"a{self._next_anchor_id}"
        self._next_anchor_id += 1
        record = AnchorRecord(
            anchor_id=anchor_id,
            carrier=carrier,
            cert_pos=set(cert_pos),
            cert_neg=set(cert_neg),
            prop_ids=set(prop_ids or []),
            guard_ids=set(guard_ids or []),
            connected=True,
            node_id=anchor_id,
            metadata=dict(metadata or {}),
        )
        self.anchors[anchor_id] = record
        self.anchor_node_to_id[anchor_id] = anchor_id
        for prop in record.prop_ids:
            self.anchors_by_prop.setdefault(prop, []).append(anchor_id)
        for guard_id in record.guard_ids:
            self.anchors_by_guard.setdefault(guard_id, []).append(anchor_id)
        return record

    def attach_graph_node(self, anchor_id: Any, node_id: Any) -> AnchorRecord:
        """Record the graph node that realizes a semantic anchor."""
        if anchor_id not in self.anchors:
            raise KeyError(f"unknown anchor_id: {anchor_id}")
        record = self.anchors[anchor_id]
        record.node_id = node_id
        record.connected = True
        self.anchor_node_to_id[node_id] = anchor_id
        return record

    def anchor_record_for_node(self, node_id: Any) -> Optional[AnchorRecord]:
        """Return anchor metadata when `node_id` is an inserted anchor node."""
        if node_id in self.anchors:
            return self.anchors[node_id]
        anchor_id = self.anchor_node_to_id.get(node_id)
        if anchor_id is None:
            return None
        return self.anchors.get(anchor_id)

    def build_proposition_anchors(
        self,
        carriers: Sequence[SemanticCarrier],
        max_per_prop: int = 10,
    ) -> Dict[str, List[Any]]:
        for prop in self.store.propositions:
            self.anchors_by_prop.setdefault(prop, [])
            if self.candidate_selector is not None:
                report = self.candidate_selector.select_proposition(
                    proposition=prop,
                    carriers=carriers,
                    max_count=max_per_prop,
                    store=self.store,
                    connectivity_fn=self.connectivity_fn,
                )
                self.selector_reports[f"prop:{prop}"] = report
                for candidate in report.selected:
                    carrier = candidate.carrier
                    self._new_anchor(
                        carrier,
                        self.store.cert_pos_set(carrier),
                        self.store.cert_neg_set(carrier),
                        prop_ids={prop},
                        metadata=candidate.metadata,
                    )
                num_selected = len(report.selected)
                if num_selected > 0:
                    status = "supported"
                elif report.certified > 0:
                    status = "no_connected_anchor"
                elif report.saw_uncertainty:
                    status = "blocked_by_uncertainty"
                else:
                    status = "no_candidate"
                self.prop_reports[prop] = PropositionSupportReport(
                    proposition=prop,
                    status=status,
                    num_candidates_scanned=report.scanned,
                    num_certified_candidates=report.certified,
                    num_connected_anchors=num_selected,
                    example_failures=report.example_failures,
                )
                continue

            scanned = 0
            certified = 0
            connected = 0
            saw_uncertainty = False
            example_failures = []
            for carrier in carriers:
                if len(self.anchors_by_prop[prop]) >= max_per_prop:
                    break
                scanned += 1
                cert = self.store.carrier_cert(carrier, prop)
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
                if not self.connectivity_fn(carrier):
                    if len(example_failures) < 5:
                        example_failures.append(
                            {
                                "carrier_index": carrier.index,
                                "reason": "no_connected_anchor",
                            }
                        )
                    continue
                connected += 1
                cert_pos = self.store.cert_pos_set(carrier)
                cert_neg = self.store.cert_neg_set(carrier)
                self._new_anchor(carrier, cert_pos, cert_neg, prop_ids={prop})
            if connected > 0:
                status = "supported"
            elif certified > 0:
                status = "no_connected_anchor"
            elif saw_uncertainty:
                status = "blocked_by_uncertainty"
            else:
                status = "no_candidate"
            self.prop_reports[prop] = PropositionSupportReport(
                proposition=prop,
                status=status,
                num_candidates_scanned=scanned,
                num_certified_candidates=certified,
                num_connected_anchors=connected,
                example_failures=example_failures,
            )
        return self.anchors_by_prop

    def _guard_candidate_status(self, carrier: SemanticCarrier, guard: GuardTerm):
        cert_pos = set()
        cert_neg = set()
        uncertain = []
        missing_pos = []
        missing_neg = []
        for prop in sorted(guard.pos):
            cert = self.store.carrier_cert(carrier, prop)
            if cert.cert_pos:
                cert_pos.add(prop)
            elif cert.uncertain:
                uncertain.append(prop)
            else:
                missing_pos.append(prop)
        for prop in sorted(guard.neg):
            cert = self.store.carrier_cert(carrier, prop)
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
        return True, "supported", self.store.cert_pos_set(carrier), self.store.cert_neg_set(carrier), []

    def build_guard_anchors(
        self,
        guard_terms: Sequence[GuardTerm],
        carriers: Sequence[SemanticCarrier],
        max_per_guard: int = 10,
    ) -> Dict[str, List[Any]]:
        for guard in guard_terms:
            if not guard.pos:
                self.guard_reports[guard.guard_id] = GuardSupportReport(
                    guard_id=guard.guard_id,
                    source_transition=(guard.source_state, guard.target_state),
                    pos=guard.pos,
                    neg=guard.neg,
                    status="not_anchor_required",
                )
                continue

            scanned = 0
            certified = 0
            connected = 0
            saw_uncertainty = False
            example_failures = []
            self.anchors_by_guard.setdefault(guard.guard_id, [])
            if self.candidate_selector is not None:
                report = self.candidate_selector.select_guard(
                    guard=guard,
                    carriers=carriers,
                    max_count=max_per_guard,
                    store=self.store,
                    connectivity_fn=self.connectivity_fn,
                )
                self.selector_reports[f"guard:{guard.guard_id}"] = report
                for candidate in report.selected:
                    carrier = candidate.carrier
                    record = self._new_anchor(
                        carrier,
                        self.store.cert_pos_set(carrier),
                        self.store.cert_neg_set(carrier),
                        prop_ids=set(guard.pos),
                        guard_ids={guard.guard_id},
                        metadata=candidate.metadata,
                    )
                    for prop in guard.pos:
                        if record.anchor_id not in self.anchors_by_prop.setdefault(prop, []):
                            self.anchors_by_prop[prop].append(record.anchor_id)
                num_selected = len(report.selected)
                if num_selected > 0:
                    status = "supported"
                elif report.certified > 0:
                    status = "no_connected_anchor"
                elif report.saw_uncertainty:
                    status = "blocked_by_uncertainty"
                else:
                    status = "no_candidate"
                self.guard_reports[guard.guard_id] = GuardSupportReport(
                    guard_id=guard.guard_id,
                    source_transition=(guard.source_state, guard.target_state),
                    pos=guard.pos,
                    neg=guard.neg,
                    status=status,
                    num_candidates_scanned=report.scanned,
                    num_certified_candidates=report.certified,
                    num_connected_anchors=num_selected,
                    example_failures=report.example_failures,
                )
                continue

            for carrier in carriers:
                if len(self.anchors_by_guard[guard.guard_id]) >= max_per_guard:
                    break
                scanned += 1
                ok, reason, cert_pos, cert_neg, props = self._guard_candidate_status(
                    carrier,
                    guard,
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
                if not self.connectivity_fn(carrier):
                    if len(example_failures) < 5:
                        example_failures.append(
                            {
                                "carrier_index": carrier.index,
                                "reason": "no_connected_anchor",
                            }
                        )
                    continue
                connected += 1
                record = self._new_anchor(
                    carrier,
                    cert_pos,
                    cert_neg,
                    prop_ids=set(guard.pos),
                    guard_ids={guard.guard_id},
                )
                for prop in guard.pos:
                    if record.anchor_id not in self.anchors_by_prop.setdefault(prop, []):
                        self.anchors_by_prop[prop].append(record.anchor_id)

            if connected > 0:
                status = "supported"
            elif certified > 0:
                status = "no_connected_anchor"
            elif saw_uncertainty:
                status = "blocked_by_uncertainty"
            else:
                status = "no_candidate"
            self.guard_reports[guard.guard_id] = GuardSupportReport(
                guard_id=guard.guard_id,
                source_transition=(guard.source_state, guard.target_state),
                pos=guard.pos,
                neg=guard.neg,
                status=status,
                num_candidates_scanned=scanned,
                num_certified_candidates=certified,
                num_connected_anchors=connected,
                example_failures=example_failures,
            )
        return self.anchors_by_guard
