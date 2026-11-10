"""Evaluator-conditioned product successor oracle."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from S_utils.anchor_manager import AnchorManager
from S_utils.certificate_store import CertificateStore
from S_utils.guard_terms import GuardTerm


class ProductOracle:
    """Pure-Python successor oracle for semantic product transitions.

    The oracle expects graph nodes to have semantic certificates available
    through `node_cert_lookup(node, proposition)` and anchor metadata through an
    `AnchorManager`.
    """

    def __init__(
        self,
        certificate_store: CertificateStore,
        anchor_manager: AnchorManager,
        node_support_lookup=None,
        node_cert_lookup=None,
    ):
        self.store = certificate_store
        self.anchor_manager = anchor_manager
        self.node_support_lookup = node_support_lookup
        self.node_cert_lookup = node_cert_lookup
        self.block_counts: Dict[str, int] = {}
        self.block_counts_by_guard: Dict[str, Dict[str, int]] = {}
        self.blocked_guard_terms: set[str] = set()
        self.block_samples: List[Dict[str, Any]] = []

    def _record_block(
        self,
        guard: GuardTerm,
        reason: str,
        *,
        node_id: Any = None,
        proposition: Optional[str] = None,
    ) -> None:
        self.block_counts[reason] = self.block_counts.get(reason, 0) + 1
        self.blocked_guard_terms.add(guard.guard_id)
        by_reason = self.block_counts_by_guard.setdefault(guard.guard_id, {})
        by_reason[reason] = by_reason.get(reason, 0) + 1
        if len(self.block_samples) < 32:
            sample = {
                "guard_id": guard.guard_id,
                "reason": reason,
                "node_id": node_id,
            }
            if proposition is not None:
                sample["proposition"] = proposition
            self.block_samples.append(sample)

    def _anchor_record_for_node(self, node_id: Any):
        if hasattr(self.anchor_manager, "anchor_record_for_node"):
            return self.anchor_manager.anchor_record_for_node(node_id)
        if node_id in self.anchor_manager.anchors:
            return self.anchor_manager.anchors[node_id]
        return None

    def _node_neg_status(self, node_id: Any, proposition: str) -> Tuple[bool, str]:
        anchor = self._anchor_record_for_node(node_id)
        if anchor is not None:
            if proposition in anchor.cert_neg:
                return True, "certified_negative"
            return False, "anchor_missing_negative_certificate"
        if self.node_cert_lookup is not None:
            cert = self.node_cert_lookup(node_id, proposition)
            if bool(cert.cert_neg):
                return True, "certified_negative"
            if bool(getattr(cert, "uncertain", False)):
                return False, "negative_uncertainty_block"
            if bool(getattr(cert, "cert_pos", False)):
                return False, "negative_positive_certificate"
            return False, "negative_not_certified"
        if self.node_support_lookup is None:
            return False, "missing_node_support"
        cert = self.store.aggregate_node(
            node_id,
            self.node_support_lookup(node_id),
            proposition,
        )
        if bool(cert.cert_neg):
            return True, "certified_negative"
        if bool(cert.uncertain):
            return False, "negative_uncertainty_block"
        if bool(cert.cert_pos):
            return False, "negative_positive_certificate"
        return False, "negative_not_certified"

    def _node_cert_neg(self, node_id: Any, proposition: str) -> bool:
        ok, _reason = self._node_neg_status(node_id, proposition)
        return ok

    def neg_ok(self, node_id: Any, guard: GuardTerm) -> bool:
        for prop in guard.neg:
            ok, reason = self._node_neg_status(node_id, prop)
            if not ok:
                self._record_block(guard, reason, node_id=node_id, proposition=prop)
                return False
        return True

    def witness_ok(self, node_id: Any, guard: GuardTerm) -> bool:
        if not guard.pos:
            return True
        anchor = self._anchor_record_for_node(node_id)
        if anchor is None:
            self._record_block(guard, "missing_positive_anchor", node_id=node_id)
            return False
        ok = anchor.anchor_id in set(
            self.anchor_manager.anchors_by_guard.get(guard.guard_id, [])
        )
        if not ok:
            self._record_block(guard, "anchor_not_guard_witness", node_id=node_id)
            return False
        if not set(guard.pos).issubset(anchor.cert_pos):
            self._record_block(guard, "positive_anchor_missing_certificate", node_id=node_id)
            return False
        return ok

    def enable(self, node_id: Any, guard: GuardTerm) -> bool:
        return self.witness_ok(node_id, guard) and self.neg_ok(node_id, guard)

    def diagnostics_summary(self) -> Dict[str, Any]:
        """Return accumulated product-transition block diagnostics."""

        return {
            "block_counts": dict(sorted(self.block_counts.items())),
            "block_counts_by_guard": {
                guard_id: dict(sorted(counts.items()))
                for guard_id, counts in sorted(self.block_counts_by_guard.items())
            },
            "blocked_guard_terms": sorted(self.blocked_guard_terms),
            "negative_uncertainty_blocks": int(
                self.block_counts.get("negative_uncertainty_block", 0)
            ),
            "block_samples": list(self.block_samples),
        }

    def successors(
        self,
        graph,
        buchi_runtime,
        product_state: Tuple[Any, Any],
        edge_weight_fn=None,
    ) -> Iterator[Tuple[Tuple[Any, Any], float, GuardTerm]]:
        u, q = product_state
        for v in graph.successors(u):
            for qn, guard in buchi_runtime.enabled_guard_successors(q):
                if not self.enable(v, guard):
                    continue
                if edge_weight_fn is not None:
                    cost = float(edge_weight_fn(u, v))
                else:
                    cost = float(graph[u][v].get("weight", 1.0))
                yield (v, qn), cost, guard
