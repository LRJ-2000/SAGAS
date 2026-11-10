"""Certificate cache and node-level aggregation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from S_utils.evaluators import SemanticEvaluator
from S_utils.semantic_types import (
    NodeCertificate,
    PredicateCertificate,
    SemanticCarrier,
)


class CertificateStore:
    """Cache evaluator outputs and aggregate graph-node support certificates."""

    def __init__(
        self,
        evaluator: SemanticEvaluator,
        propositions: Iterable[str],
        tau_pos: float = 0.99,
        tau_neg: float = 0.01,
        min_support: int = 1,
    ):
        self.evaluator = evaluator
        self.propositions = list(propositions)
        self.tau_pos = float(tau_pos)
        self.tau_neg = float(tau_neg)
        self.min_support = int(min_support)
        self._carrier_cache: Dict[Tuple[Any, str], PredicateCertificate] = {}
        self._node_cache: Dict[Tuple[Any, str], NodeCertificate] = {}

    def carrier_cert(
        self,
        carrier: SemanticCarrier,
        proposition: str,
    ) -> PredicateCertificate:
        if carrier.index is None:
            cache_id = ("object", id(carrier))
        else:
            cache_id = (carrier.source, int(carrier.index))
        key = (cache_id, proposition)
        if key not in self._carrier_cache:
            self._carrier_cache[key] = self.evaluator.evaluate(carrier, proposition)
        return self._carrier_cache[key]

    def cert_pos_set(self, carrier: SemanticCarrier) -> set[str]:
        return {
            prop
            for prop in self.propositions
            if self.carrier_cert(carrier, prop).cert_pos
        }

    def cert_neg_set(self, carrier: SemanticCarrier) -> set[str]:
        return {
            prop
            for prop in self.propositions
            if self.carrier_cert(carrier, prop).cert_neg
        }

    def aggregate_node(
        self,
        node_id: Any,
        support: Sequence[SemanticCarrier],
        proposition: str,
        weights: Optional[Sequence[float]] = None,
    ) -> NodeCertificate:
        key = (node_id, proposition)
        if key in self._node_cache and weights is None:
            return self._node_cache[key]

        support = list(support)
        if weights is None:
            weights_arr = [1.0 for _ in support]
        else:
            weights_arr = [float(w) for w in weights]
            if len(weights_arr) != len(support):
                raise ValueError("weights must match support length")

        weight_sum = float(sum(weights_arr))
        if len(support) == 0 or weight_sum <= 0:
            node_cert = NodeCertificate(
                node_id=node_id,
                proposition=proposition,
                score=0.0,
                lower=0.0,
                upper=1.0,
                cert_pos=False,
                cert_neg=False,
                uncertain=True,
                support_size=len(support),
                effective_support_size=0.0,
                metadata={"reason": "empty_or_zero_weight_support"},
            )
            self._node_cache[key] = node_cert
            return node_cert

        certs = [self.carrier_cert(carrier, proposition) for carrier in support]
        denom = weight_sum
        score = float(sum(w * c.score for w, c in zip(weights_arr, certs)) / denom)
        lower = float(sum(w * c.lower for w, c in zip(weights_arr, certs)) / denom)
        upper = float(sum(w * c.upper for w, c in zip(weights_arr, certs)) / denom)
        eff_denom = sum(w * w for w in weights_arr)
        eff = float(denom * denom / eff_denom) if eff_denom > 0 else 0.0
        enough_support = eff >= self.min_support
        cert_pos = bool(enough_support and lower >= self.tau_pos)
        cert_neg = bool(enough_support and upper <= self.tau_neg)
        node_cert = NodeCertificate(
            node_id=node_id,
            proposition=proposition,
            score=score,
            lower=lower,
            upper=upper,
            cert_pos=cert_pos,
            cert_neg=cert_neg,
            uncertain=not cert_pos and not cert_neg,
            support_size=len(support),
            effective_support_size=eff,
            metadata={
                "tau_pos": self.tau_pos,
                "tau_neg": self.tau_neg,
                "min_support": self.min_support,
            },
        )
        if weights is None:
            self._node_cache[key] = node_cert
        return node_cert

    def aggregate_all_for_node(
        self,
        node_id: Any,
        support: Sequence[SemanticCarrier],
        weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, NodeCertificate]:
        return {
            prop: self.aggregate_node(node_id, support, prop, weights=weights)
            for prop in self.propositions
        }
