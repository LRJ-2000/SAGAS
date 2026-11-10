"""Guard-term representation for evaluator-conditioned product search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Optional, Set


@dataclass(frozen=True)
class GuardTerm:
    """One conjunctive DNF guard term on user-facing proposition names."""

    guard_id: str
    source_state: Optional[str] = None
    target_state: Optional[str] = None
    pos: FrozenSet[str] = frozenset()
    neg: FrozenSet[str] = frozenset()
    raw_symbol: str = ""
    dnf_index: int = 0
    edge_key: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_unconditional(self) -> bool:
        return not self.pos and not self.neg

    @property
    def needs_witness(self) -> bool:
        return bool(self.pos)

    def to_truth_dict(self) -> Dict[str, bool]:
        truth = {p: True for p in self.pos}
        truth.update({n: False for n in self.neg})
        return truth

    def satisfied_by_sets(self, cert_pos: Iterable[str], cert_neg: Iterable[str]) -> bool:
        pos_set = set(cert_pos)
        neg_set = set(cert_neg)
        return set(self.pos).issubset(pos_set) and set(self.neg).issubset(neg_set)

    @staticmethod
    def from_truth_dict(
        truth: Optional[Dict[str, bool]],
        guard_id: str,
        source_state: Optional[str] = None,
        target_state: Optional[str] = None,
        raw_symbol: str = "",
        dnf_index: int = 0,
        edge_key: Optional[Any] = None,
        literal_to_prop: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GuardTerm":
        truth = dict(truth or {})
        literal_to_prop = literal_to_prop or {}
        pos: Set[str] = set()
        neg: Set[str] = set()
        raw_pos: Set[str] = set()
        raw_neg: Set[str] = set()
        for lit, need in truth.items():
            prop = literal_to_prop.get(lit, lit)
            if need:
                pos.add(prop)
                raw_pos.add(lit)
            else:
                neg.add(prop)
                raw_neg.add(lit)
        md = dict(metadata or {})
        if raw_pos or raw_neg:
            md.setdefault("raw_pos_literals", sorted(raw_pos))
            md.setdefault("raw_neg_literals", sorted(raw_neg))
        return GuardTerm(
            guard_id=guard_id,
            source_state=source_state,
            target_state=target_state,
            pos=frozenset(pos),
            neg=frozenset(neg),
            raw_symbol=raw_symbol,
            dnf_index=dnf_index,
            edge_key=edge_key,
            metadata=md,
        )
