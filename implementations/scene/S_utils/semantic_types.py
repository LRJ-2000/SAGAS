"""Shared semantic-layer data structures.

These types are intentionally pure Python.  They are used by the task-time
semantic layer and should not be passed directly to GAS/JAX policy networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass
class SemanticCarrier:
    """A semantic-layer wrapper around deployable observations and metadata."""

    observation: Any
    state: Any = None
    info: Dict[str, Any] = field(default_factory=dict)
    index: Optional[int] = None
    source: str = "unknown"


@dataclass(frozen=True)
class PredicateCertificate:
    """Evaluator output for a single proposition on a carrier."""

    proposition: str
    score: float
    lower: float
    upper: float
    margin: float
    cert_pos: bool
    cert_neg: bool
    uncertain: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    @staticmethod
    def positive(proposition: str, metadata: Optional[Dict[str, Any]] = None):
        return PredicateCertificate(
            proposition=proposition,
            score=1.0,
            lower=1.0,
            upper=1.0,
            margin=1.0,
            cert_pos=True,
            cert_neg=False,
            uncertain=False,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def negative(proposition: str, metadata: Optional[Dict[str, Any]] = None):
        return PredicateCertificate(
            proposition=proposition,
            score=0.0,
            lower=0.0,
            upper=0.0,
            margin=1.0,
            cert_pos=False,
            cert_neg=True,
            uncertain=False,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def uncertain_value(
        proposition: str,
        score: float,
        lower: float,
        upper: float,
        margin: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        return PredicateCertificate(
            proposition=proposition,
            score=float(score),
            lower=float(lower),
            upper=float(upper),
            margin=float(margin),
            cert_pos=False,
            cert_neg=False,
            uncertain=True,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class NodeCertificate:
    """Aggregated certificate for one graph node and one proposition."""

    node_id: Any
    proposition: str
    score: float
    lower: float
    upper: float
    cert_pos: bool
    cert_neg: bool
    uncertain: bool
    support_size: int
    effective_support_size: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnchorRecord:
    """Semantic metadata for an inserted anchor or landmark."""

    anchor_id: Any
    carrier: SemanticCarrier
    cert_pos: Set[str] = field(default_factory=set)
    cert_neg: Set[str] = field(default_factory=set)
    guard_ids: Set[str] = field(default_factory=set)
    prop_ids: Set[str] = field(default_factory=set)
    connected: bool = False
    node_id: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardSupportReport:
    """Diagnostics for proposition/guard anchor support."""

    guard_id: str
    source_transition: Optional[Tuple[Any, Any]] = None
    pos: FrozenSet[str] = frozenset()
    neg: FrozenSet[str] = frozenset()
    status: str = "unknown"
    num_candidates_scanned: int = 0
    num_certified_candidates: int = 0
    num_connected_anchors: int = 0
    example_failures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PropositionSupportReport:
    """Diagnostics for proposition-level anchor support."""

    proposition: str
    status: str = "unknown"
    num_candidates_scanned: int = 0
    num_certified_candidates: int = 0
    num_connected_anchors: int = 0
    example_failures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RolloutTrace:
    """Rollout data split into deployable observations and oracle states."""

    observations: List[Any] = field(default_factory=list)
    states: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    evaluator_labels: List[Set[str]] = field(default_factory=list)
    oracle_labels: List[Set[str]] = field(default_factory=list)
    product_progress: List[Any] = field(default_factory=list)
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
