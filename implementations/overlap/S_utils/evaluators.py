"""Semantic evaluator implementations for task-time grounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


from S_utils.manipulation_predicates import ObjectStateEvaluator
from S_utils.semantic_types import PredicateCertificate, SemanticCarrier


class SemanticEvaluator:
    """Base interface for task-time proposition evaluators."""

    def evaluate(self, carrier: SemanticCarrier, proposition: str) -> PredicateCertificate:
        raise NotImplementedError

    def certified_positive_set(
        self, carrier: SemanticCarrier, propositions: Iterable[str]
    ) -> set[str]:
        return {
            prop
            for prop in propositions
            if self.evaluate(carrier, prop).cert_pos
        }

    def certified_negative_set(
        self, carrier: SemanticCarrier, propositions: Iterable[str]
    ) -> set[str]:
        return {
            prop
            for prop in propositions
            if self.evaluate(carrier, prop).cert_neg
        }


@dataclass(frozen=True)
class RegionSpec:
    """State-space geometric predicate specification."""

    proposition: str
    kind: str
    params: Dict[str, Any]


def _point_from_carrier(
    carrier: SemanticCarrier,
    state_to_point: Callable[[Any], Sequence[float]],
    *,
    require_state: bool = False,
) -> Tuple[float, float]:
    if carrier.state is None and require_state:
        raise ValueError(
            "Oracle state-lift evaluation requires `SemanticCarrier.state`; "
            f"carrier source={carrier.source!r}, index={carrier.index!r}."
        )
    state = carrier.state if carrier.state is not None else carrier.observation
    point = state_to_point(state)
    if len(point) < 2:
        raise ValueError("state_to_point must return at least two coordinates")
    return float(point[0]), float(point[1])


def _point_in_rect(point: Tuple[float, float], params: Dict[str, Any]) -> bool:
    x, y = point
    if "bounds" in params:
        xmin, xmax, ymin, ymax = params["bounds"]
    else:
        xmin = params["xmin"]
        xmax = params["xmax"]
        ymin = params["ymin"]
        ymax = params["ymax"]
    return xmin <= x <= xmax and ymin <= y <= ymax


def _point_in_circle(point: Tuple[float, float], params: Dict[str, Any]) -> bool:
    x, y = point
    cx, cy = params.get("center", (params.get("cx"), params.get("cy")))
    radius = params.get("radius", params.get("r"))
    return (x - cx) ** 2 + (y - cy) ** 2 <= float(radius) ** 2


def _point_on_segment(px, py, ax, ay, bx, by, eps=1e-9) -> bool:
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= eps


def _point_in_polygon(point: Tuple[float, float], vertices: Sequence[Sequence[float]]) -> bool:
    """Ray-casting polygon containment with boundary included."""
    x, y = point
    verts = [(float(v[0]), float(v[1])) for v in vertices]
    if len(verts) < 3:
        return False
    inside = False
    j = len(verts) - 1
    for i in range(len(verts)):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if _point_on_segment(x, y, xi, yi, xj, yj):
            return True
        intersects = ((yi > y) != (yj > y)) and (
            x <= (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _contains(point: Tuple[float, float], spec: RegionSpec) -> bool:
    kind = spec.kind.lower()
    if kind in {"rect", "rectangle", "box"}:
        return _point_in_rect(point, spec.params)
    if kind in {"circle", "disk"}:
        return _point_in_circle(point, spec.params)
    if kind in {"polygon", "poly"}:
        return _point_in_polygon(point, spec.params["vertices"])
    raise ValueError(f"Unsupported region kind: {spec.kind}")


class StateRegionEvaluator(SemanticEvaluator):
    """Deterministic evaluator for state-space geometric predicates."""

    def __init__(
        self,
        region_specs: Sequence[RegionSpec | Dict[str, Any]],
        state_to_point: Callable[[Any], Sequence[float]] = lambda s: s[:2],
    ):
        self.state_to_point = state_to_point
        specs: List[RegionSpec] = []
        for spec in region_specs:
            if isinstance(spec, RegionSpec):
                specs.append(spec)
            else:
                proposition = spec.get("proposition", spec.get("id", spec.get("name")))
                if proposition is None:
                    raise ValueError("Region spec must define proposition/id/name")
                params = dict(spec.get("params", {}))
                for key, value in spec.items():
                    if key not in {"proposition", "id", "name", "kind", "type", "params"}:
                        params.setdefault(key, value)
                specs.append(
                    RegionSpec(
                        proposition=str(proposition),
                        kind=str(spec.get("kind", spec.get("type"))),
                        params=params,
                    )
                )
        self.region_specs = {spec.proposition: spec for spec in specs}

    @property
    def propositions(self) -> List[str]:
        return list(self.region_specs.keys())

    def evaluate(self, carrier: SemanticCarrier, proposition: str) -> PredicateCertificate:
        if proposition not in self.region_specs:
            return PredicateCertificate.negative(
                proposition,
                metadata={"reason": "unknown_proposition"},
            )
        point = _point_from_carrier(carrier, self.state_to_point)
        inside = _contains(point, self.region_specs[proposition])
        metadata = {"point": point, "evaluator": "StateRegionEvaluator"}
        if inside:
            return PredicateCertificate.positive(proposition, metadata=metadata)
        return PredicateCertificate.negative(proposition, metadata=metadata)


class OracleStateLiftEvaluator(StateRegionEvaluator):
    """Region evaluator that intentionally ignores deployable observations.

    This evaluator is used for observation-based settings where the policy sees
    images but the semantic evaluator is supplied with privileged task state by
    the environment/evaluator contract.  It fails fast when the carrier has no
    explicit state instead of silently interpreting an image as a task state.
    """

    def evaluate(self, carrier: SemanticCarrier, proposition: str) -> PredicateCertificate:
        if proposition not in self.region_specs:
            return PredicateCertificate.negative(
                proposition,
                metadata={"reason": "unknown_proposition"},
            )
        point = _point_from_carrier(
            carrier,
            self.state_to_point,
            require_state=True,
        )
        inside = _contains(point, self.region_specs[proposition])
        metadata = {"point": point, "evaluator": "OracleStateLiftEvaluator"}
        if inside:
            return PredicateCertificate.positive(proposition, metadata=metadata)
        return PredicateCertificate.negative(proposition, metadata=metadata)
