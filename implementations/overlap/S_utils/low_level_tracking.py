"""Low-level waypoint-direction utilities for evaluator-conditioned SAGAS.

The real policy still receives raw observations and a latent direction vector.
This module only computes that direction, including the guard-aware semantic
steering term used before querying the frozen low-level executor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set


Vector = Sequence[float]


@dataclass(frozen=True)
class LowLevelTrackingConfig:
    """Configuration for latent-direction tracking and semantic steering."""

    direction_smoothing_beta: float = 0.0
    semantic_repulsion_enable: bool = True
    semantic_repulsion_radius: float = 1.0
    semantic_repulsion_gain: float = 0.3
    semantic_repulsion_power: float = 1.0
    semantic_repulsion_max_norm: float = 0.5
    epsilon: float = 1e-8


@dataclass
class TrackingCommand:
    """Computed policy input for one low-level waypoint-tracking step."""

    skill: List[float]
    next_prev_skill: Optional[List[float]]
    base_delta: List[float]
    repulsion_delta: List[float]
    command_delta: List[float]
    active_forbidden_labels: Set[str] = field(default_factory=set)

    @property
    def repulsion_active(self) -> bool:
        return vector_norm(self.repulsion_delta) > 0.0


@dataclass(frozen=True)
class WaypointTargetSelection:
    """Target chosen by the monotone waypoint tracker."""

    target_index: int
    target_latent: List[float]
    locally_reachable: bool
    skipped_indices: List[int] = field(default_factory=list)
    blocked_by_checkpoint: Optional[int] = None


def as_vector(vec: Vector) -> List[float]:
    """Convert a vector-like object to a flat float list."""

    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if isinstance(vec, (int, float)):
        return [float(vec)]
    return [float(x) for x in vec]


def vector_norm(vec: Vector) -> float:
    vv = as_vector(vec)
    return math.sqrt(sum(x * x for x in vv))


def _check_same_dim(a: Vector, b: Vector) -> tuple[List[float], List[float]]:
    aa = as_vector(a)
    bb = as_vector(b)
    if len(aa) != len(bb):
        raise ValueError("vectors must have the same dimensionality")
    return aa, bb


def vector_sub(a: Vector, b: Vector) -> List[float]:
    aa, bb = _check_same_dim(a, b)
    return [x - y for x, y in zip(aa, bb)]


def vector_add(a: Vector, b: Vector) -> List[float]:
    aa, bb = _check_same_dim(a, b)
    return [x + y for x, y in zip(aa, bb)]


def vector_scale(vec: Vector, scale: float) -> List[float]:
    return [float(scale) * x for x in as_vector(vec)]


def direction_to_skill(
    delta: Vector,
    prev_skill: Optional[Vector] = None,
    smoothing_beta: float = 0.0,
    epsilon: float = 1e-8,
) -> tuple[List[float], Optional[List[float]]]:
    """Convert a latent displacement into the unit skill used by the policy."""

    delta_v = as_vector(delta)
    norm = vector_norm(delta_v)
    if norm <= epsilon:
        return [0.0 for _ in delta_v], prev_skill and as_vector(prev_skill)

    skill = vector_scale(delta_v, 1.0 / (norm + epsilon))
    beta = max(0.0, min(float(smoothing_beta), 0.99))
    if beta > 0.0 and prev_skill is not None:
        mixed = vector_add(vector_scale(skill, 1.0 - beta), vector_scale(prev_skill, beta))
        mixed_norm = vector_norm(mixed)
        if mixed_norm > epsilon:
            skill = vector_scale(mixed, 1.0 / (mixed_norm + epsilon))
    return skill, list(skill)


def active_forbidden_labels(
    edge_forbidden_labels: Sequence[Iterable[str]],
    progress_idx: int,
    target_idx: int,
    use_all: bool = False,
) -> Set[str]:
    """Collect active negative labels for the currently commanded segment."""

    if not edge_forbidden_labels:
        return set()
    if use_all:
        edge_range = edge_forbidden_labels
    else:
        lo = max(0, int(progress_idx))
        hi = min(len(edge_forbidden_labels), int(target_idx))
        if hi <= lo:
            return set()
        edge_range = edge_forbidden_labels[lo:hi]

    labels: Set[str] = set()
    for edge_labels in edge_range:
        labels.update(edge_labels or [])
    return labels


def _first_unverified_checkpoint_between(
    current_index: int,
    target_index: int,
    semantic_checkpoint_indices: Iterable[int],
    verified_checkpoint_indices: Optional[Iterable[int]] = None,
) -> Optional[int]:
    verified = set(int(i) for i in (verified_checkpoint_indices or []))
    for checkpoint in sorted(int(i) for i in semantic_checkpoint_indices):
        if int(current_index) < checkpoint < int(target_index) and checkpoint not in verified:
            return checkpoint
    return None


def select_tracking_target(
    *,
    current_latent: Vector,
    waypoint_latents: Sequence[Vector],
    current_index: int,
    local_reachability_radius: float,
    semantic_checkpoint_indices: Iterable[int] = (),
    verified_checkpoint_indices: Optional[Iterable[int]] = None,
) -> WaypointTargetSelection:
    """Select the farthest locally reachable waypoint without bypassing checks."""

    if not waypoint_latents:
        raise ValueError("waypoint_latents must be nonempty")
    idx0 = max(0, min(int(current_index), len(waypoint_latents) - 1))
    current_v = as_vector(current_latent)
    target_idx = idx0
    blocked_by = None
    checkpoints = set(int(i) for i in semantic_checkpoint_indices)
    verified = set(int(i) for i in (verified_checkpoint_indices or []))
    locally_reachable = vector_norm(vector_sub(waypoint_latents[idx0], current_v)) <= float(
        local_reachability_radius
    )

    if idx0 in checkpoints and idx0 not in verified:
        return WaypointTargetSelection(
            target_index=idx0,
            target_latent=as_vector(waypoint_latents[idx0]),
            locally_reachable=bool(locally_reachable),
            skipped_indices=[],
            blocked_by_checkpoint=idx0,
        )

    for idx in range(idx0 + 1, len(waypoint_latents)):
        blocked = _first_unverified_checkpoint_between(
            idx0,
            idx,
            checkpoints,
            verified,
        )
        if blocked is not None:
            blocked_by = blocked
            break
        dist = vector_norm(vector_sub(waypoint_latents[idx], current_v))
        if dist <= float(local_reachability_radius):
            target_idx = idx
            locally_reachable = True

    return WaypointTargetSelection(
        target_index=target_idx,
        target_latent=as_vector(waypoint_latents[target_idx]),
        locally_reachable=bool(locally_reachable),
        skipped_indices=list(range(idx0, target_idx)),
        blocked_by_checkpoint=blocked_by,
    )


def semantic_repulsion_delta(
    phi_obs: Vector,
    forbidden_labels: Iterable[str],
    anchor_positions_by_label: Dict[str, Sequence[Vector]],
    radius: float,
    gain: float,
    power: float,
    max_norm: float,
    epsilon: float = 1e-8,
) -> List[float]:
    """Compute bounded latent repulsion from anchors of forbidden labels."""

    phi_v = as_vector(phi_obs)
    if not forbidden_labels or radius <= 0.0 or gain <= 0.0:
        return [0.0 for _ in phi_v]

    repulsion = [0.0 for _ in phi_v]
    for label in sorted(set(forbidden_labels)):
        anchors = anchor_positions_by_label.get(label, [])
        nearest_diff: Optional[List[float]] = None
        nearest_dist = math.inf
        for anchor in anchors:
            diff = vector_sub(phi_v, anchor)
            dist = vector_norm(diff)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_diff = diff
        if nearest_diff is None or nearest_dist >= radius or nearest_dist <= epsilon:
            continue

        scale = float(gain) * max(0.0, 1.0 - nearest_dist / radius) ** float(power)
        repulsion = vector_add(repulsion, vector_scale(nearest_diff, scale / (nearest_dist + epsilon)))

    rep_norm = vector_norm(repulsion)
    if max_norm > 0.0 and rep_norm > max_norm:
        repulsion = vector_scale(repulsion, float(max_norm) / (rep_norm + epsilon))
    return repulsion


def compute_tracking_command(
    *,
    phi_obs: Vector,
    target_latent: Vector,
    active_negatives: Iterable[str] = (),
    anchor_positions_by_label: Optional[Dict[str, Sequence[Vector]]] = None,
    prev_skill: Optional[Vector] = None,
    config: Optional[LowLevelTrackingConfig] = None,
) -> TrackingCommand:
    """Compute the latent direction passed to the frozen low-level policy."""

    cfg = config or LowLevelTrackingConfig()
    base_delta = vector_sub(target_latent, phi_obs)
    active_neg = set(active_negatives)
    repulsion = [0.0 for _ in base_delta]
    if cfg.semantic_repulsion_enable:
        repulsion = semantic_repulsion_delta(
            phi_obs,
            active_neg,
            anchor_positions_by_label or {},
            radius=float(cfg.semantic_repulsion_radius),
            gain=float(cfg.semantic_repulsion_gain),
            power=float(cfg.semantic_repulsion_power),
            max_norm=float(cfg.semantic_repulsion_max_norm),
            epsilon=float(cfg.epsilon),
        )
    command_delta = vector_add(base_delta, repulsion)
    skill, next_prev = direction_to_skill(
        command_delta,
        prev_skill=prev_skill,
        smoothing_beta=float(cfg.direction_smoothing_beta),
        epsilon=float(cfg.epsilon),
    )
    return TrackingCommand(
        skill=skill,
        next_prev_skill=next_prev,
        base_delta=base_delta,
        repulsion_delta=repulsion,
        command_delta=command_delta,
        active_forbidden_labels=active_neg,
    )
