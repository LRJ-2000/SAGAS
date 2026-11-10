"""Semantic graph insertion helpers.

This module bridges the evaluator-conditioned semantic layer with the
conference-version KeyGraph object.  It keeps the low-level graph representation
unchanged while attaching task-time semantic nodes and metadata needed by the
product oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from S_utils.anchor_manager import AnchorManager
from S_utils.certificate_store import CertificateStore
from S_utils.semantic_types import AnchorRecord, SemanticCarrier

try:  # Optional locally; available in the full SAGAS environment.
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - exercised on minimal Python.
    _np = None


@dataclass
class InsertedSemanticNode:
    """Result of adding a task-time semantic node to a reachability graph."""

    node_id: Any
    position: Any
    connection_status: str
    neighbors: List[Any] = field(default_factory=list)
    min_distance: float = math.inf
    metadata: Dict[str, Any] = field(default_factory=dict)


def _as_list(vec: Any) -> List[float]:
    if _np is not None:
        try:
            arr = _np.asarray(vec, dtype=float).reshape(-1)
            return [float(x) for x in arr]
        except Exception:
            pass
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if isinstance(vec, (int, float)):
        return [float(vec)]
    return [float(x) for x in vec]


def latent_l2(a: Any, b: Any) -> float:
    """Pure-Python L2 distance used for graph-local semantic insertion."""
    aa = _as_list(a)
    bb = _as_list(b)
    if len(aa) != len(bb):
        raise ValueError("latent vectors must have the same dimensionality")
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(aa, bb)))


class KeyGraphSemanticAugmentor:
    """Insert anchors and start nodes into a KeyGraph-like object.

    The object is intentionally thin: it assumes the graph already stores
    dataset-supported local connectivity and only adds task-time semantic nodes
    when they can be connected under the same temporal-distance scale.  Anchors
    never use nearest-neighbor fallback; the start node may use fallback because
    it only aligns the current rollout state with the graph.
    """

    def __init__(
        self,
        key_graph: Any,
        get_phi: Callable[[Any], Any],
        certificate_store: Optional[CertificateStore] = None,
        cutoff: Optional[float] = None,
    ):
        self.key_graph = key_graph
        self.get_phi = get_phi
        self.store = certificate_store
        self.cutoff = float(cutoff if cutoff is not None else key_graph.way_steps)

    def embed(self, carrier_or_observation: Any) -> Any:
        observation = (
            carrier_or_observation.observation
            if isinstance(carrier_or_observation, SemanticCarrier)
            else carrier_or_observation
        )
        return self.get_phi(observation)

    def node_positions(self, base_only: bool = False) -> List[Any]:
        nodes = self.key_graph.nodes
        limit = len(nodes)
        if base_only:
            limit = int(getattr(self.key_graph, "base_node_cnt", limit))
        return [nodes[i] for i in range(limit)]

    def local_neighbors(
        self,
        position: Any,
        base_only: bool = False,
        cutoff: Optional[float] = None,
    ) -> List[Any]:
        threshold = float(cutoff if cutoff is not None else self.cutoff)
        neighbors = []
        for node_id, node_pos in enumerate(self.node_positions(base_only=base_only)):
            if latent_l2(position, node_pos) <= threshold:
                neighbors.append(node_id)
        return neighbors

    def nearest_neighbor(self, position: Any, base_only: bool = False) -> Optional[Any]:
        best_node = None
        best_dist = math.inf
        for node_id, node_pos in enumerate(self.node_positions(base_only=base_only)):
            dist = latent_l2(position, node_pos)
            if dist < best_dist:
                best_dist = dist
                best_node = node_id
        return best_node

    def min_distance_to_graph(self, position: Any, base_only: bool = False) -> float:
        distances = [
            latent_l2(position, node_pos)
            for node_pos in self.node_positions(base_only=base_only)
        ]
        return min(distances) if distances else math.inf

    def has_local_neighbor(
        self,
        carrier_or_observation: Any,
        base_only: bool = True,
    ) -> bool:
        return bool(
            self.local_neighbors(
                self.embed(carrier_or_observation),
                base_only=base_only,
            )
        )

    def _next_node_id(self) -> int:
        return int(self.key_graph.graph.number_of_nodes())

    def _append_position(self, position: Any) -> Any:
        if _np is not None and hasattr(self.key_graph.nodes, "shape"):
            pos_arr = _np.asarray(position, dtype=float).reshape(1, -1)
            self.key_graph.nodes = _np.vstack([self.key_graph.nodes, pos_arr])
            return pos_arr.reshape(-1)
        pos_list = _as_list(position)
        self.key_graph.nodes = list(self.key_graph.nodes) + [pos_list]
        return pos_list

    def _add_node_with_edges(
        self,
        position: Any,
        neighbors: Sequence[Any],
        attrs: Optional[Dict[str, Any]] = None,
    ) -> int:
        node_id = self._next_node_id()
        stored_pos = self._append_position(position)
        self.key_graph.graph.add_node(node_id, pos=stored_pos)
        for key, value in dict(attrs or {}).items():
            self.key_graph.graph.nodes[node_id][key] = value
        for nbr in neighbors:
            weight = latent_l2(position, self.key_graph.nodes[int(nbr)])
            self.key_graph.graph.add_edge(int(nbr), node_id, weight=float(weight))
            self.key_graph.graph.add_edge(node_id, int(nbr), weight=float(weight))
        return node_id

    def insert_anchor_record(
        self,
        anchor: AnchorRecord,
        manager: Optional[AnchorManager] = None,
        base_only: bool = True,
    ) -> InsertedSemanticNode:
        """Insert one connected anchor; fail conservatively if it lacks support."""
        position = self.embed(anchor.carrier)
        neighbors = self.local_neighbors(position, base_only=base_only)
        min_dist = self.min_distance_to_graph(position, base_only=base_only)
        if not neighbors:
            anchor.connected = False
            anchor.metadata["connection_status"] = "no_local_neighbor"
            anchor.metadata["min_distance"] = min_dist
            return InsertedSemanticNode(
                node_id=None,
                position=position,
                connection_status="no_local_neighbor",
                neighbors=[],
                min_distance=min_dist,
                metadata={"anchor_id": anchor.anchor_id},
            )

        attrs = {
            "is_anchor": True,
            "semantic_anchor_id": anchor.anchor_id,
            "semantic_cert_pos": sorted(anchor.cert_pos),
            "semantic_cert_neg": sorted(anchor.cert_neg),
            "semantic_prop_ids": sorted(anchor.prop_ids),
            "semantic_guard_ids": sorted(anchor.guard_ids),
            "raw_state": anchor.carrier.state
            if anchor.carrier.state is not None
            else anchor.carrier.observation,
            "anchor_connection_mode": "strict_htd",
            "anchor_min_base_distance": float(min_dist),
        }
        if len(anchor.prop_ids) == 1:
            # Compatibility with existing conference-version execution and
            # visualization utilities.
            attrs["hard_label"] = next(iter(anchor.prop_ids))

        node_id = self._add_node_with_edges(position, neighbors, attrs=attrs)
        anchor.connected = True
        anchor.node_id = node_id
        anchor.metadata["connection_status"] = "local_connected"
        anchor.metadata["neighbors"] = list(neighbors)
        anchor.metadata["min_distance"] = min_dist
        if manager is not None:
            manager.attach_graph_node(anchor.anchor_id, node_id)
        for prop in anchor.prop_ids:
            self.key_graph.label_node_idx_dict.setdefault(prop, []).append(node_id)
        return InsertedSemanticNode(
            node_id=node_id,
            position=position,
            connection_status="local_connected",
            neighbors=list(neighbors),
            min_distance=min_dist,
            metadata={"anchor_id": anchor.anchor_id},
        )

    def insert_anchor_records(
        self,
        anchors: Iterable[AnchorRecord],
        manager: Optional[AnchorManager] = None,
        base_only: bool = True,
    ) -> Dict[Any, InsertedSemanticNode]:
        out = {}
        for anchor in anchors:
            out[anchor.anchor_id] = self.insert_anchor_record(
                anchor,
                manager=manager,
                base_only=base_only,
            )
        return out

    def insert_start_node(
        self,
        carrier: SemanticCarrier,
        allow_nearest_fallback: bool = True,
    ) -> InsertedSemanticNode:
        """Insert the physical rollout start and connect it to the graph."""
        position = self.embed(carrier)
        min_dist = self.min_distance_to_graph(position, base_only=False)
        neighbors = self.local_neighbors(position, base_only=False)
        status = "local_connected"
        if not neighbors and allow_nearest_fallback:
            nearest = self.nearest_neighbor(position, base_only=False)
            if nearest is not None:
                neighbors = [nearest]
                status = "nearest_fallback"
        elif not neighbors:
            status = "failed"

        cert_pos = set()
        cert_neg = set()
        if self.store is not None:
            cert_pos = self.store.cert_pos_set(carrier)
            cert_neg = self.store.cert_neg_set(carrier)

        attrs = {
            "is_start": True,
            "semantic_source": "start",
            "semantic_cert_pos": sorted(cert_pos),
            "semantic_cert_neg": sorted(cert_neg),
            "raw_state": carrier.state if carrier.state is not None else carrier.observation,
            "start_connection_status": status,
        }
        node_id = self._add_node_with_edges(position, neighbors, attrs=attrs)
        return InsertedSemanticNode(
            node_id=node_id,
            position=position,
            connection_status=status,
            neighbors=list(neighbors),
            min_distance=min_dist,
            metadata={"source": "start"},
        )

    def precompute_dist_to_label_if_available(self) -> None:
        if hasattr(self.key_graph, "precompute_dist_to_label"):
            self.key_graph.precompute_dist_to_label()
