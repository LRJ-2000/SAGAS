"""Evaluator-conditioned prefix--suffix product planner.

The planner operates on the implicit product of a semantic reachability graph
and a Buchi runtime.  It delegates transition feasibility to ProductOracle, so
positive progress, negative screening, and guard-supported anchors remain in a
single semantic successor interface.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from S_utils.certificate_store import CertificateStore
from S_utils.guard_terms import GuardTerm
from S_utils.product_oracle import ProductOracle
from S_utils.semantic_types import SemanticCarrier


ProductState = Tuple[Any, Any]


@dataclass
class ProductPath:
    """A finite product-state path and the guards used by its edges."""

    states: List[ProductState]
    guards: List[GuardTerm] = field(default_factory=list)
    cost: float = 0.0

    @property
    def graph_nodes(self) -> List[Any]:
        return [v for v, _q in self.states]

    @property
    def buchi_states(self) -> List[Any]:
        return [q for _v, q in self.states]


@dataclass
class ProductLassoPlan:
    """Selected accepting prefix--suffix plan in the semantic product graph."""

    prefix: ProductPath
    suffix: ProductPath
    total_cost: float
    candidate_rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def product_states(self) -> List[ProductState]:
        return list(self.prefix.states) + list(self.suffix.states)

    @property
    def guard_terms(self) -> List[GuardTerm]:
        return list(self.prefix.guards) + list(self.suffix.guards)

    @property
    def waypoint_nodes(self) -> List[Any]:
        return [v for v, _q in self.product_states]


@dataclass
class FiniteProductExecutionPlan:
    """Finite execution schedule obtained from a prefix--suffix product lasso."""

    states: List[ProductState]
    guards: List[GuardTerm]
    prefix_end_index: int
    suffix_cycles_required: int
    suffix_cycle_num_edges: int

    @property
    def graph_nodes(self) -> List[Any]:
        return [v for v, _q in self.states]

    @property
    def buchi_states(self) -> List[Any]:
        return [q for _v, q in self.states]


def materialize_finite_lasso(
    plan: ProductLassoPlan,
    suffix_cycles: int = 2,
) -> FiniteProductExecutionPlan:
    """Expand a product lasso into the finite schedule used for evaluation.

    The prefix is executed once.  The suffix path is then appended
    `suffix_cycles` times, each time dropping the duplicated first suffix state
    because it is already the current product state.
    """

    suffix_cycles = max(0, int(suffix_cycles))
    states = list(plan.prefix.states)
    guards = list(plan.prefix.guards)
    prefix_end_index = len(states) - 1
    for _ in range(suffix_cycles):
        states.extend(plan.suffix.states[1:])
        guards.extend(plan.suffix.guards)
    return FiniteProductExecutionPlan(
        states=states,
        guards=guards,
        prefix_end_index=prefix_end_index,
        suffix_cycles_required=suffix_cycles,
        suffix_cycle_num_edges=len(plan.suffix.guards),
    )


def initial_product_states(
    start_node_id: Any,
    start_carrier: SemanticCarrier,
    certificate_store: CertificateStore,
    buchi_runtime,
) -> List[ProductState]:
    """Consume the start label and build the initial product-state set."""
    label_set = certificate_store.cert_pos_set(start_carrier)
    q_init = buchi_runtime.initial_state_set_after_label(label_set)
    return [(start_node_id, q) for q in sorted(q_init)]


def _reconstruct_path(
    parent: Dict[ProductState, Optional[ProductState]],
    parent_guard: Dict[ProductState, Optional[GuardTerm]],
    parent_step_cost: Dict[ProductState, float],
    end_state: ProductState,
) -> ProductPath:
    states: List[ProductState] = []
    guards_reversed: List[GuardTerm] = []
    cost = 0.0
    cur: Optional[ProductState] = end_state
    while cur is not None:
        states.append(cur)
        guard = parent_guard.get(cur)
        if guard is not None:
            guards_reversed.append(guard)
            cost += float(parent_step_cost.get(cur, 0.0))
        cur = parent[cur]
    states.reverse()
    guards_reversed.reverse()
    return ProductPath(states=states, guards=guards_reversed, cost=cost)


def search_prefix_candidates(
    graph,
    buchi_runtime,
    oracle: ProductOracle,
    start_states: Iterable[ProductState],
    max_candidates: int = 3,
    heuristic_fn: Optional[Callable[[ProductState], float]] = None,
    edge_weight_fn: Optional[Callable[[Any, Any], float]] = None,
) -> List[ProductPath]:
    """Run Dijkstra/A* until `max_candidates` accepting prefixes are found."""
    start_states = list(start_states)
    if not start_states:
        return []
    max_candidates = max(1, int(max_candidates))
    heuristic_fn = heuristic_fn or (lambda _state: 0.0)
    counter = itertools.count()

    parent: Dict[ProductState, Optional[ProductState]] = {}
    parent_guard: Dict[ProductState, Optional[GuardTerm]] = {}
    parent_step_cost: Dict[ProductState, float] = {}
    g_cost: Dict[ProductState, float] = {}
    open_pq: List[Tuple[float, int, ProductState]] = []

    for state in start_states:
        parent[state] = None
        parent_guard[state] = None
        parent_step_cost[state] = 0.0
        g_cost[state] = 0.0
        heapq.heappush(open_pq, (float(heuristic_fn(state)), next(counter), state))

    closed = set()
    prefixes: List[ProductPath] = []
    while open_pq and len(prefixes) < max_candidates:
        _priority, _ord, state = heapq.heappop(open_pq)
        if state in closed:
            continue
        closed.add(state)

        if buchi_runtime.is_accepting(state[1]):
            prefixes.append(_reconstruct_path(parent, parent_guard, parent_step_cost, state))
            continue

        for next_state, step_cost, guard in oracle.successors(
            graph,
            buchi_runtime,
            state,
            edge_weight_fn=edge_weight_fn,
        ):
            new_g = g_cost[state] + float(step_cost)
            if new_g + 1e-12 >= g_cost.get(next_state, float("inf")):
                continue
            g_cost[next_state] = new_g
            parent[next_state] = state
            parent_guard[next_state] = guard
            parent_step_cost[next_state] = float(step_cost)
            priority = new_g + float(heuristic_fn(next_state))
            heapq.heappush(open_pq, (priority, next(counter), next_state))

    return prefixes


def _logical_self_loop_suffix(
    buchi_runtime,
    oracle: ProductOracle,
    endpoint: ProductState,
) -> Optional[ProductPath]:
    """Instantiate a zero-waypoint-cost logical suffix when possible."""
    u, q = endpoint
    for qn, guard in buchi_runtime.enabled_guard_successors(q):
        if qn == q and oracle.enable(u, guard):
            return ProductPath(states=[endpoint, endpoint], guards=[guard], cost=0.0)
    return None


def search_suffix_cycle(
    graph,
    buchi_runtime,
    oracle: ProductOracle,
    endpoint: ProductState,
    edge_weight_fn: Optional[Callable[[Any, Any], float]] = None,
    allow_logical_self_loop: bool = True,
) -> Optional[ProductPath]:
    """Find a non-empty product cycle returning to `endpoint`."""
    if allow_logical_self_loop:
        suffix = _logical_self_loop_suffix(buchi_runtime, oracle, endpoint)
        if suffix is not None:
            return suffix

    counter = itertools.count()
    parent: Dict[ProductState, Optional[ProductState]] = {endpoint: None}
    parent_guard: Dict[ProductState, Optional[GuardTerm]] = {endpoint: None}
    parent_step_cost: Dict[ProductState, float] = {endpoint: 0.0}
    g_cost: Dict[ProductState, float] = {endpoint: 0.0}
    open_pq: List[Tuple[float, int, ProductState]] = []
    closed = {endpoint}

    for next_state, step_cost, guard in oracle.successors(
        graph,
        buchi_runtime,
        endpoint,
        edge_weight_fn=edge_weight_fn,
    ):
        if next_state == endpoint:
            return ProductPath(
                states=[endpoint, endpoint],
                guards=[guard],
                cost=float(step_cost),
            )
        parent[next_state] = endpoint
        parent_guard[next_state] = guard
        parent_step_cost[next_state] = float(step_cost)
        g_cost[next_state] = float(step_cost)
        heapq.heappush(open_pq, (float(step_cost), next(counter), next_state))

    while open_pq:
        _cost, _ord, state = heapq.heappop(open_pq)
        if state in closed:
            continue
        closed.add(state)

        for next_state, step_cost, guard in oracle.successors(
            graph,
            buchi_runtime,
            state,
            edge_weight_fn=edge_weight_fn,
        ):
            new_g = g_cost[state] + float(step_cost)
            if next_state == endpoint:
                path_to_state = _reconstruct_path(
                    parent,
                    parent_guard,
                    parent_step_cost,
                    state,
                )
                return ProductPath(
                    states=path_to_state.states + [endpoint],
                    guards=path_to_state.guards + [guard],
                    cost=path_to_state.cost + float(step_cost),
                )
            if new_g + 1e-12 >= g_cost.get(next_state, float("inf")):
                continue
            g_cost[next_state] = new_g
            parent[next_state] = state
            parent_guard[next_state] = guard
            parent_step_cost[next_state] = float(step_cost)
            heapq.heappush(open_pq, (new_g, next(counter), next_state))

    return None


def plan_prefix_suffix(
    graph,
    buchi_runtime,
    oracle: ProductOracle,
    start_states: Iterable[ProductState],
    max_prefix_candidates: int = 3,
    lambda_weight: float = 1.0,
    heuristic_fn: Optional[Callable[[ProductState], float]] = None,
    edge_weight_fn: Optional[Callable[[Any, Any], float]] = None,
) -> Optional[ProductLassoPlan]:
    """Select the lowest-cost prefix--suffix candidate found by the search."""
    prefixes = search_prefix_candidates(
        graph,
        buchi_runtime,
        oracle,
        start_states,
        max_candidates=max_prefix_candidates,
        heuristic_fn=heuristic_fn,
        edge_weight_fn=edge_weight_fn,
    )
    best: Optional[ProductLassoPlan] = None
    for rank, prefix in enumerate(prefixes):
        endpoint = prefix.states[-1]
        suffix = search_suffix_cycle(
            graph,
            buchi_runtime,
            oracle,
            endpoint,
            edge_weight_fn=edge_weight_fn,
        )
        if suffix is None:
            continue
        total_cost = prefix.cost + float(lambda_weight) * suffix.cost
        if best is None or total_cost + 1e-12 < best.total_cost:
            best = ProductLassoPlan(
                prefix=prefix,
                suffix=suffix,
                total_cost=total_cost,
                candidate_rank=rank,
                metadata={
                    "num_prefix_candidates": len(prefixes),
                    "lambda_weight": float(lambda_weight),
                },
            )
    return best
