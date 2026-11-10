# -*- coding: utf-8 -*-
"""
Incremental Product-A* planner on (KeyGraph x Buchi) without pre-building the full product graph.

Updates:
- No global forbid_labels. Instead, for each ENABLED Buchi edge we read its negative literal
  requirements (lit=False) and conservatively prune neighbor graph nodes whose soft labels
  strongly indicate that lit's label is actually present in that node.
"""

from typing import Dict, List, Optional, Set, Tuple
import heapq
import numpy as np
import networkx as nx


def _build_literal_assignment_for_node(
    key_graph, node_idx: int, buchi_rt
) -> Dict[str, bool]:
    """
    Build a sparse assignment for Buchi literals at this graph node.
    Only the anchor node's label literal is set to True; all others are implicitly False.
    """
    assign = {}
    node_data = key_graph.graph.nodes[node_idx]
    if node_data.get("is_anchor", False):
        label = node_data.get("hard_label", None)
        if label is not None:
            lit = buchi_rt.literal_for_label(label)
            assign[lit] = True
    return assign


def _heuristic(key_graph, node_idx: int, q: str, buchi_rt) -> float:
    """
    Admissible prefix heuristic on the latent product graph.

    If q is accepting, or q can reach acceptance through positive-free Buchi
    transitions, the lower bound is zero. Otherwise, any accepting continuation
    must first visit an anchor for one of the precomputed first-positive labels;
    the graph distance to the nearest such anchor is therefore a lower bound.
    """
    buchi_rt.ensure_prefix_heuristic_info()
    if buchi_rt.is_accepting(q) or q in buchi_rt.positive_free_accept_reachable:
        return 0.0

    next_lits = buchi_rt.first_positive_literals_toward_accept(q)
    if not next_lits:
        return 0.0

    best = np.inf
    for lit in next_lits:
        label = buchi_rt.label_for_literal(lit)
        dist_map = key_graph.dist_to_label.get(label, None)
        if dist_map is None:
            continue
        d = dist_map.get(node_idx, np.inf)
        if d < best:
            best = d
    return 0.0 if best == np.inf else float(best)


def _edge_weight(key_graph, u: int, v: int) -> float:
    edge_data = key_graph.graph[u][v]
    weight = edge_data.get("weight", None)
    if weight is None:
        weight = float(np.linalg.norm(key_graph.nodes[u] - key_graph.nodes[v]))
    return float(weight)


def _truth_pruned_by_soft_label(
    key_graph, node_idx: int, buchi_rt, truth: Dict[str, bool], threshold: float
) -> bool:
    soft = key_graph.node_soft_labels.get(node_idx, {})
    for lit, need in truth.items():
        if not need:
            label = buchi_rt.label_for_literal(lit)
            if soft.get(label, 0.0) >= threshold:
                return True
    return False


def _forbidden_labels_from_truth(buchi_rt, truth: Dict[str, bool]) -> List[str]:
    labels = []
    for lit, need in truth.items():
        if not need:
            labels.append(buchi_rt.label_for_literal(lit))
    return sorted(set(labels))


def _product_successors_with_truth(
    key_graph, buchi_rt, state, avoid_soft_label_threshold
):
    """
    Generate feasible one-step successors and retain the Buchi guard truth.
    """
    u, q = state
    for v in key_graph.graph.successors(u):
        assign = _build_literal_assignment_for_node(key_graph, v, buchi_rt)
        q_succ_list = buchi_rt.succ_with_truth(q, assign)
        if not q_succ_list:
            continue

        step_cost = _edge_weight(key_graph, u, v)
        for qn, truth in q_succ_list:
            if _truth_pruned_by_soft_label(
                key_graph, v, buchi_rt, truth, avoid_soft_label_threshold
            ):
                continue
            yield (v, qn), step_cost, dict(truth)


def _product_successors(key_graph, buchi_rt, state, avoid_soft_label_threshold):
    """
    Generate feasible one-step successors in the implicit product graph.
    This is the shared transition rule for prefix and suffix search.
    """
    for next_state, step_cost, _truth in _product_successors_with_truth(
        key_graph, buchi_rt, state, avoid_soft_label_threshold
    ):
        yield next_state, step_cost


def _reconstruct_product_path(parent, cur):
    path = []
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _reconstruct_product_path_with_truth(parent, parent_edge_truth, cur):
    states = []
    truths = []
    while cur is not None:
        states.append(cur)
        truths.append(parent_edge_truth.get(cur, {}))
        cur = parent[cur]
    states.reverse()
    truths.reverse()
    return states, truths[1:]


def ltl_plan_astar_full(
    key_graph,
    buchi_rt,
    start_raw_state,
    get_phi,
    avoid_soft_label_threshold: float = 0.5,
    lambda_weight: float = 1.0,
    max_prefix_candidates: int = 3,
) -> Optional[Dict]:
    """
    Perform A* search over the product of KeyGraph x Buchi Automaton,
    returning a full (prefix + suffix) LTL-satisfying path if found.

    Args:
        key_graph: KeyGraph instance with node/edge info and soft labels.
        buchi_rt: BuchiRuntime instance with automaton transitions.
        start_raw_state: raw env state to locate as the graph starting node.
        get_phi: function to encode raw state into latent vector.
        avoid_soft_label_threshold: pruning threshold for negative constraints.
        lambda_weight: suffix loop cost weight.
        max_prefix_candidates: number of accepting prefixes to consider.

    Returns:
        A dict with "positions", "node_indices", "anchor_mask",
        and prefix/suffix label boundaries or None if no path found.
    """

    start_idx = key_graph.ensure_start_node(
        start_raw_state, get_phi, try_snap_to_near=True
    )
    q0 = buchi_rt.initial_state()
    start = (start_idx, q0)

    g_cost: Dict[Tuple[int, str], float] = {start: 0.0}
    parent: Dict[Tuple[int, str], Optional[Tuple[int, str]]] = {start: None}
    parent_edge_truth: Dict[Tuple[int, str], Dict[str, bool]] = {start: {}}
    open_pq: List[Tuple[float, Tuple[int, str]]] = [(0.0, start)]
    closed: Set[Tuple[int, str]] = set()

    prefix_candidates = []

    while open_pq:
        f_cur, (u, q) = heapq.heappop(open_pq)
        if (u, q) in closed:
            continue
        closed.add((u, q))

        if buchi_rt.is_accepting(q):
            prefix_path, prefix_truths = _reconstruct_product_path_with_truth(
                parent, parent_edge_truth, (u, q)
            )
            prefix_cost = g_cost[(u, q)]
            prefix_candidates.append((u, q, prefix_path, prefix_truths, prefix_cost))
            if len(prefix_candidates) >= max_prefix_candidates:
                break

        for s2, step_cost, truth in _product_successors_with_truth(
            key_graph, buchi_rt, (u, q), avoid_soft_label_threshold
        ):
            v, qn = s2
            new_g = g_cost[(u, q)] + step_cost
            if new_g + 1e-12 < g_cost.get(s2, np.inf):
                g_cost[s2] = new_g
                parent[s2] = (u, q)
                parent_edge_truth[s2] = dict(truth)
                h = _heuristic(key_graph, v, qn, buchi_rt)
                heapq.heappush(open_pq, (new_g + h, s2))

    if not prefix_candidates:
        print("No prefix candidates found")
        prune_stats = getattr(buchi_rt, "anchor_prune_stats", {})
        if prune_stats.get("removed_edges", 0) > 0:
            labels = prune_stats.get("unavailable_labels", [])
            label_text = ",".join(labels) if labels else "unknown"
            return None, f"No prefix after anchor pruning: {label_text}"
        return None, "No prefix"
    print(f"Found {len(prefix_candidates)} prefix candidates")

    def reconstruct_node_path(prod_path):
        return [key_graph.nodes[u] for (u, _) in prod_path]

    def compute_cost(prod_path):
        total = 0.0
        for i in range(1, len(prod_path)):
            u1, _ = prod_path[i - 1]
            u2, _ = prod_path[i]
            if u1 == u2:
                continue
            total += _edge_weight(key_graph, u1, u2)
        return total

    best_plan = None
    best_cost = float("inf")

    for i, prefix_candidate in enumerate(prefix_candidates):
        u_acc, q_acc, prefix_path, prefix_truths, prefix_cost = prefix_candidate
        suffix_path, suffix_truths = find_suffix_cycle_with_truth(
            key_graph, buchi_rt, (u_acc, q_acc), avoid_soft_label_threshold
        )
        if suffix_path is None or len(suffix_path) < 1:
            print(f"No suffix path found for prefix candidate: {i}")
            continue
        suffix_cost = compute_cost(suffix_path)
        # print(f'Suffix cost: {suffix_cost}')
        # print(f'Suffix path: {suffix_path}')
        total_cost = prefix_cost + lambda_weight * suffix_cost

        if total_cost < best_cost:
            # Keep the accepting node as the first suffix waypoint for execution split.
            full_path = prefix_path + suffix_path
            idx_path = [node_id for (node_id, _) in full_path]
            pos_path = [key_graph.nodes[i] for i in idx_path]
            anchor_mask = [
                bool(key_graph.graph.nodes[i].get("is_anchor", False)) for i in idx_path
            ]

            best_plan = {
                "positions": pos_path,
                "node_indices": idx_path,
                "anchor_mask": anchor_mask,
                "prefix_length": len(prefix_path),
                "suffix_length": len(suffix_path),
                "prefix_product_states": prefix_path,
                "suffix_product_states": suffix_path,
                "prefix_edge_forbidden_labels": [
                    _forbidden_labels_from_truth(buchi_rt, truth)
                    for truth in prefix_truths
                ],
                "suffix_edge_forbidden_labels": [
                    _forbidden_labels_from_truth(buchi_rt, truth)
                    for truth in suffix_truths
                ],
            }
            best_cost = total_cost

    if best_plan is None:
        return best_plan, "No suffix"
    return best_plan, None


def find_suffix_cycle_with_truth(
    key_graph, buchi_rt, start, avoid_soft_label_threshold
):
    """
    Find a non-empty suffix cycle in the product graph starting and ending at
    the same accepting product state (u, q).

    Every product edge is checked with the same rule as prefix search:
    the graph edge must exist, the target-node assignment must enable a Buchi
    transition, and negative literals are pruned with soft-label risk.
    """
    u_start, q = start

    # check whether stay at the same node can stay in the same Buchi state
    assign = _build_literal_assignment_for_node(key_graph, u_start, buchi_rt)
    q_succ_list = buchi_rt.succ_with_truth(q, assign)
    for nq, truth in q_succ_list:
        if nq == q and not _truth_pruned_by_soft_label(
            key_graph, u_start, buchi_rt, truth, avoid_soft_label_threshold
        ):
            return [(u_start, q), (u_start, q)], [dict(truth)]

    # Dijkstra search over implicit product states. The start state is allowed
    # only as the terminal state so the returned suffix is a non-empty cycle.
    dist: Dict[Tuple[int, str], float] = {}
    parent: Dict[Tuple[int, str], Optional[Tuple[int, str]]] = {start: None}
    parent_edge_truth: Dict[Tuple[int, str], Dict[str, bool]] = {start: {}}
    open_pq: List[Tuple[float, int, Tuple[int, str]]] = []
    counter = 0
    best_return_cost = np.inf
    best_return_parent = None
    best_return_truth = None

    for next_state, step_cost, truth in _product_successors_with_truth(
        key_graph, buchi_rt, start, avoid_soft_label_threshold
    ):
        if next_state == start:
            return [start, start], [dict(truth)]
        if step_cost + 1e-12 < dist.get(next_state, np.inf):
            dist[next_state] = step_cost
            parent[next_state] = start
            parent_edge_truth[next_state] = dict(truth)
            counter += 1
            heapq.heappush(open_pq, (step_cost, counter, next_state))

    while open_pq:
        cur_cost, _, cur = heapq.heappop(open_pq)
        if cur_cost > dist.get(cur, np.inf) + 1e-12:
            continue
        if cur_cost >= best_return_cost:
            break

        for next_state, step_cost, truth in _product_successors_with_truth(
            key_graph, buchi_rt, cur, avoid_soft_label_threshold
        ):
            new_cost = cur_cost + step_cost
            if next_state == start:
                if new_cost + 1e-12 < best_return_cost:
                    best_return_cost = new_cost
                    best_return_parent = cur
                    best_return_truth = dict(truth)
                continue

            if new_cost + 1e-12 < dist.get(next_state, np.inf):
                dist[next_state] = new_cost
                parent[next_state] = cur
                parent_edge_truth[next_state] = dict(truth)
                counter += 1
                heapq.heappush(open_pq, (new_cost, counter, next_state))

    if best_return_parent is None:
        return None, []

    suffix_path, suffix_truths = _reconstruct_product_path_with_truth(
        parent, parent_edge_truth, best_return_parent
    )
    suffix_path.append(start)
    suffix_truths.append({} if best_return_truth is None else best_return_truth)
    return suffix_path, suffix_truths


def find_suffix_cycle(key_graph, buchi_rt, start, avoid_soft_label_threshold):
    path, _truths = find_suffix_cycle_with_truth(
        key_graph, buchi_rt, start, avoid_soft_label_threshold
    )
    return path


def ltl_plan_astar(
    key_graph,
    buchi_rt,
    start_raw_state,
    get_phi,
    avoid_soft_label_threshold: float = 0.5,
) -> Optional[List[np.ndarray]]:
    """
    Perform incremental A* on the product of (KeyGraph states) x (Buchi states).

    Negative constraints handling:
        For each enabled Buchi edge (q -> qn) with truth dict, we inspect all literals
        required False (lit=False). For each such literal, map to its label and check
        the neighbor node's soft label probability for that label:
            if P(label | node) >= avoid_soft_label_threshold -> prune this successor.

    Args:
        key_graph: constructed KeyGraph (with optional soft labels and precomputed dist_to_label).
        buchi_rt: BuchiRuntime instance.
        start_raw_state: raw env state to be connected/snap-in to the graph as start.
        get_phi: callable(raw_state) -> latent vector.
        avoid_soft_label_threshold: probability threshold for soft label avoidance on base nodes.

    Returns:
        A list of latent vectors (node positions) representing a path; None if not found.
    """
    # Ensure start node exists in the graph
    start_idx = key_graph.ensure_start_node(
        start_raw_state, get_phi, try_snap_to_near=True
    )

    # Buchi initial state
    q0 = buchi_rt.initial_state()

    # A* state: (graph_node, buchi_state)
    start = (start_idx, q0)
    g_cost: Dict[Tuple[int, str], float] = {start: 0.0}
    parent: Dict[Tuple[int, str], Optional[Tuple[int, str]]] = {start: None}

    f0 = _heuristic(key_graph, start_idx, q0, buchi_rt)
    open_pq: List[Tuple[float, Tuple[int, str]]] = [(f0, start)]
    closed: Set[Tuple[int, str]] = set()

    while open_pq:
        f_cur, (u, q) = heapq.heappop(open_pq)
        if (u, q) in closed:
            continue
        closed.add((u, q))

        if buchi_rt.is_accepting(q):
            # Reconstruct index path instead of only positions
            idx_path = []
            cur = (u, q)
            while cur is not None:
                idx_path.append(cur[0])
                cur = parent[cur]
            idx_path.reverse()

            # Build outputs
            pos_path = [key_graph.nodes[i] for i in idx_path]
            anchor_mask = []
            for nid in idx_path:
                data = key_graph.graph.nodes[nid]
                anchor_mask.append(bool(data.get("is_anchor", False)))

            return {
                "positions": pos_path,  # list[np.ndarray]
                "node_indices": idx_path,  # list[int]
                "anchor_mask": anchor_mask,  # list[bool], True if this node is an anchor added from a label region
            }

        # Expand neighbors in the KeyGraph
        for v in key_graph.graph.successors(u):
            # Build literal assignment for this neighbor node (only hard positive at anchors)
            assign = _build_literal_assignment_for_node(key_graph, v, buchi_rt)

            # Enabled Buchi successors with their edge truths
            q_succ_list = buchi_rt.succ_with_truth(q, assign)
            if not q_succ_list:
                continue

            # Edge cost in the KeyGraph
            step_cost = key_graph.graph[u][v].get(
                "weight", float(np.linalg.norm(key_graph.nodes[u] - key_graph.nodes[v]))
            )

            # Node-level soft labels used for conservative safety pruning
            soft = key_graph.node_soft_labels.get(v, {})

            for qn, truth in q_succ_list:
                # Conservative pruning for required negatives: for each lit with need=False,
                # if this node likely carries that label, skip this successor.
                prune = False
                for lit, need in truth.items():
                    if need is False:
                        label = buchi_rt.label_for_literal(lit)
                        prob = soft.get(label, 0.0)
                        if prob >= avoid_soft_label_threshold:
                            prune = True
                            break
                if prune:
                    continue

                s2 = (v, qn)
                new_g = g_cost[(u, q)] + step_cost
                if new_g + 1e-12 < g_cost.get(s2, np.inf):
                    g_cost[s2] = new_g
                    parent[s2] = (u, q)
                    h = _heuristic(key_graph, v, qn, buchi_rt)
                    heapq.heappush(open_pq, (new_g + h, s2))

    # No plan found
    return None
