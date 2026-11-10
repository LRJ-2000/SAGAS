import os
import jax
import pickle
import numpy as np
import networkx as nx
import jax.numpy as jnp

from tqdm import tqdm

from typing import Dict, List, Optional
from collections import defaultdict

from K_utils.keynodes_utils import KeyNodes


class KeyGraph:
    """Construct keygraph for task planning."""

    def __init__(self):
        self.way_steps = None
        self.te_threshold = None
        self.nodes = None
        self.graph = None

        self.base_node_cnt = 0  # number of nodes before adding task goals
        self.task_goal_dict = {}  # task id → raw goal state
        self.task_node_dict = {}  # task id → phi-goal node
        self.task_node_idx_dict = {}  # task id → phi-goal node idx
        self.task_paths_dict = {}  # task id → {node idx: [path idx list]}
        self.task_paths_dist_dict = {}  # task id → {node idx: distance}

        # --- LTL / Label planning extensions ---
        # Hard anchors: label/AP -> list of node indices added as anchors for that label.
        self.label_node_idx_dict: Dict[str, List[int]] = {}
        # Soft labels (for avoidance/reference only): node_idx -> {label: probability}
        self.node_soft_labels: Dict[int, Dict[str, float]] = {}
        # For heuristics: label/AP -> {node_idx: distance to nearest anchor of that label}
        self.dist_to_label: Dict[str, Dict[int, float]] = {}
        # Diagnostics for conservative anchor insertion.
        self.anchor_connection_stats: Dict[str, Dict] = {}
        self.anchor_unavailable_labels: List[str] = []

    def construct_graph(self, key_nodes, batch_size, component_merge_strategy="exact"):
        """Construct a directed graph from key nodes."""
        self.way_steps = key_nodes.way_steps
        self.te_threshold = key_nodes.te_threshold
        self.nodes = key_nodes.nodes
        self.key_nodes = key_nodes
        self.base_node_cnt = len(self.nodes)
        self.graph = nx.DiGraph()
        for node_idx, node_pos in tqdm(
            enumerate(self.nodes), desc="Adding Nodes to Graph", total=len(self.nodes)
        ):
            self.graph.add_node(node_idx, pos=node_pos)
        self.graph, pdist_matrix = add_distance_based_edges(
            self.graph, self.nodes, self.way_steps, batch_size
        )

        self.graph = nx.DiGraph(self.graph)
        if component_merge_strategy == "exact":
            self.graph = connect_strongly_connected_components(self.graph, pdist_matrix)
        elif component_merge_strategy == "fast":
            self.graph = connect_strongly_connected_components_fast(self.graph)
        else:
            raise ValueError(
                "Unknown component_merge_strategy: "
                f"{component_merge_strategy!r}. Expected 'exact' or 'fast'."
            )

        for u, v in tqdm(
            self.graph.edges,
            desc="Assigning Weights to Edges",
            total=len(self.graph.edges),
        ):
            distance = np.linalg.norm(self.nodes[u] - self.nodes[v])
            self.graph[u][v]["weight"] = distance

    def precompute_shortest_paths_to_all_tasks(self, task_goal_dict, task_node_dict):
        """Precompute shortest paths from all nodes to each task goal."""
        self.clear_task_goals()
        self.task_goal_dict = task_goal_dict
        self.task_node_dict = task_node_dict
        for task_id, target in tqdm(
            self.task_node_dict.items(), desc="Precomputing shortest paths for tasks"
        ):
            target_idx = self.graph.number_of_nodes()
            self.graph, self.nodes = add_target_node(
                self.graph, self.nodes, target, target_idx, self.way_steps
            )
            self.task_node_idx_dict[task_id] = target_idx
            self.task_paths_dict[task_id], self.task_paths_dist_dict[task_id] = (
                compute_shortest_paths_to_target(self.graph, target_idx)
            )

    def clear_task_goals(self):
        """Remove all task-related nodes and paths."""
        for idx in sorted(self.task_node_idx_dict.values(), reverse=True):
            if self.graph.has_node(idx):
                self.graph.remove_node(idx)
        self.nodes = self.nodes[: self.base_node_cnt]

        self.task_goal_dict.clear()
        self.task_node_dict.clear()
        self.task_node_idx_dict.clear()
        self.task_paths_dict.clear()
        self.task_paths_dist_dict.clear()

    def get_shortest_path(self, task_id, source, force_closest=False):
        """Get the precomputed shortest path to a task goal from a source point."""
        nodes = self.nodes
        shortest_paths = self.task_paths_dict[task_id]
        shortest_paths_dist = self.task_paths_dist_dict[task_id]

        sp_keys = list(shortest_paths.keys())
        start_distances = np.linalg.norm(nodes[sp_keys] - source, axis=1)
        valid_indices = np.where(start_distances <= self.way_steps)[0]
        if len(valid_indices) == 0:
            if force_closest:
                closest_index = np.argmin(start_distances)
                valid_indices = [closest_index]
            else:
                return None

        best_total_distance = float("inf")
        best_path = None
        for idx in valid_indices:
            path_key = sp_keys[idx]
            path_distance = shortest_paths_dist[path_key]
            total_distance = float(start_distances[idx]) + float(path_distance)
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_path = shortest_paths[path_key]

        return nodes[best_path]

    def assign_soft_labels(
        self, label_fn, tau: float = 0.01
    ):
        """
        Build soft labels per base node by voting over cluster members with `label_fn`.
        Only labels with freq >= tau are kept (others are ignored).
        NOTE:
            - `key_nodes.cluster_members` should preferably be RAW objects (if raw_states were provided).
            - If it's indices, you must adapt label_fn to accept indices or reconstruct raw states upstream.
        """
        if self.key_nodes.cluster_members is None:
            print("[KeyGraph] No cluster_members found; soft labels skipped.")
            return

        for node_idx in range(self.base_node_cnt):
            members = self.key_nodes.cluster_members.get(node_idx, [])
            if not members:
                self.node_soft_labels[node_idx] = {}
                continue

            counts = defaultdict(int)
            total = 0
            for elem in members:
                # `elem` is raw state if raw_states were provided; otherwise it's an int index.
                try:
                    lab = label_fn(elem)
                except Exception:
                    # If label_fn cannot handle elem type (e.g., indices), skip voting for this node.
                    lab = None
                if lab is not None:
                    counts[lab] += 1
                total += 1

            soft = {}
            if total > 0 and counts:
                for lab, c in counts.items():
                    p = c / total
                    if p >= tau:
                        soft[lab] = p
            self.node_soft_labels[node_idx] = soft

    def add_label_anchors(
        self,
        label_to_raw_states: Dict[str, List],
        get_phi,
        max_anchors_per_label: Optional[int] = None,
        require_neighbor_within_cutoff: bool = True,
    ):
        """
        Add 'hard' anchor nodes for each label/AP:
            - Only reaching these anchors is considered 'achieving' the label.
            - By default, anchors are connected only when they have at least one
              base-graph neighbor within way_steps; otherwise more sampled
              candidates are tried, and the label may become unavailable.
        Node attributes for anchors:
            graph.nodes[new_idx]['is_anchor'] = True
            graph.nodes[new_idx]['hard_label'] = <label>
        """
        self.anchor_connection_stats = {}
        self.anchor_unavailable_labels = []

        for label, raw_states in label_to_raw_states.items():
            idx_list = self.label_node_idx_dict.setdefault(label, [])
            target_count = (
                len(raw_states)
                if max_anchors_per_label is None
                else int(max_anchors_per_label)
            )
            stats = {
                "attempted": 0,
                "added": 0,
                "unavailable": False,
                "min_base_distance": float("inf"),
                "cutoff": float(self.way_steps),
            }
            for s in raw_states:
                if stats["added"] >= target_count:
                    break

                h = np.array(get_phi(s))
                stats["attempted"] += 1

                base_nodes = self.nodes[: self.base_node_cnt]
                base_dists = np.linalg.norm(base_nodes - h, axis=1)
                min_base_dist = float(np.min(base_dists))
                stats["min_base_distance"] = min(
                    stats["min_base_distance"], min_base_dist
                )

                if require_neighbor_within_cutoff and min_base_dist >= self.way_steps:
                    continue

                new_idx = self.graph.number_of_nodes()
                self.graph, self.nodes = add_target_node(
                    self.graph,
                    self.nodes,
                    h,
                    new_idx,
                    self.way_steps,
                    force_nearest=not require_neighbor_within_cutoff,
                )
                self.graph.nodes[new_idx]["is_anchor"] = True
                self.graph.nodes[new_idx]["hard_label"] = label
                self.graph.nodes[new_idx]["raw_state"] = s
                self.graph.nodes[new_idx]["anchor_min_base_distance"] = min_base_dist
                self.graph.nodes[new_idx]["anchor_connection_mode"] = (
                    "strict_htd"
                    if require_neighbor_within_cutoff
                    else "nearest_fallback"
                )
                idx_list.append(new_idx)
                stats["added"] += 1

            if stats["added"] == 0:
                stats["unavailable"] = True
                self.anchor_unavailable_labels.append(label)
                print(
                    f"[KeyGraph] Anchor label {label} unavailable: "
                    f"no sampled state connected within way_steps={self.way_steps}."
                )
            self.anchor_connection_stats[label] = stats

        return self.anchor_connection_stats

    def precompute_dist_to_label(self):
        """
        For each label/AP, compute shortest-path distance from any node to the nearest anchor node of that label.
        We compute on the REVERSED graph with all anchors as zero-cost sources (multi-source Dijkstra on reverse graph).
        Store as: self.dist_to_label[label][node_idx] = distance.
        """
        for label, anchors in self.label_node_idx_dict.items():
            # Initialize distances to +inf.
            dist = {int(i): float("inf") for i in self.graph.nodes()}
            # Multi-source initialization: distance to anchor itself = 0.
            import heapq

            pq = []
            for a in anchors:
                dist[int(a)] = 0.0
                heapq.heappush(pq, (0.0, int(a)))

            # Relaxation along reversed edges so that dist[v] = min distance from v to any anchor(label).
            while pq:
                du, u = heapq.heappop(pq)
                if du > dist[u] + 1e-12:
                    continue
                for v in self.graph.predecessors(u):
                    w = self.graph[v][u].get(
                        "weight", float(np.linalg.norm(self.nodes[v] - self.nodes[u]))
                    )
                    nd = du + float(w)
                    if nd + 1e-12 < dist[v]:
                        dist[v] = nd
                        heapq.heappush(pq, (nd, int(v)))
            self.dist_to_label[label] = dist

    def ensure_start_node(
        self, start_raw_state, get_phi, try_snap_to_near: bool = True
    ) -> int:
        """
        Ensure the start point is represented as a graph node:
            - If there exists a node within way_steps in latent space, reuse it.
            - Otherwise, add a new node with the same connectivity rule as goals/anchors.
        Returns:
            start_node_idx (int)
        """
        h0 = np.array(get_phi(start_raw_state))
        if try_snap_to_near and self.nodes is not None and len(self.nodes) > 0:
            dists = np.linalg.norm(self.nodes - h0, axis=1)
            j = int(np.argmin(dists))
            if dists[j] <= self.way_steps:
                return j
        new_idx = self.graph.number_of_nodes()
        self.graph, self.nodes = add_target_node(
            self.graph, self.nodes, h0, new_idx, self.way_steps
        )
        self.graph.nodes[new_idx]["is_start"] = True
        self.graph.nodes[new_idx]["raw_state"] = start_raw_state
        return new_idx

    def save_keygraph(self, save_dir, filename="keygraph"):
        """
        Save the keygraph to a file.
        Args:
            save_dir: Directory to save the keygraph.
            filename: File name.
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[KeyGraph] Saved to {save_path}")

    def load_keygraph(self, save_dir, filename="keygraph"):
        """
        Load the keygraph from a file.
        Args:
            save_dir: Path to the directory containing the saved keygraph.
            filename: File name.
        """
        load_path = os.path.join(save_dir, f"{filename}.pkl")
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)
        print(f"[KeyGraph] Loaded from {load_path}")


def compute_pdist_matrix(query_points, reference_points, batch_size):
    """Compute pairwise distance matrix."""
    num_queries = query_points.shape[0]
    num_references = reference_points.shape[0]
    distance_matrix = np.zeros((num_queries, num_references), dtype=np.float32)
    for start_idx in tqdm(
        range(0, num_queries, batch_size),
        desc="Calculating Pairwise Distances in Batches",
        leave=False,
    ):
        end_idx = min(start_idx + batch_size, num_queries)
        batch = query_points[start_idx:end_idx]
        distances = np.array(
            pairwise_distances(query_points=batch, reference_points=reference_points)
        )
        distance_matrix[start_idx:end_idx, :] = distances
    return distance_matrix


@jax.jit
def pairwise_distances(query_points, reference_points):
    """Compute pairwise distance."""
    return jnp.sqrt(
        jnp.sum(
            (query_points[:, jnp.newaxis, :] - reference_points[jnp.newaxis, :, :])
            ** 2,
            axis=-1,
        )
    )


def add_distance_based_edges(graph, nodes, cutoff, batch_size):
    """Add bidirectional edges based on temporal distance threshold."""
    pdist_matrix = compute_pdist_matrix(nodes, nodes, batch_size)
    np.fill_diagonal(pdist_matrix, np.inf)
    for i, row in tqdm(
        enumerate(pdist_matrix),
        total=len(pdist_matrix),
        desc="Adding Distance-Based Edges to Graph",
    ):
        neighbors = np.where(row <= cutoff)[0]
        for j in neighbors:
            graph.add_edge(i, j)
    return graph, pdist_matrix


def connect_strongly_connected_components(graph, pdist_matrix):
    """Connect disconnected components to ensure full graph connectivity."""
    components = list(nx.strongly_connected_components(graph))
    component_groups = [list(comp) for comp in components]
    with tqdm(total=len(component_groups), desc="Merging components") as pbar:
        while len(component_groups) > 1:
            main_nodes = component_groups[0]
            other_nodes = [node for comp in component_groups[1:] for node in comp]
            dist_matrix = pdist_matrix[np.ix_(main_nodes, other_nodes)]

            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            main_node, other_node = main_nodes[min_idx[0]], other_nodes[min_idx[1]]
            graph.add_edge(main_node, other_node)
            graph.add_edge(other_node, main_node)

            for comp in component_groups[1:]:
                if other_node in comp:
                    component_groups[0].extend(comp)
                    component_groups.remove(comp)
                    break
            pbar.update(1)

    return graph


def _node_position(graph, node):
    pos = graph.nodes[node].get("pos")
    if pos is None:
        raise ValueError(f"Graph node {node} does not have a `pos` attribute.")
    return np.asarray(pos)


def connect_strongly_connected_components_fast(graph, num_neighbors=8):
    """Quickly connect components by representative-node nearest neighbors.

    This strategy is intended for smoke-test graph rebuilds where exact
    component merging would dominate runtime.  It preserves the same interface
    as the exact routine by adding bidirectional latent-distance links, but it
    chooses links from component representatives instead of scanning all
    cross-component node pairs at every merge step.
    """
    components = [list(comp) for comp in nx.strongly_connected_components(graph)]
    if len(components) <= 1:
        return graph

    reps = [comp[0] for comp in components]
    rep_pos = np.asarray([_node_position(graph, rep) for rep in reps])
    k = min(int(num_neighbors) + 1, len(reps))

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(rep_pos)
        _dists, neigh = tree.query(rep_pos, k=k)
        neigh = np.asarray(neigh)
        if neigh.ndim == 1:
            neigh = neigh[:, None]
        for i, row in enumerate(neigh):
            for j in row:
                j = int(j)
                if i == j:
                    continue
                u, v = int(reps[i]), int(reps[j])
                graph.add_edge(u, v)
                graph.add_edge(v, u)
    except Exception:
        # SciPy is optional.  The fallback connects consecutive component
        # representatives, which is sufficient to make the graph connected for
        # smoke-test planning.
        for i in range(1, len(reps)):
            u, v = int(reps[i - 1]), int(reps[i])
            graph.add_edge(u, v)
            graph.add_edge(v, u)

    # kNN links are usually enough, but they are not a formal connectivity
    # guarantee.  Connect any remaining component groups by representative
    # chain to ensure the same connected-graph postcondition.
    remaining = [list(comp) for comp in nx.strongly_connected_components(graph)]
    if len(remaining) > 1:
        remaining_reps = [comp[0] for comp in remaining]
        for i in range(1, len(remaining_reps)):
            u, v = int(remaining_reps[i - 1]), int(remaining_reps[i])
            graph.add_edge(u, v)
            graph.add_edge(v, u)

    return graph


def add_target_node(graph, nodes, target, target_idx, cutoff, force_nearest=True):
    """Add a goal node to the graph and connect it to nearby nodes."""
    graph.add_node(target_idx, pos=target)
    distances_to_target = np.linalg.norm(nodes - target, axis=1)
    threshold = cutoff
    if force_nearest:
        min_dists = np.min(distances_to_target) * 1.2
        threshold = max(cutoff, min_dists)

    for i, dist in enumerate(distances_to_target):
        if float(dist) < threshold:
            graph.add_edge(int(i), int(target_idx), weight=float(dist))
            graph.add_edge(int(target_idx), int(i), weight=float(dist))
    updated_nodes = np.vstack([nodes, target])

    return graph, updated_nodes


def compute_shortest_paths_to_target(graph, target_idx):
    """Compute shortest paths to the given goal node."""
    lengths, paths = nx.single_source_dijkstra(
        graph, source=target_idx, weight="weight"
    )

    shortest_paths = {}
    shortest_paths_dist = {}
    for node_idx, path in paths.items():
        if node_idx == target_idx:
            continue
        shortest_paths[node_idx] = path[::-1]
        shortest_paths_dist[node_idx] = lengths[node_idx]

    return shortest_paths, shortest_paths_dist
