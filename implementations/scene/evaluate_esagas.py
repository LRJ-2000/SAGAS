"""Evaluate SAGAS with set-valued labels, connected anchors and rollout verification."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from S_utils.anchor_manager import AnchorManager
from S_utils.certificate_store import CertificateStore
from S_utils.evaluators import (
    ObjectStateEvaluator,
    OracleStateLiftEvaluator,
    StateRegionEvaluator,
)
from S_utils.execution import SegmentExecutionResult, execute_finite_product_plan
from S_utils.guard_terms import GuardTerm
from S_utils.low_level_tracking import (
    LowLevelTrackingConfig,
    compute_tracking_command,
    select_tracking_target,
)
from S_utils.manipulation_anchor_selection import ManipulationAnchorSelector
from S_utils.manipulation_predicates import (
    scene_object_label_set,
    scene_task_info_for_env,
)
from S_utils.oracle_verifier import oracle_finite_lasso_verdict
from S_utils.predicate_specs import (
    SUPPORTED_PREDICATE_BACKENDS,
    extract_predicate_specs,
    normalize_evaluator_backend,
    task_regions_to_region_specs as normalize_task_regions_to_region_specs,
    task_to_region_specs as normalized_task_to_region_specs,
    validate_predicate_specs,
)
from S_utils.product_oracle import ProductOracle
from S_utils.product_planner import (
    ProductLassoPlan,
    initial_product_states,
    materialize_finite_lasso,
    plan_prefix_suffix,
)
from S_utils.runtime_monitor import RuntimeMonitor, RuntimeMonitorConfig
from S_utils.semantic_graph import KeyGraphSemanticAugmentor
from S_utils.semantic_types import AnchorRecord, SemanticCarrier
from prepare_esagas_assets import env_stem


@dataclass
class SemanticPlanningResult:
    """Outputs of the state-based evaluator-conditioned planning pipeline."""

    plan: Optional[ProductLassoPlan]
    start_node_id: Any
    start_product_states: List[tuple]
    anchor_manager: AnchorManager
    inserted_anchor_nodes: Dict[Any, Any]
    certificate_store: CertificateStore
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class TaskScheduleEntry:
    """A scheduled subset of tasks from one saved task JSON file."""

    task_path: Path
    task_indices: tuple
    num_tasks_in_file: int


def make_carriers(
    observations: Sequence[Any],
    states: Optional[Sequence[Any]] = None,
    source: str = "dataset",
) -> List[SemanticCarrier]:
    """Wrap observations/states into semantic carriers with stable indices."""
    if states is not None and len(states) != len(observations):
        raise ValueError("states must have the same length as observations")
    carriers = []
    for idx, observation in enumerate(observations):
        carriers.append(
            SemanticCarrier(
                observation=observation,
                state=None if states is None else states[idx],
                index=idx,
                source=source,
            )
        )
    return carriers


def make_semantic_evaluator(
    semantic_evaluator: str,
    region_specs: Sequence[Dict[str, Any]],
    *,
    state_to_point: Callable[[Any], Sequence[float]] = lambda s: s[:2],
    predicate_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    propositions: Optional[Sequence[str]] = None,
):
    key = _normalize_evaluator_backend(semantic_evaluator)
    if key == "state_region":
        return StateRegionEvaluator(region_specs, state_to_point=state_to_point)
    if key == "oracle_state_lift":
        return OracleStateLiftEvaluator(region_specs, state_to_point=state_to_point)
    if key == "object_state":
        specs = predicate_specs or {}
        selected = list(propositions or specs)
        missing = [
            prop for prop in selected
            if _normalize_evaluator_backend(specs.get(prop, {}).get("evaluator")) != "object_state"
        ]
        if missing:
            raise ValueError(f"Object-state predicates required for: {missing}")
        return ObjectStateEvaluator({prop: dict(specs[prop]) for prop in selected})
    raise ValueError(f"Unsupported semantic evaluator: {semantic_evaluator!r}")


def _normalize_evaluator_backend(value: Any) -> str:
    return normalize_evaluator_backend(value)


def default_ltl_env_size(env_name: Optional[str]) -> int:
    """Return the evaluator environment render size for state or visual observations."""
    return 64 if str(env_name or "").startswith("visual-") else 480


def is_scene_env_name(env_name: Optional[str]) -> bool:
    """Return whether an OGBench env is a manipulation scene env."""

    return "scene" in str(env_name or "")


def disable_env_goal_termination_for_ltl(env) -> bool:
    """Disable original goal termination when the rollout is driven by an LTL plan."""

    if hasattr(env.unwrapped, "_terminate_at_goal"):
        env.unwrapped._terminate_at_goal = False
        return True
    return False


def collect_guard_terms(buchi_runtime, states: Optional[Iterable[Any]] = None) -> List[GuardTerm]:
    """Collect unique GuardTerm objects exposed by a BuchiRuntime-like object."""
    if states is None:
        if hasattr(buchi_runtime, "G"):
            states = list(buchi_runtime.G.nodes())
        elif hasattr(buchi_runtime, "edges"):
            states = list(buchi_runtime.edges.keys())
        else:
            raise ValueError("states must be provided for this Buchi runtime")

    guards: List[GuardTerm] = []
    seen = set()
    for q in states:
        for _qn, guard in buchi_runtime.enabled_guard_successors(q):
            if guard.guard_id in seen:
                continue
            seen.add(guard.guard_id)
            guards.append(guard)
    return guards


def build_node_support_lookup(
    support_by_node: Optional[Dict[Any, Sequence[SemanticCarrier]]] = None,
):
    """Return a node-support lookup callable for ProductOracle."""
    support_by_node = dict(support_by_node or {})

    def lookup(node_id):
        return list(support_by_node.get(node_id, []))

    return lookup


def jsonable(value: Any) -> Any:
    """Convert diagnostics containing NumPy scalars/arrays to plain JSON."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    return value


def load_keygraph_from_dir(keygraph_dir: Path):
    """Load a saved `keygraph.pkl` without importing the full GAS stack."""
    keygraph_path = Path(keygraph_dir) / "keygraph.pkl"
    if not keygraph_path.is_file():
        raise FileNotFoundError(f"keygraph.pkl not found: {keygraph_path}")
    with keygraph_path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        key_graph = SimpleNamespace(**data)
    else:
        key_graph = data
    if not hasattr(key_graph, "label_node_idx_dict"):
        key_graph.label_node_idx_dict = {}
    if not hasattr(key_graph, "base_node_cnt"):
        key_graph.base_node_cnt = len(key_graph.nodes)
    return key_graph


def find_keygraph_dir_from_manifest(
    manifest_path: Path,
    target_root: Path,
) -> Path:
    """Find the latest completed KeyGraph directory recorded for a manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    env_name = manifest["env_name"]
    record_path = (
        Path(target_root)
        / "assets"
        / "manifests"
        / f"{env_stem(env_name)}_keygraph_rebuild_command.json"
    )
    if not record_path.is_file():
        raise FileNotFoundError(f"KeyGraph rebuild record not found: {record_path}")
    record = json.loads(record_path.read_text())
    output_root = Path(record["output_root"])
    run_group = record["run_group"]
    candidates = sorted(
        (output_root / run_group).glob(f"{env_name}_sd*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates = [p for p in candidates if (p / "keygraph.pkl").is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No completed keygraph.pkl found under {output_root / run_group}"
        )
    return candidates[0]


def resolve_checkpoint_from_manifest(
    manifest_path: Path,
    target_root: Path,
    key: str,
) -> Path:
    manifest = json.loads(Path(manifest_path).read_text())
    raw = Path(manifest[key])
    manifest_target_root = Path(manifest.get("target_root") or target_root)
    if raw.is_absolute():
        try:
            relocated = Path(target_root) / raw.relative_to(manifest_target_root)
            if relocated.exists():
                return relocated
        except ValueError:
            pass
        return raw
    return Path(target_root) / raw


def _anchor_status_counts(anchor_manager: AnchorManager) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for report in anchor_manager.guard_reports.values():
        counts[report.status] = counts.get(report.status, 0) + 1
    return counts


def _prop_status_counts(anchor_manager: AnchorManager) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for report in anchor_manager.prop_reports.values():
        counts[report.status] = counts.get(report.status, 0) + 1
    return counts


def run_state_semantic_planning(
    *,
    key_graph,
    buchi_runtime,
    region_specs: Sequence[Dict[str, Any]],
    support_observations: Sequence[Any],
    start_observation: Any,
    get_phi: Callable[[Any], Any],
    support_states: Optional[Sequence[Any]] = None,
    support_carriers: Optional[Sequence[SemanticCarrier]] = None,
    start_state: Optional[Any] = None,
    propositions: Optional[Sequence[str]] = None,
    node_support_by_id: Optional[Dict[Any, Sequence[SemanticCarrier]]] = None,
    state_to_point: Callable[[Any], Sequence[float]] = lambda s: s[:2],
    semantic_evaluator: str = "state_region",
    predicate_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    max_prop_anchors: int = 10,
    max_guard_anchors: int = 10,
    max_prefix_candidates: int = 3,
    lambda_weight: float = 1.0,
    anchor_selection_policy: str = "default",
) -> SemanticPlanningResult:
    """Run state-based semantic planning."""
    planning_started = time.perf_counter()
    if support_carriers is None:
        support_carriers = make_carriers(
            support_observations,
            states=support_states,
            source="offline_support",
        )
    else:
        support_carriers = list(support_carriers)
    evaluator = make_semantic_evaluator(
        semantic_evaluator,
        region_specs,
        state_to_point=state_to_point,
        predicate_specs=predicate_specs,
        propositions=propositions,
    )
    ap = list(propositions or evaluator.propositions)
    store = CertificateStore(evaluator, ap)
    if isinstance(evaluator, OracleStateLiftEvaluator) or bool(
        getattr(evaluator, "requires_state", False)
    ):
        missing = sum(1 for carrier in support_carriers if carrier.state is None)
        if missing:
            raise ValueError(
                "oracle_state_lift requires state-paired offline support; "
                f"{missing}/{len(support_carriers)} support carriers have no state."
            )
        if start_state is None:
            raise ValueError("oracle_state_lift requires an explicit start_state.")
    evaluator_ready = time.perf_counter()

    augmentor = KeyGraphSemanticAugmentor(key_graph, get_phi=get_phi, certificate_store=store)
    normalized_anchor_policy = str(anchor_selection_policy or "default").strip().lower()
    candidate_selector = None
    if normalized_anchor_policy in {
        "manipulation_event_diverse",
        "manipulation_event_context_diverse",
    }:
        candidate_selector = ManipulationAnchorSelector()
        normalized_anchor_policy = "manipulation_event_context_diverse"
    elif normalized_anchor_policy not in {"default", "sequential"}:
        raise ValueError(f"Unsupported anchor_selection_policy: {anchor_selection_policy}")
    anchor_manager = AnchorManager(
        store,
        connectivity_fn=lambda carrier: augmentor.has_local_neighbor(carrier),
        candidate_selector=candidate_selector,
    )
    guard_terms = collect_guard_terms(buchi_runtime)
    anchor_manager.build_proposition_anchors(
        support_carriers,
        max_per_prop=max_prop_anchors,
    )
    anchor_manager.build_guard_anchors(
        guard_terms,
        support_carriers,
        max_per_guard=max_guard_anchors,
    )
    inserted = augmentor.insert_anchor_records(
        anchor_manager.anchors.values(),
        manager=anchor_manager,
    )
    augmentor.precompute_dist_to_label_if_available()

    start_carrier = SemanticCarrier(
        observation=start_observation,
        state=start_observation if start_state is None else start_state,
        index=-1,
        source="start",
    )
    start_node = augmentor.insert_start_node(start_carrier)
    start_states = initial_product_states(
        start_node.node_id,
        start_carrier,
        store,
        buchi_runtime,
    )

    oracle = ProductOracle(
        store,
        anchor_manager,
        node_support_lookup=build_node_support_lookup(node_support_by_id),
    )
    product_search_started = time.perf_counter()
    plan = plan_prefix_suffix(
        key_graph.graph,
        buchi_runtime,
        oracle,
        start_states,
        max_prefix_candidates=max_prefix_candidates,
        lambda_weight=lambda_weight,
    )
    planning_finished = time.perf_counter()
    oracle_diagnostics = oracle.diagnostics_summary()
    evaluator_diagnostics = (
        evaluator.diagnostics_summary()
        if hasattr(evaluator, "diagnostics_summary")
        else {"backend": evaluator.__class__.__name__}
    )

    diagnostics = {
        "num_support_carriers": len(support_carriers),
        "semantic_evaluator": evaluator.__class__.__name__,
        "semantic_evaluator_diagnostics": evaluator_diagnostics,
        "anchor_selection_policy": normalized_anchor_policy,
        "anchor_selector_report_summary": {
            key: {
                "scanned": report.scanned,
                "certified": report.certified,
                "connected": report.connected,
                "selected": len(report.selected),
                "metadata": report.metadata,
            }
            for key, report in sorted(anchor_manager.selector_reports.items())
        },
        "num_guard_terms": len(guard_terms),
        "num_anchors": len(anchor_manager.anchors),
        "num_inserted_anchors": sum(
            1 for item in inserted.values() if item.node_id is not None
        ),
        "prop_report_status_counts": _prop_status_counts(anchor_manager),
        "prop_anchor_supported": {
            prop: report.status == "supported"
            for prop, report in sorted(anchor_manager.prop_reports.items())
        },
        "guard_report_status_counts": _anchor_status_counts(anchor_manager),
        "unsupported_guard_terms": sorted(
            guard_id
            for guard_id, report in anchor_manager.guard_reports.items()
            if report.status not in {"supported", "not_anchor_required"}
        ),
        "start_connection_status": start_node.connection_status,
        "num_start_product_states": len(start_states),
        "planner_found_plan": plan is not None,
        "product_oracle_diagnostics": oracle_diagnostics,
        "product_transition_block_counts": oracle_diagnostics["block_counts"],
        "product_blocked_guard_terms": oracle_diagnostics["blocked_guard_terms"],
        "negative_uncertainty_blocks": oracle_diagnostics["negative_uncertainty_blocks"],
        "planning_timing_seconds": {
            "evaluator_setup": evaluator_ready - planning_started,
            "semantic_augmentation": product_search_started - evaluator_ready,
            "product_search": planning_finished - product_search_started,
            "total": planning_finished - planning_started,
        },
        "planning_timing_contract": (
            "task-time evaluator construction, semantic graph augmentation, "
            "and prefix--suffix product search; excludes checkpoint/keygraph "
            "loading, environment reset, and physical rollout"
        ),
    }
    if plan is not None:
        diagnostics.update(
            {
                "prefix_num_states": len(plan.prefix.states),
                "suffix_num_states": len(plan.suffix.states),
                "total_cost": plan.total_cost,
                "guard_sequence": [guard.guard_id for guard in plan.guard_terms],
                "waypoint_nodes": list(plan.waypoint_nodes),
            }
        )

    return SemanticPlanningResult(
        plan=plan,
        start_node_id=start_node.node_id,
        start_product_states=start_states,
        anchor_manager=anchor_manager,
        inserted_anchor_nodes={k: v.node_id for k, v in inserted.items()},
        certificate_store=store,
        diagnostics=diagnostics,
    )


class _MiniNodeView:
    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, node_id):
        return self._graph._nodes[node_id]

    def __iter__(self):
        return iter(self._graph._nodes)


class _MiniDiGraph:
    """Small NetworkX-like directed graph used by the semantic smoke test."""

    def __init__(self):
        self._nodes: Dict[Any, Dict[str, Any]] = {}
        self._edges: Dict[Any, Dict[Any, Dict[str, float]]] = {}
        self.nodes = _MiniNodeView(self)

    def add_node(self, node_id, **attrs):
        self._nodes.setdefault(node_id, {}).update(attrs)

    def add_edge(self, u, v, **attrs):
        self._edges.setdefault(u, {})[v] = dict(attrs)

    def successors(self, u):
        return list(self._edges.get(u, {}))

    def number_of_nodes(self):
        return len(self._nodes)

    def __getitem__(self, u):
        return self._edges[u]


class _MiniKeyGraph:
    def __init__(self, nodes, way_steps=0.8):
        self.nodes = [list(node) for node in nodes]
        self.way_steps = float(way_steps)
        self.base_node_cnt = len(self.nodes)
        self.label_node_idx_dict: Dict[str, List[Any]] = {}
        self.graph = _MiniDiGraph()
        for idx, node in enumerate(self.nodes):
            self.graph.add_node(idx, pos=node)

    def add_base_edge(self, u, v, weight=1.0):
        self.graph.add_edge(u, v, weight=weight)
        self.graph.add_edge(v, u, weight=weight)

    def precompute_dist_to_label(self):
        return None


class _MiniBuchiRuntime:
    def __init__(self):
        self.g_not_a = GuardTerm("g_not_a", "q0", "q0", neg=frozenset({"a"}))
        self.g_a = GuardTerm("g_a", "q0", "q1", pos=frozenset({"a"}))
        self.g_stay = GuardTerm("g_stay", "q1", "q1")
        self.edges = {
            "q0": [("q0", self.g_not_a), ("q1", self.g_a)],
            "q1": [("q1", self.g_stay)],
        }
        self.accepts = {"q1"}

    def enabled_guard_successors(self, q):
        return list(self.edges.get(q, []))

    def is_accepting(self, q):
        return q in self.accepts

    def succ_with_label_set(self, q, label_set):
        labels = set(label_set or [])
        return [
            (qn, guard)
            for qn, guard in self.enabled_guard_successors(q)
            if set(guard.pos).issubset(labels) and not set(guard.neg).intersection(labels)
        ]

    def step_set(self, q_set, label_set):
        out = set()
        for q in q_set:
            for qn, _guard in self.succ_with_label_set(q, label_set):
                out.add(qn)
        return out

    def branch_reaches(self, q_start, label_sequence, q_target):
        q_set = {q_start}
        for labels in label_sequence:
            q_set = self.step_set(q_set, labels)
            if not q_set:
                return False
        return q_target in q_set

    def initial_state_set_after_label(self, label_set):
        return {"q1"} if "a" in set(label_set) else {"q0"}


class _MatrixBuchiRuntime:
    def __init__(self, edges, accepts, init_states=("q0",)):
        self.edges = edges
        self.accepts = set(accepts)
        self.init_states = set(init_states)

    def enabled_guard_successors(self, q):
        return list(self.edges.get(q, []))

    def is_accepting(self, q):
        return q in self.accepts

    def succ_with_label_set(self, q, label_set):
        labels = set(label_set or [])
        return [
            (qn, guard)
            for qn, guard in self.enabled_guard_successors(q)
            if set(guard.pos).issubset(labels) and not set(guard.neg).intersection(labels)
        ]

    def step_set(self, q_set, label_set):
        out = set()
        for q in q_set:
            for qn, _guard in self.succ_with_label_set(q, label_set):
                out.add(qn)
        return out

    def branch_reaches(self, q_start, label_sequence, q_target):
        q_set = {q_start}
        for labels in label_sequence:
            q_set = self.step_set(q_set, labels)
            if not q_set:
                return False
        return q_target in q_set

    def initial_state_set_after_label(self, label_set):
        return self.step_set(self.init_states, label_set)


def run_semantic_smoke() -> Dict[str, Any]:
    """Run a pure-Python state-based semantic planning smoke test."""
    key_graph = _MiniKeyGraph(nodes=[(0.0, 0.0), (1.0, 0.0)], way_steps=0.8)
    key_graph.add_base_edge(0, 1, weight=1.0)
    region_specs = [
        {"id": "a", "kind": "rect", "bounds": (1.45, 1.55, -0.1, 0.1)}
    ]
    support = [(0.0, 0.0), (1.0, 0.0), (1.5, 0.0)]
    node_support = {
        0: [SemanticCarrier(observation=(0.0, 0.0), index=0)],
        1: [SemanticCarrier(observation=(1.0, 0.0), index=1)],
    }
    result = run_state_semantic_planning(
        key_graph=key_graph,
        buchi_runtime=_MiniBuchiRuntime(),
        region_specs=region_specs,
        support_observations=support,
        start_observation=(0.0, 0.0),
        get_phi=lambda obs: obs,
        node_support_by_id=node_support,
        max_prop_anchors=2,
        max_guard_anchors=2,
    )
    if result.plan is None:
        raise RuntimeError(f"semantic smoke failed: {result.diagnostics}")
    return result.diagnostics


def _chain_key_graph(num_nodes: int, way_steps: float = 0.8) -> _MiniKeyGraph:
    key_graph = _MiniKeyGraph(
        nodes=[(float(i), 0.0) for i in range(num_nodes)],
        way_steps=way_steps,
    )
    for idx in range(num_nodes - 1):
        key_graph.add_base_edge(idx, idx + 1, weight=1.0)
    return key_graph


def _node_support_from_graph(key_graph: _MiniKeyGraph) -> Dict[int, List[SemanticCarrier]]:
    return {
        idx: [
            SemanticCarrier(
                observation=tuple(pos),
                state=tuple(pos),
                index=idx,
                source="m1_node_support",
            )
        ]
        for idx, pos in enumerate(key_graph.nodes[: key_graph.base_node_cnt])
    }


def _run_m1_acceptance_case(
    *,
    name: str,
    key_graph: _MiniKeyGraph,
    buchi_runtime,
    region_specs: Sequence[Dict[str, Any]],
    support: Sequence[Any],
    start_observation: Any = (0.0, 0.0),
    expected_guards: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    result = run_state_semantic_planning(
        key_graph=key_graph,
        buchi_runtime=buchi_runtime,
        region_specs=region_specs,
        support_observations=support,
        support_states=support,
        start_observation=start_observation,
        get_phi=lambda obs: obs,
        node_support_by_id=_node_support_from_graph(key_graph),
        max_prop_anchors=4,
        max_guard_anchors=4,
        max_prefix_candidates=4,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics["case"] = name
    diagnostics["passed"] = bool(result.plan is not None)
    if expected_guards is not None:
        expected = set(expected_guards)
        observed = set(diagnostics.get("guard_sequence", []))
        diagnostics["expected_guards_seen"] = sorted(expected.intersection(observed))
        diagnostics["passed"] = diagnostics["passed"] and expected.issubset(observed)
    if not diagnostics["passed"]:
        raise RuntimeError(f"M1 acceptance case failed: {name}: {diagnostics}")
    return diagnostics


def run_m1_acceptance_smoke() -> Dict[str, Any]:
    """Run the pure-Python semantic acceptance checks."""

    wait = GuardTerm("g_wait", "q0", "q0")
    stay = GuardTerm("g_stay", "q1", "q1")

    cases = []

    cases.append(
        _run_m1_acceptance_case(
            name="conference_single_positive",
            key_graph=_chain_key_graph(2),
            buchi_runtime=_MatrixBuchiRuntime(
                {
                    "q0": [
                        ("q0", wait),
                        ("q1", GuardTerm("g_a", "q0", "q1", pos=frozenset({"a"}))),
                    ],
                    "q1": [("q1", stay)],
                },
                accepts={"q1"},
            ),
            region_specs=[{"id": "a", "kind": "rect", "bounds": (1.45, 1.55, -0.1, 0.1)}],
            support=[(0.0, 0.0), (1.0, 0.0), (1.5, 0.0)],
            expected_guards={"g_a"},
        )
    )

    cases.append(
        _run_m1_acceptance_case(
            name="overlap_multi_positive",
            key_graph=_chain_key_graph(2),
            buchi_runtime=_MatrixBuchiRuntime(
                {
                    "q0": [
                        ("q0", wait),
                        (
                            "q1",
                            GuardTerm(
                                "g_a_and_b",
                                "q0",
                                "q1",
                                pos=frozenset({"a", "b"}),
                            ),
                        ),
                    ],
                    "q1": [("q1", stay)],
                },
                accepts={"q1"},
            ),
            region_specs=[
                {"id": "a", "kind": "rect", "bounds": (1.30, 1.70, -0.2, 0.2)},
                {"id": "b", "kind": "circle", "center": (1.5, 0.0), "radius": 0.25},
            ],
            support=[(0.0, 0.0), (1.0, 0.0), (1.5, 0.0)],
            expected_guards={"g_a_and_b"},
        )
    )

    cases.append(
        _run_m1_acceptance_case(
            name="dnf_disjunction_parallel_edges",
            key_graph=_chain_key_graph(2),
            buchi_runtime=_MatrixBuchiRuntime(
                {
                    "q0": [
                        ("q0", wait),
                        ("q1", GuardTerm("g_a_choice", "q0", "q1", pos=frozenset({"a"}))),
                        ("q1", GuardTerm("g_b_choice", "q0", "q1", pos=frozenset({"b"}))),
                    ],
                    "q1": [("q1", stay)],
                },
                accepts={"q1"},
            ),
            region_specs=[
                {"id": "a", "kind": "rect", "bounds": (9.0, 10.0, -0.5, 0.5)},
                {
                    "id": "b",
                    "kind": "polygon",
                    "vertices": [(1.3, -0.2), (1.7, 0.0), (1.3, 0.2)],
                },
            ],
            support=[(0.0, 0.0), (1.0, 0.0), (9.5, 0.0), (1.5, 0.0)],
            expected_guards={"g_b_choice"},
        )
    )

    cases.append(
        _run_m1_acceptance_case(
            name="negative_only",
            key_graph=_chain_key_graph(3),
            buchi_runtime=_MatrixBuchiRuntime(
                {
                    "q_init": [("q0", GuardTerm("g_init", "q_init", "q0"))],
                    "q0": [
                        ("q0", wait),
                        ("q1", GuardTerm("g_not_bad", "q0", "q1", neg=frozenset({"bad"}))),
                    ],
                    "q1": [("q1", stay)],
                },
                accepts={"q1"},
                init_states=("q_init",),
            ),
            region_specs=[{"id": "bad", "kind": "circle", "center": (2.0, 0.0), "radius": 0.2}],
            support=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            expected_guards={"g_not_bad"},
        )
    )

    cases.append(
        _run_m1_acceptance_case(
            name="recurrent_suffix",
            key_graph=_chain_key_graph(2),
            buchi_runtime=_MatrixBuchiRuntime(
                {
                    "q0": [
                        ("q0", wait),
                        ("q1", GuardTerm("g_patrol_a", "q0", "q1", pos=frozenset({"a"}))),
                    ],
                    "q1": [("q0", GuardTerm("g_reset", "q1", "q0"))],
                },
                accepts={"q1"},
            ),
            region_specs=[{"id": "a", "kind": "rect", "bounds": (1.45, 1.55, -0.1, 0.1)}],
            support=[(0.0, 0.0), (1.0, 0.0), (1.5, 0.0)],
            expected_guards={"g_patrol_a", "g_reset"},
        )
    )

    return {
        "mode": "m1_acceptance_smoke",
        "num_cases": len(cases),
        "num_passed": sum(1 for case in cases if case["passed"]),
        "all_passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def _mini_semantic_setup():
    key_graph = _MiniKeyGraph(nodes=[(0.0, 0.0), (1.0, 0.0)], way_steps=0.8)
    key_graph.add_base_edge(0, 1, weight=1.0)
    region_specs = [
        {"id": "a", "kind": "rect", "bounds": (1.45, 1.55, -0.1, 0.1)}
    ]
    support = [(0.0, 0.0), (1.0, 0.0), (1.5, 0.0)]
    node_support = {
        0: [SemanticCarrier(observation=(0.0, 0.0), state=(0.0, 0.0), index=0)],
        1: [SemanticCarrier(observation=(1.0, 0.0), state=(1.0, 0.0), index=1)],
    }
    buchi_runtime = _MiniBuchiRuntime()
    result = run_state_semantic_planning(
        key_graph=key_graph,
        buchi_runtime=buchi_runtime,
        region_specs=region_specs,
        support_observations=support,
        support_states=support,
        start_observation=(0.0, 0.0),
        get_phi=lambda obs: obs,
        node_support_by_id=node_support,
        max_prop_anchors=2,
        max_guard_anchors=2,
    )
    return result, buchi_runtime


def run_execution_smoke() -> Dict[str, Any]:
    """Run semantic planning, finite execution monitoring, and oracle checking."""
    result, buchi_runtime = _mini_semantic_setup()
    if result.plan is None:
        raise RuntimeError(f"execution smoke planning failed: {result.diagnostics}")
    execution_plan = materialize_finite_lasso(result.plan, suffix_cycles=2)
    runtime_monitor = RuntimeMonitor(result.certificate_store)

    def segment_executor(request):
        if request.guard.pos:
            carrier = SemanticCarrier(
                observation=(1.5, 0.0),
                state=(1.5, 0.0),
                index=100 + request.transition_index,
                source="execution_smoke",
            )
        else:
            carrier = SemanticCarrier(
                observation=(0.0, 0.0),
                state=(0.0, 0.0),
                index=100 + request.transition_index,
                source="execution_smoke",
            )
        return SegmentExecutionResult(
            carriers=[carrier],
            waypoint_reached=True,
            actions=[request.transition_index],
        )

    execution = execute_finite_product_plan(
        execution_plan=execution_plan,
        segment_executor=segment_executor,
        runtime_monitor=runtime_monitor,
        buchi_runtime=buchi_runtime,
    )

    def oracle_labeler(state):
        x, y = state
        return {"a"} if 1.45 <= x <= 1.55 and -0.1 <= y <= 0.1 else set()

    oracle = oracle_finite_lasso_verdict(
        execution.trace.states,
        oracle_labeler,
        buchi_runtime,
        prefix_completed=execution.prefix_completed,
        suffix_cycles_completed=execution.suffix_cycles_completed,
        suffix_cycles_required=execution.suffix_cycles_required,
        execution_plan=execution_plan,
        product_progress=execution.trace.product_progress,
    )
    semantic_grounding_error = bool(
        execution.finite_lasso_success and not oracle["oracle_success"]
    )
    diagnostics = {
        "planner_found_plan": result.plan is not None,
        "runtime_finite_lasso_success": execution.finite_lasso_success,
        "runtime_failure_reason": execution.failure_reason,
        "oracle_success": oracle["oracle_success"],
        "oracle_failure_reason": oracle_failure_reason_with_grounding(
            execution.finite_lasso_success,
            oracle,
        ),
        "oracle_plan_checked": oracle.get("oracle_plan_checked", False),
        "oracle_prefix_completed": oracle.get("oracle_prefix_completed"),
        "oracle_suffix_cycles_completed": oracle.get("oracle_suffix_cycles_completed"),
        "semantic_grounding_error": semantic_grounding_error,
        "prefix_completed": execution.prefix_completed,
        "suffix_cycles_completed": execution.suffix_cycles_completed,
        "suffix_cycles_required": execution.suffix_cycles_required,
        "num_trace_states": len(execution.trace.states),
        "num_transition_results": len(execution.transition_results),
        "guard_sequence": [g.guard_id for g in execution_plan.guards],
    }
    if not execution.finite_lasso_success or not oracle["oracle_success"]:
        raise RuntimeError(f"execution smoke failed: {diagnostics}")
    return diagnostics


def _as_list(vec: Any) -> List[float]:
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]


def _choose_far_latent_indices(nodes: Sequence[Any], base_count: int) -> tuple[int, int]:
    if base_count <= 1:
        return 0, 0
    start_idx = 0
    start = _as_list(nodes[start_idx])
    best_idx = 1
    best_dist = -math.inf
    for idx in range(1, base_count):
        pos = _as_list(nodes[idx])
        dist = math.sqrt(sum((a - b) * (a - b) for a, b in zip(start, pos)))
        if dist > best_dist:
            best_dist = dist
            best_idx = idx
    return start_idx, best_idx


def make_latent_region_spec(center: Sequence[float], radius: float) -> Dict[str, Any]:
    x, y = float(center[0]), float(center[1])
    r = float(radius)
    return {"id": "a", "kind": "rect", "bounds": (x - r, x + r, y - r, y + r)}


def run_real_graph_smoke(
    *,
    keygraph_dir: Path,
    start_index: Optional[int] = None,
    target_index: Optional[int] = None,
    region_radius: Optional[float] = None,
    max_support_nodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Run semantic planning over a rebuilt KeyGraph without environment rollout.

    This is a structural integration smoke test.  It loads the real graph and
    exercises semantic insertion plus product planning, while using latent-space
    rectangular predicates so no TDR agent needs to be reconstructed here.
    """
    key_graph = load_keygraph_from_dir(keygraph_dir)
    base_count = int(getattr(key_graph, "base_node_cnt", len(key_graph.nodes)))
    if base_count <= 0:
        raise ValueError("Loaded KeyGraph has no base nodes.")
    default_start, default_target = _choose_far_latent_indices(key_graph.nodes, base_count)
    start_index = default_start if start_index is None else int(start_index)
    target_index = default_target if target_index is None else int(target_index)
    if not (0 <= start_index < base_count):
        raise ValueError(f"start_index out of range: {start_index}")
    if not (0 <= target_index < base_count):
        raise ValueError(f"target_index out of range: {target_index}")

    way_steps = float(getattr(key_graph, "way_steps", 1.0))
    radius = float(region_radius if region_radius is not None else max(1e-3, 0.05 * way_steps))
    edge_count = (
        int(key_graph.graph.number_of_edges())
        if hasattr(key_graph.graph, "number_of_edges")
        else None
    )
    support_limit = base_count if max_support_nodes is None else min(base_count, int(max_support_nodes))
    support_observations = [_as_list(key_graph.nodes[i]) for i in range(support_limit)]
    if target_index >= support_limit:
        support_observations.append(_as_list(key_graph.nodes[target_index]))
    start_observation = _as_list(key_graph.nodes[start_index])
    target_position = _as_list(key_graph.nodes[target_index])

    node_support = {
        i: [SemanticCarrier(observation=_as_list(key_graph.nodes[i]), index=i)]
        for i in range(support_limit)
    }
    if target_index not in node_support:
        node_support[target_index] = [
            SemanticCarrier(observation=target_position, index=target_index)
        ]

    result = run_state_semantic_planning(
        key_graph=key_graph,
        buchi_runtime=_MiniBuchiRuntime(),
        region_specs=[make_latent_region_spec(target_position, radius)],
        support_observations=support_observations,
        start_observation=start_observation,
        get_phi=lambda obs: obs,
        node_support_by_id=node_support,
        max_prop_anchors=4,
        max_guard_anchors=4,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "mode": "real_graph_smoke",
            "keygraph_dir": str(keygraph_dir),
            "num_graph_nodes_before_semantic_insert": base_count,
            "num_graph_edges_before_semantic_insert": edge_count,
            "start_index": start_index,
            "target_index": target_index,
            "region_radius": radius,
        }
    )
    if result.plan is None:
        raise RuntimeError(f"real graph smoke failed: {diagnostics}")
    return diagnostics


def load_tdr_agent_and_dataset(
    *,
    manifest_path: Path,
    target_root: Path,
    example_observation: Any,
    env_name: Optional[str] = None,
    action_dim: Optional[int] = None,
    seed: int = 0,
    gpu: int = 0,
):
    """Load the copied TDR agent without constructing a MuJoCo environment."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import random
    import numpy as np

    from M_utils.agents import agents_dict
    from M_utils.agents.gas import get_config
    from M_utils.agent_config_utils import default_action_dim, resolve_agent_config_for_env
    from M_utils.flax_utils import restore_agent

    manifest = json.loads(Path(manifest_path).read_text())
    env_name = env_name or manifest["env_name"]
    if action_dim is None:
        action_dim = default_action_dim(env_name)
    tdr_path = resolve_checkpoint_from_manifest(manifest_path, target_root, "tdr_path")
    if not tdr_path.is_file():
        raise FileNotFoundError(f"TDR checkpoint not found: {tdr_path}")

    random.seed(seed)
    np.random.seed(seed)
    config = resolve_agent_config_for_env(
        get_config(),
        env_name,
        overrides=manifest.get("agent_config_overrides"),
    )
    example_observations = np.asarray(example_observation)[None]
    example_actions = np.zeros((1, int(action_dim)), dtype=np.float32)
    agent_class = agents_dict[config["agent_name"]]
    agent = agent_class.create(
        seed,
        example_observations,
        example_actions,
        config,
    )
    restore_dir = str(tdr_path.parent)
    restore_epoch = tdr_path.stem.split("_")[-1]
    agent = restore_agent(agent, restore_dir, restore_epoch)

    def get_phi_single(observation):
        arr = np.asarray(observation)[None]
        return np.asarray(agent.get_phi(arr))[0]

    def get_phi_batch(observations):
        return np.asarray(agent.get_phi(np.asarray(observations)))

    return {
        "env_name": env_name,
        "config": config,
        "get_phi": get_phi_single,
        "get_phi_batch": get_phi_batch,
        "tdr_path": str(tdr_path),
        "action_dim": int(action_dim),
    }


def load_policy_agent_backend(
    *,
    manifest_path: Path,
    target_root: Path,
    example_observation: Any,
    env_name: Optional[str] = None,
    action_dim: Optional[int] = None,
    seed: int = 0,
    gpu: int = 0,
):
    """Load the copied low-level policy checkpoint for real rollout smoke tests."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import random
    import numpy as np
    import jax

    from M_utils.agents import agents_dict
    from M_utils.agents.gas import get_config
    from M_utils.agent_config_utils import default_action_dim, resolve_agent_config_for_env
    from M_utils.flax_utils import restore_agent

    manifest = json.loads(Path(manifest_path).read_text())
    env_name = env_name or manifest["env_name"]
    if action_dim is None:
        action_dim = default_action_dim(env_name)
    policy_path = resolve_checkpoint_from_manifest(manifest_path, target_root, "policy_path")
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")

    random.seed(seed)
    np.random.seed(seed)
    config = resolve_agent_config_for_env(
        get_config(),
        env_name,
        overrides=manifest.get("agent_config_overrides"),
    )
    example_observations = np.asarray(example_observation)[None]
    example_actions = np.zeros((1, int(action_dim)), dtype=np.float32)
    agent_class = agents_dict[config["agent_name"]]
    agent = agent_class.create(
        seed,
        example_observations,
        example_actions,
        config,
    )
    restore_dir = str(policy_path.parent)
    restore_epoch = policy_path.stem.split("_")[-1]
    agent = restore_agent(agent, restore_dir, restore_epoch)
    rng = jax.random.PRNGKey(seed)

    def get_phi_single(observation):
        arr = np.asarray(observation)[None]
        return np.asarray(agent.get_phi(arr))[0]

    def get_phi_batch(observations):
        return np.asarray(agent.get_phi(np.asarray(observations)))

    def sample_action(observation, skill, temperature: float = 0.0):
        nonlocal rng
        rng, key = jax.random.split(rng)
        obs_arr = np.asarray(observation)[None]
        skill_arr = np.asarray(skill, dtype=np.float32)[None]
        action = agent.sample_actions(
            observations=obs_arr,
            goals=skill_arr,
            temperature=float(temperature),
            seed=key,
        )
        return np.asarray(action)[0]

    return {
        "env_name": env_name,
        "config": config,
        "get_phi": get_phi_single,
        "get_phi_batch": get_phi_batch,
        "sample_action": sample_action,
        "policy_path": str(policy_path),
        "action_dim": int(action_dim),
    }


def collect_node_support_from_keygraph(
    key_graph,
    *,
    max_members_per_node: int = 4,
) -> Dict[Any, List[SemanticCarrier]]:
    key_nodes = getattr(key_graph, "key_nodes", None)
    cluster_members = getattr(key_nodes, "cluster_members_raw", None) if key_nodes is not None else None
    if cluster_members is None and key_nodes is not None:
        cluster_members = getattr(key_nodes, "cluster_members", None)
    cluster_member_indices = (
        getattr(key_nodes, "cluster_members_indices", None) if key_nodes is not None else None
    )
    if cluster_members is None:
        raise ValueError("Loaded KeyGraph does not contain cluster member support.")

    def member_original_index(node_id: int, local_idx: int) -> Optional[int]:
        if cluster_member_indices is None:
            return None
        try:
            raw_indices = cluster_member_indices.get(node_id, [])
        except AttributeError:
            try:
                raw_indices = cluster_member_indices[node_id]
            except (IndexError, KeyError, TypeError):
                return None
        if local_idx >= len(raw_indices):
            return None
        try:
            return int(raw_indices[local_idx])
        except (TypeError, ValueError):
            return None

    def carrier_index(node_id: int, local_idx: int, original_index: Optional[int]) -> int:
        return int(original_index) if original_index is not None else node_id * 100000 + local_idx

    def carrier_from_member(
        member,
        *,
        node_id: int,
        local_idx: int,
        original_index: Optional[int],
    ) -> SemanticCarrier:
        if isinstance(member, SemanticCarrier):
            if member.index is None and original_index is not None:
                member.index = int(original_index)
                member.info.setdefault("original_index", int(original_index))
                member.info.setdefault("node_id", int(node_id))
                member.info.setdefault("local_idx", int(local_idx))
            return member
        if isinstance(member, dict) and "observation" in member:
            state = member.get("state")
            info = {
                key: value
                for key, value in member.items()
                if key not in {"observation", "state"}
            }
            if original_index is not None:
                info.setdefault("original_index", int(original_index))
            info.setdefault("node_id", int(node_id))
            info.setdefault("local_idx", int(local_idx))
            return SemanticCarrier(
                observation=member["observation"],
                state=state,
                info=info,
                index=carrier_index(node_id, local_idx, original_index),
                source="keygraph_cluster",
            )
        return SemanticCarrier(
            observation=member,
            state=member,
            info={
                **({} if original_index is None else {"original_index": int(original_index)}),
                "node_id": int(node_id),
                "local_idx": int(local_idx),
            },
            index=carrier_index(node_id, local_idx, original_index),
            source="keygraph_cluster",
        )

    support_by_node: Dict[Any, List[SemanticCarrier]] = {}
    for node_id in range(int(getattr(key_graph, "base_node_cnt", len(key_graph.nodes)))):
        raw_members = list(cluster_members.get(node_id, []))[:max_members_per_node]
        support_by_node[node_id] = [
            carrier_from_member(
                member,
                node_id=int(node_id),
                local_idx=local_idx,
                original_index=member_original_index(int(node_id), local_idx),
            )
            for local_idx, member in enumerate(raw_members)
        ]
    return support_by_node


def flatten_support_for_anchor_search(
    support_by_node: Dict[Any, List[SemanticCarrier]],
    *,
    priority_nodes: Sequence[int] = (),
    max_total: int = 5000,
) -> List[SemanticCarrier]:
    ordered_nodes = list(dict.fromkeys([int(x) for x in priority_nodes] + sorted(support_by_node)))
    out: List[SemanticCarrier] = []
    seen = set()
    for node_id in ordered_nodes:
        for carrier in support_by_node.get(node_id, []):
            key = carrier.index if carrier.index is not None else id(carrier)
            if key in seen:
                continue
            seen.add(key)
            out.append(carrier)
            if len(out) >= max_total:
                return out
    return out


def run_state_graph_smoke(
    *,
    manifest_path: Path,
    target_root: Path,
    keygraph_dir: Optional[Path] = None,
    env_name: Optional[str] = None,
    seed: int = 0,
    gpu: int = 0,
    action_dim: Optional[int] = None,
    start_index: Optional[int] = None,
    target_index: Optional[int] = None,
    region_radius: float = 1.0,
    max_members_per_node: int = 4,
    max_anchor_support: int = 5000,
) -> Dict[str, Any]:
    """Run real-KeyGraph planning with task-space support and TDR embedding."""
    keygraph_dir = keygraph_dir or find_keygraph_dir_from_manifest(manifest_path, target_root)
    key_graph = load_keygraph_from_dir(keygraph_dir)
    base_count = int(getattr(key_graph, "base_node_cnt", len(key_graph.nodes)))
    default_start, default_target = _choose_far_latent_indices(key_graph.nodes, base_count)
    start_index = default_start if start_index is None else int(start_index)
    target_index = default_target if target_index is None else int(target_index)

    support_by_node = collect_node_support_from_keygraph(
        key_graph,
        max_members_per_node=max_members_per_node,
    )
    if not support_by_node.get(start_index):
        raise ValueError(f"No cluster support for start_index={start_index}")
    if not support_by_node.get(target_index):
        raise ValueError(f"No cluster support for target_index={target_index}")

    start_carrier = support_by_node[start_index][0]
    target_carrier = support_by_node[target_index][0]
    backend = load_tdr_agent_and_dataset(
        manifest_path=manifest_path,
        target_root=target_root,
        example_observation=start_carrier.observation,
        env_name=env_name,
        action_dim=action_dim,
        seed=seed,
        gpu=gpu,
    )
    tx, ty = float(target_carrier.state[0]), float(target_carrier.state[1])
    region_specs = [
        {
            "id": "a",
            "kind": "rect",
            "bounds": (
                tx - float(region_radius),
                tx + float(region_radius),
                ty - float(region_radius),
                ty + float(region_radius),
            ),
        }
    ]
    anchor_support = flatten_support_for_anchor_search(
        support_by_node,
        priority_nodes=[target_index, start_index],
        max_total=max_anchor_support,
    )

    result = run_state_semantic_planning(
        key_graph=key_graph,
        buchi_runtime=_MiniBuchiRuntime(),
        region_specs=region_specs,
        support_observations=[c.observation for c in anchor_support],
        support_states=[c.state for c in anchor_support],
        start_observation=start_carrier.observation,
        get_phi=backend["get_phi"],
        node_support_by_id=support_by_node,
        max_prop_anchors=4,
        max_guard_anchors=4,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "mode": "state_graph_smoke",
            "env_name": backend["env_name"],
            "keygraph_dir": str(keygraph_dir),
            "tdr_path": backend["tdr_path"],
            "action_dim": backend["action_dim"],
            "num_graph_nodes_before_semantic_insert": base_count,
            "num_anchor_support_carriers": len(anchor_support),
            "start_index": start_index,
            "target_index": target_index,
            "region_radius": float(region_radius),
        }
    )
    if result.plan is None:
        raise RuntimeError(f"state graph smoke failed: {diagnostics}")
    return diagnostics


def build_buchi_runtime_from_formula(formula: str, propositions: Sequence[str]):
    """Build a BuchiRuntime from a direct proposition-level LTL formula."""
    from L_utils.buchi_parse import Buchi
    from L_utils.ltl_runtime import BuchiRuntime

    class _Task:
        def __init__(self, formula_text):
            self.formula = formula_text
            self.subformula = {}
            self.number_of_robots = 1

    buchi = Buchi(_Task(formula))
    buchi.construct_buchi_graph()
    label_to_literal = {prop: prop for prop in propositions}
    return BuchiRuntime(buchi, label_to_literal)


def _xy_of(carrier: SemanticCarrier) -> tuple[float, float]:
    state = carrier.state if carrier.state is not None else carrier.observation
    return float(state[0]), float(state[1])


def choose_real_smoke_node_pair(
    key_graph,
    support_by_node: Dict[Any, List[SemanticCarrier]],
    *,
    min_xy_distance: float = 8.0,
    max_xy_distance: float = 16.0,
    min_path_hops: int = 3,
    max_path_hops: int = 10,
) -> tuple[int, int]:
    """Choose a nontrivial reachable start-target pair for the real smoke task."""
    import networkx as nx

    base_count = int(getattr(key_graph, "base_node_cnt", len(key_graph.nodes)))
    xy_by_node = {
        int(node_id): _xy_of(carriers[0])
        for node_id, carriers in support_by_node.items()
        if int(node_id) < base_count and carriers
    }
    nodes = sorted(xy_by_node)
    if not nodes:
        raise ValueError("Could not find support-backed base graph nodes.")

    max_sources = 180
    stride = max(1, len(nodes) // max_sources)
    source_nodes = nodes[::stride]
    preferred_distance = 0.5 * (float(min_xy_distance) + float(max_xy_distance))
    best = None
    best_score = -float("inf")
    for u in source_nodes:
        try:
            hop_lengths = nx.single_source_shortest_path_length(
                key_graph.graph,
                u,
                cutoff=int(max_path_hops),
            )
        except Exception:
            continue
        ux, uy = xy_by_node[u]
        for v, hops in hop_lengths.items():
            v = int(v)
            if v == u or v not in xy_by_node:
                continue
            if int(hops) < int(min_path_hops):
                continue
            vx, vy = _xy_of(support_by_node[v][0])
            dist = math.sqrt((ux - vx) * (ux - vx) + (uy - vy) * (uy - vy))
            if min_xy_distance <= dist <= max_xy_distance:
                score = (
                    -abs(dist - preferred_distance)
                    + 0.35 * float(hops)
                    + 0.05 * dist
                )
                if score > best_score:
                    best = (u, v, int(hops), float(dist))
                    best_score = score
    if best is None:
        fallback_min = min(5.0, float(min_xy_distance))
        fallback_max = max(float(max_xy_distance), 8.0)
        for u in nodes:
            ux, uy = xy_by_node[u]
            for v in nodes:
                if v == u:
                    continue
                vx, vy = xy_by_node[v]
                dist = math.sqrt((ux - vx) * (ux - vx) + (uy - vy) * (uy - vy))
                if fallback_min <= dist <= fallback_max:
                    return u, v
        raise ValueError("Could not find a support-backed start-target pair.")
    return int(best[0]), int(best[1])


def _line_projection_stats(
    point_xy: Sequence[float],
    start_xy: Sequence[float],
    target_xy: Sequence[float],
) -> tuple[float, float]:
    px, py = float(point_xy[0]), float(point_xy[1])
    sx, sy = float(start_xy[0]), float(start_xy[1])
    tx, ty = float(target_xy[0]), float(target_xy[1])
    dx, dy = tx - sx, ty - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return 0.0, math.sqrt((px - sx) * (px - sx) + (py - sy) * (py - sy))
    t = ((px - sx) * dx + (py - sy) * dy) / denom
    proj_x, proj_y = sx + t * dx, sy + t * dy
    line_dist = math.sqrt((px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y))
    return float(t), float(line_dist)


def choose_real_smoke_avoid_node(
    support_by_node: Dict[Any, List[SemanticCarrier]],
    *,
    start_index: int,
    target_index: int,
    target_radius: float,
    avoid_radius: float,
) -> int:
    """Choose a forbidden-label node near the start-target corridor."""
    sx, sy = _xy_of(support_by_node[int(start_index)][0])
    tx, ty = _xy_of(support_by_node[int(target_index)][0])
    min_target_gap = float(target_radius) + float(avoid_radius) + 1.0
    best = None
    best_score = -float("inf")
    for node_id, carriers in support_by_node.items():
        node_id = int(node_id)
        if node_id in {int(start_index), int(target_index)} or not carriers:
            continue
        ax, ay = _xy_of(carriers[0])
        dist_start = math.sqrt((ax - sx) * (ax - sx) + (ay - sy) * (ay - sy))
        dist_target = math.sqrt((ax - tx) * (ax - tx) + (ay - ty) * (ay - ty))
        if dist_start < min_target_gap or dist_target < min_target_gap:
            continue
        corridor_t, line_dist = _line_projection_stats((ax, ay), (sx, sy), (tx, ty))
        if not 0.35 <= corridor_t <= 0.82:
            continue
        if line_dist > 7.5:
            continue
        score = (
            -1.5 * abs(line_dist - 3.0)
            -8.0 * abs(corridor_t - 0.58)
            -0.35 * abs(dist_target - 6.5)
        )
        if score > best_score:
            best = node_id
            best_score = score
    if best is not None:
        return int(best)

    best = None
    best_score = float("inf")
    for node_id, carriers in support_by_node.items():
        node_id = int(node_id)
        if node_id in {int(start_index), int(target_index)} or not carriers:
            continue
        ax, ay = _xy_of(carriers[0])
        dist_target = math.sqrt((ax - tx) * (ax - tx) + (ay - ty) * (ay - ty))
        if dist_target < min_target_gap:
            continue
        _t, line_dist = _line_projection_stats((ax, ay), (sx, sy), (tx, ty))
        score = line_dist + 0.1 * abs(dist_target - 7.0)
        if score < best_score:
            best = node_id
            best_score = score
    if best is None:
        raise ValueError("Could not find a supported forbidden-label node.")
    return int(best)


def make_real_smoke_region_specs(
    *,
    target_xy: Sequence[float],
    avoid_xy: Sequence[float],
    target_radius: float = 2.2,
    avoid_radius: float = 1.5,
) -> List[Dict[str, Any]]:
    """Create overlapping target predicates plus a separate forbidden predicate."""
    tx, ty = float(target_xy[0]), float(target_xy[1])
    ax, ay = float(avoid_xy[0]), float(avoid_xy[1])
    poly_r = float(target_radius) * 1.15
    return [
        {
            "id": "a",
            "kind": "circle",
            "center": (tx, ty),
            "radius": float(target_radius),
        },
        {
            "id": "b",
            "kind": "polygon",
            "vertices": [
                (tx - poly_r, ty),
                (tx - 0.2 * poly_r, ty + 0.9 * poly_r),
                (tx + poly_r, ty + 0.25 * poly_r),
                (tx + 0.55 * poly_r, ty - 0.95 * poly_r),
            ],
        },
        {
            "id": "c",
            "kind": "circle",
            "center": (ax, ay),
            "radius": float(avoid_radius),
        },
    ]


def real_smoke_render_regions(region_specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert real-smoke region specs to existing LTL render region objects."""
    from L_utils.ltl_task_utils import region_from_dict

    colors = {
        "a": [0.1, 0.45, 0.95, 0.55],
        "b": [0.05, 0.75, 0.35, 0.55],
        "c": [0.95, 0.1, 0.1, 0.45],
    }
    regions = {}
    for spec in region_specs:
        prop = str(spec["id"])
        kind = str(spec["kind"])
        if kind == "circle":
            data = {
                "type": "circle",
                "center": list(spec["center"]),
                "radius": float(spec["radius"]),
                "color": colors.get(prop, [1.0, 1.0, 1.0, 0.5]),
            }
        elif kind == "polygon":
            data = {
                "type": "polygon",
                "vertices": [list(v) for v in spec["vertices"]],
                "color": colors.get(prop, [1.0, 1.0, 1.0, 0.5]),
            }
        elif kind in {"rect", "rectangle"}:
            xmin, xmax, ymin, ymax = spec["bounds"]
            data = {
                "type": "rect",
                "xmin": float(xmin),
                "xmax": float(xmax),
                "ymin": float(ymin),
                "ymax": float(ymax),
                "color": colors.get(prop, [1.0, 1.0, 1.0, 0.5]),
            }
        else:
            continue
        regions[prop] = region_from_dict(data)
    return regions


def _state_xy_point(state: Any):
    if isinstance(state, Mapping):
        if "xy" in state:
            return state["xy"][:2]
        if "state" in state:
            return _state_xy_point(state["state"])
        if "qpos" in state:
            return state["qpos"][:2]
    return state[:2]


def _region_labeler(region_specs: Sequence[Dict[str, Any]]):
    evaluator = StateRegionEvaluator(region_specs, state_to_point=_state_xy_point)

    def labeler(state):
        carrier = SemanticCarrier(observation=state, state=state, source="oracle")
        return {
            prop
            for prop in evaluator.propositions
            if evaluator.evaluate(carrier, prop).cert_pos
        }

    return labeler


def _oracle_labeler_for_task(
    region_specs: Sequence[Dict[str, Any]],
    predicate_specs: Optional[Mapping[str, Mapping[str, Any]]] = None,
):
    """Build the final verifier labeler from declared task predicates.

    Region predicates use the state-space geometry evaluator. Manipulation
    object-state predicates use privileged scene facts from the recorded
    trajectory state. This oracle is intentionally separate from the
    method-internal runtime evaluator.
    """

    predicate_specs = dict(predicate_specs or {})
    object_specs = {
        str(prop): dict(spec)
        for prop, spec in predicate_specs.items()
        if normalize_evaluator_backend(spec.get("evaluator", "state_region"))
        == "object_state"
    }
    region_labeler = _region_labeler(region_specs)

    def labeler(state):
        labels = set(region_labeler(state))
        if object_specs:
            labels.update(scene_object_label_set(state, object_specs))
        return labels

    return labeler


def reset_env_at_xy(env, seed: int, xy: Sequence[float]):
    """Reset a maze environment and best-effort align the simulator xy state."""
    import numpy as np

    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        observation, _info = out
    else:
        observation = out
    xy = (float(xy[0]), float(xy[1]))
    if hasattr(env.unwrapped, "set_xy"):
        try:
            env.unwrapped.set_xy(xy)
        except TypeError:
            env.unwrapped.set_xy(xy[0], xy[1])
    if hasattr(env.unwrapped, "get_ob"):
        try:
            observation = env.unwrapped.get_ob()
        except TypeError:
            observation = env.unwrapped.get_ob(None)
    observation = np.asarray(observation).copy()
    if observation.ndim == 1 and observation.shape[0] >= 2:
        observation[0], observation[1] = xy
    return observation


def extract_task_state(
    env,
    observation: Any,
    info: Optional[Dict[str, Any]] = None,
    *,
    strict: bool = False,
) -> tuple[Any, str]:
    """Extract the task state used by semantic evaluators.

    In state-based runs, the deployable observation can also serve as the task
    state.  In visual runs, callers should set ``strict=True`` so semantic
    grounding fails if the environment does not expose privileged state.
    """
    import numpy as np

    info = dict(info or {})
    unwrapped = getattr(env, "unwrapped", env)
    scene_keys = {
        key: value
        for key, value in info.items()
        if key in {"qpos", "qvel", "button_states", "prev_button_states"}
        or str(key).startswith("privileged/")
    }
    if scene_keys and (
        "button_states" in scene_keys
        or any(str(key).startswith("privileged/button_") for key in scene_keys)
        or any(str(key).startswith("privileged/block_") for key in scene_keys)
    ):
        return scene_keys, "info.scene_object_state"
    if hasattr(unwrapped, "compute_ob_info"):
        try:
            ob_info = dict(unwrapped.compute_ob_info())
        except Exception:
            ob_info = {}
        scene_info = {
            key: value
            for key, value in ob_info.items()
            if key in {"qpos", "qvel", "button_states", "prev_button_states"}
            or str(key).startswith("privileged/")
        }
        if scene_info and (
            "button_states" in scene_info
            or any(str(key).startswith("privileged/button_") for key in scene_info)
            or any(str(key).startswith("privileged/block_") for key in scene_info)
        ):
            return scene_info, "env.compute_ob_info"
    for key in ("qpos", "state", "xy"):
        if key in info and info[key] is not None:
            return np.asarray(info[key]).copy(), f"info.{key}"
    if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "qpos"):
        return np.asarray(unwrapped.data.qpos).copy(), "env.data.qpos"
    if hasattr(unwrapped, "_data") and hasattr(unwrapped._data, "qpos"):
        return np.asarray(unwrapped._data.qpos).copy(), "env._data.qpos"
    if hasattr(unwrapped, "get_xy"):
        return np.asarray(unwrapped.get_xy()).copy(), "env.get_xy"
    if strict:
        raise ValueError(
            "Oracle state-lift evaluator requested, but no privileged task "
            "state was available from info/env."
        )
    return np.asarray(observation).copy(), "observation"


def build_anchor_positions_by_label(key_graph) -> Dict[str, List[Any]]:
    anchors = {}
    for label, node_ids in getattr(key_graph, "label_node_idx_dict", {}).items():
        positions = []
        for node_id in node_ids:
            try:
                positions.append(key_graph.nodes[int(node_id)])
            except Exception:
                continue
        anchors[label] = positions
    return anchors


def execution_waypoint_latents(key_graph, execution_plan) -> List[Any]:
    """Return latent graph positions aligned with an execution plan's states."""

    return [key_graph.nodes[int(node_id)] for node_id, _q in execution_plan.states]


def semantic_checkpoint_indices(execution_plan) -> List[int]:
    """State indices whose incoming product edge carries positive progress."""

    return [
        idx + 1
        for idx, guard in enumerate(execution_plan.guards)
        if bool(guard.pos)
    ]


def guard_span_obligations(execution_plan, start_edge_index: int, target_state_index: int):
    """Union positive and negative obligations for a planned transition span."""

    pos = set()
    neg = set()
    for guard in execution_plan.guards[int(start_edge_index):int(target_state_index)]:
        pos.update(guard.pos)
        neg.update(guard.neg)
    return pos, neg


def normalize_execution_refresh_mode(mode: Optional[str]) -> str:
    """Normalize opt-in execution refresh modes."""

    normalized = str(mode or "none").strip().lower()
    if normalized in {"", "off", "false", "0"}:
        normalized = "none"
    if normalized not in {"none", "gas_target"}:
        raise ValueError(
            "execution_refresh_mode must be one of {'none', 'gas_target'}, "
            f"got {mode!r}"
        )
    return normalized


def pending_semantic_target_index(
    *,
    current_index: int,
    execution_plan,
    semantic_checkpoint_indices: Sequence[int],
) -> int:
    """Return the next semantic checkpoint, or the immediate successor."""

    idx0 = max(0, min(int(current_index), len(execution_plan.states) - 1))
    for checkpoint in sorted(int(i) for i in semantic_checkpoint_indices):
        if checkpoint >= idx0:
            return min(checkpoint, len(execution_plan.states) - 1)
    return idx0


def _latent_distance(a: Any, b: Any) -> float:
    aa = _as_list(a)
    bb = _as_list(b)
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(aa, bb)))


def _keygraph_edge_weight(key_graph, u: Any, v: Any) -> float:
    try:
        return float(key_graph.graph[int(u)][int(v)].get("weight", 0.0))
    except Exception:
        return _latent_distance(key_graph.nodes[int(u)], key_graph.nodes[int(v)])


def _select_refreshed_path_target(
    *,
    current_latent: Any,
    path_latents: Sequence[Any],
    reach_threshold: float,
) -> Any:
    """Choose a GAS-style local waypoint from a refreshed latent path."""

    if not path_latents:
        return current_latent
    if len(path_latents) <= 2:
        return path_latents[-1]
    distances = [_latent_distance(current_latent, latent) for latent in path_latents]
    valid = [
        idx for idx, dist in enumerate(distances)
        if float(dist) <= float(reach_threshold)
    ]
    target_idx = valid[-1] if valid else 0
    return path_latents[target_idx]


def refreshed_path_to_target_node(
    *,
    key_graph,
    current_latent: Any,
    target_node: Any,
    local_radius: float,
) -> Dict[str, Any]:
    """Find a temporary local path from the current latent state to a graph node."""

    try:
        import networkx as nx
    except ModuleNotFoundError:
        return {
            "success": False,
            "reason": "networkx_unavailable",
            "path_nodes": [],
            "path_latents": [],
        }

    target = int(target_node)
    if not key_graph.graph.has_node(target):
        return {
            "success": False,
            "reason": "target_node_missing",
            "path_nodes": [],
            "path_latents": [],
        }

    source_candidates = []
    for node_id in key_graph.graph.nodes:
        try:
            node_int = int(node_id)
            node_latent = key_graph.nodes[node_int]
        except Exception:
            continue
        distance = _latent_distance(current_latent, node_latent)
        if distance <= float(local_radius):
            source_candidates.append((float(distance), node_int))

    if not source_candidates:
        return {
            "success": False,
            "reason": "no_local_source_neighbor",
            "path_nodes": [],
            "path_latents": [],
        }

    best_cost = math.inf
    best_path = None
    best_source_distance = math.inf
    for source_distance, source in sorted(source_candidates):
        try:
            path = nx.shortest_path(
                key_graph.graph,
                source=source,
                target=target,
                weight="weight",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        graph_cost = 0.0
        for u, v in zip(path[:-1], path[1:]):
            graph_cost += _keygraph_edge_weight(key_graph, u, v)
        total_cost = float(source_distance) + float(graph_cost)
        if total_cost < best_cost:
            best_cost = total_cost
            best_path = [int(node) for node in path]
            best_source_distance = float(source_distance)

    if best_path is None:
        return {
            "success": False,
            "reason": "no_graph_path_to_target",
            "path_nodes": [],
            "path_latents": [],
        }

    return {
        "success": True,
        "reason": "ok",
        "path_nodes": best_path,
        "path_latents": [key_graph.nodes[int(node)] for node in best_path],
        "path_cost": float(best_cost),
        "source_distance": float(best_source_distance),
    }


def oracle_failure_reason_with_grounding(
    runtime_success: bool,
    oracle: Dict[str, Any],
) -> Optional[str]:
    if runtime_success and not oracle.get("oracle_success"):
        return "semantic_grounding_error"
    return oracle.get("failure_reason")


def evaluator_oracle_mismatch_summary(
    evaluator_labels: Sequence[Iterable[str]],
    oracle_labels: Sequence[Iterable[str]],
    propositions: Sequence[str],
    product_progress: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compare method-internal evaluator labels with final oracle labels."""

    n = min(len(evaluator_labels), len(oracle_labels))
    props = [str(p) for p in propositions]
    by_prop = {prop: {"false_positive": 0, "false_negative": 0} for prop in props}
    samples = []
    mismatch_steps = 0
    for idx in range(n):
        eval_set = {str(x) for x in evaluator_labels[idx]}
        oracle_set = {str(x) for x in oracle_labels[idx]}
        fp = sorted(eval_set - oracle_set)
        fn = sorted(oracle_set - eval_set)
        if fp or fn:
            mismatch_steps += 1
            for prop in fp:
                by_prop.setdefault(prop, {"false_positive": 0, "false_negative": 0})
                by_prop[prop]["false_positive"] += 1
            for prop in fn:
                by_prop.setdefault(prop, {"false_positive": 0, "false_negative": 0})
                by_prop[prop]["false_negative"] += 1
            if len(samples) < 16:
                progress = None
                if product_progress is not None and idx < len(product_progress):
                    progress = product_progress[idx]
                samples.append(
                    {
                        "step": int(idx),
                        "false_positive": fp,
                        "false_negative": fn,
                        "product_progress": progress,
                    }
                )
    return {
        "num_compared_steps": int(n),
        "num_mismatch_steps": int(mismatch_steps),
        "mismatch_rate": float(mismatch_steps / n) if n else 0.0,
        "by_proposition": by_prop,
        "samples": samples,
    }


def task_regions_to_region_specs(regions: Any) -> List[Dict[str, Any]]:
    """Normalize saved task regions into StateRegionEvaluator specs."""
    return normalize_task_regions_to_region_specs(regions)


def task_predicate_specs(task: Dict[str, Any]) -> tuple[Dict[str, Dict[str, Any]], str]:
    """Return per-proposition predicate specs with backward-compatible fallbacks."""
    return extract_predicate_specs(task)


def task_to_region_specs(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return oracle region specs from `regions` plus any predicate geometries."""
    return normalized_task_to_region_specs(task)


def persist_real_task_outputs(
    *,
    result_dir: Optional[Path],
    diagnostics: Dict[str, Any],
    execution=None,
    oracle: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, str]]:
    """Persist diagnostics and rollout traces for a real task run."""
    if result_dir is None:
        return None
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = result_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(jsonable(diagnostics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths = {"diagnostics_path": str(diagnostics_path)}

    if execution is not None:
        trace = execution.trace
        trace_payload = {
            "observations": trace.observations,
            "states": trace.states,
            "actions": trace.actions,
            "evaluator_labels": [sorted(labels) for labels in trace.evaluator_labels],
            "oracle_labels": [] if oracle is None else oracle.get("oracle_labels", []),
            "product_progress": trace.product_progress,
            "failure_reason": trace.failure_reason,
            "metadata": trace.metadata,
        }
        trace_path = result_dir / "trace.json"
        trace_path.write_text(
            json.dumps(jsonable(trace_payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths["trace_path"] = str(trace_path)

        transition_payload = [
            {
                "realized": item.realized,
                "reason": item.reason,
                "details": item.details,
            }
            for item in execution.transition_results
        ]
        transition_path = result_dir / "transition_results.json"
        transition_path.write_text(
            json.dumps(jsonable(transition_payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths["transition_results_path"] = str(transition_path)

    if oracle is not None:
        oracle_path = result_dir / "oracle_verdict.json"
        oracle_path.write_text(
            json.dumps(jsonable(oracle), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths["oracle_verdict_path"] = str(oracle_path)

    return paths


def run_real_task_instance(
    *,
    key_graph,
    support_by_node: Dict[Any, List[SemanticCarrier]],
    backend: Dict[str, Any],
    semantic_get_phi: Callable[[Any], Any],
    semantic_tdr_path: Optional[str],
    formula: str,
    labels: Sequence[str],
    region_specs: Sequence[Dict[str, Any]],
    seed: int,
    start_observation: Optional[Any] = None,
    start_xy: Optional[Sequence[float]] = None,
    max_anchor_support: int = 5000,
    max_prefix_candidates: int = 5,
    suffix_cycles: int = 1,
    local_step_budget: int = 8000,
    reach_threshold: float = 8.0,
    anchor_reached_threshold: Optional[float] = None,
    action_clip: float = 1.0,
    render_video: Optional[Path] = None,
    render_every: int = 3,
    render_fps: int = 30,
    env_size: Optional[int] = None,
    env_max_episode_steps: Optional[int] = None,
    result_dir: Optional[Path] = None,
    task_metadata: Optional[Dict[str, Any]] = None,
    semantic_evaluator: str = "state_region",
    predicate_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    reset_task_info: Optional[Dict[str, Any]] = None,
    semantic_repulsion_enable: Optional[bool] = None,
    env_backend: str = "ogbenchTL",
    execution_refresh_mode: str = "none",
    execution_refresh_interval: int = 10,
    execution_refresh_failure_budget: int = 3,
) -> Dict[str, Any]:
    """Run one state-based real environment task with persistent diagnostics."""
    import numpy as np

    from O_utils.env_utils import make_ltl_env_and_datasets
    from O_utils.evaluation import env_step, get_frame

    if anchor_reached_threshold is None or float(anchor_reached_threshold) <= 0:
        anchor_reached_threshold = float(reach_threshold)
    else:
        anchor_reached_threshold = float(anchor_reached_threshold)
    execution_refresh_mode = normalize_execution_refresh_mode(execution_refresh_mode)
    execution_refresh_interval = max(1, int(execution_refresh_interval))
    execution_refresh_failure_budget = max(1, int(execution_refresh_failure_budget))
    labels = [str(label) for label in labels]
    buchi_runtime = build_buchi_runtime_from_formula(formula, labels)

    evaluator_key = _normalize_evaluator_backend(semantic_evaluator)
    strict_state_lift = evaluator_key == "oracle_state_lift"
    scene_env = is_scene_env_name(backend["env_name"])

    resolved_env_size = (
        default_ltl_env_size(backend["env_name"]) if env_size is None else int(env_size)
    )
    if scene_env:
        import gymnasium as gym
        from O_utils.env_utils import EpisodeMonitor, dataset_directory_options

        normalized_env_backend = str(env_backend or "ogbenchTL")
        if normalized_env_backend == "ogbench":
            import ogbench

            env_raw, _train_dataset, _val_dataset = ogbench.make_env_and_datasets(
                backend["env_name"],
                compact_dataset=False,
                **dataset_directory_options(),
            )
            env = EpisodeMonitor(env_raw)
        elif normalized_env_backend == "ogbenchTL":
            import ogbenchTL  # noqa: F401 - registers local OGBench scene envs.

            env = EpisodeMonitor(gym.make(backend["env_name"]))
        else:
            raise ValueError(f"Unknown scene env backend: {normalized_env_backend!r}")
        scene_goal_termination_disabled = disable_env_goal_termination_for_ltl(env)
    else:
        normalized_env_backend = "ogbenchTL"
        env, _train_dataset, _val_dataset = make_ltl_env_and_datasets(
            backend["env_name"],
            seed,
            size=resolved_env_size,
            region_type="mixed",
            max_regions=max(8, len(region_specs)),
            add_info=str(backend["env_name"]).startswith("visual-") or strict_state_lift,
        )
        scene_goal_termination_disabled = False
    if env_max_episode_steps is not None:
        env.set_max_episode_steps(int(env_max_episode_steps))

    scene_reset_options = None
    if scene_env and reset_task_info is not None:
        scene_reset_options = {"task_info": scene_task_info_for_env(reset_task_info)}

    def reset_scene_env():
        if scene_reset_options is not None:
            out = env.reset(seed=seed, options=scene_reset_options)
        else:
            out = env.reset(seed=seed)
        disable_env_goal_termination_for_ltl(env)
        return out

    if start_observation is None:
        if scene_env:
            out = reset_scene_env()
            observation = out[0] if isinstance(out, tuple) else out
            observation = np.asarray(observation).copy()
            start_xy = None
        else:
            if start_xy is None:
                first_support = next(
                    (carriers[0] for carriers in support_by_node.values() if carriers),
                    None,
                )
                if first_support is None:
                    raise ValueError("Cannot infer a start observation from empty support.")
                start_observation = first_support.observation
                start_xy = _xy_of(first_support)
            observation = reset_env_at_xy(env, seed, start_xy)
    else:
        start_observation = np.asarray(start_observation).copy()
        if start_xy is None and start_observation.ndim == 1 and len(start_observation) >= 2:
            start_xy = [float(start_observation[0]), float(start_observation[1])]
        if scene_env:
            out = reset_scene_env()
            observation = out[0] if isinstance(out, tuple) else out
            observation = np.asarray(observation).copy()
        elif start_xy is not None:
            observation = reset_env_at_xy(env, seed, start_xy)
        else:
            out = env.reset(seed=seed)
            observation = out[0] if isinstance(out, tuple) else out
            observation = np.asarray(observation).copy()
    start_state, start_state_source = extract_task_state(
        env,
        observation,
        strict=strict_state_lift,
    )
    actual_initial_oracle_labels = sorted(
        _oracle_labeler_for_task(region_specs, predicate_specs)(start_state)
    )
    generation_validation = dict(
        (task_metadata or {}).get("generation_validation", {})
        if isinstance((task_metadata or {}).get("generation_validation", {}), Mapping)
        else {}
    )
    generation_initial_labels = generation_validation.get("oracle_initial_labels")
    initial_label_mismatch = None
    if generation_initial_labels is not None:
        initial_label_mismatch = sorted(generation_initial_labels) != actual_initial_oracle_labels

    anchor_support = flatten_support_for_anchor_search(
        support_by_node,
        max_total=max_anchor_support,
    )
    result = run_state_semantic_planning(
        key_graph=key_graph,
        buchi_runtime=buchi_runtime,
        region_specs=region_specs,
        support_observations=[c.observation for c in anchor_support],
        support_states=[c.state for c in anchor_support],
        support_carriers=anchor_support,
        start_observation=observation,
        start_state=start_state,
        get_phi=semantic_get_phi,
        node_support_by_id=support_by_node,
        propositions=labels,
        semantic_evaluator=semantic_evaluator,
        predicate_specs=predicate_specs,
        max_prop_anchors=6,
        max_guard_anchors=6,
        max_prefix_candidates=max_prefix_candidates,
        anchor_selection_policy=(
            "manipulation_event_context_diverse" if scene_env else "default"
        ),
    )

    base_diagnostics = dict(result.diagnostics)
    base_diagnostics.update(
        {
            "formula": formula,
            "labels": list(labels),
            "region_specs": list(region_specs),
            "seed": int(seed),
            "env_name": backend["env_name"],
            "env_backend": normalized_env_backend,
            "env_size": int(resolved_env_size),
            "env_max_episode_steps": (
                None if env_max_episode_steps is None else int(env_max_episode_steps)
            ),
            "policy_path": backend["policy_path"],
            "semantic_tdr_path": semantic_tdr_path,
            "start_xy": None if start_xy is None else [float(start_xy[0]), float(start_xy[1])],
            "start_state_source": start_state_source,
            "start_state_shape": list(np.asarray(start_state).shape),
            "reset_task_info_used": reset_task_info is not None,
            "scene_goal_termination_disabled": scene_goal_termination_disabled,
            "generation_initial_oracle_labels": generation_initial_labels,
            "actual_initial_oracle_labels": actual_initial_oracle_labels,
            "initial_oracle_label_mismatch": initial_label_mismatch,
            "anchor_support_with_state": sum(
                1 for carrier in anchor_support if carrier.state is not None
            ),
            "anchor_support_without_state": sum(
                1 for carrier in anchor_support if carrier.state is None
            ),
            "task_metadata": dict(task_metadata or {}),
            "max_prefix_candidates": int(max_prefix_candidates),
        }
    )
    if task_metadata and "predicate_spec_validation" in task_metadata:
        base_diagnostics["predicate_spec_validation"] = task_metadata[
            "predicate_spec_validation"
        ]
    if result.plan is None:
        planner_failure = "planner_infeasible"
        base_diagnostics.update(
            {
                "runtime_finite_lasso_success": False,
                "runtime_failure_reason": planner_failure,
                "oracle_success": False,
                "oracle_failure_reason": planner_failure,
            }
        )
        persisted = persist_real_task_outputs(
            result_dir=result_dir,
            diagnostics=base_diagnostics,
        )
        if persisted:
            base_diagnostics.update(persisted)
        return base_diagnostics

    video_writer = None
    render_video_path = None
    if render_video is not None:
        from L_utils.ltl_task_utils import add_regions_to_env_render
        from O_utils.log_utils import get_video_writer

        render_video_path = Path(render_video)
        render_video_path.parent.mkdir(parents=True, exist_ok=True)
        add_regions_to_env_render(env, real_smoke_render_regions(region_specs))
        video_writer = get_video_writer(render_video_path, fps=int(render_fps))
        video_writer.append_data(get_frame(env, backend["env_name"]))

    execution_plan = materialize_finite_lasso(result.plan, suffix_cycles=suffix_cycles)
    waypoint_latents = execution_waypoint_latents(key_graph, execution_plan)
    checkpoint_indices = semantic_checkpoint_indices(execution_plan)
    runtime_monitor = RuntimeMonitor(
        result.certificate_store,
        RuntimeMonitorConfig(local_step_budget=int(local_step_budget)),
    )
    resolved_semantic_repulsion_enable = (
        not scene_env
        if semantic_repulsion_enable is None
        else bool(semantic_repulsion_enable)
    )
    tracking_config = LowLevelTrackingConfig(
        semantic_repulsion_enable=resolved_semantic_repulsion_enable,
        semantic_repulsion_radius=64.0,
        semantic_repulsion_gain=0.1,
        semantic_repulsion_power=1.0,
        semantic_repulsion_max_norm=0.2,
    )
    anchor_positions_by_label = build_anchor_positions_by_label(key_graph)
    step_counter = 0
    prev_skill = None
    carrier_state_source_counts: Dict[str, int] = {}
    execution_refresh_stats: Dict[str, Any] = {
        "mode": execution_refresh_mode,
        "interval": int(execution_refresh_interval),
        "failure_budget": int(execution_refresh_failure_budget),
        "attempts": 0,
        "successes": 0,
        "failures": 0,
        "forced_transition_failures": 0,
        "last_failure_reason": None,
    }

    def record_state_source(source: str) -> None:
        carrier_state_source_counts[source] = carrier_state_source_counts.get(source, 0) + 1

    def segment_executor(request):
        nonlocal observation, step_counter, prev_skill
        current_target_index = request.transition_index + 1
        if request.source[0] == request.target[0] and request.guard.is_unconditional:
            target_index = current_target_index
        elif execution_refresh_mode == "gas_target":
            target_index = pending_semantic_target_index(
                current_index=current_target_index,
                execution_plan=execution_plan,
                semantic_checkpoint_indices=checkpoint_indices,
            )
        else:
            selection = select_tracking_target(
                current_latent=semantic_get_phi(observation),
                waypoint_latents=waypoint_latents,
                current_index=current_target_index,
                local_reachability_radius=float(reach_threshold),
                semantic_checkpoint_indices=checkpoint_indices,
                verified_checkpoint_indices=range(current_target_index),
            )
            target_index = selection.target_index
        target_node = execution_plan.states[target_index][0]
        span_pos, span_neg = guard_span_obligations(
            execution_plan,
            request.transition_index,
            target_index,
        )
        if request.source[0] == request.target[0] and request.guard.is_unconditional:
            dwell_state, dwell_state_source = extract_task_state(
                env,
                observation,
                strict=strict_state_lift,
            )
            carrier = SemanticCarrier(
                observation=observation,
                state=dwell_state,
                info={"state_source": dwell_state_source},
                index=step_counter,
                source="real_task_dwell",
            )
            record_state_source(dwell_state_source)
            return SegmentExecutionResult(
                carriers=[carrier],
                waypoint_reached=True,
                actions=[],
                semantic_checkpoint=False,
                advance_to_transition_index=target_index,
                metadata={"target_node": target_node, "symbolic_dwell": True},
            )

        semantic_target_latent = key_graph.nodes[int(target_node)]
        carriers: List[SemanticCarrier] = []
        actions: List[Any] = []
        waypoint_reached = False
        refresh_path_latents: Optional[List[Any]] = None
        refresh_path_nodes: List[int] = []
        refresh_failures_in_segment = 0
        refresh_stop_reason = None

        def maybe_refresh_path(local_step: int, phi_obs: Any) -> None:
            nonlocal refresh_path_latents, refresh_path_nodes
            nonlocal refresh_failures_in_segment, refresh_stop_reason
            if execution_refresh_mode != "gas_target":
                return
            if local_step != 0 and local_step % int(execution_refresh_interval) != 0:
                return
            execution_refresh_stats["attempts"] += 1
            refresh = refreshed_path_to_target_node(
                key_graph=key_graph,
                current_latent=phi_obs,
                target_node=target_node,
                local_radius=float(reach_threshold),
            )
            if refresh.get("success"):
                execution_refresh_stats["successes"] += 1
                refresh_failures_in_segment = 0
                refresh_path_latents = list(refresh.get("path_latents", []))
                refresh_path_nodes = [int(x) for x in refresh.get("path_nodes", [])]
                return
            execution_refresh_stats["failures"] += 1
            execution_refresh_stats["last_failure_reason"] = refresh.get("reason")
            refresh_failures_in_segment += 1
            if refresh_failures_in_segment > int(execution_refresh_failure_budget):
                refresh_stop_reason = "refresh_path_unavailable"
                execution_refresh_stats["forced_transition_failures"] += 1

        for _local_step in range(int(local_step_budget)):
            phi_obs = semantic_get_phi(observation)
            maybe_refresh_path(_local_step, phi_obs)
            if refresh_stop_reason is not None:
                break
            command_target_latent = semantic_target_latent
            if refresh_path_latents:
                command_target_latent = _select_refreshed_path_target(
                    current_latent=phi_obs,
                    path_latents=refresh_path_latents,
                    reach_threshold=float(reach_threshold),
                )
            command = compute_tracking_command(
                phi_obs=phi_obs,
                target_latent=command_target_latent,
                active_negatives=span_neg,
                anchor_positions_by_label=anchor_positions_by_label,
                prev_skill=prev_skill,
                config=tracking_config,
            )
            prev_skill = command.next_prev_skill
            action = backend["sample_action"](observation, command.skill, temperature=0.0)
            action = np.clip(np.asarray(action), -float(action_clip), float(action_clip))
            observation, _reward, done, info = env_step(env, backend["env_name"], action)
            step_counter += 1
            if video_writer is not None and step_counter % max(1, int(render_every)) == 0:
                video_writer.append_data(get_frame(env, backend["env_name"]))
            carrier_state, carrier_state_source = extract_task_state(
                env,
                observation,
                info,
                strict=strict_state_lift,
            )
            carrier = SemanticCarrier(
                observation=observation,
                state=carrier_state,
                info={"state_source": carrier_state_source},
                index=step_counter,
                source="real_task_rollout",
            )
            record_state_source(carrier_state_source)
            carriers.append(carrier)
            actions.append(action.tolist())
            phi_next = semantic_get_phi(observation)
            latent_dist = float(
                np.linalg.norm(np.asarray(phi_next) - np.asarray(semantic_target_latent))
            )
            labels_at_state = result.certificate_store.cert_pos_set(carrier)
            semantic_ok = set(span_pos).issubset(labels_at_state)
            waypoint_latent_reached = latent_dist <= float(reach_threshold)
            anchor_latent_reached = latent_dist <= float(anchor_reached_threshold)
            if span_pos:
                waypoint_reached = bool(anchor_latent_reached and semantic_ok)
            else:
                waypoint_reached = bool(waypoint_latent_reached)
            if waypoint_reached or done:
                break
        return SegmentExecutionResult(
            carriers=carriers,
            waypoint_reached=waypoint_reached,
            actions=actions,
            semantic_checkpoint=bool(span_pos),
            advance_to_transition_index=target_index,
            metadata={
                "target_node": target_node,
                "requested_target_node": request.target[0],
                "target_index": target_index,
                "requested_target_index": current_target_index,
                "span_positive": sorted(span_pos),
                "span_negative": sorted(span_neg),
                "semantic_checkpoint_requires_latent": bool(span_pos),
                "execution_refresh_mode": execution_refresh_mode,
                "refresh_path_nodes": refresh_path_nodes,
                "refresh_failures_in_segment": int(refresh_failures_in_segment),
                "refresh_stop_reason": refresh_stop_reason,
            },
        )

    execution = execute_finite_product_plan(
        execution_plan=execution_plan,
        segment_executor=segment_executor,
        runtime_monitor=runtime_monitor,
        buchi_runtime=buchi_runtime,
    )
    if video_writer is not None:
        video_writer.append_data(get_frame(env, backend["env_name"]))
        video_writer.close()

    oracle = oracle_finite_lasso_verdict(
        execution.trace.states,
        _oracle_labeler_for_task(region_specs, predicate_specs),
        buchi_runtime,
        prefix_completed=execution.prefix_completed,
        suffix_cycles_completed=execution.suffix_cycles_completed,
        suffix_cycles_required=execution.suffix_cycles_required,
        execution_plan=execution_plan,
        product_progress=execution.trace.product_progress,
    )
    semantic_grounding_error = bool(
        execution.finite_lasso_success and not oracle["oracle_success"]
    )
    mismatch_summary = evaluator_oracle_mismatch_summary(
        execution.trace.evaluator_labels,
        oracle.get("oracle_labels", []),
        labels,
        execution.trace.product_progress,
    )
    oracle_failure_reason = oracle_failure_reason_with_grounding(
        execution.finite_lasso_success,
        oracle,
    )
    diagnostics = dict(base_diagnostics)
    diagnostics.update(
        {
            "execution_guard_sequence": [g.guard_id for g in execution_plan.guards],
            "execution_num_transitions": len(execution_plan.guards),
            "runtime_finite_lasso_success": execution.finite_lasso_success,
            "runtime_failure_reason": execution.failure_reason,
            "prefix_completed": execution.prefix_completed,
            "suffix_cycles_completed": execution.suffix_cycles_completed,
            "suffix_cycles_required": execution.suffix_cycles_required,
            "oracle_success": oracle["oracle_success"],
            "oracle_failure_reason": oracle_failure_reason,
            "oracle_plan_checked": oracle.get("oracle_plan_checked", False),
            "oracle_prefix_completed": oracle.get("oracle_prefix_completed"),
            "oracle_suffix_cycles_completed": oracle.get("oracle_suffix_cycles_completed"),
            "semantic_grounding_error": semantic_grounding_error,
            "evaluator_oracle_mismatch": mismatch_summary,
            "oracle_labels_collapsed": _collapse_label_sets(oracle["oracle_labels"]),
            "num_trace_states": len(execution.trace.states),
            "carrier_state_source_counts": dict(sorted(carrier_state_source_counts.items())),
            "num_actions": len(execution.trace.actions),
            "local_step_budget": int(local_step_budget),
            "reach_threshold": float(reach_threshold),
            "anchor_reached_threshold": float(anchor_reached_threshold),
            "action_clip": float(action_clip),
            "semantic_repulsion_enable": bool(tracking_config.semantic_repulsion_enable),
            "execution_refresh_mode": execution_refresh_mode,
            "execution_refresh_interval": int(execution_refresh_interval),
            "execution_refresh_failure_budget": int(execution_refresh_failure_budget),
            "execution_refresh_stats": dict(execution_refresh_stats),
            "semantic_checkpoint_requires_latent": True,
            "semantic_repulsion_radius": float(tracking_config.semantic_repulsion_radius),
            "semantic_repulsion_gain": float(tracking_config.semantic_repulsion_gain),
            "semantic_repulsion_max_norm": float(tracking_config.semantic_repulsion_max_norm),
            "render_video_path": str(render_video_path) if render_video_path is not None else None,
        }
    )
    persisted = persist_real_task_outputs(
        result_dir=result_dir,
        diagnostics=diagnostics,
        execution=execution,
        oracle=oracle,
    )
    if persisted:
        diagnostics.update(persisted)
    return diagnostics


def run_real_task_smoke(
    *,
    manifest_path: Path,
    target_root: Path,
    keygraph_dir: Optional[Path] = None,
    env_name: Optional[str] = None,
    seed: int = 0,
    gpu: int = 0,
    action_dim: Optional[int] = None,
    start_index: Optional[int] = None,
    target_index: Optional[int] = None,
    avoid_index: Optional[int] = None,
    max_members_per_node: int = 4,
    max_anchor_support: int = 5000,
    suffix_cycles: int = 1,
    local_step_budget: int = 8000,
    reach_threshold: float = 8.0,
    anchor_reached_threshold: Optional[float] = None,
    action_clip: float = 1.0,
    render_video: Optional[Path] = None,
    render_every: int = 3,
    render_fps: int = 30,
    env_size: Optional[int] = None,
    env_max_episode_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one real environment task that exercises overlapping and negative guards."""
    import numpy as np

    from O_utils.env_utils import make_ltl_env_and_datasets
    from O_utils.evaluation import env_step, get_frame

    if anchor_reached_threshold is None or float(anchor_reached_threshold) <= 0:
        anchor_reached_threshold = float(reach_threshold)
    else:
        anchor_reached_threshold = float(anchor_reached_threshold)
    keygraph_dir = keygraph_dir or find_keygraph_dir_from_manifest(manifest_path, target_root)
    key_graph = load_keygraph_from_dir(keygraph_dir)
    support_by_node = collect_node_support_from_keygraph(
        key_graph,
        max_members_per_node=max_members_per_node,
    )
    if start_index is None or target_index is None:
        auto_start, auto_target = choose_real_smoke_node_pair(key_graph, support_by_node)
        start_index = auto_start if start_index is None else int(start_index)
        target_index = auto_target if target_index is None else int(target_index)
    start_index = int(start_index)
    target_index = int(target_index)
    if not support_by_node.get(start_index):
        raise ValueError(f"No cluster support for start_index={start_index}")
    if not support_by_node.get(target_index):
        raise ValueError(f"No cluster support for target_index={target_index}")

    start_carrier = support_by_node[start_index][0]
    target_carrier = support_by_node[target_index][0]
    sx, sy = _xy_of(start_carrier)
    tx, ty = _xy_of(target_carrier)
    target_radius = 2.2
    avoid_radius = 1.5
    if avoid_index is None:
        avoid_index = choose_real_smoke_avoid_node(
            support_by_node,
            start_index=start_index,
            target_index=target_index,
            target_radius=target_radius,
            avoid_radius=avoid_radius,
        )
    avoid_carrier = support_by_node[int(avoid_index)][0]
    ax, ay = _xy_of(avoid_carrier)
    start_target_xy_distance = math.sqrt((sx - tx) * (sx - tx) + (sy - ty) * (sy - ty))
    avoid_target_xy_distance = math.sqrt((ax - tx) * (ax - tx) + (ay - ty) * (ay - ty))
    avoid_corridor_fraction, avoid_line_distance = _line_projection_stats(
        (ax, ay),
        (sx, sy),
        (tx, ty),
    )

    region_specs = make_real_smoke_region_specs(
        target_xy=(tx, ty),
        avoid_xy=(ax, ay),
        target_radius=target_radius,
        avoid_radius=avoid_radius,
    )
    formula = "(!c U (a && b))"
    buchi_runtime = build_buchi_runtime_from_formula(formula, ["a", "b", "c"])
    backend = load_policy_agent_backend(
        manifest_path=manifest_path,
        target_root=target_root,
        example_observation=start_carrier.observation,
        env_name=env_name,
        action_dim=action_dim,
        seed=seed,
        gpu=gpu,
    )
    anchor_support = flatten_support_for_anchor_search(
        support_by_node,
        priority_nodes=[target_index, int(avoid_index), start_index],
        max_total=max_anchor_support,
    )
    result = run_state_semantic_planning(
        key_graph=key_graph,
        buchi_runtime=buchi_runtime,
        region_specs=region_specs,
        support_observations=[c.observation for c in anchor_support],
        support_states=[c.state for c in anchor_support],
        start_observation=start_carrier.observation,
        get_phi=backend["get_phi"],
        node_support_by_id=support_by_node,
        max_prop_anchors=6,
        max_guard_anchors=6,
        max_prefix_candidates=4,
    )
    if result.plan is None:
        diagnostics = dict(result.diagnostics)
        diagnostics.update({"mode": "real_task_smoke", "formula": formula})
        raise RuntimeError(f"real task planning failed: {diagnostics}")

    execution_plan = materialize_finite_lasso(result.plan, suffix_cycles=suffix_cycles)
    waypoint_latents = execution_waypoint_latents(key_graph, execution_plan)
    checkpoint_indices = semantic_checkpoint_indices(execution_plan)
    runtime_monitor = RuntimeMonitor(
        result.certificate_store,
        RuntimeMonitorConfig(local_step_budget=int(local_step_budget)),
    )
    tracking_config = LowLevelTrackingConfig(
        semantic_repulsion_enable=True,
        semantic_repulsion_radius=64.0,
        semantic_repulsion_gain=0.1,
        semantic_repulsion_power=1.0,
        semantic_repulsion_max_norm=0.2,
    )
    anchor_positions_by_label = build_anchor_positions_by_label(key_graph)

    env, _train_dataset, _val_dataset = make_ltl_env_and_datasets(
        backend["env_name"],
        seed,
        size=default_ltl_env_size(backend["env_name"]) if env_size is None else int(env_size),
        region_type="mixed",
        max_regions=8,
    )
    if env_max_episode_steps is not None:
        env.set_max_episode_steps(int(env_max_episode_steps))
    video_writer = None
    render_video_path = None
    if render_video is not None:
        from L_utils.ltl_task_utils import add_regions_to_env_render
        from O_utils.log_utils import get_video_writer

        render_video_path = Path(render_video)
        render_video_path.parent.mkdir(parents=True, exist_ok=True)
        add_regions_to_env_render(env, real_smoke_render_regions(region_specs))
        video_writer = get_video_writer(render_video_path, fps=int(render_fps))
    observation = reset_env_at_xy(env, seed, (sx, sy))
    if video_writer is not None:
        video_writer.append_data(get_frame(env, backend["env_name"]))
    step_counter = 0
    prev_skill = None

    def segment_executor(request):
        nonlocal observation, step_counter, prev_skill
        current_target_index = request.transition_index + 1
        if request.source[0] == request.target[0] and request.guard.is_unconditional:
            target_index = current_target_index
        else:
            selection = select_tracking_target(
                current_latent=backend["get_phi"](observation),
                waypoint_latents=waypoint_latents,
                current_index=current_target_index,
                local_reachability_radius=float(reach_threshold),
                semantic_checkpoint_indices=checkpoint_indices,
                verified_checkpoint_indices=range(current_target_index),
            )
            target_index = selection.target_index
        target_node = execution_plan.states[target_index][0]
        span_pos, span_neg = guard_span_obligations(
            execution_plan,
            request.transition_index,
            target_index,
        )
        if request.source[0] == request.target[0] and request.guard.is_unconditional:
            carrier = SemanticCarrier(
                observation=observation,
                state=observation,
                index=step_counter,
                source="real_task_smoke_dwell",
            )
            return SegmentExecutionResult(
                carriers=[carrier],
                waypoint_reached=True,
                actions=[],
                semantic_checkpoint=False,
                advance_to_transition_index=target_index,
                metadata={"target_node": target_node, "symbolic_dwell": True},
            )

        target_latent = key_graph.nodes[int(target_node)]
        carriers: List[SemanticCarrier] = []
        actions: List[Any] = []
        waypoint_reached = False
        for _local_step in range(int(local_step_budget)):
            phi_obs = backend["get_phi"](observation)
            command = compute_tracking_command(
                phi_obs=phi_obs,
                target_latent=target_latent,
                active_negatives=span_neg,
                anchor_positions_by_label=anchor_positions_by_label,
                prev_skill=prev_skill,
                config=tracking_config,
            )
            prev_skill = command.next_prev_skill
            action = backend["sample_action"](observation, command.skill, temperature=0.0)
            action = np.clip(np.asarray(action), -float(action_clip), float(action_clip))
            observation, _reward, done, _info = env_step(env, backend["env_name"], action)
            step_counter += 1
            if video_writer is not None and step_counter % max(1, int(render_every)) == 0:
                video_writer.append_data(get_frame(env, backend["env_name"]))
            carrier = SemanticCarrier(
                observation=observation,
                state=observation,
                index=step_counter,
                source="real_task_smoke",
            )
            carriers.append(carrier)
            actions.append(action.tolist())
            phi_next = backend["get_phi"](observation)
            latent_dist = float(np.linalg.norm(np.asarray(phi_next) - np.asarray(target_latent)))
            labels = result.certificate_store.cert_pos_set(carrier)
            semantic_ok = set(span_pos).issubset(labels)
            waypoint_latent_reached = latent_dist <= float(reach_threshold)
            anchor_latent_reached = latent_dist <= float(anchor_reached_threshold)
            if span_pos:
                waypoint_reached = bool(anchor_latent_reached and semantic_ok)
            else:
                waypoint_reached = bool(waypoint_latent_reached)
            if waypoint_reached or done:
                break
        return SegmentExecutionResult(
            carriers=carriers,
            waypoint_reached=waypoint_reached,
            actions=actions,
            semantic_checkpoint=bool(span_pos),
            advance_to_transition_index=target_index,
            metadata={
                "target_node": target_node,
                "requested_target_node": request.target[0],
                "target_index": target_index,
                "requested_target_index": current_target_index,
                "span_positive": sorted(span_pos),
                "span_negative": sorted(span_neg),
            },
        )

    execution = execute_finite_product_plan(
        execution_plan=execution_plan,
        segment_executor=segment_executor,
        runtime_monitor=runtime_monitor,
        buchi_runtime=buchi_runtime,
    )
    if video_writer is not None:
        video_writer.append_data(get_frame(env, backend["env_name"]))
        video_writer.close()
    oracle = oracle_finite_lasso_verdict(
        execution.trace.states,
        _region_labeler(region_specs),
        buchi_runtime,
        prefix_completed=execution.prefix_completed,
        suffix_cycles_completed=execution.suffix_cycles_completed,
        suffix_cycles_required=execution.suffix_cycles_required,
        execution_plan=execution_plan,
        product_progress=execution.trace.product_progress,
    )
    semantic_grounding_error = bool(
        execution.finite_lasso_success and not oracle["oracle_success"]
    )

    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "mode": "real_task_smoke",
            "env_name": backend["env_name"],
            "formula": formula,
            "keygraph_dir": str(keygraph_dir),
            "policy_path": backend["policy_path"],
            "start_index": start_index,
            "target_index": target_index,
            "avoid_index": int(avoid_index),
            "start_xy": [sx, sy],
            "target_xy": [tx, ty],
            "avoid_xy": [ax, ay],
            "start_target_xy_distance": float(start_target_xy_distance),
            "avoid_target_xy_distance": float(avoid_target_xy_distance),
            "avoid_corridor_fraction": float(avoid_corridor_fraction),
            "avoid_line_distance": float(avoid_line_distance),
            "region_specs": region_specs,
            "execution_guard_sequence": [g.guard_id for g in execution_plan.guards],
            "execution_num_transitions": len(execution_plan.guards),
            "runtime_finite_lasso_success": execution.finite_lasso_success,
            "runtime_failure_reason": execution.failure_reason,
            "prefix_completed": execution.prefix_completed,
            "suffix_cycles_completed": execution.suffix_cycles_completed,
            "suffix_cycles_required": execution.suffix_cycles_required,
            "oracle_success": oracle["oracle_success"],
            "oracle_failure_reason": oracle_failure_reason_with_grounding(
                execution.finite_lasso_success,
                oracle,
            ),
            "oracle_plan_checked": oracle.get("oracle_plan_checked", False),
            "oracle_prefix_completed": oracle.get("oracle_prefix_completed"),
            "oracle_suffix_cycles_completed": oracle.get("oracle_suffix_cycles_completed"),
            "semantic_grounding_error": semantic_grounding_error,
            "oracle_labels_collapsed": _collapse_label_sets(oracle["oracle_labels"]),
            "num_trace_states": len(execution.trace.states),
            "num_actions": len(execution.trace.actions),
            "local_step_budget": int(local_step_budget),
            "reach_threshold": float(reach_threshold),
            "anchor_reached_threshold": float(anchor_reached_threshold),
            "action_clip": float(action_clip),
            "semantic_repulsion_radius": float(tracking_config.semantic_repulsion_radius),
            "semantic_repulsion_gain": float(tracking_config.semantic_repulsion_gain),
            "semantic_repulsion_max_norm": float(tracking_config.semantic_repulsion_max_norm),
            "render_video_path": str(render_video_path) if render_video_path is not None else None,
        }
    )
    return diagnostics


def _collapse_label_sets(label_sets: Sequence[Sequence[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    prev = None
    for labels in label_sets:
        cur = sorted(labels)
        if cur != prev:
            out.append(cur)
            prev = cur
    return out


def select_task_indices(
    *,
    num_tasks: int,
    task_id: int = 0,
    task_limit: int = 1,
    task_shard_index: int = 0,
    task_num_shards: int = 1,
) -> List[int]:
    """Select local task indices using a contiguous window and deterministic sharding."""
    num_tasks = int(num_tasks)
    task_id = int(task_id)
    task_limit = int(task_limit)
    task_shard_index = int(task_shard_index)
    task_num_shards = int(task_num_shards)
    if num_tasks < 0:
        raise ValueError("num_tasks must be nonnegative")
    if task_id < 0:
        raise ValueError("task_id must be nonnegative")
    if task_num_shards <= 0:
        raise ValueError("task_num_shards must be positive")
    if task_shard_index < 0 or task_shard_index >= task_num_shards:
        raise ValueError("task_shard_index must be in [0, task_num_shards)")
    if task_id >= num_tasks:
        raise IndexError(f"task_id={task_id} is outside task file with {num_tasks} tasks")
    end = num_tasks if task_limit <= 0 else min(num_tasks, task_id + task_limit)
    candidates = list(range(task_id, end))
    return [
        index
        for offset, index in enumerate(candidates)
        if offset % task_num_shards == task_shard_index
    ]


def task_result_dir_for(
    result_dir: Optional[Path],
    task: Dict[str, Any],
    task_index: int,
) -> Optional[Path]:
    """Return the stable result directory for one selected task."""
    if result_dir is None:
        return None
    return Path(result_dir) / f"task_{int(task.get('task_id', task_index)):04d}"


def load_completed_real_task_diagnostics(task_result_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load a completed task diagnostic payload for resume mode, if available."""
    if task_result_dir is None:
        return None
    diagnostics_path = Path(task_result_dir) / "diagnostics.json"
    if not diagnostics_path.is_file():
        return None
    try:
        payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    terminal_keys = {
        "oracle_success",
        "runtime_finite_lasso_success",
        "oracle_failure_reason",
        "runtime_failure_reason",
    }
    if not any(key in payload for key in terminal_keys):
        return None
    payload = dict(payload)
    payload["resume_skipped"] = True
    payload["resume_diagnostics_path"] = str(diagnostics_path)
    return payload


def load_saved_task_payload(task_path: Path) -> Dict[str, Any]:
    """Load a saved LTL task JSON without importing environment utilities."""
    return json.loads(Path(task_path).read_text(encoding="utf-8"))


def run_real_task_file(
    *,
    manifest_path: Path,
    target_root: Path,
    task_path: Path,
    keygraph_dir: Optional[Path] = None,
    env_name: Optional[str] = None,
    task_id: int = 0,
    task_limit: int = 1,
    seed: int = 0,
    gpu: int = 0,
    action_dim: Optional[int] = None,
    max_members_per_node: int = 4,
    max_anchor_support: int = 5000,
    max_prefix_candidates: int = 5,
    suffix_cycles: int = 1,
    local_step_budget: int = 8000,
    reach_threshold: float = 8.0,
    anchor_reached_threshold: Optional[float] = None,
    action_clip: float = 1.0,
    render_video: Optional[Path] = None,
    render_every: int = 3,
    render_fps: int = 30,
    env_size: Optional[int] = None,
    env_max_episode_steps: Optional[int] = None,
    result_dir: Optional[Path] = None,
    task_indices: Optional[Sequence[int]] = None,
    resume: bool = False,
    semantic_evaluator: str = "state_region",
    semantic_repulsion_enable: Optional[bool] = None,
    env_backend: str = "ogbenchTL",
    execution_refresh_mode: str = "none",
    execution_refresh_interval: int = 10,
    execution_refresh_failure_budget: int = 3,
) -> Dict[str, Any]:
    """Run one or more tasks from a saved task file."""
    import numpy as np

    task_payload = load_saved_task_payload(task_path)
    tasks = list(task_payload.get("tasks", []))
    if not tasks:
        raise ValueError(f"No tasks found in {task_path}")
    task_id = int(task_id)
    requested_task_limit = int(task_limit)
    if task_indices is None:
        selected_indices = select_task_indices(
            num_tasks=len(tasks),
            task_id=task_id,
            task_limit=requested_task_limit,
        )
    else:
        selected_indices = [int(index) for index in task_indices]
        invalid = [index for index in selected_indices if index < 0 or index >= len(tasks)]
        if invalid:
            raise IndexError(f"task indices out of range for {task_path}: {invalid}")
    if not selected_indices:
        raise ValueError(f"No selected task indices for {task_path}")

    result_dir = Path(result_dir) if result_dir is not None else None
    selected_items = [(index, tasks[index]) for index in selected_indices]
    results_by_index: Dict[int, Dict[str, Any]] = {}
    pending_items = []
    for actual_task_idx, task in selected_items:
        task_result_dir = task_result_dir_for(result_dir, task, actual_task_idx)
        resumed = load_completed_real_task_diagnostics(task_result_dir) if resume else None
        if resumed is not None:
            resumed.update(
                {
                    "mode": "real_task_file",
                    "task_index": actual_task_idx,
                    "task_id": task.get("task_id", actual_task_idx),
                    "ltl_task_path": str(task_path),
                }
            )
            results_by_index[actual_task_idx] = resumed
        else:
            pending_items.append((actual_task_idx, task))

    backend = None
    semantic_backend = None
    if pending_items:
        keygraph_dir = keygraph_dir or find_keygraph_dir_from_manifest(manifest_path, target_root)
        first_graph = load_keygraph_from_dir(keygraph_dir)
        first_support = collect_node_support_from_keygraph(
            first_graph,
            max_members_per_node=max_members_per_node,
        )
        example_carrier = next(
            (carriers[0] for carriers in first_support.values() if carriers),
            None,
        )
        if example_carrier is None:
            raise ValueError("Cannot load policy backend without graph support carriers.")
        backend = load_policy_agent_backend(
            manifest_path=manifest_path,
            target_root=target_root,
            example_observation=example_carrier.observation,
            env_name=env_name or task_payload.get("env_name"),
            action_dim=action_dim,
            seed=int(seed),
            gpu=gpu,
        )
        # The KeyGraph and semantic anchors must use the frozen TDR that
        # produced its coordinates. The policy checkpoint supplies actions
        # only; its post-training representation is not a graph embedding.
        semantic_backend = load_tdr_agent_and_dataset(
            manifest_path=manifest_path,
            target_root=target_root,
            example_observation=example_carrier.observation,
            env_name=env_name or task_payload.get("env_name"),
            action_dim=action_dim,
            seed=int(seed),
            gpu=gpu,
        )
    elif keygraph_dir is None:
        keygraph_dir = None

    seed = int(seed)
    executed_count = 0
    for actual_task_idx, task in pending_items:
        key_graph = load_keygraph_from_dir(keygraph_dir)
        support_by_node = collect_node_support_from_keygraph(
            key_graph,
            max_members_per_node=max_members_per_node,
        )
        formula = str(task.get("formula_e", task.get("formula", "")))
        if not formula:
            raise ValueError(f"Task {actual_task_idx} has no formula_e/formula")
        predicate_specs, predicate_specs_source = task_predicate_specs(task)
        labels = [str(x) for x in task.get("labels", []) if str(x).strip()]
        if not labels:
            labels = sorted(predicate_specs.keys())
        if not labels:
            labels = sorted(str(k) for k in task.get("regions", {}).keys())
        predicate_spec_validation = validate_predicate_specs(
            predicate_specs,
            fallback_used=predicate_specs_source,
            labels=labels,
        ).to_dict()
        unsupported = sorted(set(predicate_spec_validation["backends"].values()) - SUPPORTED_PREDICATE_BACKENDS)
        if unsupported:
            raise ValueError(f"Unsupported task predicate backend(s): {unsupported}")
        region_specs = task_to_region_specs(task)
        reset_task_info = task.get("reset_task_info")
        start_observation = task.get("start_observation")
        start_xy = task.get("start_xy")
        if start_xy is None and start_observation is not None:
            start_array = np.asarray(start_observation)
            if start_array.ndim == 1 and len(start_array) >= 2:
                start_xy = [start_array[0], start_array[1]]
        eval_seed = int(task.get("seed", seed + actual_task_idx))

        task_result_dir = task_result_dir_for(result_dir, task, actual_task_idx)
        task_render_video = None
        if render_video is not None:
            if len(selected_items) == 1 and not Path(render_video).is_dir():
                task_render_video = Path(render_video)
            else:
                base_dir = task_result_dir or Path(render_video)
                task_render_video = Path(base_dir) / "rollout.mp4"

        diagnostics = run_real_task_instance(
            key_graph=key_graph,
            support_by_node=support_by_node,
            backend=backend,
            semantic_get_phi=semantic_backend["get_phi"],
            semantic_tdr_path=semantic_backend["tdr_path"],
            formula=formula,
            labels=labels,
            region_specs=region_specs,
            seed=eval_seed,
            start_observation=start_observation,
            start_xy=start_xy,
            max_anchor_support=max_anchor_support,
            max_prefix_candidates=max_prefix_candidates,
            suffix_cycles=suffix_cycles,
            local_step_budget=local_step_budget,
            reach_threshold=reach_threshold,
            anchor_reached_threshold=anchor_reached_threshold,
            action_clip=action_clip,
            render_video=task_render_video,
            render_every=render_every,
            render_fps=render_fps,
            env_size=env_size,
            env_max_episode_steps=env_max_episode_steps,
            result_dir=task_result_dir,
            semantic_evaluator=semantic_evaluator,
            predicate_specs=predicate_specs,
            reset_task_info=reset_task_info,
            semantic_repulsion_enable=semantic_repulsion_enable,
            env_backend=env_backend,
            execution_refresh_mode=execution_refresh_mode,
            execution_refresh_interval=execution_refresh_interval,
            execution_refresh_failure_budget=execution_refresh_failure_budget,
            task_metadata={
                "task_id": task.get("task_id", actual_task_idx),
                "task_index": actual_task_idx,
                "formula_p": task.get("formula_p"),
                "ltl_task_path": str(task_path),
                "predicate_specs_source": predicate_specs_source,
                "predicate_spec_validation": predicate_spec_validation,
                "generation_validation": task.get("generation_validation", {}),
                "reset_task_info_present": reset_task_info is not None,
            },
        )
        diagnostics.update(
            {
                "mode": "real_task_file",
                "task_index": actual_task_idx,
                "task_id": task.get("task_id", actual_task_idx),
                "ltl_task_path": str(task_path),
                "resume_skipped": False,
            }
        )
        if task_result_dir is not None:
            persist_real_task_outputs(
                result_dir=task_result_dir,
                diagnostics=diagnostics,
            )
        results_by_index[actual_task_idx] = diagnostics
        executed_count += 1

    results = [results_by_index[index] for index in selected_indices]
    resume_skipped_count = sum(1 for item in results if item.get("resume_skipped"))

    success_rate = (
        sum(1 for item in results if item.get("oracle_success")) / float(len(results))
        if results
        else 0.0
    )
    aggregate = {
        "mode": "real_task_file",
        "ltl_task_path": str(task_path),
        "num_tasks": len(results),
        "task_start": task_id,
        "task_limit": requested_task_limit,
        "task_indices": list(selected_indices),
        "executed_count": executed_count,
        "resume_skipped_count": resume_skipped_count,
        "resume": bool(resume),
        "oracle_success_rate": success_rate,
        "runtime_success_rate": (
            sum(1 for item in results if item.get("runtime_finite_lasso_success"))
            / float(len(results))
            if results
            else 0.0
        ),
        "semantic_evaluator": str(semantic_evaluator),
        "execution_refresh_mode": str(execution_refresh_mode),
        "execution_refresh_interval": int(execution_refresh_interval),
        "execution_refresh_failure_budget": int(execution_refresh_failure_budget),
        "results": results,
    }
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
        aggregate_path = result_dir / "aggregate_diagnostics.json"
        aggregate_path.write_text(
            json.dumps(jsonable(aggregate), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        aggregate["aggregate_diagnostics_path"] = str(aggregate_path)
    return aggregate


def resolve_task_file_paths(task_path: Path) -> List[Path]:
    """Resolve a task JSON path or directory into concrete task files."""
    task_path = Path(task_path)
    if task_path.is_file():
        return [task_path]
    if not task_path.is_dir():
        raise FileNotFoundError(f"Task path does not exist: {task_path}")
    excluded = {"aggregate_diagnostics.json", "batch_summary.json", "summary.json"}
    paths = sorted(
        path
        for path in task_path.glob("*.json")
        if path.is_file() and path.name not in excluded
    )
    if not paths:
        raise ValueError(f"No task JSON files found in {task_path}")
    return paths


def build_real_task_schedule(
    task_path: Path,
    *,
    task_id: int = 0,
    task_limit: int = 1,
    task_shard_index: int = 0,
    task_num_shards: int = 1,
) -> List[TaskScheduleEntry]:
    """Build a deterministic per-file task schedule for benchmark-scale runs."""
    entries: List[TaskScheduleEntry] = []
    for path in resolve_task_file_paths(task_path):
        payload = load_saved_task_payload(path)
        num_tasks = len(payload.get("tasks", []))
        if num_tasks == 0:
            continue
        indices = select_task_indices(
            num_tasks=num_tasks,
            task_id=task_id,
            task_limit=task_limit,
            task_shard_index=task_shard_index,
            task_num_shards=task_num_shards,
        )
        if indices:
            entries.append(
                TaskScheduleEntry(
                    task_path=path,
                    task_indices=tuple(indices),
                    num_tasks_in_file=num_tasks,
                )
            )
    return entries


def _count_values(values: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def task_result_key(item: Dict[str, Any]) -> str:
    """Return a stable task-result key for cross-shard aggregation."""
    task_path = item.get("ltl_task_path") or item.get("task_path") or "unknown_task_file"
    task_index = item.get("task_index")
    if task_index is None:
        task_index = item.get("task_id", "unknown_index")
    return f"{task_path}::{task_index}"


def task_file_breakdown(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate compact metrics by saved task file."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in tasks:
        key = str(item.get("ltl_task_path") or item.get("task_path") or "unknown_task_file")
        grouped.setdefault(key, []).append(dict(item))
    out: Dict[str, Dict[str, Any]] = {}
    for task_path, items in sorted(grouped.items()):
        num_items = len(items)
        oracle_success_count = sum(1 for item in items if bool(item.get("oracle_success")))
        runtime_success_count = sum(
            1 for item in items if bool(item.get("runtime_finite_lasso_success"))
        )
        action_values = [
            float(item["num_actions"])
            for item in items
            if item.get("num_actions") is not None
        ]
        out[task_path] = {
            "num_tasks": num_items,
            "oracle_success_count": oracle_success_count,
            "oracle_success_rate": oracle_success_count / float(num_items) if num_items else 0.0,
            "runtime_success_count": runtime_success_count,
            "runtime_success_rate": runtime_success_count / float(num_items) if num_items else 0.0,
            "mean_num_actions": _mean_or_none(action_values),
        }
    return out


def summarize_real_task_aggregates(aggregates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize one or more real-task aggregate diagnostic payloads."""
    tasks: List[Dict[str, Any]] = []
    for aggregate in aggregates:
        for item in aggregate.get("results", []):
            tasks.append(dict(item))

    num_tasks = len(tasks)
    oracle_success_count = sum(1 for item in tasks if bool(item.get("oracle_success")))
    runtime_success_count = sum(
        1 for item in tasks if bool(item.get("runtime_finite_lasso_success"))
    )
    num_actions = [
        float(item["num_actions"])
        for item in tasks
        if item.get("num_actions") is not None
    ]
    success_actions = [
        float(item["num_actions"])
        for item in tasks
        if item.get("num_actions") is not None and bool(item.get("oracle_success"))
    ]
    oracle_failure_reasons = [
        item.get("oracle_failure_reason") or "success"
        for item in tasks
        if not bool(item.get("oracle_success"))
    ]
    runtime_failure_reasons = [
        item.get("runtime_failure_reason") or "success"
        for item in tasks
        if not bool(item.get("runtime_finite_lasso_success"))
    ]
    unsupported_guard_terms = sorted(
        {
            str(term)
            for item in tasks
            for term in item.get("unsupported_guard_terms", [])
        }
    )
    semantic_grounding_error_count = sum(
        1 for item in tasks if bool(item.get("semantic_grounding_error"))
    )
    evaluator_oracle_mismatch_rates = [
        float(item.get("evaluator_oracle_mismatch", {}).get("mismatch_rate"))
        for item in tasks
        if isinstance(item.get("evaluator_oracle_mismatch"), dict)
        and item.get("evaluator_oracle_mismatch", {}).get("mismatch_rate") is not None
    ]
    total_negative_uncertainty_blocks = sum(
        int(item.get("negative_uncertainty_blocks") or 0)
        for item in tasks
    )
    compact_tasks = [
        {
            "ltl_task_path": item.get("ltl_task_path"),
            "task_index": item.get("task_index"),
            "task_id": item.get("task_id"),
            "task_key": task_result_key(item),
            "oracle_success": bool(item.get("oracle_success")),
            "runtime_success": bool(item.get("runtime_finite_lasso_success")),
            "runtime_finite_lasso_success": bool(item.get("runtime_finite_lasso_success")),
            "oracle_failure_reason": item.get("oracle_failure_reason"),
            "runtime_failure_reason": item.get("runtime_failure_reason"),
            "unsupported_guard_terms": list(item.get("unsupported_guard_terms", [])),
            "semantic_grounding_error": bool(item.get("semantic_grounding_error")),
            "evaluator_oracle_mismatch_rate": (
                item.get("evaluator_oracle_mismatch", {}).get("mismatch_rate")
                if isinstance(item.get("evaluator_oracle_mismatch"), dict)
                else None
            ),
            "state_lift_mismatch_rate": (
                item.get("evaluator_oracle_mismatch", {}).get("mismatch_rate")
                if isinstance(item.get("evaluator_oracle_mismatch"), dict)
                else None
            ),
            "negative_uncertainty_blocks": int(
                item.get("negative_uncertainty_blocks") or 0
            ),
            "num_actions": item.get("num_actions"),
            "prefix_completed": item.get("prefix_completed"),
            "suffix_cycles_completed": item.get("suffix_cycles_completed"),
            "suffix_cycles_required": item.get("suffix_cycles_required"),
        }
        for item in tasks
    ]
    task_key_counts = _count_values(item["task_key"] for item in compact_tasks)
    duplicate_task_keys = sorted(
        key for key, count in task_key_counts.items() if count > 1
    )
    return {
        "num_aggregate_files": len(aggregates),
        "num_tasks": num_tasks,
        "num_unique_task_results": len(task_key_counts),
        "num_duplicate_task_results": sum(
            count - 1 for count in task_key_counts.values() if count > 1
        ),
        "duplicate_task_keys": duplicate_task_keys,
        "oracle_success_count": oracle_success_count,
        "oracle_success_rate": (
            oracle_success_count / float(num_tasks) if num_tasks else 0.0
        ),
        "runtime_success_count": runtime_success_count,
        "runtime_success_rate": (
            runtime_success_count / float(num_tasks) if num_tasks else 0.0
        ),
        "mean_num_actions": _mean_or_none(num_actions),
        "mean_success_num_actions": _mean_or_none(success_actions),
        "oracle_failure_reason_counts": _count_values(oracle_failure_reasons),
        "runtime_failure_reason_counts": _count_values(runtime_failure_reasons),
        "unsupported_guard_terms": unsupported_guard_terms,
        "semantic_grounding_error_count": semantic_grounding_error_count,
        "mean_evaluator_oracle_mismatch_rate": _mean_or_none(
            evaluator_oracle_mismatch_rates
        ),
        "total_negative_uncertainty_blocks": total_negative_uncertainty_blocks,
        "by_task_file": task_file_breakdown(compact_tasks),
        "tasks": compact_tasks,
    }


def load_schedule_manifests(result_dir: Path) -> List[Dict[str, Any]]:
    """Load dry-run run manifests saved under a result root."""
    result_dir = Path(result_dir)
    manifest_paths = sorted(result_dir.rglob("run_manifest.json"))
    manifests = []
    for path in manifest_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload = dict(payload)
        payload["_manifest_path"] = str(path)
        manifests.append(payload)
    return manifests


def scheduled_task_keys_from_manifests(manifests: Sequence[Dict[str, Any]]) -> List[str]:
    """Extract scheduled task keys from dry-run manifests."""
    keys: List[str] = []
    for manifest in manifests:
        for file_entry in manifest.get("schedule", []):
            task_path = file_entry.get("task_path")
            for task_entry in file_entry.get("tasks", []):
                keys.append(
                    task_result_key(
                        {
                            "ltl_task_path": task_path,
                            "task_index": task_entry.get("task_index"),
                            "task_id": task_entry.get("task_id"),
                        }
                    )
                )
    return keys


def summarize_real_task_results_dir(
    result_dir: Path,
    *,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load saved real-task aggregates under a directory and summarize them."""
    result_dir = Path(result_dir)
    aggregate_paths = sorted(result_dir.rglob("aggregate_diagnostics.json"))
    if not aggregate_paths:
        raise FileNotFoundError(f"No aggregate_diagnostics.json found under {result_dir}")
    aggregates = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in aggregate_paths
    ]
    summary = summarize_real_task_aggregates(aggregates)
    schedule_manifests = load_schedule_manifests(result_dir)
    scheduled_keys = scheduled_task_keys_from_manifests(schedule_manifests)
    scheduled_key_counts = _count_values(scheduled_keys)
    result_keys = _count_values(task.get("task_key") for task in summary.get("tasks", []))
    missing_scheduled_keys = sorted(
        key for key in scheduled_key_counts if result_keys.get(key, 0) == 0
    )
    extra_result_keys = sorted(
        key for key in result_keys if key not in scheduled_key_counts
    )
    duplicate_scheduled_keys = sorted(
        key for key, count in scheduled_key_counts.items() if count > 1
    )
    summary.update(
        {
            "mode": "summarize_real_results",
            "result_dir": str(result_dir),
            "aggregate_paths": [str(path) for path in aggregate_paths],
            "schedule_manifest_paths": [
                str(manifest.get("_manifest_path")) for manifest in schedule_manifests
            ],
            "num_schedule_manifests": len(schedule_manifests),
            "num_scheduled_tasks_from_manifests": len(scheduled_keys),
            "num_unique_scheduled_tasks_from_manifests": len(scheduled_key_counts),
            "num_missing_scheduled_tasks": len(missing_scheduled_keys),
            "missing_scheduled_task_keys": missing_scheduled_keys,
            "num_extra_result_tasks": len(extra_result_keys),
            "extra_result_task_keys": extra_result_keys,
            "duplicate_scheduled_task_keys": duplicate_scheduled_keys,
        }
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(jsonable(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["summary_path"] = str(output_path)
    return summary


def run_real_task_batch(
    *,
    manifest_path: Path,
    target_root: Path,
    task_path: Path,
    keygraph_dir: Optional[Path] = None,
    env_name: Optional[str] = None,
    task_id: int = 0,
    task_limit: int = 1,
    seed: int = 0,
    gpu: int = 0,
    action_dim: Optional[int] = None,
    max_members_per_node: int = 4,
    max_anchor_support: int = 5000,
    max_prefix_candidates: int = 5,
    suffix_cycles: int = 1,
    local_step_budget: int = 8000,
    reach_threshold: float = 8.0,
    anchor_reached_threshold: Optional[float] = None,
    action_clip: float = 1.0,
    render_video: Optional[Path] = None,
    render_every: int = 3,
    render_fps: int = 30,
    env_size: Optional[int] = None,
    env_max_episode_steps: Optional[int] = None,
    result_dir: Optional[Path] = None,
    task_shard_index: int = 0,
    task_num_shards: int = 1,
    resume: bool = False,
    semantic_evaluator: str = "state_region",
    execution_refresh_mode: str = "none",
    execution_refresh_interval: int = 10,
    execution_refresh_failure_budget: int = 3,
) -> Dict[str, Any]:
    """Run saved task JSON files through the verified real-task primitive."""
    schedule = build_real_task_schedule(
        task_path,
        task_id=task_id,
        task_limit=task_limit,
        task_shard_index=task_shard_index,
        task_num_shards=task_num_shards,
    )
    result_root = Path(result_dir) if result_dir is not None else None
    aggregates: List[Dict[str, Any]] = []
    for entry in schedule:
        task_file = entry.task_path
        file_result_dir = None if result_root is None else result_root / task_file.stem
        file_render_video = None
        if render_video is not None:
            file_render_video = Path(render_video)
            if len(schedule) > 1 or file_render_video.is_dir():
                file_render_video = file_render_video / task_file.stem / "rollout.mp4"
        aggregate = run_real_task_file(
            manifest_path=manifest_path,
            target_root=target_root,
            task_path=task_file,
            keygraph_dir=keygraph_dir,
            env_name=env_name,
            task_id=entry.task_indices[0],
            task_limit=len(entry.task_indices),
            task_indices=entry.task_indices,
            seed=seed,
            gpu=gpu,
            action_dim=action_dim,
            max_members_per_node=max_members_per_node,
            max_anchor_support=max_anchor_support,
            max_prefix_candidates=max_prefix_candidates,
            suffix_cycles=suffix_cycles,
            local_step_budget=local_step_budget,
            reach_threshold=reach_threshold,
            anchor_reached_threshold=anchor_reached_threshold,
            action_clip=action_clip,
            render_video=file_render_video,
            render_every=render_every,
            render_fps=render_fps,
            env_size=env_size,
            env_max_episode_steps=env_max_episode_steps,
            result_dir=file_result_dir,
            resume=resume,
            semantic_evaluator=semantic_evaluator,
            execution_refresh_mode=execution_refresh_mode,
            execution_refresh_interval=execution_refresh_interval,
            execution_refresh_failure_budget=execution_refresh_failure_budget,
        )
        aggregates.append(aggregate)

    summary = summarize_real_task_aggregates(aggregates)
    summary.update(
        {
            "mode": "real_task_batch",
            "task_path": str(task_path),
            "task_files": [str(entry.task_path) for entry in schedule],
            "schedule": [
                {
                    "task_path": str(entry.task_path),
                    "task_indices": list(entry.task_indices),
                    "num_tasks_in_file": entry.num_tasks_in_file,
                }
                for entry in schedule
            ],
            "task_start": int(task_id),
            "task_limit": int(task_limit),
            "task_shard_index": int(task_shard_index),
            "task_num_shards": int(task_num_shards),
            "resume": bool(resume),
            "semantic_evaluator": str(semantic_evaluator),
            "max_prefix_candidates": int(max_prefix_candidates),
            "result_dir": None if result_root is None else str(result_root),
            "file_aggregates": aggregates,
        }
    )
    if result_root is not None:
        result_root.mkdir(parents=True, exist_ok=True)
        summary_path = result_root / "batch_summary.json"
        summary_path.write_text(
            json.dumps(jsonable(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_path)
    return summary


def build_real_task_batch_manifest(
    *,
    task_path: Path,
    result_dir: Optional[Path],
    task_id: int = 0,
    task_limit: int = 1,
    task_shard_index: int = 0,
    task_num_shards: int = 1,
    resume: bool = False,
    manifest_path: Optional[Path] = None,
    target_root: Optional[Path] = None,
    keygraph_dir: Optional[Path] = None,
    env_name: Optional[str] = None,
    semantic_evaluator: str = "state_region",
) -> Dict[str, Any]:
    """Create a dry-run manifest for benchmark-scale saved-task evaluation."""
    schedule = build_real_task_schedule(
        task_path,
        task_id=task_id,
        task_limit=task_limit,
        task_shard_index=task_shard_index,
        task_num_shards=task_num_shards,
    )
    result_root = Path(result_dir) if result_dir is not None else None
    entries = []
    total_scheduled = 0
    completed_if_resume = 0
    for entry in schedule:
        file_result_dir = None if result_root is None else result_root / entry.task_path.stem
        task_entries = []
        payload = load_saved_task_payload(entry.task_path)
        tasks = payload.get("tasks", [])
        for index in entry.task_indices:
            task = tasks[index]
            task_dir = task_result_dir_for(file_result_dir, task, int(index))
            completed = load_completed_real_task_diagnostics(task_dir) is not None
            total_scheduled += 1
            completed_if_resume += int(bool(resume and completed))
            task_entries.append(
                {
                    "task_index": int(index),
                    "task_id": task.get("task_id", int(index)),
                    "result_dir": None if task_dir is None else str(task_dir),
                    "completed_for_resume": bool(completed),
                }
            )
        entries.append(
            {
                "task_path": str(entry.task_path),
                "num_tasks_in_file": entry.num_tasks_in_file,
                "task_indices": list(entry.task_indices),
                "result_dir": None if file_result_dir is None else str(file_result_dir),
                "tasks": task_entries,
            }
        )
    return {
        "mode": "real_task_schedule",
        "task_path": str(task_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "target_root": None if target_root is None else str(target_root),
        "keygraph_dir": None if keygraph_dir is None else str(keygraph_dir),
        "env_name": env_name,
        "semantic_evaluator": str(semantic_evaluator),
        "result_dir": None if result_root is None else str(result_root),
        "task_start": int(task_id),
        "task_limit": int(task_limit),
        "task_shard_index": int(task_shard_index),
        "task_num_shards": int(task_num_shards),
        "resume": bool(resume),
        "num_task_files": len(schedule),
        "num_scheduled_tasks": total_scheduled,
        "num_completed_for_resume": completed_if_resume,
        "num_pending_if_resume": total_scheduled - completed_if_resume,
        "schedule": entries,
    }


def write_real_task_batch_manifest(
    output_path: Path,
    manifest_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist a dry-run real-task schedule manifest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(jsonable(manifest_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    out = dict(manifest_payload)
    out["schedule_manifest_path"] = str(output_path)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "semantic_smoke",
            "m1_acceptance_smoke",
            "real_graph_smoke",
            "state_graph_smoke",
            "execution_smoke",
            "real_task_smoke",
            "real_task_file",
            "real_task_batch",
            "real_task_schedule",
            "summarize_real_results",
        ],
        default="semantic_smoke",
        help="Execution mode for semantic evaluation.",
    )
    parser.add_argument("--manifest", default="assets/manifests/antmaze-giant-stitch_assets.json")
    parser.add_argument("--target-root", default=".")
    parser.add_argument("--keygraph-dir", default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--region-radius", type=float, default=None)
    parser.add_argument("--max-real-support-nodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--max-members-per-node", type=int, default=4)
    parser.add_argument("--max-anchor-support", type=int, default=5000)
    parser.add_argument("--max-prefix-candidates", type=int, default=5)
    parser.add_argument("--suffix-cycles", type=int, default=1)
    parser.add_argument("--local-step-budget", type=int, default=8000)
    parser.add_argument("--reach-threshold", type=float, default=8.0)
    parser.add_argument(
        "--anchor-reached-threshold",
        type=float,
        default=None,
        help=(
            "Latent radius for declaring a positive semantic checkpoint reached. "
            "Defaults to --reach-threshold."
        ),
    )
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--avoid-index", type=int, default=None)
    parser.add_argument("--render-video", default=None)
    parser.add_argument("--render-every", type=int, default=3)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--env-size", type=int, default=None)
    parser.add_argument("--env-max-episode-steps", type=int, default=None)
    parser.add_argument("--ltl-task-path", default=None)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=1)
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-num-shards", type=int, default=1)
    parser.add_argument(
        "--semantic-evaluator",
        choices=["state_region", "oracle_state_lift", "object_state"],
        default="state_region",
        help="State-based predicate evaluator.",
    )
    parser.add_argument(
        "--execution-refresh-mode",
        choices=["none", "gas_target"],
        default="none",
        help=(
            "Opt-in execution adapter. gas_target periodically refreshes the "
            "local graph path to the current pending semantic target."
        ),
    )
    parser.add_argument("--execution-refresh-interval", type=int, default=10)
    parser.add_argument("--execution-refresh-failure-budget", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--schedule-output", default=None)
    parser.add_argument("--json", action="store_true", help="Print diagnostics as JSON.")
    args = parser.parse_args(argv)

    if args.mode == "semantic_smoke":
        diagnostics = run_semantic_smoke()
    elif args.mode == "m1_acceptance_smoke":
        diagnostics = run_m1_acceptance_smoke()
    elif args.mode == "execution_smoke":
        diagnostics = run_execution_smoke()
    elif args.mode == "real_graph_smoke":
        keygraph_dir = (
            Path(args.keygraph_dir)
            if args.keygraph_dir
            else find_keygraph_dir_from_manifest(Path(args.manifest), Path(args.target_root))
        )
        diagnostics = run_real_graph_smoke(
            keygraph_dir=keygraph_dir,
            start_index=args.start_index,
            target_index=args.target_index,
            region_radius=args.region_radius,
            max_support_nodes=args.max_real_support_nodes,
        )
    elif args.mode == "state_graph_smoke":
        diagnostics = run_state_graph_smoke(
            manifest_path=Path(args.manifest),
            target_root=Path(args.target_root),
            keygraph_dir=Path(args.keygraph_dir) if args.keygraph_dir else None,
            env_name=args.env_name,
            seed=args.seed,
            gpu=args.gpu,
            action_dim=args.action_dim,
            start_index=args.start_index,
            target_index=args.target_index,
            region_radius=0.2 if args.region_radius is None else args.region_radius,
            max_members_per_node=args.max_members_per_node,
            max_anchor_support=args.max_anchor_support,
        )
    elif args.mode == "real_task_smoke":
        diagnostics = run_real_task_smoke(
            manifest_path=Path(args.manifest),
            target_root=Path(args.target_root),
            keygraph_dir=Path(args.keygraph_dir) if args.keygraph_dir else None,
            env_name=args.env_name,
            seed=args.seed,
            gpu=args.gpu,
            action_dim=args.action_dim,
            start_index=args.start_index,
            target_index=args.target_index,
            avoid_index=args.avoid_index,
            max_members_per_node=args.max_members_per_node,
            max_anchor_support=args.max_anchor_support,
            suffix_cycles=args.suffix_cycles,
            local_step_budget=args.local_step_budget,
            reach_threshold=args.reach_threshold,
            anchor_reached_threshold=args.anchor_reached_threshold,
            action_clip=args.action_clip,
            render_video=Path(args.render_video) if args.render_video else None,
            render_every=args.render_every,
            render_fps=args.render_fps,
            env_size=args.env_size,
            env_max_episode_steps=args.env_max_episode_steps,
        )
    elif args.mode == "real_task_file":
        if args.ltl_task_path is None:
            raise ValueError("--ltl-task-path is required for --mode real_task_file")
        default_result_dir = (
            Path("artifacts")
            / "evaluate_esagas"
            / (Path(args.ltl_task_path).stem + f"_task{int(args.task_id):04d}")
        )
        diagnostics = run_real_task_file(
            manifest_path=Path(args.manifest),
            target_root=Path(args.target_root),
            task_path=Path(args.ltl_task_path),
            keygraph_dir=Path(args.keygraph_dir) if args.keygraph_dir else None,
            env_name=args.env_name,
            task_id=args.task_id,
            task_limit=args.task_limit,
            seed=args.seed,
            gpu=args.gpu,
            action_dim=args.action_dim,
            max_members_per_node=args.max_members_per_node,
            max_anchor_support=args.max_anchor_support,
            max_prefix_candidates=args.max_prefix_candidates,
            suffix_cycles=args.suffix_cycles,
            local_step_budget=args.local_step_budget,
            reach_threshold=args.reach_threshold,
            anchor_reached_threshold=args.anchor_reached_threshold,
            action_clip=args.action_clip,
            render_video=Path(args.render_video) if args.render_video else None,
            render_every=args.render_every,
            render_fps=args.render_fps,
            env_size=args.env_size,
            env_max_episode_steps=args.env_max_episode_steps,
            result_dir=Path(args.result_dir) if args.result_dir else default_result_dir,
            resume=args.resume,
            semantic_evaluator=args.semantic_evaluator,
            execution_refresh_mode=args.execution_refresh_mode,
            execution_refresh_interval=args.execution_refresh_interval,
            execution_refresh_failure_budget=args.execution_refresh_failure_budget,
        )
    elif args.mode == "real_task_batch":
        if args.ltl_task_path is None:
            raise ValueError("--ltl-task-path is required for --mode real_task_batch")
        default_result_dir = (
            Path("artifacts")
            / "evaluate_esagas"
            / ("batch_" + Path(args.ltl_task_path).stem)
        )
        diagnostics = run_real_task_batch(
            manifest_path=Path(args.manifest),
            target_root=Path(args.target_root),
            task_path=Path(args.ltl_task_path),
            keygraph_dir=Path(args.keygraph_dir) if args.keygraph_dir else None,
            env_name=args.env_name,
            task_id=args.task_id,
            task_limit=args.task_limit,
            seed=args.seed,
            gpu=args.gpu,
            action_dim=args.action_dim,
            max_members_per_node=args.max_members_per_node,
            max_anchor_support=args.max_anchor_support,
            max_prefix_candidates=args.max_prefix_candidates,
            suffix_cycles=args.suffix_cycles,
            local_step_budget=args.local_step_budget,
            reach_threshold=args.reach_threshold,
            anchor_reached_threshold=args.anchor_reached_threshold,
            action_clip=args.action_clip,
            render_video=Path(args.render_video) if args.render_video else None,
            render_every=args.render_every,
            render_fps=args.render_fps,
            env_size=args.env_size,
            env_max_episode_steps=args.env_max_episode_steps,
            result_dir=Path(args.result_dir) if args.result_dir else default_result_dir,
            task_shard_index=args.task_shard_index,
            task_num_shards=args.task_num_shards,
            resume=args.resume,
            semantic_evaluator=args.semantic_evaluator,
            execution_refresh_mode=args.execution_refresh_mode,
            execution_refresh_interval=args.execution_refresh_interval,
            execution_refresh_failure_budget=args.execution_refresh_failure_budget,
        )
    elif args.mode == "real_task_schedule":
        if args.ltl_task_path is None:
            raise ValueError("--ltl-task-path is required for --mode real_task_schedule")
        default_result_dir = (
            Path("artifacts")
            / "evaluate_esagas"
            / ("batch_" + Path(args.ltl_task_path).stem)
        )
        result_dir = Path(args.result_dir) if args.result_dir else default_result_dir
        manifest_payload = build_real_task_batch_manifest(
            task_path=Path(args.ltl_task_path),
            result_dir=result_dir,
            task_id=args.task_id,
            task_limit=args.task_limit,
            task_shard_index=args.task_shard_index,
            task_num_shards=args.task_num_shards,
            resume=args.resume,
            manifest_path=Path(args.manifest),
            target_root=Path(args.target_root),
            keygraph_dir=Path(args.keygraph_dir) if args.keygraph_dir else None,
            env_name=args.env_name,
            semantic_evaluator=args.semantic_evaluator,
        )
        schedule_output = (
            Path(args.schedule_output)
            if args.schedule_output
            else result_dir / "run_manifest.json"
        )
        diagnostics = write_real_task_batch_manifest(schedule_output, manifest_payload)
    elif args.mode == "summarize_real_results":
        if args.result_dir is None:
            raise ValueError("--result-dir is required for --mode summarize_real_results")
        result_dir = Path(args.result_dir)
        diagnostics = summarize_real_task_results_dir(
            result_dir,
            output_path=result_dir / "summary.json",
        )
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"unsupported mode: {args.mode}")

    if args.json:
        print(json.dumps(jsonable(diagnostics), sort_keys=True))
    else:
        print(f"Evaluator-conditioned SAGAS mode '{args.mode}' completed.")
        print(json.dumps(jsonable(diagnostics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
