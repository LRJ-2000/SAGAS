# -*- coding: utf-8 -*-
"""
Runtime wrapper around the Buchi automaton produced by buchi_parse.Buchi.

Adds:
- succ_with_truth(q, assign): returns enabled successors WITH their edge truth dicts,
  so the planner can see which literals are required False (negative constraints).
"""

from typing import Dict, Iterable, List, Set, Tuple
import networkx as nx

from S_utils.guard_terms import GuardTerm


class BuchiRuntime:
    def __init__(self, buchi_obj, label_to_literal: Dict[str, str]):
        """
        Args:
            buchi_obj: Instance of buchi_parse.Buchi with a built buchi_graph (nx.MultiDiGraph).
            label_to_literal: Mapping like {"A": "l1_1", "B": "l2_1", ...}.
        """
        self.G = buchi_obj.buchi_graph
        self.init_states: List[str] = list(self.G.graph.get('init', []))
        self.accept_states: Set[str] = set(self.G.graph.get('accept', []))
        self.label_to_literal: Dict[str, str] = dict(label_to_literal)
        self.literal_to_label: Dict[str, str] = {v: k for k, v in self.label_to_literal.items()}
        self.anchor_prune_stats: Dict = {}
        self.can_reach_accept: Set[str] = set()
        self.positive_free_accept_reachable: Set[str] = set()
        self.first_positive_literals_to_accept: Dict[str, Set[str]] = {}
        self._prefix_heuristic_info_ready = False

    # ---- Basic API ----

    def initial_state(self) -> str:
        """Return a single initial state (if multiple exist, pick the first)."""
        if not self.init_states:
            raise RuntimeError("BuchiRuntime: no initial state in the automaton.")
        return self.init_states[0]

    def is_accepting(self, q: str) -> bool:
        """Check whether a Buchi state is accepting."""
        return q in self.accept_states

    # ---- Transition semantics ----

    def succ(self, q: str, assign_literals: Dict[str, bool]) -> List[str]:
        """
        Return successors enabled by `assign_literals`. Kept for compatibility.
        """
        return [qn for qn, _truth in self.succ_with_truth(q, assign_literals)]

    def succ_with_truth(self, q: str, assign_literals: Dict[str, bool]) -> List[Tuple[str, Dict]]:
        """
        Return successors enabled by `assign_literals`, together with the edge truth.

        Edge truth format:
            dict{lit: bool}   -> required values for that transition
            '1'               -> treated as {} for backward compatibility

        Returns:
            List of (qn, truth_dict) for all enabled successors.
        """
        out: List[Tuple[str, Dict]] = []
        for _, v, data in self.G.out_edges(q, data=True):
            truth = data.get('truth', '1')
            # Normalize to dict form for downstream checks
            if truth == '1' or truth is None:
                truth_dict = {}
                out.append((v, truth_dict))
                continue

            ok = True
            for lit, need in truth.items():
                cur = assign_literals.get(lit, False)
                if cur != need:
                    ok = False
                    break
            if ok:
                out.append((v, dict(truth)))
        return out

    def enabled_guard_successors(self, q: str) -> List[Tuple[str, GuardTerm]]:
        """Return outgoing successors with explicit GuardTerm objects."""
        out: List[Tuple[str, GuardTerm]] = []
        for _, v, data in self.G.out_edges(q, data=True):
            guard = data.get("guard_term", None)
            if guard is None:
                truth = data.get("truth", {})
                if truth == "1" or truth is None:
                    truth = {}
                guard = GuardTerm.from_truth_dict(
                    truth,
                    guard_id=f"{q}->{v}:legacy",
                    source_state=q,
                    target_state=v,
                )
            out.append((v, guard))
        return out

    def succ_with_label_set(self, q: str, label_set: Iterable[str]) -> List[Tuple[str, GuardTerm]]:
        """Return successors whose GuardTerm is satisfied by a set-valued label."""
        labels = set(label_set or [])
        out: List[Tuple[str, GuardTerm]] = []
        for qn, guard in self.enabled_guard_successors(q):
            if set(guard.pos).issubset(labels) and not set(guard.neg).intersection(labels):
                out.append((qn, guard))
        return out

    def step_set(self, q_set: Iterable[str], label_set: Iterable[str]) -> Set[str]:
        """Advance a nondeterministic automaton state set by one label set."""
        next_states: Set[str] = set()
        for q in q_set:
            for qn, _guard in self.succ_with_label_set(q, label_set):
                next_states.add(qn)
        return next_states

    def initial_state_set_after_label(self, label_set: Iterable[str]) -> Set[str]:
        """Consume the initial label from all automaton initial states."""
        return self.step_set(self.init_states, label_set)

    def branch_reaches(
        self,
        q_start: str,
        label_sequence: Iterable[Iterable[str]],
        q_target: str,
    ) -> bool:
        """Check whether some nondeterministic branch reaches `q_target`.

        This is used by runtime and final-verification code to align a planned
        product branch with the label sequence produced by execution.
        """
        q_set: Set[str] = {q_start}
        for labels in label_sequence:
            q_set = self.step_set(q_set, labels)
            if not q_set:
                return False
        return q_target in q_set

    def next_positive_literals(self, q: str) -> Set[str]:
        """
        Collect the set of positive literals (must be True) required by ANY outgoing transition from q.
        This is a local semantic guide only; admissible prefix heuristics should use
        first_positive_literals_toward_accept instead.
        """
        pos = set()
        for _, v, data in self.G.out_edges(q, data=True):
            guard = data.get("guard_term", None)
            if guard is not None:
                pos.update(guard.pos)
                continue
            truth = data.get('truth', '1')
            if truth == '1':
                continue
            for lit, need in truth.items():
                if need is True:
                    pos.add(lit)
        return pos

    def _truth_dict(self, truth) -> Dict:
        """Normalize stored edge truth into a dictionary."""
        if truth == "1" or truth is None:
            return {}
        return dict(truth)

    def _positive_literals(self, truth) -> Set[str]:
        """Return positive literals required by a transition truth."""
        if isinstance(truth, GuardTerm):
            return set(truth.pos)
        truth_dict = self._truth_dict(truth)
        return {lit for lit, need in truth_dict.items() if need is True}

    def _is_positive_free_truth(self, truth) -> bool:
        """Check whether a transition requires no positive proposition."""
        return len(self._positive_literals(truth)) == 0

    def _demote_nonrecurring_accept_states(self) -> List[str]:
        """
        Demote accepting states that cannot be visited infinitely often.

        Buchi acceptance requires recurrent visits. A state is therefore a
        viable accepting state only if it belongs to a non-trivial SCC or has a
        self-loop in the currently pruned automaton.
        """
        if not self.accept_states or self.G.number_of_nodes() == 0:
            self.accept_states = {q for q in self.accept_states if q in self.G}
            self.G.graph["accept"] = sorted(self.accept_states)
            return []

        digraph = nx.DiGraph(self.G)
        cycle_capable = set()
        for scc in nx.strongly_connected_components(digraph):
            if len(scc) > 1:
                cycle_capable.update(scc)
            else:
                q = next(iter(scc))
                if self.G.has_edge(q, q):
                    cycle_capable.add(q)

        viable_accept_states = {
            q for q in self.accept_states if q in self.G and q in cycle_capable
        }
        demoted = sorted(self.accept_states - viable_accept_states)
        self.accept_states = viable_accept_states
        self.G.graph["accept"] = sorted(self.accept_states)
        if demoted:
            self._prefix_heuristic_info_ready = False
        return demoted

    def _precompute_prefix_heuristic_info(self):
        """
        Precompute Buchi-side sets used by the admissible prefix heuristic.

        positive_free_accept_reachable[q] means q can reach an accepting state
        without requiring any positive literal. first_positive_literals_to_accept[q]
        contains labels that can be the first required positive literal after any
        positive-free prefix from q, provided the successor can still reach accept.
        """
        self._demote_nonrecurring_accept_states()
        self.can_reach_accept = set()
        self.positive_free_accept_reachable = set()
        self.first_positive_literals_to_accept = {}
        self._prefix_heuristic_info_ready = True

        if not self.accept_states or self.G.number_of_nodes() == 0:
            for q in self.G.nodes():
                self.first_positive_literals_to_accept[q] = set()
            return

        digraph = nx.DiGraph(self.G)
        reverse_graph = digraph.reverse(copy=False)
        for q_acc in self.accept_states:
            if q_acc not in self.G:
                continue
            self.can_reach_accept.add(q_acc)
            self.can_reach_accept.update(nx.descendants(reverse_graph, q_acc))

        positive_free_graph = nx.DiGraph()
        positive_free_graph.add_nodes_from(self.G.nodes())
        for u, v, data in self.G.edges(data=True):
            guard = data.get("guard_term", None)
            if guard is not None:
                positive_free = len(guard.pos) == 0
            else:
                positive_free = self._is_positive_free_truth(data.get("truth", "1"))
            if positive_free:
                positive_free_graph.add_edge(u, v)

        reverse_positive_free = positive_free_graph.reverse(copy=False)
        for q_acc in self.accept_states:
            if q_acc not in positive_free_graph:
                continue
            self.positive_free_accept_reachable.add(q_acc)
            self.positive_free_accept_reachable.update(
                nx.descendants(reverse_positive_free, q_acc)
            )

        for q in self.G.nodes():
            first_lits = set()
            closure = {q}
            if q in positive_free_graph:
                closure.update(nx.descendants(positive_free_graph, q))

            for qc in closure:
                for _, qn, data in self.G.out_edges(qc, data=True):
                    if qn not in self.can_reach_accept:
                        continue
                    guard = data.get("guard_term", None)
                    if guard is not None:
                        pos_lits = set(guard.pos)
                    else:
                        pos_lits = self._positive_literals(data.get("truth", "1"))
                    first_lits.update(pos_lits)
            self.first_positive_literals_to_accept[q] = first_lits

    def ensure_prefix_heuristic_info(self):
        """Build prefix heuristic metadata after normalizing accepting states."""
        if not self._prefix_heuristic_info_ready:
            self._precompute_prefix_heuristic_info()

    def first_positive_literals_toward_accept(self, q: str) -> Set[str]:
        """
        Return admissible first-positive candidates for reaching acceptance.

        If q can already reach acceptance through positive-free transitions, no
        positive proposition is required before acceptance, so callers should use
        a zero heuristic.
        """
        self.ensure_prefix_heuristic_info()
        if q in self.accept_states or q in self.positive_free_accept_reachable:
            return set()
        return set(self.first_positive_literals_to_accept.get(q, set()))

    def prune_unavailable_positive_labels(self, unavailable_labels: Set[str]) -> Dict:
        """
        Remove automaton transitions that require unavailable labels to be True.

        Unavailable labels mean no conservatively connected anchor could be created
        for that proposition under the current offline graph. Negative requirements
        are left intact; only positive requirements are pruned.
        """
        unavailable_labels = set(unavailable_labels or [])
        unavailable_literals = {
            self.label_to_literal[label]
            for label in unavailable_labels
            if label in self.label_to_literal
        }
        stats = {
            "unavailable_labels": sorted(unavailable_labels),
            "unavailable_literals": sorted(unavailable_literals),
            "removed_edges": 0,
            "demoted_accepting_states": [],
            "demoted_accepting_count": 0,
            "removed_unreachable_states": 0,
            "removed_deadend_states": 0,
            "states_before": self.G.number_of_nodes(),
            "states_after": self.G.number_of_nodes(),
            "accepting_before": len(self.accept_states),
            "accepting_after": len(self.accept_states),
            "has_accepting_run": bool(self.accept_states),
        }
        self._prefix_heuristic_info_ready = False

        edges_to_remove = []
        if unavailable_literals:
            for u, v, key, data in self.G.edges(keys=True, data=True):
                guard = data.get("guard_term", None)
                if guard is not None:
                    # GuardTerm stores user-facing labels. unavailable_labels is
                    # already in the same space.
                    if set(guard.pos).intersection(unavailable_labels):
                        edges_to_remove.append((u, v, key))
                    continue
                truth = data.get("truth", "1")
                if truth == "1":
                    continue
                for lit, need in truth.items():
                    if need is True and lit in unavailable_literals:
                        edges_to_remove.append((u, v, key))
                        break
            self.G.remove_edges_from(edges_to_remove)
        stats["removed_edges"] = len(edges_to_remove)

        reachable = set()
        for q0 in self.init_states:
            if q0 in self.G:
                reachable.add(q0)
                reachable.update(nx.descendants(self.G, q0))
        unreachable = [q for q in list(self.G.nodes()) if q not in reachable]
        self.G.remove_nodes_from(unreachable)
        stats["removed_unreachable_states"] = len(unreachable)

        self.init_states = [q for q in self.init_states if q in self.G]
        self.G.graph["init"] = list(self.init_states)

        demoted_accepting_states = self._demote_nonrecurring_accept_states()
        stats["demoted_accepting_states"] = demoted_accepting_states
        stats["demoted_accepting_count"] = len(demoted_accepting_states)

        if self.accept_states:
            digraph = nx.DiGraph(self.G)
            reverse_graph = digraph.reverse(copy=False)
            can_reach_accept = set()
            for q_acc in self.accept_states:
                can_reach_accept.add(q_acc)
                can_reach_accept.update(nx.descendants(reverse_graph, q_acc))
            deadend = [q for q in list(self.G.nodes()) if q not in can_reach_accept]
            self.G.remove_nodes_from(deadend)
            stats["removed_deadend_states"] = len(deadend)
            self.init_states = [q for q in self.init_states if q in self.G]
            self.accept_states = {q for q in self.accept_states if q in self.G}

        self.G.graph["init"] = list(self.init_states)
        self.G.graph["accept"] = sorted(self.accept_states)
        stats["states_after"] = self.G.number_of_nodes()
        stats["accepting_after"] = len(self.accept_states)
        stats["has_accepting_run"] = bool(self.init_states and self.accept_states)
        self.anchor_prune_stats = stats
        self._precompute_prefix_heuristic_info()
        return stats

    # ---- Label/literal helpers ----

    def literal_for_label(self, label: str) -> str:
        """Map human-readable label to Buchi literal name."""
        return self.label_to_literal[label]

    def label_for_literal(self, literal: str) -> str:
        """Map Buchi literal name back to human-readable label (if known)."""
        return self.literal_to_label.get(literal, literal)
