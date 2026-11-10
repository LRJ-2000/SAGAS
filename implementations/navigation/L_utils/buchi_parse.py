# -*- coding: utf-8 -*-
"""
Single-agent friendly Buchi automaton builder.

Differences from the original multi-robot version:
- Assumes a single agent; literals are plain propositional names such as "l1_1", "A", etc.
- Keeps only clauses whose guards contain at most ONE positive literal (the original paper already
    filtered those). This matches the assumption that labels are non-overlapping and we only "achieve"
    one positive AP at a time (by reaching an anchor node).
- Builds a NetworkX MultiDiGraph to allow multiple parallel edges per (state, state) pair, one per
    feasible DNF clause.
- Edge attribute "truth":
        - '1'   : unconditional transition
        - dict  : mapping {literal_name: bool} for a single DNF clause accepted for that edge.
"""

import os
import re
import subprocess
import networkx as nx
import numpy as np
from sympy.logic.boolalg import to_dnf


class Buchi(object):
    """Construct a Buchi automaton graph for a given LTL task."""

    def __init__(self, task):
        """
        Args:
            task.formula: LTL formula string (no X/next)
            task.subformula: dict[int->str], id to atomic proposition string used by ltl2ba (e.g., {"1": "p", ...})
            task.number_of_robots: ignored except stored (kept for API compatibility)
        """
        self.formula = task.formula
        self.subformula = task.subformula
        self.number_of_robots = getattr(task, "number_of_robots", 1)

        # Use MultiDiGraph to support multiple transitions (one per feasible clause)
        self.buchi_graph = nx.MultiDiGraph(type="buchi", init=[], accept=[])
        self.min_length = dict()

    def construct_buchi_graph(self):
        """Call ltl2ba to compile the LTL formula and parse its output into a NetworkX DiGraph."""
        dirname = os.path.dirname(__file__)
        binary = os.environ.get("SAGAS_LTL2BA") or os.path.join(dirname, "ltl2ba")
        out = subprocess.check_output(
            [binary, "-f", self.formula]
        ).decode("utf-8")
        # State headers
        states_pat = r"\n(\w+):"
        block_pat = r":\n\tif(.*?)fi"

        state_group = re.findall(re.compile(states_pat), out)
        init = [s for s in state_group if "init" in s]
        accept = [s for s in state_group if "accept" in s]
        self.buchi_graph.graph["init"] = init
        self.buchi_graph.graph["accept"] = accept

        # Replace subformula placeholders e{k} with their concrete expressions
        order_key = list(self.subformula.keys())
        order_key.sort(reverse=True)

        for state in state_group:
            self.buchi_graph.add_node(state)
            state_if_fi = re.findall(state + block_pat, out, re.DOTALL)
            if state_if_fi:
                relation_group = re.findall(
                    r":: (\(.*?\)) -> goto (\w+)\n\t", state_if_fi[0]
                )
                for symbol, next_state in relation_group:
                    raw_sym = symbol

                    # Expand e{k} to actual atomic propositions (strings expected by ltl2ba)
                    exp = symbol
                    for k in order_key:
                        exp = exp.replace(f"e{k}", self.subformula[k])

                    truths = self._get_truth_assignments(exp)
                    if not truths:
                        continue
                    # If any clause is unconditional, keep a single unconditional edge
                    if any(t == "1" for t in truths):
                        self.buchi_graph.add_edge(
                            state, next_state, truth="1", symbol=raw_sym
                        )
                    else:
                        # Add one parallel edge per feasible clause (deduplicated)
                        seen = set()
                        for t in truths:
                            key = tuple(sorted(t.items()))
                            if key in seen:
                                continue
                            seen.add(key)
                            self.buchi_graph.add_edge(
                                state, next_state, truth=t, symbol=raw_sym
                            )
            else:
                # Self-loop skip
                state_skip = re.findall(state + r"(.*?)skip\n", out, re.DOTALL)
                if state_skip:
                    self.buchi_graph.add_edge(state, state, truth="1")

    def _get_truth_assignments(self, symbol):
        """
        Return ALL feasible DNF clauses:
            - list of dicts {literal: bool} for each valid clause (<=1 positive literal)
            - or ["1"] if unconditional (any clause reduces to True)
        """
        if symbol == "(1)":
            return ["1"]

        # Convert to sympy style and DNF
        exp = symbol.replace("||", "|").replace("&&", "&").replace("!", "~")
        exp_dnf = str(to_dnf(exp))

        results = []
        unconditional = False

        # Collect all valid clauses that do not contain a literal and its negation simultaneously
        for clause in exp_dnf.split(" | "):
            clause = clause.strip("(").strip(")")
            if clause == "True":
                unconditional = True
                break
            if clause == "False" or clause == "":
                continue

            truth_table = {}
            literals = [t for t in clause.split(" & ") if len(t) > 0]
            valid = True
            seen_pos = set()
            seen_neg = set()

            for lit in literals:
                if lit.startswith("~"):
                    name = lit[1:]
                    seen_neg.add(name)
                    truth_table[name] = False
                else:
                    name = lit
                    seen_pos.add(name)
                    truth_table[name] = True

            # Invalidate if a literal appears both positive and negative
            if len(seen_pos.intersection(seen_neg)) > 0:
                valid = False

            # Enforce the "at most one positive literal" rule (single-agent, non-overlapping labels)
            if valid and len(seen_pos) <= 1:
                results.append(truth_table)

        if unconditional:
            return ["1"]
        return results

    # --- Optional utilities kept for compatibility, not required by the planner ---

    def get_minimal_length(self):
        """Compute minimal number of transitions between state pairs (used in some analyses)."""
        for head in self.buchi_graph.nodes():
            for tail in self.buchi_graph.nodes():
                if head != tail and "accept" in tail:
                    try:
                        length, _ = nx.algorithms.single_source_dijkstra(
                            self.buchi_graph, source=head, target=tail
                        )
                    except nx.exception.NetworkXNoPath:
                        length = np.inf
                    self.min_length[(head, tail)] = length
                elif head == tail and "accept" in tail:
                    length = np.inf
                    for suc in self.buchi_graph.succ[head]:
                        try:
                            len1, _ = nx.algorithms.single_source_dijkstra(
                                self.buchi_graph, source=suc, target=tail
                            )
                        except nx.exception.NetworkXNoPath:
                            len1 = np.inf
                        if len1 < length:
                            length = len1 + 1
                    self.min_length[(head, tail)] = length

    def get_feasible_accepting_state(self):
        """
        Filter accepting states by reachability (optional).

        Returns:
            bool: whether there exists at least one feasible accepting state
        """
        accept = self.buchi_graph.graph["accept"]
        self.buchi_graph.graph["accept"] = []
        for ac in accept:
            for init in self.buchi_graph.graph["init"]:
                if (
                    self.min_length.get((init, ac), np.inf) < np.inf
                    and self.min_length.get((ac, ac), np.inf) < np.inf
                ):
                    self.buchi_graph.graph["accept"].append(ac)
                    break
        if not self.buchi_graph.graph["accept"]:
            # print('No feasible accepting states!')
            return False
        return True
