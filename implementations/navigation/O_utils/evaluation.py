import os
import re
import cv2
import jax
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from tqdm import trange
from collections import defaultdict
from rich import print as rprint
from rich.live import Live
import rich
import time
from pathlib import Path
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg

from L_utils.ltl_planner import (
    ltl_plan_astar,
    ltl_plan_astar_full,
)
from L_utils.ltl_utils import (
    extract_label_trace,
    pick_pos_extractor,
    set_pos_xy,
    verify_with_buchi_runtime,
    verify_with_buchi_runtime_full,
)
from O_utils.log_utils import get_video_writer


def _kitchen_set_obs_and_goal(*args, **kwargs):
    from D_utils.kitchen_utils import kitchen_set_obs_and_goal

    return kitchen_set_obs_and_goal(*args, **kwargs)


def _kitchen_render(*args, **kwargs):
    from D_utils.kitchen_utils import kitchen_render

    return kitchen_render(*args, **kwargs)


def update_failure_reason(failure_reasons, idx, reason):
    """Update failure reason for episode idx."""
    if failure_reasons[idx] is None:
        failure_reasons[idx] = reason


def log_stat_to_csv(log_dict, csv_path):
    """Log statistics dictionary to CSV file."""
    df = pd.DataFrame([log_dict])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="w", index=False)


def collapse_labels(labels):
    # Collapse consecutive identical labels in prefix_labels
    labels_readable = []
    _prev = object()
    for lab in labels:
        if lab != _prev:
            labels_readable.append(lab)
            _prev = lab

    # Convert sets/lists to readable strings for printing
    return [
        ",".join(sorted(l)) if isinstance(l, (set, list, tuple)) else str(l)
        for l in labels_readable
    ]


def _direction_to_skill(delta, prev_skill=None, smoothing_beta=0.0, epsilon=1e-8):
    """Convert a latent displacement into a unit skill, optionally smoothing turns."""
    norm = np.linalg.norm(delta)
    if norm <= epsilon:
        return np.zeros_like(delta), prev_skill

    skill = delta / (norm + epsilon)
    beta = float(np.clip(smoothing_beta, 0.0, 0.99))
    if beta > 0.0 and prev_skill is not None:
        mixed = (1.0 - beta) * skill + beta * prev_skill
        mixed_norm = np.linalg.norm(mixed)
        if mixed_norm > epsilon:
            skill = mixed / (mixed_norm + epsilon)

    return skill, skill


def _semantic_repulsion_delta(
    phi_obs,
    forbidden_labels,
    key_graph,
    radius,
    gain,
    power,
    max_norm,
    epsilon=1e-8,
):
    if not forbidden_labels or radius <= 0.0 or gain <= 0.0:
        return np.zeros_like(phi_obs)

    label_to_anchors = getattr(key_graph, "label_node_idx_dict", {})
    repulsion = np.zeros_like(phi_obs)
    for label in sorted(set(forbidden_labels)):
        anchor_ids = label_to_anchors.get(label, [])
        if not anchor_ids:
            continue
        valid_ids = [
            int(idx)
            for idx in anchor_ids
            if 0 <= int(idx) < len(key_graph.nodes)
        ]
        if not valid_ids:
            continue

        anchors = np.asarray([key_graph.nodes[idx] for idx in valid_ids])
        diff = phi_obs[None, :] - anchors
        dists = np.linalg.norm(diff, axis=1)
        nearest = int(np.argmin(dists))
        dist = float(dists[nearest])
        if dist >= radius or dist <= epsilon:
            continue

        scale = float(gain) * max(0.0, 1.0 - dist / radius) ** float(power)
        repulsion += scale * diff[nearest] / (dist + epsilon)

    rep_norm = float(np.linalg.norm(repulsion))
    if max_norm > 0.0 and rep_norm > max_norm:
        repulsion = repulsion * (float(max_norm) / (rep_norm + epsilon))
    return repulsion


def _active_forbidden_labels(
    edge_forbidden_labels, progress_idx, target_idx, use_all=False
):
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

    labels = set()
    for edge_labels in edge_range:
        labels.update(edge_labels or [])
    return labels


def evaluate_with_graph(
    agent,
    key_graph,
    env,
    env_name,
    task_id,
    eval_episodes,
    eval_video_episodes,
    seed,
    eval_on_cpu,
    eval_subgoal_threshold,
    eval_final_goal_threshold,
    config,
):
    """
    Evaluate the GAS in the environment.
    In OGbench environments, the final goal includes slight random noise per episode.
    Empirically, we found little to no performance difference between:
    (1) Recomputing the shortest path every episode
    (2) Reusing a precomputed path for the same task_id

    The function `evaluate_with_graph()` follows (2) by computing the shortest path once via `precompute_shortest_paths_to_all_tasks()`
    When using (2), we recommend setting `eval_final_goal_threshold >= 2` to allow agents to reach slightly perturbed final goals.
    If final goals vary significantly per episode, or if GAS is extended to online RL, strategy (1) is preferred.
    """
    eval_agent = (
        jax.device_put(agent, device=jax.devices("cpu")[0]) if eval_on_cpu else agent
    )
    get_phi_fn = eval_agent.get_phi
    actor_fn = supply_rng(eval_agent.sample_actions, rng=jax.random.PRNGKey(seed))

    stats = defaultdict(list)
    renders = []
    for i in trange(
        eval_episodes + eval_video_episodes,
        leave=False,
        desc=f"Task {task_id} Episodes",
    ):
        step = 0
        render = []
        should_render = i >= eval_episodes
        eval_seed = seed + i
        env, observation, goal, reward, done, goal_rendered = setup_task_env(
            env, env_name, task_id, should_render, seed=eval_seed
        )

        epsilon = 1e-10
        phi_obs = np.array(get_phi_fn(observation))
        phi_goal = np.array(get_phi_fn(goal))
        final_goal_on = False
        shortest_path = key_graph.get_shortest_path(
            task_id=task_id, source=phi_obs, force_closest=True
        )
        while not done:
            phi_obs = np.array(get_phi_fn(observation))
            if final_goal_on == True:
                cur_obs_goal = phi_goal
            else:
                cached_shortest_path = key_graph.get_shortest_path(
                    task_id=task_id, source=phi_obs
                )
                if cached_shortest_path is None:
                    pass
                else:
                    shortest_path = cached_shortest_path

                distances = np.linalg.norm(np.array(shortest_path) - phi_obs, axis=1)
                valid_indices = np.where(distances <= eval_subgoal_threshold)[0]
                cur_node_idx = valid_indices[-1] if len(valid_indices) > 0 else 0
                if len(shortest_path) <= eval_final_goal_threshold:
                    final_goal_on = True
                    cur_obs_goal = phi_goal
                else:
                    cur_obs_goal = shortest_path[cur_node_idx]

            skills = (cur_obs_goal - phi_obs) / (
                np.linalg.norm(cur_obs_goal - phi_obs) + epsilon
            )
            action = actor_fn(observations=observation, goals=skills, temperature=0.0)
            action = np.clip(np.array(action), -0.5, 0.5)
            next_observation, reward, done, info = env_step(env, env_name, action)

            step += 1
            if should_render and (step % 3 == 0 or done):
                frame = get_frame(env, env_name)
                render.append(frame)
            observation = next_observation
        add_to(stats, flatten(info))
        if should_render:
            renders.append(np.array(render))
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, renders


def evaluate_with_ltl_full(
    agent,
    key_graph,
    env,
    env_name,
    buchi_runtime,
    start_state_getter=None,
    get_phi_override=None,
    seed=0,
    eval_on_cpu=False,
    eval_subgoal_threshold=1.0,
    eval_final_goal_threshold=2,
    avoid_soft_label_threshold=0.5,
    max_prefix_candidates=3,
    lambda_weight=1.0,
    ltl_labels=None,
    ltl_formula_eform=None,
    label_fn=None,
    ltl_label_to_literal=None,
    use_spot_first=True,
    config=None,
    regions=None,
    imgpath=None,
    logpath=None,
    max_steps=5000,
    action_clip=1.0,
    suffix_cycles=2,
    direction_smoothing_beta=0.0,
    semantic_repulsion_enable=False,
    semantic_repulsion_radius=-1.0,
    semantic_repulsion_gain=0.3,
    semantic_repulsion_power=1.0,
    semantic_repulsion_max_norm=0.5,
    anchor_reached_threshold=None,
    vis_simple_mode=False,
    render_wp=True,
):
    """
    Evaluate full prefix + suffix plan under a finite lasso surrogate.
    """
    suffix_cycles = max(1, int(suffix_cycles))
    if anchor_reached_threshold is None or float(anchor_reached_threshold) <= 0:
        anchor_reached_threshold = float(eval_subgoal_threshold)
    else:
        anchor_reached_threshold = float(anchor_reached_threshold)
    semantic_repulsion_enable = bool(semantic_repulsion_enable)
    if semantic_repulsion_radius is None or float(semantic_repulsion_radius) <= 0:
        semantic_repulsion_radius = float(eval_subgoal_threshold)
    else:
        semantic_repulsion_radius = float(semantic_repulsion_radius)
    semantic_repulsion_gain = float(semantic_repulsion_gain)
    semantic_repulsion_power = float(semantic_repulsion_power)
    semantic_repulsion_max_norm = float(semantic_repulsion_max_norm)

    def reset_env_simple(env, seed):
        out = env.reset(seed=seed)
        # Special setting for ltl task, do not terminate at goal
        env.unwrapped._terminate_at_goal = False
        return out[0], out[1] if isinstance(out, tuple) else (out, {})

    def _collect_anchor_labels(key_graph):
        node2lab = {}
        if getattr(key_graph, "graph", None):
            for nid in key_graph.graph.nodes:
                lab = key_graph.graph.nodes[nid].get("hard_label", None)
                if lab is not None:
                    node2lab[nid] = lab
        if not node2lab and getattr(key_graph, "label_node_idx_dict", None):
            for lab, nlist in key_graph.label_node_idx_dict.items():
                for nid in nlist:
                    node2lab[nid] = lab
        return node2lab

    eval_agent = (
        jax.device_put(agent, device=jax.devices("cpu")[0]) if eval_on_cpu else agent
    )
    get_phi_fn = get_phi_override or eval_agent.get_phi
    actor_fn = supply_rng(eval_agent.sample_actions, rng=jax.random.PRNGKey(seed))

    stats = defaultdict(list)
    failure_reasons = [None]  # None if success else reason string
    node2label = _collect_anchor_labels(key_graph)
    pos_extractor = pick_pos_extractor(env_name)

    def _region_center_and_margin(label):
        if not regions or label not in regions:
            return None, None

        region = regions[label]
        region_type = getattr(region, "type", None)
        if region_type == "circle" and hasattr(region, "center"):
            return np.array(region.center, dtype=np.float32), 0.5 * float(region.radius)
        if region_type == "rect":
            center = np.array(
                [
                    0.5 * (float(region.xmin) + float(region.xmax)),
                    0.5 * (float(region.ymin) + float(region.ymax)),
                ],
                dtype=np.float32,
            )
            margin = 0.25 * min(
                float(region.xmax) - float(region.xmin),
                float(region.ymax) - float(region.ymin),
            )
            return center, margin
        if region_type == "polygon" and hasattr(region, "vertices"):
            vertices = np.array(region.vertices, dtype=np.float32)
            center = np.mean(vertices, axis=0)
            margin = 0.25 * float(np.min(np.linalg.norm(vertices - center, axis=1)))
            return center, margin
        return None, None

    def _self_loop_center_ready(label, obs):
        center, margin = _region_center_and_margin(label)
        if center is None or margin is None:
            return True, center
        xy = np.array(pos_extractor(obs), dtype=np.float32)
        return bool(np.linalg.norm(xy - center) <= margin), center

    # Recording variables
    success_flags = []
    exec_times_list = []
    steps_list = []
    exec_times = {}
    steps = {}
    success = False
    frame_cnt = 0
    repulsion_stats = {
        "active_steps": 0,
        "sum_norm": 0.0,
        "max_norm": 0.0,
    }

    # Reset env and apply start state
    observation, _ = reset_env_simple(env, seed=seed)
    if start_state_getter:
        start_raw_state = start_state_getter(env, observation)
        observation = start_raw_state
    else:
        start_raw_state = observation

    start_t = time.time()
    plan_metadata = {}
    plan, plan_failure = ltl_plan_astar_full(
        key_graph,
        buchi_runtime,
        start_raw_state,
        get_phi_fn,
        avoid_soft_label_threshold=avoid_soft_label_threshold,
        lambda_weight=lambda_weight,
        max_prefix_candidates=max_prefix_candidates,
    )

    if not plan:
        if plan_failure is not None:
            update_failure_reason(failure_reasons, 0, plan_failure)
        rprint(f"[yellow]Planning failed: {plan_failure}[/yellow]")
        prefix_step = 0
        suffix_step = 0
        steps["prefix_steps"] = 0
        steps["suffix_steps"] = 0
    else:
        exec_times["planning"] = time.time() - start_t
        plan_metadata = dict(plan.get("metadata", {}))

        # Split prefix/suffix
        path_arr = np.array(plan["positions"], dtype=np.float32)
        node_idx_path = plan["node_indices"]
        anchor_mask = plan["anchor_mask"]
        prefix_len = plan.get("prefix_length", len(path_arr))

        path_prefix = path_arr[:prefix_len]
        path_suffix = path_arr[prefix_len:]
        node_prefix = node_idx_path[:prefix_len]
        node_suffix = node_idx_path[prefix_len:]
        anchor_prefix = anchor_mask[:prefix_len]
        anchor_suffix = anchor_mask[prefix_len:]
        prefix_edge_forbidden = plan.get("prefix_edge_forbidden_labels", [])
        suffix_edge_forbidden = plan.get("suffix_edge_forbidden_labels", [])

        def run_path_segment(
            path_segment,
            node_indices,
            anchor_mask,
            edge_forbidden_labels=None,
            prior_obs=None,
        ):
            nonlocal observation
            obs_segment = [observation]
            anchor_wp = [k for k, is_a in enumerate(anchor_mask) if is_a]
            if len(anchor_wp) == 0:
                anchor_wp = [len(path_segment) - 1]
            current_wp_ptr = 0
            target_idx = anchor_wp[current_wp_ptr]
            self_loop_check = False
            if len(anchor_wp) == 2:
                start = anchor_wp[0]
                end = anchor_wp[1]
                path_segment_start = path_segment[start]
                path_segment_end = path_segment[end]
                self_loop_check = np.allclose(path_segment_start, path_segment_end)
            progress_idx = 0
            lock_on_anchor = False
            step, success = 0, False
            epsilon = 1e-10
            prev_skill = None
            notmove_steps = 0
            overturn_check_steps = 75
            potential_overturn = False
            live = None

            try:
                while step < max_steps and not success:
                    phi_obs = np.array(get_phi_fn(observation))
                    anchor_node_idx = node_indices[target_idx]
                    anchor_label = node2label.get(anchor_node_idx, None)
                    self_loop_ready_before_step, self_loop_center = (
                        _self_loop_center_ready(anchor_label, observation)
                        if self_loop_check
                        else (False, None)
                    )
                    segment = path_segment[progress_idx : target_idx + 1]
                    dists = np.linalg.norm(segment - phi_obs, axis=1)
                    cur_goal_render_info = (None, None)
                    if lock_on_anchor:
                        cur_obs_goal = path_segment[target_idx]
                        cur_goal_render_info = (target_idx, "anchor")
                    else:
                        reachable_local = np.where(dists <= eval_subgoal_threshold)[0]
                        if len(reachable_local) > 0:
                            cur_node_idx = progress_idx + int(reachable_local[-1])
                        else:
                            nearest_local = int(np.argmin(dists))
                            cur_node_idx = max(progress_idx, progress_idx + nearest_local)
                        if (target_idx - cur_node_idx) <= int(eval_final_goal_threshold):
                            lock_on_anchor = True
                            cur_obs_goal = path_segment[target_idx]
                            cur_goal_render_info = (target_idx, "anchor")
                        else:
                            cur_obs_goal = path_segment[cur_node_idx]
                            cur_goal_render_info = (cur_node_idx, "subgoal")

                    if (
                        self_loop_check
                        and not self_loop_ready_before_step
                        and self_loop_center is not None
                    ):
                        center_obs = set_pos_xy(observation, self_loop_center)
                        cur_obs_goal = np.array(get_phi_fn(center_obs))
                        cur_goal_render_info = (target_idx, "self-loop-center")

                    delta = cur_obs_goal - phi_obs
                    if semantic_repulsion_enable:
                        forbidden_labels = _active_forbidden_labels(
                            edge_forbidden_labels,
                            progress_idx,
                            target_idx,
                            use_all=self_loop_check,
                        )
                        rep_delta = _semantic_repulsion_delta(
                            phi_obs,
                            forbidden_labels,
                            key_graph,
                            semantic_repulsion_radius,
                            semantic_repulsion_gain,
                            semantic_repulsion_power,
                            semantic_repulsion_max_norm,
                            epsilon,
                        )
                        rep_norm = float(np.linalg.norm(rep_delta))
                        if rep_norm > epsilon:
                            delta = delta + rep_delta
                            repulsion_stats["active_steps"] += 1
                            repulsion_stats["sum_norm"] += rep_norm
                            repulsion_stats["max_norm"] = max(
                                repulsion_stats["max_norm"], rep_norm
                            )
                    skills, prev_skill = _direction_to_skill(
                        delta, prev_skill, direction_smoothing_beta, epsilon
                    )
                    action = actor_fn(
                        observations=observation, goals=skills, temperature=0.0
                    )
                    action = np.clip(np.array(action), -action_clip, action_clip)

                    used_zero_action = self_loop_check and self_loop_ready_before_step
                    if used_zero_action:
                        action = np.zeros_like(action)

                    next_observation, reward, env_done, info = env_step(
                        env, env_name, action
                    )

                    last_xy = observation[:2]
                    curr_xy = next_observation[:2]
                    if (
                        np.linalg.norm(curr_xy - last_xy) < 1e-5
                        and np.linalg.norm(action) > 0.1
                        and not success
                    ):
                        notmove_steps += 1
                        if notmove_steps >= overturn_check_steps:
                            potential_overturn = True
                            break
                    else:
                        potential_overturn = False
                        notmove_steps = 0

                    obs_segment.append(next_observation)
                    observation = next_observation
                    step += 1

                    seg_d = np.linalg.norm(
                        path_segment[progress_idx : target_idx + 1] - phi_obs,
                        axis=1,
                    )
                    attained = np.where(seg_d <= eval_subgoal_threshold)[0]
                    if len(attained) > 0:
                        progress_idx += int(attained[-1])

                    label_ok = True
                    if label_fn and anchor_label:
                        label_ok = label_fn(observation) == anchor_label
                    anchor_hit = (
                        np.linalg.norm(phi_obs - path_segment[target_idx])
                        <= anchor_reached_threshold
                    )

                    if self_loop_check:
                        self_loop_ready_after_step, _ = _self_loop_center_ready(
                            anchor_label, observation
                        )
                        anchor_hit = used_zero_action and self_loop_ready_after_step
                    elif current_wp_ptr == len(anchor_wp) - 1:
                        anchor_hit = True

                    if anchor_hit and label_ok:
                        current_wp_ptr += 1
                        if current_wp_ptr >= len(anchor_wp):
                            success = True
                        else:
                            target_idx = anchor_wp[current_wp_ptr]
                            progress_idx = min(progress_idx, target_idx)
                            lock_on_anchor = False
            finally:
                if live is not None:
                    live.stop()

            return obs_segment, step, potential_overturn, success

        prefix_start_t = time.time()
        prefix_obs, prefix_step, potential_overturn_prefix, prefix_completed = (
            run_path_segment(
                path_prefix,
                node_prefix,
                anchor_prefix,
                edge_forbidden_labels=prefix_edge_forbidden,
                prior_obs=None,
            )
        )
        prefix_end_t = time.time()
        if potential_overturn_prefix:
            rprint("[yellow]Detected overturn in prefix[/yellow]")
            update_failure_reason(failure_reasons, 0, "Overturn during prefix")
        if not prefix_completed:
            update_failure_reason(failure_reasons, 0, "Prefix execution incomplete")
        exec_times["prefix_execution"] = prefix_end_t - prefix_start_t

        suffix_start_t = time.time()
        suffix_obs = []
        suffix_one_cycle_obs = []
        suffix_step = 0
        suffix_one_cycle_step = np.nan
        suffix_one_cycle_completed = 0
        suffix_cycle_steps = []
        suffix_cycles_completed = 0
        potential_overturn_suffix = False
        for cycle_idx in range(suffix_cycles):
            (
                cur_suffix_obs,
                cur_suffix_step,
                cur_potential_overturn,
                cur_suffix_completed,
            ) = run_path_segment(
                path_suffix,
                node_suffix,
                anchor_suffix,
                edge_forbidden_labels=suffix_edge_forbidden,
                prior_obs=prefix_obs,
            )
            suffix_cycle_steps.append(cur_suffix_step)
            suffix_step += cur_suffix_step
            potential_overturn_suffix = (
                potential_overturn_suffix or cur_potential_overturn
            )
            if cycle_idx == 0:
                suffix_obs = list(cur_suffix_obs)
                suffix_one_cycle_obs = list(cur_suffix_obs)
                suffix_one_cycle_step = cur_suffix_step
                suffix_one_cycle_completed = int(cur_suffix_completed)
            else:
                suffix_obs.extend(cur_suffix_obs[1:])
            if cur_suffix_completed:
                suffix_cycles_completed += 1
            else:
                update_failure_reason(
                    failure_reasons,
                    0,
                    f"Suffix cycle {cycle_idx + 1} incomplete",
                )
                break

        if potential_overturn_suffix:
            rprint("[yellow]Detected overturn in suffix[/yellow]")
            update_failure_reason(failure_reasons, 0, "Overturn during suffix")
        exec_times["suffix_execution"] = time.time() - suffix_start_t
        steps["prefix_steps"] = prefix_step
        steps["suffix_steps"] = suffix_step
        steps["suffix_one_cycle_steps"] = suffix_one_cycle_step
        steps["suffix_one_cycle_completed"] = suffix_one_cycle_completed
        steps["suffix_cycles_requested"] = suffix_cycles
        steps["suffix_cycles_completed"] = suffix_cycles_completed
        steps["suffix_cycle_steps"] = suffix_cycle_steps

        # ---- verify prefix+suffix label trace ----
        verify_ok = False
        suffix_one_cycle_verify_ok = False
        suffix_one_cycle_failure = "Suffix one-cycle check not evaluated"
        verification_start_t = time.time()
        if ltl_labels and ltl_formula_eform and label_fn:
            prefix_labels = extract_label_trace(prefix_obs, label_fn)
            suffix_one_cycle_labels = extract_label_trace(suffix_one_cycle_obs, label_fn)
            suffix_labels = extract_label_trace(suffix_obs, label_fn)
            suffix_one_cycle_verify_ok, suffix_one_cycle_failure = (
                verify_with_buchi_runtime_full(
                    buchi_runtime,
                    ltl_label_to_literal,
                    prefix_labels,
                    suffix_one_cycle_labels,
                )
            )
            lasso_verify_ok, check_failure = verify_with_buchi_runtime_full(
                buchi_runtime, ltl_label_to_literal, prefix_labels, suffix_labels
            )
            cycles_ok = suffix_cycles_completed == suffix_cycles
            verify_ok = bool(lasso_verify_ok and cycles_ok and prefix_completed)

            if not cycles_ok:
                check_failure = (
                    f"Only completed {suffix_cycles_completed}/{suffix_cycles} "
                    "suffix cycles"
                )
            elif not prefix_completed:
                check_failure = "Prefix execution incomplete"

            if suffix_one_cycle_verify_ok:
                rprint(
                    "[cyan]One suffix traversal check: success "
                    f"({suffix_one_cycle_step} steps)[/cyan]"
                )
            else:
                rprint(
                    "[yellow]One suffix traversal check failed: "
                    f"{suffix_one_cycle_failure}[/yellow]"
                )

            if not verify_ok and check_failure is not None:
                rprint(f"[yellow]{check_failure}[/yellow]")
                rprint(
                    f"[yellow]Prefix labels: {collapse_labels(prefix_labels)}[/yellow]"
                )
                rprint(
                    f"[yellow]Suffix labels: {collapse_labels(suffix_labels)}[/yellow]"
                )
                update_failure_reason(failure_reasons, 0, check_failure)

        success = bool(verify_ok)
        steps["suffix_one_cycle_success"] = int(suffix_one_cycle_verify_ok)
        exec_times["verification"] = time.time() - verification_start_t

        if regions:
            os.makedirs(imgpath, exist_ok=True)
            imgfile = os.path.join(imgpath, "ltl_traj.pdf")
            generate_visualization(
                env,
                prefix_obs,
                regions,
                start_raw_state,
                imgfile,
                suffix_obs=suffix_obs,
                formula=ltl_formula_eform,
                more_info="(success)" if verify_ok else "(failure)",
                simple_mode=vis_simple_mode,
            )
        if not bool(verify_ok):
            exec_times = {}

    # Record result
    prefix_step = steps.get("prefix_steps", 0)
    suffix_step = steps.get("suffix_steps", 0)
    suffix_one_cycle_success = steps.get("suffix_one_cycle_success", 0)
    suffix_one_cycle_step = steps.get("suffix_one_cycle_steps", np.nan)
    suffix_cycles_completed = steps.get("suffix_cycles_completed", 0)
    suffix_cycles_requested = steps.get("suffix_cycles_requested", suffix_cycles)
    rprint(
        f"[cyan]one suffix traversal: {suffix_one_cycle_success}/1 "
        f"({suffix_one_cycle_step} steps)[/cyan]"
    )
    if success:
        rprint(
            f"[green]success with steps {prefix_step} + {suffix_step} "
            f"({suffix_cycles_completed}/{suffix_cycles_requested} suffix cycles)[/green]"
        )
    else:
        rprint(f"[red]failed[/red]")

    success_flags.append(success)
    exec_times_list.append(exec_times)
    steps_list.append(steps)

    plan_times = [et.get("planning", np.nan) for et in exec_times_list]
    prefix_times = [et.get("prefix_execution", np.nan) for et in exec_times_list]
    suffix_times = [et.get("suffix_execution", np.nan) for et in exec_times_list]
    verification_times = [et.get("verification", np.nan) for et in exec_times_list]
    prefix_steps = [s.get("prefix_steps", np.nan) for s in steps_list]
    suffix_steps = [s.get("suffix_steps", np.nan) for s in steps_list]
    suffix_one_cycle_steps = [
        s.get("suffix_one_cycle_steps", np.nan) for s in steps_list
    ]
    suffix_one_cycle_completed = [
        s.get("suffix_one_cycle_completed", np.nan) for s in steps_list
    ]
    suffix_one_cycle_success = [
        s.get("suffix_one_cycle_success", 0) for s in steps_list
    ]
    suffix_cycles_requested = [
        s.get("suffix_cycles_requested", np.nan) for s in steps_list
    ]
    suffix_cycles_completed = [
        s.get("suffix_cycles_completed", np.nan) for s in steps_list
    ]
    suffix_cycle_steps = [s.get("suffix_cycle_steps", []) for s in steps_list]
    anchor_stats = getattr(key_graph, "anchor_connection_stats", {})
    anchor_unavailable_labels = getattr(key_graph, "anchor_unavailable_labels", [])
    anchor_attempts = sum(v.get("attempted", 0) for v in anchor_stats.values())
    anchor_added = sum(v.get("added", 0) for v in anchor_stats.values())
    automaton_anchor_prune_stats = getattr(buchi_runtime, "anchor_prune_stats", {})
    stats = {
        "ltl_formula": ltl_formula_eform,
        "planner_variant": "sagas_joint",
        "executor_variant": "gas_policy",
        "legacy_plan_algorithm": "astar",
        "planning_failure_reason": plan_failure,
        "num_prefix_candidates_found": plan_metadata.get(
            "num_prefix_candidates_found", np.nan
        ),
        "selected_prefix_candidate_rank": plan_metadata.get(
            "selected_prefix_candidate_rank", np.nan
        ),
        "num_expanded_product_states": plan_metadata.get(
            "num_expanded_product_states", np.nan
        ),
        "num_expanded_prefix_states": plan_metadata.get(
            "num_expanded_prefix_states", np.nan
        ),
        "num_expanded_suffix_states": plan_metadata.get(
            "num_expanded_suffix_states", np.nan
        ),
        "num_greedy_prefix_steps": plan_metadata.get(
            "num_greedy_prefix_steps", np.nan
        ),
        "num_greedy_suffix_steps": plan_metadata.get(
            "num_greedy_suffix_steps", np.nan
        ),
        "direction_smoothing_beta": float(direction_smoothing_beta),
        "semantic_repulsion_enable": int(semantic_repulsion_enable),
        "semantic_repulsion_radius": float(semantic_repulsion_radius),
        "semantic_repulsion_gain": float(semantic_repulsion_gain),
        "semantic_repulsion_power": float(semantic_repulsion_power),
        "semantic_repulsion_max_norm": float(semantic_repulsion_max_norm),
        "semantic_repulsion_active_steps": int(repulsion_stats["active_steps"]),
        "semantic_repulsion_mean_norm": (
            float(repulsion_stats["sum_norm"] / repulsion_stats["active_steps"])
            if repulsion_stats["active_steps"] > 0
            else 0.0
        ),
        "semantic_repulsion_max_observed_norm": float(repulsion_stats["max_norm"]),
        "eval_subgoal_threshold": float(eval_subgoal_threshold),
        "anchor_reached_threshold": float(anchor_reached_threshold),
        "success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
        "success_flags": [1 if s else 0 for s in success_flags],
        "failure_reasons": failure_reasons,
        "plan_times": plan_times,
        "prefix_times": prefix_times,
        "suffix_times": suffix_times,
        "verification_times": verification_times,
        "total_times": [
            np.sum([pt, preft, suft, vt])
            for pt, preft, suft, vt in zip(
                plan_times, prefix_times, suffix_times, verification_times
            )
        ],
        "prefix_steps": prefix_steps,
        "suffix_steps": suffix_steps,
        "suffix_one_cycle_steps": suffix_one_cycle_steps,
        "suffix_one_cycle_completed": suffix_one_cycle_completed,
        "suffix_one_cycle_success": suffix_one_cycle_success,
        "suffix_cycles_requested": suffix_cycles_requested,
        "suffix_cycles_completed": suffix_cycles_completed,
        "suffix_cycle_steps": suffix_cycle_steps,
        "anchor_attempts": anchor_attempts,
        "anchor_added": anchor_added,
        "anchor_unavailable_count": len(anchor_unavailable_labels),
        "anchor_unavailable_labels": anchor_unavailable_labels,
        "anchor_connection_stats": anchor_stats,
        "automaton_anchor_pruned_edges": automaton_anchor_prune_stats.get(
            "removed_edges", 0
        ),
        "automaton_anchor_accepting_after_prune": automaton_anchor_prune_stats.get(
            "accepting_after", np.nan
        ),
        "automaton_anchor_has_accepting_run": automaton_anchor_prune_stats.get(
            "has_accepting_run", np.nan
        ),
        "automaton_anchor_prune_stats": automaton_anchor_prune_stats,
        "total_steps": [np.sum([ps, ss]) for ps, ss in zip(prefix_steps, suffix_steps)],
    }

    if logpath is not None and os.path.exists(logpath):
        log_stat_to_csv(stats, os.path.join(logpath, "ltl_evaluation_stats.csv"))

    return stats


def evaluate_with_ltl(
    agent,
    key_graph,
    env,  # original env
    env_name,
    buchi_runtime,  # for planning only
    start_state_getter=None,
    get_phi_override=None,
    seed=0,
    eval_on_cpu=False,
    eval_subgoal_threshold=1.0,
    eval_final_goal_threshold=2,
    avoid_soft_label_threshold=0.5,
    # offline validation args
    ltl_labels=None,  # ["A","B","C"]
    ltl_formula_eform=None,  # "(F e1) & (F e2) & (G !e3)"
    label_fn=None,  # raw_state -> Optional[label]
    ltl_label_to_literal=None,  # {"A":"l1_1",...}
    use_spot_first=True,
    config=None,
    regions=None,
    imgpath=None,
    action_clip=1.0,
    direction_smoothing_beta=0.0,
):
    """
    LTL evaluation that follows the sequence of ANCHOR nodes from the planned path (single rollout).
    """

    # --------- helpers ---------
    def reset_env_simple(env, seed):
        out = env.reset(seed=seed)
        # Special setting for ltl task, do not terminate at goal
        env.unwrapped._terminate_at_goal = False
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]
        return out, {}

    def _collect_anchor_labels(key_graph):
        """Return mapping: node_idx -> label string (if available)."""
        node2lab = {}
        # Prefer node attribute 'anchor_label' if present
        if getattr(key_graph, "graph", None) is not None:
            for nid in key_graph.graph.nodes:
                lab = key_graph.graph.nodes[nid].get("hard_label", None)
                if lab is not None:
                    node2lab[nid] = lab
        # Fallback to label_node_idx_dict if provided
        if not node2lab and getattr(key_graph, "label_node_idx_dict", None):
            for lab, nlist in key_graph.label_node_idx_dict.items():
                for nid in nlist:
                    node2lab[nid] = lab
        return node2lab

    # --------- setup ---------
    eval_agent = (
        jax.device_put(agent, device=jax.devices("cpu")[0]) if eval_on_cpu else agent
    )
    get_phi_fn = get_phi_override or eval_agent.get_phi
    actor_fn = supply_rng(eval_agent.sample_actions, rng=jax.random.PRNGKey(seed))

    stats = defaultdict(list)
    renders, success_flags = [], []
    anchor_hit_tol = float(eval_subgoal_threshold)

    node2label = _collect_anchor_labels(key_graph)
    failure_reasons = [None]
    exec_times_list = []
    steps_list = []

    eval_seed = seed
    observation, _ = reset_env_simple(env, seed=eval_seed)

    if start_state_getter:
        start_raw_state = start_state_getter(env, observation)
        observation = start_raw_state
    else:
        start_raw_state = observation

    exec_times = {}
    start_t = time.time()

    plan = ltl_plan_astar(
        key_graph,
        buchi_runtime,
        start_raw_state,
        get_phi_fn,
        avoid_soft_label_threshold=avoid_soft_label_threshold,
    )
    if not plan:
        update_failure_reason(failure_reasons, 0, "Planning failed")
        success_flags.append(False)
    else:
        exec_times["planning"] = time.time() - start_t

        path_arr = np.array(plan["positions"], dtype=np.float32)
        node_idx_path = plan["node_indices"]
        anchor_mask = plan["anchor_mask"]

        anchor_waypoints = [k for k, is_a in enumerate(anchor_mask) if is_a]
        if len(anchor_waypoints) == 0:
            anchor_waypoints = [len(path_arr) - 1]

        epsilon = 1e-10
        obs_traj = [observation]
        render_frames = []
        max_steps = 5000
        current_wp_ptr = 0
        target_idx = anchor_waypoints[current_wp_ptr]
        progress_idx = 0
        lock_on_anchor = False
        step = 0
        prev_skill = None
        success = False
        notmove_steps = 0
        overturn_check_steps = 75
        potential_overturn = False

        execution_start_t = time.time()
        while step < max_steps and not success:
            phi_obs = np.array(get_phi_fn(observation))

            if lock_on_anchor:
                cur_obs_goal = path_arr[target_idx]
            else:
                segment = path_arr[progress_idx : target_idx + 1]
                dists = np.linalg.norm(segment - phi_obs, axis=1)
                reachable_local = np.where(dists <= eval_subgoal_threshold)[0]
                if len(reachable_local) > 0:
                    cur_node_idx = progress_idx + int(reachable_local[-1])
                else:
                    nearest_local = int(np.argmin(dists))
                    cur_node_idx = max(progress_idx, progress_idx + nearest_local)
                if (target_idx - cur_node_idx) <= int(eval_final_goal_threshold):
                    lock_on_anchor = True
                    cur_obs_goal = path_arr[target_idx]
                else:
                    cur_obs_goal = path_arr[cur_node_idx]

            delta = cur_obs_goal - phi_obs
            skills, prev_skill = _direction_to_skill(
                delta, prev_skill, direction_smoothing_beta, epsilon
            )
            action = actor_fn(observations=observation, goals=skills, temperature=0.0)
            action = np.clip(np.array(action), -action_clip, action_clip)

            next_observation, reward, env_done, info = env_step(env, env_name, action)

            last_xy = observation[:2]
            curr_xy = next_observation[:2]
            if (
                np.linalg.norm(curr_xy - last_xy) < 1e-5
                and np.linalg.norm(action) > 0.1
                and not success
            ):
                notmove_steps += 1
                if notmove_steps >= overturn_check_steps:
                    potential_overturn = True
                    break
            else:
                potential_overturn = False
                notmove_steps = 0

            obs_traj.append(next_observation)
            observation = next_observation
            step += 1

            seg = path_arr[progress_idx : target_idx + 1]
            seg_d = np.linalg.norm(seg - phi_obs, axis=1)
            attained = np.where(seg_d <= eval_subgoal_threshold)[0]
            if len(attained) > 0:
                progress_idx = progress_idx + int(attained[-1])

            anchor_dist_ok = (
                np.linalg.norm(phi_obs - path_arr[target_idx]) <= anchor_hit_tol
            )

            label_ok = True
            if label_fn is not None:
                anchor_node_idx = node_idx_path[target_idx]
                anchor_label = node2label.get(anchor_node_idx, None)
                if anchor_label is not None:
                    label_ok = label_fn(observation) == anchor_label

            if anchor_dist_ok and label_ok:
                current_wp_ptr += 1
                if current_wp_ptr >= len(anchor_waypoints):
                    success = True
                else:
                    target_idx = anchor_waypoints[current_wp_ptr]
                    progress_idx = min(progress_idx, target_idx)
                    lock_on_anchor = False

        if potential_overturn:
            update_failure_reason(failure_reasons, 0, "Potential overturn during execution")

        exec_times["execution"] = time.time() - execution_start_t
        steps_list.append(step)

        ok = False
        verification_start_t = time.time()
        if (
            (ltl_labels is not None)
            and (ltl_formula_eform is not None)
            and (label_fn is not None)
        ):
            label_trace = extract_label_trace(obs_traj, label_fn)
            ok = verify_with_buchi_runtime(
                buchi_runtime, ltl_label_to_literal, label_trace
            )
            if not ok:
                update_failure_reason(failure_reasons, 0, "LTL verification failed")
        exec_times["verification"] = time.time() - verification_start_t

        success_flags.append(bool(success and ok))

        if regions is not None:
            if not os.path.exists(imgpath):
                os.makedirs(imgpath)
            imgpath_case = os.path.join(imgpath, "ltl_execution_traj.pdf")
            generate_visualization(
                env,
                obs_traj,
                regions,
                start_raw_state,
                imgpath_case,
                formula=ltl_formula_eform,
            )

        if bool(ok):
            exec_times_list.append(exec_times)

    # Record result
    if success:
        rprint(f"[green]success with steps {step}[/green]")
    else:
        rprint(f"[red]failed[/red]")

    success_flags.append(success)
    if exec_times:
        exec_times_list.append(exec_times)
    steps_list.append(step)

    plan_times = [et.get("planning", np.nan) for et in exec_times_list]
    exec_times_vals = [et.get("execution", np.nan) for et in exec_times_list]
    verification_times = [et.get("verification", np.nan) for et in exec_times_list]
    stats = {
        "ltl_formula": ltl_formula_eform,
        "direction_smoothing_beta": float(direction_smoothing_beta),
        "success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
        "success_flags": [1 if s else 0 for s in success_flags],
        "failure_reasons": failure_reasons,
        "avg_plan_times": (
            np.mean([et["planning"] for et in exec_times_list])
            if exec_times_list
            else 0.0
        ),
        "avg_execution_times": (
            np.mean([et["execution"] for et in exec_times_list])
            if exec_times_list
            else 0.0
        ),
        "avg_verification_times": (
            np.mean([et["verification"] for et in exec_times_list])
            if exec_times_list
            else 0.0
        ),
        "avg_total_times": (
            np.mean([sum(et.values()) for et in exec_times_list])
            if exec_times_list
            else 0.0
        ),
        "ltl_formula": ltl_formula_eform,
        "avg_total_steps": float(np.mean(steps_list)) if steps_list else 0.0,
    }
    return stats, renders


def plot_trajectory(
    fig,
    ax,
    env,
    prefix_obs,
    regions,
    start_state,
    suffix_obs=None,
    color_gradient=True,
    formula=None,
    more_info="",
    simple_mode=False,
    show_title=True,
):
    # Convert observation_sequence to numpy array if it's a list
    prefix_obs = np.array(prefix_obs)

    linestyle = {
        "markersize": 1,
        "linewidth": 1.5,
        "alpha": 0.4,
    }

    textstyle = {
        "color": "black",
        "fontsize": 15,
        "ha": "center",
        "va": "top",
        "weight": "bold",
        "alpha": 0.7,
    }

    if env:
        maze_map = env.get_maze_map()

        # OGBench maze plotting parameters
        y_grids, x_grids = maze_map.shape
        maze_params = env.unwrapped.get_maze_params()
        x_range = maze_params["x_range"]
        y_range = maze_params["y_range"]
        cmap = ListedColormap(["white", "gray"])
        ax.imshow(
            maze_map,
            extent=[*x_range, *y_range],
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

    ax.set_aspect("equal", adjustable="box")
    ax.scatter(start_state[0], start_state[1], color="red", s=50, label="Start State")

    handles = []
    # Plot prefix & suffix trajectory
    if suffix_obs is not None:
        suffix_obs = np.array(suffix_obs)
    if color_gradient:
        blue_colors = ["#B3D9FF", "#66B3FF", "#3399FF", "#0066CC", "#003399"]
        custom_cmap_blues = LinearSegmentedColormap.from_list(
            "custom_blues", blue_colors
        )
        points_prefix = np.array([prefix_obs[:, 0], prefix_obs[:, 1]]).T.reshape(
            -1, 1, 2
        )
        segments_prefix = np.concatenate(
            [points_prefix[:-1], points_prefix[1:]], axis=1
        )
        colors_prefix = np.linspace(0, 1, len(segments_prefix))
        lc_prefix = LineCollection(
            segments_prefix, cmap=custom_cmap_blues, array=colors_prefix
        )
        ax.add_collection(lc_prefix)
        handles.append(
            Line2D([0], [0], color=blue_colors[1], lw=2, label="Prefix Path")
        )
        if suffix_obs is not None:
            red_colors = ["#FFB3B3", "#FF6666", "#FF0000", "#CC0000", "#990000"]
            custom_cmap_reds = LinearSegmentedColormap.from_list(
                "custom_reds", red_colors
            )
            points_suffix = np.array([suffix_obs[:, 0], suffix_obs[:, 1]]).T.reshape(
                -1, 1, 2
            )
            segments_suffix = np.concatenate(
                [points_suffix[:-1], points_suffix[1:]], axis=1
            )
            colors_suffix = np.linspace(0, 1, len(segments_suffix))
            lc_suffix = LineCollection(
                segments_suffix, cmap=custom_cmap_reds, array=colors_suffix
            )
            ax.add_collection(lc_suffix)
            handles.append(
                Line2D([0], [0], color=red_colors[1], lw=2, label="Suffix Path")
            )

            if not simple_mode:
                # 两个 colorbar 并排
                cax1 = fig.add_axes(
                    [0.92, 0.53, 0.01, 0.25]
                )  # [left, bottom, width, height]
                cax2 = fig.add_axes([0.92, 0.21, 0.01, 0.25])

                cb1 = fig.colorbar(lc_prefix, cax=cax1)
                cb2 = fig.colorbar(lc_suffix, cax=cax2)

                # 统一标题（放在上方）
                fig.text(
                    0.98,
                    0.5,
                    "Steps / Total steps",
                    rotation=90,
                    va="center",
                    fontsize=10,
                )
        elif not simple_mode:
            # 只有一个 colorbar（居中右侧）
            cax = fig.add_axes([0.95, 0.3, 0.01, 0.4])
            cb = fig.colorbar(lc_prefix, cax=cax)
            cb.set_label("Steps / Total steps", fontsize=10)
    else:
        ax.plot(
            prefix_obs[:, 0],
            prefix_obs[:, 1],
            "-o",
            color="blue",
            label="Prefix Path",
            **linestyle,
        )

        if suffix_obs is not None:
            ax.plot(
                suffix_obs[:, 0],
                suffix_obs[:, 1],
                "-o",
                color="pink",
                label="Suffix Path",
                **linestyle,
            )

    for lab, region in regions.items():
        if region.type == "circle":
            (x, y) = region.center
            r = region.radius
            circle = plt.Circle((x, y), r, color=region.color)
            ax.add_patch(circle)
            ax.text(
                x,
                y,
                lab,
                **textstyle,
            )
        elif region.type == "rect":
            (xmin, xmax, ymin, ymax) = (
                region.xmin,
                region.xmax,
                region.ymin,
                region.ymax,
            )
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                color=region.color,
            )
            ax.add_patch(rect)
            ax.text(
                xmin + (xmax - xmin) / 2,
                ymin + (ymax - ymin) / 2,
                lab,
                **textstyle,
            )
        elif region.type == "polygon":
            vertices = np.array(region.vertices)
            poly = plt.Polygon(
                vertices,
                fill=True,
                facecolor=region.color,
                edgecolor="black",
            )
            ax.add_patch(poly)
            # ax.plot(vertices[:, 0], vertices[:, 1], color=region_colors[lab], alpha=0.4)
            ax.text(
                np.mean(vertices[:, 0]),
                np.mean(vertices[:, 1]),
                lab,
                **textstyle,
            )

    ax.legend(handles=handles, loc="upper right")

    if simple_mode:
        ax.axis("off")
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    if formula:
        # Convert LTL formula to LaTeX format compatible with matplotlib mathtext
        # Replace operators with mathtext-compatible symbols
        latex_formula = formula
        # Replace in order to avoid conflicts
        latex_formula = latex_formula.replace("&&", r"\wedge")
        latex_formula = latex_formula.replace("||", r"\vee")
        latex_formula = latex_formula.replace(
            "[]", r"\mathrm{G}"
        )  # Globally - add space after
        latex_formula = latex_formula.replace(
            "<>", r"\mathrm{F}"
        )  # Finally - add space after
        latex_formula = latex_formula.replace("true", r"\top")
        latex_formula = latex_formula.replace("false", r"\bot")
        latex_formula = latex_formula.replace("NOT", r"\neg")
        latex_formula = latex_formula.replace("->", r"\rightarrow")
        latex_formula = latex_formula.replace(
            " U ", r"\mathrm{U}"
        )  # Only replace standalone U

        # Clean up multiple spaces
        latex_formula = re.sub(r"\s+", " ", latex_formula).strip()
        # Wrap in math mode for LaTeX rendering
        simplified_formula = f"${latex_formula}$"

    if show_title:
        ax.set_title(
            (
                f"Execution Trajectory {more_info}\n{simplified_formula}"
                if formula
                else "Execution Trajectory {more_info}"
            ),
            pad=10,
        )


def get_visualization_frame(
    env,
    prefix_obs,
    regions,
    start_state,
    suffix_obs=None,
    color_gradient=True,
    formula=None,
    more_info="",
    simple_mode=False,
    figsize=(4, 4),
    dpi=100,
    pad=1.0,
):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_trajectory(
        fig,
        ax,
        env,
        prefix_obs,
        regions,
        start_state,
        suffix_obs,
        color_gradient,
        formula,
        more_info,
        simple_mode,
        show_title=False,
    )
    fig.tight_layout(pad=pad)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image = np.asarray(buf)
    plt.close(fig)
    return image


def generate_visualization(
    env,
    prefix_obs,
    regions,
    start_state,
    imgpath,
    suffix_obs=None,
    color_gradient=True,
    formula=None,
    more_info="",
    simple_mode=False,
    show_title=True,
):
    """Generate visualization for a specific mode result."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    plot_trajectory(
        fig,
        ax,
        env,
        prefix_obs,
        regions,
        start_state,
        suffix_obs,
        color_gradient,
        formula,
        more_info,
        simple_mode,
        show_title,
    )

    # plt.show()
    plt.savefig(imgpath, bbox_inches="tight")
    # print(f"Trajectory plot saved to {imgpath}")
    plt.close(fig)


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def reset_env_simple(env, seed):
    """
    Minimal reset helper for LTL eval.
    For D4RL/Gymnasium envs: returns (obs, info).
    """
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        # extremely rare legacy fallback
        obs, info = out, {}
    return obs, info


def setup_task_env(env, env_name, task_id, should_render, seed):
    """Reset the environment for a specific task."""
    if env_name in [
        "kitchen-partial-v0",
    ]:
        env, observation, goal = _kitchen_set_obs_and_goal(
            env, env_name, task_id, seed=seed
        )
        goal_rendered = None
    else:
        observation, info = env.reset(
            seed=seed, options=dict(task_id=task_id, render_goal=should_render)
        )
        goal = info.get("goal")
        if should_render:
            goal_rendered = info.get("goal_rendered")
            goal_rendered = resize_frame(goal_rendered)
        else:
            goal_rendered = None
    reward = 0
    done = False
    return env, observation, goal, reward, done, goal_rendered


def env_step(env, env_name, action):
    """Step the environment once."""
    if env_name in ["kitchen-partial-v0"]:
        next_observation, reward, done, info = env.step(action)
        next_observation = next_observation[:30]
    else:
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return next_observation, reward, done, info


def resize_frame(frame):
    """Resize a frame"""
    resized_frame = cv2.resize(frame, (200, 200), interpolation=cv2.INTER_LINEAR)
    return resized_frame


def get_frame(env, env_name):
    """Render a frame from the environment."""
    if env_name in [
        "kitchen-partial-v0",
    ]:
        frame = _kitchen_render(env, wh=200)
    else:
        frame = np.ascontiguousarray(env.render())
        # frame = resize_frame(frame)
    return frame


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)
