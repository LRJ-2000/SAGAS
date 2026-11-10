# test_ltl.py
# -*- coding: utf-8 -*-
""" """

import os
import sys
import platform
import imageio
import pandas as pd
from copy import deepcopy
from rich import print as rprint
from rich.table import Table
from rich.console import Console

# ---------- GPU/Render setup (same as original) ----------
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpu_index = sys.argv[sys.argv.index("--gpu") + 1] if "--gpu" in sys.argv else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
print(f"\033[38;5;208m{'=' * 14}\n Using GPU: {gpu_index}\n{'=' * 14}\033[0m")

if "mac" in platform.platform():
    pass
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]

# ---------- Std imports ----------
import random
import numpy as np
from absl import app, flags
from ml_collections import config_flags

# ---------- Project imports ----------
from O_utils.datasets import Dataset, GCDataset
from O_utils.env_utils import make_env_and_datasets, make_ltl_env_and_datasets
from O_utils.log_utils import (
    get_exp_name,
    setup_save_directory,
)
from O_utils.evaluation import (
    evaluate_with_ltl,
    evaluate_with_ltl_full,
)
from K_utils.keygraph_utils import KeyGraph
from M_utils.agents import agents_dict
from M_utils.flax_utils import restore_agent
from L_utils.buchi_parse import Buchi
from L_utils.ltl_runtime import BuchiRuntime
import sys
import os
from L_utils.ltl_utils import (
    build_label_literal_maps,
    generate_regions,
    make_label_fn,
    pick_pos_extractor,
    retrieve_anchor_raw_states_from_dataset,
    sample_anchor_raw_states,
    set_pos_xy,
    # verify_with_buchi_runtime,
    # verify_with_buchi_runtime_full,
)
from L_utils.ltl_task_utils import deserialize_regions, load_tasks_json

# ---------- Flags ----------
FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "LTL", "Run group.")
flags.DEFINE_string("env_name", "antmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("gpu", 0, "GPU index")
flags.DEFINE_string("save_eval_dir", "exp_LTL/", "Save directory.")

flags.DEFINE_integer("eval_on_cpu", 1, "Whether to evaluate on CPU.")
flags.DEFINE_integer(
    "eval_final_goal_threshold", 2, "Threshold to switch to final goal."
)

flags.DEFINE_string(
    "keygraph_path",
    "exp_graph/EXP_antmaze-medium-navigate/antmaze-medium-navigate-v0_sd000__2025-11-07/keygraph.pkl",
    "Path to TD-aware graph.",
)
flags.DEFINE_string(
    "policy_path",
    "exp_policy/EXP_antmaze-medium-navigate/antmaze-medium-navigate-v0_sd000__2025-11-07/params_1000000.pkl",
    "Low-level policy path.",
)

# LTL-specific
flags.DEFINE_string("ltl_plan_type", "full", 'Plan type: "prefix" or "full".')
flags.DEFINE_integer("ltl_max_region_num", 20, "Potential max number of regions.")
flags.DEFINE_string(
    "ltl_task_path",
    None,
    help="Path to generated task json. This script evaluates all tasks from this file.",
    required=True,
)
flags.DEFINE_boolean(
    "ltl_use_task_seed",
    True,
    "Whether to use per-task seed stored in task json.",
)
flags.DEFINE_string(
    "ltl_region_kind", "circle", 'Region type: "circle" or "rect" or "polygon".'
)
flags.DEFINE_integer(
    "ltl_anchor_per_label", 3, "Number of anchors to sample per label."
)
flags.DEFINE_integer(
    "ltl_anchor_sample_attempts",
    50,
    "Maximum candidate samples tried per label before declaring it unavailable.",
)
flags.DEFINE_enum(
    "ltl_anchor_source",
    "dataset_retrieval",
    ["region_sample", "dataset_retrieval"],
    "Anchor candidate source. region_sample writes sampled xy into a template "
    "observation; dataset_retrieval uses offline observations inside each region.",
)
flags.DEFINE_integer(
    "ltl_anchor_dataset_max_scan",
    -1,
    "Maximum number of offline observations to scan when --ltl_anchor_source="
    "dataset_retrieval; -1 scans the full dataset.",
)
flags.DEFINE_float(
    "ltl_avoid_soft_thresh", 0.2, "Threshold for soft-label avoidance in planner."
)
flags.DEFINE_integer("max_steps", 8000, "Maximum steps per segment (prefix/suffix).")
flags.DEFINE_float("action_clip", 1.0, "Action clipping value.")
flags.DEFINE_float(
    "direction_smoothing_beta",
    0.0,
    "Low-level skill direction smoothing factor. 0 disables smoothing; larger values keep more of the previous direction.",
)
flags.DEFINE_boolean(
    "semantic_repulsion_enable",
    False,
    "Whether to bias low-level latent directions away from anchors of labels "
    "forbidden by the current product transition. Disabled by default.",
)
flags.DEFINE_float(
    "semantic_repulsion_radius",
    -1.0,
    "Latent radius for semantic repulsion. Non-positive values reuse "
    "--eval_subgoal_threshold.",
)
flags.DEFINE_float(
    "semantic_repulsion_gain",
    0.3,
    "Gain for forbidden-anchor repulsion when semantic repulsion is enabled.",
)
flags.DEFINE_float(
    "semantic_repulsion_power",
    1.0,
    "Distance falloff power for semantic repulsion.",
)
flags.DEFINE_float(
    "semantic_repulsion_max_norm",
    0.5,
    "Maximum norm of the added semantic repulsion vector.",
)
flags.DEFINE_float("eval_subgoal_threshold", 8, "Subgoal radius in latent space.")
flags.DEFINE_float(
    "eval_anchor_reached_threshold",
    -1.0,
    "Latent radius for declaring an anchor reached. Non-positive values reuse "
    "--eval_subgoal_threshold.",
)
flags.DEFINE_integer(
    "ltl_suffix_cycles",
    2,
    "Number of suffix cycle traversals for finite lasso evaluation.",
)
flags.DEFINE_integer(
    "max_prefix_candidates", 1, "Number of accepting prefixes to consider."
)
flags.DEFINE_float("lambda_weight", 1.0, "Suffix loop cost weight.")
flags.DEFINE_boolean(
    "vis_simple_mode", False, "Whether to use simple mode for visualization."
)
flags.DEFINE_integer("render_size", 800, "Render size for environment observations.")

config_flags.DEFINE_config_file(
    "agent_config", "M_utils/agents/gas.py", lock_config=False
)


def log_current_eval_results(results, total_tasks=None):
    success_rate = np.mean([res["success_rate"] for res in results])
    exec_times = [
        np.nanmean(res["total_times"])
        for res in results
        if np.any(~np.isnan(res["total_times"]))
    ]
    total_steps = [
        np.nanmean(res["total_steps"])
        for res in results
        if np.any(~np.isnan(res["total_steps"]))
    ]

    avg_exec_time = np.nanmean(exec_times) if exec_times else float("nan")
    avg_total_steps = np.nanmean(total_steps) if total_steps else float("nan")

    # aggregate failure reasons
    failure_reasons = {}
    for res in results:
        eps = len(res["failure_reasons"])
        for reason in res["failure_reasons"]:
            if reason is None:
                continue
            if reason not in failure_reasons:
                failure_reasons[reason] = 0
            failure_reasons[reason] += 1 / eps
    failure_reasons_table = Table(title="Failure Reasons", title_style="cyan")
    failure_reasons_table.add_column(
        "Reason", justify="left", style="cyan", no_wrap=True
    )
    failure_reasons_table.add_column("Value", justify="right", style="cyan")
    sorted_reasons = dict(
        sorted(failure_reasons.items(), key=lambda item: item[1], reverse=True)
    )
    for k, v in sorted_reasons.items():
        failure_reasons_table.add_row(k, f"{v:.2f}")

    total_tasks = len(results) if total_tasks is None else int(total_tasks)
    rprint(f"[cyan]  Success Rate   : {success_rate:.4f}[/cyan]")
    rprint(f"[cyan]  Execution Time : {avg_exec_time:.4f}[/cyan]")
    rprint(f"[cyan]  Total Steps    : {avg_total_steps:.4f}\n[/cyan]")
    Console().print(failure_reasons_table)


def run_eval_on_one_formula(
    env,
    labels,
    agent,
    key_graph,
    config,
    train_dataset,
    get_phi_fn,
    seed,
    region_kind,
    ltl_formula,
    save_eval_dir,
    regions=None,
    start_xy=None,
    start_observation=None,
):
    if regions is None:
        # Deterministic region generation given seed
        rng_state = np.random.get_state()
        np.random.seed(seed)
        regions = generate_regions(env, labels, kind=region_kind)
        # Restore RNG state
        np.random.set_state(rng_state)

    # Add regions to env for visualization
    if hasattr(env.unwrapped, "clear_regions"):
        env.unwrapped.clear_regions()
    for region in regions.values():
        if region.type == "circle":
            region_info = {
                "shape": "circle",
                "center": region.center,
                "radius": region.radius,
                "color": region.color,
            }
            env.unwrapped.add_region(region_info)
        elif region.type == "rect":
            x_min, x_max, y_min, y_max = (
                region.xmin,
                region.xmax,
                region.ymin,
                region.ymax,
            )
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            width_x = x_max - x_min
            width_y = y_max - y_min
            region_info = {
                "shape": "rect",
                "center": (center_x, center_y),
                "size": (width_x, width_y),
                "color": region.color,
            }
            env.unwrapped.add_region(region_info)
        else:
            # Keep compatibility if renderer does not support polygons directly.
            continue

    # Copy keygraph and agent to avoid modifying original
    key_graph = deepcopy(key_graph)
    agent = deepcopy(agent)

    pos_extractor = pick_pos_extractor(FLAGS.env_name)
    label_fn = make_label_fn(regions, pos_extractor)

    # ---------- Anchors: sample raw states in each region and attach to KeyGraph ----------
    anchor_sample_attempts = max(
        int(FLAGS.ltl_anchor_sample_attempts), int(FLAGS.ltl_anchor_per_label)
    )
    if FLAGS.ltl_anchor_source == "region_sample":
        # Use a template observation taken from the env reset to fill (x,y).
        obs0, _ = env.reset(seed=seed)
        label_to_states = sample_anchor_raw_states(
            regions, obs0, per_label=anchor_sample_attempts
        )
    elif FLAGS.ltl_anchor_source == "dataset_retrieval":
        if train_dataset is None or "observations" not in train_dataset:
            raise ValueError(
                "--ltl_anchor_source=dataset_retrieval requires train_dataset "
                "with an observations array."
            )
        label_to_states = retrieve_anchor_raw_states_from_dataset(
            regions,
            train_dataset["observations"],
            pos_extractor,
            per_label=anchor_sample_attempts,
            rng=np.random.default_rng(seed),
            max_scan=int(FLAGS.ltl_anchor_dataset_max_scan),
        )
    else:
        raise ValueError(f"Unknown ltl_anchor_source: {FLAGS.ltl_anchor_source}")
    anchor_counts = {lab: len(states) for lab, states in label_to_states.items()}
    rprint(
        f"[cyan]Anchor source: {FLAGS.ltl_anchor_source}; "
        f"candidate counts: {anchor_counts}[/cyan]"
    )

    # ---------- Buchi automaton/runtime for planning ----------
    # Map labels -> literals and align e{i} placeholders with label indices (handles gaps like e4)
    label_to_literal, subformula = build_label_literal_maps(labels)
    # print(subformula)

    class _Task:
        def __init__(self, formula, subformula):
            self.formula = formula
            self.subformula = subformula
            self.number_of_robots = 1

    buchi = Buchi(_Task(ltl_formula, subformula))
    buchi.construct_buchi_graph()
    # print(buchi.buchi_graph)
    buchi_runtime = BuchiRuntime(buchi, label_to_literal)

    start_state_getter = None
    if start_observation is not None:
        start_observation = np.asarray(start_observation, dtype=np.float32)
        if start_xy is None:
            start_xy = (float(start_observation[0]), float(start_observation[1]))

    if start_xy is not None:
        start_xy = (float(start_xy[0]), float(start_xy[1]))

        def start_state_getter(env, observation):
            # Best-effort: update simulator state if the backend exposes set_xy.
            if hasattr(env.unwrapped, "set_xy"):
                try:
                    env.unwrapped.set_xy(start_xy)
                except TypeError:
                    env.unwrapped.set_xy(start_xy[0], start_xy[1])
            if start_observation is not None:
                start_raw = start_observation.copy()
                start_raw[0], start_raw[1] = start_xy
                return start_raw
            return set_pos_xy(observation, start_xy)

    # ---------- Evaluate with offline LTL verification ----------
    vis_path = os.path.join(save_eval_dir, "visualizations")

    key_graph.assign_soft_labels(label_fn, 0.01)
    key_graph.add_label_anchors(
        label_to_states,
        get_phi=get_phi_fn,
        max_anchors_per_label=FLAGS.ltl_anchor_per_label,
        require_neighbor_within_cutoff=True,
    )
    key_graph.precompute_dist_to_label()
    anchor_prune_stats = buchi_runtime.prune_unavailable_positive_labels(
        getattr(key_graph, "anchor_unavailable_labels", [])
    )
    if anchor_prune_stats.get("removed_edges", 0) > 0:
        rprint(
            "[yellow]Pruned "
            f"{anchor_prune_stats['removed_edges']} automaton transitions requiring "
            f"unavailable labels {anchor_prune_stats['unavailable_labels']}[/yellow]"
        )

    if FLAGS.ltl_plan_type == "prefix":
        eval_info, cur_renders = evaluate_with_ltl(
            agent=agent,
            key_graph=key_graph,
            env=env,  # base env: unchanged API
            env_name=FLAGS.env_name,
            buchi_runtime=buchi_runtime,  # for planner
            start_state_getter=start_state_getter,
            get_phi_override=None,
            seed=seed,
            eval_on_cpu=bool(FLAGS.eval_on_cpu),
            eval_subgoal_threshold=float(FLAGS.eval_subgoal_threshold),
            eval_final_goal_threshold=int(FLAGS.eval_final_goal_threshold),
            avoid_soft_label_threshold=float(FLAGS.ltl_avoid_soft_thresh),
            # offline validation
            ltl_labels=labels,
            ltl_formula_eform=ltl_formula,
            label_fn=label_fn,
            ltl_label_to_literal=label_to_literal,
            config=config,
            regions=regions,
            imgpath=vis_path,
            action_clip=FLAGS.action_clip,
            direction_smoothing_beta=FLAGS.direction_smoothing_beta,
        )
    elif FLAGS.ltl_plan_type == "full":
        eval_info = evaluate_with_ltl_full(
            agent=agent,
            key_graph=key_graph,
            env=env,  # base env: unchanged API
            env_name=FLAGS.env_name,
            buchi_runtime=buchi_runtime,  # for planner
            start_state_getter=start_state_getter,
            get_phi_override=None,
            seed=seed,
            eval_on_cpu=bool(FLAGS.eval_on_cpu),
            eval_subgoal_threshold=float(FLAGS.eval_subgoal_threshold),
            eval_final_goal_threshold=int(FLAGS.eval_final_goal_threshold),
            avoid_soft_label_threshold=float(FLAGS.ltl_avoid_soft_thresh),
            # offline validation
            ltl_labels=labels,
            ltl_formula_eform=ltl_formula,
            label_fn=label_fn,
            ltl_label_to_literal=label_to_literal,
            config=config,
            regions=regions,
            imgpath=vis_path,
            logpath=save_eval_dir,
            max_prefix_candidates=FLAGS.max_prefix_candidates,
            lambda_weight=FLAGS.lambda_weight,
            max_steps=FLAGS.max_steps,
            action_clip=FLAGS.action_clip,
            suffix_cycles=FLAGS.ltl_suffix_cycles,
            direction_smoothing_beta=FLAGS.direction_smoothing_beta,
            semantic_repulsion_enable=FLAGS.semantic_repulsion_enable,
            semantic_repulsion_radius=FLAGS.semantic_repulsion_radius,
            semantic_repulsion_gain=FLAGS.semantic_repulsion_gain,
            semantic_repulsion_power=FLAGS.semantic_repulsion_power,
            semantic_repulsion_max_norm=FLAGS.semantic_repulsion_max_norm,
            anchor_reached_threshold=FLAGS.eval_anchor_reached_threshold,
            vis_simple_mode=FLAGS.vis_simple_mode,
        )

    try:
        env.unwrapped.custom_renderer.close()
    except Exception:
        pass

    # rprint(f"[green][LTL EVAL] Metrics: {eval_info}[/green]")
    return eval_info


def main(_):
    # ---------- Seeds & config ----------
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    config = FLAGS.agent_config

    # ---------- Logging ----------
    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_eval_dir = setup_save_directory(
        exp_name, FLAGS.env_name, FLAGS.run_group, FLAGS.save_eval_dir
    )

    # ---------- Env & datasets (unchanged) ----------
    if FLAGS.env_name in [
        "kitchen-partial-v0",
    ]:
        from D_utils.d4rl_env_utils import d4rl_make_env_and_dataset

        env, train_dataset = d4rl_make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
        val_dataset = None
    else:
        # env, train_dataset, val_dataset = make_env_and_datasets(
        #     FLAGS.env_name, FLAGS.seed
        # )
        env, train_dataset, val_dataset = make_ltl_env_and_datasets(
            FLAGS.env_name,
            FLAGS.seed,
            size=FLAGS.render_size,
            region_type=FLAGS.ltl_region_kind,
            max_regions=FLAGS.ltl_max_region_num,
        )
    train_gc_dataset = GCDataset(Dataset.create(**train_dataset), config)

    # ---------- Agent ----------
    example_batch = train_gc_dataset.sample(1)
    agent_class = agents_dict[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )

    # Restore low-level policy
    policy_restore_path = os.path.dirname(FLAGS.policy_path)
    policy_restore_epoch = (
        os.path.basename(FLAGS.policy_path).split("_")[-1].split(".")[0]
    )
    agent = restore_agent(agent, policy_restore_path, policy_restore_epoch)
    get_phi_fn = agent.get_phi

    # ---------- Load KeyGraph ----------
    key_graph = KeyGraph()
    keygraph_load_path = os.path.dirname(FLAGS.keygraph_path)
    keygraph_load_filename = (
        os.path.basename(FLAGS.keygraph_path).split("_")[-1].split(".")[0]
    )
    key_graph.load_keygraph(keygraph_load_path, keygraph_load_filename)

    # Backward-compat: ensure new attrs exist even if old pickle didn't have them
    for name, default in [
        ("label_node_idx_dict", {}),
        ("node_soft_labels", {}),
        ("dist_to_label", {}),
    ]:
        if not hasattr(key_graph, name):
            setattr(key_graph, name, default)

    assert FLAGS.ltl_task_path, "Please provide --ltl_task_path to evaluate tasks."
    task_payload = load_tasks_json(FLAGS.ltl_task_path)
    tasks = task_payload.get("tasks", [])
    assert len(tasks) > 0, f"No tasks found in {FLAGS.ltl_task_path}"

    results = []
    total_tasks = len(tasks)

    for task_idx, task in enumerate(tasks):
        ltl_formula = str(task["formula_e"])
        labels = [s.strip() for s in task["labels"] if str(s).strip() != ""]
        regions = deserialize_regions(task["regions"])
        start_observation = task.get("start_observation", None)
        start_xy = task.get("start_xy", None)
        if start_xy is None and start_observation is not None:
            start_xy = [start_observation[0], start_observation[1]]

        eval_seed = (
            int(task.get("seed", FLAGS.seed + task_idx))
            if FLAGS.ltl_use_task_seed
            else FLAGS.seed + task_idx
        )

        formula_show = str(task.get("formula_p", ltl_formula))
        rprint(
            f"\n[blue]=== {FLAGS.env_name:^30} [ {task_idx+1:>3} / {total_tasks:<3} ] ===[/blue]"
        )
        rprint(f"[blue]  Task {task.get('task_id', task_idx)}: {formula_show}[/blue]")

        status = run_eval_on_one_formula(
            env=env,
            labels=labels,
            agent=agent,
            key_graph=key_graph,
            config=config,
            train_dataset=train_dataset,
            get_phi_fn=get_phi_fn,
            seed=eval_seed,
            region_kind=str(task.get("region_kind", FLAGS.ltl_region_kind)),
            ltl_formula=ltl_formula,
            save_eval_dir=FLAGS.save_eval_dir,
            regions=regions,
            start_xy=start_xy,
            start_observation=start_observation,
        )

        if status:
            results.append(status)

        if len(results) != 0 and task_idx + 1 < total_tasks:
            cur_success = np.mean([res["success_rate"] for res in results])
            rprint(
                f"[cyan]Progress: {task_idx+1}/{total_tasks} | success_rate={cur_success:.4f}[/cyan]"
            )

    print("\n=== Summary of LTL Evaluation Results ===")
    log_current_eval_results(results, total_tasks=total_tasks)

    # Save overall results to a CSV
    with open(os.path.join(FLAGS.save_eval_dir, "overall_ltl_results.csv"), "w") as f:
        df = pd.DataFrame(results)
        df.to_csv(f, index=False)


if __name__ == "__main__":
    app.run(main)
