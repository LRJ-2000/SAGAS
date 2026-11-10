# test_ltl.py
# -*- coding: utf-8 -*-
"""
Evaluate GAS with LTL planning + offline verification on existing trained model and prebuilt KeyGraph.

Pipeline:
1) Load trained agent (get_phi) and prebuilt KeyGraph (unchanged).
2) Define LTL spec (manual e-form) + labels; randomly generate non-overlapping regions.
3) Build label_fn from regions; sample per-label anchors -> add to KeyGraph; precompute distances.
4) Build Buchi automaton/runtime (for planning only).
5) Run episodes with Product-A* path and low-level policy; collect full observation trajectory.
6) Offline verify the label trace (Spot first; fallback to BuchiRuntime).
7) Log ltl_success_rate and optional rollout video to wandb/CSV.
"""

import os
import sys
import platform

# ---------- GPU/Render setup (same as original) ----------
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpu_index = sys.argv[sys.argv.index("--gpu") + 1] if "--gpu" in sys.argv else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
print(f"\033[38;5;208m{'=' * 14}\n Using GPU: {gpu_index}\n{'=' * 14}\033[0m")

if "mac" in platform.platform():
    pass
else:
    os.environ["MUJOCO_GL"] = "egl"
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
)  # <- added earlier
from K_utils.keygraph_utils import KeyGraph
from M_utils.agents import agents_dict
from M_utils.flax_utils import restore_agent

# ---------- LTL modules (added earlier) ----------
from L_utils.ltl_utils import (
    build_label_literal_maps,
    pick_pos_extractor,
    make_label_fn,
    sample_anchor_raw_states,
    generate_regions,
)
from L_utils.buchi_parse import Buchi
from L_utils.ltl_runtime import BuchiRuntime

# ---------- Flags ----------
FLAGS = flags.FLAGS

flags.DEFINE_string("run_eval_project", "Debug", "Run Evaluation Project.")
flags.DEFINE_string("run_group", "LTL", "Run group.")
flags.DEFINE_string("env_name", "antmaze-giant-stitch-v0", "Environment name.")
flags.DEFINE_integer("seed", 37, "Random seed.")
flags.DEFINE_integer("gpu", 0, "GPU index")
flags.DEFINE_string("save_eval_dir", "exp_LTL/", "Save directory.")

flags.DEFINE_integer("eval_on_cpu", 1, "Whether to evaluate on CPU.")
flags.DEFINE_integer("eval_episodes", 1, "Number of episodes.")
flags.DEFINE_integer("eval_video_episodes", 0, "Number of video episodes.")
flags.DEFINE_integer(
    "eval_final_goal_threshold", 2, "Threshold to switch to final goal."
)

flags.DEFINE_string(
    "keygraph_path",
    "checkpoints/antmaze-giant-stitch/keygraph.pkl",
    "Path to TD-aware graph.",
)
flags.DEFINE_string(
    "policy_path",
    "checkpoints/antmaze-giant-stitch/params_1000000.pkl",
    "Low-level policy path.",
)

# LTL-specific
flags.DEFINE_string("ltl_plan_type", "full", 'Plan type: "prefix" or "full".')
flags.DEFINE_string(
    "ltl_formula",
    "<>(e0 || e4) && <>(e4 || e7 || e5) && <>(e3 && (!e1 U e6)) && <>(e4) && <>(e3) && <>(e0)",
    "LTL formula in e-form (use e1,e2,...)",
)
flags.DEFINE_string(
    "ltl_labels",
    "e0,e1,e2,e3,e4,e5,e6,e7",
    "Comma-separated label names, mapped to e1,e2,... in order.",
)
flags.DEFINE_string(
    "ltl_region_kind", "rect", 'Region type: "circle" or "rect" or "polygon".'
)
flags.DEFINE_integer(
    "ltl_anchor_per_label", 5, "Number of anchors to sample per label."
)
flags.DEFINE_float(
    "ltl_avoid_soft_thresh", 0.05, "Threshold for soft-label avoidance in planner."
)
flags.DEFINE_float("eval_subgoal_threshold", 8, "Subgoal radius in latent space.")
flags.DEFINE_integer(
    "max_prefix_candidates", 30, "Number of accepting prefixes to consider."
)
flags.DEFINE_float("lambda_weight", 1.0, "Suffix loop cost weight.")
flags.DEFINE_float("action_clip", 1.0, "Action clipping value.")
flags.DEFINE_integer("render_size", 800, "Render size.")
flags.DEFINE_integer("max_steps", 5000, "Maximum steps per episode.")

config_flags.DEFINE_config_file(
    "agent_config", "M_utils/agents/gas.py", lock_config=False
)


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
        env, train_dataset, val_dataset = make_ltl_env_and_datasets(
            FLAGS.env_name,
            FLAGS.seed,
            size=FLAGS.render_size,
            region_type=FLAGS.ltl_region_kind,
            max_regions=len(FLAGS.ltl_labels.split(",")),
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

    # ---------- LTL: labels, regions, label_fn ----------
    labels = [s.strip() for s in FLAGS.ltl_labels.split(",") if s.strip() != ""]
    assert len(labels) >= 1, "ltl_labels must contain at least one label, e.g., 'A,B'."

    # Deterministic region generation given seed
    rng_state = np.random.get_state()
    np.random.seed(FLAGS.seed)
    regions = generate_regions(env, labels, kind=FLAGS.ltl_region_kind, alpha=0.6)
    np.random.set_state(rng_state)

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
            pass

    pos_extractor = pick_pos_extractor(FLAGS.env_name)
    label_fn = make_label_fn(regions, pos_extractor)

    # ---------- Anchors: sample raw states in each region and attach to KeyGraph ----------
    # Use a template observation taken from the env reset to fill (x,y)
    obs0, _ = env.reset(seed=FLAGS.seed)
    label_to_states = sample_anchor_raw_states(
        regions, obs0, per_label=FLAGS.ltl_anchor_per_label
    )

    key_graph.assign_soft_labels(label_fn, 0.01)
    key_graph.add_label_anchors(label_to_states, get_phi=get_phi_fn)
    key_graph.precompute_dist_to_label()


    # ---------- Buchi automaton/runtime for planning ----------
    # Map labels -> literals and align e{i} placeholders with label indices (handles gaps like e4)
    label_to_literal, subformula = build_label_literal_maps(labels)

    class _Task:
        def __init__(self, formula, subformula):
            self.formula = formula
            self.subformula = subformula
            self.number_of_robots = 1

    buchi = Buchi(_Task(FLAGS.ltl_formula, subformula))
    buchi.construct_buchi_graph()
    buchi_runtime = BuchiRuntime(buchi, label_to_literal)

    # ---------- Evaluate with offline LTL verification ----------
    eval_metrics = {}
    vis_path = os.path.join(FLAGS.save_eval_dir, "visualizations")

    if FLAGS.ltl_plan_type == "prefix":
        eval_info, cur_renders = evaluate_with_ltl(
            agent=agent,
            key_graph=key_graph,
            env=env,  # base env: unchanged API
            env_name=FLAGS.env_name,
            buchi_runtime=buchi_runtime,  # for planner
            start_state_getter=None,
            get_phi_override=None,
            eval_episodes=FLAGS.eval_episodes,
            eval_video_episodes=FLAGS.eval_video_episodes,
            seed=FLAGS.seed,
            eval_on_cpu=bool(FLAGS.eval_on_cpu),
            eval_subgoal_threshold=float(FLAGS.eval_subgoal_threshold),
            eval_final_goal_threshold=int(FLAGS.eval_final_goal_threshold),
            avoid_soft_label_threshold=float(FLAGS.ltl_avoid_soft_thresh),
            # offline validation
            ltl_labels=labels,
            ltl_formula_eform=FLAGS.ltl_formula,
            label_fn=label_fn,
            ltl_label_to_literal=label_to_literal,
            config=config,
            regions=regions,
            imgpath=vis_path,
        )
    elif FLAGS.ltl_plan_type == "full":
        eval_info = evaluate_with_ltl_full(
            agent=agent,
            key_graph=key_graph,
            env=env,  # base env: unchanged API
            env_name=FLAGS.env_name,
            buchi_runtime=buchi_runtime,  # for planner
            start_state_getter=None,
            get_phi_override=None,
            eval_episodes=FLAGS.eval_episodes,
            eval_video_episodes=FLAGS.eval_video_episodes,
            seed=FLAGS.seed,
            eval_on_cpu=bool(FLAGS.eval_on_cpu),
            eval_subgoal_threshold=float(FLAGS.eval_subgoal_threshold),
            eval_final_goal_threshold=int(FLAGS.eval_final_goal_threshold),
            avoid_soft_label_threshold=float(FLAGS.ltl_avoid_soft_thresh),
            # offline validation
            ltl_labels=labels,
            ltl_formula_eform=FLAGS.ltl_formula,
            label_fn=label_fn,
            ltl_label_to_literal=label_to_literal,
            config=config,
            regions=regions,
            imgpath=vis_path,
            max_prefix_candidates=FLAGS.max_prefix_candidates,
            lambda_weight=FLAGS.lambda_weight,
            max_steps=FLAGS.max_steps,
            action_clip=FLAGS.action_clip,
        )

    if eval_info is None:
        print("[LTL EVAL] Formula not satisfiable, skipping...")
        return

    # Log metrics
    # We at least expect 'ltl_success_rate' from evaluate_with_ltl
    for k, v in eval_info.items():
        eval_metrics[f"eval/ltl_{k}"] = v

    # wandb.log(eval_metrics, step=0)
    # eval_logger.log(eval_metrics, step=0)
    # eval_logger.close()
    env.unwrapped.custom_renderer.close()

    print("[LTL EVAL] Metrics:", eval_metrics)


if __name__ == "__main__":
    app.run(main)
