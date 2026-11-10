# -*- coding: utf-8 -*-
"""
Generate offline LTL tasks and save to JSON.

Each task contains:
- formula in p-form and e-form
- labels and sampled non-overlapping regions
- sampled start observation plus start/goal xy
"""

import random

import numpy as np
from absl import app, flags

from O_utils.env_utils import make_ltl_env_and_datasets
from L_utils.ltl_utils import pick_pos_extractor
from L_utils.ltl_utils import generate_regions
from L_utils.ltl_task_utils import (
    e_labels_from_indices,
    extract_p_label_indices,
    formula_p_to_e,
    read_formula_file,
    sample_free_observation_from_dataset,
    sample_free_xy,
    save_tasks_json,
    serialize_regions,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "antmaze-giant-stitch-v0", "Environment name.")
flags.DEFINE_integer("seed", 0, "Base random seed.")
flags.DEFINE_string("formula_path", None, "Path to formula txt file.")
flags.DEFINE_string("save_path", None, "Output task json path.")
flags.DEFINE_string(
    "complexity", "easy", 'Task complexity: "easy" / "medium" / "hard".'
)

flags.DEFINE_integer("num_tasks", 10, "Number of tasks to generate.")
flags.DEFINE_string(
    "formula_sample_mode",
    "random",
    'Formula sampling mode: "random" or "sequential".',
)
flags.DEFINE_integer("formula_start_idx", 0, "Start index for sequential sampling.")
flags.DEFINE_string(
    "region_kind", "circle", 'Region type: "circle" / "rect" / "polygon".'
)
flags.DEFINE_integer("render_size", 800, "Render size used to build environment.")
flags.DEFINE_integer(
    "ltl_max_region_num", 30, "Max region number passed to environment init."
)
flags.DEFINE_float("region_alpha", 0.6, "RGBA alpha for visualization regions.")
flags.DEFINE_float(
    "region_size_scale",
    1.0,
    "Scale factor for randomly generated LTL proposition regions.",
)
flags.DEFINE_float(
    "region_center_jitter",
    1.0,
    "Uniform xy jitter around sampled free-cell centers for LTL regions.",
)
flags.DEFINE_float(
    "region_radius_min",
    0.5,
    "Minimum random LTL region radius/size before applying region_size_scale.",
)
flags.DEFINE_float(
    "region_radius_max",
    2.0,
    "Maximum random LTL region radius/size before applying region_size_scale.",
)


def _pick_formula(formulas, task_idx, rng):
    if FLAGS.formula_sample_mode == "sequential":
        pos = FLAGS.formula_start_idx + task_idx
        if pos >= len(formulas):
            raise IndexError(
                f"formula_start_idx + num_tasks exceeds file size: {pos} >= {len(formulas)}"
            )
        return pos, formulas[pos]

    pos = rng.randrange(len(formulas))
    return pos, formulas[pos]


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    if not FLAGS.formula_path:
        FLAGS.formula_path = f"L_utils/formula/{FLAGS.complexity}.txt"
    if not FLAGS.save_path:
        env_key = "-".join(FLAGS.env_name.split("-")[:-1])
        FLAGS.save_path = f"ltl_tasks/{env_key}/{FLAGS.complexity}.json"

    formulas = read_formula_file(FLAGS.formula_path)
    if not formulas:
        raise ValueError(f"No formulas found in {FLAGS.formula_path}")

    env, train_dataset, _ = make_ltl_env_and_datasets(
        FLAGS.env_name,
        FLAGS.seed,
        size=FLAGS.render_size,
        region_type=FLAGS.region_kind,
        max_regions=FLAGS.ltl_max_region_num,
    )
    pos_extractor = pick_pos_extractor(FLAGS.env_name)
    dataset_observations = np.asarray(train_dataset["observations"])

    rng = random.Random(FLAGS.seed)
    tasks = []

    for task_id in range(FLAGS.num_tasks):
        cur_seed = FLAGS.seed + task_id
        random.seed(cur_seed)
        np.random.seed(cur_seed)

        formula_idx, formula_p = _pick_formula(formulas, task_id, rng)
        label_indices = extract_p_label_indices(formula_p)
        if not label_indices:
            # Skip formulas that do not use p{i} symbols.
            continue

        labels_e = e_labels_from_indices(label_indices)
        formula_e = formula_p_to_e(formula_p)

        regions = generate_regions(
            env,
            labels_e,
            kind=FLAGS.region_kind,
            alpha=FLAGS.region_alpha,
            size_scale=FLAGS.region_size_scale,
            center_jitter=FLAGS.region_center_jitter,
            radius_min=FLAGS.region_radius_min,
            radius_max=FLAGS.region_radius_max,
        )

        try:
            start_observation, start_dataset_index, start_xy = (
                sample_free_observation_from_dataset(
                    dataset_observations, regions, pos_extractor, rng
                )
            )
        except RuntimeError:
            start_observation = None
            start_dataset_index = None
            start_xy = sample_free_xy(env, regions, rng)
        goal_xy = sample_free_xy(env, regions, rng)

        task = {
            "task_id": len(tasks),
            "seed": cur_seed,
            "formula_index": int(formula_idx),
            "formula_p": formula_p,
            "formula_e": formula_e,
            "labels": labels_e,
            "region_kind": FLAGS.region_kind,
            "regions": serialize_regions(regions),
            "start_xy": [float(start_xy[0]), float(start_xy[1])],
            "start_observation": (
                [float(x) for x in np.asarray(start_observation).reshape(-1)]
                if start_observation is not None
                else None
            ),
            "start_dataset_index": start_dataset_index,
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        }
        tasks.append(task)

    payload = {
        "env_name": FLAGS.env_name,
        "formula_path": FLAGS.formula_path,
        "complexity": FLAGS.complexity,
        "region_kind": FLAGS.region_kind,
        "region_size_scale": float(FLAGS.region_size_scale),
        "region_center_jitter": float(FLAGS.region_center_jitter),
        "region_radius_min": float(FLAGS.region_radius_min),
        "region_radius_max": float(FLAGS.region_radius_max),
        "seed": FLAGS.seed,
        "num_tasks": len(tasks),
        "tasks": tasks,
    }

    save_tasks_json(payload, FLAGS.save_path)
    print(f"[TASK GEN] Saved {len(tasks)} tasks to {FLAGS.save_path}")


if __name__ == "__main__":
    app.run(main)
