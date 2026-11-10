# -*- coding: utf-8 -*-

import json
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np

from L_utils.ltl_utils import CircleRegion, PolyRegion, RectRegion, convert_formula


def read_formula_file(formula_path: str) -> List[str]:
    """Read formulas from txt file, skipping empty lines and comments."""
    formulas: List[str] = []
    with open(formula_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            formulas.append(line)
    return formulas


def extract_p_label_indices(formula_p: str) -> List[int]:
    """Extract sorted unique p-label indices (e.g., p0,p2 -> [0,2])."""
    ids = {int(x) for x in re.findall(r"\bp(\d+)\b", formula_p)}
    return sorted(ids)


def formula_p_to_e(formula_p: str) -> str:
    """Convert p-form LTL to e-form used by planner (p0 -> e0)."""
    return convert_formula(formula_p)


def e_labels_from_indices(indices: List[int]) -> List[str]:
    return [f"e{i}" for i in indices]


def region_to_dict(region) -> Dict:
    """Serialize a Region object to JSON-friendly dict."""
    color = tuple(float(c) for c in getattr(region, "color", (1.0, 1.0, 1.0, 1.0)))
    if getattr(region, "type", None) == "circle":
        return {
            "type": "circle",
            "center": [float(region.center[0]), float(region.center[1])],
            "radius": float(region.radius),
            "color": list(color),
        }
    if getattr(region, "type", None) == "rect":
        return {
            "type": "rect",
            "xmin": float(region.xmin),
            "xmax": float(region.xmax),
            "ymin": float(region.ymin),
            "ymax": float(region.ymax),
            "color": list(color),
        }
    if getattr(region, "type", None) == "polygon":
        return {
            "type": "polygon",
            "vertices": [[float(x), float(y)] for x, y in region.vertices],
            "color": list(color),
        }
    raise ValueError(f"Unsupported region type: {getattr(region, 'type', None)}")


def region_from_dict(data: Dict):
    """Deserialize region dict to Region object."""
    rtype = data["type"]
    if rtype == "circle":
        region = CircleRegion(
            center=(float(data["center"][0]), float(data["center"][1])),
            radius=float(data["radius"]),
        )
    elif rtype == "rect":
        region = RectRegion(
            xmin=float(data["xmin"]),
            xmax=float(data["xmax"]),
            ymin=float(data["ymin"]),
            ymax=float(data["ymax"]),
        )
    elif rtype == "polygon":
        region = PolyRegion(vertices=[(float(x), float(y)) for x, y in data["vertices"]])
    else:
        raise ValueError(f"Unsupported region type in task: {rtype}")

    color = data.get("color")
    if color is not None and len(color) == 4:
        region.color = tuple(float(c) for c in color)
    return region


def serialize_regions(regions: Dict[str, object]) -> Dict[str, Dict]:
    return {lab: region_to_dict(reg) for lab, reg in regions.items()}


def deserialize_regions(regions_dict: Dict[str, Dict]) -> Dict[str, object]:
    return {lab: region_from_dict(reg_d) for lab, reg_d in regions_dict.items()}


def sample_free_xy(env, regions: Dict[str, object], rng: random.Random, max_tries: int = 2000) -> Tuple[float, float]:
    """Sample a free-space xy, preferring points outside all labeled regions."""
    maze_map = env.get_maze_map()
    h, w = maze_map.shape
    for _ in range(max_tries):
        i = rng.randrange(h)
        j = rng.randrange(w)
        if maze_map[i, j] != 0:
            continue
        x, y = env.ij_to_xy(i, j)
        x += rng.uniform(-0.8, 0.8)
        y += rng.uniform(-0.8, 0.8)
        inside = False
        for reg in regions.values():
            if reg.contains((x, y)):
                inside = True
                break
        if not inside:
            return float(x), float(y)

    # Fallback: sample any free cell center.
    for _ in range(max_tries):
        i = rng.randrange(h)
        j = rng.randrange(w)
        if maze_map[i, j] == 0:
            x, y = env.ij_to_xy(i, j)
            return float(x), float(y)

    raise RuntimeError("Failed to sample free xy from maze.")


def sample_free_observation_from_dataset(
    observations,
    regions: Dict[str, object],
    pos_extractor,
    rng: random.Random,
    max_tries: int = 20000,
):
    """Sample a dataset observation whose xy position is outside labeled regions."""
    observations = np.asarray(observations)
    if len(observations) == 0:
        raise RuntimeError("Cannot sample a start observation from an empty dataset.")

    for _ in range(max_tries):
        idx = rng.randrange(len(observations))
        obs = np.asarray(observations[idx])
        xy = pos_extractor(obs)
        inside = False
        for reg in regions.values():
            if reg.contains(xy):
                inside = True
                break
        if not inside:
            return obs.copy(), int(idx), (float(xy[0]), float(xy[1]))

    raise RuntimeError(
        "Failed to sample a dataset start observation outside labeled regions."
    )


def save_tasks_json(payload: Dict, save_path: str) -> None:
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_tasks_json(task_path: str) -> Dict:
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_task(task_payload: Dict, task_id: int) -> Dict:
    tasks = task_payload.get("tasks", [])
    if not tasks:
        raise ValueError("Task file has no tasks.")
    if task_id < 0 or task_id >= len(tasks):
        raise IndexError(f"task_id out of range: {task_id} not in [0, {len(tasks)-1}]")
    return tasks[task_id]
