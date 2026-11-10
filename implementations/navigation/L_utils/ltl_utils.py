# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Callable, Optional, Set
import numpy as np
import random
import re
from .buchi_parse import Buchi
import matplotlib.pyplot as plt

# ---------- Geometry helpers ----------


def _ray_casting_point_in_polygon(
    point: Tuple[float, float], poly: List[Tuple[float, float]]
) -> bool:
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        if cond:
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if xinters > x:
                inside = not inside
    return inside


def _sample_point_in_rect(
    xmin: float, xmax: float, ymin: float, ymax: float
) -> Tuple[float, float]:
    return (random.uniform(xmin, xmax), random.uniform(ymin, ymax))


def _sample_point_in_circle(
    center: Tuple[float, float], radius: float
) -> Tuple[float, float]:
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, radius)
    return (center[0] + r * np.cos(angle), center[1] + r * np.sin(angle))


def _bbox(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), max(xs), min(ys), max(ys))


def _regions_non_overlapping(regions) -> bool:
    """Conservative non-overlap via AABB test (fast and sufficient for random generation)."""
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if regions[i].intersects(regions[j]):
                return False
    return True


def _sample_convex_polygon(
    center: Tuple[float, float], radius_range=(0.6, 1.6), num_vertices=6
) -> List[Tuple[float, float]]:
    cx, cy = center
    angles = np.sort(np.random.uniform(0, 2 * np.pi, size=num_vertices))
    radii = np.random.uniform(radius_range[0], radius_range[1], size=num_vertices)
    return [
        (float(cx + r * np.cos(a)), float(cy + r * np.sin(a)))
        for a, r in zip(angles, radii)
    ]


# ---------- Region classes ----------


class Region:
    """Abstract region with containment test and sampling."""

    def contains(self, pos: Tuple[float, float]) -> bool:
        raise NotImplementedError

    def sample_point(self) -> Tuple[float, float]:
        raise NotImplementedError

    def intersects(self, other: "Region") -> bool:
        raise NotImplementedError


class RectRegion(Region):
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float):
        assert xmin < xmax and ymin < ymax
        self.type = "rect"
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.color = (1, 1, 1, 1)

    def contains(self, pos: Tuple[float, float]) -> bool:
        x, y = pos
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)

    def sample_point(self) -> Tuple[float, float]:
        return _sample_point_in_rect(self.xmin, self.xmax, self.ymin, self.ymax)

    def intersects(self, other: "Region") -> bool:
        if isinstance(other, RectRegion):
            return not (
                self.xmax < other.xmin
                or self.xmin > other.xmax
                or self.ymin > other.ymax
                or self.ymax < other.ymin
            )
        return False


class PolyRegion(Region):
    def __init__(self, vertices: List[Tuple[float, float]]):
        assert len(vertices) >= 3
        self.type = "polygon"
        self.vertices = vertices
        self.color = (1, 1, 1, 1)

    def contains(self, pos: Tuple[float, float]) -> bool:
        return _ray_casting_point_in_polygon(pos, self.vertices)

    def sample_point(self) -> Tuple[float, float]:
        xmin, xmax, ymin, ymax = _bbox(self.vertices)
        for _ in range(512):
            pt = _sample_point_in_rect(xmin, xmax, ymin, ymax)
            if self.contains(pt):
                return pt
        # fallback: centroid
        xs = [p[0] for p in self.vertices]
        ys = [p[1] for p in self.vertices]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def intersects(self, other: "Region") -> bool:
        if isinstance(other, RectRegion):
            return not (
                self.xmax < other.xmin
                or self.xmin > other.xmax
                or self.ymin > other.ymax
                or self.ymax < other.ymin
            )
        return False


class CircleRegion(Region):
    def __init__(self, center: Tuple[float, float], radius: float):
        self.type = "circle"
        self.center = center
        self.radius = radius
        self.color = (1, 1, 1, 1)

    def contains(self, pos: Tuple[float, float]) -> bool:
        x, y = pos
        return (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 <= self.radius**2

    def sample_point(self) -> Tuple[float, float]:
        return _sample_point_in_circle(self.center, self.radius)

    def intersects(self, other: "Region") -> bool:
        if isinstance(other, CircleRegion):
            return (self.center[0] - other.center[0]) ** 2 + (
                self.center[1] - other.center[1]
            ) ** 2 <= (self.radius + other.radius) ** 2
        return False


# ---------- Random non-overlapping generator ----------


def generate_regions(
    env,
    labels: List[str],
    kind: str = "circle",
    max_tries: int = 200,
    alpha: float = 0.6,
    size_scale: float = 1.0,
    center_jitter: float = 1.0,
    radius_min: float = 0.5,
    radius_max: float = 2.0,
) -> Dict[str, Region]:
    """
    Randomly generate non-overlapping regions within a bounding box.
    Args:
        area_bbox: (xmin, xmax, ymin, ymax)
        maze_map: maze map
        labels: list of region labels; assumed non-overlapping
        kind: "circle" or "rect" or "polygon"
    """
    maze_map = env.get_maze_map()
    x, y = maze_map.shape
    xmin, ymin = env.ij_to_xy(0, 0)
    xmax, ymax = env.ij_to_xy(x - 1, y - 1)

    size_scale = float(size_scale)
    if size_scale <= 0:
        raise ValueError(f"size_scale must be positive, got {size_scale}")
    center_jitter = float(center_jitter)
    if center_jitter < 0:
        raise ValueError(f"center_jitter must be non-negative, got {center_jitter}")
    radius_min = float(radius_min)
    radius_max = float(radius_max)
    if radius_min <= 0 or radius_max < radius_min:
        raise ValueError(
            f"radius range must satisfy 0 < min <= max, got {radius_min}, {radius_max}"
        )

    regions = []
    for lab in labels:
        ok = False
        for _ in range(max_tries):
            while True:
                i, j = np.random.randint(0, x), np.random.randint(0, y)
                if maze_map[i, j] == 0:
                    break
            cx, cy = env.ij_to_xy(i, j)
            cx += np.random.uniform(-center_jitter, center_jitter)
            cy += np.random.uniform(-center_jitter, center_jitter)
            if kind == "rect":
                w = np.random.uniform(radius_min, radius_max) * size_scale
                h = np.random.uniform(radius_min, radius_max) * size_scale
                rxmin, rxmax = max(xmin, cx - w / 2), min(xmax, cx + w / 2)
                rymin, rymax = max(ymin, cy - h / 2), min(ymax, cy + h / 2)
                poly = [(rxmin, rymin), (rxmax, rymin), (rxmax, rymax), (rxmin, rymax)]
                region = RectRegion(rxmin, rxmax, rymin, rymax)
            elif kind == "circle":
                r = np.random.uniform(radius_min, radius_max) * size_scale
                region = CircleRegion((cx, cy), r)
            else:
                poly = _sample_convex_polygon(
                    (cx, cy),
                    radius_range=(radius_min * size_scale, radius_max * size_scale),
                    num_vertices=np.random.randint(4, 8),
                )
                region = PolyRegion(poly)
            if _regions_non_overlapping(regions + [region]):
                regions.append(region)
                ok = True
                break
        if not ok:
            raise RuntimeError(
                "Failed to sample non-overlapping regions; relax bounds or sizes."
            )

    region_colors = [plt.cm.rainbow(i / len(regions)) for i in range(len(regions))]
    for region, color in zip(regions, region_colors):
        r, g, b, a = color
        region.color = (r, g, b, alpha)

    return {lab: region for lab, region in zip(labels, regions)}


# -------- Position adapters for common maze-like datasets --------


def pos_antmaze(obs: np.ndarray) -> Tuple[float, float]:
    return float(obs[0]), float(obs[1])


def pos_humanoidmaze(obs: np.ndarray) -> Tuple[float, float]:
    return float(obs[0]), float(obs[1])


def pos_pointmaze(obs: np.ndarray) -> Tuple[float, float]:
    return float(obs[0]), float(obs[1])


def set_pos_xy(obs: np.ndarray, xy: Tuple[float, float]) -> np.ndarray:
    obs = obs.copy()
    obs[0], obs[1] = xy[0], xy[1]
    return obs


def pick_pos_extractor(env_name: str) -> Callable[[np.ndarray], Tuple[float, float]]:
    name = env_name.lower()
    if "antmaze" in name:
        return pos_antmaze
    if "humanoid" in name:
        return pos_humanoidmaze
    if "pointmaze" in name:
        return pos_pointmaze
    return pos_pointmaze


# -------- Label function (single-label, non-overlapping) --------


def make_label_fn(
    label_to_region: Dict[str, Region],
    pos_extractor: Callable[[np.ndarray], Tuple[float, float]],
) -> Callable[[np.ndarray], Optional[str]]:
    """Return raw_state -> Optional[label] under the non-overlap assumption."""

    def label_fn(raw_obs: np.ndarray) -> Optional[str]:
        x, y = pos_extractor(raw_obs)
        for lab, reg in label_to_region.items():
            if reg.contains((x, y)):
                return lab
        return None

    return label_fn


# -------- Anchor sampling for planner --------


def sample_anchor_raw_states(
    label_to_region: Dict[str, Region], template_obs: np.ndarray, per_label: int = 12
) -> Dict[str, List[np.ndarray]]:
    """Sample anchors in each region by writing (x,y) into a template observation."""
    out = {}
    for lab, reg in label_to_region.items():
        arr = []
        for _ in range(per_label):
            xy = reg.sample_point()
            arr.append(set_pos_xy(template_obs, xy))
        out[lab] = arr
    return out


def retrieve_anchor_raw_states_from_dataset(
    label_to_region: Dict[str, Region],
    observations: np.ndarray,
    pos_extractor: Callable[[np.ndarray], Tuple[float, float]],
    per_label: int = 12,
    rng: Optional[np.random.Generator] = None,
    max_scan: int = -1,
) -> Dict[str, List[np.ndarray]]:
    """Retrieve anchor candidates from real offline observations inside each region."""
    observations = np.asarray(observations)
    rng = rng or np.random.default_rng()
    n_obs = len(observations)

    if max_scan is not None and int(max_scan) > 0 and int(max_scan) < n_obs:
        scan_indices = rng.choice(n_obs, size=int(max_scan), replace=False)
    else:
        scan_indices = np.arange(n_obs)

    positions = np.array(
        [pos_extractor(np.asarray(observations[i])) for i in scan_indices],
        dtype=np.float32,
    )

    out = {}
    for lab, reg in label_to_region.items():
        if getattr(reg, "type", None) == "circle":
            center = np.asarray(reg.center, dtype=np.float32)
            mask = np.sum((positions - center) ** 2, axis=1) <= float(reg.radius) ** 2
        elif getattr(reg, "type", None) == "rect":
            mask = (
                (positions[:, 0] >= float(reg.xmin))
                & (positions[:, 0] <= float(reg.xmax))
                & (positions[:, 1] >= float(reg.ymin))
                & (positions[:, 1] <= float(reg.ymax))
            )
        else:
            mask = np.array([reg.contains(tuple(pos)) for pos in positions])

        candidate_indices = scan_indices[np.nonzero(mask)[0]]
        if len(candidate_indices) > per_label:
            candidate_indices = rng.choice(
                candidate_indices, size=int(per_label), replace=False
            )

        out[lab] = [np.array(observations[i]).copy() for i in candidate_indices]
    return out


def build_label_literal_maps(
    labels: List[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (label -> literal, subformula mapping) honoring numeric suffixes when present.

    If a label ends with digits (e.g., "p4" or "e4"), we bind that digit index to the
    corresponding placeholder e4 in the LTL formula so users can reference the human-readable
    label directly. When no digits are present (e.g., "kitchen"), we fall back to the
    enumeration order, preserving the previous behavior.
    """

    def _extract_numeric_suffix(token: str) -> Optional[str]:
        nums = re.findall(r"\d+", token)
        if not nums:
            return None
        suffix = nums[-1]
        # Keep "0" if suffix is composed only of zeros
        trimmed = suffix.lstrip("0")
        return trimmed if trimmed else "0"

    label_to_literal: Dict[str, str] = {}
    subformula: Dict[str, str] = {}
    used_indices: Set[str] = set()
    auto_index = 1

    for idx, label in enumerate(labels):
        literal = f"l{idx + 1}_1"
        label_to_literal[label] = literal

        placeholder_idx = _extract_numeric_suffix(label)
        if placeholder_idx is None or placeholder_idx in used_indices:
            # Fallback to sequential numbering (previous behavior) without collisions.
            while str(auto_index) in used_indices:
                auto_index += 1
            placeholder_idx = str(auto_index)
            auto_index += 1

        subformula[placeholder_idx] = literal
        used_indices.add(placeholder_idx)

    return label_to_literal, subformula


def eform_to_label_formula(formula_eform: str, labels: List[str]) -> str:
    """Map e1,e2,... to label tokens (A,B,...) for human/spot-side formula."""
    s = " " + formula_eform + " "
    for i, lab in enumerate(labels, start=1):
        s = re.sub(rf"\be{i}\b", lab, s)
    return s.strip()


def extract_label_trace(observations: List, label_fn) -> List[Optional[str]]:
    """Map a list of raw observations to a sequence of labels (or None)."""
    return [label_fn(obs) for obs in observations]


def verify_with_buchi_runtime(
    buchi_runtime, label_to_literal: Dict[str, str], label_trace: List[Optional[str]]
) -> bool:
    """
    Fallback conservative check: run Buchi on the finite trace; success if any accepting state is reached.
    """
    Q = {buchi_runtime.initial_state()}
    for lab in label_trace:
        assign = {}
        if lab is not None:
            assign[label_to_literal[lab]] = True
        newQ = set()
        for q in Q:
            for qn in buchi_runtime.succ(q, assign):
                newQ.add(qn)
        if not newQ:
            # newQ = set(Q)
            return False
        Q = newQ
        if any(buchi_runtime.is_accepting(q) for q in Q):
            return True
    return False


def verify_with_buchi_runtime_full(
    buchi_runtime,
    label_to_literal: Dict[str, str],
    label_trace_prefix: List[Optional[str]],
    label_trace_suffix: List[Optional[str]],
) -> bool:
    """
    Full Buchi acceptance check:
    - prefix leads from initial to some accepting state q_a
    - suffix is a loop starting from q_a and passes through q_a again
    """

    # --- Phase 1: prefix simulation ---
    Q = {buchi_runtime.initial_state()}
    for lab in label_trace_prefix:
        assign = {}
        if lab is not None:
            assign[label_to_literal[lab]] = True
        newQ = set()
        for q in Q:
            for qn in buchi_runtime.succ(q, assign):
                newQ.add(qn)
        if not newQ:
            return False, "Prefix check failed"
        Q = newQ

    # --- Phase 2: suffix loop validation from each accepting state reached ---
    for q_accept in Q:
        if not buchi_runtime.is_accepting(q_accept):
            continue
        Qsuf = {q_accept}
        for lab in label_trace_suffix:
            assign = {}
            if lab is not None:
                assign[label_to_literal[lab]] = True
            newQ = set()
            for q in Qsuf:
                for qn in buchi_runtime.succ(q, assign):
                    newQ.add(qn)
            if not newQ:
                break  # loop failed
            Qsuf = newQ
        else:
            # check if loop includes accepting state again
            if any(buchi_runtime.is_accepting(q) for q in Qsuf):
                return True, None

    return False, "Suffix check failed"


# -------- Formula conversion between different notations --------


def convert_formula(formula):
    """
    Convert LTL formula from F/G/&&/||/TRUE/FALSE/p_n to <>/[]/&/|/true/false/e_n format.
    """
    replacements = {
        "||": "|",
        "&&": "&",
        "[]": "G",
        "<>": "F",
        "TRUE": "1",
        "FALSE": "0",
    }

    formula = formula.replace("p0", "pf")
    formula = formula.replace("p1", "pt")

    for new, old in replacements.items():
        formula = formula.replace(old, new)

    formula = (
        formula.replace("pt", "p1")
        .replace("pf", "p0")
        .replace("p", "e")
        .replace("TRUE", "true")
        .replace("FALSE", "false")
        .strip()
    )

    return formula


def check_formula_validity(formula, labels):
    conv = convert_formula(formula)

    class _Task:
        def __init__(self, formula, subformula):
            self.formula = formula
            self.subformula = subformula
            self.number_of_robots = 1

    label_to_literal, subformula = build_label_literal_maps(labels)
    buchi = Buchi(_Task(conv, subformula))
    buchi.construct_buchi_graph()
    buchi.get_minimal_length()
    return buchi.get_feasible_accepting_state()
