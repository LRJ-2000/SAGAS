"""Object-state predicates for OGBench manipulation scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np

from S_utils.semantic_types import PredicateCertificate, SemanticCarrier


XYZ_CENTER = np.array([0.425, 0.0, 0.0], dtype=np.float64)
XYZ_SCALER = 10.0
DRAWER_SCALER = 18.0
WINDOW_SCALER = 15.0
QPOS_OBJ_START = 14
QPOS_CUBE_LENGTH = 7
DEFAULT_NUM_CUBES = 1
DEFAULT_NUM_BUTTONS = 2
DEFAULT_DRAWER_HANDLE_X = 0.3302867
DEFAULT_DRAWER_HANDLE_Y_AT_CLOSED = -0.24
DEFAULT_DRAWER_HANDLE_Z = 0.08
DEFAULT_WINDOW_HANDLE_X_AT_CLOSED = 0.286
DEFAULT_WINDOW_HANDLE_Y = 0.225
DEFAULT_WINDOW_HANDLE_Z = 0.237
DEFAULT_DRAWER_OPEN_THRESHOLD = -0.10
DEFAULT_DRAWER_CLOSED_THRESHOLD = -0.08
DEFAULT_WINDOW_OPEN_THRESHOLD = 0.12
DEFAULT_WINDOW_CLOSED_THRESHOLD = 0.08


@dataclass(frozen=True)
class ObjectStatePredicateSpec:
    """Normalized object-state proposition declaration."""

    proposition: str
    object_name: str
    property_name: str
    index: Optional[int] = None
    region: Optional[Dict[str, Any]] = None
    thresholds: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def _as_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return arr.astype(np.float64, copy=False).reshape(-1)


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _parse_button_index(value: Any, fallback: Optional[int] = None) -> Optional[int]:
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return fallback


def _infer_spec(prop: str, raw: Mapping[str, Any]) -> ObjectStatePredicateSpec:
    spec = dict(raw)
    object_name = str(spec.get("object", spec.get("target", ""))).strip()
    property_name = str(spec.get("property", spec.get("predicate", ""))).strip()
    index = _parse_button_index(spec.get("index"))

    if not property_name:
        if isinstance(spec.get("all_of"), list):
            object_name = object_name or "scene"
            property_name = "all_of"
        elif isinstance(spec.get("any_of"), list):
            object_name = object_name or "scene"
            property_name = "any_of"

    prop_key = prop.lower()
    if not property_name:
        if prop_key in {"drawer_open", "drawer_closed", "window_open", "window_closed"}:
            object_name, property_name = prop_key.split("_", 1)
        elif prop_key.startswith("button") and (
            prop_key.endswith("_on") or prop_key.endswith("_off")
        ):
            object_name = "button"
            property_name = "off" if prop_key.endswith("_off") else "on"
            digits = "".join(ch for ch in prop_key if ch.isdigit())
            index = int(digits) if digits else index
        elif prop_key in {"cube_in_drawer", "block_in_drawer"}:
            object_name = "cube"
            property_name = "in_drawer"
        elif prop_key in {"cube_in_region", "block_in_region"}:
            object_name = "cube"
            property_name = "in_region"

    object_key = object_name.lower()
    if not object_name and property_name.lower() in {"cube_in_drawer", "block_in_drawer"}:
        object_name = "cube"
        object_key = "cube"
    if not object_name and property_name.lower() in {"cube_in_region", "block_in_region"}:
        object_name = "cube"
        object_key = "cube"
    if object_key.startswith("button") and object_key != "button":
        digits = "".join(ch for ch in object_key if ch.isdigit())
        if digits:
            index = int(digits)
        object_name = "button"
    if object_key in {"block", "block_0"}:
        object_name = "cube"

    thresholds = {
        str(k): float(v)
        for k, v in dict(spec.get("thresholds", {})).items()
        if isinstance(v, (int, float))
    }
    for key in (
        "drawer_open_threshold",
        "drawer_closed_threshold",
        "window_open_threshold",
        "window_closed_threshold",
        "radius",
        "r",
    ):
        if key in spec and isinstance(spec[key], (int, float)):
            thresholds[key] = float(spec[key])

    return ObjectStatePredicateSpec(
        proposition=prop,
        object_name=object_name or "scene",
        property_name=property_name,
        index=index,
        region=dict(spec["region"]) if isinstance(spec.get("region"), dict) else None,
        thresholds=thresholds,
        raw=spec,
    )


def normalize_object_state_specs(
    specs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, ObjectStatePredicateSpec]:
    """Normalize raw predicate specs into object-state predicate declarations."""

    return {str(prop): _infer_spec(str(prop), raw) for prop, raw in specs.items()}


def _parse_qpos(qpos: Sequence[float], out: Dict[str, Any]) -> None:
    arr = _as_array(qpos)
    if arr is None:
        return
    if arr.size >= QPOS_OBJ_START + QPOS_CUBE_LENGTH:
        out.setdefault("block_positions", {})[0] = arr[QPOS_OBJ_START : QPOS_OBJ_START + 3].copy()
    drawer_idx = QPOS_OBJ_START + DEFAULT_NUM_CUBES * QPOS_CUBE_LENGTH + DEFAULT_NUM_BUTTONS
    window_idx = drawer_idx + 1
    if arr.size > drawer_idx:
        out["drawer_pos"] = float(arr[drawer_idx])
    if arr.size > window_idx:
        out["window_pos"] = float(arr[window_idx])


def _parse_scene_observation(observation: Sequence[float], out: Dict[str, Any]) -> None:
    arr = _as_array(observation)
    if arr is None:
        return
    if arr.size == 7:
        out.setdefault("block_positions", {})[0] = arr[0:3] / XYZ_SCALER + XYZ_CENTER
        out["button_states"] = np.rint(arr[3:5]).astype(int)
        out["drawer_pos"] = float(arr[5] / DRAWER_SCALER)
        out["window_pos"] = float(arr[6] / WINDOW_SCALER)
        out.setdefault("state_format", "scene_oracle_rep")
        return
    if arr.size >= 40:
        out.setdefault("block_positions", {})[0] = arr[19:22] / XYZ_SCALER + XYZ_CENTER
        out["button_states"] = np.array(
            [int(np.argmax(arr[28:30])), int(np.argmax(arr[32:34]))],
            dtype=int,
        )
        out["drawer_pos"] = float(arr[36] / DRAWER_SCALER)
        out["window_pos"] = float(arr[38] / WINDOW_SCALER)
        out.setdefault("state_format", "scene_observation")
        return
    _parse_qpos(arr, out)
    out.setdefault("state_format", "qpos_array")


def _augment_runtime_handle_positions(out: Dict[str, Any]) -> Dict[str, Any]:
    """Fill handle positions implied by SceneEnv scalar joints when absent.

    Runtime evaluation prefers the environment-provided handle site positions.
    Saved task-info dictionaries and some support carriers only contain scalar
    drawer/window joint positions, so this helper reconstructs the same local
    handle frame used by SceneEnv's object-state oracle closely enough for
    generation-time filtering.  Existing privileged handle positions are never
    overwritten.
    """

    if "drawer_pos" in out and "drawer_handle_pos" not in out:
        drawer_joint = float(out["drawer_pos"])
        out["drawer_handle_pos"] = np.array(
            [
                DEFAULT_DRAWER_HANDLE_X,
                DEFAULT_DRAWER_HANDLE_Y_AT_CLOSED - drawer_joint,
                DEFAULT_DRAWER_HANDLE_Z,
            ],
            dtype=np.float64,
        )
    if "window_pos" in out and "window_handle_pos" not in out:
        window_joint = float(out["window_pos"])
        out["window_handle_pos"] = np.array(
            [
                DEFAULT_WINDOW_HANDLE_X_AT_CLOSED + window_joint,
                DEFAULT_WINDOW_HANDLE_Y,
                DEFAULT_WINDOW_HANDLE_Z,
            ],
            dtype=np.float64,
        )
    return out


def scene_state_from_value(value: Any) -> Dict[str, Any]:
    """Extract scene object facts from a carrier, dict, qpos, or scene observation."""

    out: Dict[str, Any] = {"block_positions": {}}
    if isinstance(value, SemanticCarrier):
        if value.info:
            out.update(scene_state_from_value(value.info))
        if value.state is not None:
            out.update({k: v for k, v in scene_state_from_value(value.state).items() if v is not None})
        elif value.observation is not None:
            out.update(
                {k: v for k, v in scene_state_from_value(value.observation).items() if v is not None}
            )
        return _augment_runtime_handle_positions(out)

    if isinstance(value, Mapping):
        qpos = _first(value, "qpos", "state")
        if qpos is not None:
            _parse_qpos(qpos, out)
        button_states = _first(value, "button_states", "privileged/button_states")
        if button_states is not None:
            arr = _as_array(button_states)
            if arr is not None:
                out["button_states"] = np.rint(arr).astype(int)
        raw_block_positions = _first(value, "block_positions", "blocks")
        if isinstance(raw_block_positions, Mapping):
            block_positions = out.setdefault("block_positions", {})
            for key, raw_pos in raw_block_positions.items():
                arr = _as_array(raw_pos)
                if arr is not None and arr.size >= 3:
                    block_positions[int(key)] = arr[:3].copy()
        elif raw_block_positions is not None:
            arr = np.asarray(raw_block_positions, dtype=np.float64)
            block_positions = out.setdefault("block_positions", {})
            if arr.ndim == 1 and arr.size >= 3:
                block_positions[0] = arr[:3].copy()
            elif arr.ndim >= 2:
                for idx, raw_pos in enumerate(arr):
                    pos_arr = np.asarray(raw_pos, dtype=np.float64).reshape(-1)
                    if pos_arr.size >= 3:
                        block_positions[int(idx)] = pos_arr[:3].copy()
        for idx in range(DEFAULT_NUM_BUTTONS):
            key = f"privileged/button_{idx}_state"
            if key in value:
                states = out.setdefault(
                    "button_states",
                    np.full(DEFAULT_NUM_BUTTONS, -1, dtype=int),
                )
                states[idx] = int(np.asarray(value[key]).reshape(-1)[0])
        block_positions = out.setdefault("block_positions", {})
        for idx in range(DEFAULT_NUM_CUBES):
            key = f"privileged/block_{idx}_pos"
            if key in value:
                arr = _as_array(value[key])
                if arr is not None and arr.size >= 3:
                    block_positions[idx] = arr[:3].copy()
        drawer_pos = _first(value, "privileged/drawer_pos", "drawer_pos")
        if drawer_pos is not None:
            arr = _as_array(drawer_pos)
            if arr is not None:
                out["drawer_pos"] = float(arr[0])
        window_pos = _first(value, "privileged/window_pos", "window_pos")
        if window_pos is not None:
            arr = _as_array(window_pos)
            if arr is not None:
                out["window_pos"] = float(arr[0])
        drawer_handle = _first(value, "privileged/drawer_handle_pos", "drawer_handle_pos")
        if drawer_handle is not None:
            arr = _as_array(drawer_handle)
            if arr is not None and arr.size >= 3:
                out["drawer_handle_pos"] = arr[:3].copy()
        window_handle = _first(value, "privileged/window_handle_pos", "window_handle_pos")
        if window_handle is not None:
            arr = _as_array(window_handle)
            if arr is not None and arr.size >= 3:
                out["window_handle_pos"] = arr[:3].copy()
        return _augment_runtime_handle_positions(out)

    _parse_scene_observation(value, out)
    return _augment_runtime_handle_positions(out)


def scene_state_from_task_info(task_info: Mapping[str, Any]) -> Dict[str, Any]:
    """Build oracle scene facts from a saved SceneEnv task-info dictionary.

    The random task generator uses this helper to validate initial labels
    without constructing or resetting a MuJoCo environment.  The input may be a
    full task-info dictionary with an ``init`` field or the ``init`` dictionary
    itself.
    """

    init = task_info.get("init", task_info)
    out: Dict[str, Any] = {"block_positions": {}, "state_format": "scene_task_info"}
    block_xyzs = init.get("block_xyzs")
    if block_xyzs is not None:
        arr = np.asarray(block_xyzs, dtype=np.float64)
        if arr.ndim == 1 and arr.size >= 3:
            out["block_positions"][0] = arr[:3].copy()
        elif arr.ndim >= 2:
            for idx, xyz in enumerate(arr):
                xyz_arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
                if xyz_arr.size >= 3:
                    out["block_positions"][int(idx)] = xyz_arr[:3].copy()
    button_states = init.get("button_states")
    if button_states is not None:
        out["button_states"] = np.rint(np.asarray(button_states, dtype=np.float64)).astype(int)
    if "drawer_pos" in init:
        out["drawer_pos"] = float(init["drawer_pos"])
    if "window_pos" in init:
        out["window_pos"] = float(init["window_pos"])
    return _augment_runtime_handle_positions(out)


def scene_task_info_for_env(task_info: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert JSON-friendly SceneEnv task info into arrays expected by reset."""

    converted: Dict[str, Any] = dict(task_info)
    for section in ("init", "goal"):
        raw = dict(converted.get(section, {}))
        if "block_xyzs" in raw:
            raw["block_xyzs"] = np.asarray(raw["block_xyzs"], dtype=np.float64)
        if "button_states" in raw:
            raw["button_states"] = np.asarray(raw["button_states"], dtype=int)
        if "drawer_pos" in raw:
            raw["drawer_pos"] = float(raw["drawer_pos"])
        if "window_pos" in raw:
            raw["window_pos"] = float(raw["window_pos"])
        converted[section] = raw
    return converted


def _block_pos(scene_state: Mapping[str, Any], index: int = 0) -> Optional[np.ndarray]:
    blocks = scene_state.get("block_positions", {})
    if isinstance(blocks, Mapping) and index in blocks:
        arr = _as_array(blocks[index])
        if arr is not None and arr.size >= 3:
            return arr[:3]
    return None


def _button_state(scene_state: Mapping[str, Any], index: int) -> Optional[int]:
    states = _as_array(scene_state.get("button_states"))
    if states is None or states.size <= index:
        return None
    return int(round(float(states[index])))


def _point_in_3d_region(point: Sequence[float], region: Mapping[str, Any]) -> bool:
    p = _as_array(point)
    if p is None or p.size < 3:
        return False
    kind = str(region.get("kind", region.get("type", "sphere"))).lower()
    if kind in {"sphere", "ball"}:
        center = _as_array(region.get("center"))
        radius = region.get("radius", region.get("r"))
        if center is None or center.size < 3 or radius is None:
            return False
        return float(np.linalg.norm(p[:3] - center[:3])) <= float(radius)
    if kind in {"box", "aabb"}:
        if "bounds" in region:
            bounds = list(region["bounds"])
            if len(bounds) == 6:
                low = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
                high = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
            else:
                return False
        else:
            low = _as_array(region.get("lower", region.get("low")))
            high = _as_array(region.get("upper", region.get("high")))
            if low is None or high is None or low.size < 3 or high.size < 3:
                return False
        return bool(np.all(low[:3] <= p[:3]) and np.all(p[:3] <= high[:3]))
    if kind in {"disk", "circle", "cylinder"}:
        center = _as_array(region.get("center"))
        radius = region.get("radius", region.get("r"))
        if center is None or center.size < 2 or radius is None:
            return False
        xy_inside = float(np.linalg.norm(p[:2] - center[:2])) <= float(radius)
        z_min = region.get("z_min", region.get("zmin"))
        z_max = region.get("z_max", region.get("zmax"))
        z_ok = True
        if z_min is not None:
            z_ok = z_ok and float(p[2]) >= float(z_min)
        if z_max is not None:
            z_ok = z_ok and float(p[2]) <= float(z_max)
        return bool(xy_inside and z_ok)
    raise ValueError(f"Unsupported 3-D region kind: {kind}")


def _cube_in_drawer(scene_state: Mapping[str, Any], cube_index: int = 0) -> bool:
    pos = _block_pos(scene_state, cube_index)
    if pos is None:
        return False
    if "drawer_handle_pos" in scene_state:
        drawer_pos_y = float(_as_array(scene_state["drawer_handle_pos"])[1])
        low = np.array([0.21, drawer_pos_y - 0.27, 0.0], dtype=np.float64)
        high = np.array([0.45, drawer_pos_y - 0.07, 0.15], dtype=np.float64)
    else:
        low = np.array([0.21, -0.43, 0.0], dtype=np.float64)
        high = np.array([0.45, -0.15, 0.15], dtype=np.float64)
    return bool(np.all(low <= pos[:3]) and np.all(pos[:3] <= high))


def evaluate_object_state_predicate(
    scene_state: Mapping[str, Any],
    spec: ObjectStatePredicateSpec,
) -> Optional[bool]:
    """Evaluate one normalized object-state predicate.

    Returns ``None`` when the required object facts are not available.
    """

    if isinstance(spec.raw.get("all_of"), list):
        child_results = [
            evaluate_object_state_predicate(
                scene_state,
                _infer_spec(
                    f"{spec.proposition}__all_{idx}",
                    child if isinstance(child, Mapping) else {},
                ),
            )
            for idx, child in enumerate(spec.raw["all_of"])
        ]
        if any(result is False for result in child_results):
            return False
        if any(result is None for result in child_results):
            return None
        return True
    if isinstance(spec.raw.get("any_of"), list):
        child_results = [
            evaluate_object_state_predicate(
                scene_state,
                _infer_spec(
                    f"{spec.proposition}__any_{idx}",
                    child if isinstance(child, Mapping) else {},
                ),
            )
            for idx, child in enumerate(spec.raw["any_of"])
        ]
        if any(result is True for result in child_results):
            return True
        if any(result is None for result in child_results):
            return None
        return False

    prop = spec.property_name.lower()
    obj = spec.object_name.lower()
    index = 0 if spec.index is None else int(spec.index)
    if obj == "button" or prop in {"button0_on", "button1_on", "button0_off", "button1_off"}:
        if prop == "button0_on":
            index = 0
        elif prop == "button1_on":
            index = 1
        elif prop == "button0_off":
            index = 0
        elif prop == "button1_off":
            index = 1
        state = _button_state(scene_state, index)
        if state is None:
            return None
        on_value = int(spec.thresholds.get("on_value", spec.raw.get("on_value", 1)))
        if prop in {"off", "button0_off", "button1_off"}:
            return bool(state != on_value)
        return bool(state == on_value)
    if obj == "drawer" or prop.startswith("drawer_"):
        drawer_pos = scene_state.get("drawer_pos")
        if drawer_pos is None:
            return None
        open_threshold = float(
            spec.thresholds.get("drawer_open_threshold", DEFAULT_DRAWER_OPEN_THRESHOLD)
        )
        closed_threshold = float(
            spec.thresholds.get("drawer_closed_threshold", DEFAULT_DRAWER_CLOSED_THRESHOLD)
        )
        if prop in {"open", "drawer_open"}:
            return bool(float(drawer_pos) <= open_threshold)
        if prop in {"closed", "drawer_closed"}:
            return bool(float(drawer_pos) >= closed_threshold)
    if obj == "window" or prop.startswith("window_"):
        window_pos = scene_state.get("window_pos")
        if window_pos is None:
            return None
        open_threshold = float(
            spec.thresholds.get("window_open_threshold", DEFAULT_WINDOW_OPEN_THRESHOLD)
        )
        closed_threshold = float(
            spec.thresholds.get("window_closed_threshold", DEFAULT_WINDOW_CLOSED_THRESHOLD)
        )
        if prop in {"open", "window_open"}:
            return bool(float(window_pos) >= open_threshold)
        if prop in {"closed", "window_closed"}:
            return bool(float(window_pos) <= closed_threshold)
    if obj in {"cube", "block", "block_0"}:
        if prop in {"in_drawer", "cube_in_drawer"}:
            return _cube_in_drawer(scene_state, index)
        if prop in {"in_region", "cube_in_region"}:
            pos = _block_pos(scene_state, index)
            if pos is None or spec.region is None:
                return None
            return _point_in_3d_region(pos, spec.region)
    return None


class ObjectStateEvaluator:
    """Deterministic evaluator for manipulation object-state propositions."""

    def __init__(self, predicate_specs: Mapping[str, Mapping[str, Any]]):
        self.predicate_specs = normalize_object_state_specs(predicate_specs)
        self.certificate_counts = {"positive": 0, "negative": 0, "uncertain": 0}

    @property
    def propositions(self) -> List[str]:
        return list(self.predicate_specs.keys())

    def evaluate(self, carrier: SemanticCarrier, proposition: str) -> PredicateCertificate:
        prop = str(proposition)
        spec = self.predicate_specs.get(prop)
        if spec is None:
            self.certificate_counts["uncertain"] += 1
            return PredicateCertificate.uncertain_value(
                prop,
                score=0.0,
                lower=0.0,
                upper=1.0,
                metadata={"reason": "unknown_object_state_proposition"},
            )
        scene_state = scene_state_from_value(carrier)
        result = evaluate_object_state_predicate(scene_state, spec)
        metadata = {
            "evaluator": "ObjectStateEvaluator",
            "object": spec.object_name,
            "property": spec.property_name,
            "index": spec.index,
            "composite": bool(
                isinstance(spec.raw.get("all_of"), list)
                or isinstance(spec.raw.get("any_of"), list)
            ),
            "state_format": scene_state.get("state_format", "mapping_or_carrier"),
        }
        if result is None:
            self.certificate_counts["uncertain"] += 1
            return PredicateCertificate.uncertain_value(
                prop,
                score=0.0,
                lower=0.0,
                upper=1.0,
                metadata={**metadata, "reason": "insufficient_scene_object_state"},
            )
        if result:
            self.certificate_counts["positive"] += 1
            return PredicateCertificate.positive(prop, metadata=metadata)
        self.certificate_counts["negative"] += 1
        return PredicateCertificate.negative(prop, metadata=metadata)

    def diagnostics_summary(self) -> Dict[str, Any]:
        total = sum(self.certificate_counts.values())
        return {
            "backend": "ObjectStateEvaluator",
            "num_propositions": len(self.predicate_specs),
            "certificate_counts": dict(self.certificate_counts),
            "uncertain_certificate_ratio": (
                self.certificate_counts["uncertain"] / float(total) if total else 0.0
            ),
        }


def scene_object_label_set(
    value: Any,
    predicate_specs: Mapping[str, Mapping[str, Any]],
) -> Set[str]:
    """Return all object-state propositions satisfied by one scene state."""

    normalized = normalize_object_state_specs(predicate_specs)
    scene_state = scene_state_from_value(value)
    labels = set()
    for prop, spec in normalized.items():
        if evaluate_object_state_predicate(scene_state, spec):
            labels.add(prop)
    return labels


def scene_object_label_trace(
    values: Iterable[Any],
    predicate_specs: Mapping[str, Mapping[str, Any]],
) -> List[Set[str]]:
    """Label a trajectory or dataset slice with object-state propositions."""

    return [scene_object_label_set(value, predicate_specs) for value in values]
