"""Predicate specification normalization and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SUPPORTED_PREDICATE_BACKENDS = {
    "state_region",
    "oracle_state_lift",
    "object_state",
}


def normalize_evaluator_backend(value: Any) -> str:
    """Normalize evaluator backend aliases used in task files and CLI flags."""

    key = str(value or "").strip().lower()
    aliases = {
        "state": "state_region",
        "state_region_evaluator": "state_region",
        "state-predicate": "state_region",
        "state_predicate": "state_region",
        "oracle": "oracle_state_lift",
        "oracle_state": "oracle_state_lift",
        "oracle-state-lift": "oracle_state_lift",
        "object": "object_state",
        "object-state": "object_state",
        "scene_object": "object_state",
        "manipulation_object": "object_state",
    }
    return aliases.get(key, key)


def _label_from_region_spec(idx: int, spec: Mapping[str, Any]) -> str:
    return str(spec.get("id", spec.get("proposition", spec.get("name", f"e{idx}"))))


def task_regions_to_region_specs(regions: Any) -> List[Dict[str, Any]]:
    """Normalize saved task regions into evaluator geometry specs."""

    if isinstance(regions, dict):
        items = list(regions.items())
    elif isinstance(regions, list):
        items = [(_label_from_region_spec(idx, spec), spec) for idx, spec in enumerate(regions)]
    else:
        raise ValueError("regions must be a dict or list of region specs")

    out: List[Dict[str, Any]] = []
    for label, raw_spec in items:
        spec = dict(raw_spec)
        prop = str(spec.get("proposition", spec.get("id", label)))
        spec.setdefault("id", prop)
        spec.setdefault("proposition", prop)
        if "kind" not in spec and "type" in spec:
            spec["kind"] = spec["type"]
        if spec.get("kind") == "rect" and "bounds" not in spec:
            spec["bounds"] = (
                float(spec["xmin"]),
                float(spec["xmax"]),
                float(spec["ymin"]),
                float(spec["ymax"]),
            )
        out.append(spec)
    return out


def _legacy_regions_to_predicate_specs(regions: Any) -> Dict[str, Dict[str, Any]]:
    return {
        str(spec["proposition"]): {
            "id": str(spec["proposition"]),
            "proposition": str(spec["proposition"]),
            "evaluator": "state_region",
            "geometry": dict(spec),
        }
        for spec in task_regions_to_region_specs(regions)
    }


def _metadata(task: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = task.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _normalize_geometry(prop: str, geometry: Mapping[str, Any]) -> Dict[str, Any]:
    geom = dict(geometry)
    geom.setdefault("id", prop)
    geom.setdefault("proposition", prop)
    if "kind" not in geom and "type" in geom:
        geom["kind"] = geom["type"]
    return geom


def extract_predicate_specs(task: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Return normalized per-proposition specs and the fallback source used."""

    if isinstance(task.get("predicate_specs"), dict):
        raw_specs = task["predicate_specs"]
        source = "task.predicate_specs"
    elif isinstance(_metadata(task).get("predicate_specs"), dict):
        raw_specs = _metadata(task)["predicate_specs"]
        source = "task.metadata.predicate_specs"
    elif "regions" in task:
        raw_specs = _legacy_regions_to_predicate_specs(task["regions"])
        source = "task.regions"
    else:
        return {}, "none"

    normalized: Dict[str, Dict[str, Any]] = {}
    for label, raw_spec in raw_specs.items():
        spec = dict(raw_spec)
        prop = str(spec.get("proposition", spec.get("id", label)))
        spec.setdefault("proposition", prop)
        spec.setdefault("id", prop)
        spec["evaluator"] = normalize_evaluator_backend(spec.get("evaluator", "state_region"))

        geometry = spec.get("geometry")
        if isinstance(geometry, dict):
            spec["geometry"] = _normalize_geometry(prop, geometry)

        normalized[prop] = spec
    return normalized, source


def region_specs_by_proposition(
    region_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for spec in region_specs:
        prop = spec.get("proposition", spec.get("id", spec.get("name")))
        if prop is not None:
            out[str(prop)] = dict(spec)
    return out


def task_to_region_specs(task: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return oracle/visualization region specs from regions and predicate geometries."""

    specs_by_prop: Dict[str, Dict[str, Any]] = {}
    if "regions" in task:
        for spec in task_regions_to_region_specs(task["regions"]):
            prop = spec.get("proposition", spec.get("id"))
            if prop is not None:
                specs_by_prop[str(prop)] = spec

    predicate_specs, _source = extract_predicate_specs(task)
    for prop, pred_spec in predicate_specs.items():
        geometry = pred_spec.get("geometry")
        if isinstance(geometry, dict):
            geom = _normalize_geometry(str(prop), geometry)
            specs_by_prop[str(prop)] = task_regions_to_region_specs({prop: geom})[0]

    if not specs_by_prop:
        predicate_specs, source = extract_predicate_specs(task)
        if predicate_specs and source != "none":
            return []
        raise ValueError("Task has neither `regions` nor predicate geometries")
    return [specs_by_prop[prop] for prop in sorted(specs_by_prop)]


@dataclass
class PredicateSpecValidationReport:
    """JSON-serializable validation summary for task predicate declarations."""

    fallback_used: str
    num_predicates: int
    backends: Dict[str, str] = field(default_factory=dict)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fallback_used": self.fallback_used,
            "num_predicates": int(self.num_predicates),
            "backends": dict(sorted(self.backends.items())),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "ok": bool(self.ok),
        }


def _add_issue(
    bucket: List[Dict[str, Any]],
    *,
    proposition: str,
    code: str,
    message: str,
    **extra: Any,
) -> None:
    issue = {
        "proposition": str(proposition),
        "code": str(code),
        "message": str(message),
    }
    issue.update(extra)
    bucket.append(issue)


def validate_predicate_specs(
    predicate_specs: Mapping[str, Mapping[str, Any]],
    *,
    fallback_used: str = "unknown",
    labels: Optional[Iterable[str]] = None,
) -> PredicateSpecValidationReport:
    """Validate normalized predicate specs without constructing evaluators."""

    specs = {str(prop): dict(spec) for prop, spec in predicate_specs.items()}
    report = PredicateSpecValidationReport(
        fallback_used=str(fallback_used),
        num_predicates=len(specs),
    )

    label_set = {str(label) for label in labels or [] if str(label).strip()}
    for label in sorted(label_set - set(specs)):
        _add_issue(
            report.warnings,
            proposition=label,
            code="missing_predicate_spec_for_label",
            message="Label appears in the task but has no predicate spec.",
        )

    for prop, spec in sorted(specs.items()):
        backend = normalize_evaluator_backend(spec.get("evaluator", "state_region"))
        report.backends[prop] = backend
        if backend not in SUPPORTED_PREDICATE_BACKENDS:
            _add_issue(
                report.warnings,
                proposition=prop,
                code="unknown_evaluator_backend",
                message="Unsupported predicate evaluator backend.",
                backend=backend,
            )
            continue

        if backend in {"state_region", "oracle_state_lift"}:
            if not isinstance(spec.get("geometry"), dict):
                _add_issue(
                    report.errors,
                    proposition=prop,
                    code="missing_geometry",
                    message=f"{backend} requires a geometry field.",
                    backend=backend,
                )

        if backend == "object_state":
            prop_name = str(spec.get("property", spec.get("predicate", ""))).strip()
            object_name = str(spec.get("object", spec.get("target", ""))).strip()
            if not prop_name:
                _add_issue(
                    report.errors,
                    proposition=prop,
                    code="missing_object_property",
                    message="object_state requires a property/predicate field.",
                    backend=backend,
                )
            if not object_name and prop_name not in {
                "drawer_open",
                "drawer_closed",
                "window_open",
                "window_closed",
                "button0_on",
                "button0_off",
                "button1_on",
                "button1_off",
                "cube_in_drawer",
            }:
                _add_issue(
                    report.warnings,
                    proposition=prop,
                    code="missing_object_name",
                    message="object_state usually declares an object or target field.",
                    backend=backend,
                )
            if prop_name in {"in_region", "cube_in_region"} and not isinstance(spec.get("region"), dict):
                _add_issue(
                    report.errors,
                    proposition=prop,
                    code="missing_object_region",
                    message="cube_in_region/in_region requires a region field.",
                    backend=backend,
                )

    return report
