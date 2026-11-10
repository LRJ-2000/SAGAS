"""Environment-aware GAS agent configuration helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def is_visual_env(env_name: Optional[str]) -> bool:
    """Return whether an OGBench environment uses pixel observations."""
    return bool(env_name) and str(env_name).startswith("visual-")


def default_action_dim(env_name: Optional[str]) -> int:
    """Return the action dimension used to instantiate GAS agents."""

    env = str(env_name or "").lower()
    if any(name in env for name in ("cube", "scene", "puzzle", "manip")):
        return 5
    if "humanoid" in env:
        return 21
    return 8


def default_agent_config_overrides(env_name: Optional[str]) -> Dict[str, Any]:
    """Return GAS-compatible agent-config overrides for an environment.

    The base `M_utils/agents/gas.py` config is state-based. Official GAS pixel
    checkpoints use an Impala encoder and smaller visual-training batches, so
    visual environments need these overrides whenever the checkpoint is loaded
    or a KeyGraph is rebuilt from that checkpoint.
    """
    if not is_visual_env(env_name):
        return {}

    env = str(env_name)
    return {
        "encoder": "impala_small",
        "batch_size": 256,
        "p_aug": 0.5,
        "discount": 0.995 if "giant" in env else 0.99,
        "tdr_expectile": 0.95,
        "alpha": 0.01 if "explore" in env else 1.0,
    }


def default_te_threshold(env_name: Optional[str]) -> float:
    """Return the GAS TE threshold convention for graph construction."""
    return 0.9 if is_visual_env(env_name) else 0.99


def _copy_config(config: Any) -> Any:
    if hasattr(config, "copy"):
        return config.copy()
    return dict(config)


def apply_agent_config_overrides(config: Any, overrides: Mapping[str, Any]) -> Any:
    """Return a copied config with the provided top-level overrides applied."""
    resolved = _copy_config(config)
    for key, value in overrides.items():
        resolved[key] = value
    return resolved


def resolve_agent_config_for_env(
    config: Any,
    env_name: Optional[str],
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Apply default and explicit agent-config overrides for an environment."""
    merged = dict(default_agent_config_overrides(env_name))
    if overrides:
        merged.update(dict(overrides))
    if not merged:
        return _copy_config(config)
    return apply_agent_config_overrides(config, merged)
