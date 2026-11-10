"""Prepare local copies of graph and checkpoint assets.

The journal code should not train TDR or low-level policy models as part of
the first integration pass.  Instead, this helper copies existing conference
SAGAS checkpoints into the specified workspace and writes a manifest
that later scripts can consume when rebuilding KeyGraphs or running rollouts.

The script never moves source files and never overwrites destination files by
default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from M_utils.agent_config_utils import default_agent_config_overrides


def env_stem(env_name: str) -> str:
    """Convert `antmaze-giant-stitch-v0` to `antmaze-giant-stitch`."""
    stem = str(env_name)
    if stem.endswith("-v0"):
        stem = stem[:-3]
    return stem


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _candidate_dirs(source_root: Path, family: str, env_name: str) -> List[Path]:
    stem = env_stem(env_name)
    base = source_root / family / f"EXP_{stem}"
    if not base.exists():
        return []
    prefix = f"{env_name}_sd"
    dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def find_checkpoint(
    source_root: Path,
    family: str,
    env_name: str,
    epoch: int = 1000000,
) -> Path:
    """Find the newest matching checkpoint for one model family."""
    filename = f"params_{int(epoch)}.pkl"
    for directory in _candidate_dirs(source_root, family, env_name):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} under {source_root / family / ('EXP_' + env_stem(env_name))}"
    )


def copy_file_no_overwrite(src: Path, dst: Path) -> Dict:
    """Copy `src` to `dst` unless an identical destination already exists."""
    src = src.resolve()
    dst = dst.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_stat = src.stat()
    if dst.exists():
        if not dst.is_file():
            raise FileExistsError(f"Destination exists and is not a file: {dst}")
        dst_stat = dst.stat()
        if dst_stat.st_size == src_stat.st_size and _sha256(dst) == _sha256(src):
            return {
                "status": "skipped_existing_identical",
                "src": str(src),
                "dst": str(dst),
                "bytes": int(dst_stat.st_size),
                "sha256": _sha256(dst),
            }
        raise FileExistsError(
            f"Destination exists with different content; refusing to overwrite: {dst}"
        )
    shutil.copy2(src, dst)
    return {
        "status": "copied",
        "src": str(src),
        "dst": str(dst),
        "bytes": int(src_stat.st_size),
        "sha256": _sha256(dst),
    }


def relative_checkpoint_destination(
    source_root: Path,
    checkpoint_path: Path,
    target_root: Path,
) -> Path:
    """Preserve checkpoint family/run structure under `assets/copied_models`."""
    # Preserve the user-facing path under source_root even when EXP_tdr or
    # EXP_policy is a symlink to a shared model store.
    rel = checkpoint_path.absolute().relative_to(source_root.absolute())
    return target_root / "assets" / "copied_models" / rel


def prepare_assets(
    source_root: Path,
    target_root: Path,
    env_name: str,
    epoch: int = 1000000,
    families: Optional[Iterable[str]] = None,
    manifest_name: Optional[str] = None,
) -> Dict:
    """Copy selected model checkpoints and write an asset manifest."""
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    families = list(families or ["EXP_tdr", "EXP_policy"])
    copied = {}
    for family in families:
        ckpt = find_checkpoint(source_root, family, env_name, epoch=epoch)
        dst = relative_checkpoint_destination(source_root, ckpt, target_root)
        copied[family] = copy_file_no_overwrite(ckpt, dst)

    manifest = {
        "env_name": env_name,
        "epoch": int(epoch),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "agent_config_overrides": default_agent_config_overrides(env_name),
        "checkpoints": copied,
        "tdr_path": copied.get("EXP_tdr", {}).get("dst"),
        "policy_path": copied.get("EXP_policy", {}).get("dst"),
    }
    manifest_dir = target_root / "assets" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if manifest_name is None:
        manifest_name = f"{env_stem(env_name)}_assets.json"
    manifest_path = manifest_dir / manifest_name
    # The manifest records the latest preparation status and is safe to refresh:
    # checkpoint files themselves remain protected by copy_file_no_overwrite.
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def prepare_official_gas_assets(
    checkpoint_dir: Path,
    target_root: Path,
    env_name: str,
    epoch: int = 500000,
    manifest_name: Optional[str] = None,
) -> Dict:
    """Copy an official flat GAS checkpoint and write a SAGAS asset manifest.

    Official GAS release folders store one full agent checkpoint per
    environment.  SAGAS uses that same restored agent both as the temporal
    distance encoder for KeyGraph rebuilding and as the frozen low-level policy
    for execution, so the manifest intentionally points `tdr_path` and
    `policy_path` to the same copied file.
    """
    checkpoint_dir = checkpoint_dir.resolve()
    target_root = target_root.resolve()
    src = checkpoint_dir / f"params_{int(epoch)}.pkl"
    dst = (
        target_root
        / "assets"
        / "copied_models"
        / "official_gas"
        / env_stem(env_name)
        / src.name
    )
    copied = copy_file_no_overwrite(src, dst)
    keygraph_source = checkpoint_dir / "keygraph.pkl"
    manifest = {
        "env_name": env_name,
        "epoch": int(epoch),
        "source_root": str(checkpoint_dir),
        "target_root": str(target_root),
        "asset_layout": "official_gas_flat",
        "agent_config_overrides": default_agent_config_overrides(env_name),
        "checkpoints": {
            "official_gas_agent": copied,
        },
        "tdr_path": copied["dst"],
        "policy_path": copied["dst"],
        "official_keygraph_source": str(keygraph_source) if keygraph_source.is_file() else None,
        "keygraph_policy": "rebuild_with_sagas",
    }
    manifest_dir = target_root / "assets" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if manifest_name is None:
        manifest_name = f"{env_stem(env_name)}_assets.json"
    manifest_path = manifest_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=None)
    parser.add_argument(
        "--official-gas-checkpoint-dir",
        default=None,
        help="Flat official GAS checkpoint directory containing params_<epoch>.pkl.",
    )
    parser.add_argument("--target-root", default=".")
    parser.add_argument("--env-name", default="antmaze-giant-stitch-v0")
    parser.add_argument("--epoch", type=int, default=1000000)
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        choices=["EXP_tdr", "EXP_policy"],
        help="Checkpoint family to copy. Repeat to select multiple families.",
    )
    parser.add_argument("--manifest-name", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.official_gas_checkpoint_dir:
        manifest = prepare_official_gas_assets(
            checkpoint_dir=Path(args.official_gas_checkpoint_dir),
            target_root=Path(args.target_root),
            env_name=args.env_name,
            epoch=args.epoch,
            manifest_name=args.manifest_name,
        )
    else:
        if args.source_root is None:
            raise ValueError("--source-root is required unless --official-gas-checkpoint-dir is used")
        manifest = prepare_assets(
            source_root=Path(args.source_root),
            target_root=Path(args.target_root),
            env_name=args.env_name,
            epoch=args.epoch,
            families=args.families,
            manifest_name=args.manifest_name,
        )
    if args.json:
        print(json.dumps(manifest, sort_keys=True))
    else:
        print(f"Prepared assets for {args.env_name}")
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
