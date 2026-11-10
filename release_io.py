"""Small file and asset helpers for the public entrypoint."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent
IMPLEMENTATIONS = ROOT / "implementations"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_path(item, assets_dir, data_dir):
    base = data_dir if item["storage"] == "data" else assets_dir
    rel = Path(item["path"])
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Invalid asset path: {rel}")
    return base / rel


def check_file(path, item):
    if not path.is_file():
        raise FileNotFoundError(f"Missing asset: {path}")
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise ValueError(f"Asset size or SHA-256 differs: {path}")


def fetch_file(path, item, hub=None):
    url = item.get("url")
    if not hub and (not url or not url.startswith("https://")):
        raise FileNotFoundError(f"No download for {path}; see assets/manifest.json and ASSETS.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
            if hub:
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError as exc:
                    raise RuntimeError("Install huggingface_hub to fetch the pinned model assets.") from exc
                cached = hf_hub_download(repo_id=hub["repo_id"], repo_type=hub["repo_type"],
                                         revision=hub["revision"], filename=item["path"])
                with Path(cached).open("rb") as response:
                    shutil.copyfileobj(response, f)
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    shutil.copyfileobj(response, f)
        check_file(tmp, item)
        # Publish only a complete file, without replacing an existing asset.
        os.link(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def verify_assets(ids, files, assets_dir, data_dir, fetch=False, hub=None):
    missing = []
    downloads = []
    selected = set(hub["assets"]) if hub else set()
    revision = hub.get("revision") if hub else None
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub revision must be a full commit SHA, not a branch or tag.")
    for asset_id in sorted(set(ids)):
        item = files[asset_id]
        path = asset_path(item, assets_dir, data_dir)
        if path.exists():
            check_file(path, item)
        elif fetch and asset_id in selected and revision:
            downloads.append((path, item, hub))
        elif fetch and item.get("url"):
            downloads.append((path, item, None))
        else:
            detail = ""
            if fetch and item["storage"] == "assets":
                detail = (" (Hub revision not pinned)" if asset_id in selected
                          else " (not in the pretrained release)")
            missing.append(f"{asset_id}: {path}{detail}")
    if missing:
        raise FileNotFoundError("Missing assets (see ASSETS.md):\n  " + "\n  ".join(missing))
    for path, item, source in downloads:
        fetch_file(path, item, source)
