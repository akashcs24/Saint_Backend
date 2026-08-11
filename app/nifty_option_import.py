from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings


def _default_source_root() -> Path:
    return settings.price_cache.parent


def _copy_tree_subset(src_root: Path, dst_root: Path) -> dict[str, Any]:
    copied = 0
    skipped = 0
    bytes_copied = 0
    samples: list[str] = []
    patterns = ("NSE_NIFTY*", "NSE_NIFTY50_INDEX*")

    if not src_root.exists():
        return {
            "source": str(src_root),
            "target": str(dst_root),
            "copied": 0,
            "skipped": 0,
            "bytesCopied": 0,
            "missing": True,
            "samples": [],
        }

    for pattern in patterns:
        for src in src_root.rglob(pattern):
            if not src.is_file():
                continue
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue
            shutil.copy2(src, dst)
            copied += 1
            bytes_copied += src.stat().st_size
            if len(samples) < 10:
                samples.append(str(rel))
    return {
        "source": str(src_root),
        "target": str(dst_root),
        "copied": copied,
        "skipped": skipped,
        "bytesCopied": bytes_copied,
        "missing": False,
        "samples": samples,
    }


def import_project_nifty_option_history(source_root: Path | None = None) -> dict[str, Any]:
    source_root = (source_root or _default_source_root()).resolve()
    plan = [
        (source_root / "fyers_1m", settings.nifty_option_1m_dir.resolve()),
        (source_root / "fyers_5m", settings.nifty_option_5m_dir.resolve()),
        (source_root / "fyers_fo", settings.nifty_option_15m_dir.resolve()),
    ]
    results = [_copy_tree_subset(src, dst) for src, dst in plan]
    total_copied = sum(int(r["copied"]) for r in results)
    total_skipped = sum(int(r["skipped"]) for r in results)
    total_bytes = sum(int(r["bytesCopied"]) for r in results)
    manifest = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceRoot": str(source_root),
        "copied": total_copied,
        "skipped": total_skipped,
        "bytesCopied": total_bytes,
        "buckets": results,
    }
    manifest_path = settings.nifty_option_15m_dir.resolve().parent / "nifty_option_import_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifestPath"] = str(manifest_path)
    return manifest