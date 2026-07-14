#!/usr/bin/env python3
"""Round 19 config / stage 19A setup helpers."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round19_feature_builder import build_round19_feature_set
from tools.round19_fusion_models import COMPATIBLE_CELLS, assert_compatible
from tools.round19_graph_features import (
    BOND_FEATURE_DIM,
    cache_metadata,
    ensure_cache_dir,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_git_baseline(outdir: Path) -> Dict[str, Any]:
    meta_dir = Path(outdir) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    def _run(cmd: List[str]) -> str:
        # Avoid mutating git config; allow reading a host-mounted repo inside Docker.
        full = ["git", "-c", "safe.directory=*"] + cmd[1:] if cmd and cmd[0] == "git" else cmd
        return subprocess.check_output(full, cwd=str(Path.cwd()), text=True).strip()

    head = os.environ.get("ROUND19_GIT_HEAD", "").strip()
    origin = os.environ.get("ROUND19_GIT_ORIGIN_MAIN", "").strip()
    dirty = os.environ.get("ROUND19_GIT_STATUS", "")
    try:
        if not head:
            head = _run(["git", "rev-parse", "HEAD"])
        if not origin:
            try:
                origin = _run(["git", "rev-parse", "origin/main"])
            except subprocess.CalledProcessError:
                origin = ""
        if dirty == "":
            dirty = _run(["git", "status", "--porcelain"])
    except subprocess.CalledProcessError as exc:
        if not head:
            head_file = Path(".git/HEAD")
            if head_file.is_file():
                ref = head_file.read_text().strip()
                if ref.startswith("ref:"):
                    ref_path = Path(".git") / ref.split(" ", 1)[1].strip()
                    head = ref_path.read_text().strip() if ref_path.is_file() else "UNKNOWN"
                else:
                    head = ref
            else:
                head = "UNKNOWN"
        dirty = dirty or f"git_unavailable: {exc}"
    payload = {
        "round18e_commit": head,
        "round19_start_commit": head,
        "origin_main": origin,
        "head_equals_origin_main": bool(origin) and head == origin,
        "working_tree_clean": dirty == "",
        "round18e_external_success": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dirty_sample": dirty.splitlines()[:20],
    }
    path = meta_dir / "round19_baseline_git.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_stage19a(settings: dict, outdir: str) -> Dict[str, Any]:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    git_meta = write_git_baseline(root)

    feature_root = Path(settings["feature_root"]) / settings["feature_model_key"]
    feat_rep = build_round19_feature_set(
        feature_root=str(feature_root),
        out_root=settings.get("round19_feature_out_root", str(root / "features")),
    )

    cache_root = root / "cache"
    gin_meta = cache_metadata(
        encoder_type="gin",
        atom_feature_dim=78,
        bond_feature_dim=None,
        cache_version="round19_gin_atom78_v1",
    )
    gine_meta = cache_metadata(
        encoder_type="gine",
        atom_feature_dim=78,
        bond_feature_dim=BOND_FEATURE_DIM,
        cache_version="round19_gine_v1",
    )
    gin_cache = ensure_cache_dir(cache_root, "gin_atom78_v1", gin_meta)
    gine_cache = ensure_cache_dir(cache_root, "gine_atom78_bond_v1", gine_meta)

    # Validate compatibility table matches settings
    cells = [tuple(x) for x in settings.get("compatible_cells", COMPATIBLE_CELLS)]
    for d, p in cells:
        assert_compatible(d, p)
    if sorted(cells) != sorted(COMPATIBLE_CELLS):
        raise RuntimeError("settings.compatible_cells != code COMPATIBLE_CELLS")

    report = {
        "stage": "19a",
        "git": git_meta,
        "features": feat_rep,
        "cache": {"gin": str(gin_cache), "gine": str(gine_cache)},
        "n_compatible_cells": len(COMPATIBLE_CELLS),
        "smoke_cells": settings.get("stage19a_smoke_cells", []),
    }
    (root / "reports").mkdir(parents=True, exist_ok=True)
    path = root / "reports" / "round19a_setup_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19 config builder")
    parser.add_argument("--settings", default="config/round19_factorial_settings.json")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--stage", default="19a", choices=["19a"])
    args = parser.parse_args()
    settings = _load_json(Path(args.settings))
    outdir = args.outdir or settings.get("outdir", "result/optimization_runs/round19_factorial")
    if args.stage == "19a":
        rep = build_stage19a(settings, outdir)
        print(json.dumps(rep, indent=2, default=str))
    else:
        raise SystemExit(f"Unsupported stage {args.stage}")


if __name__ == "__main__":
    main()
