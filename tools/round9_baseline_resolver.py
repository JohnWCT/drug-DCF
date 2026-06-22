#!/usr/bin/env python3
"""Resolve Round 9 baseline checkpoints from result tree and known hints."""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.optimization_selection import DEFAULT_FORCE_BASELINE_PATHS
from tools.round9_diagnostics_common import (
    KNOWN_BASELINE_HINTS,
    RESOLVED_BASELINE_COLUMNS,
    build_resolved_row,
    checkpoint_completeness_score,
    load_json,
    relpath_from_root,
    resolve_path,
    write_csv,
    write_md,
)


def _candidate_dirs(exp_id: str, search_root: str, explicit_path: Optional[str]) -> List[str]:
    candidates: List[str] = []
    seen = set()

    def add(path: Optional[str]) -> None:
        if not path:
            return
        full = resolve_path(path)
        if os.path.isdir(full) and full not in seen:
            seen.add(full)
            candidates.append(full)

    add(explicit_path)
    add(KNOWN_BASELINE_HINTS.get(exp_id))
    add(DEFAULT_FORCE_BASELINE_PATHS.get(exp_id))

    patterns = [
        os.path.join(search_root, "optimization_runs", "**", "pretrain", exp_id),
        os.path.join(search_root, "optimization_runs", "**", "pretrain_results", exp_id),
        os.path.join(search_root, "pretrain*", exp_id),
        os.path.join(search_root, "**", "pretrain", exp_id),
    ]
    for pattern in patterns:
        for path in glob(resolve_path(pattern), recursive=True):
            add(path)

    for csv_path in glob(resolve_path(os.path.join(search_root, "**", "model_select.csv")), recursive=True):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        id_col = "ID" if "ID" in df.columns else None
        folder_col = "result_folder" if "result_folder" in df.columns else None
        if not id_col or not folder_col:
            continue
        match = df[df[id_col].astype(str) == exp_id]
        if not match.empty:
            add(str(match.iloc[0][folder_col]))

    for csv_path in glob(resolve_path(os.path.join(search_root, "**", "aggregate_scores.csv")), recursive=True):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "Model_ID" not in df.columns:
            continue
        match = df[df["Model_ID"].astype(str) == exp_id]
        if match.empty:
            continue
        for col in ("pretrain_result_dir", "result_folder", "checkpoint_dir"):
            if col in match.columns and pd.notna(match.iloc[0][col]):
                add(str(match.iloc[0][col]))
    return candidates


def _pick_best_candidate(exp_id: str, candidates: List[str]) -> Tuple[Optional[str], List[str]]:
    if not candidates:
        return None, []
    hint = KNOWN_BASELINE_HINTS.get(exp_id) or DEFAULT_FORCE_BASELINE_PATHS.get(exp_id)
    if hint:
        hint_full = resolve_path(hint)
        if hint_full in candidates and checkpoint_completeness_score(hint_full) > 0:
            return hint_full, []
    scored = sorted(
        ((checkpoint_completeness_score(c), c) for c in candidates),
        key=lambda x: (-x[0], x[1]),
    )
    best_score = scored[0][0]
    best = [c for s, c in scored if s == best_score]
    if len(best) == 1:
        return best[0], []
    return best[0], [relpath_from_root(c) for c in best]


def resolve_baselines(
    baseline_config_path: str,
    search_root: str = "result",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    spec = load_json(baseline_config_path)
    resolved_rows: List[dict] = []
    missing_rows: List[dict] = []
    ambiguous_rows: List[dict] = []
    exit_code = 0

    for item in spec.get("baselines", []):
        exp_id = item["exp_id"]
        role = item.get("role", "")
        required = bool(item.get("required", False))
        explicit_path = item.get("explicit_path")
        candidates = _candidate_dirs(exp_id, search_root, explicit_path)
        best, ambiguous = _pick_best_candidate(exp_id, candidates)
        if best is None:
            row = {
                "exp_id": exp_id,
                "role": role,
                "required": required,
                "notes": "not found",
            }
            if required:
                exit_code = 2
            missing_rows.append(row)
            continue
        if ambiguous:
            for cand in ambiguous:
                ambiguous_rows.append(
                    {
                        "exp_id": exp_id,
                        "candidate_checkpoint_dir": cand,
                        "completeness_score": checkpoint_completeness_score(resolve_path(cand)),
                    }
                )
        resolved_rows.append(build_resolved_row(exp_id, role, required, best))

    resolved_df = pd.DataFrame(resolved_rows, columns=RESOLVED_BASELINE_COLUMNS)
    missing_df = pd.DataFrame(missing_rows)
    ambiguous_df = pd.DataFrame(ambiguous_rows)
    return resolved_df, missing_df, ambiguous_df, exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve Round 9 baseline checkpoints")
    parser.add_argument("--baseline-config", default="config/round9_baselines.json")
    parser.add_argument("--search-root", default="result")
    parser.add_argument("--outdir", default="result/optimization_runs/round9_diagnostics/baselines")
    args = parser.parse_args()

    outdir = resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    resolved_df, missing_df, ambiguous_df, exit_code = resolve_baselines(
        args.baseline_config, args.search_root
    )

    write_csv(resolved_df, os.path.join(outdir, "resolved_baselines.csv"))
    write_csv(
        missing_df if not missing_df.empty else pd.DataFrame(columns=["exp_id", "role", "required", "notes"]),
        os.path.join(outdir, "missing_baselines.csv"),
    )
    write_csv(
        ambiguous_df if not ambiguous_df.empty else pd.DataFrame(columns=["exp_id", "candidate_checkpoint_dir", "completeness_score"]),
        os.path.join(outdir, "ambiguous_baseline_candidates.csv"),
    )

    lines = [
        "# Round 9 Baseline Resolution Report",
        "",
        f"- Resolved: **{len(resolved_df)}**",
        f"- Missing: **{len(missing_df)}**",
        f"- Ambiguous candidates logged: **{len(ambiguous_df)}**",
        "",
    ]
    if not missing_df.empty:
        lines.append("## Missing baselines")
        for _, row in missing_df.iterrows():
            req = "required" if row.get("required") else "optional"
            lines.append(f"- `{row['exp_id']}` ({req})")
        lines.append("")
    if not ambiguous_df.empty:
        lines.append("## Ambiguous candidates")
        for _, row in ambiguous_df.iterrows():
            lines.append(f"- `{row['exp_id']}` → `{row['candidate_checkpoint_dir']}`")
        lines.append("")
    write_md(os.path.join(outdir, "resolution_report.md"), lines)
    print(f"Wrote {outdir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
