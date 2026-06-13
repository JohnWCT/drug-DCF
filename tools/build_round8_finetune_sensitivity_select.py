#!/usr/bin/env python3
"""Build Round 8 second-pass finetune sensitivity model_select.csv from first-pass aggregate."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _pick_top_models(df: pd.DataFrame, col: str, n: int) -> List[str]:
    if df.empty or col not in df.columns:
        return []
    ranked = df.sort_values(col, ascending=False, na_position="last")
    return [str(x) for x in ranked["Model_ID"].head(n).astype(str).tolist()]


def _best_vicreg(selection_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> Optional[str]:
    if selection_df.empty:
        return None
    col = "round8_vicreg_active" if "round8_vicreg_active" in selection_df.columns else None
    if col:
        vicreg_ids = set(selection_df[selection_df[col].fillna(False)]["ID"].astype(str))
        sub = aggregate_df[aggregate_df["Model_ID"].astype(str).isin(vicreg_ids)]
        picks = _pick_top_models(sub, "Average_TCGA_AUC_mean", 1)
        return picks[0] if picks else None
    return None


def _best_control(selection_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> Optional[str]:
    if selection_df.empty:
        return None
    col = "round8_control_like" if "round8_control_like" in selection_df.columns else None
    if col:
        control_ids = set(selection_df[selection_df[col].fillna(False)]["ID"].astype(str))
        sub = aggregate_df[aggregate_df["Model_ID"].astype(str).isin(control_ids)]
        picks = _pick_top_models(sub, "Average_TCGA_AUC_mean", 1)
        return picks[0] if picks else None
    return None


def build_sensitivity_select(
    aggregate_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    force_models: Optional[Iterable[str]] = None,
    max_models: int = 12,
) -> pd.DataFrame:
    force_models = list(force_models or [])
    if aggregate_df.empty:
        raise ValueError("aggregate_scores.csv is empty")

    agg = aggregate_df.copy()
    if "Model_ID" not in agg.columns:
        raise ValueError("aggregate_scores.csv must contain Model_ID column")

    chosen: List[str] = []
    chosen.extend(_pick_top_models(agg, "Average_TCGA_AUC_mean", 5))
    chosen.extend(_pick_top_models(agg, "Global_TCGA_AUC_mean", 3))

    best_vicreg = _best_vicreg(selection_df, agg)
    if best_vicreg:
        chosen.append(best_vicreg)
    best_control = _best_control(selection_df, agg)
    if best_control:
        chosen.append(best_control)
    chosen.extend(force_models)

    deduped: List[str] = []
    for mid in chosen:
        mid = str(mid).strip()
        if mid and mid not in deduped:
            deduped.append(mid)
        if len(deduped) >= max_models:
            break

    if selection_df.empty:
        out = pd.DataFrame({"ID": deduped})
        out["selection_rank"] = range(1, len(out) + 1)
        return out

    sel = selection_df.copy()
    id_col = "ID" if "ID" in sel.columns else "Model_ID"
    sel_map = sel.set_index(sel[id_col].astype(str))
    rows = []
    for rank, mid in enumerate(deduped, start=1):
        if mid in sel_map.index:
            row = sel_map.loc[mid].to_dict()
            row["ID"] = mid
        else:
            sub = agg[agg["Model_ID"].astype(str) == mid]
            row = {"ID": mid}
            if not sub.empty:
                row["Average_TCGA_AUC_mean"] = sub.iloc[0].get("Average_TCGA_AUC_mean")
        row["selection_rank"] = rank
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Round 8 finetune sensitivity model_select.csv")
    parser.add_argument(
        "--aggregate",
        default="result/optimization_runs/round8_combined/aggregate/aggregate_scores.csv",
    )
    parser.add_argument(
        "--selection",
        default="result/optimization_runs/round8_combined/selection/pretrain_top10.csv",
    )
    parser.add_argument(
        "--outdir",
        default="result/optimization_runs/round8_finetune_sensitivity/selection",
    )
    parser.add_argument("--max-models", type=int, default=12)
    parser.add_argument(
        "--force-models",
        default="exp_048,exp_021,exp_746",
        help="Comma-separated model IDs to force include",
    )
    args = parser.parse_args(argv)

    aggregate_path = _resolve(args.aggregate)
    selection_path = _resolve(args.selection)
    out_dir = _resolve(args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    agg = pd.read_csv(aggregate_path)
    sel = pd.read_csv(selection_path) if os.path.exists(selection_path) else pd.DataFrame()
    force_models = [x.strip() for x in args.force_models.split(",") if x.strip()]
    out = build_sensitivity_select(agg, sel, force_models=force_models, max_models=args.max_models)
    out_path = os.path.join(out_dir, "model_select.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
