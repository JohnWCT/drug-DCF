#!/usr/bin/env python3
"""Append forced models (baselines / best InfoNCE) to pretrain_top10.csv for finetune."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def augment_top10(top10_path: str, all_candidates_path: str, additions: list) -> pd.DataFrame:
    top10 = pd.read_csv(_resolve(top10_path))
    all_df = (
        pd.read_csv(_resolve(all_candidates_path))
        if os.path.exists(_resolve(all_candidates_path))
        else pd.DataFrame()
    )

    for spec in additions:
        model_id = spec["ID"]
        if model_id in set(top10["ID"].astype(str)):
            if spec.get("pretrain_result_dir"):
                top10.loc[top10["ID"].astype(str) == model_id, "pretrain_result_dir"] = spec["pretrain_result_dir"]
            continue

        if not all_df.empty and model_id in set(all_df["ID"].astype(str)):
            row = all_df[all_df["ID"].astype(str) == model_id].iloc[0].to_dict()
        else:
            row = {"ID": model_id}

        row = dict(row)
        row["ID"] = model_id
        if spec.get("pretrain_result_dir"):
            row["pretrain_result_dir"] = spec["pretrain_result_dir"]
            row["result_folder"] = spec["pretrain_result_dir"]
        if spec.get("role"):
            row["finetune_role"] = spec["role"]
        if spec.get("lambda_proto") is not None:
            row["lambda_proto"] = spec["lambda_proto"]
        row["is_control"] = spec.get("is_control", row.get("lambda_proto", 0) == 0)
        top10 = pd.concat([top10, pd.DataFrame([row])], ignore_index=True)

    top10 = top10.drop_duplicates(subset=["ID"], keep="first")
    top10["selection_rank"] = range(1, len(top10) + 1)
    top10.to_csv(_resolve(top10_path), index=False)
    return top10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top10", required=True)
    parser.add_argument("--all-candidates", required=True)
    parser.add_argument("--append-baseline-exp746", action="store_true")
    parser.add_argument("--append-baseline-exp018", action="store_true")
    parser.add_argument("--append-best-infonce-exp045", action="store_true")
    args = parser.parse_args()

    additions = []
    if args.append_baseline_exp746:
        additions.append({
            "ID": "exp_746",
            "pretrain_result_dir": "result/pretrain_vaewc/exp_746",
            "role": "historical_baseline",
            "lambda_proto": 0,
            "is_control": True,
        })
    if args.append_baseline_exp018:
        additions.append({
            "ID": "exp_018",
            "pretrain_result_dir": (
                "result/optimization_runs/vaewc_proto_infonce_round3_exp746/pretrain/exp_018"
            ),
            "role": "round3_control_baseline",
            "lambda_proto": 0,
            "is_control": True,
        })
    if args.append_best_infonce_exp045:
        additions.append({
            "ID": "exp_045",
            "pretrain_result_dir": (
                "result/optimization_runs/vaewc_round4_1_t2s_infonce_collapse_guard/pretrain/exp_045"
            ),
            "role": "round4_1_best_infonce",
            "lambda_proto": 0.001,
            "is_control": False,
        })

    df = augment_top10(args.top10, args.all_candidates, additions)
    print(f"Augmented top10: {len(df)} models -> {args.top10}")
    for _, r in df.iterrows():
        pdir = r.get("pretrain_result_dir", r.get("result_folder", f"pretrain/{r['ID']}"))
        print(f"  {r['ID']} pretrain={pdir}")


if __name__ == "__main__":
    main()
