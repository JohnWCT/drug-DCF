"""Repair pretrain manifest from logs/status when manifest rows were reset."""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repair_manifest(run_dir: str) -> str:
    run_dir = os.path.join(PROJECT_ROOT, run_dir) if not os.path.isabs(run_dir) else run_dir
    manifest_path = os.path.join(run_dir, "manifests", "pretrain_sweep_manifest.csv")
    df = pd.read_csv(manifest_path, dtype={"result_dir": str, "error_message": str, "start_time": str, "end_time": str})
    df["result_dir"] = df["result_dir"].fillna("").astype(str).replace("nan", "")
    logs_dir = os.path.join(run_dir, "logs", "pretrain")
    status_dir = os.path.join(run_dir, "status", "pretrain")

    for idx, row in df.iterrows():
        job_id = row["job_id"]
        if str(row.get("status")) == "success" and str(row.get("result_dir", "")).strip():
            continue
        status_json = os.path.join(status_dir, f"{job_id}_status.json")
        if os.path.exists(status_json):
            with open(status_json, encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("return_code") == 0 and payload.get("result_dir"):
                df.at[idx, "status"] = "success"
                df.at[idx, "result_dir"] = os.path.relpath(payload["result_dir"], PROJECT_ROOT)
                continue
        log_path = os.path.join(logs_dir, f"{job_id}.log")
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if "All experiments done" in text:
                m = re.search(r"start experiment (exp_\d+)", text)
                if m:
                    exp_name = m.group(1)
                    df.at[idx, "status"] = "success"
                    df.at[idx, "result_dir"] = os.path.relpath(
                        os.path.join(run_dir, "pretrain", exp_name), PROJECT_ROOT
                    )

    df.to_csv(manifest_path, index=False)
    print(df["status"].value_counts().to_dict())
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="result/optimization_runs/vaewc_proto_infonce_round1")
    args = parser.parse_args()
    repair_manifest(args.run_dir)
