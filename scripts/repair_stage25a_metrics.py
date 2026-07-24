#!/usr/bin/env python3
"""Repair Stage25A metrics from completed checkpoints after post-GAN summary crash."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/optimization_runs/round25_stage25a"
REPORTS = ROOT / "reports"


def _job_row(job_dir: Path) -> dict:
    variant, seed = job_dir.name.split("_seed")
    exps = sorted(job_dir.glob("exp_*"))
    exp = exps[-1] if exps else None
    row = {
        "variant": variant,
        "seed": int(seed),
        "status": "FAIL",
        "elapsed_sec": None,
        "exp_dir": str(exp) if exp else "",
        "best_gan_loss": None,
        "proto_align_loss": None,
        "prototype_hinge_active_fraction": None,
        "source_reconstruction_error": None,
        "target_reconstruction_error": None,
        "mean_target_to_source_anchor_distance": None,
    }
    if exp is None:
        return row
    ckpt = exp / "after_traingan_shared_vae.pth"
    lat = exp / "ccle_latent_dict.pkl"
    if not (ckpt.exists() and lat.exists()):
        return row
    row["status"] = "DONE"
    gm = exp / "gan_metrics.json"
    if gm.exists():
        g = json.loads(gm.read_text(encoding="utf-8"))
        row["best_gan_loss"] = g.get("best_gan_loss") or g.get("best_loss") or g.get("best_eval_loss")
    gcsv = exp / "g_loss.csv"
    if gcsv.exists():
        df = pd.read_csv(gcsv)
        if len(df):
            last = df.iloc[-1].to_dict()
            for k in (
                "proto_align_loss",
                "prototype_hinge_active_fraction",
                "source_reconstruction_error",
                "target_reconstruction_error",
                "mean_target_to_source_anchor_distance",
            ):
                if k in last:
                    row[k] = last[k]
    # margin artifact presence for S2
    mp = exp / "prototype_upper_margins.json"
    if mp.exists():
        art = json.loads(mp.read_text(encoding="utf-8"))
        row["prototype_upper_margins_sha256"] = art.get("sha256")
    return row


def main() -> int:
    rows = []
    for job_dir in sorted(OUT.glob("S*_seed*")):
        if not job_dir.is_dir():
            continue
        rows.append(_job_row(job_dir))
    REPORTS.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS / "round25_stage25a_metrics.csv"
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_jobs": len(rows),
        "n_done": sum(1 for r in rows if r["status"] == "DONE"),
        "n_fail": sum(1 for r in rows if r["status"] != "DONE"),
        "shared_ae": str(OUT / "shared_ae"),
        "metrics_csv": str(out_csv.relative_to(ROOT)),
        "repaired": True,
        "note": "Repaired after post-GAN vicreg summary AttributeError; checkpoints were already saved.",
    }
    (REPORTS / "round25_stage25a_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
