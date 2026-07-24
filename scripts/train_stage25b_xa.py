#!/usr/bin/env python3
"""Stage 25B paired XA training: B0 (S0 features) vs B1 (selected Stage2 features).

Fixed fresh no-pooling BioCDA-XA. Shared splits/seeds/budget. No TCGA selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _write_cfg(base: dict, feature_dir: str, out_yaml: Path, out_root: str, model_type: str) -> None:
    cfg = json.loads(json.dumps(base))  # deep copy via json
    cfg["data"]["feature_dir"] = feature_dir
    cfg["data"]["forbid_tcga_during_selection"] = True
    cfg["outputs"]["root"] = out_root
    cfg["models"] = [model_type]
    cfg["model"]["type"] = model_type
    # Round25: fresh only, no transfer/KD
    cfg["training"]["predictive_warmstart_gin"] = False
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _run_job(job: dict) -> dict:
    log = Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/train_xa_v2_job.py",
        "--config",
        job["config"],
        "--model-type",
        job["model_type"],
        "--seed",
        str(job["seed"]),
    ]
    if job.get("smoke"):
        cmd.append("--smoke")
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    return {**job, "returncode": proc.returncode, "status": "DONE" if proc.returncode == 0 else "FAIL"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="reports/round25_artifact_lineage.json")
    ap.add_argument("--base-config", default="configs/biocda/xa_v2_closure.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    lineage = json.loads((ROOT / args.lineage).read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / args.base_config).read_text(encoding="utf-8"))
    exports = lineage["exports"]
    b0 = exports["B0_S0"]["feature_dir"]
    b1_key = [k for k in exports if k.startswith("B1_")][0]
    b1 = exports[b1_key]["feature_dir"]
    b2 = exports[b1_key]["z_only_dir"]

    cfg_root = ROOT / "configs/round25"
    out_root = ROOT / "outputs/round25_stage25b"
    jobs = []
    for arm, feat, model in [
        ("B0", b0, "biocda_xa_fresh"),
        ("B1", b1, "biocda_xa_fresh"),
        ("B2", b2, "biocda_xa_z_only"),
    ]:
        for seed in args.seeds:
            cfg_path = cfg_root / f"stage25b_{arm}_seed{seed}.yaml"
            run_out = str((out_root / arm).relative_to(ROOT))
            _write_cfg(base, feat, cfg_path, run_out, model)
            jobs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "config": str(cfg_path.relative_to(ROOT)),
                    "model_type": model,
                    "log": str(out_root / arm / f"seed{seed}_train.log"),
                    "smoke": bool(args.smoke),
                }
            )

    print(f"[25B] launching {len(jobs)} XA jobs parallel={args.max_parallel}")
    results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(_run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[25B] {r['arm']}_seed{r['seed']} -> {r['status']}")

    (ROOT / "reports/round25_stage25b_train_summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
                "b1_export": b1_key,
                "tcga_used_for_selection": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    n_fail = sum(1 for r in results if r["status"] != "DONE")
    if args.strict and n_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
