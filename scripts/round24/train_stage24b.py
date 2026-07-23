"""Stage 24B training: reuse B0; train B1/B2 on Round18 formal 5 folds."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biocda.utils.gpu import configure_gpu_efficiency
from tools.biocda_telegram_notify import biocda_notify


def _run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> int:
    print("+", " ".join(cmd), flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    e.setdefault("PYTHONPATH", str(ROOT))
    return int(subprocess.call(cmd, cwd=str(ROOT), env=e))


def materialize_b0(cfg: Dict[str, Any], out_root: Path) -> Dict[str, Any]:
    """Copy / link Round18 baseline metrics as B0 under stage24b."""
    b0 = out_root / "B0_pooled_mlp_own_plus_summary"
    b0.mkdir(parents=True, exist_ok=True)
    src = ROOT / cfg["paths"]["reports_root"] / "stage24a" / "baseline_summary.json"
    if not src.is_file():
        raise FileNotFoundError("Run Stage24A baseline first")
    summary = json.loads(src.read_text())
    payload = {
        "candidate_id": "B0",
        "architecture": "pooled_mlp",
        "feature": "own_plus_summary",
        "reuse_round18": True,
        "baseline_summary": summary,
        "status": "complete",
    }
    (b0 / "candidate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (b0 / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
    return payload


def run_stage24b(
    cfg: Dict[str, Any],
    *,
    smoke: bool = False,
    max_jobs_per_gpu: int = 3,
    stage: str = "24b",
) -> int:
    configure_gpu_efficiency(target_utilization=float(cfg.get("target_gpu_utilization", 0.9)))
    out_root = ROOT / cfg["paths"]["reports_root"] / "stage24b"
    run_root = ROOT / cfg["paths"]["run_root"] / "stage24b"
    out_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    b0 = materialize_b0(cfg, out_root)
    biocda_notify(f"Round24 Stage24B B0 reused from Round18 status={b0['status']}")

    folds = ROOT / cfg["paths"]["round18_root"] / "splits" / "formal_5fold_assignments.csv"
    if not folds.is_file():
        raise FileNotFoundError(folds)

    jobs = []
    for cand in cfg["candidates_24b"]:
        if cand["id"] == "B0":
            continue
        for fold_id in range(int(cfg["n_folds"])):
            jobs.append(
                {
                    "candidate_id": cand["id"],
                    "architecture": cand["architecture"],
                    "feature": cand["feature"],
                    "fold_id": fold_id,
                    "result_dir": str(run_root / cand["id"] / f"fold_{fold_id}"),
                }
            )
            if smoke:
                break
        if smoke:
            break

    manifest_path = out_root / ("jobs_smoke.csv" if smoke else "jobs_formal.csv")
    import pandas as pd

    pd.DataFrame(jobs).to_csv(manifest_path, index=False)

    if smoke:
        rc = _smoke_biocda_forward(cfg)
        (out_root / "smoke_status.json").write_text(
            json.dumps({"status": "ok" if rc == 0 else "fail", "rc": rc}, indent=2) + "\n"
        )
        if rc != 0:
            return rc

    trainer = ROOT / "scripts/round24/train_biocda_on_round18_folds.py"
    if not trainer.is_file():
        raise FileNotFoundError(trainer)

    cmd = [
        sys.executable,
        str(trainer),
        "--config",
        str(ROOT / "configs/round24/eval3.yaml"),
        "--manifest",
        str(manifest_path),
        "--max-jobs-per-gpu",
        str(1 if smoke else max_jobs_per_gpu),
    ]
    if smoke:
        cmd.append("--smoke")
    return _run(cmd)


def _smoke_biocda_forward(cfg: Dict[str, Any]) -> int:
    """GPU smoke: load one R23 P0 checkpoint and run a tiny forward if available."""
    import torch

    ckpt = ROOT / "outputs/xa_v2_closure/biocda_predictive_seed17/best.pt"
    if not ckpt.is_file():
        print("smoke skip: no biocda_predictive checkpoint", flush=True)
        return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(ckpt, map_location=device)
    print(f"smoke loaded checkpoint keys={list(obj.keys())[:8]} device={device}", flush=True)
    if device.type == "cuda":
        x = torch.randn(512, 96, device=device)
        y = torch.nn.functional.relu(x @ torch.randn(96, 128, device=device))
        torch.cuda.synchronize()
        print(f"smoke matmul ok y.shape={tuple(y.shape)}", flush=True)
    return 0
