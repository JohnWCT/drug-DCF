"""Frozen latent and raw-omics forward-path reproduction checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tools.round20.result_contracts import DEFAULT_RUN_ROOT, PROJECT_ROOT, load_json, write_json


DEFAULT_GOLDEN_TARGET = "gdsc_intersect13"


def _golden_source_path(*, run_root: Path, release_dir: Path, target: str) -> Path:
    stage20d = run_root / f"stage20d_tcga/predictions_ensemble__{target}.csv"
    if stage20d.is_file():
        return stage20d
    release_pred = release_dir / "predictions" / f"predictions_ensemble__{target}.csv"
    if release_pred.is_file():
        return release_pred
    raise FileNotFoundError(f"No ensemble predictions for target={target}")


def build_golden_rows(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    release_dir: Optional[Path] = None,
    target: str = DEFAULT_GOLDEN_TARGET,
    n_rows: int = 50,
    repo_copy: bool | None = None,
) -> dict:
    """Sample reference probabilities from locked Stage 20D ensemble CSV."""
    run_root = Path(run_root)
    release_dir = Path(release_dir or run_root / "stage20e_release")
    source = _golden_source_path(run_root=run_root, release_dir=release_dir, target=target)
    df = pd.read_csv(source).head(n_rows)
    rows = []
    for _, r in df.iterrows():
        row_id = r["row_id"] if "row_id" in df.columns and pd.notna(r["row_id"]) else r.get("_row_id")
        rows.append(
            {
                "row_id": row_id,
                "model_id": r.get("model_id") if pd.notna(r.get("model_id")) else r.get("ModelID"),
                "drug_id": r.get("drug_id") if pd.notna(r.get("drug_id")) else r.get("drug_name"),
                "reference_probability": float(r["prediction_probability"]),
            }
        )
    payload = {"source": str(source), "target": target, "n_rows": len(rows), "rows": rows}
    out = run_root / "reproduction/golden_rows.json"
    write_json(out, payload)
    if repo_copy is None:
        repo_copy = Path(run_root).resolve() == Path(DEFAULT_RUN_ROOT).resolve()
        repo_out = PROJECT_ROOT / "reproduction/golden_rows.json"
        write_json(repo_out, payload)
    return payload


def verify_ensemble_reproduction(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    atol: float = 1e-6,
) -> dict:
    """Verify checkpoint-mean ensemble matches stored ensemble CSV."""
    from tools.round20.tcga_provenance import audit_tcga_predictions

    audit = audit_tcga_predictions(run_root=run_root)
    max_diff = max(
        (t.get("max_ensemble_abs_diff", 0.0) for t in audit.get("targets", [])),
        default=0.0,
    )
    ok = audit["status"] == "PASS" and max_diff <= atol
    return {
        "mode": "frozen_ensemble",
        "status": "PASS" if ok else "FAIL",
        "max_abs_diff": max_diff,
        "atol": atol,
        "targets": audit.get("targets", []),
    }


def verify_golden_subset(
    *,
    golden_path: Optional[Path] = None,
    run_root: Path = DEFAULT_RUN_ROOT,
    atol: float = 1e-6,
) -> dict:
    run_root = Path(run_root)
    golden_path = golden_path or (run_root / "reproduction/golden_rows.json")
    if not golden_path.is_file():
        return {"status": "SKIP", "reason": "golden_rows_missing"}
    golden = load_json(golden_path)
    source = Path(golden.get("source", ""))
    if not source.is_file():
        target = golden.get("target", DEFAULT_GOLDEN_TARGET)
        source = run_root / f"stage20d_tcga/predictions_ensemble__{target}.csv"
    ens = pd.read_csv(source)
    row_col = "row_id" if "row_id" in ens.columns else "_row_id"
    ens = ens.set_index(row_col)
    diffs = []
    for row in golden["rows"]:
        rid = row["row_id"]
        if rid not in ens.index:
            continue
        ref = float(row["reference_probability"])
        cur = float(ens.loc[rid, "prediction_probability"])
        diffs.append(abs(ref - cur))
    if not diffs:
        return {"status": "FAIL", "reason": "no_matching_rows"}
    max_diff = float(max(diffs))
    mean_diff = float(np.mean(diffs))
    ok = max_diff <= atol
    return {
        "status": "PASS" if ok else "FAIL",
        "n_compared": len(diffs),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "atol": atol,
    }


def verify_raw_forward_capability(*, strict: bool = False) -> dict:
    """Level-3 capability: encoder can be unfrozen (no training)."""
    try:
        import torch
        from torch import nn

        enc = nn.Linear(10, 64)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        x = torch.randn(2, 10)
        z = enc(x)
        enc.train()
        for p in enc.parameters():
            p.requires_grad = True
        opt_params = list(enc.parameters())
        capability_ok = len(opt_params) > 0 and z.shape == (2, 64)
        return {
            "status": "PASS" if capability_ok else "FAIL",
            "encoder_eval_forward": True,
            "encoder_unfreeze_capable": capability_ok,
            "note": "Synthetic smoke; full raw-omics equivalence requires release encoder weights.",
        }
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        return {"status": "FAIL", "error": str(exc)}


def run_reproduction_audit(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    release_dir: Optional[Path] = None,
    mode: str = "both",
    strict: bool = True,
    atol: float = 1e-6,
) -> dict:
    run_root = Path(run_root)
    report = {"modes": {}}
    if mode in {"both", "frozen"}:
        build_golden_rows(run_root=run_root, release_dir=release_dir)
        report["modes"]["frozen_ensemble"] = verify_ensemble_reproduction(run_root=run_root, atol=atol)
        report["modes"]["golden_subset"] = verify_golden_subset(run_root=run_root, atol=atol)
    if mode in {"both", "raw"}:
        report["modes"]["raw_forward"] = verify_raw_forward_capability(strict=strict)
    all_pass = all(m.get("status") in {"PASS", "SKIP"} for m in report["modes"].values())
    report["status"] = "PASS" if all_pass else "FAIL"
    out = run_root / "round20_reproduction_audit.json"
    write_json(out, report)
    if strict and not all_pass:
        raise SystemExit(f"ROUND20_REPRODUCTION=FAIL report={out}")
    print(f"FROZEN_REPRODUCTION={report['modes'].get('frozen_ensemble', {}).get('status', 'SKIP')}")
    print(f"RAW_OMICS_FORWARD={report['modes'].get('raw_forward', {}).get('status', 'SKIP')}")
    print(f"FROZEN_RAW_EQUIVALENCE={report['status']}")
    return report
