"""Stage 24C feature coverage / attribution helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FEATURE_SPECS = [
    {"id": "F0", "name": "own_plus_summary", "path": "result/optimization_runs/round17r_18class/features/r13_exp_008/own_plus_summary", "expected_dim": 75},
    {"id": "F1", "name": "z_plus_summary", "path": "result/optimization_runs/round19_factorial/features/z_plus_summary", "expected_dim": 75},
    {"id": "F2", "name": "z_plus_context16", "path": "result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context16", "expected_dim": 80},
    {"id": "F3", "name": "z_plus_context32", "path": "result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context32", "expected_dim": 96},
    {"id": "F4", "name": "z_plus_summary_context16", "path": "result/optimization_runs/round19_factorial/features/z_plus_summary_context16", "expected_dim": 91},
]


def run_feature_coverage(cfg: Dict[str, Any], out_dir: Path) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in FEATURE_SPECS:
        p = ROOT / spec["path"]
        meta_path = p / "feature_metadata.json"
        names_path = p / "feature_names.json"
        ccle = p / "ccle_latent_proto.pkl"
        tcga = p / "tcga_latent_proto.pkl"
        dim = None
        includes_summary = None
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            dim = meta.get("response_input_dim") or meta.get("feature_dim")
            includes_summary = meta.get("includes_own_plus_summary")
        elif names_path.is_file():
            dim = len(json.loads(names_path.read_text()))
        rows.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "path": str(p),
                "exists": p.is_dir(),
                "expected_dim": spec["expected_dim"],
                "observed_dim": dim,
                "dim_match": dim == spec["expected_dim"] if dim is not None else False,
                "includes_own_plus_summary": includes_summary,
                "has_ccle_pkl": ccle.is_file(),
                "has_tcga_pkl": tcga.is_file(),
                "note": (
                    "plan listed own_plus_summary as 86-d; Round17R artifact is 75-d (64+11). "
                    "Use observed_dim for contracts."
                    if spec["id"] == "F0"
                    else ""
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "feature_coverage.csv", index=False)
    (out_dir / "feature_coverage.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def run_feature_attribution_stub(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PENDING_STAGE24B",
        "message": "Full F0-F4 attribution trains after Stage24B gate decides feature experiments are needed.",
    }
    (out_dir / "feature_attribution_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
