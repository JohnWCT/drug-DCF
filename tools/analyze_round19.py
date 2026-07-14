#!/usr/bin/env python3
"""Round 19 analyzer stubs (structure / effect templates for smoke)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round19_manifest_validator import assert_selection_frame_has_no_tcga
from tools.round19_selection_lock import scan_mapping_for_forbidden


EFFECT_REPORTS_19B = [
    "round19b_architecture_ranking.csv",
    "round19b_node_capacity_effect.csv",
    "round19b_graph_bottleneck_effect.csv",
    "round19b_bond_content_effect.csv",
    "round19b_predictor_integration_effect.csv",
    "round19b_context_dependency.csv",
    "round19b_resource_summary.csv",
]


def write_empty_effect_templates(outdir: str) -> dict:
    reports = Path(outdir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    written = []
    for name in EFFECT_REPORTS_19B:
        path = reports / name
        if not path.is_file():
            pd.DataFrame(columns=["architecture_id", "mean_DrugMacro_AUC"]).to_csv(path, index=False)
        df = pd.read_csv(path)
        assert_selection_frame_has_no_tcga(df)
        written.append(str(path))
    return {"templates": written}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="19b", choices=["19b", "selection"])
    parser.add_argument("--outdir", default="result/optimization_runs/round19_factorial")
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()
    if args.stage == "19b":
        out = write_empty_effect_templates(args.outdir)
        print(json.dumps(out, indent=2))
        return
    if args.stage == "selection":
        if not args.write_lock:
            raise SystemExit("selection stage requires --write-lock after 19B/19C complete")
        # Smoke-safe: refuse to write a real lock without ranking inputs.
        raise SystemExit("Refuse lock: Round 19B/19C results not complete (smoke guard)")
    raise SystemExit(f"Unsupported stage {args.stage}")


if __name__ == "__main__":
    main()
