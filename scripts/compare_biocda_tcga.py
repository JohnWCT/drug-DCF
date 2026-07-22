#!/usr/bin/env python3
"""Compare TCGA metrics (AUC + AUPRC) across all BioCDA-tested models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from biocda.validation.tcga_benchmark import (
    discover_model_specs,
    prepare_tcga_frames,
    results_to_long_df,
    results_to_wide_markdown,
    run_model_tcga,
    _resolve_round20_lock,
)
from biocda.utils.gpu import configure_gpu_efficiency
from tools.biocda_telegram_notify import biocda_notify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/biocda_tcga_comparison",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--smoke", action="store_true", help="Run only first non-import model")
    args = parser.parse_args()

    configure_gpu_efficiency(target_utilization=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = discover_model_specs()
    if not specs:
        print("No model checkpoints found", file=sys.stderr)
        return 1

    biocda_notify(f"BioCDA TCGA compare START n_models={len(specs)} device={device}")

    lock = _resolve_round20_lock()
    frames, patient_latent = prepare_tcga_frames(lock)
    (out_dir / "tcga_frame_sizes.json").write_text(
        json.dumps({k: len(v) for k, v in frames.items()}, indent=2) + "\n",
        encoding="utf-8",
    )

    run_specs = specs[:2] if args.smoke else specs
    results = []
    for i, spec in enumerate(run_specs):
        print(f"[{i+1}/{len(run_specs)}] {spec.display_name} ({spec.source})", flush=True)
        res = run_model_tcga(
            spec,
            frames=frames,
            patient_latent=patient_latent,
            device=device,
            output_dir=out_dir,
        )
        results.append(res)
        means = res.mean_metrics()
        biocda_notify(
            f"BioCDA TCGA done {spec.model_id}\n"
            f"mean DrugMacro AUC={means.get('mean_drug_macro_auc'):.4f} "
            f"AUPRC={means.get('mean_drug_macro_auprc'):.4f}"
        )

    long_df = results_to_long_df(results)
    long_df.to_csv(out_dir / "biocda_tcga_comparison_long.csv", index=False)

    summary_rows = []
    for res in results:
        means = res.mean_metrics()
        summary_rows.append(
            {
                "model_id": res.spec.model_id,
                "display_name": res.spec.display_name,
                "round": res.spec.round_tag,
                "architecture": res.spec.architecture,
                "ensemble": res.spec.notes,
                **{k.replace("mean_", "mean_"): v for k, v in means.items()},
            }
        )
    summary = sorted(summary_rows, key=lambda r: r.get("mean_drug_macro_auc") or -1, reverse=True)
    (out_dir / "biocda_tcga_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md = results_to_wide_markdown(long_df)
    (out_dir / "biocda_tcga_comparison.md").write_text(md + "\n", encoding="utf-8")
    (ROOT / "docs/biocda_tcga_comparison.md").write_text(md + "\n", encoding="utf-8")

    biocda_notify(
        f"BioCDA TCGA compare COMPLETE\n"
        f"models={len(results)} out={out_dir}\n"
        f"top={summary[0]['display_name'] if summary else 'n/a'}"
    )
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
