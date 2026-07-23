#!/usr/bin/env python3
"""Round 24 unified CLI — eval3 TCGA recovery."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biocda.utils.gpu import configure_gpu_efficiency
from biocda.validation.round24_gate import (
    build_lock_manifest,
    write_lock_manifest,
)
from biocda.validation.round24_protocol import (
    load_eval3_config,
    rebuild_round18_baseline,
    run_preflight,
)
from tools.biocda_telegram_notify import biocda_notify


def _out_stage(cfg, stage: str) -> Path:
    return ROOT / cfg["paths"]["reports_root"] / stage


def cmd_preflight(args) -> int:
    cfg = load_eval3_config(args.config)
    configure_gpu_efficiency(target_utilization=float(cfg.get("target_gpu_utilization", 0.9)))
    out = _out_stage(cfg, "stage24a")
    biocda_notify("Round24 Stage24A preflight START")
    manifest = run_preflight(cfg, out)
    status = manifest["status"]
    biocda_notify(
        f"Round24 Stage24A preflight {status}\n"
        f"gdsc raw={manifest['gdsc_intersect13_note']['headline_pairs']} "
        f"eligible={manifest['gdsc_intersect13_note']['eligible_pairs']} "
        f"dropped={manifest['gdsc_intersect13_note']['dropped']}\n"
        f"blockers={len(manifest['blockers'])} out={out}"
    )
    print(json.dumps({"status": status, "blockers": manifest["blockers"]}, indent=2))
    return 0 if status == "PASS" else 2


def cmd_baseline(args) -> int:
    cfg = load_eval3_config(args.config)
    out = _out_stage(cfg, "stage24a")
    if args.smoke:
        from biocda.validation.round24_protocol import metrics_from_predictions
        import pandas as pd

        r18 = ROOT / cfg["paths"]["round18_root"] / "reports"
        arch = cfg["baseline"]["architecture_id"]
        ens = pd.read_csv(r18 / f"round18e_tcga_{arch}_gdsc_intersect13_ensemble_predictions.csv")
        m = metrics_from_predictions(ens, cfg)
        print(json.dumps({"smoke": True, "gdsc_intersect13_ensemble": m}, indent=2, default=str))
        biocda_notify(f"Round24 baseline SMOKE AUC={m['DrugMacro_AUC']:.4f}")
        return 0

    biocda_notify("Round24 Stage24A baseline START")
    if not (out / "eval3_manifest.json").is_file():
        run_preflight(cfg, out)
    summary = rebuild_round18_baseline(cfg, out)
    n_pass = sum(1 for v in summary["vs_gate"].values() if v["pass"])
    biocda_notify(
        f"Round24 Stage24A baseline DONE\n"
        f"targets_pass_gate={n_pass}/5\n"
        f"out={out}"
    )
    print(json.dumps({"targets_pass_gate": n_pass, "vs_gate": summary["vs_gate"]}, indent=2))
    return 0


def cmd_protocol(args) -> int:
    rc = cmd_preflight(args)
    if rc != 0:
        return rc
    return cmd_baseline(args)


def cmd_diagnose(args) -> int:
    from scripts.round24.diagnose_gdsc_intersect13 import run_diagnose

    cfg = load_eval3_config(args.config)
    out = _out_stage(cfg, "stage24d")
    biocda_notify("Round24 Stage24D diagnose START")
    report = run_diagnose(cfg, out, target=args.target)
    biocda_notify(f"Round24 Stage24D diagnose DONE out={out}")
    print(json.dumps({k: report[k] for k in report if k != "per_drug"}, indent=2, default=str)[:3000])
    return 0


def cmd_features(args) -> int:
    from scripts.round24.analyze_features import run_feature_coverage, run_feature_attribution_stub

    cfg = load_eval3_config(args.config)
    out = _out_stage(cfg, "stage24c")
    if args.coverage_only or args.smoke:
        biocda_notify("Round24 Stage24C feature coverage START")
        report = run_feature_coverage(cfg, out)
        biocda_notify(f"Round24 Stage24C feature coverage DONE n={len(report)}")
        print(json.dumps(report, indent=2, default=str))
        return 0
    biocda_notify("Round24 Stage24C feature attribution requires Stage24B complete")
    run_feature_attribution_stub(cfg, out)
    return 0


def cmd_train(args) -> int:
    from scripts.round24.train_stage24b import run_stage24b

    cfg = load_eval3_config(args.config)
    configure_gpu_efficiency(target_utilization=float(cfg.get("target_gpu_utilization", 0.9)))
    max_jobs = args.max_jobs_per_gpu or int(cfg.get("max_jobs_per_gpu", 3))
    biocda_notify(f"Round24 train START stage={args.stage} smoke={args.smoke} jobs/gpu={max_jobs}")
    rc = run_stage24b(cfg, smoke=args.smoke, max_jobs_per_gpu=max_jobs, stage=args.stage or "24b")
    biocda_notify(f"Round24 train END stage={args.stage} rc={rc}")
    return int(rc)


def cmd_evaluate(args) -> int:
    from scripts.round24.evaluate_stage24b import run_evaluate_24b

    cfg = load_eval3_config(args.config)
    biocda_notify(f"Round24 evaluate START stage={args.stage}")
    report = run_evaluate_24b(cfg, stage=args.stage or "24b")
    biocda_notify(f"Round24 evaluate END stage={args.stage}")
    print(json.dumps(report, indent=2, default=str)[:4000])
    return 0


def cmd_select(args) -> int:
    from scripts.round24.lock_round24_model import run_select

    cfg = load_eval3_config(args.config)
    report = run_select(cfg, preregister_only=args.preregister_only, strict=args.strict_all_targets)
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_lock(args) -> int:
    from scripts.round24.lock_round24_model import run_lock

    cfg = load_eval3_config(args.config)
    report = run_lock(cfg, force=args.force if hasattr(args, "force") else False)
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_all(args) -> int:
    # Stage24A full then stop unless --continue later stages exist
    rc = cmd_protocol(args)
    if rc != 0:
        return rc
    if args.analysis_only:
        from scripts.round24.analyze_objective_alignment import run_alignment

        cfg = load_eval3_config(args.config)
        out = _out_stage(cfg, "stage24g")
        report = run_alignment(cfg, out)
        print(json.dumps(report, indent=2, default=str)[:3000])
        return 0
    print("Stage24A complete. Use train/evaluate for Stage24B+.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/round24/eval3.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in [
        ("preflight", cmd_preflight),
        ("protocol", cmd_protocol),
    ]:
        p = sub.add_parser(name)
        p.set_defaults(func=fn)

    p = sub.add_parser("baseline")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("diagnose")
    p.add_argument("--target", default="gdsc_intersect13")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("features")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--coverage-only", action="store_true")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("train")
    p.add_argument("--stage", default="24b")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-jobs-per-gpu", type=int, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate")
    p.add_argument("--stage", default="24b")
    p.add_argument("--formal", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("select")
    p.add_argument("--preregister-only", action="store_true")
    p.add_argument("--strict-all-targets", action="store_true")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("lock")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_lock)

    p = sub.add_parser("all")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
