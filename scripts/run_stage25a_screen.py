#!/usr/bin/env python3
"""Stage 25A formal screen launcher (Round 25).

Design
------
1. XA topology is FIXED (not searched).
2. Only Stage2 alignment variants are screened: S0 / S2 / S1 first.
3. Shared AE checkpoint is trained once, then all GAN variants load it
   (parity: same base encoder / classifier init).
4. GPU: parallel (variant, seed) workers with per-process mem fraction
   targeting ~90% utilization on a single RTX 6000 Ada.

Run inside Docker only:
  docker exec DAPL bash -lc 'cd /workspace/DAPL && PYTHONPATH=/workspace/DAPL \\
    python3 scripts/run_stage25a_screen.py --config config/round25_stage2_margin_screen.yaml \\
    --variants S0 S2 S1 --max-parallel 3 --mem-fraction 0.30'
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biocda.stage2.variant_registry import (
    initial_screen_variants,
    load_registry_from_yaml,
    registry_payload,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_parameters(module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def build_parameter_comparison(registry, out_csv: Path) -> None:
    from biocda.stage2.latent_autoencoder import LatentAutoencoder
    from biocda.stage2.target_adapter import TargetResidualAdapter

    latent_dim = 64
    rows = []
    for vid, var in registry.items():
        ae_n = ad_n = 0
        if var.global_alignment == "aada_autoencoder":
            ae_n = count_parameters(LatentAutoencoder(latent_dim))
            ad_n = count_parameters(TargetResidualAdapter(latent_dim))
        rows.append(
            {
                "variant": vid,
                "global_alignment": var.global_alignment,
                "prototype_alignment": var.prototype_alignment,
                "latent_ae_parameters": ae_n,
                "target_adapter_parameters": ad_n,
                "variant_extra_parameters": ae_n + ad_n,
                "note": "critic/encoder/proto bank shared with S0 baseline",
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def base_params_from_reference(ref_params_path: Path) -> Dict[str, Any]:
    payload = json.loads(ref_params_path.read_text(encoding="utf-8"))
    params = dict(payload.get("params") or payload)
    # Keep Stage2-relevant keys; zero unrelated optional losses for parity.
    keep_defaults = {
        "lambda_proto": 0,
        "lambda_class_gap": 0,
        "lambda_cmmd": 0,
        "lambda_tumor_topology": 0,
        "lambda_tumor_supcon": 0,
        "lambda_subspace_ortho": 0,
        "lambda_tumor_var": 0,
        "lambda_tumor_cov": 0,
        "use_tumor_subspace": False,
    }
    params.update(keep_defaults)
    return params


def apply_screen_budget(params: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Shared reduced-but-valid budget for 25A screen (same for all variants)."""
    budget = dict(cfg.get("screen_budget") or {})
    out = dict(params)
    # Defaults: enough for cond/proto ramp while finishing overnight on 1 GPU × 3 workers.
    defaults = {
        "pretrain_num_epochs": 100,
        "pretrain_patience": 20,
        "train_num_epochs": 200,
        "gan_patience": 35,
        "gan_early_stop_start_epoch": 50,
        "cond_adv_start_epoch": 20,
        "cond_adv_full_epoch": 70,
        "proto_align_start_epoch": 30,
        "proto_align_full_epoch": 90,
        "cls_start_epoch": 20,
        "cls_full_epoch": 70,
        "batch_size": 256,
    }
    defaults.update(budget)
    out.update(defaults)
    return out


def apply_variant(params: Dict[str, Any], variant_id: str, var_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    out["round25_variant"] = variant_id
    out["source_anchor_proto_enabled"] = True
    if float(out.get("lambda_proto_align", 0.0)) <= 0:
        out["lambda_proto_align"] = 0.001

    proto = str(var_cfg.get("prototype_alignment", "always_on"))
    glob = str(var_cfg.get("global_alignment", "wgan"))

    if proto == "always_on":
        out["proto_align_mode"] = "always_on"
    elif proto == "margin_gated":
        out["proto_align_mode"] = "margin_gated"
        out["prototype_margin"] = dict(var_cfg.get("prototype_margin") or {})
    elif proto == "distance_band":
        out["proto_align_mode"] = "distance_band"
        out["prototype_band"] = dict(var_cfg.get("prototype_band") or {})
    else:
        raise ValueError(f"unsupported prototype_alignment={proto}")

    if glob == "wgan":
        out["aada_enabled"] = False
        out["global_adv_mode"] = out.get("global_adv_mode") or "conditional_plus_weak_global"
    elif glob == "aada_autoencoder":
        out["aada_enabled"] = True
        out["aada"] = dict(var_cfg.get("aada") or {})
        # reconstruction_margin calibrated later if needed; use configured default.
        if "reconstruction_margin" not in out["aada"]:
            out["aada"]["reconstruction_margin"] = float(
                out["aada"].get("reconstruction_margin", 0.1)
            )
        out["global_adv_mode"] = "conditional_replacement"
    else:
        raise ValueError(f"unsupported global_alignment={glob}")
    return out


def write_combo_config(params: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pretrain_param_combinations": [params]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _worker_train(job: Dict[str, Any]) -> Dict[str, Any]:
    """Subprocess worker: one (variant, seed) GAN job."""
    os.chdir(job["root"])
    os.environ["PYTHONPATH"] = job["root"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job.get("cuda_device", "0"))
    try:
        import torch

        frac = float(job.get("mem_fraction", 0.3))
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(min(max(frac, 0.05), 0.95), 0)
    except Exception as exc:  # pragma: no cover
        return {**job, "status": "FAIL", "error": f"cuda setup failed: {exc}"}

    outfolder = Path(job["outfolder"])
    outfolder.mkdir(parents=True, exist_ok=True)
    log_path = outfolder / "train.log"
    cmd = [
        sys.executable,
        "pretrain_VAEwC.py",
        "--outfolder",
        str(outfolder),
        "--config",
        job["config_path"],
        "--target_domain",
        "tcga",
    ]
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=job["root"],
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    elapsed = time.time() - t0
    # Locate newest exp_* under outfolder
    exps = sorted(outfolder.glob("exp_*"), key=lambda p: p.stat().st_mtime)
    exp_dir = str(exps[-1]) if exps else ""
    status = "DONE" if proc.returncode == 0 and exp_dir else "FAIL"
    return {
        **{k: job[k] for k in ("variant", "seed", "job_id")},
        "status": status,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "exp_dir": exp_dir,
        "log_path": str(log_path),
        "error": None if status == "DONE" else f"rc={proc.returncode}",
    }


def bootstrap_shared_ae(
    *,
    root: Path,
    base_params: Dict[str, Any],
    out_dir: Path,
    mem_fraction: float,
) -> Path:
    """Train AE once; all Stage2 variants load this checkpoint."""
    ae_dir = out_dir / "shared_ae"
    marker = ae_dir / "after_pretrain_shared_vae.pth"
    if marker.exists():
        print(f"[25A] reuse shared AE: {ae_dir}")
        return ae_dir
    ae_dir.mkdir(parents=True, exist_ok=True)
    params = dict(base_params)
    params["random_seed"] = 17
    params["ae_only"] = True
    params["skip_ae_train"] = False
    # AE-only: still run pretrain epochs; GAN skipped via ae_only return.
    params["train_num_epochs"] = 1  # unused when ae_only returns early
    cfg_path = ae_dir / "ae_params.json"
    write_combo_config(params, cfg_path)
    job = {
        "root": str(root),
        "job_id": "AE_SHARED",
        "variant": "AE_SHARED",
        "seed": 17,
        "outfolder": str(ae_dir),
        "config_path": str(cfg_path),
        "mem_fraction": min(0.9, mem_fraction * 2.5),
        "cuda_device": "0",
    }
    print("[25A] bootstrapping shared AE …")
    result = _worker_train(job)
    if result["status"] != "DONE":
        raise RuntimeError(f"shared AE bootstrap failed: {result}")
    # Move/copy after_pretrain from exp_* into ae_dir root for stable path.
    exp = Path(result["exp_dir"])
    for fname in (
        "after_pretrain_shared_vae.pth",
        "after_pretrain_source_vae.pth",
        "after_pretrain_target_vae.pth",
        "after_pretrain_classifier.pth",
    ):
        src = exp / fname
        if not src.exists():
            raise FileNotFoundError(f"shared AE missing {src}")
        dst = ae_dir / fname
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
    (ae_dir / "bootstrap_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[25A] shared AE ready: {ae_dir}")
    return ae_dir


def collect_job_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "variant": result["variant"],
        "seed": result["seed"],
        "status": result["status"],
        "elapsed_sec": result.get("elapsed_sec"),
        "exp_dir": result.get("exp_dir"),
        "best_gan_loss": None,
        "proto_align_loss": None,
        "prototype_hinge_active_fraction": None,
        "source_reconstruction_error": None,
        "target_reconstruction_error": None,
    }
    exp = Path(result.get("exp_dir") or "")
    gm = exp / "gan_metrics.json"
    if gm.exists():
        try:
            g = json.loads(gm.read_text(encoding="utf-8"))
            row["best_gan_loss"] = g.get("best_loss") or g.get("best_eval_loss")
        except Exception:
            pass
    # Pull last gen loss row if present
    gcsv = exp / "g_loss.csv"
    if gcsv.exists():
        try:
            import pandas as pd

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
        except Exception:
            pass
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/round25_stage2_margin_screen.yaml")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--mem-fraction", type=float, default=0.30)
    ap.add_argument("--out-dir", default="result/optimization_runs/round25_stage25a")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--skip-shared-ae", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny epoch budget to validate pipeline before formal screen",
    )
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    cfg = _load_yaml(cfg_path)
    registry = load_registry_from_yaml(cfg_path)
    seeds = list(cfg.get("experiment", {}).get("seeds") or [17, 29, 43])
    variants = list(args.variants) if args.variants else initial_screen_variants(registry)

    for v in variants:
        if v in ("S3", "S2b") and args.strict:
            decision = ROOT / "reports/round25_stage25a_decision.json"
            if not decision.exists():
                raise SystemExit(f"refusing {v}: stage25a decision missing")

    reports_dir = ROOT / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    payload = registry_payload(registry)
    payload.update(
        {
            "screen_variants": variants,
            "seeds": seeds,
            "created_at": _now(),
            "gpu": {
                "max_parallel": args.max_parallel,
                "mem_fraction": args.mem_fraction,
                "target_utilization": 0.9,
            },
        }
    )
    (reports_dir / "round25_variant_registry.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    build_parameter_comparison(registry, reports_dir / "round25_stage2_parameter_comparison.csv")

    ref = cfg.get("baseline", {})
    ref_root = ROOT / str(
        ref.get(
            "reference_pretrain_root",
            "result/optimization_runs/round12_proto_alignment/pretrain",
        )
    )
    ref_exp = str(ref.get("reference_exp_id", "exp_027"))
    ref_params = ref_root / ref_exp / "params.json"
    if not ref_params.exists():
        # fallback to exp_037 structure with proto force-on
        alt = ROOT / "result/optimization_runs/round12_proto_alignment/pretrain/exp_027/params.json"
        ref_params = alt if alt.exists() else ref_params
    if not ref_params.exists():
        raise FileNotFoundError(f"missing reference params: {ref_params}")

    base = apply_screen_budget(base_params_from_reference(ref_params), cfg)
    if args.smoke:
        base.update(
            {
                "pretrain_num_epochs": 2,
                "pretrain_patience": 1,
                "train_num_epochs": 3,
                "gan_patience": 2,
                "gan_early_stop_start_epoch": 1,
                "cond_adv_start_epoch": 1,
                "cond_adv_full_epoch": 2,
                "proto_align_start_epoch": 1,
                "proto_align_full_epoch": 2,
                "cls_start_epoch": 1,
                "cls_full_epoch": 2,
                "batch_size": 128,
            }
        )
        print("[25A] SMOKE budget enabled")
        out_root = ROOT / (args.out_dir + "_smoke")
        out_root.mkdir(parents=True, exist_ok=True)

    jobs_meta: List[Dict[str, Any]] = []
    for vid in variants:
        var_cfg = dict((cfg.get("variants") or {}).get(vid) or {})
        for seed in seeds:
            p = apply_variant(base, vid, var_cfg)
            p["random_seed"] = int(seed)
            p["skip_ae_train"] = True
            job_id = f"{vid}_seed{seed}"
            job_dir = out_root / job_id
            cfg_job = job_dir / "params_combo.json"
            jobs_meta.append(
                {
                    "job_id": job_id,
                    "variant": vid,
                    "seed": int(seed),
                    "status": "PLANNED",
                    "outfolder": str(job_dir),
                    "config_path": str(cfg_job),
                    "params_preview": {
                        "proto_align_mode": p.get("proto_align_mode"),
                        "aada_enabled": p.get("aada_enabled"),
                        "global_adv_mode": p.get("global_adv_mode"),
                        "train_num_epochs": p.get("train_num_epochs"),
                        "pretrain_num_epochs": p.get("pretrain_num_epochs"),
                    },
                    "_params": p,
                }
            )

    plan = {
        "stage": "25A",
        "config": str(cfg_path.relative_to(ROOT)),
        "dry_run": bool(args.dry_run),
        "jobs": [{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs_meta],
        "n_jobs": len(jobs_meta),
        "forbidden": cfg.get("forbidden"),
        "selection": cfg.get("selection"),
        "screen_budget": cfg.get("screen_budget"),
        "tcga_used_for_selection": False,
        "round23_gdsc_lock_must_remain": "REJECTED",
        "created_at": _now(),
    }
    (reports_dir / "round25_stage25a_job_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "n_jobs": len(jobs_meta)}, indent=2))
        return 0

    # 1) Shared AE
    if args.skip_shared_ae:
        ae_dir = out_root / "shared_ae"
        if not (ae_dir / "after_pretrain_shared_vae.pth").exists():
            raise FileNotFoundError("shared AE missing and --skip-shared-ae set")
    else:
        ae_params = dict(base)
        ae_dir = bootstrap_shared_ae(
            root=ROOT,
            base_params=ae_params,
            out_dir=out_root,
            mem_fraction=args.mem_fraction,
        )

    # 2) Materialize per-job configs with ae_init_dir
    worker_jobs = []
    for j in jobs_meta:
        p = dict(j["_params"])
        p["ae_init_dir"] = str(ae_dir)
        p["skip_ae_train"] = True
        write_combo_config(p, Path(j["config_path"]))
        worker_jobs.append(
            {
                "root": str(ROOT),
                "job_id": j["job_id"],
                "variant": j["variant"],
                "seed": j["seed"],
                "outfolder": j["outfolder"],
                "config_path": j["config_path"],
                "mem_fraction": args.mem_fraction,
                "cuda_device": "0",
            }
        )

    print(
        f"[25A] launching {len(worker_jobs)} jobs "
        f"max_parallel={args.max_parallel} mem_fraction={args.mem_fraction}"
    )
    results: List[Dict[str, Any]] = []
    # Threads + subprocess workers: avoids forking after CUDA init in parent.
    with ThreadPoolExecutor(max_workers=int(args.max_parallel)) as ex:
        futs = {ex.submit(_worker_train, job): job for job in worker_jobs}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            print(
                f"[25A] {res['job_id']} -> {res['status']} "
                f"({res.get('elapsed_sec', 0):.0f}s) exp={res.get('exp_dir')}"
            )

    metrics_rows = [collect_job_metrics(r) for r in results]
    metrics_path = reports_dir / "round25_stage25a_metrics.csv"
    fieldnames: List[str] = []
    for row in metrics_rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(metrics_rows)

    summary = {
        "created_at": _now(),
        "n_jobs": len(results),
        "n_done": sum(1 for r in results if r["status"] == "DONE"),
        "n_fail": sum(1 for r in results if r["status"] != "DONE"),
        "shared_ae": str(ae_dir),
        "results": results,
        "metrics_csv": str(metrics_path.relative_to(ROOT)),
    }
    (reports_dir / "round25_stage25a_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("n_jobs", "n_done", "n_fail", "metrics_csv")}, indent=2))
    if args.strict and summary["n_fail"]:
        return 1
    return 0


if __name__ == "__main__":
    # ProcessPool requires picklable main under spawn; fork is default on Linux.
    raise SystemExit(main())
