"""Single-GPU sequential optimization runner for VAEwC prototype InfoNCE sweeps."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from glob import glob
from itertools import product
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.optimization_config_generator import MANIFEST_COLUMNS, generate_configs
from tools.optimization_report import generate_final_reports
from tools.finetune_tcga_eval import FIXED_DRUG_SMILES_AACDR_EXTENDED
from tools.optimization_selection import (
    SelectionInsufficientError,
    build_model_select_from_top10,
    write_selection_outputs,
)
from tools.update_running_report import write_running_report


def _refresh_running_report(run_dir: str, note: str = "") -> None:
    try:
        write_running_report(run_dir, note=note)
    except Exception as exc:
        print(f"[warn] running_report update failed: {exc}")


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _append_drug_smiles_arg(cmd: List[str], drug_smiles_path: str) -> None:
    if drug_smiles_path:
        cmd.extend(["--drug-smiles-path", _resolve_path(drug_smiles_path)])


def _resolve_pretrain_result_folder(model_id: str, model_row, job_row) -> str:
    """Resolve pretrain folder; supports external baselines via pretrain_result_dir."""
    candidates = []
    for source in (job_row, model_row):
        if source is None:
            continue
        for key in ("pretrain_result_dir", "result_folder"):
            try:
                val = source.get(key) if hasattr(source, "get") else None
            except Exception:
                val = None
            if val is not None and str(val).strip() and str(val).lower() != "nan":
                candidates.append(str(val).strip())
    for val in candidates:
        resolved = _resolve_path(val)
        if os.path.isdir(resolved):
            return resolved
    return _resolve_path(os.path.join("pretrain", str(model_id)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


FINETUNE_MANIFEST_COLUMNS = [
    "job_id",
    "model_id",
    "combo_id",
    "pretrain_result_dir",
    "status",
    "start_time",
    "end_time",
    "error_message",
]

ROUND13_FINETUNE_MANIFEST_COLUMNS = FINETUNE_MANIFEST_COLUMNS + [
    "source_model_id",
    "source_round",
    "feature_mode",
    "prototype_feature_mode",
    "response_input_mode",
    "combined_latent_dir",
    "model_select_path",
    "finetune_config_path",
    "result_dir",
    "random_seed",
]


class ManifestManager:
    def __init__(self, manifest_path: str, default_columns: Optional[List[str]] = None):
        self.manifest_path = _resolve_path(manifest_path)
        self.default_columns = default_columns or MANIFEST_COLUMNS
        if os.path.exists(self.manifest_path):
            self.df = pd.read_csv(self.manifest_path)
        else:
            self.df = pd.DataFrame(columns=self.default_columns)
        for col in self.default_columns:
            if col not in self.df.columns:
                self.df[col] = ""

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        self.df.to_csv(self.manifest_path, index=False)

    def pending_jobs(self, rerun_completed: bool = False) -> pd.DataFrame:
        if rerun_completed:
            return self.df[self.df["status"].isin(["pending", "success", "failed", "skipped"])].copy()
        return self.df[self.df["status"] == "pending"].copy()

    def update_job(self, job_id: str, **fields) -> None:
        mask = self.df["job_id"] == job_id
        if not mask.any():
            raise KeyError(f"job_id not found in manifest: {job_id}")
        for key, value in fields.items():
            self.df.loc[mask, key] = value
        self.save()


def _write_status_json(status_dir: str, job_id: str, payload: dict) -> str:
    os.makedirs(status_dir, exist_ok=True)
    path = os.path.join(status_dir, f"{job_id}_status.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _run_command(cmd: List[str], log_path: str, dry_run: bool = False) -> int:
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        )
    return int(proc.returncode)


def _latest_exp_dir(pretrain_dir: str, before: Optional[List[str]] = None) -> Optional[str]:
    before_set = set(before or [])
    exp_dirs = sorted(glob(os.path.join(pretrain_dir, "exp_*")))
    for path in reversed(exp_dirs):
        if os.path.basename(path) not in before_set:
            return path
    return None


def _exp_dir_from_log(log_path: str, pretrain_dir: str) -> Optional[str]:
    """Resolve the experiment directory created by a pretrain job from its log."""
    if not os.path.exists(log_path):
        return None
    exp_name = None
    with open(log_path, "r", errors="replace") as log_file:
        for line in log_file:
            match = re.search(r"start experiment (exp_\d+)", line)
            if match:
                exp_name = match.group(1)
    if not exp_name:
        return None
    exp_dir = os.path.join(pretrain_dir, exp_name)
    return exp_dir if os.path.isdir(exp_dir) else None


def _build_pretrain_cmd(
    config_path: str,
    pretrain_dir: str,
    overlap_tcga: str,
    smoke_test: bool = False,
    batch_size: Optional[int] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "pretrain_VAEwC.py"),
        "--config",
        config_path,
        "--outfolder",
        pretrain_dir,
        "--target_domain",
        "tcga",
        "--overlap_tcga",
        _resolve_path(overlap_tcga),
    ]
    if smoke_test:
        cmd.extend(["--smoke-test"])
    if batch_size is not None and batch_size > 0:
        cmd.extend(["--batch-size", str(int(batch_size))])
    return cmd


def _run_one_pretrain_job(
    job_row,
    manager: ManifestManager,
    pretrain_dir: str,
    logs_dir: str,
    status_dir: str,
    run_dir: str,
    overlap_tcga: str,
    smoke_test: bool,
    batch_size: Optional[int],
    dry_run: bool,
    lock: threading.Lock,
) -> None:
    job_id = job_row["job_id"]
    config_path = _resolve_path(str(job_row["config_path"]))
    if not os.path.exists(config_path):
        if dry_run:
            print(f"[dry-run] skip missing config: {config_path}")
            return
        with lock:
            manager.update_job(
                job_id,
                status="failed",
                end_time=_utc_now(),
                error_message=f"Missing config: {config_path}",
            )
        return

    existing_exps = set(os.path.basename(p) for p in glob(os.path.join(pretrain_dir, "exp_*")))
    log_path = os.path.join(logs_dir, f"{job_id}.log")
    cmd = _build_pretrain_cmd(config_path, pretrain_dir, overlap_tcga, smoke_test, batch_size)

    if dry_run:
        _run_command(cmd, log_path, dry_run=True)
        return

    with lock:
        manager.update_job(job_id, status="running", start_time=_utc_now(), error_message="")
    return_code = _run_command(cmd, log_path, dry_run=False)
    status_payload = {"job_id": job_id, "return_code": return_code, "log_path": log_path}

    with lock:
        if return_code != 0:
            manager.update_job(
                job_id,
                status="failed",
                end_time=_utc_now(),
                error_message=f"pretrain failed with code {return_code}",
            )
            _write_status_json(status_dir, job_id, status_payload)
            return

        result_dir = _exp_dir_from_log(log_path, pretrain_dir)
        if result_dir is None:
            result_dir = _latest_exp_dir(pretrain_dir, before=list(existing_exps))
        if result_dir is None:
            manager.update_job(
                job_id,
                status="failed",
                end_time=_utc_now(),
                error_message="No new exp_* directory found after pretrain",
            )
            return

        manager.update_job(
            job_id,
            status="success",
            end_time=_utc_now(),
            result_dir=os.path.relpath(result_dir, PROJECT_ROOT),
            error_message="",
        )
        status_payload["result_dir"] = result_dir
        _write_status_json(status_dir, job_id, status_payload)

    _refresh_running_report(run_dir, note=f"Pretrain job `{job_id}` completed → `{os.path.basename(result_dir)}`")


def run_pretrain_stage(
    manifest_path: str,
    run_dir: str,
    device: str = "cuda:0",
    dry_run: bool = False,
    rerun_completed: bool = False,
    smoke_test: bool = False,
    overlap_tcga: str = "data/TCGA/PMID27354694_DR_OMICS_ad.csv",
    batch_size: Optional[int] = None,
    max_parallel: int = 1,
) -> None:
    manager = ManifestManager(manifest_path)
    pretrain_dir = _resolve_path(os.path.join(run_dir, "pretrain"))
    logs_dir = _resolve_path(os.path.join(run_dir, "logs", "pretrain"))
    status_dir = _resolve_path(os.path.join(run_dir, "status", "pretrain"))
    os.makedirs(pretrain_dir, exist_ok=True)

    pending = manager.pending_jobs(rerun_completed=rerun_completed)
    max_parallel = max(1, int(max_parallel))
    lock = threading.Lock()

    jobs = [row for _, row in pending.iterrows()]
    if max_parallel == 1 or dry_run:
        for job in jobs:
            _run_one_pretrain_job(
                job, manager, pretrain_dir, logs_dir, status_dir, run_dir,
                overlap_tcga, smoke_test, batch_size, dry_run, lock,
            )
    else:
        print(f"[pretrain] parallel dispatch: {len(jobs)} jobs, max_parallel={max_parallel}")
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [
                pool.submit(
                    _run_one_pretrain_job,
                    job, manager, pretrain_dir, logs_dir, status_dir, run_dir,
                    overlap_tcga, smoke_test, batch_size, dry_run, lock,
                )
                for job in jobs
            ]
            for fut in futures:
                fut.result()

    _refresh_running_report(
        run_dir,
        note=f"Pretrain stage batch finished (batch_size={batch_size or 'config'}, max_parallel={max_parallel}).",
    )


def _parse_response_head_dims(spec) -> List[int]:
    if spec is None:
        return [256, 128]
    if isinstance(spec, list):
        return [int(x) for x in spec]
    text = str(spec).strip().lower()
    if text in ("", "default"):
        return [256, 128]
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _expand_curated_finetune_combos(combos_cfg: List[dict]) -> List[dict]:
    combos: List[dict] = []
    for row in combos_cfg:
        combo_id = int(row["combo_id"])
        combos.append(
            {
                "combo_id": combo_id,
                "finetune_params": {
                    "ftlr": float(row.get("learning_rate", row.get("ftlr", 0.001))),
                    "scheduler_flag": bool(row.get("scheduler_flag", True)),
                    "loss_type": str(row.get("loss_type", "bce")),
                    "focal_loss_gamma": float(row.get("focal_loss_gamma", 2.0)),
                    "weight_decay": float(row.get("weight_decay", 1e-5)),
                    "patience": int(row.get("early_stop_patience", row.get("patience", 50))),
                },
                "classifier_params": {
                    "hidden_dims": _parse_response_head_dims(row.get("response_head_dims", "default")),
                    "dropout_rate": float(row.get("dropout", row.get("dropout_rate", 0.1))),
                    "use_batch_norm": bool(row.get("use_batch_norm", True)),
                    "activation": str(row.get("activation", "leaky_relu")),
                },
                "model_params": {"gin_type": str(row.get("gin_type", "dapl"))},
                "batch_size": int(row.get("batch_size", 12288)),
                "mini_batch_size": int(row.get("mini_batch_size", 3072)),
                "epochs": int(row.get("epochs", 1000)),
            }
        )
    return combos


def _expand_finetune_combinations(config_path: str) -> List[dict]:
    with open(_resolve_path(config_path), "r", encoding="utf-8") as f:
        config = json.load(f)
    if "finetune_combos" in config:
        return _expand_curated_finetune_combos(config["finetune_combos"])
    ft = [dict(zip(config["finetune_params"].keys(), v)) for v in product(*config["finetune_params"].values())]
    clf = [dict(zip(config["classifier_params"].keys(), v)) for v in product(*config["classifier_params"].values())]
    model = [dict(zip(config["model_params"].keys(), v)) for v in product(*config["model_params"].values())]
    combos = []
    combo_id = 0
    for ft_params in ft:
        for clf_params in clf:
            for model_params in model:
                combos.append(
                    {
                        "combo_id": combo_id,
                        "finetune_params": ft_params,
                        "classifier_params": clf_params,
                        "model_params": model_params,
                    }
                )
                combo_id += 1
    return combos


def build_finetune_manifest(
    top10_path: str,
    run_dir: str,
    finetune_config: str = "config/params_finetune_mini.json",
    force: bool = False,
) -> str:
    top10_df = pd.read_csv(_resolve_path(top10_path))
    combos = _expand_finetune_combinations(finetune_config)
    manifest_dir = _resolve_path(os.path.join(run_dir, "manifests"))
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "finetune_dispatch_manifest.csv")
    if os.path.exists(manifest_path) and not force:
        return manifest_path

    rows = []
    for _, model_row in top10_df.iterrows():
        model_id = model_row["ID"]
        for combo in combos:
            job_id = f"ft_{model_id}_c{combo['combo_id']:02d}"
            rows.append(
                {
                    "job_id": job_id,
                    "model_id": model_id,
                    "combo_id": combo["combo_id"],
                    "pretrain_result_dir": (
                        model_row.get("pretrain_result_dir")
                        or model_row.get("result_folder")
                        or os.path.join("pretrain", model_id)
                    ),
                    "status": "pending",
                    "start_time": "",
                    "end_time": "",
                    "error_message": "",
                }
            )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _single_combo_finetune_config(base_config_path: str, combo: dict, out_path: str) -> str:
    with open(_resolve_path(base_config_path), "r", encoding="utf-8") as f:
        base = json.load(f)
    payload = {
        "finetune_params": {k: [v] for k, v in combo["finetune_params"].items()},
        "classifier_params": {k: [v] for k, v in combo["classifier_params"].items()},
        "model_params": {k: [v] for k, v in combo["model_params"].items()},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _run_one_finetune_job(
    job_row,
    manager: ManifestManager,
    run_dir: str,
    top10_df: pd.DataFrame,
    combos: List[dict],
    finetune_config: str,
    finetune_dir: str,
    logs_dir: str,
    status_dir: str,
    scratch_dir: str,
    batch_size: int,
    mini_batch_size: int,
    epochs: int,
    dry_run: bool,
    lock: threading.Lock,
    drug_smiles_path: str = FIXED_DRUG_SMILES_AACDR_EXTENDED,
) -> None:

    job_id = job_row["job_id"]
    model_id = job_row["model_id"]
    combo_id = int(job_row["combo_id"])
    manifest_model_select = str(job_row.get("model_select_path", "")).strip()
    if manifest_model_select:
        model_select_path = _resolve_path(manifest_model_select)
        if not os.path.isfile(model_select_path):
            with lock:
                manager.update_job(
                    job_id,
                    status="failed",
                    end_time=_utc_now(),
                    error_message=f"Missing model_select_path: {model_select_path}",
                )
            return
    elif top10_df.empty or model_id not in top10_df.index:
        with lock:
            manager.update_job(
                job_id,
                status="failed",
                end_time=_utc_now(),
                error_message=f"Missing pretrain candidate in top10: {model_id}",
            )
        return
    else:
        model_row = top10_df.loc[[model_id]].reset_index()
        model_select_path = os.path.join(_resolve_path(run_dir), f"_ft_{job_id}_model_select.csv")
        ms_df = build_model_select_from_top10(model_row)
        ms_df["result_folder"] = _resolve_pretrain_result_folder(model_id, model_row.iloc[0], job_row)
        ms_df.to_csv(model_select_path, index=False)

    combo = combos[combo_id]
    combo_config_path = os.path.join(scratch_dir, f"{job_id}_config.json")
    _single_combo_finetune_config(finetune_config, combo, combo_config_path)

    log_path = os.path.join(logs_dir, f"{job_id}.log")
    job_out = os.path.join(finetune_dir, model_id, f"combo_{combo_id:02d}")
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "step1_finetune_latent_pipeline_All_split.py"),
        "--config",
        combo_config_path,
        "--model_select_path",
        model_select_path,
        "--outfolder",
        job_out,
        "--batch_size",
        str(batch_size),
        "--mini_batch_size",
        str(mini_batch_size),
        "--epochs",
        str(epochs),
    ]
    _append_drug_smiles_arg(cmd, drug_smiles_path)
    if dry_run:
        _run_command(cmd, log_path, dry_run=True)
        return

    with lock:
        manager.update_job(job_id, status="running", start_time=_utc_now(), error_message="")
    return_code = _run_command(cmd, log_path, dry_run=False)
    with lock:
        if return_code != 0:
            manager.update_job(
                job_id,
                status="failed",
                end_time=_utc_now(),
                error_message=f"finetune failed with code {return_code}",
            )
        else:
            manager.update_job(job_id, status="success", end_time=_utc_now(), error_message="")
        _write_status_json(status_dir, job_id, {"job_id": job_id, "return_code": return_code, "log_path": log_path})
    _refresh_running_report(run_dir, note=f"Finetune job `{job_id}` finished (code={return_code})")


def _run_one_round13_finetune_job(
    job_row,
    manager: ManifestManager,
    run_dir: str,
    combos: List[dict],
    finetune_config: str,
    logs_dir: str,
    status_dir: str,
    scratch_dir: str,
    batch_size: int,
    mini_batch_size: int,
    epochs: int,
    dry_run: bool,
    lock: threading.Lock,
    drug_smiles_path: str = FIXED_DRUG_SMILES_AACDR_EXTENDED,
) -> None:
    job_id = job_row["job_id"]
    combo_id = int(job_row["combo_id"])
    model_select_path = _resolve_path(str(job_row["model_select_path"]))
    if not os.path.isfile(model_select_path):
        with lock:
            manager.update_job(
                job_id,
                status="failed",
                end_time=_utc_now(),
                error_message=f"Missing model_select_path: {model_select_path}",
            )
        return

    combo = combos[combo_id]
    combo_config_path = os.path.join(scratch_dir, f"{job_id}_config.json")
    _single_combo_finetune_config(finetune_config, combo, combo_config_path)

    log_path = os.path.join(logs_dir, f"{job_id}.log")
    job_out = _resolve_path(str(job_row["result_dir"]))
    os.makedirs(job_out, exist_ok=True)

    job_epochs = epochs
    if "epochs" in job_row and str(job_row.get("epochs", "")).strip() not in ("", "nan"):
        job_epochs = int(job_row["epochs"])
    # CLI batch args from stage scripts take precedence over manifest defaults.
    job_batch_size = batch_size
    if job_batch_size <= 0:
        if "batch_size" in job_row and str(job_row.get("batch_size", "")).strip() not in ("", "nan"):
            job_batch_size = int(job_row["batch_size"])
        elif "batch_size" in combo:
            job_batch_size = int(combo["batch_size"])
    job_mini_batch_size = mini_batch_size
    if job_mini_batch_size <= 0:
        if "mini_batch_size" in job_row and str(job_row.get("mini_batch_size", "")).strip() not in ("", "nan"):
            job_mini_batch_size = int(job_row["mini_batch_size"])
        elif "mini_batch_size" in combo:
            job_mini_batch_size = int(combo["mini_batch_size"])

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "step1_finetune_latent_pipeline_All_split.py"),
        "--config",
        combo_config_path,
        "--model_select_path",
        model_select_path,
        "--outfolder",
        job_out,
        "--batch_size",
        str(job_batch_size),
        "--mini_batch_size",
        str(job_mini_batch_size),
        "--epochs",
        str(job_epochs),
    ]
    seed_val = job_row.get("seed", job_row.get("random_seed"))
    if seed_val is not None and str(seed_val).strip() not in ("", "nan"):
        cmd.extend(["--random_seed", str(int(seed_val))])
    _append_drug_smiles_arg(cmd, drug_smiles_path)
    if dry_run:
        _run_command(cmd, log_path, dry_run=True)
        return

    with lock:
        manager.update_job(job_id, status="running", start_time=_utc_now(), error_message="")
    return_code = _run_command(cmd, log_path, dry_run=False)
    with lock:
        if return_code != 0:
            manager.update_job(
                job_id,
                status="failed",
                end_time=_utc_now(),
                error_message=f"finetune failed with code {return_code}",
            )
        else:
            manager.update_job(job_id, status="success", end_time=_utc_now(), error_message="")
        _write_status_json(
            status_dir,
            job_id,
            {
                "job_id": job_id,
                "return_code": return_code,
                "log_path": log_path,
                "model_select_path": model_select_path,
                "combined_latent_dir": str(job_row.get("combined_latent_dir", "")),
                "prototype_feature_mode": str(job_row.get("prototype_feature_mode", "")),
            },
        )
    _refresh_running_report(run_dir, note=f"Round13 finetune job `{job_id}` finished (code={return_code})")


def run_round13_finetune_stage(
    manifest_path: str,
    run_dir: str,
    finetune_config: str = "config/params_finetune_proto_features.json",
    batch_size: int = 2048,
    mini_batch_size: int = 512,
    epochs: int = 1000,
    dry_run: bool = False,
    rerun_completed: bool = False,
    max_parallel: int = 1,
    drug_smiles_path: str = FIXED_DRUG_SMILES_AACDR_EXTENDED,
) -> None:
    manager = ManifestManager(manifest_path, default_columns=ROUND13_FINETUNE_MANIFEST_COLUMNS)
    manifest_df = manager.df
    required = {"job_id", "model_select_path", "result_dir", "combo_id"}
    missing = required - set(manifest_df.columns)
    if missing:
        raise ValueError(f"Round 13 finetune manifest missing columns: {sorted(missing)}")
    if manifest_df["model_select_path"].fillna("").astype(str).str.strip().eq("").all():
        raise ValueError("Round 13 finetune manifest has no model_select_path entries")

    logs_dir = _resolve_path(os.path.join(run_dir, "logs", "finetune"))
    status_dir = _resolve_path(os.path.join(run_dir, "status", "finetune"))
    scratch_dir = _resolve_path(os.path.join(run_dir, "scratch", "finetune"))
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    combos = _expand_finetune_combinations(finetune_config)
    pending = manager.pending_jobs(rerun_completed=rerun_completed)
    max_parallel = max(1, int(max_parallel))
    lock = threading.Lock()

    jobs = [row for _, row in pending.iterrows()]
    if max_parallel == 1 or dry_run:
        for job in jobs:
            _run_one_round13_finetune_job(
                job,
                manager,
                run_dir,
                combos,
                finetune_config,
                logs_dir,
                status_dir,
                scratch_dir,
                batch_size,
                mini_batch_size,
                epochs,
                dry_run,
                lock,
                drug_smiles_path,
            )
    else:
        print(f"[round13 finetune] parallel dispatch: {len(jobs)} jobs, max_parallel={max_parallel}")
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [
                pool.submit(
                    _run_one_round13_finetune_job,
                    job,
                    manager,
                    run_dir,
                    combos,
                    finetune_config,
                    logs_dir,
                    status_dir,
                    scratch_dir,
                    batch_size,
                    mini_batch_size,
                    epochs,
                    dry_run,
                    lock,
                    drug_smiles_path,
                )
                for job in jobs
            ]
            for fut in futures:
                fut.result()

    _refresh_running_report(
        run_dir,
        note=f"Round13 finetune stage finished (max_parallel={max_parallel}, batch_size={batch_size}).",
    )


def run_finetune_stage(
    manifest_path: str,
    run_dir: str,
    top10_path: Optional[str] = None,
    finetune_config: str = "config/params_finetune_mini.json",
    batch_size: int = 2048,
    mini_batch_size: int = 512,
    epochs: int = 1000,
    dry_run: bool = False,
    rerun_completed: bool = False,
    max_parallel: int = 1,
    drug_smiles_path: str = FIXED_DRUG_SMILES_AACDR_EXTENDED,
) -> None:
    manager = ManifestManager(manifest_path, default_columns=FINETUNE_MANIFEST_COLUMNS)
    finetune_dir = _resolve_path(os.path.join(run_dir, "finetune"))
    logs_dir = _resolve_path(os.path.join(run_dir, "logs", "finetune"))
    status_dir = _resolve_path(os.path.join(run_dir, "status", "finetune"))
    scratch_dir = _resolve_path(os.path.join(run_dir, "scratch", "finetune"))
    os.makedirs(finetune_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    if top10_path and os.path.isfile(_resolve_path(top10_path)):
        top10_df = pd.read_csv(_resolve_path(top10_path)).set_index("ID")
    else:
        top10_df = pd.DataFrame()
    combos = _expand_finetune_combinations(finetune_config)
    pending = manager.pending_jobs(rerun_completed=rerun_completed)
    max_parallel = max(1, int(max_parallel))
    lock = threading.Lock()

    jobs = [row for _, row in pending.iterrows()]
    if max_parallel == 1 or dry_run:
        for job in jobs:
            _run_one_finetune_job(
                job, manager, run_dir, top10_df, combos, finetune_config,
                finetune_dir, logs_dir, status_dir, scratch_dir,
                batch_size, mini_batch_size, epochs, dry_run, lock, drug_smiles_path,
            )
    else:
        # Pool size = max concurrent hyperparameter combos (each subprocess is independent).
        print(f"[finetune] parallel dispatch: {len(jobs)} jobs, max_parallel={max_parallel}")
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [
                pool.submit(
                    _run_one_finetune_job,
                    job, manager, run_dir, top10_df, combos, finetune_config,
                    finetune_dir, logs_dir, status_dir, scratch_dir,
                    batch_size, mini_batch_size, epochs, dry_run, lock, drug_smiles_path,
                )
                for job in jobs
            ]
            for fut in futures:
                fut.result()

    _refresh_running_report(
        run_dir,
        note=f"Finetune stage batch finished (max_parallel={max_parallel}, batch_size={batch_size}).",
    )


def run_aggregate_stage(run_dir: str, dry_run: bool = False) -> str:
    finetune_root = _resolve_path(os.path.join(run_dir, "finetune"))
    input_candidates = glob(os.path.join(finetune_root, "**", "parameter_comparison_tcga_focus.csv"), recursive=True)
    if not input_candidates:
        raise FileNotFoundError(f"No parameter_comparison_tcga_focus.csv found under {finetune_root}")
    frames = [pd.read_csv(path) for path in input_candidates]
    merged = pd.concat(frames, ignore_index=True)
    aggregate_dir = _resolve_path(os.path.join(run_dir, "aggregate"))
    os.makedirs(aggregate_dir, exist_ok=True)
    merged_input = os.path.join(aggregate_dir, "merged_finetune_tcga_focus.csv")
    merged.to_csv(merged_input, index=False)
    output_path = os.path.join(aggregate_dir, "aggregate_scores.csv")
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "aggregate_pretrain_tcga_scores.py"),
        "--input",
        merged_input,
        "--output",
        output_path,
        "--top_n",
        "10",
    ]
    return_code = _run_command(cmd, os.path.join(aggregate_dir, "aggregate.log"), dry_run=dry_run)
    if not dry_run and return_code != 0:
        raise RuntimeError(f"aggregate failed with code {return_code}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("optimization_runner")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate sweep configs and pretrain manifest")
    gen.add_argument("--sweep-spec", default="config/pretrain_sweeps/vaewc_proto_infonce_round1.json")
    gen.add_argument("--run-dir", default="result/optimization_runs/vaewc_proto_infonce_round1")
    gen.add_argument("--force", action="store_true")

    pre = sub.add_parser("pretrain", help="Run pending pretrain jobs sequentially")
    pre.add_argument("--manifest", required=True)
    pre.add_argument("--run-dir", required=True)
    pre.add_argument("--device", default="cuda:0")
    pre.add_argument("--dry-run", action="store_true")
    pre.add_argument("--rerun-completed", action="store_true")
    pre.add_argument("--smoke-test", action="store_true")
    pre.add_argument("--batch-size", type=int, default=None, help="Override pretrain batch size (e.g. 2048)")
    pre.add_argument("--max-parallel", type=int, default=1, help="Concurrent pretrain subprocesses on GPU")

    sel = sub.add_parser("select", help="Visualize/filter and build Top-10 with controls")
    sel.add_argument("--run-dir", required=True)
    sel.add_argument("--result-dir", default=None)
    sel.add_argument("--filter-config", default="config/visualize_vaewc_filter.json")
    sel.add_argument("--no-filter", action="store_true")
    sel.add_argument("--min-passing", type=int, default=10, help="Min experiments passing filter before finetune")
    sel.add_argument("--require-controls", type=int, default=2, help="Min lambda_proto=0 controls in filtered pool")
    sel.add_argument(
        "--selection-mode",
        default="score_total",
        choices=[
            "score_total",
            "round4_kmeans_first",
            "round4_weighted",
            "round4_1_structure_first",
            "round5_structure_first",
            "round6_sweetspot",
            "round7_diverse_downstream_probe",
            "round8_architecture_broad_probe",
            "round10_cond_adv_qc",
            "round11_stability_qc",
            "round12_proto_alignment_qc",
            "round13_proto_response_qc",
            "round14_vicreg_stabilizer_qc",
            "round15_repro_rescue_qc",
            "round16_bruteforce_qc",
        ],
        help=(
            "Top-K ranking "
            "(Round 7: round7_diverse_downstream_probe; "
            "Round 8: round8_architecture_broad_probe; "
            "Round 10: round10_cond_adv_qc; "
            "Round 11: round11_stability_qc; "
            "Round 12: round12_proto_alignment_qc; "
            "Round 13: round13_proto_response_qc; "
            "Round 14: round14_vicreg_stabilizer_qc; "
            "Round 15: round15_repro_rescue_qc; "
            "Round 16: round16_bruteforce_qc)"
        ),
    )
    sel.add_argument(
        "--exclude-proto-ineffective",
        action="store_true",
        help="Drop checkpoints where best_gan_epoch < proto_start_epoch while lambda_proto>0",
    )
    sel.add_argument(
        "--force-baseline-models",
        default=None,
        help="Comma-separated model IDs to force into selection (e.g. exp_018,exp_746)",
    )
    sel.add_argument("--top-k", type=int, default=10, help="Top-K models for selection (Round 5: 10–15)")
    sel.add_argument(
        "--result-dirs",
        default=None,
        help="Comma-separated extra pretrain result dirs to merge before selection",
    )
    sel.add_argument("--run-tag", default=None, help="Optional run tag recorded in running report notes")

    ft = sub.add_parser("finetune", help="Dispatch finetune jobs for Top-10")
    ft.add_argument("--manifest", required=True)
    ft.add_argument("--run-dir", required=True)
    ft.add_argument("--top10", default=None, help="Top-K selection CSV (not required for --round13-mode)")
    ft.add_argument("--finetune-config", default="config/params_finetune_mini.json")
    ft.add_argument("--batch-size", type=int, default=2048)
    ft.add_argument("--mini-batch-size", type=int, default=512)
    ft.add_argument("--epochs", type=int, default=1000, help="Finetune epochs (use low value for smoke tests)")
    ft.add_argument(
        "--drug-smiles-path",
        default=FIXED_DRUG_SMILES_AACDR_EXTENDED,
        help="Drug SMILES CSV forwarded to step1 finetune subprocesses",
    )
    ft.add_argument("--max-parallel", type=int, default=1, help="Concurrent finetune subprocesses")
    ft.add_argument("--dry-run", action="store_true")
    ft.add_argument("--rerun-completed", action="store_true")
    ft.add_argument("--build-manifest-only", action="store_true")
    ft.add_argument("--force-manifest", action="store_true")
    ft.add_argument(
        "--round13-mode",
        action="store_true",
        help="Round 13 finetune dispatch using manifest model_select_path (no top10 CSV)",
    )

    agg = sub.add_parser("aggregate", help="Aggregate downstream finetune scores")
    agg.add_argument("--run-dir", required=True)
    agg.add_argument("--dry-run", action="store_true")

    rep = sub.add_parser("report", help="Generate final optimization reports")
    rep.add_argument("--run-dir", required=True)

    full = sub.add_parser("full", help="Run generate→pretrain→select→finetune→aggregate→report")
    full.add_argument("--run-dir", default="result/optimization_runs/vaewc_proto_infonce_round1")
    full.add_argument("--sweep-spec", default="config/pretrain_sweeps/vaewc_proto_infonce_round1.json")
    full.add_argument("--dry-run", action="store_true")
    full.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = _resolve_path(getattr(args, "run_dir", "result/optimization_runs/vaewc_proto_infonce_round1"))

    if args.command == "generate":
        manifest_path, _ = generate_configs(
            args.sweep_spec,
            manifest_dir=os.path.join(run_dir, "manifests"),
            force=args.force,
        )
        print(manifest_path)
        return

    if args.command == "pretrain":
        _refresh_running_report(run_dir, note="Pretrain stage started.")
        run_pretrain_stage(
            args.manifest,
            run_dir,
            device=args.device,
            dry_run=args.dry_run,
            rerun_completed=args.rerun_completed,
            smoke_test=args.smoke_test,
            batch_size=args.batch_size,
            max_parallel=args.max_parallel,
        )
        return

    if args.command == "select":
        result_dir = args.result_dir or os.path.join(run_dir, "pretrain")
        force_baselines = None
        if args.force_baseline_models:
            force_baselines = [x.strip() for x in args.force_baseline_models.split(",") if x.strip()]
        extra_dirs = None
        if args.result_dirs:
            extra_dirs = [x.strip() for x in args.result_dirs.split(",") if x.strip()]
        try:
            write_selection_outputs(
                run_dir,
                result_dir,
                filter_config=args.filter_config,
                no_filter=args.no_filter,
                min_passing=args.min_passing,
                require_controls=args.require_controls,
                selection_mode=args.selection_mode,
                exclude_proto_ineffective=args.exclude_proto_ineffective,
                top_k=args.top_k,
                force_baseline_models=force_baselines,
                result_dirs=extra_dirs,
            )
        except SelectionInsufficientError as err:
            print(f"[select] INSUFFICIENT: {err}", file=sys.stderr)
            note = f"Selection insufficient: {err}"
            if getattr(args, "run_tag", None):
                note = f"[{args.run_tag}] {note}"
            _refresh_running_report(run_dir, note=note)
            sys.exit(2)
        note = "Selection stage completed (filter passed)."
        if getattr(args, "run_tag", None):
            note = f"[{args.run_tag}] {note} mode={args.selection_mode}"
        _refresh_running_report(run_dir, note=note)
        return

    if args.command == "finetune":
        top10_path = getattr(args, "top10", None)
        if args.build_manifest_only or not os.path.exists(_resolve_path(args.manifest)):
            if not top10_path:
                raise SystemExit("finetune --top10 is required unless using a prebuilt Round 13 manifest")
            manifest_path = build_finetune_manifest(
                top10_path,
                run_dir,
                finetune_config=args.finetune_config,
                force=args.force_manifest,
            )
            print(f"Finetune manifest: {manifest_path}")
            if args.build_manifest_only:
                return
            args.manifest = manifest_path
        if not top10_path and not getattr(args, "round13_mode", False):
            manifest_df = pd.read_csv(_resolve_path(args.manifest))
            if "model_select_path" not in manifest_df.columns or manifest_df["model_select_path"].fillna("").eq("").all():
                raise SystemExit("finetune requires --top10 or --round13-mode with model_select_path manifest")
        _refresh_running_report(run_dir, note="Finetune stage started.")
        if getattr(args, "round13_mode", False):
            run_round13_finetune_stage(
                args.manifest,
                run_dir,
                finetune_config=args.finetune_config,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                epochs=args.epochs,
                dry_run=args.dry_run,
                rerun_completed=args.rerun_completed,
                max_parallel=args.max_parallel,
                drug_smiles_path=args.drug_smiles_path,
            )
            return
        run_finetune_stage(
            args.manifest,
            run_dir,
            top10_path,
            finetune_config=args.finetune_config,
            batch_size=args.batch_size,
            mini_batch_size=args.mini_batch_size,
            epochs=args.epochs,
            dry_run=args.dry_run,
            rerun_completed=args.rerun_completed,
            max_parallel=args.max_parallel,
            drug_smiles_path=args.drug_smiles_path,
        )
        return

    if args.command == "aggregate":
        run_aggregate_stage(run_dir, dry_run=args.dry_run)
        _refresh_running_report(run_dir, note="Aggregation completed.")
        return

    if args.command == "report":
        generate_final_reports(run_dir)
        _refresh_running_report(run_dir, note="Final reports generated.")
        return

    if args.command == "full":
        manifest_path, _ = generate_configs(args.sweep_spec, manifest_dir=os.path.join(run_dir, "manifests"))
        run_pretrain_stage(manifest_path, run_dir, dry_run=args.dry_run, smoke_test=args.smoke_test)
        write_selection_outputs(run_dir, os.path.join(run_dir, "pretrain"))
        top10_path = os.path.join(run_dir, "selection", "pretrain_top10.csv")
        ft_manifest = build_finetune_manifest(top10_path, run_dir, force=True)
        run_finetune_stage(ft_manifest, run_dir, top10_path, dry_run=args.dry_run)
        if not args.dry_run:
            run_aggregate_stage(run_dir)
            generate_final_reports(run_dir)
        return


if __name__ == "__main__":
    main()
