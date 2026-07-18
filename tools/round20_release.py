#!/usr/bin/env python3
"""Round 20 Stage 20E: release archive + hash audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result/optimization_runs/round20_unseen_drug_closure"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def build_release(
    *,
    model_lock_path: Path,
    tcga_dir: Path,
    output_dir: Path,
) -> dict:
    lock = json.loads(Path(model_lock_path).read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED":
        raise ValueError("model lock not LOCKED")
    output_dir = Path(output_dir)
    for sub in (
        "checkpoints", "encoder", "projections", "features", "splits",
        "configs", "metrics", "predictions", "reports", "hashes",
    ):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # Copy core artifacts
    _copy(Path(model_lock_path), output_dir / "configs" / "final_model_lock.json")
    _copy(RESULT_ROOT / "stage20_0/resolved_e3.json", output_dir / "configs" / "resolved_e3.json")
    _copy(RESULT_ROOT / "stage20_0/stage20_0_go.json", output_dir / "configs" / "stage20_0_go.json")
    _copy(
        RESULT_ROOT / "stage20a_dimension/stage20a_dimension_decision.json",
        output_dir / "reports" / "stage20a_dimension_decision.json",
    )
    _copy(
        RESULT_ROOT / "stage20b_predictor/stage20b_guardrail_report.json",
        output_dir / "reports" / "stage20b_guardrail_report.json",
    )
    feat = Path(lock["selected_context"]["feature_dir"])
    _copy(feat, output_dir / "features" / feat.name)
    ctx_dim = int(lock["selected_context"]["dimension"])
    proj = RESULT_ROOT / f"projections/context{ctx_dim}"
    if proj.is_dir():
        _copy(proj, output_dir / "projections" / f"context{ctx_dim}")
    for seed in (52, 62, 72):
        sp = RESULT_ROOT / "splits" / f"round20a_drug_heldout_seed{seed}_assignments.csv"
        _copy(sp, output_dir / "splits" / sp.name)

    # Checkpoints for locked model
    ctx = lock["selected_context"]["id"]
    cand = lock["selected_model"]["candidate_id"]
    ckpt_paths: List[Path] = []
    if cand == "B_E3":
        root = RESULT_ROOT / "stage20a_dimension/jobs"
        pattern = f"r20a__A_{ctx}_E3__ss{{seed}}__f{{fold}}__ms101"
    else:
        root = RESULT_ROOT / "stage20b_predictor/jobs"
        pattern = f"r20b__B_GATED__{ctx}__ss{{seed}}__f{{fold}}__ms101"
    for seed in (52, 62, 72):
        for fold in range(5):
            src = root / pattern.format(seed=seed, fold=fold) / "best_checkpoint.pt"
            dst = output_dir / "checkpoints" / f"seed{seed}_fold{fold}.pt"
            _copy(src, dst)
            ckpt_paths.append(dst)

    # TCGA outputs
    tcga_dir = Path(tcga_dir)
    if tcga_dir.is_dir():
        for p in tcga_dir.glob("*.csv"):
            _copy(p, output_dir / "predictions" / p.name)
        for p in tcga_dir.glob("*.json"):
            _copy(p, output_dir / "metrics" / p.name)

    # Environment snapshot (best-effort inside Docker)
    env_dir = output_dir
    try:
        (env_dir / "environment.txt").write_text(
            subprocess.check_output(["python3", "--version"], text=True), encoding="utf-8"
        )
        (env_dir / "requirements.lock.txt").write_text(
            subprocess.check_output(["pip", "freeze"], text=True), encoding="utf-8"
        )
        (env_dir / "cuda_info.txt").write_text(
            subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        (env_dir / "environment_error.txt").write_text(str(exc), encoding="utf-8")

    # Cards
    (output_dir / "MODEL_CARD.md").write_text(
        f"""# Round 20 Locked Model Card

## Purpose
Unseen-drug response prediction (drug-held-out optimised).

## Selected configuration
- Context: {lock['selected_context']['id']} (omics dim {lock['selected_context']['omics_dimension']})
- Drug encoder: D0 GIN32
- Predictor: {lock['selected_model']['predictor_type']}
- Candidate: {lock['selected_model']['candidate_id']}
- Reason: {lock['selection_reason']}

## Validation
Repeated drug-held-out, seeds 52/62/72 × 5 folds, model seed 101.

## TCGA
Post-selection evaluation only. Not used for model selection.

## Limitations
- Not validated for unseen cancer type.
- Omics encoder remains frozen; raw-omics path is capability-only.
- Gate values (if gated) are not causal importance scores.
""",
        encoding="utf-8",
    )
    (output_dir / "DATASET_CARD.md").write_text(
        """# Round 20 Dataset Card

- Eligible development rows from Round 19 development split.
- Drug identity: canonical SMILES grouping (alias/salt-safe).
- Split policy: GroupKFold drug-held-out, seeds 52/62/72.
- Projection fit population: source-only (inherited Round 17/19 contract).
- TCGA response: existing five-target Round 18/19 format, not rebuilt.
""",
        encoding="utf-8",
    )
    (output_dir / "INFERENCE_GUIDE.md").write_text(
        """# Round 20 Inference Guide

## Frozen latent path (official)
1. Load precomputed Z64 and locked context features → O2.
2. Encode drug SMILES with D0 GIN.
3. Run locked predictor.
4. For CV ensemble: arithmetic mean of probabilities across 15 fold checkpoints.

## Raw-omics path (capability)
`Round20OmicsAdapter(mode=\"raw_omics\")` is archived but was not unfrozen in Round 20.
""",
        encoding="utf-8",
    )
    (output_dir / "LIMITATIONS.md").write_text(
        """# Limitations
- No unseen-cancer validation in Round 20.
- No encoder unfreeze search.
- TCGA must not be used to revisit C16/C32 or E3/gated selection.
""",
        encoding="utf-8",
    )

    artifacts = []
    hashes = {}
    for p in sorted(output_dir.rglob("*")):
        if p.is_file() and p.name != "RELEASE_MANIFEST.json":
            rel = str(p.relative_to(output_dir))
            digest = _sha256_file(p)
            artifacts.append({"path": rel, "sha256": digest, "size": p.stat().st_size})
            hashes[rel] = digest
    _write_json(output_dir / "hashes" / "artifact_hashes.json", hashes)

    manifest = {
        "project": "drug-DCF",
        "round": "20",
        "release_status": "LOCKED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "context": lock["selected_context"]["id"],
            "context_dim": lock["selected_context"]["dimension"],
            "omics_dim": lock["selected_context"]["omics_dimension"],
            "predictor": lock["selected_model"]["predictor_type"],
            "drug_encoder": "D0",
            "candidate_id": lock["selected_model"]["candidate_id"],
        },
        "validation": {
            "split_type": "drug_held_out",
            "split_seeds": [52, 62, 72],
            "folds": 5,
            "guardrails_passed": True,
        },
        "tcga": {
            "inference_complete": (tcga_dir / "stage20d_tcga_summary.json").is_file(),
            "predictions_path": "predictions/",
            "metrics_path": "metrics/",
        },
        "capabilities": {
            "frozen_latent_inference": True,
            "raw_omics_forward": True,
            "encoder_unfreeze_supported": True,
            "encoder_unfreeze_validated_in_round20": False,
        },
        "artifacts": artifacts,
        "n_checkpoints": len(ckpt_paths),
    }
    _write_json(output_dir / "RELEASE_MANIFEST.json", manifest)
    return manifest


def audit_release(*, release_dir: Path, strict: bool = True) -> dict:
    release_dir = Path(release_dir)
    manifest_path = release_dir / "RELEASE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = json.loads((release_dir / "hashes" / "artifact_hashes.json").read_text(encoding="utf-8"))
    errors = []
    for rel, expected in hashes.items():
        path = release_dir / rel
        if not path.is_file():
            errors.append(f"missing:{rel}")
            continue
        got = _sha256_file(path)
        if got != expected:
            errors.append(f"hash_mismatch:{rel}")
    required = [
        "configs/final_model_lock.json",
        "MODEL_CARD.md",
        "DATASET_CARD.md",
        "INFERENCE_GUIDE.md",
        "LIMITATIONS.md",
    ]
    for rel in required:
        if not (release_dir / rel).is_file():
            errors.append(f"required_missing:{rel}")
    n_ckpt = len(list((release_dir / "checkpoints").glob("*.pt")))
    if n_ckpt != 15:
        errors.append(f"checkpoint_count={n_ckpt}")
    status = "PASS" if not errors else "FAIL"
    report = {
        "status": status,
        "n_hashed_artifacts": len(hashes),
        "n_checkpoints": n_ckpt,
        "errors": errors,
        "release_status": manifest.get("release_status"),
    }
    _write_json(release_dir / "hashes" / "release_audit.json", report)
    print(f"ROUND20_RELEASE_AUDIT={status}")
    if strict and status != "PASS":
        raise SystemExit(1)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model-lock", required=True)
    b.add_argument("--tcga-dir", required=True)
    b.add_argument("--output-dir", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--release-dir", required=True)
    a.add_argument("--strict", action="store_true", default=True)
    args = p.parse_args()
    if args.cmd == "build":
        print(json.dumps(build_release(
            model_lock_path=Path(args.model_lock),
            tcga_dir=Path(args.tcga_dir),
            output_dir=Path(args.output_dir),
        ), indent=2)[:2000])
    else:
        audit_release(release_dir=Path(args.release_dir), strict=args.strict)


if __name__ == "__main__":
    main()
