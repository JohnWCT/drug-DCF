#!/usr/bin/env python3
"""Round 20 Stage 20D: locked TCGA response inference (strict, ensemble)."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TCGA_TARGETS = {
    "gdsc_intersect13": "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv",
    "tcga_only3": "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv",
    "dapl": "data/TCGA/TCGA_drug_response_from_DAPL.csv",
    "aacdr_tcga_only": "data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv",
    "aacdr_gdsc_intersect": "data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_lock(path: Path) -> dict:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED":
        raise ValueError(f"Model lock not LOCKED: {path}")
    return lock


def build_tcga_o2_features(lock: dict) -> Dict[str, np.ndarray]:
    """Build TCGA O2 vectors for the locked context dimension (transform-only).

    C16: reuse Round 19 O3 TCGA artifact sliced to 80-d (identical to Round 19 O2 path).
    C32: Z64 + StandardScaler(PCA32(raw_context)) using source-fitted PCA/scaler only.
    """
    from tools.extract_round13_proto_features import (
        _load_or_extract_prototypes,
        _sample_cancer_id,
    )
    from tools.prototype_response_features import (
        build_raw_context_vector,
        get_own_source_target_vectors,
    )
    from tools.round19_feature_builder import OMICS_ALIAS
    from tools.round20_context_rebuild import DEFAULT_CHECKPOINT
    from tools.round9_diagnostics_common import _load_cancer_maps

    ctx_id = lock["selected_context"]["id"]
    omics_dim = int(lock["selected_context"]["omics_dimension"])
    feature_dir = Path(lock["selected_context"]["feature_dir"])

    o3_tcga = (
        PROJECT_ROOT
        / "result/optimization_runs/round19_factorial/features"
        / OMICS_ALIAS["O3"]
        / "tcga_latent_proto.pkl"
    )
    with o3_tcga.open("rb") as f:
        o3 = pickle.load(f)

    if ctx_id == "C16":
        latent = {
            str(k): np.asarray(v, dtype=np.float32).reshape(-1)[:80].copy()
            for k, v in o3.items()
        }
    elif ctx_id == "C32":
        z_by_id = {
            str(k): np.asarray(v, dtype=np.float32).reshape(-1)[:64].copy()
            for k, v in o3.items()
        }
        with (feature_dir / "projection_model.pkl").open("rb") as f:
            projection = pickle.load(f)
        meta = json.loads((feature_dir / "feature_metadata.json").read_text(encoding="utf-8"))
        scaler_cfg = meta.get("context_scaler")
        if not scaler_cfg or "mean" not in scaler_cfg or "scale" not in scaler_cfg:
            raise RuntimeError("C32 feature_metadata missing source-fitted context_scaler")
        scaler_mean = np.asarray(scaler_cfg["mean"], dtype=np.float32)
        scaler_scale = np.asarray(scaler_cfg["scale"], dtype=np.float32)
        if scaler_mean.shape != (32,) or scaler_scale.shape != (32,):
            raise AssertionError(
                f"Unexpected scaler shapes mean={scaler_mean.shape} scale={scaler_scale.shape}"
            )

        proto_cache = (
            PROJECT_ROOT
            / "result/optimization_runs/round17r_18class/features/r13_exp_008"
            / "own_proto_context_projected_16/_proto_cache"
        )
        proto = _load_or_extract_prototypes(
            str(DEFAULT_CHECKPOINT), str(proto_cache), strict=False
        )
        mapping = proto["cancer_type_mapping"]
        name_to_id = mapping.get("name_to_id", {})
        ccle_map, tcga_map = _load_cancer_maps()

        raw_rows = []
        keep_ids = []
        skipped = 0
        for sid, z in z_by_id.items():
            try:
                cancer_id = _sample_cancer_id(sid, "target", ccle_map, tcga_map, name_to_id)
                vecs = get_own_source_target_vectors(
                    int(cancer_id),
                    proto["source_anchor_prototypes"],
                    proto["target_prototypes"],
                    source_initialized=proto["source_initialized"],
                    target_initialized=proto["target_initialized"],
                    strict=False,
                    latent_dim=64,
                )
                raw_rows.append(
                    build_raw_context_vector(z, vecs["source_anchor"], vecs["target_proto"])
                )
                keep_ids.append(sid)
            except Exception:
                skipped += 1
                continue
        if not raw_rows:
            raise RuntimeError("Failed to build any TCGA raw context rows for C32")
        raw_mat = np.stack(raw_rows, axis=0).astype(np.float32)
        ctx = projection.transform(raw_mat).astype(np.float32)
        ctx = (ctx - scaler_mean) / np.maximum(scaler_scale, 1e-8)
        latent = {
            sid: np.concatenate([z_by_id[sid], ctx[i]], axis=0).astype(np.float32)
            for i, sid in enumerate(keep_ids)
        }
        print(
            json.dumps(
                {
                    "tcga_c32_feature_build": True,
                    "n_kept": len(latent),
                    "n_skipped": skipped,
                    "omics_dim": omics_dim,
                }
            ),
            flush=True,
        )
    else:
        raise ValueError(f"Unsupported locked context: {ctx_id}")

    for vec in latent.values():
        if vec.shape != (omics_dim,):
            raise AssertionError(f"TCGA O2 dim {vec.shape} != {omics_dim}")
    return latent


def _resolve_checkpoints(lock: dict) -> List[Path]:
    """Locate the 15 fold checkpoints for the locked candidate on winner context."""
    ctx = lock["selected_context"]["id"]
    cand = lock["selected_model"]["candidate_id"]
    paths: List[Path] = []
    if cand == "B_E3":
        root = PROJECT_ROOT / "result/optimization_runs/round20_unseen_drug_closure/stage20a_dimension/jobs"
        for seed in (52, 62, 72):
            for fold in range(5):
                p = root / f"r20a__A_{ctx}_E3__ss{seed}__f{fold}__ms101" / "best_checkpoint.pt"
                paths.append(p)
    else:
        root = PROJECT_ROOT / "result/optimization_runs/round20_unseen_drug_closure/stage20b_predictor/jobs"
        for seed in (52, 62, 72):
            for fold in range(5):
                p = root / f"r20b__B_GATED__{ctx}__ss{seed}__f{fold}__ms101" / "best_checkpoint.pt"
                paths.append(p)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing locked checkpoints ({len(missing)}): {missing[:3]}")
    return paths


def _prepare_tcga_frame(
    target_path: Path,
    *,
    target_key: str,
    latent: Dict[str, np.ndarray],
    drug_smiles_path: str,
) -> pd.DataFrame:
    from tools.finetune_tcga_eval import load_tcga_response_csv
    from tools.round18_eligible_data import load_smiles_lookup

    raw = load_tcga_response_csv(str(target_path)).reset_index(drop=True).copy()
    drug_column = "drug_name" if "drug_name" in raw.columns else "DRUG_NAME"
    required = {"Patient_id", "Label", drug_column}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"TCGA target missing columns {sorted(missing)}: {target_path}")

    patient_to_latent = {}
    for latent_key in latent:
        parts = str(latent_key).split("-")
        if len(parts) >= 3:
            patient_to_latent.setdefault("-".join(parts[:3]), latent_key)
        patient_to_latent.setdefault(str(latent_key), latent_key)

    smiles_lookup = load_smiles_lookup(drug_smiles_path)
    rows = []
    miss_latent = miss_smiles = 0
    for source_row_id, row in raw.iterrows():
        patient_id = str(row["Patient_id"])
        latent_key = patient_to_latent.get(patient_id)
        if latent_key is None or latent_key not in latent:
            miss_latent += 1
            continue
        drug_name = str(row[drug_column]).strip()
        inline_smiles = str(row.get("smiles", "")).strip()
        if inline_smiles.lower() == "nan":
            inline_smiles = ""
        smiles = inline_smiles or smiles_lookup.get(drug_name.lower(), "")
        if not smiles:
            miss_smiles += 1
            continue
        rows.append(
            {
                "_row_id": len(rows),
                "ModelID": patient_id,
                "_latent_key": str(latent_key),
                "Patient_id": patient_id,
                "DRUG_NAME": drug_name,
                "drug_name": drug_name,
                "Label": int(row["Label"]),
                "target_key": target_key,
                "drug_smiles_key": drug_name.lower(),
                "smiles": smiles,
            }
        )
    if not rows:
        raise RuntimeError(
            f"No usable TCGA rows for {target_key} "
            f"(miss_latent={miss_latent} miss_smiles={miss_smiles})"
        )
    frame = pd.DataFrame(rows)
    frame.attrs["n_miss_latent"] = miss_latent
    frame.attrs["n_miss_smiles"] = miss_smiles
    return frame


def run_tcga_inference(
    *,
    model_lock_path: Path,
    output_dir: Path,
    settings_path: Path = PROJECT_ROOT / "config/round19_factorial_settings.json",
    strict: bool = True,
) -> dict:
    from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, metrics_to_jsonable
    from tools.round18_response_head import Round18ResponseHead
    from tools.round19_dataset import Round19ResponseDataset, round19_collate_fn
    from tools.round19_drug_encoders import build_drug_encoder
    from tools.round19_fusion_models import build_predictor
    from tools.round19_train_loop import forward_round19_batch

    lock = _load_lock(model_lock_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    omics_dim = int(lock["selected_context"]["omics_dimension"])
    predictor_kind = (
        "gated_pooled_fusion"
        if lock["selected_model"]["candidate_id"] == "B_GATED"
        else "pooled_e3"
    )

    latent = build_tcga_o2_features(lock)
    # Remap so dataset ModelID (Patient_id) hits the vector.
    patient_latent: Dict[str, np.ndarray] = {}
    for key, vec in latent.items():
        parts = str(key).split("-")
        if len(parts) >= 3:
            patient_latent.setdefault("-".join(parts[:3]), vec)
        patient_latent[str(key)] = vec

    ckpts = _resolve_checkpoints(lock)
    preflight = {
        "status": "PASS",
        "n_tcga_latents": len(latent),
        "omics_dim": omics_dim,
        "n_checkpoints": len(ckpts),
        "checkpoint_sha256": [_sha256_file(p) for p in ckpts],
        "model_lock_sha256": _sha256_file(model_lock_path),
        "context_id": lock["selected_context"]["id"],
        "candidate_id": lock["selected_model"]["candidate_id"],
    }
    _write_json(output_dir / "stage20d_tcga_preflight.json", preflight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drug_cfg = settings["drug_reps"]["D0"]
    per_target_metrics = {}

    for target_key, target_rel in TCGA_TARGETS.items():
        target_path = PROJECT_ROOT / target_rel
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        frame = _prepare_tcga_frame(
            target_path,
            target_key=target_key,
            latent=latent,
            drug_smiles_path=settings["drug_smiles_path"],
        )
        # latent_by_id keyed by ModelID (= Patient_id)
        row_latent = {
            str(row["ModelID"]): patient_latent[str(row["ModelID"])]
            for _, row in frame.iterrows()
        }
        ds = Round19ResponseDataset(
            frame,
            feature_dir=str(lock["selected_context"]["feature_dir"]),
            drug_smiles_path=settings["drug_smiles_path"],
            encoder_type="gin",
            with_bonds=False,
            latent_by_id=row_latent,
            omics_id="O2",
        )
        if ds.omics_dim != omics_dim:
            raise AssertionError(f"dataset omics_dim {ds.omics_dim} != lock {omics_dim}")
        loader = DataLoader(
            ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=round19_collate_fn
        )

        ckpt_prob_cols = []
        ckpt_rows_ref = None
        for ci, ckpt_path in enumerate(ckpts):
            blob = torch.load(ckpt_path, map_location=device)
            state = blob["model_state_dict"] if "model_state_dict" in blob else blob
            encoder = build_drug_encoder(
                "gin",
                node_hidden_dim=int(drug_cfg["node_hidden_dim"]),
                graph_output_dim=int(drug_cfg["graph_output_dim"]),
            ).to(device)
            if predictor_kind == "pooled_e3":
                fusion = build_predictor(
                    "P0", omics_dim=omics_dim, drug_dim=32, node_dim=32
                ).to(device)
                head = Round18ResponseHead(input_dim=fusion.output_dim).to(device)
            else:
                from tools.round20_gated_fusion import GatedPooledFusionPredictor

                fusion = GatedPooledFusionPredictor(omics_dim=omics_dim).to(device)
                head = torch.nn.Identity().to(device)
            encoder.load_state_dict(state["encoder"], strict=True)
            fusion.load_state_dict(state["fusion"], strict=True)
            if predictor_kind == "pooled_e3":
                head.load_state_dict(state["head"], strict=True)
            encoder.eval()
            fusion.eval()
            head.eval()

            rows = []
            with torch.no_grad():
                for batch in loader:
                    omics = batch["omics"].to(device)
                    local = dict(batch)
                    local["drug_batch"] = local["drug_batch"].to(device)
                    repr_vec = forward_round19_batch(
                        encoder=encoder,
                        fusion=fusion,
                        encoder_type="gin",
                        predictor_id="P0",
                        omics=omics,
                        batch=local,
                    )
                    logits = head(repr_vec).view(-1)
                    probs = torch.sigmoid(logits)
                    for i in range(probs.size(0)):
                        rows.append(
                            {
                                "_row_id": int(batch["_row_id"][i]),
                                "sample_id": str(batch["ModelID"][i]),
                                "drug_name": str(batch["drug_name"][i]),
                                "true_label": int(batch["label"][i].item()),
                                f"prob_ckpt{ci}": float(probs[i].item()),
                            }
                        )
            cdf = pd.DataFrame(rows)
            if ckpt_rows_ref is None:
                ckpt_rows_ref = cdf[
                    ["_row_id", "sample_id", "drug_name", "true_label"]
                ].copy()
            ckpt_prob_cols.append(cdf.set_index("_row_id")[f"prob_ckpt{ci}"])

        ens = ckpt_rows_ref.copy()
        for col in ckpt_prob_cols:
            ens = ens.join(col, on="_row_id")
        prob_cols = [c for c in ens.columns if c.startswith("prob_ckpt")]
        ens["prediction_probability"] = ens[prob_cols].mean(axis=1)
        ens["checkpoint_count"] = len(prob_cols)
        ens["model_lock_id"] = lock["selected_model"]["candidate_id"]
        ens["context_id"] = lock["selected_context"]["id"]
        ens["predictor_id"] = predictor_kind
        ens["target_key"] = target_key
        ens.to_csv(output_dir / f"predictions_by_checkpoint__{target_key}.csv", index=False)
        ens_out = ens[
            [
                "_row_id",
                "sample_id",
                "drug_name",
                "true_label",
                "prediction_probability",
                "checkpoint_count",
                "model_lock_id",
                "context_id",
                "predictor_id",
                "target_key",
            ]
        ].rename(columns={"_row_id": "row_id", "drug_name": "drug_id"})
        metric_df = ens.rename(
            columns={
                "drug_name": "DRUG_NAME",
                "true_label": "Label",
                "prediction_probability": "probability",
            }
        )
        metrics = calculate_robust_drug_macro_metrics(metric_df)
        per_target_metrics[target_key] = metrics_to_jsonable(metrics)
        ens_out.to_csv(output_dir / f"predictions_ensemble__{target_key}.csv", index=False)
        print(
            json.dumps(
                {
                    "target_key": target_key,
                    "n_rows": int(len(ens_out)),
                    "DrugMacro_AUC": per_target_metrics[target_key].get("DrugMacro_AUC"),
                    "miss_latent": frame.attrs.get("n_miss_latent"),
                    "miss_smiles": frame.attrs.get("n_miss_smiles"),
                }
            ),
            flush=True,
        )

    summary = {
        "status": "COMPLETE",
        "n_targets": len(TCGA_TARGETS),
        "metrics_by_target": per_target_metrics,
        "preflight": str(output_dir / "stage20d_tcga_preflight.json"),
        "model_lock": str(model_lock_path),
        "selected_context": lock["selected_context"]["id"],
        "selected_model": lock["selected_model"]["candidate_id"],
    }
    _write_json(output_dir / "stage20d_tcga_summary.json", summary)
    _write_json(output_dir / "tcga_metrics.json", per_target_metrics)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-lock", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--strict", action="store_true", default=True)
    args = p.parse_args()
    print(json.dumps(run_tcga_inference(
        model_lock_path=Path(args.model_lock),
        output_dir=Path(args.output_dir),
        strict=args.strict,
    ), indent=2))


if __name__ == "__main__":
    main()
