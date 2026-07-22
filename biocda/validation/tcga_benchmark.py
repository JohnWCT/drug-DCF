"""TCGA benchmark across all BioCDA-tested models (post-hoc, not for selection)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from biocda.data.xa_dataset import biocda_collate_fn, split_z_context
from biocda.training.checkpoint import load_biocda_checkpoint
from biocda.utils.gpu import configure_gpu_efficiency
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, metrics_to_jsonable
from tools.round20_tcga import TCGA_TARGETS, _prepare_tcga_frame, build_tcga_o2_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUND20_LOCK = PROJECT_ROOT / "reports/round20_final_model_lock_public.json"
ROUND20_TCGA_METRICS = (
    PROJECT_ROOT
    / "result/optimization_runs/round20_unseen_drug_closure/stage20d_tcga/tcga_metrics.json"
)
FEATURE_DIR = (
    PROJECT_ROOT / "result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context32"
)
DRUG_SMILES = PROJECT_ROOT / "data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv"
SEEDS_R21_R23 = (17, 29, 43)


@dataclass
class ModelSpec:
    model_id: str
    display_name: str
    round_tag: str
    architecture: str
    checkpoint_paths: List[Path]
    source: str  # import_round20 | checkpoint_ensemble
    notes: str = ""


@dataclass
class TargetMetrics:
    target_key: str
    drug_macro_auc: Optional[float]
    drug_macro_auprc: Optional[float]
    global_auc: Optional[float]
    global_auprc: Optional[float]
    n_rows: int = 0


@dataclass
class ModelTcgaResult:
    spec: ModelSpec
    per_target: Dict[str, TargetMetrics] = field(default_factory=dict)

    def mean_metrics(self) -> Dict[str, Optional[float]]:
        keys = ("drug_macro_auc", "drug_macro_auprc", "global_auc", "global_auprc")
        out: Dict[str, Optional[float]] = {}
        for k in keys:
            vals = [getattr(t, k) for t in self.per_target.values() if getattr(t, k) is not None]
            out[f"mean_{k}"] = float(sum(vals) / len(vals)) if vals else None
        return out


def _resolve_round20_lock() -> dict:
    lock = json.loads(ROUND20_LOCK.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / "result/optimization_runs/round20_unseen_drug_closure"
    feat = lock["selected_context"]["feature_dir"].replace("${ROUND20_RELEASE_ROOT}", str(root))
    lock = dict(lock)
    lock["selected_context"] = dict(lock["selected_context"])
    lock["selected_context"]["feature_dir"] = feat
    lock["status"] = "LOCKED"
    return lock


def discover_model_specs() -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    # Round 20 LOCKED reference — reuse validated 15-fold TCGA ensemble
    if ROUND20_TCGA_METRICS.is_file():
        specs.append(
            ModelSpec(
                model_id="r20_predictive_locked",
                display_name="BioCDA-Predictive (R20 locked, 15-fold)",
                round_tag="round20",
                architecture="biocda-predictive-e3",
                checkpoint_paths=[],
                source="import_round20",
                notes="15-fold probability ensemble from stage20d_tcga",
            )
        )

    # Round 21
    r21_root = PROJECT_ROOT / "outputs/xa_validation"
    for mid, name in [
        ("pooled_baseline", "pooled_baseline / M0 (R21)"),
        ("biocda_xa_z", "biocda_xa_z / M1 (R21)"),
        ("biocda_xa_zc", "biocda_xa_zc / M2 (R21)"),
    ]:
        ckpts = [r21_root / f"{mid}_seed{s}" / "best.pt" for s in SEEDS_R21_R23]
        if all(p.is_file() for p in ckpts):
            specs.append(
                ModelSpec(
                    model_id=f"r21_{mid}",
                    display_name=name,
                    round_tag="round21",
                    architecture="biocda-xa-v1",
                    checkpoint_paths=ckpts,
                    source="checkpoint_ensemble",
                    notes="3-seed probability ensemble",
                )
            )

    # Round 23
    r23_root = PROJECT_ROOT / "outputs/xa_v2_closure"
    for mid, name in [
        ("biocda_predictive", "biocda_predictive / P0 (R23)"),
        ("biocda_xa_fresh", "biocda_xa_fresh / X0 (R23)"),
        ("biocda_xa_transfer", "biocda_xa_transfer / X1 (R23)"),
        ("biocda_xa_kd", "biocda_xa_kd / X2 (R23)"),
    ]:
        ckpts = [r23_root / f"{mid}_seed{s}" / "best.pt" for s in SEEDS_R21_R23]
        if all(p.is_file() for p in ckpts):
            specs.append(
                ModelSpec(
                    model_id=f"r23_{mid}",
                    display_name=name,
                    round_tag="round23",
                    architecture="biocda-xa-v2" if mid != "biocda_predictive" else "biocda-predictive-e3",
                    checkpoint_paths=ckpts,
                    source="checkpoint_ensemble",
                    notes="3-seed probability ensemble",
                )
            )
    return specs


def _load_model_from_checkpoint(ckpt_path: Path) -> torch.nn.Module:
    blob = torch.load(ckpt_path, map_location="cpu")
    model_type = blob.get("model_type", "")
    arch = blob.get("architecture_version", "")

    if arch == "biocda-predictive-e3" or model_type == "biocda_predictive":
        from biocda.models.predictive import BioCDAPredictive

        model = BioCDAPredictive()
        model.load_state_dict(blob["model_state_dict"], strict=True)
        return model

    if arch == "biocda-xa-v2" or model_type in {
        "biocda_xa_fresh",
        "biocda_xa_transfer",
        "biocda_xa_kd",
        "biocda_xa_z_only",
    }:
        from biocda.models.xa.factory import build_xa_v2

        cfg = blob.get("config") or {
            "model": {"type": model_type, "drug_encoder": {}, "cross_attention": {}, "response_head": {}}
        }
        if "model" not in cfg:
            cfg = {"model": {"type": model_type, **cfg}}
        cfg["model"]["type"] = model_type
        model = build_xa_v2(cfg, model_type=model_type)
        model.load_state_dict(blob["model_state_dict"], strict=True)
        return model

    # Round 21 v1 factory models
    from biocda.models.model_factory import build_model

    cfg = blob["config"]
    model = build_model(cfg)
    load_biocda_checkpoint(model, ckpt_path, strict=True)
    return model


def _build_tcga_dataset(frame: pd.DataFrame, patient_latent: Dict[str, Any]) -> DataLoader:
    from tools.round19_dataset import Round19ResponseDataset

    row_latent = {str(row["ModelID"]): patient_latent[str(row["ModelID"])] for _, row in frame.iterrows()}
    ds = Round19ResponseDataset(
        frame,
        feature_dir=str(FEATURE_DIR),
        drug_smiles_path=str(DRUG_SMILES),
        encoder_type="gin",
        graph_cache={},
        latent_by_id=row_latent,
        omics_id="O2",
    )
    return DataLoader(
        ds,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=biocda_collate_fn,
    )


@torch.no_grad()
def _predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval().to(device)
    rows: List[dict] = []
    use_amp = device.type == "cuda"
    for batch in loader:
        omics = batch["omics"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        drug_graph = batch["drug_graph"].to(device)
        with autocast(enabled=use_amp):
            out = model(omics, context, drug_graph, output_mode="prediction")
            probs = torch.sigmoid(out.logits.reshape(-1))
        for i in range(len(batch["labels"])):
            rows.append(
                {
                    "_row_id": int(batch["_row_id"][i]),
                    "DRUG_NAME": batch["DRUG_NAME"][i],
                    "Label": int(batch["labels"][i].item()),
                    "probability": float(probs[i].item()),
                }
            )
    return pd.DataFrame(rows)


def _ensemble_predictions(pred_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    base = pred_frames[0][["_row_id", "DRUG_NAME", "Label"]].copy()
    prob_cols = []
    for i, df in enumerate(pred_frames):
        col = f"prob{i}"
        base = base.merge(
            df[["_row_id", "probability"]].rename(columns={"probability": col}),
            on="_row_id",
            how="left",
        )
        prob_cols.append(col)
    base["probability"] = base[prob_cols].mean(axis=1)
    base["checkpoint_count"] = len(prob_cols)
    base = base.drop(columns=prob_cols)
    return base


def _metrics_from_predictions(df: pd.DataFrame) -> TargetMetrics:
    m = metrics_to_jsonable(calculate_robust_drug_macro_metrics(df))
    return TargetMetrics(
        target_key="",
        drug_macro_auc=m.get("DrugMacro_AUC"),
        drug_macro_auprc=m.get("DrugMacro_AUPRC"),
        global_auc=m.get("Global_AUC"),
        global_auprc=m.get("Global_AUPRC"),
        n_rows=len(df),
    )


def _import_round20_metrics() -> ModelTcgaResult:
    spec = ModelSpec(
        model_id="r20_predictive_locked",
        display_name="BioCDA-Predictive (R20 locked, 15-fold)",
        round_tag="round20",
        architecture="biocda-predictive-e3",
        checkpoint_paths=[],
        source="import_round20",
    )
    raw = json.loads(ROUND20_TCGA_METRICS.read_text(encoding="utf-8"))
    result = ModelTcgaResult(spec=spec)
    for target_key, m in raw.items():
        result.per_target[target_key] = TargetMetrics(
            target_key=target_key,
            drug_macro_auc=m.get("DrugMacro_AUC"),
            drug_macro_auprc=m.get("DrugMacro_AUPRC"),
            global_auc=m.get("Global_AUC"),
            global_auprc=m.get("Global_AUPRC"),
        )
    return result


def run_model_tcga(
    spec: ModelSpec,
    *,
    frames: Dict[str, pd.DataFrame],
    patient_latent: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> ModelTcgaResult:
    if spec.source == "import_round20":
        return _import_round20_metrics()

    out = ModelTcgaResult(spec=spec)
    model_dir = output_dir / spec.model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    for target_key, frame in frames.items():
        loader = _build_tcga_dataset(frame, patient_latent)
        per_ckpt: List[pd.DataFrame] = []
        for ci, ckpt in enumerate(spec.checkpoint_paths):
            model = _load_model_from_checkpoint(ckpt)
            pred = _predict_loader(model, loader, device)
            pred.to_csv(model_dir / f"predictions_{target_key}_ckpt{ci}.csv", index=False)
            per_ckpt.append(pred)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ens = _ensemble_predictions(per_ckpt)
        ens["target_key"] = target_key
        ens.to_csv(model_dir / f"predictions_ensemble_{target_key}.csv", index=False)
        tm = _metrics_from_predictions(ens)
        tm.target_key = target_key
        out.per_target[target_key] = tm
    return out


def prepare_tcga_frames(lock: dict) -> tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    latent = build_tcga_o2_features(lock)
    patient_latent: Dict[str, Any] = {}
    for key, vec in latent.items():
        parts = str(key).split("-")
        if len(parts) >= 3:
            patient_latent.setdefault("-".join(parts[:3]), vec)
        patient_latent[str(key)] = vec
    frames = {}
    for target_key, rel in TCGA_TARGETS.items():
        path = PROJECT_ROOT / rel
        frames[target_key] = _prepare_tcga_frame(
            path,
            target_key=target_key,
            latent=latent,
            drug_smiles_path=str(DRUG_SMILES),
        )
    return frames, patient_latent


def results_to_long_df(results: Sequence[ModelTcgaResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        means = res.mean_metrics()
        for target_key, tm in res.per_target.items():
            rows.append(
                {
                    "model_id": res.spec.model_id,
                    "display_name": res.spec.display_name,
                    "round": res.spec.round_tag,
                    "architecture": res.spec.architecture,
                    "target": target_key,
                    "DrugMacro_AUC": tm.drug_macro_auc,
                    "DrugMacro_AUPRC": tm.drug_macro_auprc,
                    "Global_AUC": tm.global_auc,
                    "Global_AUPRC": tm.global_auprc,
                    "n_rows": tm.n_rows,
                    "mean_DrugMacro_AUC_5targets": means.get("mean_drug_macro_auc"),
                    "mean_DrugMacro_AUPRC_5targets": means.get("mean_drug_macro_auprc"),
                    "mean_Global_AUC_5targets": means.get("mean_global_auc"),
                    "mean_Global_AUPRC_5targets": means.get("mean_global_auprc"),
                }
            )
    return pd.DataFrame(rows)


def results_to_wide_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# BioCDA TCGA Comparison (5 targets)",
        "",
        "Post-hoc descriptive evaluation only — **not used for model selection**.",
        "",
        "## Per-target metrics",
        "",
    ]
    metrics = [
        ("DrugMacro_AUC", "DrugMacro AUC"),
        ("DrugMacro_AUPRC", "DrugMacro AUPRC"),
        ("Global_AUC", "Global AUC"),
        ("Global_AUPRC", "Global AUPRC"),
    ]
    for metric_col, metric_name in metrics:
        lines.append(f"### {metric_name}")
        lines.append("")
        pivot = df.pivot_table(index="display_name", columns="target", values=metric_col, aggfunc="first")
        target_order = list(TCGA_TARGETS.keys())
        pivot = pivot.reindex(columns=target_order)
        mean_col = df.groupby("display_name")[f"mean_{metric_col}_5targets"].first()
        pivot["mean_5targets"] = mean_col
        lines.append(pivot.round(4).to_markdown())
        lines.append("")
    return "\n".join(lines)
