#!/usr/bin/env python3
"""Fail-closed pooled E3 resolver for Round 20.

Never guesses architecture from the candidate name alone. Resolution must
cross-check role lock, deployment policy, factorial settings, and checkpoint
payload fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROLE_LOCK = (
    PROJECT_ROOT
    / "result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json"
)
DEFAULT_ROLE_LOCK_POINTER = PROJECT_ROOT / "reports/round19_final_role_lock.json"
DEFAULT_DEPLOYMENT_POLICY = PROJECT_ROOT / "reports/round19_deployment_policy.json"
DEFAULT_FACTORIAL_SETTINGS = PROJECT_ROOT / "config/round19_factorial_settings.json"
DEFAULT_PROPOSAL = (
    PROJECT_ROOT
    / "result/optimization_runs/round19_factorial/reports/round19_final_role_proposal.json"
)
SOURCE_CANDIDATE_ID = "F3_best_pooled_o2"
PUBLIC_ALIAS = "E3"
REQUIRED_OMICS = "O2"
REQUIRED_DRUG = "D0"
REQUIRED_PREDICTOR = "P0"


class E3ResolutionError(RuntimeError):
    """Raised when pooled E3 cannot be uniquely resolved from artifacts."""


@dataclass(frozen=True)
class ResolvedE3:
    source: str
    reconstructed: bool
    architecture_family: str
    predictor_class: str
    predictor_id: str
    omics_id: str
    drug_encoder_id: str
    drug_encoder_training_mode: str
    pooling: str
    context_dim: int
    omics_dim: int
    graph_dim: int
    node_hidden_dim: int
    adapter_dim: int
    hidden_dims: tuple[int, ...]
    dropout: float
    activation: str
    normalization: str
    optimizer: Mapping[str, Any]
    training: Mapping[str, Any]
    checkpoint_paths: tuple[str, ...]
    config_hash: str
    source_candidate_id: str
    public_alias: str
    baseline_label: str
    is_original_e3: bool
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        payload["checkpoint_paths"] = list(self.checkpoint_paths)
        return payload


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise E3ResolutionError(f"Missing required artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise E3ResolutionError(f"Expected JSON object: {path}")
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return _sha256_bytes(blob)


def _resolve_role_lock_path(repo_root: Path) -> Path:
    primary = repo_root / "result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json"
    pointer = repo_root / "reports/round19_final_role_lock.json"
    if primary.is_file():
        return primary
    if pointer.is_file():
        payload = _load_json(pointer)
        immutable = payload.get("immutable_lock_path")
        if immutable:
            cand = repo_root / str(immutable)
            if cand.is_file():
                return cand
        return pointer
    raise E3ResolutionError("No Round 19 final role lock found")


def _extract_role_candidate(lock: Mapping[str, Any], role: str) -> str:
    roles = lock.get("roles")
    if not isinstance(roles, Mapping) or role not in roles:
        raise E3ResolutionError(f"Role lock missing role {role!r}")
    record = roles[role]
    if not isinstance(record, Mapping):
        raise E3ResolutionError(f"Role {role!r} is not an object")
    candidate = record.get("source_candidate_id") or record.get("candidate_id")
    if not candidate:
        raise E3ResolutionError(f"Role {role!r} missing candidate id")
    return str(candidate)


def _inventory_for_candidate(lock: Mapping[str, Any], candidate_id: str) -> list[str]:
    inventory = lock.get("hashes", {}).get("checkpoint_inventory", [])
    paths: list[str] = []
    for item in inventory:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("source_candidate_id")) == candidate_id:
            path = item.get("checkpoint_path")
            if path:
                paths.append(str(path))
    if not paths:
        raise E3ResolutionError(
            f"Role lock checkpoint inventory has no entries for {candidate_id}"
        )
    return paths


def _predictors_from_settings(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    predictors = settings.get("predictors")
    if not isinstance(predictors, Mapping):
        raise E3ResolutionError("factorial settings.predictors must be an object")
    return predictors


def _proposal_identity(proposal: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if proposal is None:
        return None
    evidence = proposal.get("candidate_identity_evidence", {})
    if not isinstance(evidence, Mapping):
        return None
    e3 = evidence.get(PUBLIC_ALIAS)
    if e3 is None:
        return None
    if not isinstance(e3, Mapping):
        raise E3ResolutionError("proposal E3 identity is not an object")
    return e3


def _inspect_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise E3ResolutionError("torch is required to inspect checkpoints") from exc
    if not path.is_file():
        raise E3ResolutionError(f"Checkpoint missing: {path}")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise E3ResolutionError(f"Checkpoint is not a dict: {path}")
    required = ("encoder", "fusion", "head", "drug_id", "predictor_id", "omics_id", "encoder_type")
    missing = [k for k in required if k not in obj]
    if missing:
        raise E3ResolutionError(f"Checkpoint {path} missing keys: {missing}")
    fusion = obj["fusion"]
    head = obj["head"]
    if not isinstance(fusion, Mapping) or not isinstance(head, Mapping):
        raise E3ResolutionError(f"Checkpoint {path} fusion/head must be state dict mappings")
    omics_w = fusion.get("omics_adapter.0.weight")
    drug_w = fusion.get("drug_adapter.0.weight")
    head_w = head.get("net.0.weight")
    if omics_w is None or drug_w is None or head_w is None:
        raise E3ResolutionError(
            f"Checkpoint {path} does not match AdapterMLPFusion+response-head schema"
        )
    omics_in = int(omics_w.shape[1])
    adapter_dim = int(omics_w.shape[0])
    graph_dim = int(drug_w.shape[1])
    head_in = int(head_w.shape[1])
    head_hidden = int(head_w.shape[0])
    if head_in != adapter_dim * 2:
        raise E3ResolutionError(
            f"Checkpoint head input {head_in} != 2*adapter_dim ({adapter_dim * 2})"
        )
    context_dim = omics_in - 64
    if context_dim not in (16, 32):
        # Still accept exact O2=80; Round 20 may later rebuild C32 with adapted input.
        if omics_in != 80:
            raise E3ResolutionError(
                f"Unexpected omics input dim {omics_in} in checkpoint {path}"
            )
        context_dim = 16
    return {
        "path": str(path),
        "drug_id": str(obj["drug_id"]),
        "predictor_id": str(obj["predictor_id"]),
        "omics_id": str(obj["omics_id"]),
        "encoder_type": str(obj["encoder_type"]).lower(),
        "node_hidden_dim": int(obj.get("node_hidden_dim", 32)),
        "graph_output_dim": int(obj.get("graph_output_dim", graph_dim)),
        "omics_dim": omics_in,
        "context_dim": int(context_dim),
        "adapter_dim": adapter_dim,
        "graph_dim": graph_dim,
        "head_hidden": head_hidden,
        "fusion_keys": sorted(str(k) for k in fusion.keys()),
        "head_keys": sorted(str(k) for k in head.keys()),
        "has_dropout_in_head": any("net.1" in str(k) or "net.2" in str(k) for k in head.keys()),
    }


def _approved_reconstruction(settings: Mapping[str, Any]) -> ResolvedE3:
    head = settings.get("response_head", {})
    opt = settings.get("optimizer", {})
    drug = settings.get("drug_reps", {}).get("D0", {})
    screening = settings.get("screening_cv", {})
    payload = {
        "baseline_label": "approved_reconstructed_pooled_mlp",
        "is_original_e3": False,
        "reason": "exact E3 artifact unavailable",
        "user_approval_reference": "Round 20 planning decision",
        "architecture_family": "pooled_mlp",
        "predictor_class": "AdapterMLPFusion+ResponseHead",
        "predictor_id": REQUIRED_PREDICTOR,
        "omics_id": REQUIRED_OMICS,
        "drug_encoder_id": REQUIRED_DRUG,
        "drug_encoder_training_mode": "end_to_end_finetune",
        "pooling": "global_max",
        "context_dim": 16,
        "omics_dim": 80,
        "graph_dim": int(drug.get("graph_output_dim", 32)),
        "node_hidden_dim": int(drug.get("node_hidden_dim", 32)),
        "adapter_dim": 64,
        "hidden_dims": (int(head.get("hidden_dim", 128)),),
        "dropout": float(head.get("dropout", 0.1)),
        "activation": str(head.get("activation", "relu")),
        "normalization": "layernorm_adapters",
        "optimizer": {
            "encoder_lr": float(opt.get("encoder_lr", 1e-4)),
            "fusion_lr": float(opt.get("fusion_lr", 3e-4)),
            "head_lr": float(opt.get("head_lr", 3e-4)),
            "weight_decay": float(opt.get("weight_decay", 1e-4)),
            "grad_clip_max_norm": float(opt.get("grad_clip_max_norm", 1.0)),
        },
        "training": {
            "max_epochs": int(screening.get("max_epochs", 500)),
            "early_stop_patience": int(screening.get("early_stop_patience", 50)),
            "early_stop_start_epoch": int(screening.get("early_stop_start_epoch", 30)),
            "model_seed": int(settings.get("model_seed", 101)),
            "target_effective_batch": int(settings.get("oom", {}).get("target_effective_batch", 1024)),
        },
        "checkpoint_paths": (),
    }
    return ResolvedE3(
        source="approved_reconstruction",
        reconstructed=True,
        architecture_family=payload["architecture_family"],
        predictor_class=payload["predictor_class"],
        predictor_id=payload["predictor_id"],
        omics_id=payload["omics_id"],
        drug_encoder_id=payload["drug_encoder_id"],
        drug_encoder_training_mode=payload["drug_encoder_training_mode"],
        pooling=payload["pooling"],
        context_dim=payload["context_dim"],
        omics_dim=payload["omics_dim"],
        graph_dim=payload["graph_dim"],
        node_hidden_dim=payload["node_hidden_dim"],
        adapter_dim=payload["adapter_dim"],
        hidden_dims=payload["hidden_dims"],
        dropout=payload["dropout"],
        activation=payload["activation"],
        normalization=payload["normalization"],
        optimizer=payload["optimizer"],
        training=payload["training"],
        checkpoint_paths=payload["checkpoint_paths"],
        config_hash=_sha256_json(payload),
        source_candidate_id=SOURCE_CANDIDATE_ID,
        public_alias=PUBLIC_ALIAS,
        baseline_label=payload["baseline_label"],
        is_original_e3=False,
        evidence={
            "baseline_label": payload["baseline_label"],
            "is_original_e3": False,
            "reason": payload["reason"],
            "user_approval_reference": payload["user_approval_reference"],
        },
    )


def resolve_e3(
    repo_root: Path | str,
    *,
    allow_approved_reconstruction: bool = False,
) -> ResolvedE3:
    root = Path(repo_root).resolve()
    lock_path = _resolve_role_lock_path(root)
    lock = _load_json(lock_path)
    policy_path = root / "reports/round19_deployment_policy.json"
    policy = _load_json(policy_path)
    settings_path = root / "config/round19_factorial_settings.json"
    settings = _load_json(settings_path)

    proposal_path = (
        root
        / "result/optimization_runs/round19_factorial/reports/round19_final_role_proposal.json"
    )
    proposal = _load_json(proposal_path) if proposal_path.is_file() else None

    general = _extract_role_candidate(lock, "general_recommended_model")
    chemical = _extract_role_candidate(lock, "chemical_shift_specialist")
    if general != chemical:
        raise E3ResolutionError(
            f"general_recommended_model={general!r} != chemical_shift_specialist={chemical!r}"
        )
    if general != SOURCE_CANDIDATE_ID:
        raise E3ResolutionError(
            f"Locked E3 source candidate is {general!r}, expected {SOURCE_CANDIDATE_ID!r}"
        )

    policy_map = policy.get("locked_role_candidates", {})
    if not isinstance(policy_map, Mapping):
        raise E3ResolutionError("deployment policy locked_role_candidates invalid")
    for role in ("general_recommended_model", "chemical_shift_specialist"):
        if str(policy_map.get(role)) != SOURCE_CANDIDATE_ID:
            raise E3ResolutionError(
                f"deployment policy {role}={policy_map.get(role)!r} != {SOURCE_CANDIDATE_ID!r}"
            )

    predictors = _predictors_from_settings(settings)
    family = predictors.get(REQUIRED_PREDICTOR)
    if family != "pooled_mlp":
        raise E3ResolutionError(
            f"settings predictors.{REQUIRED_PREDICTOR}={family!r}, expected pooled_mlp"
        )

    identity = _proposal_identity(proposal)
    if identity is not None:
        for key, want in (
            ("omics_id", REQUIRED_OMICS),
            ("drug_id", REQUIRED_DRUG),
            ("predictor_id", REQUIRED_PREDICTOR),
            ("source_candidate_id", SOURCE_CANDIDATE_ID),
        ):
            got = identity.get(key)
            if got is not None and str(got) != want:
                raise E3ResolutionError(
                    f"proposal identity {key}={got!r} conflicts with required {want!r}"
                )

    ckpt_rel_paths = _inventory_for_candidate(lock, SOURCE_CANDIDATE_ID)
    ckpt_paths = [root / p for p in ckpt_rel_paths]
    missing = [str(p) for p in ckpt_paths if not p.is_file()]
    if missing:
        if allow_approved_reconstruction:
            return _approved_reconstruction(settings)
        raise E3ResolutionError(
            f"E3 checkpoints missing ({len(missing)}): e.g. {missing[:3]}"
        )

    # Cross-check a representative subset: all three seeds, fold0.
    sample_paths = []
    for seed in (52, 62, 72):
        marker = f"seed{seed}__fold0"
        matched = [p for p in ckpt_paths if marker in str(p)]
        if matched:
            sample_paths.append(matched[0])
    if len(sample_paths) < 3:
        raise E3ResolutionError(
            f"Could not locate seed52/62/72 fold0 checkpoints for cross-check; found {len(sample_paths)}"
        )
    inspections = [_inspect_checkpoint(p) for p in sample_paths]
    ref = inspections[0]
    for other in inspections[1:]:
        for key in (
            "drug_id",
            "predictor_id",
            "omics_id",
            "encoder_type",
            "node_hidden_dim",
            "graph_output_dim",
            "omics_dim",
            "adapter_dim",
            "graph_dim",
            "head_hidden",
        ):
            if other[key] != ref[key]:
                raise E3ResolutionError(
                    f"Checkpoint conflict on {key}: {ref['path']}={ref[key]!r} vs "
                    f"{other['path']}={other[key]!r}"
                )

    if ref["drug_id"] != REQUIRED_DRUG:
        raise E3ResolutionError(f"checkpoint drug_id={ref['drug_id']!r} != {REQUIRED_DRUG}")
    if ref["predictor_id"] != REQUIRED_PREDICTOR:
        raise E3ResolutionError(
            f"checkpoint predictor_id={ref['predictor_id']!r} != {REQUIRED_PREDICTOR}"
        )
    if ref["omics_id"] != REQUIRED_OMICS:
        raise E3ResolutionError(f"checkpoint omics_id={ref['omics_id']!r} != {REQUIRED_OMICS}")
    if ref["encoder_type"] != "gin":
        raise E3ResolutionError(f"checkpoint encoder_type={ref['encoder_type']!r} != gin")
    if int(ref["node_hidden_dim"]) != 32 or int(ref["graph_output_dim"]) != 32:
        raise E3ResolutionError("D0 dims must be node32/graph32")

    # Presence of encoder param group with non-zero lr implies end-to-end finetune.
    opt = settings.get("optimizer", {})
    encoder_lr = float(opt.get("encoder_lr", 0.0))
    if encoder_lr <= 0:
        raise E3ResolutionError("optimizer.encoder_lr must be > 0 for locked E3 D0 mode")
    drug_training_mode = "end_to_end_finetune"

    head_cfg = settings.get("response_head", {})
    screening = settings.get("screening_cv", {})
    confirmation = settings.get("confirmation_cv", screening)

    resolved_core = {
        "architecture_family": "pooled_mlp",
        "predictor_class": "AdapterMLPFusion+ResponseHead",
        "predictor_id": REQUIRED_PREDICTOR,
        "omics_id": REQUIRED_OMICS,
        "drug_encoder_id": REQUIRED_DRUG,
        "drug_encoder_training_mode": drug_training_mode,
        "pooling": "global_max",
        "context_dim": int(ref["context_dim"]),
        "omics_dim": int(ref["omics_dim"]),
        "graph_dim": int(ref["graph_dim"]),
        "node_hidden_dim": int(ref["node_hidden_dim"]),
        "adapter_dim": int(ref["adapter_dim"]),
        "hidden_dims": (int(ref["head_hidden"]),),
        "dropout": float(head_cfg.get("dropout", 0.1)),
        "activation": str(head_cfg.get("activation", "relu")),
        "normalization": "layernorm_adapters",
        "optimizer": {
            "encoder_lr": encoder_lr,
            "fusion_lr": float(opt.get("fusion_lr", 3e-4)),
            "head_lr": float(opt.get("head_lr", 3e-4)),
            "weight_decay": float(opt.get("weight_decay", 1e-4)),
            "grad_clip_max_norm": float(opt.get("grad_clip_max_norm", 1.0)),
        },
        "training": {
            "max_epochs": int(confirmation.get("max_epochs", screening.get("max_epochs", 500))),
            "early_stop_patience": int(
                confirmation.get("early_stop_patience", screening.get("early_stop_patience", 50))
            ),
            "early_stop_start_epoch": int(
                confirmation.get(
                    "early_stop_start_epoch", screening.get("early_stop_start_epoch", 30)
                )
            ),
            "model_seed": int(settings.get("model_seed", 101)),
            "target_effective_batch": int(settings.get("oom", {}).get("target_effective_batch", 1024)),
            "canonical_deployment_split_seed": 52,
        },
        "source_candidate_id": SOURCE_CANDIDATE_ID,
        "public_alias": PUBLIC_ALIAS,
        "is_original_e3": True,
        "baseline_label": "artifact_resolved_pooled_e3",
    }

    def _rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return str(path)

    evidence = {
        "role_lock_path": _rel(lock_path),
        "deployment_policy_path": _rel(policy_path),
        "factorial_settings_path": _rel(settings_path),
        "proposal_path": _rel(proposal_path) if proposal is not None else None,
        "general_recommended_model": general,
        "chemical_shift_specialist": chemical,
        "predictor_family_from_settings": family,
        "proposal_identity": dict(identity) if identity is not None else None,
        "checkpoint_inspections": inspections,
        "n_checkpoints": len(ckpt_paths),
        "cross_source_agreement": True,
    }

    return ResolvedE3(
        source="artifact_resolved",
        reconstructed=False,
        architecture_family=resolved_core["architecture_family"],
        predictor_class=resolved_core["predictor_class"],
        predictor_id=resolved_core["predictor_id"],
        omics_id=resolved_core["omics_id"],
        drug_encoder_id=resolved_core["drug_encoder_id"],
        drug_encoder_training_mode=resolved_core["drug_encoder_training_mode"],
        pooling=resolved_core["pooling"],
        context_dim=resolved_core["context_dim"],
        omics_dim=resolved_core["omics_dim"],
        graph_dim=resolved_core["graph_dim"],
        node_hidden_dim=resolved_core["node_hidden_dim"],
        adapter_dim=resolved_core["adapter_dim"],
        hidden_dims=resolved_core["hidden_dims"],
        dropout=resolved_core["dropout"],
        activation=resolved_core["activation"],
        normalization=resolved_core["normalization"],
        optimizer=resolved_core["optimizer"],
        training=resolved_core["training"],
        checkpoint_paths=tuple(str(p.relative_to(root)) for p in ckpt_paths),
        config_hash=_sha256_json(resolved_core),
        source_candidate_id=SOURCE_CANDIDATE_ID,
        public_alias=PUBLIC_ALIAS,
        baseline_label=resolved_core["baseline_label"],
        is_original_e3=True,
        evidence=evidence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--allow-approved-e3-reconstruction",
        action="store_true",
        help="Only if exact artifacts are unavailable; marks reconstructed baseline",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    try:
        resolved = resolve_e3(
            Path(args.repo_root),
            allow_approved_reconstruction=args.allow_approved_e3_reconstruction,
        )
    except E3ResolutionError as exc:
        payload = {"ok": False, "error": str(exc)}
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        print(text, end="")
        raise SystemExit(2) from exc
    payload = {"ok": True, "resolved_e3": resolved.to_dict()}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
