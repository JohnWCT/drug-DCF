#!/usr/bin/env python3
"""Lock-bound Round 19G inference executor and atomic shard finalizer.

Formal jobs are accepted only from the four manifests pinned by an immutable
Stage 19G experiment lock.  This module never writes a lock, checkpoint, or
role definition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step1_finetune_latent_pipeline_round19 import (  # noqa: E402
    _build_encoder_fusion_head,
    _feature_dir,
    _load_settings,
    preflight_tcga_features,
)
from tools.connected_substructure_masking import (  # noqa: E402
    FRACTIONS,
    connected_mask,
    matched_connected_random,
)
from tools.omics_group_ablation import ablate_omics_blocks  # noqa: E402
from tools.round18_eligible_data import load_omics_latent_dict  # noqa: E402
from tools.round19_atom_occlusion import (  # noqa: E402
    MAX_PERTURBATION_BATCH,
    batched,
    feature_zero_graph,
    matched_random_controls,
    rank_atom_sets,
)
from tools.round19_attention_consistency import pairwise_member_consistency  # noqa: E402
from tools.round19_attention_ensemble import ensemble_atom_attention  # noqa: E402
from tools.round19_attention_export import (  # noqa: E402
    export_attention_batches,
    strict_load_locked_models,
)
from tools.round19_dataset import Round19ResponseDataset, round19_collate_fn  # noqa: E402
from tools.round19_stage19f_ensemble import REQUIRED_MEMBER_IDS  # noqa: E402
from tools.round19_stage19g_experiment_lock import canonical_sha256, sha256_file  # noqa: E402
from tools.round19_stage19g_lock_adapter import load_verified_lock, route_locked  # noqa: E402
from tools.round19_train_loop import forward_round19_batch  # noqa: E402
from tools.routing_counterfactual import routing_regret  # noqa: E402
from tools.scaffold_sidechain_ablation import ablation_rows  # noqa: E402
from tools.stage19g_routing_audit import novelty_class  # noqa: E402

METHODS = ("attention", "occlusion", "omics", "routing")
OUTPUT_CSVS = (
    "round19g_atom_occlusion.csv",
    "round19g_connected_substructure_masking.csv",
    "round19g_scaffold_sidechain_ablation.csv",
    "round19g_bond_occlusion.csv",
    "round19g_pooled_drug_occlusion.csv",
    "round19g_maccs_ablation.csv",
    "round19g_omics_group_ablation.csv",
    "round19g_context_sensitivity.csv",
    "round19g_routing_audit.csv",
    "round19g_routing_counterfactual.csv",
    "round19g_case_summary.csv",
)
ATTENTION_CSVS = (
    "round19g_attention_long.csv",
    "round19g_attention_ensemble.csv",
    "round19g_attention_consistency.csv",
    "round19g_attention_context.csv",
    "round19g_attention_variance.csv",
)
TASK_REQUIRED = {
    "task_id", "method", "source_candidate_id", "role_aliases", "member_id",
    "checkpoint_path", "checkpoint_sha256", "case_shard_id", "case_start",
    "case_stop_exclusive", "case_count", "case_ids_sha256",
    "case_manifest_sha256", "config_sha256", "final_lock_file_sha256",
    "final_lock_payload_sha256",
}


def _rooted(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _case_column(frame: pd.DataFrame) -> str:
    for name in ("case_id", "eval_row_id"):
        if name in frame:
            return name
    raise KeyError("case manifest requires case_id or eval_row_id")


def _canonical_cases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    case_column = _case_column(out)
    if "case_id" not in out:
        out["case_id"] = out[case_column].astype(str)
    if "eval_row_id" not in out:
        out["eval_row_id"] = out[case_column].astype(str)
    aliases = {
        "DRUG_NAME": ("drug_name",),
        "ModelID": ("model_id", "Patient_id"),
        "canonical_smiles": ("graph_smiles", "smiles"),
        "graph_smiles": ("canonical_smiles", "smiles"),
    }
    for target, alternatives in aliases.items():
        if target not in out:
            source = next((name for name in alternatives if name in out), None)
            if source:
                out[target] = out[source]
    required = {"case_id", "eval_row_id", "ModelID", "DRUG_NAME", "Label"}
    missing = required - set(out)
    if missing:
        raise KeyError(f"case manifest missing runtime fields: {sorted(missing)}")
    if out["case_id"].astype(str).duplicated().any():
        raise AssertionError("case_id must be unique")
    out["case_id"] = out["case_id"].astype(str)
    out["eval_row_id"] = out["eval_row_id"].astype(str)
    out["ModelID"] = out["ModelID"].astype(str)
    out["DRUG_NAME"] = out["DRUG_NAME"].astype(str)
    out["Label"] = pd.to_numeric(out["Label"], errors="raise").astype(int)
    out["_row_id"] = np.arange(len(out), dtype=int)
    if "canonical_smiles" in out:
        out["smiles"] = out["canonical_smiles"].fillna("").astype(str)
    elif "graph_smiles" in out:
        out["smiles"] = out["graph_smiles"].fillna("").astype(str)
    target = (
        out["source_target"] if "source_target" in out
        else out["target_key"] if "target_key" in out
        else pd.Series("internal", index=out.index)
    )
    out["target_key"] = target.astype(str)
    return out


def verify_experiment_lock(
    experiment_lock_path: Path, *, project_root: Path
) -> dict[str, Any]:
    """Verify the experiment lock, final lock, all manifests, and checkpoints."""
    path = Path(experiment_lock_path)
    lock = _read_json(path)
    if (
        lock.get("lock_type") != "round19_stage19g_experiment_lock"
        or lock.get("schema_version") != 1
        or lock.get("immutable") is not True
    ):
        raise AssertionError("Formal inference requires immutable schema-v1 19G experiment lock")
    payload = json.loads(json.dumps(lock, allow_nan=False))
    expected_payload = payload.pop("lock_payload_sha256", None)
    if expected_payload != canonical_sha256(payload):
        raise AssertionError("19G experiment lock payload hash mismatch")
    final_record = lock.get("final_role_lock", {})
    final_path = _rooted(project_root, final_record.get("path", ""))
    if sha256_file(final_path) != final_record.get("file_sha256"):
        raise AssertionError("Final lock file hash differs from experiment lock")
    final_lock = load_verified_lock(final_path, project_root)
    if final_lock["hashes"]["lock_payload_sha256"] != final_record.get("payload_sha256"):
        raise AssertionError("Final lock payload differs from experiment lock")
    if lock.get("protocol", {}).get("role_changes_forbidden") is not True:
        raise AssertionError("Experiment lock does not forbid role changes")
    manifests = lock.get("task_manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != set(METHODS):
        raise AssertionError("Experiment lock must pin exactly four task manifests")
    loaded: dict[str, pd.DataFrame] = {}
    for method in METHODS:
        record = manifests[method]
        manifest_path = _rooted(project_root, record["path"])
        if sha256_file(manifest_path) != record["sha256"]:
            raise AssertionError(f"Pinned {method} task manifest hash mismatch")
        frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        missing = TASK_REQUIRED - set(frame)
        if missing or frame.empty:
            raise AssertionError(f"{method} manifest invalid; missing={sorted(missing)}")
        if set(frame["method"]) != {method}:
            raise AssertionError(f"{method} manifest contains another method")
        if frame["task_id"].duplicated().any() or len(frame) != int(record["tasks"]):
            raise AssertionError(f"{method} task coverage/count mismatch")
        loaded[method] = frame
    case_record = lock["case_manifest"]
    case_path = _rooted(project_root, case_record["path"])
    if sha256_file(case_path) != case_record["sha256"]:
        raise AssertionError("Case manifest hash mismatch")
    cases = _canonical_cases(pd.read_csv(case_path))
    if len(cases) != int(case_record["rows"]):
        raise AssertionError("Case manifest row count mismatch")
    inventory = {
        (str(row["source_candidate_id"]), str(row["member_id"])): row
        for row in final_lock["hashes"]["checkpoint_inventory"]
    }
    if len(inventory) != 90:
        raise AssertionError("Final checkpoint inventory is incomplete")
    for method, frame in loaded.items():
        for row in frame.to_dict("records"):
            key = (row["source_candidate_id"], row["member_id"])
            pinned = inventory.get(key)
            if pinned is None:
                raise AssertionError(f"{method} task references unlocked member {key}")
            if any(str(row[name]) != str(pinned[name]) for name in ("checkpoint_path", "checkpoint_sha256")):
                raise AssertionError(f"{method} task checkpoint differs from final lock: {row['task_id']}")
            if row["case_manifest_sha256"] != case_record["sha256"]:
                raise AssertionError(f"{method} task is not pinned to case manifest")
            if row["final_lock_file_sha256"] != final_record["file_sha256"]:
                raise AssertionError(f"{method} task is not pinned to final lock")
    lock["_path"] = str(path)
    lock["_file_sha256"] = sha256_file(path)
    lock["_final_path"] = str(final_path)
    lock["_final"] = final_lock
    lock["_manifests"] = loaded
    lock["_cases"] = cases
    return lock


def _task_and_cases(
    verified: Mapping[str, Any], method: str, task_id: str, case_limit: int | None
) -> tuple[dict[str, str], pd.DataFrame]:
    frame = verified["_manifests"][method]
    rows = frame[frame["task_id"] == task_id]
    if len(rows) != 1:
        raise AssertionError(f"task_id must occur exactly once in pinned manifest: {task_id}")
    task = rows.iloc[0].to_dict()
    start, stop = int(task["case_start"]), int(task["case_stop_exclusive"])
    cases = verified["_cases"].iloc[start:stop].copy()
    if len(cases) != int(task["case_count"]):
        raise AssertionError("task case range/count mismatch")
    digest = hashlib.sha256("\n".join(cases["eval_row_id"]).encode()).hexdigest()
    if digest != task["case_ids_sha256"]:
        raise AssertionError("task case row-range identity mismatch")
    if case_limit is not None:
        cases = cases.head(int(case_limit)).copy()
    return task, cases


def _components(verified: Mapping[str, Any], source: str) -> dict[str, str]:
    config_path = next(
        _rooted(PROJECT_ROOT, path)
        for path, record in verified["configs"].items()
        if record.get("artifact_type") == "round19_stage19g_interpretability_config"
    )
    config = _read_json(config_path)
    value = config["candidate_components"].get(source)
    if not isinstance(value, dict):
        raise KeyError(f"No locked component identity for {source}")
    return {key: str(value[key]) for key in ("drug_id", "predictor_id", "omics_id")}


def _latent_map(cases: pd.DataFrame, settings: dict, omics_id: str) -> dict[str, np.ndarray]:
    latent = load_omics_latent_dict(_feature_dir(settings, omics_id))
    tcga = cases.get("is_tcga_exploratory", pd.Series(False, index=cases.index))
    tcga = tcga.astype(str).str.lower().isin({"true", "1"})
    if tcga.any():
        external, _ = preflight_tcga_features(settings, omics_id)
        by_patient: dict[str, np.ndarray] = {}
        for key, vector in external.items():
            parts = str(key).split("-")
            by_patient.setdefault("-".join(parts[:3]) if len(parts) >= 3 else str(key), vector)
        for model_id in cases.loc[tcga, "ModelID"].astype(str):
            if model_id not in by_patient:
                raise KeyError(f"Missing TCGA latent vector for {model_id}")
            latent[model_id] = np.asarray(by_patient[model_id], dtype=np.float32)
    return latent


def _build_runtime(
    verified: Mapping[str, Any],
    task: Mapping[str, str],
    cases: pd.DataFrame,
    *,
    settings_path: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[Any, Any, Any, str, dict, Round19ResponseDataset, DataLoader]:
    components = _components(verified, task["source_candidate_id"])
    settings = _load_settings(str(settings_path))
    latent = _latent_map(cases, settings, components["omics_id"])
    drug_cfg = settings["drug_reps"][components["drug_id"]]
    dataset = Round19ResponseDataset(
        cases,
        feature_dir=_feature_dir(settings, components["omics_id"]),
        drug_smiles_path=settings["drug_smiles_path"],
        encoder_type=drug_cfg["type"],
        with_bonds=bool(drug_cfg.get("edge_features")),
        omics_id=components["omics_id"],
        latent_by_id=latent,
    )
    loader = DataLoader(
        dataset, batch_size=max(1, int(batch_size)), shuffle=False,
        num_workers=0, collate_fn=round19_collate_fn,
    )

    def factory(checkpoint: Mapping[str, Any]):
        encoder, fusion, head, _, _ = _build_encoder_fusion_head(
            settings,
            drug_id=components["drug_id"],
            predictor_id=components["predictor_id"],
            omics_dim=dataset.omics_dim,
            device=device,
        )
        for key in ("drug_id", "predictor_id", "omics_id"):
            if key in checkpoint and str(checkpoint[key]) != components[key]:
                raise AssertionError(f"checkpoint {key} identity mismatch")
        return encoder, fusion, head

    encoder, fusion, head, _ = strict_load_locked_models(
        task, project_root=PROJECT_ROOT, model_factory=factory, map_location=device
    )
    for module in (encoder, fusion, head):
        module.to(device).eval()
    return encoder, fusion, head, str(drug_cfg["type"]), components, dataset, loader


def _local_batch(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch = round19_collate_fn([item])
    batch["omics"] = batch["omics"].to(device)
    if batch.get("maccs") is not None:
        batch["maccs"] = batch["maccs"].to(device)
    if batch.get("drug_batch") is not None:
        batch["drug_batch"] = batch["drug_batch"].to(device)
    return batch


@torch.no_grad()
def _predict_batch(
    encoder, fusion, head, batch: dict[str, Any], *, device, encoder_type, predictor_id,
    interpretability: bool = False,
) -> tuple[np.ndarray, dict[str, torch.Tensor] | None]:
    local = dict(batch)
    local["omics"] = local["omics"].to(device)
    if local.get("maccs") is not None:
        local["maccs"] = local["maccs"].to(device)
    if local.get("drug_batch") is not None:
        local["drug_batch"] = local["drug_batch"].to(device)
    result = forward_round19_batch(
        encoder=encoder, fusion=fusion, encoder_type=encoder_type,
        predictor_id=predictor_id, omics=local["omics"], batch=local,
        return_interpretability=interpretability,
    )
    representation = result["representation"] if isinstance(result, dict) else result
    logits = head(representation).view(-1)
    return torch.sigmoid(logits).detach().cpu().numpy(), result if isinstance(result, dict) else None


def _predict_graphs(
    encoder, fusion, head, item: dict[str, Any], graphs: Sequence[Any], *,
    device, encoder_type, predictor_id, batch_size: int,
) -> list[float]:
    values: list[float] = []
    for group in batched(list(graphs), min(int(batch_size), MAX_PERTURBATION_BATCH)):
        n = len(group)
        batch = {
            "omics": torch.stack([item["omics"]] * n),
            "drug_batch": Batch.from_data_list(list(group)),
            "maccs": None,
        }
        probabilities, _ = _predict_batch(
            encoder, fusion, head, batch, device=device,
            encoder_type=encoder_type, predictor_id=predictor_id,
        )
        values.extend(map(float, probabilities))
    return values


def _common(task: Mapping[str, str], item: Mapping[str, Any], baseline: float) -> dict[str, Any]:
    return {
        "case_id": str(item.get("case_id", item["eval_row_id"])),
        "eval_row_id": str(item["eval_row_id"]),
        "candidate_id": task["source_candidate_id"],
        "source_candidate_id": task["source_candidate_id"],
        "role_aliases": task["role_aliases"],
        "member_id": task["member_id"],
        "original_probability": float(baseline),
        "checkpoint_sha256": task["checkpoint_sha256"],
        "method_classification": "post_lock_descriptive",
    }


def _na_row(common: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        **common, "applicable": False, "status": "not_applicable",
        "not_applicable_reason": reason, "attribution_available": False,
    }


def run_attention(
    task: Mapping[str, str], cases: pd.DataFrame, runtime: tuple, output_dir: Path,
    *, device: torch.device,
) -> dict[str, Any]:
    encoder, fusion, head, encoder_type, components, _, loader = runtime
    if components["predictor_id"] != "P2":
        raise AssertionError("Pinned attention task is not P2")
    destination = output_dir / ATTENTION_CSVS[0]
    temp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    count = export_attention_batches(
        encoder=encoder, fusion=fusion, head=head, dataloader=loader,
        output_path=temp, device=device, encoder_type=encoder_type,
        predictor_id="P2",
        provenance={
            "candidate_id": task["source_candidate_id"],
            "member_id": task["member_id"],
            "checkpoint_path": task["checkpoint_path"],
            "checkpoint_sha256": task["checkpoint_sha256"],
            "lock_payload_sha256": task["final_lock_payload_sha256"],
        },
        case_shard=cases,
    )
    os.replace(temp, destination)
    return {"rows": count, "outputs": [str(destination)]}


def run_occlusion(
    task: Mapping[str, str], cases: pd.DataFrame, runtime: tuple, output_dir: Path,
    *, device: torch.device, perturbation_batch: int,
) -> dict[str, Any]:
    encoder, fusion, head, encoder_type, components, dataset, _ = runtime
    predictor = components["predictor_id"]
    rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in OUTPUT_CSVS[:6]
    }
    for index in range(len(dataset)):
        item = dataset[index]
        baseline, interpreted = _predict_batch(
            encoder, fusion, head, _local_batch(item, device), device=device,
            encoder_type=encoder_type, predictor_id=predictor,
            interpretability=(predictor == "P2"),
        )
        original = float(baseline[0])
        common = _common(task, {**item, "case_id": cases.iloc[index]["case_id"]}, original)
        if item.get("drug_graph") is None:
            fingerprint = item["maccs"]
            active = torch.flatnonzero(fingerprint > 0).tolist()
            if not active:
                active = list(range(len(fingerprint)))
            altered = []
            for bit in active:
                value = fingerprint.clone()
                value[bit] = 0
                altered.append(value)
            values = []
            for group in batched(altered, perturbation_batch):
                batch = {
                    "omics": torch.stack([item["omics"]] * len(group)),
                    "maccs": torch.stack(group), "drug_batch": None,
                }
                probs, _ = _predict_batch(
                    encoder, fusion, head, batch, device=device,
                    encoder_type=encoder_type, predictor_id=predictor,
                )
                values.extend(map(float, probs))
            ranked = sorted(zip(active, values), key=lambda pair: (-(original-pair[1]), pair[0]))
            for rank, (bit, probability) in enumerate(ranked, 1):
                rows["round19g_maccs_ablation.csv"].append({
                    **common, "applicable": True, "status": "ok",
                    "method": "maccs_input_perturbation", "attribution_level": "fingerprint_bit",
                    "atom_heatmap_available": False, "bit_index": bit, "rank": rank,
                    "perturbed_probability": probability,
                    "prediction_delta": original - probability,
                })
            reason = "MACCS has fingerprint-bit inputs and no atom/bond topology"
            for filename in (
                "round19g_atom_occlusion.csv", "round19g_connected_substructure_masking.csv",
                "round19g_scaffold_sidechain_ablation.csv", "round19g_bond_occlusion.csv",
                "round19g_pooled_drug_occlusion.csv",
            ):
                rows[filename].append(_na_row(common, reason))
            continue

        graph = item["drug_graph"]
        n_atoms = int(graph.x.shape[0])
        if predictor == "P2":
            if interpreted is None:
                raise RuntimeError("P2 occlusion did not return primary attention")
            scores = interpreted["attention_primary"][0, :n_atoms].detach().cpu().numpy()
            ranking_strategy = "primary_attention"
        else:
            single_graphs = [feature_zero_graph(graph, [atom]) for atom in range(n_atoms)]
            single_probs = _predict_graphs(
                encoder, fusion, head, item, single_graphs, device=device,
                encoder_type=encoder_type, predictor_id=predictor,
                batch_size=perturbation_batch,
            )
            scores = np.asarray([original - value for value in single_probs])
            ranking_strategy = "single_atom_input_perturbation"
        ranked_sets = rank_atom_sets(scores)
        metadata = graph.graph_metadata["atom_metadata"]
        interventions: list[tuple[str, list[int], str, int | None]] = []
        for group_name, target in ranked_sets.items():
            interventions.append((group_name, target, "ranked", None))
            try:
                controls = matched_random_controls(
                    target, metadata, repeats=20,
                    seed=19091 + index + int(hashlib.sha256(task["member_id"].encode()).hexdigest()[:6], 16),
                )
            except ValueError:
                controls = [
                    {"repeat": repeat, "atom_indices": target, "target_excluded": False,
                     "fallback_levels": ["unavailable"]}
                    for repeat in range(20)
                ]
            for control in controls:
                interventions.append(
                    (group_name, control["atom_indices"], "matched_random", int(control["repeat"]))
                )
        probabilities = _predict_graphs(
            encoder, fusion, head, item,
            [feature_zero_graph(graph, indices) for _, indices, _, _ in interventions],
            device=device, encoder_type=encoder_type, predictor_id=predictor,
            batch_size=perturbation_batch,
        )
        for (group_name, indices, control_type, repeat), probability in zip(interventions, probabilities):
            rows["round19g_atom_occlusion.csv"].append({
                **common, "applicable": True, "status": "ok",
                "ranking_strategy": ranking_strategy, "mask_group": group_name,
                "atom_indices": json.dumps(indices), "control_type": control_type,
                "repeat": repeat, "perturbed_probability": probability,
                "prediction_delta": original - probability,
            })
        connected: list[tuple[float, list[int], str, int | None]] = []
        for fraction in FRACTIONS:
            target = connected_mask(scores, graph.edge_index, fraction)
            connected.append((fraction, target, "ranked", None))
            try:
                controls = matched_connected_random(
                    target, graph.edge_index, n_atoms, repeats=20,
                    seed=19091 + index,
                )
            except ValueError:
                controls = [{"repeat": repeat, "atom_indices": target} for repeat in range(20)]
            connected.extend(
                (fraction, control["atom_indices"], "matched_random", int(control["repeat"]))
                for control in controls
            )
        connected_probs = _predict_graphs(
            encoder, fusion, head, item,
            [feature_zero_graph(graph, indices) for _, indices, _, _ in connected],
            device=device, encoder_type=encoder_type, predictor_id=predictor,
            batch_size=perturbation_batch,
        )
        for (fraction, indices, control_type, repeat), probability in zip(connected, connected_probs):
            rows["round19g_connected_substructure_masking.csv"].append({
                **common, "applicable": True, "status": "ok",
                "ranking_strategy": ranking_strategy, "fraction": fraction,
                "atom_indices": json.dumps(indices), "control_type": control_type,
                "repeat": repeat, "perturbed_probability": probability,
                "prediction_delta": original - probability,
            })
        scaffold_specs = ablation_rows(item["graph_smiles"])
        applicable = [spec for spec in scaffold_specs if spec["applicable"]]
        scaffold_probs = _predict_graphs(
            encoder, fusion, head, item,
            [feature_zero_graph(graph, spec["atom_indices"]) for spec in applicable],
            device=device, encoder_type=encoder_type, predictor_id=predictor,
            batch_size=perturbation_batch,
        )
        probability_iter = iter(scaffold_probs)
        for spec in scaffold_specs:
            probability = next(probability_iter) if spec["applicable"] else math.nan
            rows["round19g_scaffold_sidechain_ablation.csv"].append({
                **common, "applicable": bool(spec["applicable"]),
                "status": "ok" if spec["applicable"] else "not_applicable",
                "ablation": spec["ablation"], "atom_indices": json.dumps(spec["atom_indices"]),
                "perturbed_probability": probability,
                "prediction_delta": original - probability if spec["applicable"] else math.nan,
            })
        if encoder_type.lower() == "gine" and getattr(graph, "edge_attr", None) is not None:
            bond_ids = sorted({
                min(int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge])) * n_atoms
                + max(int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge]))
                for edge in range(graph.edge_index.shape[1])
            })
            bond_graphs = []
            for bond_id in bond_ids:
                altered = graph.clone()
                altered.edge_attr = graph.edge_attr.clone()
                for edge in range(graph.edge_index.shape[1]):
                    current = min(int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge])) * n_atoms + max(
                        int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge])
                    )
                    if current == bond_id:
                        altered.edge_attr[edge] = 0
                bond_graphs.append(altered)
            bond_probs = _predict_graphs(
                encoder, fusion, head, item, bond_graphs, device=device,
                encoder_type=encoder_type, predictor_id=predictor,
                batch_size=perturbation_batch,
            )
            for bond_id, probability in zip(bond_ids, bond_probs):
                rows["round19g_bond_occlusion.csv"].append({
                    **common, "applicable": True, "status": "ok",
                    "method": "bond_feature_input_perturbation", "bond_id": bond_id,
                    "perturbed_probability": probability,
                    "prediction_delta": original - probability,
                })
        else:
            rows["round19g_bond_occlusion.csv"].append(
                _na_row(common, "bond feature perturbation is applicable only to GINE")
            )
        pooled_groups = list(ranked_sets.items()) if predictor in {"P0", "P1"} else []
        if pooled_groups:
            pooled_probs = _predict_graphs(
                encoder, fusion, head, item,
                [feature_zero_graph(graph, indices) for _, indices in pooled_groups],
                device=device, encoder_type=encoder_type, predictor_id=predictor,
                batch_size=perturbation_batch,
            )
            for (name, indices), probability in zip(pooled_groups, pooled_probs):
                rows["round19g_pooled_drug_occlusion.csv"].append({
                    **common, "applicable": True, "status": "ok",
                    "method": "input_perturbation", "has_attention": False,
                    "ranking_strategy": ranking_strategy, "group_id": name,
                    "feature_indices": json.dumps(indices),
                    "perturbed_probability": probability,
                    "prediction_delta": original - probability,
                })
        else:
            rows["round19g_pooled_drug_occlusion.csv"].append(
                _na_row(common, "pooled-drug occlusion is defined only for P0/P1")
            )
        rows["round19g_maccs_ablation.csv"].append(
            _na_row(common, "MACCS bit ablation is applicable only to D4")
        )
    outputs = []
    for filename, records in rows.items():
        path = output_dir / filename
        _atomic_csv(path, pd.DataFrame(records))
        outputs.append(str(path))
    return {"rows": {name: len(value) for name, value in rows.items()}, "outputs": outputs}


def _omics_blocks(omics_id: str, dimension: int) -> dict[str, list[int]]:
    blocks = {"latent_core": list(range(min(64, dimension)))}
    oid = str(omics_id).upper()
    if oid == "O2":
        blocks["context16"] = list(range(64, min(80, dimension)))
    elif oid == "O3":
        blocks["summary11"] = list(range(64, min(75, dimension)))
        blocks["context16"] = list(range(75, min(91, dimension)))
    elif oid == "O4":
        blocks["source_prototype7"] = list(range(64, min(71, dimension)))
    elif dimension > 64:
        blocks["summary_features"] = list(range(64, dimension))
    return {name: indices for name, indices in blocks.items() if indices}


def run_omics(
    task: Mapping[str, str], cases: pd.DataFrame, runtime: tuple, output_dir: Path,
    *, device: torch.device, batch_size: int,
) -> dict[str, Any]:
    encoder, fusion, head, encoder_type, components, dataset, _ = runtime
    values = np.stack([dataset[index]["omics"].numpy() for index in range(len(dataset))])
    partitions = (
        cases.get("source_stage", "unknown").astype(str) + ":"
        + cases.get("source_target", "unknown").astype(str)
    ).tolist()
    blocks = _omics_blocks(components["omics_id"], values.shape[1])
    interventions = ablate_omics_blocks(
        values, blocks, omics_role=components["omics_id"],
        partition_ids=partitions, seed=19091,
    )
    rows = []
    for block_name, conditions in interventions.items():
        for condition, matrix in conditions.items():
            for start in range(0, len(dataset), max(1, int(batch_size))):
                stop = min(start + int(batch_size), len(dataset))
                items = [dataset[index] for index in range(start, stop)]
                batch = round19_collate_fn(items)
                batch["omics"] = torch.as_tensor(matrix[start:stop], dtype=torch.float32)
                probabilities, _ = _predict_batch(
                    encoder, fusion, head, batch, device=device,
                    encoder_type=encoder_type, predictor_id=components["predictor_id"],
                )
                for offset, probability in enumerate(probabilities):
                    index = start + offset
                    common = _common(
                        task, {**items[offset], "case_id": cases.iloc[index]["case_id"]},
                        float(probability) if condition == "true" else math.nan,
                    )
                    rows.append({
                        **common, "omics_id": components["omics_id"],
                        "omics_feature_block": block_name, "condition": condition,
                        "partition_id": partitions[index], "shuffle_seed": 19091,
                        "probability": float(probability),
                    })
    frame = pd.DataFrame(rows)
    baseline = frame[frame["condition"] == "true"][
        ["case_id", "omics_feature_block", "probability"]
    ].rename(columns={"probability": "true_probability"})
    frame = frame.merge(
        baseline, on=["case_id", "omics_feature_block"], how="left", validate="many_to_one"
    )
    frame["probability_delta"] = frame["true_probability"] - frame["probability"]
    omics_path = output_dir / "round19g_omics_group_ablation.csv"
    _atomic_csv(omics_path, frame)
    context = frame[frame["omics_feature_block"].str.contains("context")].copy()
    if context.empty:
        context = frame.copy()
        context["status"] = "not_applicable"
        context["not_applicable_reason"] = f"{components['omics_id']} has no context block"
    context_path = output_dir / "round19g_context_sensitivity.csv"
    _atomic_csv(context_path, context)
    return {"rows": len(frame), "outputs": [str(omics_path), str(context_path)]}


def _routing_audit(verified: Mapping[str, Any]) -> pd.DataFrame:
    cases = verified["_cases"].copy()
    required = {"normalized_drug_id", "scaffold_id", "cancer_type"}
    missing = required - set(cases)
    if missing:
        raise KeyError(f"routing cases missing fields: {sorted(missing)}")
    development = pd.read_csv(
        PROJECT_ROOT / "result/optimization_runs/round19_factorial/data/round19_eligible_response.csv"
    )
    maps = PROJECT_ROOT / "result/optimization_runs/round19_factorial/splits"
    drug_map = pd.read_csv(maps / "round19e_drug_group_table.csv")
    scaffold_map = pd.read_csv(maps / "round19e_scaffold_group_table.csv")
    cancer_map = pd.read_csv(maps / "round19e_modelid_cancer_type_map.csv")
    support = development.merge(
        drug_map[["DRUG_NAME", "normalized_drug_id"]],
        on="DRUG_NAME", how="left", validate="many_to_one",
    ).merge(
        scaffold_map[["DRUG_NAME", "scaffold_id"]],
        on="DRUG_NAME", how="left", validate="many_to_one",
    ).merge(
        cancer_map[["ModelID", "cancer_type"]], on="ModelID",
        how="left", validate="many_to_one",
    )
    assignments = {
        shift: pd.read_csv(maps / f"round19e_{shift}_5cv.csv")
        for shift in ("drug_heldout", "scaffold_heldout", "cancer_type_heldout")
    }
    rows = []
    final_path = Path(verified["_final_path"])
    for case in cases.to_dict("records"):
        is_19e = str(case.get("source_stage", "")).upper() == "19E"
        if is_19e:
            shift = str(case.get("shift_strategy") or case.get("source_target"))
            if shift not in assignments:
                raise ValueError(f"19E routing case has unknown shift strategy: {shift}")
            assignment = assignments[shift]
            source_row = int(float(case["source_row_id"]))
            validation = assignment[
                (assignment["_row_id"] == source_row)
                & (assignment["split_role"].astype(str) == "val")
            ]
            if len(validation) != 1:
                raise AssertionError(
                    f"19E case does not map to one held-out fold: {case['case_id']}"
                )
            fold_id = int(validation.iloc[0]["fold_id"])
            train_ids = set(
                assignment[
                    (assignment["fold_id"] == fold_id)
                    & (assignment["split_role"].astype(str) == "train")
                ]["_row_id"].astype(int)
            )
            eligible = support[support["_row_id"].astype(int).isin(train_ids)]
            support_basis = "19E_fold_relative"
        else:
            eligible = support
            fold_id = math.nan
            support_basis = "TCGA_full_development"
        all_drugs = set(eligible["normalized_drug_id"].astype(str))
        all_scaffolds = set(eligible["scaffold_id"].astype(str))
        all_cancers = set(eligible["cancer_type"].astype(str))
        seen = {
            "seen_drug": str(case["normalized_drug_id"]) in all_drugs,
            "seen_scaffold": str(case["scaffold_id"]) in all_scaffolds,
            "seen_cancer_type": str(case["cancer_type"]) in all_cancers,
        }
        novelty = novelty_class(seen)
        routed = route_locked(final_path, novelty, project_root=PROJECT_ROOT)
        rows.append({
            **case, **seen, "fold_id": fold_id, "support_basis": support_basis,
            "novelty_class": novelty,
            "selected_role": routed["selected_role"],
            "selected_source_candidate_id": routed["source_candidate_id"],
            "routing_match": True,
            "lock_file_sha256": routed["lock_file_sha256"],
        })
    return pd.DataFrame(rows)


def run_routing(verified: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    # Completeness of all nominal candidate×member×shard rows was checked before
    # this single deterministic all-case job was admitted.
    audit = _routing_audit(verified)
    path = output_dir / "round19g_routing_audit.csv"
    _atomic_csv(path, audit)
    return {"rows": len(audit), "outputs": [str(path)], "deduplicated": True}


def execute_job(
    *, experiment_lock: Path, method: str, task_id: str, output_root: Path,
    settings: Path, device_name: str, batch_size: int, perturbation_batch: int,
    case_limit: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    verified = verify_experiment_lock(experiment_lock, project_root=PROJECT_ROOT)
    task, cases = _task_and_cases(verified, method, task_id, case_limit)
    shard_dir = output_root / "shards" / method / task_id
    status_path = shard_dir / "status.json"
    if status_path.is_file():
        previous = _read_json(status_path)
        if (
            previous.get("status") == "done"
            and previous.get("experiment_lock_sha256") == verified["_file_sha256"]
            and previous.get("checkpoint_sha256") == task["checkpoint_sha256"]
            and int(previous.get("case_count", -1)) == len(cases)
            and all(Path(path).is_file() for path in previous.get("outputs", []))
        ):
            return {**previous, "resume_skipped": True}
    checkpoint = _rooted(PROJECT_ROOT, task["checkpoint_path"])
    before = {
        "experiment": sha256_file(Path(experiment_lock)),
        "final": sha256_file(Path(verified["_final_path"])),
        "checkpoint": sha256_file(checkpoint),
    }
    if before["checkpoint"] != task["checkpoint_sha256"]:
        raise AssertionError("Checkpoint hash changed immediately before job")
    shard_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(status_path, {
        "status": "running", "task_id": task_id, "method": method,
        "experiment_lock_sha256": before["experiment"],
        "checkpoint_sha256": before["checkpoint"],
    })
    try:
        if method == "routing":
            result = run_routing(verified, shard_dir)
        else:
            device = torch.device(
                "cuda" if device_name == "auto" and torch.cuda.is_available()
                else "cpu" if device_name == "auto" else device_name
            )
            runtime = _build_runtime(
                verified, task, cases, settings_path=settings,
                device=device, batch_size=batch_size,
            )
            if method == "attention":
                result = run_attention(task, cases, runtime, shard_dir, device=device)
            elif method == "occlusion":
                result = run_occlusion(
                    task, cases, runtime, shard_dir, device=device,
                    perturbation_batch=perturbation_batch,
                )
            elif method == "omics":
                result = run_omics(
                    task, cases, runtime, shard_dir, device=device, batch_size=batch_size
                )
            else:
                raise ValueError(f"Unknown method={method}")
        after = {
            "experiment": sha256_file(Path(experiment_lock)),
            "final": sha256_file(Path(verified["_final_path"])),
            "checkpoint": sha256_file(checkpoint),
        }
        if after != before:
            raise AssertionError(f"Immutable input hash changed during job: {before} -> {after}")
        status = {
            "status": "done", "task_id": task_id, "method": method,
            "source_candidate_id": task["source_candidate_id"],
            "member_id": task["member_id"], "case_count": len(cases),
            "experiment_lock_sha256": before["experiment"],
            "final_lock_sha256": before["final"],
            "checkpoint_sha256": before["checkpoint"],
            "elapsed_sec": round(time.time() - started, 3), **result,
        }
        _atomic_json(status_path, status)
        return status
    except Exception as exc:
        _atomic_json(status_path, {
            "status": "failed", "task_id": task_id, "method": method,
            "error_type": type(exc).__name__, "error": str(exc),
            "experiment_lock_sha256": before["experiment"],
            "checkpoint_sha256": before["checkpoint"],
        })
        raise


def _merge_shards(output_root: Path, method: str, filename: str) -> pd.DataFrame:
    paths = sorted((output_root / "shards" / method).glob(f"*/{filename}"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _assert_no_duplicates(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    selected = [key for key in keys if key in frame]
    if selected and frame.duplicated(selected).any():
        raise AssertionError(f"{name} contains duplicate rows on {selected}")


def finalize_outputs(
    *, experiment_lock: Path, output_root: Path, allow_partial: bool = False
) -> dict[str, Any]:
    verified = verify_experiment_lock(experiment_lock, project_root=PROJECT_ROOT)
    expected_cases = set(verified["_cases"]["case_id"])
    statuses = {}
    for method in METHODS:
        expected_tasks = set(verified["_manifests"][method]["task_id"])
        if method == "routing":
            expected_tasks = {sorted(expected_tasks)[0]}
        done = set()
        for path in (output_root / "shards" / method).glob("*/status.json"):
            status = _read_json(path)
            if status.get("status") == "done":
                done.add(str(status["task_id"]))
        if not allow_partial and done != expected_tasks:
            raise AssertionError(
                f"{method} job coverage mismatch: missing={len(expected_tasks-done)} "
                f"extra={len(done-expected_tasks)}"
            )
        statuses[method] = {"expected": len(expected_tasks), "done": len(done)}
    for filename in OUTPUT_CSVS[:8]:
        method = "omics" if filename in OUTPUT_CSVS[6:8] else "occlusion"
        frame = _merge_shards(output_root, method, filename)
        if frame.empty:
            if allow_partial:
                frame = pd.DataFrame({"case_id": []})
            else:
                raise AssertionError(f"No shard rows for {filename}")
        _assert_no_duplicates(
            frame,
            ["case_id", "source_candidate_id", "member_id", "mask_group",
             "control_type", "repeat", "fraction", "ablation", "bond_id",
             "bit_index", "group_id", "omics_feature_block", "condition"],
            filename,
        )
        _atomic_csv(output_root / filename, frame)
    attention = _merge_shards(output_root, "attention", ATTENTION_CSVS[0])
    if not attention.empty:
        _assert_no_duplicates(
            attention,
            ["candidate_id", "eval_row_id", "member_id", "attention_kind",
             "layer", "head", "atom_index"],
            ATTENTION_CSVS[0],
        )
        _atomic_csv(output_root / ATTENTION_CSVS[0], attention)
        primary = attention[attention["attention_kind"] == "primary"].copy()
        ensemble = ensemble_atom_attention(primary)
        consistency = pairwise_member_consistency(primary)
        variance = (
            primary.groupby(["candidate_id", "eval_row_id", "atom_index"], as_index=False)
            .agg(attention_mean=("attention", "mean"), attention_variance=("attention", "var"),
                 n_members=("member_id", "nunique"))
        )
        _atomic_csv(output_root / ATTENTION_CSVS[1], ensemble)
        _atomic_csv(output_root / ATTENTION_CSVS[2], consistency)
        _atomic_csv(output_root / ATTENTION_CSVS[4], variance)
    elif not allow_partial:
        raise AssertionError("No attention shard rows")
    routing = _merge_shards(output_root, "routing", "round19g_routing_audit.csv")
    if routing.empty:
        raise AssertionError("No routing audit shard")
    _assert_no_duplicates(routing, ["case_id"], "routing audit")
    if set(routing["case_id"].astype(str)) != expected_cases:
        raise AssertionError("routing audit case coverage mismatch")
    _atomic_csv(output_root / "round19g_routing_audit.csv", routing)
    occlusion = pd.read_csv(output_root / "round19g_atom_occlusion.csv")
    originals = (
        occlusion.groupby(["case_id", "source_candidate_id"], as_index=False)
        .agg(probability=("original_probability", "first"),
             role_aliases=("role_aliases", "first"))
    )
    role_rows = []
    selected_by_case = routing.set_index("case_id")["selected_role"].astype(str).to_dict()
    labels = verified["_cases"].set_index("case_id")["Label"].astype(int).to_dict()
    for record in originals.to_dict("records"):
        aliases = [value for value in str(record["role_aliases"]).split(",") if value]
        for role in aliases:
            probability = float(record["probability"])
            label = int(labels[str(record["case_id"])])
            role_rows.append({
                **record, "role": role, "selected_role": selected_by_case[str(record["case_id"])],
                "Label": label, "loss": -(label * math.log(max(probability, 1e-12))
                    + (1-label) * math.log(max(1-probability, 1e-12))),
                "classification": "post_lock_descriptive",
            })
    counterfactual = pd.DataFrame(role_rows)
    if counterfactual.empty and not allow_partial:
        raise AssertionError("No routing counterfactual probabilities")
    if not counterfactual.empty:
        regret = routing_regret(counterfactual)
        counterfactual = counterfactual.merge(regret, on=["case_id", "selected_role"], how="left")
    _atomic_csv(output_root / "round19g_routing_counterfactual.csv", counterfactual)
    case_summary = verified["_cases"].merge(
        routing[["case_id", "novelty_class", "selected_role", "routing_match"]],
        on="case_id", how="left", validate="one_to_one",
    )
    if not counterfactual.empty:
        summary_prob = counterfactual[
            counterfactual["role"] == counterfactual["selected_role"]
        ][["case_id", "probability", "descriptive_regret"]]
        case_summary = case_summary.merge(summary_prob, on="case_id", how="left")
    _atomic_csv(output_root / "round19g_case_summary.csv", case_summary)
    # Context-attention is a joined descriptive product, never a new forward pass.
    if not attention.empty:
        context = pd.read_csv(output_root / "round19g_context_sensitivity.csv")
        joined = ensemble.merge(
            context.groupby(["case_id"], as_index=False).agg(
                mean_context_probability_delta=("probability_delta", "mean")
            ),
            left_on="eval_row_id", right_on="case_id", how="left",
        )
        _atomic_csv(output_root / ATTENTION_CSVS[3], joined)
    (output_root / "experiment_lock.sha256").write_text(
        verified["_file_sha256"] + "\n", encoding="utf-8"
    )
    summary = {
        "complete": not allow_partial, "classification": "post_lock_descriptive",
        "roles_changed": False, "experiment_lock_sha256": verified["_file_sha256"],
        "case_count": len(expected_cases), "jobs": statuses,
        "outputs": list(OUTPUT_CSVS + ATTENTION_CSVS),
    }
    _atomic_json(output_root / "round19g_execution_summary.json", summary)
    return summary


def smoke(real_checkpoint: bool = False) -> dict[str, Any]:
    """Synthetic primitives plus an optional one-forward real checkpoint smoke."""
    from tools.round19_graph_features import build_pyg_data

    graph = build_pyg_data("CCO", with_bonds=False)
    altered = feature_zero_graph(graph, [1])
    if not torch.equal(graph.edge_index, altered.edge_index) or altered.x[1].count_nonzero():
        raise AssertionError("synthetic topology-preserving perturbation failed")
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "atomic.csv"
        _atomic_csv(path, pd.DataFrame({"case_id": ["synthetic"], "value": [1]}))
        if pd.read_csv(path).iloc[0]["value"] != 1:
            raise AssertionError("atomic CSV smoke failed")
    result: dict[str, Any] = {"synthetic": "passed", "real_checkpoint": "not_requested"}
    if real_checkpoint:
        final_path = PROJECT_ROOT / (
            "result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json"
        )
        lock = load_verified_lock(final_path, PROJECT_ROOT)
        record = lock["hashes"]["checkpoint_inventory"][0]
        checkpoint = _rooted(PROJECT_ROOT, record["checkpoint_path"])
        loaded = torch.load(checkpoint, map_location="cpu")
        for key in ("encoder", "fusion", "head"):
            if key not in loaded:
                raise KeyError(f"real checkpoint smoke missing {key}")
        prediction_path = (
            PROJECT_ROOT / "result/optimization_runs/round19_factorial/stage19f_posthoc"
            / "internal_test" / str(record["source_candidate_id"])
            / str(record["member_id"]) / "internal_test_predictions.csv"
        )
        split_path = PROJECT_ROOT / (
            "result/optimization_runs/round19_factorial/splits/internal_test_split.csv"
        )
        if not prediction_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(
                "single-checkpoint smoke requires existing post-hoc prediction and locked split"
            )
        existing = pd.read_csv(prediction_path).head(1).copy()
        if existing.empty or "probability" not in existing:
            raise AssertionError("existing checkpoint prediction smoke row is unavailable")
        split = pd.read_csv(split_path)
        source = split[split["_row_id"] == int(existing.iloc[0]["_row_id"])].copy()
        if len(source) != 1:
            raise AssertionError("existing prediction does not map to exactly one locked split row")
        source["eval_row_id"] = existing.iloc[0]["eval_row_id"]
        source["case_id"] = source["eval_row_id"]
        source = _canonical_cases(source)
        settings = _load_settings(str(PROJECT_ROOT / "config/round19_factorial_settings.json"))
        drug_id = str(loaded["drug_id"])
        predictor_id = str(loaded["predictor_id"])
        omics_id = str(loaded["omics_id"])
        drug_cfg = settings["drug_reps"][drug_id]
        dataset = Round19ResponseDataset(
            source,
            feature_dir=_feature_dir(settings, omics_id),
            drug_smiles_path=settings["drug_smiles_path"],
            encoder_type=drug_cfg["type"],
            with_bonds=bool(drug_cfg.get("edge_features")),
            omics_id=omics_id,
        )

        def factory(_checkpoint):
            encoder, fusion, head, _, _ = _build_encoder_fusion_head(
                settings, drug_id=drug_id, predictor_id=predictor_id,
                omics_dim=dataset.omics_dim, device=torch.device("cpu"),
            )
            return encoder, fusion, head

        encoder, fusion, head, _ = strict_load_locked_models(
            record, project_root=PROJECT_ROOT, model_factory=factory
        )
        for module in (encoder, fusion, head):
            module.eval()
        probability, _ = _predict_batch(
            encoder, fusion, head, round19_collate_fn([dataset[0]]),
            device=torch.device("cpu"), encoder_type=str(drug_cfg["type"]),
            predictor_id=predictor_id,
        )
        expected_probability = float(existing.iloc[0]["probability"])
        # Existing formal predictions were emitted under CUDA AMP/float16.  A
        # CPU float32 replay must agree within the observed quantization bound.
        if not np.isclose(float(probability[0]), expected_probability, atol=5e-3, rtol=5e-3):
            raise AssertionError(
                "real checkpoint original probability differs from its existing prediction: "
                f"runtime={probability[0]} existing={expected_probability}"
            )
        result["real_checkpoint"] = {
            "path": str(checkpoint), "sha256": sha256_file(checkpoint),
            "member_id": record["member_id"], "state_dicts": "strictly_loaded",
            "existing_probability": expected_probability,
            "runtime_probability": float(probability[0]),
            "probability_match": True,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19G locked inference executor")
    parser.add_argument("--experiment-lock")
    parser.add_argument("--output-root", default=(
        "result/optimization_runs/round19_factorial/stage19g"
    ))
    parser.add_argument("--settings", default="config/round19_factorial_settings.json")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--task-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--perturbation-batch", type=int, default=128)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--real-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(smoke(args.real_checkpoint), indent=2))
        return
    if not args.experiment_lock:
        parser.error("--experiment-lock is required outside smoke")
    if args.finalize:
        result = finalize_outputs(
            experiment_lock=Path(args.experiment_lock),
            output_root=Path(args.output_root),
            allow_partial=args.allow_partial,
        )
    else:
        if not args.method or not args.task_id:
            parser.error("--method and --task-id are required for a job")
        result = execute_job(
            experiment_lock=Path(args.experiment_lock), method=args.method,
            task_id=args.task_id, output_root=Path(args.output_root),
            settings=Path(args.settings), device_name=args.device,
            batch_size=args.batch_size, perturbation_batch=args.perturbation_batch,
            case_limit=args.case_limit,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or ("cuda" in message and "memory" in message):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(json.dumps({"oom": True, "detail": str(exc)}), file=sys.stderr)
            raise SystemExit(42)
        raise
