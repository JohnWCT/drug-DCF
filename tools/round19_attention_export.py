#!/usr/bin/env python3
"""Memory-bounded, lock-bound Stage 19G atom-attention export foundation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

import pandas as pd
import torch
from torch import nn

from tools.round19_stage19f_ensemble import REQUIRED_MEMBER_IDS
from tools.round19_stage19g_lock_adapter import load_verified_lock
from tools.round19_train_loop import forward_round19_batch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_candidate_members(
    final_lock_path: Path,
    *,
    project_root: Path,
    source_candidate_id: str,
) -> list[dict[str, Any]]:
    """Return the exact locked 15-member inventory for one source candidate."""
    lock = load_verified_lock(Path(final_lock_path), Path(project_root))
    records = [
        dict(item)
        for item in lock["hashes"]["checkpoint_inventory"]
        if str(item["source_candidate_id"]) == str(source_candidate_id)
    ]
    members = {str(item["member_id"]) for item in records}
    if len(records) != 15 or members != set(REQUIRED_MEMBER_IDS):
        raise AssertionError("Attention export requires the complete locked 15-member grid")
    for item in records:
        item["lock_payload_sha256"] = lock["hashes"]["lock_payload_sha256"]
        item["lock_file_sha256"] = lock["_lock_file_sha256"]
    return sorted(records, key=lambda item: str(item["member_id"]))


def load_case_shard(path: Path, *, partition: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"eval_row_id", "ModelID", "DRUG_NAME"}
    missing = required - set(frame)
    if missing or frame.empty:
        raise AssertionError(
            f"Invalid Stage 19G case shard; missing={sorted(missing)} empty={frame.empty}"
        )
    if frame["eval_row_id"].astype(str).duplicated().any():
        raise AssertionError("case shard eval_row_id must be unique")
    if partition is not None:
        if "partition" not in frame:
            raise KeyError("case shard requires partition for partition-bound export")
        if set(frame["partition"].astype(str)) != {str(partition)}:
            raise AssertionError("case shard crosses the requested partition")
    return frame


def strict_load_locked_models(
    record: Mapping[str, Any],
    *,
    project_root: Path,
    model_factory: Callable[
        [Mapping[str, Any]], Tuple[nn.Module, nn.Module, nn.Module]
    ],
    map_location: torch.device | str = "cpu",
) -> tuple[nn.Module, nn.Module, nn.Module, Mapping[str, Any]]:
    """Construct models and strict-load only a hash-pinned checkpoint."""
    checkpoint_path = Path(str(record["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(project_root) / checkpoint_path
    if _sha256(checkpoint_path) != str(record["checkpoint_sha256"]):
        raise AssertionError(f"Locked checkpoint hash mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Round19 checkpoint must be a mapping")
    if {"split_seed", "fold_id"} <= set(checkpoint):
        checkpoint_member = (
            f"seed{int(checkpoint['split_seed'])}_fold{int(checkpoint['fold_id'])}"
        )
        if str(record["member_id"]) != checkpoint_member:
            raise AssertionError("checkpoint/member identity mismatch")
    elif str(record["member_id"]) not in REQUIRED_MEMBER_IDS:
        raise AssertionError("legacy checkpoint lacks a valid locked member attestation")
    encoder, fusion, head = model_factory(checkpoint)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    fusion.load_state_dict(checkpoint["fusion"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    return encoder, fusion, head, checkpoint


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    local = dict(batch)
    local["omics"] = batch["omics"].to(device)
    if local.get("drug_batch") is not None:
        local["drug_batch"] = local["drug_batch"].to(device)
    if local.get("maccs") is not None:
        local["maccs"] = local["maccs"].to(device)
    return local


@torch.no_grad()
def export_attention_batches(
    *,
    encoder: nn.Module,
    fusion: nn.Module,
    head: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    output_path: Path,
    device: torch.device,
    encoder_type: str,
    predictor_id: str,
    provenance: Mapping[str, Any],
    case_shard: pd.DataFrame,
) -> int:
    """Write logits and raw+primary attention from each single forward pass.

    Rows are streamed batch-by-batch after moving only the current attention to
    CPU; the complete attention tensor is never retained on GPU.
    """
    if str(predictor_id).upper() != "P2":
        raise ValueError("Real atom attention export is supported only for P2")
    required_case_columns = {"eval_row_id", "ModelID", "DRUG_NAME"}
    if required_case_columns - set(case_shard) or case_shard.empty:
        raise AssertionError("A non-empty validated case shard is required")
    expected_cases = set(case_shard["eval_row_id"].astype(str))
    if len(expected_cases) != len(case_shard):
        raise AssertionError("case shard eval_row_id must be unique")
    seen_cases: set[str] = set()
    encoder.eval()
    fusion.eval()
    head.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id", "member_id", "eval_row_id", "ModelID", "drug_name",
        "target_key", "logit", "attention_kind", "layer", "head", "atom_index",
        "attention", "is_valid_atom", "graph_smiles", "legacy_input_smiles",
        "actual_smiles_source", "atom_symbol", "original_atom_index",
        "checkpoint_path", "checkpoint_sha256", "lock_payload_sha256",
    ]
    written = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch in dataloader:
            batch_cases = {str(value) for value in batch["eval_row_id"]}
            if not batch_cases <= expected_cases:
                raise AssertionError("dataloader contains rows outside the case shard")
            if seen_cases & batch_cases:
                raise AssertionError("case shard row exported more than once")
            seen_cases.update(batch_cases)
            local = _move_batch(batch, device)
            result = forward_round19_batch(
                encoder=encoder,
                fusion=fusion,
                encoder_type=encoder_type,
                predictor_id=predictor_id,
                omics=local["omics"],
                batch=local,
                return_interpretability=True,
            )
            if not isinstance(result, dict):
                raise RuntimeError("P2 interpretability forward did not return metadata")
            logits = head(result["representation"]).view(-1)
            raw = result["attention_raw"].detach().float().cpu()
            primary = result["attention_primary"].detach().float().cpu()
            valid = result["atom_valid_mask"].detach().cpu().bool()
            if raw.ndim != 5 or raw.shape[3] != 1:
                raise AssertionError(f"Invalid attention shape: {tuple(raw.shape)}")
            if torch.count_nonzero(raw.masked_select(~valid[None, :, None, None, :])):
                raise AssertionError("Padding attention must be exactly zero in eval")
            sums = raw.sum(dim=-1)
            expected = valid.any(dim=-1)[None, :, None, None].expand_as(sums)
            if not torch.allclose(sums[expected], torch.ones_like(sums[expected]), atol=1e-5):
                raise AssertionError("Valid eval attention rows must sum to one")

            raw_np, primary_np = raw.numpy(), primary.numpy()
            logits_cpu = logits.detach().float().cpu().numpy()
            for batch_index in range(valid.shape[0]):
                metadata = batch["graph_metadata"][batch_index]
                atoms = metadata["atom_metadata"]
                common = {
                    "candidate_id": provenance["candidate_id"],
                    "member_id": provenance["member_id"],
                    "eval_row_id": str(batch["eval_row_id"][batch_index]),
                    "ModelID": str(batch["ModelID"][batch_index]),
                    "drug_name": str(batch["drug_name"][batch_index]),
                    "target_key": str(batch["target_key"][batch_index]),
                    "logit": float(logits_cpu[batch_index]),
                    "graph_smiles": batch["graph_smiles"][batch_index],
                    "legacy_input_smiles": batch["legacy_input_smiles"][batch_index],
                    "actual_smiles_source": batch["actual_smiles_source"][batch_index],
                    "checkpoint_path": provenance["checkpoint_path"],
                    "checkpoint_sha256": provenance["checkpoint_sha256"],
                    "lock_payload_sha256": provenance["lock_payload_sha256"],
                }
                for atom_index in range(int(valid[batch_index].sum())):
                    atom = atoms[atom_index]
                    atom_common = {
                        **common,
                        "atom_index": atom_index,
                        "is_valid_atom": True,
                        "atom_symbol": atom["symbol"],
                        "original_atom_index": atom["original_atom_index"],
                    }
                    writer.writerow(
                        {
                            **atom_common,
                            "attention_kind": "primary",
                            "layer": raw.shape[0] - 1,
                            "head": "mean",
                            "attention": float(primary_np[batch_index, atom_index]),
                        }
                    )
                    written += 1
                    for layer in range(raw.shape[0]):
                        for attention_head in range(raw.shape[2]):
                            writer.writerow(
                                {
                                    **atom_common,
                                    "attention_kind": "raw",
                                    "layer": layer,
                                    "head": attention_head,
                                    "attention": float(
                                        raw_np[layer, batch_index, attention_head, 0, atom_index]
                                    ),
                                }
                            )
                            written += 1
            del result, raw, primary, logits
    if seen_cases != expected_cases:
        raise AssertionError(
            f"case shard coverage mismatch: missing={sorted(expected_cases-seen_cases)[:5]}"
        )
    return written


__all__ = [
    "export_attention_batches",
    "load_case_shard",
    "locked_candidate_members",
    "strict_load_locked_models",
]
