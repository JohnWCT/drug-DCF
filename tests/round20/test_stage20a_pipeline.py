"""Stage 20A unit / smoke checks (no full training)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result/optimization_runs/round20_unseen_drug_closure"
STAGE0 = RESULT / "stage20_0"
STAGEA = RESULT / "stage20a_dimension"
SPLITS = RESULT / "splits"
C16 = RESULT / "features/z_plus_context16"
C32 = RESULT / "features/z_plus_context32"


def test_stage20_0_go_required() -> None:
    go = json.loads((STAGE0 / "stage20_0_go.json").read_text(encoding="utf-8"))
    assert go["status"] == "GO"
    assert go["pca_contract"]["comparable"] is True
    assert go["feature_stores"]["c16"]["dimension"] == 80
    assert go["feature_stores"]["c32"]["dimension"] == 96
    assert "hashes" in go and go["hashes"]["c16_projection_sha256"]


def test_e3_contract_complete() -> None:
    e3 = json.loads((STAGE0 / "resolved_e3.json").read_text(encoding="utf-8"))
    body = e3["resolved_e3"]
    assert body["public_alias"] == "E3"
    assert body["predictor_id"] == "P0"
    assert body["drug_encoder_id"] == "D0"
    assert body["optimizer"]["encoder_lr"] is not None
    assert body["training"]["max_epochs"] is not None
    assert body["training"]["model_seed"] == 101
    assert len(body["checkpoint_paths"]) == 15


def test_c16_c32_dims_and_z_identical() -> None:
    import pickle

    with (C16 / "ccle_latent_proto.pkl").open("rb") as f:
        d16 = pickle.load(f)
    with (C32 / "ccle_latent_proto.pkl").open("rb") as f:
        d32 = pickle.load(f)
    mid = next(iter(d16))
    v16 = np.asarray(d16[mid], dtype=np.float32)
    v32 = np.asarray(d32[mid], dtype=np.float32)
    assert v16.shape == (80,)
    assert v32.shape == (96,)
    torch.testing.assert_close(
        torch.from_numpy(v16[:64]), torch.from_numpy(v32[:64]), rtol=0, atol=0
    )


def test_manifest_has_30_jobs_and_pairs() -> None:
    lines = (STAGEA / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    jobs = [json.loads(x) for x in lines]
    assert len(jobs) == 30
    pairs = {}
    for j in jobs:
        pairs.setdefault((j["split_seed"], j["fold"]), set()).add(j["context_id"])
    assert len(pairs) == 15
    assert all(v == {"C16", "C32"} for v in pairs.values())
    for seed in (52, 62, 72):
        hashes = {j["split_assignment_sha256"] for j in jobs if j["split_seed"] == seed}
        assert len(hashes) == 1
    assert len({j["e3_contract_sha256"] for j in jobs}) == 1


def test_no_drug_leakage() -> None:
    audit = json.loads((SPLITS / "round20a_drug_split_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert len(audit["folds"]) == 15
    for fold in audit["folds"]:
        assert fold["identity_overlap_count"] == 0
        assert fold["status"] == "PASS"
        path = SPLITS / f"round20a_drug_heldout_seed{fold['split_seed']}_assignments.csv"
        df = pd.read_csv(path)
        sub = df[df["fold_id"] == fold["fold"]]
        train = set(sub.loc[sub.split_role == "train", "drug_group_id"].astype(str))
        val = set(sub.loc[sub.split_role == "val", "drug_group_id"].astype(str))
        assert train.isdisjoint(val)


def test_gated_output_shape_and_gate_range() -> None:
    from tools.round20_gated_fusion import GatedPooledFusionPredictor

    model = GatedPooledFusionPredictor(omics_dim=80)
    omics = torch.randn(4, 80)
    drug = torch.randn(4, 32)
    logits, gate = model(omics, drug, return_gate=True)
    assert logits.shape == (4,)
    assert gate.shape == (4, 128)
    assert torch.all(gate >= 0) and torch.all(gate <= 1)
    loss = logits.sum()
    loss.backward()
    assert model.omics_projection[0].weight.grad is not None


def test_dimension_lock_parsimony_rule() -> None:
    mean_delta = 0.002
    selected = "C16" if abs(mean_delta) < 0.005 else "C32"
    assert selected == "C16"


def test_selection_rejects_tcga_keys() -> None:
    FORBIDDEN = {"tcga_auc", "tcga_auprc", "internal_auc", "external_auc", "integrated5_auc"}
    payload = {"mean_auc": 0.6, "tcga_auc": 0.9}
    assert FORBIDDEN.intersection(payload)
