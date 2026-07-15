#!/usr/bin/env python3
"""Synthetic checks for Round 19 native post-hoc inference contracts."""
from __future__ import annotations

import json
import pickle
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch_geometric.data import Batch

import step1_finetune_latent_pipeline_round19 as pipeline
from tools.round19_graph_features import build_pyg_data
from tools.round19_train_loop import forward_round19_batch


def _pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def smoke_o2(root: Path) -> None:
    o2 = root / "z_plus_context16"
    o3 = root / "z_plus_summary_context16"
    vector = np.arange(91, dtype=np.float32)
    _pickle(o3 / "tcga_latent_proto.pkl", {"TCGA-AA-0001-01": vector})
    names = [f"f{i}" for i in range(80)]
    _json(o2 / "feature_names.json", names)
    _json(o3 / "feature_names.json", names + [f"s{i}" for i in range(11)])
    _json(o3 / "projection_metadata.json", {"fit_domain": "source_only"})
    for path in (o2 / "projection_model.pkl", o3 / "projection_model.pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"same-development-projection")
    latent, report = pipeline.preflight_tcga_features(
        {"round19_feature_out_root": str(root)}, "O2"
    )
    assert report["ok"] and report["fit_on_tcga_labels"] is False
    np.testing.assert_array_equal(latent["TCGA-AA-0001-01"], vector[:80])


def smoke_o4(root: Path) -> None:
    o1 = root / "z_plus_summary"
    o4 = root / "z_plus_source_proto_features"
    summary = list(pipeline.SOURCE_SUMMARY_NAMES) + [f"target_{i}" for i in range(4)]
    o1_names = [f"z_dim{i:03d}" for i in range(64)] + summary
    o4_names = [f"z_dim{i:03d}" for i in range(64)] + list(
        pipeline.SOURCE_SUMMARY_NAMES
    )
    source = np.arange(75, dtype=np.float32)
    selected = np.concatenate([source[:64], source[64:71]]).astype(np.float32)
    _pickle(o1 / "ccle_latent_proto.pkl", {"A": source})
    _pickle(o4 / "ccle_latent_proto.pkl", {"A": selected})
    _pickle(o1 / "tcga_latent_proto.pkl", {"TCGA-AA-0001-01": source})
    _json(o1 / "feature_names.json", o1_names)
    _json(o4 / "feature_names.json", o4_names)
    _json(
        o1 / "feature_metadata.json",
        {"n_tcga_samples": 1, "scaler": {"type": "standard"}},
    )
    latent, report = pipeline.preflight_tcga_features(
        {"round19_feature_out_root": str(root)}, "O4"
    )
    assert report["ok"] and latent["TCGA-AA-0001-01"].shape == (71,)


def smoke_checkpoint(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint.pt"
    modules = (
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 1),
    )
    torch.save(
        {
            "encoder": modules[0].state_dict(),
            "fusion": modules[1].state_dict(),
            "head": modules[2].state_dict(),
            "drug_id": "D0",
            "predictor_id": "P0",
            "omics_id": "O0",
            "split_seed": 52,
            "fold_id": 3,
            "encoder_type": "gin",
        },
        checkpoint_path,
    )
    args = Namespace(
        checkpoint_path=str(checkpoint_path),
        result_dir=str(root),
        drug_id="D0",
        predictor_id="P0",
        omics_id="O0",
        split_seed=52,
        fold_id=3,
    )
    cfg = {"type": "gin", "edge_features": False}
    with mock.patch.object(
        pipeline,
        "_build_encoder_fusion_head",
        return_value=(*modules, "gin", cfg),
    ):
        pipeline._strict_load_inference_models(
            args,
            {"drug_reps": {"D0": cfg}},
            omics_dim=64,
            device=torch.device("cpu"),
        )
    args.fold_id = 4
    try:
        with mock.patch.object(
            pipeline,
            "_build_encoder_fusion_head",
            return_value=(*modules, "gin", cfg),
        ):
            pipeline._strict_load_inference_models(
                args,
                {"drug_reps": {"D0": cfg}},
                omics_dim=64,
                device=torch.device("cpu"),
            )
    except AssertionError:
        pass
    else:
        raise AssertionError("checkpoint identity mismatch did not fail closed")


def smoke_native_model_matrix() -> None:
    settings = json.loads(
        Path("config/round19_factorial_settings.json").read_text(encoding="utf-8")
    )
    device = torch.device("cpu")
    omics = torch.randn(2, 64)
    cases = (
        ("D0", "P0"),  # GIN pooled MLP
        ("D3", "P2"),  # GINE bond features + atom cross-attention
        ("D4", "P1"),  # MACCS + compact pooled transformer
    )
    for drug_id, predictor_id in cases:
        encoder, fusion, head, encoder_type, drug_cfg = (
            pipeline._build_encoder_fusion_head(
                settings,
                drug_id=drug_id,
                predictor_id=predictor_id,
                omics_dim=64,
                device=device,
            )
        )
        if encoder_type == "maccs":
            batch = {"maccs": torch.zeros(2, 166), "drug_batch": None}
        else:
            graph = build_pyg_data(
                "CCO", with_bonds=bool(drug_cfg.get("edge_features"))
            )
            batch = {
                "maccs": None,
                "drug_batch": Batch.from_data_list([graph, graph]),
            }
        representation = forward_round19_batch(
            encoder=encoder,
            fusion=fusion,
            encoder_type=encoder_type,
            predictor_id=predictor_id,
            omics=omics,
            batch=batch,
        )
        probability = torch.sigmoid(head(representation).view(-1))
        assert probability.shape == (2,)
        assert torch.isfinite(probability).all()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        smoke_o2(root / "o2")
        smoke_o4(root / "o4")
        smoke_checkpoint(root / "checkpoint")
        smoke_native_model_matrix()
    print(json.dumps({"ok": True, "synthetic_jobs": 4, "full_jobs_started": 0}))


if __name__ == "__main__":
    main()
