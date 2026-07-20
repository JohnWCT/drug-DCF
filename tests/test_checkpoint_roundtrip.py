from __future__ import annotations

import torch

from biocda.models.model_factory import build_model
from biocda.training.checkpoint import load_biocda_checkpoint, save_biocda_checkpoint


def test_checkpoint_roundtrip(xa_config, tmp_path):
    model = build_model(xa_config)
    ckpt = tmp_path / "model.pt"
    save_biocda_checkpoint(ckpt, model=model, config=xa_config, epoch=3, architecture_version="biocda-xa-v1")
    model2 = build_model(xa_config)
    report = load_biocda_checkpoint(model2, ckpt, strict=True)
    assert not report.missing_keys


def test_strict_checkpoint_loading(xa_config, tmp_path):
    model = build_model(xa_config)
    ckpt = tmp_path / "model.pt"
    save_biocda_checkpoint(ckpt, model=model, config=xa_config, epoch=0)
    model2 = build_model(xa_config)
    report = load_biocda_checkpoint(model2, ckpt, strict=True)
    assert report.architecture_version == "biocda-xa-v1"


def test_architecture_version_saved(xa_config, tmp_path):
    ckpt = tmp_path / "model.pt"
    save_biocda_checkpoint(
        ckpt,
        model=build_model(xa_config),
        config=xa_config,
        epoch=0,
        architecture_version="biocda-xa-v1",
    )
    assert torch.load(ckpt, map_location="cpu")["architecture_version"] == "biocda-xa-v1"


def test_encoder_hashes_saved(xa_config, tmp_path):
    ckpt = tmp_path / "model.pt"
    save_biocda_checkpoint(
        ckpt,
        model=build_model(xa_config),
        config=xa_config,
        epoch=0,
        omics_encoder_sha="o",
        drug_encoder_sha="d",
        context_artifact_sha="c",
    )
    blob = torch.load(ckpt, map_location="cpu")
    assert blob["omics_encoder_sha"] == "o" and blob["drug_encoder_sha"] == "d"
