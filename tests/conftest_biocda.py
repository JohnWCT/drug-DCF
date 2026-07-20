"""Shared fixtures for BioCDA architecture tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from biocda.data.drug_graph import batch_drug_graphs, make_chain_graph
from biocda.models.model_factory import build_model, build_model_config_for_type


@pytest.fixture()
def xa_config():
    path = Path(__file__).resolve().parents[1] / "configs/model/biocda_cross_attention.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture()
def pooled_config():
    base = Path(__file__).resolve().parents[1] / "configs/model/biocda_cross_attention.yaml"
    return build_model_config_for_type(yaml.safe_load(base.read_text(encoding="utf-8")), "pooled_baseline")


@pytest.fixture()
def xa_model(xa_config):
    model = build_model(xa_config)
    model.eval()
    return model


@pytest.fixture()
def pooled_model(pooled_config):
    model = build_model(pooled_config)
    model.eval()
    return model


@pytest.fixture()
def sample_batch(xa_config):
    omics_dim = xa_config["model"]["omics_encoder"]["latent_dim"]
    ctx_dim = xa_config["model"]["biological_context"]["context_dim"]
    omics = torch.randn(2, omics_dim)
    context = torch.randn(2, ctx_dim)
    graphs = [make_chain_graph(14), make_chain_graph(6)]
    return omics, context, batch_drug_graphs(graphs)
