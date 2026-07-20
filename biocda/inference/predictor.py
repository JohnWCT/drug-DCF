"""Inference-only predictor wrapper."""
from __future__ import annotations

from typing import Union

import torch
from torch_geometric.data import Batch, Data

from biocda.models.biocda_model import BioCDA
from biocda.models.outputs import BioCDAOutput
from biocda.models.pooled_baseline import PooledBaselineModel


class BioCDAPredictor:
    def __init__(self, model: Union[BioCDA, PooledBaselineModel]) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        omics: torch.Tensor,
        biological_context: torch.Tensor,
        drug_graph: Union[Data, Batch],
        *,
        output_mode: str = "prediction",
    ) -> BioCDAOutput:
        return self.model(
            omics,
            biological_context,
            drug_graph,
            output_mode=output_mode,
        )
