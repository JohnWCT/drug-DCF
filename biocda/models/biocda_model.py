"""BioCDA cross-attention model."""
from __future__ import annotations

from typing import Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from biocda.models.cross_attention import SampleAtomCrossAttention
from biocda.models.outputs import BioCDAOutput
from biocda.models.response_head import BioCDAResponseHead


class BioCDA(nn.Module):
    """BioCDA-XA: sample-to-atom cross-attention."""

    VALID_OUTPUT_MODES = frozenset({"prediction", "attention", "full", "debug"})
    ARCHITECTURE_VERSION = "biocda-xa-v1"

    def __init__(
        self,
        omics_encoder: nn.Module,
        sample_encoder: nn.Module,
        drug_encoder: nn.Module,
        cross_attention: SampleAtomCrossAttention,
        response_head: BioCDAResponseHead,
        *,
        architecture_name: str = "BioCDA-XA-ZC",
    ) -> None:
        super().__init__()
        self.omics_encoder = omics_encoder
        self.sample_encoder = sample_encoder
        self.drug_encoder = drug_encoder
        self.cross_attention = cross_attention
        self.response_head = response_head
        self.architecture_name = architecture_name

    def forward(
        self,
        omics: torch.Tensor,
        biological_context: torch.Tensor,
        drug_graph: Union[Data, Batch],
        *,
        output_mode: str = "prediction",
    ) -> BioCDAOutput:
        if output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode: {output_mode}")

        omics_latent = self.omics_encoder(omics)
        sample_repr = self.sample_encoder(omics_latent, biological_context)
        drug_nodes = self.drug_encoder(drug_graph)
        attention_output = self.cross_attention(
            sample_repr,
            drug_nodes.node_embeddings,
            drug_nodes.batch_index,
        )
        logits = self.response_head(sample_repr, attention_output.drug_representation)
        logits = logits.reshape(-1)
        probabilities = torch.sigmoid(logits)

        base = {
            "logits": logits,
            "probabilities": probabilities,
            "architecture_version": self.ARCHITECTURE_VERSION,
        }

        if output_mode == "prediction":
            return BioCDAOutput(**base)

        meta = {
            "atom_attention": attention_output.attention_probabilities,
            "atom_attention_logits": attention_output.attention_logits,
            "atom_mask": attention_output.atom_mask,
            "atom_batch_index": drug_nodes.batch_index,
            "atom_ptr": drug_nodes.atom_ptr,
            "model_atom_index": drug_nodes.model_atom_index,
            "original_atom_index": drug_nodes.original_atom_index,
            "rdkit_atom_index": drug_nodes.rdkit_atom_index,
        }
        if output_mode in {"full", "debug"}:
            meta.update(
                {
                    "sample_representation": sample_repr,
                    "omics_latent": omics_latent,
                    "biological_context": biological_context,
                    "drug_representation": attention_output.drug_representation,
                    "node_embeddings": drug_nodes.node_embeddings,
                }
            )
        if output_mode == "debug":
            meta["attention_probabilities_used"] = attention_output.attention_probabilities_used

        return BioCDAOutput(**base, **meta)
