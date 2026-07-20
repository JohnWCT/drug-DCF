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
    """BioCDA-XA: biological-context-guided sample-to-atom cross-attention."""

    VALID_OUTPUT_MODES = frozenset({"prediction", "attention", "full"})
    ARCHITECTURE_VERSION = "biocda-xa-v1"

    def __init__(
        self,
        omics_encoder: nn.Module,
        sample_encoder: nn.Module,
        drug_encoder: nn.Module,
        cross_attention: SampleAtomCrossAttention,
        response_head: BioCDAResponseHead,
    ) -> None:
        super().__init__()
        self.omics_encoder = omics_encoder
        self.sample_encoder = sample_encoder
        self.drug_encoder = drug_encoder
        self.cross_attention = cross_attention
        self.response_head = response_head

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
        probabilities = torch.sigmoid(logits)

        if output_mode == "prediction":
            return BioCDAOutput(logits=logits, probabilities=probabilities)

        if output_mode == "attention":
            return BioCDAOutput(
                logits=logits,
                probabilities=probabilities,
                atom_attention=attention_output.attention_probabilities,
                atom_attention_logits=attention_output.attention_logits,
                atom_mask=attention_output.atom_mask,
                model_atom_index=drug_nodes.model_atom_index,
                original_atom_index=drug_nodes.original_atom_index,
                rdkit_atom_index=drug_nodes.rdkit_atom_index,
            )

        return BioCDAOutput(
            logits=logits,
            probabilities=probabilities,
            sample_representation=sample_repr,
            omics_latent=omics_latent,
            biological_context=biological_context,
            drug_representation=attention_output.drug_representation,
            node_embeddings=drug_nodes.node_embeddings,
            atom_attention=attention_output.attention_probabilities,
            atom_attention_logits=attention_output.attention_logits,
            atom_mask=attention_output.atom_mask,
            model_atom_index=drug_nodes.model_atom_index,
            original_atom_index=drug_nodes.original_atom_index,
            rdkit_atom_index=drug_nodes.rdkit_atom_index,
        )
