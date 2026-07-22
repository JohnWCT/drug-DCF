"""BioCDA-XA v2 model — no pooling, query-only response head."""
from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from biocda.models.xa.cross_attention import OmicsAtomCrossAttentionStack
from biocda.models.xa.gin_atom_encoder import GINAtomEncoder
from biocda.models.xa.outputs import BioCDAXAOutput
from biocda.models.xa.response_head import XAQueryResponseHead
from biocda.models.xa.sample_query import SampleQueryProjector, SampleQueryProjectorZOnly


class BioCDAXA(nn.Module):
    """Omics-conditioned atom-level cross-attention candidate (biocda-xa-v2)."""

    ARCHITECTURE_VERSION = "biocda-xa-v2"
    ARCHITECTURE_NAME = "BioCDA-XA-Candidate"
    VALID_OUTPUT_MODES = frozenset({"prediction", "attention", "full"})

    def __init__(
        self,
        sample_projector: nn.Module,
        drug_encoder: GINAtomEncoder,
        cross_attention: OmicsAtomCrossAttentionStack,
        response_head: XAQueryResponseHead,
        *,
        use_context: bool = True,
    ) -> None:
        super().__init__()
        self.sample_projector = sample_projector
        self.drug_encoder = drug_encoder
        self.cross_attention = cross_attention
        self.response_head = response_head
        self.use_context = bool(use_context)

    def forward(
        self,
        omics: torch.Tensor,
        biological_context: torch.Tensor,
        drug_graph: Union[Data, Batch],
        *,
        output_mode: str = "prediction",
        attention_override: Optional[torch.Tensor] = None,
    ) -> BioCDAXAOutput:
        if output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode: {output_mode}")

        assert omics.shape[-1] == 64, f"base_omics must be 64-d, got {omics.shape[-1]}"
        if self.use_context:
            assert biological_context.shape[-1] == 32, (
                f"prototype_context must be 32-d, got {biological_context.shape[-1]}"
            )

        sample_features, initial_query = self.sample_projector(omics, biological_context)
        atoms = self.drug_encoder(drug_graph)
        attn = self.cross_attention(
            initial_query,
            atoms.node_embeddings,
            atoms.batch_index,
            attention_override=attention_override,
        )
        logits = self.response_head(attn.final_query)
        logits = logits.reshape(-1)
        probabilities = torch.sigmoid(logits)

        base = BioCDAXAOutput(
            logits=logits,
            probabilities=probabilities,
            architecture_version=self.ARCHITECTURE_VERSION,
        )
        if output_mode == "prediction":
            return base

        base.attention_logits = attn.attention_logits
        base.attention_probabilities = attn.attention_probabilities
        base.atom_mask = attn.atom_mask
        base.atom_batch_index = atoms.batch_index
        base.atom_ptr = atoms.atom_ptr
        base.model_atom_index = atoms.model_atom_index
        base.original_atom_index = atoms.original_atom_index
        base.rdkit_atom_index = atoms.rdkit_atom_index

        if output_mode == "full":
            base.base_omics_latent = omics
            base.prototype_context = biological_context
            base.sample_features = sample_features
            base.initial_query = initial_query
            base.final_query = attn.final_query
            base.node_embeddings = atoms.node_embeddings
            base.dense_atom_tokens = attn.dense_atom_tokens
        return base
