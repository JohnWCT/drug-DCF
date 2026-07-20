"""Drug GIN node encoder — atom embeddings only, no graph pooling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from biocda.data.atom_metadata import build_atom_ptr
from drugmodels.ginconv import GINConvNet


@dataclass
class DrugNodeEncoderOutput:
    node_embeddings: Tensor
    batch_index: Tensor
    atom_ptr: Tensor
    model_atom_index: Optional[Tensor] = None
    original_atom_index: Optional[Tensor] = None
    rdkit_atom_index: Optional[Tensor] = None


def _resolve_model_atom_index(data: Data | Batch) -> Tensor:
    if hasattr(data, "model_atom_index") and data.model_atom_index is not None:
        return data.model_atom_index
    if isinstance(data, Batch):
        counts = torch.bincount(data.batch, minlength=int(data.num_graphs))
        parts = [torch.arange(n, device=data.x.device) for n in counts.tolist()]
        return torch.cat(parts, dim=0)
    return torch.arange(data.num_nodes, device=data.x.device)


class DrugGINNodeEncoder(nn.Module):
    """D0 GIN encoder returning atom-level node embeddings only."""

    def __init__(
        self,
        *,
        input_dim: int = 78,
        node_hidden_dim: int = 32,
        num_layers: int = 5,
        jk_mode: str = "last",
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        frozen: bool = False,
    ) -> None:
        super().__init__()
        self.gin = GINConvNet(
            input_dim=input_dim,
            node_hidden_dim=node_hidden_dim,
            graph_output_dim=node_hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            jk_mode=jk_mode,
            pool_type="max",
            use_batch_norm=use_batch_norm,
        )
        self.node_dim = int(self.gin.node_dim)
        self.frozen = bool(frozen)
        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, drug_graph: Data | Batch) -> DrugNodeEncoderOutput:
        node_embeddings = self.gin.encode_nodes(drug_graph.x, drug_graph.edge_index)
        if isinstance(drug_graph, Batch):
            batch_index = drug_graph.batch
            num_graphs = int(drug_graph.num_graphs)
        else:
            batch_index = torch.zeros(drug_graph.num_nodes, dtype=torch.long, device=drug_graph.x.device)
            num_graphs = 1
        atom_ptr = build_atom_ptr(batch_index, num_graphs)
        return DrugNodeEncoderOutput(
            node_embeddings=node_embeddings,
            batch_index=batch_index,
            atom_ptr=atom_ptr,
            model_atom_index=_resolve_model_atom_index(drug_graph),
            original_atom_index=getattr(drug_graph, "original_atom_index", None),
            rdkit_atom_index=getattr(drug_graph, "rdkit_atom_index", None),
        )


class DrugGINPooledEncoder(nn.Module):
    """Legacy D0 pooled encoder for baseline comparison."""

    def __init__(self, gin: GINConvNet) -> None:
        super().__init__()
        self.gin = gin

    @classmethod
    def from_node_encoder(cls, node_encoder: DrugGINNodeEncoder) -> "DrugGINPooledEncoder":
        return cls(node_encoder.gin)

    def forward(self, drug_graph: Data | Batch) -> Tensor:
        out = self.gin(drug_graph, return_dict=True, return_graph_embedding=True)
        graph_embedding = out["graph_embedding"]
        if graph_embedding is None:
            raise RuntimeError("GIN pooled encoder failed to produce graph_embedding")
        return graph_embedding
