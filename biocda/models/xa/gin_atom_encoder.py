"""GIN atom encoder — node embeddings only, no graph pooling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from biocda.data.atom_metadata import build_atom_ptr
from drugmodels.ginconv import GINConvNet

FORBIDDEN_POOL_ATTRS = (
    "pool",
    "pool_type",
    "fc1_xd",
    "out",
)


@dataclass
class AtomEncoderOutput:
    node_embeddings: Tensor
    batch_index: Tensor
    atom_ptr: Tensor
    model_atom_index: Optional[Tensor] = None
    original_atom_index: Optional[Tensor] = None
    rdkit_atom_index: Optional[Tensor] = None


def _resolve_model_atom_index(data: Union[Data, Batch]) -> Tensor:
    if hasattr(data, "model_atom_index") and data.model_atom_index is not None:
        return data.model_atom_index
    if isinstance(data, Batch):
        counts = torch.bincount(data.batch, minlength=int(data.num_graphs))
        parts = [torch.arange(n, device=data.x.device) for n in counts.tolist()]
        return torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=torch.long, device=data.x.device)
    return torch.arange(data.num_nodes, device=data.x.device)


class GINAtomEncoder(nn.Module):
    """E3-compatible GIN backbone that exposes only atom node embeddings."""

    def __init__(
        self,
        *,
        input_dim: int = 78,
        node_hidden_dim: int = 32,
        num_layers: int = 5,
        jk_mode: str = "last",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        # pool_type is required by GINConvNet constructor but must never be used
        # on the XA student forward path (audited separately).
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
        self.num_layers = int(num_layers)

    def forward(self, drug_graph: Union[Data, Batch]) -> AtomEncoderOutput:
        node_embeddings = self.gin.encode_nodes(drug_graph.x, drug_graph.edge_index)
        if isinstance(drug_graph, Batch):
            batch_index = drug_graph.batch
            num_graphs = int(drug_graph.num_graphs)
        else:
            batch_index = torch.zeros(drug_graph.num_nodes, dtype=torch.long, device=drug_graph.x.device)
            num_graphs = 1
        return AtomEncoderOutput(
            node_embeddings=node_embeddings,
            batch_index=batch_index,
            atom_ptr=build_atom_ptr(batch_index, num_graphs),
            model_atom_index=_resolve_model_atom_index(drug_graph),
            original_atom_index=getattr(drug_graph, "original_atom_index", None),
            rdkit_atom_index=getattr(drug_graph, "rdkit_atom_index", None),
        )

    # Explicitly refuse pooling APIs on the XA student encoder.
    def pool_nodes(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("GINAtomEncoder forbids pooling on BioCDA-XA path")

    def get_graph_embedding(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("GINAtomEncoder forbids graph embedding on BioCDA-XA path")
