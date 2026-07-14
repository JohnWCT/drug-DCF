"""GINE drug encoder with bond edge attributes (Round 19 D3)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import (
    GINEConv,
    GlobalAttention,
    JumpingKnowledge,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class GINEConvNet(torch.nn.Module):
    """Bond-aware GIN variant using GINEConv + edge_attr."""

    def __init__(
        self,
        input_dim: int = 78,
        edge_dim: int = 10,
        node_hidden_dim: int = 64,
        graph_output_dim: int = 64,
        dropout: float = 0.1,
        num_layers: int = 5,
        jk_mode: str = "last",
        use_batch_norm: bool = True,
        pool_type: str = "max",
        *,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        if output_dim is not None:
            graph_output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.jk_mode = jk_mode
        self.use_batch_norm = bool(use_batch_norm)
        self.pool_type = pool_type
        self.node_hidden_dim = int(node_hidden_dim)
        self.graph_output_dim = int(graph_output_dim)
        self.edge_dim = int(edge_dim)

        dim = self.node_hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        assert jk_mode in ["last", "cat", "sum", "max"], f"Invalid jk_mode: {jk_mode}"
        assert pool_type in ["add", "max", "mean", "attention", "set2set"], f"Invalid pool_type: {pool_type}"

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = input_dim if i == 0 else dim
            nn_block = Sequential(Linear(in_dim, dim), ReLU(), Linear(dim, dim))
            self.convs.append(GINEConv(nn_block, edge_dim=self.edge_dim))
            self.bns.append(nn.BatchNorm1d(dim) if use_batch_norm else None)

        if jk_mode == "cat":
            self.jk = JumpingKnowledge("cat")
            jk_output_dim = dim * self.num_layers
        elif jk_mode == "max":
            self.jk = JumpingKnowledge("max")
            jk_output_dim = dim
        elif jk_mode == "sum":
            self.jk = None
            jk_output_dim = dim
        else:
            self.jk = None
            jk_output_dim = dim

        if pool_type == "add":
            self.pool = global_add_pool
        elif pool_type == "max":
            self.pool = global_max_pool
        elif pool_type == "mean":
            self.pool = global_mean_pool
        elif pool_type == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(jk_output_dim, 1))
        elif pool_type == "set2set":
            self.pool = Set2Set(jk_output_dim, processing_steps=3)
            jk_output_dim *= 2
        else:
            raise ValueError("Invalid graph pooling type.")

        self.fc1_xd = Linear(jk_output_dim, self.graph_output_dim)
        self.out = nn.Linear(self.graph_output_dim, self.graph_output_dim)
        self.node_dim = jk_output_dim
        self.output_dim = self.graph_output_dim

    def encode_nodes(self, x, edge_index, edge_attr):
        if edge_attr is None:
            raise ValueError("GINEConvNet requires edge_attr")
        x_list = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.relu(x)
            if self.use_batch_norm and self.bns[i] is not None:
                x = self.bns[i](x)
            x_list.append(x)
        if self.jk is not None and self.jk_mode != "last":
            return self.jk(x_list)
        if self.jk_mode == "sum":
            return torch.stack(x_list, dim=0).sum(dim=0)
        return x_list[-1]

    def pool_graph(self, node_embeddings, batch=None):
        return self.pool(node_embeddings, batch=batch)

    def project_graph(self, pooled):
        x = self.relu(self.fc1_xd(pooled))
        x = self.dropout(x)
        return self.out(x)

    def forward(
        self,
        data,
        *,
        return_node_embeddings: bool = False,
        return_graph_embedding: bool = True,
        return_dict: bool = False,
    ):
        want_dict = bool(return_node_embeddings or return_dict)
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)

        node_embeddings = self.encode_nodes(x, edge_index, edge_attr)
        pooled_raw = None
        graph_embedding = None
        if return_graph_embedding or not want_dict:
            pooled_raw = self.pool_graph(node_embeddings, batch=batch)
            graph_embedding = self.project_graph(pooled_raw)

        if not want_dict:
            return graph_embedding

        return {
            "node_embeddings": node_embeddings,
            "batch_index": batch,
            "graph_embedding": graph_embedding if return_graph_embedding else None,
            "pooled_raw": pooled_raw if return_graph_embedding else None,
            "node_dim": int(node_embeddings.shape[-1]),
            "graph_dim": int(graph_embedding.shape[-1]) if graph_embedding is not None else int(self.output_dim),
        }
