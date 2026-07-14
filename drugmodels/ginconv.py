import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool, global_mean_pool, JumpingKnowledge, GlobalAttention, Set2Set

# GINConv model with JK (Jumping Knowledge) support
class GINConvNet(torch.nn.Module):
    def __init__(
        self,
        input_dim=78,
        output_dim=None,
        dropout=0.2,
        num_layers=5,
        jk_mode="last",
        use_batch_norm=True,
        pool_type="max",
        *,
        node_hidden_dim=None,
        graph_output_dim=None,
    ):
        """
        Improved GINConvNet with JK (Jumping Knowledge) mechanism.

        Round 19 split:
          - node_hidden_dim: GIN message-passing width (default 32; Round 1–18 legacy)
          - graph_output_dim / output_dim: pooled graph projection width

        Legacy call sites that only pass ``output_dim`` keep node_hidden_dim=32.
        """
        super(GINConvNet, self).__init__()
        if graph_output_dim is None:
            graph_output_dim = 32 if output_dim is None else int(output_dim)
        else:
            graph_output_dim = int(graph_output_dim)
        if output_dim is not None and int(output_dim) != graph_output_dim:
            raise ValueError(
                f"output_dim ({output_dim}) and graph_output_dim ({graph_output_dim}) disagree"
            )
        output_dim = graph_output_dim
        if node_hidden_dim is None:
            node_hidden_dim = 32
        else:
            node_hidden_dim = int(node_hidden_dim)

        self.num_layers = num_layers
        self.jk_mode = jk_mode
        self.use_batch_norm = use_batch_norm
        self.pool_type = pool_type
        self.node_hidden_dim = node_hidden_dim
        self.graph_output_dim = graph_output_dim

        dim = node_hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Validate JK mode
        assert jk_mode in ["last", "cat", "sum", "max"], f"Invalid jk_mode: {jk_mode}"
        assert pool_type in ["add", "max", "mean", "attention", "set2set"], f"Invalid pool_type: {pool_type}"

        # Create GINConv layers dynamically
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer: input_dim -> dim
                nn_block = Sequential(Linear(input_dim, dim), ReLU(), Linear(dim, dim))
            else:
                # Other layers: dim -> dim
                nn_block = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))

            conv = GINConv(nn_block)
            bn = nn.BatchNorm1d(dim) if use_batch_norm else None

            self.convs.append(conv)
            self.bns.append(bn)

        # JK mechanism
        if jk_mode in ["cat", "sum", "max"]:
            if jk_mode == "cat":
                self.jk = JumpingKnowledge("cat")
                # For cat mode, output dimension depends on number of layers
                jk_output_dim = dim * num_layers
            elif jk_mode == "max":
                self.jk = JumpingKnowledge("max")
                jk_output_dim = dim
            elif jk_mode == "sum":
                # PyTorch Geometric JumpingKnowledge doesn't support 'sum', implement manually
                self.jk = None
                self.jk_mode = "sum"  # Keep track for manual implementation
                jk_output_dim = dim
        else:
            self.jk = None
            jk_output_dim = dim

        # Set up pooling mechanism
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
            # Set2Set doubles the output dimension
            jk_output_dim *= 2
        else:
            raise ValueError("Invalid graph pooling type.")

        # Output layers
        self.fc1_xd = Linear(jk_output_dim, output_dim)
        self.out = nn.Linear(output_dim, output_dim)
        self.node_dim = jk_output_dim
        self.output_dim = output_dim

    def encode_nodes(self, x, edge_index):
        """Encode atom nodes through GIN + JK (before pooling)."""
        x_list = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
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
        """Global pool node embeddings to graph-level features (pre-projection)."""
        return self.pool(node_embeddings, batch=batch)

    def project_graph(self, pooled):
        """Apply legacy graph projection head (fc + dropout + out)."""
        x = self.relu(self.fc1_xd(pooled))
        x = self.dropout(x)
        return self.out(x)

    def forward(
        self,
        data,
        pretrain_flag=False,
        return_node_embeddings: bool = False,
        return_graph_embedding: bool = True,
        return_dict: bool = False,
    ):
        """
        Default path preserves Round 1–17 behavior (projected graph embedding).

        When return_node_embeddings=True or return_dict=True, returns a dict with:
          - node_embeddings: [N, node_dim] after JK
          - batch_index: data.batch
          - graph_embedding: projected graph embedding (or None)
          - pooled_raw: pre-projection pooled features (or None)
          - node_dim / graph_dim: int widths (Round 19)
        """
        del pretrain_flag  # retained for call-site compatibility
        want_dict = bool(return_node_embeddings or return_dict)
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        node_embeddings = self.encode_nodes(x, edge_index)

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
