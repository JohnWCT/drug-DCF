import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool, global_mean_pool, JumpingKnowledge, GlobalAttention, Set2Set

# GINConv model with JK (Jumping Knowledge) support
class GINConvNet(torch.nn.Module):
    def __init__(self, input_dim=78, output_dim=32, dropout=0.2, 
                 num_layers=5, jk_mode='last', use_batch_norm=True, pool_type='max'):
        """
        Improved GINConvNet with JK (Jumping Knowledge) mechanism
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            dropout: Dropout rate
            num_layers: Number of GINConv layers
            jk_mode: JK mode - 'last', 'cat', 'sum', 'max'
            use_batch_norm: Whether to use BatchNorm
            pool_type: Global pooling type - 'add', 'max', 'mean', 'attention', 'set2set'
        """
        super(GINConvNet, self).__init__()
        self.num_layers = num_layers
        self.jk_mode = jk_mode
        self.use_batch_norm = use_batch_norm
        self.pool_type = pool_type
        
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Validate JK mode
        assert jk_mode in ['last', 'cat', 'sum', 'max'], f"Invalid jk_mode: {jk_mode}"
        assert pool_type in ['add', 'max', 'mean', 'attention', 'set2set'], f"Invalid pool_type: {pool_type}"
        
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
        if jk_mode in ['cat', 'sum', 'max']:
            if jk_mode == 'cat':
                self.jk = JumpingKnowledge('cat')
                # For cat mode, output dimension depends on number of layers
                jk_output_dim = dim * num_layers
            elif jk_mode == 'max':
                self.jk = JumpingKnowledge('max')
                jk_output_dim = dim
            elif jk_mode == 'sum':
                # PyTorch Geometric JumpingKnowledge doesn't support 'sum', implement manually
                self.jk = None
                self.jk_mode = 'sum'  # Keep track for manual implementation
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
    ):
        """
        Default path preserves Round 1–17 behavior (projected graph embedding).

        When return_node_embeddings=True, returns a dict with:
          - node_embeddings: [N, node_dim] after JK
          - batch_index: data.batch
          - graph_embedding: projected graph embedding (or None)
          - pooled_raw: pre-projection pooled features (or None)
        """
        del pretrain_flag  # retained for call-site compatibility
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        node_embeddings = self.encode_nodes(x, edge_index)

        pooled_raw = None
        graph_embedding = None
        if return_graph_embedding or not return_node_embeddings:
            pooled_raw = self.pool_graph(node_embeddings, batch=batch)
            graph_embedding = self.project_graph(pooled_raw)

        if not return_node_embeddings:
            return graph_embedding

        return {
            "node_embeddings": node_embeddings,
            "batch_index": batch,
            "graph_embedding": graph_embedding if return_graph_embedding else None,
            "pooled_raw": pooled_raw if return_graph_embedding else None,
        }

'''
DRpreter
gin_model = GINConvNet(
    input_dim=77,      # 對應 DrugEncoder 中的 77
    output_dim=32,     # 對應 dim_drug=32
    num_layers=5,      # 對應 layer_drug=5
    jk_mode='cat',     # 對應 JumpingKnowledge('cat')
    use_batch_norm=True,
    pool_type='max'    # 對應 global_max_pool
)


GINpre
gin_model = GINConvNet(
    input_dim=77,      # 根據您的輸入特徵維度
    output_dim=32,     # 對應 emb_dim=32
    num_layers=5,      # 對應 num_layer=5
    jk_mode='sum',     # 對應 JK="sum"
    use_batch_norm=False, # GNN_graphpred 沒有 BatchNorm
    pool_type='mean',  # 對應 graph_pooling="mean"
    dropout=0.0        # 對應 drop_ratio=0.0
)
'''