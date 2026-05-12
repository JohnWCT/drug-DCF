
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool


class GATEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.1, pool_ratio=0.5):
        super(GATEncoderBlock, self).__init__()
        # concat=False: averages across heads → output dim stays out_channels (not out_channels * heads)
        # This prevents OOM when heads=4 + dense graphs expand scatter_add_ allocations by 4×.
        self.conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()
        self.pool = SAGPooling(out_channels, ratio=pool_ratio)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None, return_attention=False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if return_attention:
            x, (att_edge_index, att_weights) = self.conv(x, edge_index, return_attention_weights=True)
            attention = (att_edge_index, att_weights)
        else:
            x = self.conv(x, edge_index)
            attention = None

        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)

        # Pooling
        x, edge_index, edge_attr, batch, perm, score = self.pool(x, edge_index, batch=batch)

        return x, edge_index, batch, perm, score, attention


class GNNEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], bottleneck_dim=32,
                 heads=1, dropout=0.1, pool_ratios=[0.5, 0.5]):
        """
        Pure AE Encoder (no VAE reparameterization).

        Architecture:
          Input node features (num_nodes × 1)
          → Stack of GATEncoderBlock (GAT conv + BN + ELU + SAGPool)
          → Global mean pooling  →  graph-level vector  [Batch, last_hidden]  (avg over heads)
          → FC projection        →  latent z            [Batch, bottleneck_dim]

        The latent z is a deterministic, flat vector – a concatenation of
        pooled node information across GNN layers.
        """
        super(GNNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            block = GATEncoderBlock(current_dim, h_dim, heads=heads,
                                    dropout=dropout, pool_ratio=pool_ratios[i])
            self.layers.append(block)
            # concat=False → output dim = h_dim (not h_dim * heads)
            current_dim = h_dim

        # Single projection to bottleneck (no mu/logvar split)
        self.fc_z = nn.Linear(current_dim, bottleneck_dim)

    def forward(self, x, edge_index, batch=None, return_attention=False):
        perms = []
        attentions = []

        for layer in self.layers:
            x, edge_index, batch, perm, _, attention = layer(
                x, edge_index, batch, return_attention=return_attention)
            perms.append((perm, edge_index))
            if attention is not None:
                attentions.append(attention)

        # Global mean pooling → graph-level representation
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        g_x = global_mean_pool(x, batch)   # [Batch, current_dim]
        z = self.fc_z(g_x)                 # [Batch, bottleneck_dim]

        return z, perms, batch, attentions


class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, num_nodes, output_dim=1,
                 hidden_dims=[32, 64], heads=1, dropout=0.1):
        super(GNNDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim

        # Learnable query embedding for each gene node
        self.query_embed = nn.Parameter(torch.randn(num_nodes, latent_dim))

        # MLP decoder: (query_embed ‖ global_z) → hidden layers → output_dim per node
        # Using MLP avoids the B×E edge explosion that caused OOM in the GATConv version.
        layers = []
        in_dim = latent_dim * 2   # query + z concatenated
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, edge_index, batch_size):
        """
        z          : [BatchSize, LatentDim]
        edge_index : [2, NumEdges]  (kept for API compatibility; unused by MLP decoder)
        batch_size : int

        Memory: O(B * N * D) — no edge explosion.
        """
        N = self.num_nodes

        # Expand z to all nodes: [B, D] → [B*N, D]
        z_expanded = z.unsqueeze(1).expand(-1, N, -1).reshape(-1, self.latent_dim)

        # Expand learnable queries: [N, D] → [B*N, D]
        q_expanded = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.latent_dim)

        # Concatenate query + global z → MLP → per-node reconstruction
        dec_in = torch.cat([q_expanded, z_expanded], dim=1)  # [B*N, 2D]
        x_rec = self.mlp(dec_in)                              # [B*N, output_dim]
        return x_rec



class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=[64, 32]):
        super(Discriminator, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        # No sigmoid – use Wasserstein / BCEWithLogits loss
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=[64, 32]):
        super(Classifier, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GNNAutoencoder(nn.Module):
    """
    Pure Graph Autoencoder (AE, not VAE).

    Encoder: GNN → global mean pool → FC → deterministic latent z
    Latent:  shared z  (domain-invariant)  +  private z  (domain-specific)
             concatenated before decoding
    Decoder: two domain-specific MLP decoders (source_decoder / target_decoder)
             each receives combined z = [shared_z ‖ private_z] → per-node reconstruction

    forward() returns:
        recon_x  : reconstructed node features  [B*N, 1]
        s_z      : shared  latent               [B, latent_dim]
        p_z      : private latent               [B, latent_dim]

    Note: mu / logvar / reparameterization have been removed entirely.
    """

    def __init__(self, num_nodes, input_dim=1, latent_dim=32,
                 gnn_hidden_dims=[64, 64], gnn_heads=2, gnn_dropout=0.1,
                 gnn_pool_ratios=[0.5, 0.5], decoder_hidden_dims=[64, 64],
                 device='cuda'):
        super(GNNAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.device = device

        # Three encoders: 1 shared + 2 private (source / target)
        self.shared_encoder = GNNEncoder(
            input_dim, gnn_hidden_dims, latent_dim,
            heads=gnn_heads, dropout=gnn_dropout, pool_ratios=gnn_pool_ratios)
        self.private_source_encoder = GNNEncoder(
            input_dim, gnn_hidden_dims, latent_dim,
            heads=gnn_heads, dropout=gnn_dropout, pool_ratios=gnn_pool_ratios)
        self.private_target_encoder = GNNEncoder(
            input_dim, gnn_hidden_dims, latent_dim,
            heads=gnn_heads, dropout=gnn_dropout, pool_ratios=gnn_pool_ratios)

        # Domain-specific decoders: shared(latent_dim) ⊕ private(latent_dim) = 2*latent_dim
        self.source_decoder = GNNDecoder(
            latent_dim * 2, num_nodes, output_dim=1,
            hidden_dims=decoder_hidden_dims, heads=gnn_heads, dropout=gnn_dropout)
        self.target_decoder = GNNDecoder(
            latent_dim * 2, num_nodes, output_dim=1,
            hidden_dims=decoder_hidden_dims, heads=gnn_heads, dropout=gnn_dropout)

    def forward(self, x, edge_index, batch, domain='source'):
        """
        Returns: recon_x, s_z, p_z
          recon_x : [B*N, 1]
          s_z     : [B, latent_dim]  shared latent
          p_z     : [B, latent_dim]  private latent
        """
        # Shared encoding
        s_z, _, _, _ = self.shared_encoder(x, edge_index, batch)

        # Private encoding
        if domain == 'source':
            p_z, _, _, _ = self.private_source_encoder(x, edge_index, batch)
        else:
            p_z, _, _, _ = self.private_target_encoder(x, edge_index, batch)

        # Concatenate latents and decode with domain-specific decoder
        combined_z = torch.cat([s_z, p_z], dim=1)  # [B, 2*latent_dim]
        batch_size = s_z.size(0)
        if domain == 'source':
            recon_x = self.source_decoder(combined_z, edge_index, batch_size)
        else:
            recon_x = self.target_decoder(combined_z, edge_index, batch_size)

        return recon_x, s_z, p_z

    def get_attention_weights(self, x, edge_index, batch):
        """
        Extracts attention weights from the Shared Encoder.
        Returns: list of (edge_index, attention_weights) per GAT layer.
        """
        _, _, _, attentions = self.shared_encoder(x, edge_index, batch, return_attention=True)
        return attentions
