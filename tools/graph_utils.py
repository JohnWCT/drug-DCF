
import os
import torch
import pandas as pd
import numpy as np
import torch_geometric.data as DATA
from torch_geometric.loader import DataLoader

class PPIEdgeProcessor:
    """
    Handles PPI edge construction and processing for GNN models.
    Supports different graph construction modes (A, B, C).
    """
    def __init__(self, ppi_df: pd.DataFrame, gene_list: list, 
                 mode: str = 'A', 
                 threshold: float = 0.9, 
                 low_score: float = 0.1, 
                 complete_graph: bool = False,
                 self_loop: bool = True,
                 col_map: dict = None):
        """
        Args:
            ppi_df: DataFrame with PPI interactions.
            gene_list: List of gene names (universe) to align with.
            mode: Graph construction mode.
                  - 'A' (Binary Threshold): Edges with score > threshold are kept with weight 1.0. All others discarded.
                  - 'B' (Raw + Missing/Low): All existing edges keep their raw PPI score. If complete_graph=True, missing edges are added with `low_score`.
                  - 'C' (Threshold + Low): Existing edges > threshold keep raw score. Existing edges <= threshold are set to `low_score`.
            threshold: Threshold for Mode C and A (default 0.99 for A).
            low_score: Background score for weak/missing edges (used in Mode B/C).
            complete_graph: Whether to force a fully connected graph by filling missing edges with `low_score`.
            self_loop: Whether to add self-loops (weight=1.0) for every node.
            col_map: Dictionary mapping 'GeneA', 'GeneB', 'PPI_score' to actual columns. 
                     Default: {'GeneA': 'GeneA', 'GeneB': 'GeneB', 'PPI_score': 'PPI_score'}
        """
        self.ppi_df = ppi_df
        self.gene_list = gene_list
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
        self.num_nodes = len(gene_list)
        self.mode = mode
        self.threshold = threshold
        self.low_score = low_score
        self.complete_graph = complete_graph
        self.self_loop = self_loop
        
        self.col_map = {'GeneA': 'GeneA', 'GeneB': 'GeneB', 'PPI_score': 'PPI_score'}
        if col_map:
            self.col_map.update(col_map)
        
        # Check columns
        for key in ['GeneA', 'GeneB', 'PPI_score']:
            col_name = self.col_map[key]
            if col_name not in ppi_df.columns:
                raise ValueError(f"PPI DataFrame missing column: {col_name} (mapped from {key})")

    def process(self):
        """
        Main processing function to generate edge_index and edge_attr.
        Returns:
            edge_index: torch.LongTensor [2, num_edges]
            edge_attr: torch.FloatTensor [num_edges, 1]
        """
        print(f"Processing PPI Graph Mode: {self.mode}")
        
        gene_a_col = self.col_map['GeneA']
        gene_b_col = self.col_map['GeneB']
        score_col = self.col_map['PPI_score']
        
        # filter genes that are in our universe
        valid_mask = (self.ppi_df[gene_a_col].isin(self.gene_list)) & \
                     (self.ppi_df[gene_b_col].isin(self.gene_list))
        filtered_ppi = self.ppi_df[valid_mask].copy()
        
        # Map gene names to indices
        filtered_ppi['src'] = filtered_ppi[gene_a_col].map(self.gene_to_idx)
        filtered_ppi['dst'] = filtered_ppi[gene_b_col].map(self.gene_to_idx)
        
        filtered_ppi['u'] = filtered_ppi[['src', 'dst']].min(axis=1)
        filtered_ppi['v'] = filtered_ppi[['src', 'dst']].max(axis=1)
        
        # Valid scores
        scores = filtered_ppi.groupby(['u', 'v'])[score_col].max().reset_index()
        scores.rename(columns={score_col: 'PPI_score'}, inplace=True)
        results = []
        
        if self.mode == 'A':
            # Mode A: High Confidence Binary Network
            # - Filter edges with score > threshold (default 0.99)
            # - Treat as unweighted binary edges (weight = 1.0)
            # - Suitable for clean, sparse graphs where only high-certainty interactions matter.
            threshold = 0.99
            mask = scores['PPI_score'] > threshold
            edges = scores[mask][['u', 'v']].values
            weights = np.ones(len(edges)) # Binary weight = 1
            
            results = list(zip(edges[:,0], edges[:,1], weights))
            
        elif self.mode == 'B':
            # Mode B: Raw Scores with Optional Completion
            # - Keep ALL existing edges with their original continuous scores.
            # - If complete_graph=True: Add missing edges with `low_score` to model weak/unknown interactions.
            # - Suitable for utilizing full interaction spectrum.
            
            existing_edges = set(zip(scores['u'], scores['v'])) # optimization for lookup
            existing_weights = dict(zip(zip(scores['u'], scores['v']), scores['PPI_score']))
            
            if self.complete_graph:
                # Generate all pairs
                all_u, all_v = np.triu_indices(self.num_nodes, k=1)
                # This loop is slow in python... vectorization preferred
                # Vectorized approach:
                # Initialize weights matrix
                W = np.full((self.num_nodes, self.num_nodes), self.low_score)
                # Fill existing
                for u, v, w in zip(scores['u'], scores['v'], scores['PPI_score']):
                    W[int(u), int(v)] = w
                
                # Extract upper triangle
                u_idx, v_idx = np.triu_indices(self.num_nodes, k=1)
                w_vals = W[u_idx, v_idx]
                results = list(zip(u_idx, v_idx, w_vals))
                
            else:
                # Only existing 
                for u, v, w in zip(scores['u'], scores['v'], scores['PPI_score']):
                    results.append((u, v, w))
                    
        elif self.mode == 'C':
            # Mode C: Thresholded Continuous Network
            # - Existing edges > threshold: Keep original raw score.
            # - Existing edges <= threshold: Downgrade to `low_score`.
            # - If complete_graph=True: Add missing edges with `low_score`.
            # - Acts as a soft filter: preserves strong signals while suppressing noise without removing it entirely.
            existing_weights = dict(zip(zip(scores['u'], scores['v']), scores['PPI_score']))
            
            if self.complete_graph:
                 # Vectorized approach:
                W = np.full((self.num_nodes, self.num_nodes), self.low_score)
                for u, v, w in zip(scores['u'], scores['v'], scores['PPI_score']):
                    if w > self.threshold:
                         W[int(u), int(v)] = w
                    # else it remains low_score
                
                u_idx, v_idx = np.triu_indices(self.num_nodes, k=1)
                w_vals = W[u_idx, v_idx]
                results = list(zip(u_idx, v_idx, w_vals))

            else:
                for u, v, raw_w in zip(scores['u'], scores['v'], scores['PPI_score']):
                    if raw_w > self.threshold:
                        results.append((u, v, raw_w))
                    else:
                        results.append((u, v, self.low_score))
        
        # Convert to tensor
        if not results:
            print("Warning: No edges found. Creating empty graph.")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            results_array = np.array(results)
            u_list = results_array[:, 0].astype(int)
            v_list = results_array[:, 1].astype(int)
            w_list = results_array[:, 2].astype(float)
            
            # Make symmetric (Undirected)
            # Add (v, u) for every (u, v)
            src = np.concatenate([u_list, v_list])
            dst = np.concatenate([v_list, u_list])
            weights = np.concatenate([w_list, w_list])
            
            if self.self_loop:
                # Add self loops
                loops = np.arange(self.num_nodes)
                src = np.concatenate([src, loops])
                dst = np.concatenate([dst, loops])
                # Self loop weight = 1.0
                weights = np.concatenate([weights, np.ones(self.num_nodes)])
                
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(weights, dtype=torch.float).view(-1, 1)
            
        return edge_index, edge_attr

    def save_edge_artifact(self, output_path: str, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """Saves generated edge_index/attr to disk"""
        torch.save({
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'gene_list': self.gene_list,
            'config': {
                'mode': self.mode,
                'threshold': self.threshold,
                'low_score': self.low_score,
                'complete': self.complete_graph
            }
        }, output_path)
        print(f"Edge artifact saved to {output_path}")

class GraphDataset(DATA.Dataset):
    def __init__(self, x_data, y_data, edge_index, edge_attr, root=None, transform=None, pre_transform=None):
        """
        Args:
            x_data: Gene expression tensor [num_samples, num_genes]
            y_data: Labels tensor [num_samples]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge weights [num_edges, 1]
        """
        self.x_data = x_data
        self.y_data = y_data
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    def len(self):
        return len(self.x_data)

    def get(self, idx):
        # x shape: [num_genes, 1]
        x = self.x_data[idx].view(-1, 1)
        if self.y_data is not None:
             y = self.y_data[idx]
             data = DATA.Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y)
        else:
             data = DATA.Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)
        return data

def create_graph_loader(x_tensor, y_tensor, edge_index, edge_attr, batch_size=64, shuffle=True):
    """Factory function for DataLoader"""
    dataset = GraphDataset(x_tensor, y_tensor, edge_index, edge_attr)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
