import argparse
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import glob
import ast
import itertools
import csv
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# Import model components and utils
from drugmodels.ginconv import GINConvNet
from tools.drug_finetune_utils import load_tcga_data_deduplicated, process_drug_data, DrugResponseDataset
from tools.inference_utils import inference_on_tcga_drugs, calculate_comprehensive_metrics, calculate_overall_tcga_metrics
from tools.model_opt import Classify, init_weights, VAE, MLP
import Export_latent2dict  # Import reuseable logic
# GNN AE Imports
from drugmodels.gnn_ae_model import GNNAutoencoder
from tools.graph_utils import PPIEdgeProcessor
import matplotlib.pyplot as plt



def plot_finetune_curves(train_losses, val_losses, val_aucs, save_path):
    """Plot training and validation curves"""
    try:
        plt.figure(figsize=(10, 5))
        
        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # AUC Plot
        plt.subplot(1, 2, 2)
        plt.plot(val_aucs, label='Val AUC', color='orange')
        plt.title('Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error plotting curves: {e}")

# --- Execution Example ---
"""
Example Execution:

python finetune_GNN_AE.py \
    --config config/finetune_params_grid.json \
    --model_select_path result/pretrain_VAE/model_select.csv \
    --drug_response_path "data/GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID678 drug230 samples142188 balanced_high.csv" \
    --drug_smiles_path data/223drugs_pubchem_smiles.csv \
    --tcga_data_path "data/TCGA" \
    --mode 5fold \
    --outfolder result/finetune_selected
"""

# --- Helper Functions ---

def construct_folder_name(row):
    """
    Construct model folder name from model_select.csv row.
    Supports both VAE format and GNN AE exp_X format.
    """
    # Check if ID is in exp_X format
    if 'ID' in row and str(row['ID']).startswith('exp_'):
        return str(row['ID'])
        
    # Legacy VAE format
    try:
        pt_epochs = row['pretrain_epochs']
        t_epochs = row['train_epochs']
        pt_lr = row['pretrain_lr']
        t_lr = row['train_lr']
        dop = row['dropout']
        
        # Parse encoder dims
        enc_dims_str = row['encoder_dims']
        try:
            enc_dims = ast.literal_eval(enc_dims_str)
            if enc_dims == [128]:
                enc_suffix = 1
            elif enc_dims == [512, 256, 128, 64]:
                enc_suffix = 4
            else:
                enc_suffix = len(enc_dims)
        except:
            # Fallback if simple parsing fails
            if '128' in enc_dims_str and '512' not in enc_dims_str:
                enc_suffix = 1
            else:
                enc_suffix = 4
                
        folder_name = f"pt_epochs_{pt_epochs},t_epochs_{t_epochs},Ptlr_{pt_lr},tlr{t_lr},dop{dop},enc{enc_suffix}"
        return folder_name
    except Exception as e:
        # Fallback: just return ID
        if 'ID' in row:
            return str(row['ID'])
        raise e

def generate_latent_if_missing(model_folder, model_type, data_path, device, encoder_dims, dropout_rate=0.0, row_config=None, ppi_path=None):
    """
    Check if latent_dict.pkl exists in model_folder. If not, generate it.
    Supports VAE, MLP, and GNN_AE.
    """
    latent_path = os.path.join(model_folder, 'latent_dict.pkl')
    
    if os.path.exists(latent_path):
        print(f"Index latent representation found: {latent_path}")
        return latent_path
        
    print(f"Latent representation not found. Generating at {latent_path}...")
    
    # Locate model file
    # GNN AE priority
    model_file_path = os.path.join(model_folder, 'model_best_fid.pth')
    if not os.path.exists(model_file_path):
        model_file_path = os.path.join(model_folder, 'model_last.pth')
    
    if not os.path.exists(model_file_path):
        # VAE/MLP priority
        model_file_path = os.path.join(model_folder, 'after_traingan_shared_vae.pth')
    
    if not os.path.exists(model_file_path):
        # Fallback
        pth_files = glob.glob(os.path.join(model_folder, '*.pth'))
        if pth_files:
            model_file_path = pth_files[0]
        else:
            raise FileNotFoundError(f"No model file found in {model_folder}")
            
    print(f"Using model file: {model_file_path}")
    
    # Load Data
    try:
        df = pd.read_csv(data_path, index_col=0, header=0)
    except Exception as e:
        print(f"Error loading pretrain data from {data_path}: {e}")
        raise
        
    input_dim = df.shape[1]
    latent_size = 32 # Default
    if row_config is not None and 'latent_dim' in row_config:
        try:
            latent_size = int(row_config['latent_dim'])
        except:
            pass
    
    # Init Model
    if model_type == 'VAE':
        encoder_hidden_dims = encoder_dims
        decoder_hidden_dims = encoder_hidden_dims[::-1]
        model = VAE(input_size=input_dim, output_size=input_dim, latent_size=latent_size,
                    encoder_hidden_dims=encoder_hidden_dims, decoder_hidden_dims=decoder_hidden_dims,
                    dop=dropout_rate, act_fn=nn.ReLU).to(device)
    elif model_type == 'GNN_AE':
        # GNN AE Initialization
        if ppi_path is None:
            # Try default if missing
            ppi_path = 'data/PPI/9606.protein.links.v11.5.txt'
            print(f"Warning: ppi_path not provided, using default {ppi_path}")
            
        print("Initializing GNN Autoencoder for inference...")
        
        # Load PPI Graph
        ppi_processor = PPIEdgeProcessor(ppi_path)
        # Note: gene_list must match the columns of df
        gene_list = list(df.columns)
        num_nodes = len(gene_list)
        
        edge_index = ppi_processor.process_edge_index(gene_list).to(device)
        print(f"PPI Graph loaded: {edge_index.shape[1]} edges for {num_nodes} nodes")
        
        # Parse params from row_config
        gnn_hidden = [64, 64]
        if row_config is not None and 'gnn_hidden_dims' in row_config:
            try:
                gnn_hidden = ast.literal_eval(row_config['gnn_hidden_dims'])
            except:
                pass
                
        gnn_heads = 2
        if row_config is not None and 'gnn_heads' in row_config:
            gnn_heads = int(row_config['gnn_heads'])
            
        gnn_dropout = 0.1
        if row_config is not None and 'gnn_dropout' in row_config:
            gnn_dropout = float(row_config['gnn_dropout'])
            
        gnn_pool_ratios = [0.5, 0.5]
        if row_config is not None and 'gnn_pool_ratios' in row_config:
            try:
                gnn_pool_ratios = ast.literal_eval(row_config['gnn_pool_ratios'])
            except:
                pass
        
        model = GNNAutoencoder(
            num_nodes=num_nodes,
            input_dim=1, # GNN AE takes [Nodes, 1] per sample
            latent_dim=latent_size,
            gnn_hidden_dims=gnn_hidden,
            gnn_heads=gnn_heads,
            gnn_dropout=gnn_dropout,
            gnn_pool_ratios=gnn_pool_ratios,
            decoder_hidden_dims=[64, 64], # Default/Not used for encoding
            device=device
        ).to(device)
    else:
        model = MLP(input_dim=input_dim, output_dim=latent_size, hidden_dims=encoder_dims,
                    dop=dropout_rate, act_fn=nn.ReLU, out_fn=None, gr_flag=False).to(device)
                    
    # Load Weights
    checkpoint = torch.load(model_file_path, map_location=device)
    
    # Handle GNN AE state dict keys if needed
    if model_type == 'GNN_AE':
        # Check if keys match
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(checkpoint.keys())
        if not model_keys.intersection(ckpt_keys):
             # Try loading 'model_state_dict' if nested
             if 'model_state_dict' in checkpoint:
                 checkpoint = checkpoint['model_state_dict']
    
    try:
        model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(f"Warning: strict load failed ({e}), trying strict=False")
        model.load_state_dict(checkpoint, strict=False)
        
    model.eval()
    
    # Generate Latent
    latent_dict = {}
    with torch.no_grad():
        if model_type == 'GNN_AE':
             # Process row by row
             # GNN input: x [NumNodes, 1]
             # edge_index is fixed
             batch_vec = torch.zeros(num_nodes, dtype=torch.long, device=device) # Batch = 0 for single sample
             
             for i, (sample_id, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Generating GNN Latents")):
                 x = torch.FloatTensor(row.values).to(device).unsqueeze(1) # [Nodes, 1]
                 
                 # Shared Encoder only: s_z
                 # Forward returns: recon, s_mu, s_logvar, p_mu, p_logvar, s_z, p_z
                 # We want s_z (shared latent) or s_mu
                 # PREFER s_mu for deterministic embedding
                 
                 # Access shared encoder directly
                 # shared_encoder forward returns: z, mu, logvar, perms, batch, attentions
                 # We need to pass batch vector if global pooling relies on it
                 
                 s_z, s_mu, s_logvar, _, _, _ = model.shared_encoder(x, edge_index, batch_vec)
                 
                 # s_mu is [1, Latent]
                 latent_dict[sample_id] = s_mu.flatten().cpu().numpy().tolist()
                 
        else:
            for i, (sample_id, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Generating Latents")):
                features = torch.FloatTensor(row.values).to(device).unsqueeze(0)
                if model_type == 'VAE':
                    _, z, _, _ = model(features)
                else:
                    z = model(features)
                latent_dict[sample_id] = z.squeeze(0).cpu().numpy().tolist()
            
    # Deduplicate TCGA if needed (using utility logic)
    if any(Export_latent2dict.is_tcga_sample(sid) for sid in latent_dict):
        latent_dict = Export_latent2dict.deduplicate_tcga_latent_dict(latent_dict)
        
    # Save
    Export_latent2dict.save_to_pickle(latent_dict, latent_path)
    print(f"Saved generated latent dict to {latent_path}")
    
    return latent_path
    
# --- Model Definition ---
class LatentDrugResponseModel(nn.Module):
    def __init__(self, drug_model, classifier):
        super().__init__()
        self.drug_model = drug_model
        self.classifier = classifier
        
    def forward(self, gene_latent, drug_data):
        # gene_latent: [Batch, LatentDim]
        
        # Drug Encoder
        if self.drug_model is not None:
             # Assuming drug_data is a Batch object if using graph
             z_drug = self.drug_model(drug_data)
        else:
             # Assuming drug_data is already latent
             z_drug = drug_data
             
        # Combined
        combined = torch.cat([gene_latent, z_drug], dim=1)
        
        return self.classifier(combined)

# --- Training / Eval Loops ---
def train_epoch(model, loader, optimizer, criterion, device, gin_type='precomputed'):
    model.train()
    total_loss = 0
    preds = []
    labels = []
    
    for gene_latent, drug_data, label in runner:
        gene_latent = gene_latent.to(device)
        label = label.to(device)
        
        if gin_type == 'precomputed':
            drug_data = drug_data.to(device)
        else:
            drug_data = drug_data.to(device)
            
        optimizer.zero_grad()
        out = model(gene_latent, drug_data).squeeze()
        
        if out.ndim == 0:
            out = out.unsqueeze(0)
            
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * label.size(0)
        preds.extend(torch.sigmoid(out).detach().cpu().numpy())
        labels.extend(label.cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)
    except:
        auc, auprc = 0.5, 0.0
    return avg_loss, auc, auprc

# Helper to use tqdm or not 
runner = None # will be set in loop or just use tqdm in train/eval

def train_epoch_ops(model, loader, optimizer, criterion, device, gin_type='precomputed'):
    model.train()
    total_loss = 0
    preds = []
    labels = []
    
    for gene_latent, drug_data, label in loader: # No tqdm to reduce clutter in grid search
        gene_latent = gene_latent.to(device)
        label = label.to(device)
        
        if gin_type == 'precomputed':
            drug_data = drug_data.to(device)
        else:
            drug_data = drug_data.to(device)
            
        optimizer.zero_grad()
        out = model(gene_latent, drug_data).squeeze()
        
        if out.ndim == 0: out = out.unsqueeze(0)
            
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * label.size(0)
        
        with torch.no_grad():
             preds.extend(torch.sigmoid(out).detach().cpu().numpy())
             labels.extend(label.cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)
    except:
        auc, auprc = 0.5, 0.0
    return avg_loss, auc, auprc

def evaluate_ops(model, loader, criterion, device, gin_type='precomputed'):
    model.eval()
    total_loss = 0
    preds = []
    labels = []
    
    with torch.no_grad():
        for gene_latent, drug_data, label in loader:
            gene_latent = gene_latent.to(device)
            label = label.to(device)
            
            if gin_type == 'precomputed':
                drug_data = drug_data.to(device)
            else:
                drug_data = drug_data.to(device)
            
            out = model(gene_latent, drug_data).squeeze()
            if out.ndim == 0: out = out.unsqueeze(0)
                
            loss = criterion(out, label)
            
            total_loss += loss.item() * label.size(0)
            preds.extend(torch.sigmoid(out).cpu().numpy())
            labels.extend(label.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)
    except:
        auc, auprc = 0.5, 0.0
    return avg_loss, auc, auprc

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Finetune GNN AE - Grid Search")
    parser.add_argument('--config', type=str, required=True, help='Path to finetune_params_grid.json')
    parser.add_argument('--model_select_path', type=str, required=True, help='Path to model_select.csv')
    parser.add_argument('--drug_response_path', type=str, required=True, help='Path to response CSV')
    parser.add_argument('--drug_smiles_path', type=str, default='data/223drugs_pubchem_smiles.csv', help='Path to SMILES CSV')
    parser.add_argument('--tcga_data_path', type=str, default='data/TCGA', help='Path to TCGA data root folder')
    parser.add_argument('--pretrain_data_path', type=str, default='data/pretrain_ccle.csv', help='Path to pretrain data for latent generation')
    parser.add_argument('--tcga_latent_path', type=str, default='input/VAE_latent/tcga_latent_dict.pkl', help='Path to TCGA expression latent pickle file')
    
    parser.add_argument('--drug_latent_path', type=str, default='input/GIN_latent/latent_dict_supervised_masking_graph_GDSC+214drugs.pkl', help='Path to drug latent pickle file')
    
    parser.add_argument('--ppi_path', type=str, default='data/PPI/9606.protein.links.v11.5.txt', help='Path to PPI network file')
    
    parser.add_argument('--mode', type=str, choices=['1split', '5fold'], default='1split')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--outfolder', type=str, default='result/finetune_selected')
    
    args = parser.parse_args()
    print(f"Using device: {args.device}")
    os.makedirs(args.outfolder, exist_ok=True)
    
    # 1. Load Config
    with open(args.config, 'r') as f:
        config_grid = json.load(f)
    print("Loaded config grid.")
    
    # 2. Prepare Parameter Grid
    # We want to iterate over all combinations of parameters in the lists
    # Flatten the config to a list of (section, key, values)
    param_keys = []
    param_values = []
    
    sections = ['finetune_params', 'classifier_params', 'model_params', 'training_params', 'gnn_params']
    
    for sec in sections:
        if sec in config_grid:
            for k, v in config_grid[sec].items():
                if isinstance(v, list):
                    param_keys.append((sec, k))
                    param_values.append(v)
                else:
                    # If single value, make it a list so it works with product
                    param_keys.append((sec, k))
                    param_values.append([v])
                    
    # Generate all combinations
    grid_product = list(itertools.product(*param_values))
    print(f"Total parameter combinations found: {len(grid_product)}")
    
    # 3. Load Static Data
    drug_latent_dict = None
    drug_graph_dict = None
    drug_smiles_df = None
    
    if os.path.exists(args.drug_latent_path):
        print(f"Loading precomputed drug latents from {args.drug_latent_path}")
        with open(args.drug_latent_path, 'rb') as f:
            drug_latent_dict = pickle.load(f)
    else:
        print(f"Drug latent file not found at {args.drug_latent_path}. Will attempt end-to-end training using SMILES.")
        
    # Load SMILES data if needed
    if os.path.exists(args.drug_smiles_path):
        print(f"Loading Drug SMILES from {args.drug_smiles_path}")
        drug_smiles_df = pd.read_csv(args.drug_smiles_path, index_col=0) # Assuming index is drug ID
        # Check if SMILES col exists
        if 'SMILES' not in drug_smiles_df.columns:
             # Try to find it
             cols = [c for c in drug_smiles_df.columns if 'smiles' in c.lower()]
             if cols:
                 drug_smiles_df = drug_smiles_df.rename(columns={cols[0]: 'SMILES'})
                 
        # Pre-process graphs if we are going to use them
        if drug_latent_dict is None:
             drug_graph_dict = process_drug_data(drug_smiles_df, smile_col='SMILES')
             
    print(f"Loading Response Data from {args.drug_response_path}")
    response_df = pd.read_csv(args.drug_response_path)
    
    # Load Model Selection
    print(f"Loading model selection from {args.model_select_path}")
    model_select_df = pd.read_csv(args.model_select_path)
    
    # Prepare Summary CSV
    summary_file = os.path.join(args.outfolder, 'final_summary.csv')
    summary_headers = ['Exp_ID', 'Model_ID', 'Model_Type', 'Mode', 'Split/Fold', 'AUC', 'AUPRC', 'TCGA_AUC', 'TCGA_Avg_Score'] 
    # Add parameter keys to headers
    for sec, k in param_keys:
        summary_headers.append(f"{k}")
        
    if not os.path.exists(summary_file):
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(summary_headers)

    # 4. Main Experiment Loop
    exp_counter = 0
    
    for idx, row in model_select_df.iterrows():
        model_id = row['ID']
        model_type = row.get('model_type', 'GNN_AE')
        
        print(f"\n=== Processing Model ID: {model_id} ===")
        
        # Resolve Model Folder & Latents
        try:
            folder_name = construct_folder_name(row)
            base_dir = os.path.dirname(args.model_select_path)
            model_folder = os.path.join(base_dir, folder_name)
            
            if not os.path.exists(model_folder):
                print(f"  Model folder not found: {model_folder}. Skipping.")
                continue
                
            # Generate/Load Latent
            if 'encoder_dims' in row and pd.notna(row['encoder_dims']):
                encoder_dims = ast.literal_eval(row['encoder_dims'])
            elif 'gnn_hidden_dims' in row and pd.notna(row['gnn_hidden_dims']):
                encoder_dims = ast.literal_eval(row['gnn_hidden_dims'])
            else:
                encoder_dims = [128, 64] # Default fallback
            latent_path = generate_latent_if_missing(model_folder, model_type, args.pretrain_data_path, args.device, encoder_dims, row.get('dropout', row.get('dropout_rate', row.get('gnn_dropout', 0.1))), row, args.ppi_path)
            
            with open(latent_path, 'rb') as f:
                expression_latent_dict = pickle.load(f)
            latent_dim = len(expression_latent_dict[next(iter(expression_latent_dict))])
            
            # Prepare Base Datasets (to reuse across grid if possible, improving speed)
            # Actually, split depends on random_state, which is constant. Only params change.
            # But batch_size is a param. So loaders need re-init.
            
        except Exception as e:
            print(f"  Error setting up model {model_id}: {e}")
            continue

        # Grid Search for this Model
        for params in tqdm(grid_product, desc=f"Grid Searching Model {model_id}"):
            # Construct dictionary for this run
            current_config = {}
            flat_param_dict = {} # For logging
            
            for (sec, k), val in zip(param_keys, params):
                if sec not in current_config: current_config[sec] = {}
                current_config[sec][k] = val
                flat_param_dict[f"{k}"] = val
                
            # Setup Experiment Folder
            exp_folder = os.path.join(args.outfolder, f"exp_{exp_counter}")
            os.makedirs(exp_folder, exist_ok=True)
            
            # Save specific config
            with open(os.path.join(exp_folder, 'config.json'), 'w') as f:
                json.dump(current_config, f, indent=4)
                
            # Extract Params
            lr = current_config['finetune_params']['ftlr']
            epochs = current_config['training_params']['epochs']
            batch_size = current_config['training_params']['batch_size']
            patience = current_config['training_params']['patience']
            gin_type = current_config['model_params']['gin_type']
            
            # Determine mode based on available data
            if drug_latent_dict is None:
                gin_type = 'dapl'
                print("Using end-to-end GIN training (DAPL mode).")
            elif gin_type != 'precomputed':
                 # If latents avail but config says otherwise, respect config?
                 pass 
            
            # Helper to create model
            def create_model():
                if gin_type == 'precomputed':
                    # Load drug dim
                    d_key = next(iter(drug_latent_dict))
                    drug_dim = len(drug_latent_dict[d_key])
                    drug_model = None
                else:
                    # GIN Drug Model
                    # We assume standard GIN configuration for now or add to params
                    drug_input_dim = 77 # Atom features (matches tools.dataprocess) Or 78? GINConvNet default is 78.
                    # Let's check GINConvNet default
                    drug_input_dim = 78 
                    drug_hidden_dim = 32
                    drug_output_dim = 32 # embedding dim
                    
                    drug_model = GINConvNet(
                        input_dim=drug_input_dim,
                        output_dim=drug_output_dim,
                        num_layers=5,
                        jk_mode='cat', # DAPL style
                        pool_type='max'
                    ).to(args.device)
                    drug_dim = 32 # fixed output dim of GINConvNet above
                
                input_dim = latent_dim + drug_dim
                
                # Classifier params
                c_params = current_config['classifier_params']
                act_map = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}
                act_fn = act_map.get(c_params['activation'], nn.ReLU)
                
                classifier = Classify(
                    input_dim=input_dim,
                    hidden_dims=c_params['hidden_dims'],
                    dop=c_params['dropout_rate'],
                    act_fn=act_fn,
                    use_bn=c_params['use_batch_norm']
                ).to(args.device)
                
                return LatentDrugResponseModel(drug_model, classifier)

            # Training
            results_to_log = []
            
            if args.mode == '1split':
                train_df, val_df = train_test_split(response_df, test_size=0.2, stratify=response_df['Label'], random_state=42)
                
                train_dataset = DrugResponseDataset(train_df, expression_latent_dict, drug_latent_dict if gin_type=='precomputed' else drug_graph_dict, drug_smiles_df, gin_type=gin_type)
                val_dataset = DrugResponseDataset(val_df, expression_latent_dict, drug_latent_dict if gin_type=='precomputed' else drug_graph_dict, drug_smiles_df, gin_type=gin_type)
                
                def collate_fn(batch):
                    gene_list, drug_list, target_list = zip(*batch)
                    return torch.stack(list(gene_list)), torch.stack(list(drug_list)), torch.stack(list(target_list))

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                
                model = create_model().to(args.device)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                best_auc = 0
                best_auprc = 0
                patience_count = 0
                
                # Tracking lists
                train_losses = []
                val_losses = []
                val_aucs = []
                
                for epoch in range(epochs):
                    t_loss, t_auc, _ = train_epoch_ops(model, train_loader, optimizer, criterion, args.device)
                    v_loss, v_auc, v_auprc = evaluate_ops(model, val_loader, criterion, args.device)
                    
                    train_losses.append(t_loss)
                    val_losses.append(v_loss)
                    val_aucs.append(v_auc)
                    
                    if v_auc > best_auc:
                        best_auc = v_auc
                        best_auprc = v_auprc
                        torch.save(model.state_dict(), os.path.join(exp_folder, 'best_model.pth'))
                        patience_count = 0
                    else:
                        patience_count += 1
                        if patience_count >= patience: break
                        
                # Plot learning curves
                plot_finetune_curves(train_losses, val_losses, val_aucs, os.path.join(exp_folder, 'finetune_learning_curve.png'))
                
                results_to_log.append({
                    'Split/Fold': '1split',
                    'AUC': best_auc,
                    'AUPRC': best_auprc,
                    'TCGA_AUC': None,
                    'TCGA_Avg_Score': None
                })
                
            elif args.mode == '5fold':
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                def collate_fn(batch):
                    gene_list, drug_list, target_list = zip(*batch)
                    return torch.stack(list(gene_list)), torch.stack(list(drug_list)), torch.stack(list(target_list))
                
                fold_aucs = []
                fold_auprcs = []
                
                for fold, (train_idx, val_idx) in enumerate(skf.split(response_df, response_df['Label'])):
                    print(f"  Fold {fold+1}/5")
                    train_sub = response_df.iloc[train_idx]
                    val_sub = response_df.iloc[val_idx]
                    
                    train_dataset = DrugResponseDataset(train_sub, expression_latent_dict, drug_latent_dict if gin_type=='precomputed' else drug_graph_dict, drug_smiles_df, gin_type=gin_type)
                    val_dataset = DrugResponseDataset(val_sub, expression_latent_dict, drug_latent_dict if gin_type=='precomputed' else drug_graph_dict, drug_smiles_df, gin_type=gin_type)
                    
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                    
                    model = create_model().to(args.device)
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    
                    b_auc = 0
                    b_auprc = 0
                    
                    # Tracking lists for this fold
                    train_losses = []
                    val_losses = []
                    val_aucs = []
                    
                    for epoch in range(epochs):
                        t_loss, t_auc, _ = train_epoch_ops(model, train_loader, optimizer, criterion, args.device)
                        v_loss, v_auc, v_auprc = evaluate_ops(model, val_loader, criterion, args.device)
                        
                        train_losses.append(t_loss)
                        val_losses.append(v_loss)
                        val_aucs.append(v_auc)
                        
                        if v_auc > b_auc:
                            b_auc = v_auc
                            b_auprc = v_auprc
                    
                    # Plot learning curves for this fold
                    plot_finetune_curves(train_losses, val_losses, val_aucs, os.path.join(exp_folder, f'finetune_learning_curve_fold_{fold+1}.png'))
                    
                    fold_aucs.append(b_auc)
                    fold_auprcs.append(b_auprc)
                    
                    # Log per fold if needed, but summary usually implies average or best
                    
                mean_auc = np.mean(fold_aucs)
                mean_auprc = np.mean(fold_auprcs)
                
                results_to_log.append({
                    'Split/Fold': '5fold_mean',
                    'AUC': mean_auc,
                    'AUPRC': mean_auprc,
                    'TCGA_AUC': None,
                    'TCGA_Avg_Score': None
                })
            
            # Write to Summary Code
            with open(summary_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for res in results_to_log:
                    # 'Exp_ID', 'Model_ID', 'Model_Type', 'Mode', 'Split/Fold', 'AUC', 'AUPRC', 'TCGA_AUC', 'TCGA_Avg_Score' + keys
                    row_data = [
                        exp_counter,
                        model_id,
                        model_type,
                        args.mode,
                        res['Split/Fold'],
                        f"{res['AUC']:.4f}",
                        f"{res['AUPRC']:.4f}",
                        res['TCGA_AUC'],
                        res['TCGA_Avg_Score']
                    ]
                    # Add params
                    for sec, k in param_keys:
                        row_data.append(current_config[sec][k])
                    
                    writer.writerow(row_data)
            
            exp_counter += 1

if __name__ == '__main__':
    main()
