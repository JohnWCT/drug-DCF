import os
import torch
import numpy as np
import pandas as pd
import json
import time
from datetime import timedelta
from collections import defaultdict
from itertools import product
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric import data as DATA
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools.model_opt import VAE, Classify, init_weights # Assuming these are in a 'tools' directory
from drugmodels.ginconv import GINConvNet # Assuming this is in a 'drugmodels' directory
from tools.dataprocess import cat_tensor_with_drug, smile_to_graph, safemakedirs # Using safemakedirs from tools.dataprocess
from tools.inference_utils import inference_on_tcga_drugs as inference_on_tcga_drugs_latent

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. No GPU detected.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FIXED_DRUG_SMILES_DATA_PATH = "data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv"
FIXED_TCGA_DATA_FOLDER = "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv"
FIXED_TCGA_DATA_FOLDER_EXTRA = "data/TCGA/TCGA_drug_response_from_DAPL.csv"

# Moved default hyperparameter grids and model configurations here
DEFAULT_PARAMS_GRID = {
    "pretrain_num_epochs": [0, 100, 300],
    'pretrain_learning_rate': [0.001],
    'gan_learning_rate': [0.001],
    "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000],
    "dropout_rate": [0.0, 0.2],
    "encoder_dims": [[512, 256, 128, 64], [128]],
    "lambda_cls": [1, 5, 10]  # Will be removed if not using lambda_cls
}

DEFAULT_FINE_TUNE_PARAMS_GRID = {
    'ftlr': [0.01, 0.001],
    'scheduler_flag': [True, False]
}

# Define constants for model dimensions and training parameters
GENE_INPUT_DIM = 1426
DRUG_INPUT_DIM = 78
ENCODER_LATENT_DIM = 32
DRUG_EMBED_DIM = 10
CLASSIFIER_INPUT_DIM_EXTENSION = DRUG_EMBED_DIM

DEFAULT_PATIENCE = 20


def _normalize_drug_name(name):
    if pd.isna(name):
        return ""
    return ''.join(ch for ch in str(name).lower().strip() if ch.isalnum())


def _align_drug_embedding(drug_emb, drug_batch_obj, expected_batch_size):
    if drug_emb.size(0) == expected_batch_size:
        return drug_emb
    if hasattr(drug_batch_obj, "batch"):
        pooled = global_mean_pool(drug_emb, drug_batch_obj.batch)
        if pooled.size(0) == expected_batch_size:
            return pooled
    # Fallback for legacy GIN outputs that flatten node-level embeddings by graph blocks.
    if drug_emb.size(0) % expected_batch_size == 0:
        group_size = drug_emb.size(0) // expected_batch_size
        if drug_emb.dim() == 1:
            return drug_emb.view(expected_batch_size, group_size).mean(dim=1, keepdim=True)
        return drug_emb.view(expected_batch_size, group_size, drug_emb.size(1)).mean(dim=1)
    raise RuntimeError(
        f"Drug embedding batch mismatch: expected {expected_batch_size}, got {drug_emb.size(0)}"
    )

class DrugResponseDataset(Dataset):
    def __init__(self, df, expression_df, drug_smiles_df):
        self.df = df.reset_index(drop=True).copy()
        self.drug_smiles_df = drug_smiles_df
        self.expr_dict = {}
        for sample_id in expression_df.index:
            expr = expression_df.loc[sample_id].values.astype(np.float32)
            self.expr_dict[sample_id] = torch.tensor(expr, dtype=torch.float32)
        self.drug_graph_dict = {}
        self.drug_graph_norm_map = {}
        for drug_id in self.drug_smiles_df.index:
            drug_smile = self.drug_smiles_df.loc[drug_id]['SMILES']
            c_size, atom_features_list, edge_index = smile_to_graph(drug_smile)
            drug_x = torch.tensor(np.array(atom_features_list), dtype=torch.float32)
            if len(edge_index) > 0:
                drug_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                drug_edge_index = torch.empty((2, 0), dtype=torch.long)
            drug_data = DATA.Data(x=drug_x, edge_index=drug_edge_index)
            self.drug_graph_dict[drug_id] = drug_data
            self.drug_graph_norm_map[_normalize_drug_name(drug_id)] = drug_id

        id_col = 'DepMap_ID' if 'DepMap_ID' in self.df.columns else ('ModelID' if 'ModelID' in self.df.columns else None)
        drug_col = 'drug_name' if 'drug_name' in self.df.columns else ('DRUG_NAME' if 'DRUG_NAME' in self.df.columns else None)
        if id_col and drug_col:
            before = len(self.df)
            valid_sample = self.df[id_col].isin(self.expr_dict)
            valid_drug = self.df[drug_col].map(lambda x: _normalize_drug_name(x) in self.drug_graph_norm_map)
            self.df = self.df[valid_sample & valid_drug].reset_index(drop=True)
            removed = before - len(self.df)
            if removed > 0:
                print(f"[DrugResponseDataset] filtered {removed} rows with missing sample/drug features.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if 'DepMap_ID' in row:
            sample_id = row['DepMap_ID']
        elif 'ModelID' in row:
            sample_id = row['ModelID']
        else:
            raise ValueError("Neither 'DepMap_ID' nor 'ModelID' column found in data")
        if 'drug_name' in row:
            drug_id = row['drug_name']
        elif 'DRUG_NAME' in row:
            drug_id = row['DRUG_NAME']
        else:
            raise ValueError("Neither 'drug_name' nor 'DRUG_NAME' column found in data")
        if 'Label' in row:
            target = float(row['Label'])
        elif 'Class' in row:
            target = float(row['Class'])
        else:
            raise ValueError("Neither 'Label' nor 'Class' column found in data")
        gene_feature = self.expr_dict[sample_id]
        drug_key = self.drug_graph_norm_map.get(_normalize_drug_name(drug_id), drug_id)
        drug_data = self.drug_graph_dict[drug_key]
        target = torch.tensor(target, dtype=torch.float32)
        return gene_feature, drug_data, target

def get_encoder_output(encoder, x, encoder_type='vae'):
    if encoder_type == 'vae':
        _, z, _, _ = encoder(x)
    else:
        z = encoder(x)
    return z

def train_one_epoch(model_components, dataloader, optimizer, loss_fn):
    encoder = model_components['encoder']
    drug_model = model_components['drug_model']
    classifier = model_components['classifier']
    encoder_type = model_components['encoder_type']
    
    encoder.eval()
    drug_model.eval()
    classifier.train()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Training Batches", leave=False):
        gene_list, drug_data_list, target_list = batch
        batch_gene = torch.stack(gene_list).to(device)
        batch_target = torch.stack(target_list).to(device)
        drug_batch = Batch.from_data_list(drug_data_list).to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            z = get_encoder_output(encoder, batch_gene, encoder_type)
            drug_emb = drug_model(drug_batch)
            drug_emb = _align_drug_embedding(drug_emb, drug_batch, z.size(0))
        
        combined = torch.cat((z, drug_emb), dim=1)
        pred = classifier(combined).view(-1)
        
        # Check for NaN values in predictions
        if torch.isnan(pred).any():
            print("Warning: NaN values detected in predictions. Replacing with zeros.")
            pred = torch.nan_to_num(pred, nan=0.0)
            
        loss = loss_fn(pred, batch_target)
        
        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN or Inf loss detected. Using zero loss instead.")
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * batch_gene.size(0)
        all_preds.extend(pred.detach().cpu().numpy())
        all_targets.extend(batch_target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Convert to numpy arrays and handle NaN values
    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)
    
    # Check for and handle NaN values
    if np.isnan(all_preds_np).any():
        print("Warning: NaN values found in predictions. Replacing with zeros for metric calculation.")
        all_preds_np = np.nan_to_num(all_preds_np, nan=0.0)
    
    # Calculate metrics with error handling
    try:
        auc_score = roc_auc_score(all_targets_np, all_preds_np)
        auprc_score = average_precision_score(all_targets_np, all_preds_np)
    except ValueError as e:
        print(f"Error calculating metrics: {e}")
        print("Using default metrics (AUC=0.5, AUPRC=class ratio)")
        auc_score = 0.5  # Default AUC (random classifier)
        # Default AUPRC is the proportion of positive samples
        positive_ratio = np.mean(all_targets_np)
        auprc_score = positive_ratio if not np.isnan(positive_ratio) else 0.5
    
    return avg_loss, auc_score, auprc_score

def evaluate_model(model_components, dataloader, loss_fn):
    encoder = model_components['encoder']
    drug_model = model_components['drug_model']
    classifier = model_components['classifier']
    encoder_type = model_components['encoder_type']
    
    encoder.eval()
    drug_model.eval()
    classifier.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation Batches", leave=False):
            gene_list, drug_data_list, target_list = batch
            batch_gene = torch.stack(gene_list).to(device)
            batch_target = torch.stack(target_list).to(device)
            drug_batch = Batch.from_data_list(drug_data_list).to(device)
            
            z = get_encoder_output(encoder, batch_gene, encoder_type)
            drug_emb = drug_model(drug_batch)
            drug_emb = _align_drug_embedding(drug_emb, drug_batch, z.size(0))
            combined = torch.cat((z, drug_emb), dim=1)
            pred = classifier(combined).view(-1)
            
            # Handle NaN values in predictions
            if torch.isnan(pred).any():
                pred = torch.nan_to_num(pred, nan=0.0)
                
            loss = loss_fn(pred, batch_target)
            
            # Handle NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(0.0, device=device)
                
            total_loss += loss.item() * batch_gene.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Convert to numpy arrays and handle NaN values
    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)
    
    # Check for and handle NaN values
    if np.isnan(all_preds_np).any():
        print("Warning: NaN values found in validation predictions. Replacing with zeros for metric calculation.")
        all_preds_np = np.nan_to_num(all_preds_np, nan=0.0)
    
    # Calculate metrics with error handling
    try:
        auc_score = roc_auc_score(all_targets_np, all_preds_np)
        auprc_score = average_precision_score(all_targets_np, all_preds_np)
    except ValueError as e:
        print(f"Error calculating validation metrics: {e}")
        print("Using default metrics (AUC=0.5, AUPRC=class ratio)")
        auc_score = 0.5
        positive_ratio = np.mean(all_targets_np)
        auprc_score = positive_ratio if not np.isnan(positive_ratio) else 0.5
    
    return avg_loss, auc_score, auprc_score

def inference_on_tcga_drugs(
    model_components,
    tcga_data_folder,
    best_model_path,
    train_params,
    ft_params,
    drug_smiles_df,
    tcga_expression_path="data/TCGA/pretrain_tcga.csv",
    tcga_tag='TCGA',
):
    encoder = model_components['encoder']
    encoder_type = model_components['encoder_type']
    if best_model_path is not None and os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model_components['classifier'].load_state_dict(checkpoint['classifier_state_dict'])
            if model_components.get('drug_model') is not None and 'drug_model_state_dict' in checkpoint:
                model_components['drug_model'].load_state_dict(checkpoint['drug_model_state_dict'])
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Continuing with current classifier state...")

    expr_df = pd.read_csv(tcga_expression_path, index_col=0)
    expr_df.index = expr_df.index.astype(str)
    tcga_latent_dict = {}
    with torch.no_grad():
        batch_tensor = torch.from_numpy(expr_df.values.astype(np.float32)).to(device)
        z = get_encoder_output(encoder, batch_tensor, encoder_type).detach().cpu().numpy()
    for i, sid in enumerate(expr_df.index.tolist()):
        tcga_latent_dict[sid] = z[i].tolist()

    raw_results = inference_on_tcga_drugs_latent(
        model_components={'drug_model': model_components['drug_model'], 'classifier': model_components['classifier']},
        tcga_data_path=tcga_data_folder,
        best_model_path=None,
        ft_params=ft_params,
        tcga_latent_dict=tcga_latent_dict,
        drug_latent_dict=None,
        gin_type='dapl',
        fold_model_folder=None,
        drug_smiles_df=drug_smiles_df,
        tcga_tag=tcga_tag,
    )
    if not raw_results:
        return {}
    return {
        drug: {'AUC': mets.get('AUC', np.nan), 'AUPRC': mets.get('AUPRC', np.nan)}
        for drug, mets in raw_results.get('Drug_Metrics', {}).items()
    }

def step_1_finetune_pipeline_zscore(
    pretrain_model_root_folder, 
    drug_model_path, 
    outfolder,
    response_data_path,
    expression_data_path,
    drug_smiles_data_path,
    tcga_inference_data_folder,
    tcga_inference_data_folder_extra,
    params_grid_dict,
    fine_tune_params_grid_dict,
    batch_size=2048, 
    encoder_type='vae',
    use_lambda_cls=False,
    run_config=None):  # Added parameter to indicate if we're using models with lambda_cls
    
    response_df = pd.read_csv(response_data_path)
    expression_df = pd.read_csv(expression_data_path, index_col=0)
    drug_smiles_df = pd.read_csv(drug_smiles_data_path, index_col=0)[['SMILES']].dropna()
    
    valid_ids = expression_df.index.astype(str)
    id_column = 'DepMap_ID' if 'DepMap_ID' in response_df.columns else 'ModelID'
    response_df[id_column] = response_df[id_column].astype(str)
    response_df = response_df[response_df[id_column].isin(valid_ids)]
    all_samples = response_df[id_column].unique()
    print(f'Total number of unique samples: {len(all_samples)}')
    
    # Split samples into train+val and test sets (90-10 split)
    train_val_samples, test_samples = train_test_split(all_samples, test_size=0.1, random_state=42)
    # Further split train+val into train and validation (80-20 split)
    train_samples, val_samples = train_test_split(train_val_samples, test_size=0.2, random_state=42)
    
    print(f'Number of samples in train set: {len(train_samples)}')
    print(f'Number of samples in validation set: {len(val_samples)}')
    print(f'Number of samples in test set: {len(test_samples)}')
    
    drug_encoder_dict = torch.load(drug_model_path)
    
    # Read model parameters from model_select.csv instead of generating from config
    model_select_path = os.path.join(pretrain_model_root_folder, 'model_select.csv')
    if not os.path.exists(model_select_path):
        alt_model_select_path = os.path.join(pretrain_model_root_folder, '00_report', 'model_select.csv')
        if os.path.exists(alt_model_select_path):
            model_select_path = alt_model_select_path
    resolved_model_select_dir = os.path.dirname(model_select_path)
    model_dir_roots = [resolved_model_select_dir]
    model_select_parent_dir = os.path.dirname(resolved_model_select_dir)
    if model_select_parent_dir not in model_dir_roots:
        model_dir_roots.append(model_select_parent_dir)
    try:
        model_select_df = pd.read_csv(model_select_path)
        print(f"Successfully loaded model parameters from {model_select_path}")
        
        # Convert parameters to the correct format
        param_combinations = []
        for index, row in model_select_df.iterrows():
            # Skip rows where ID is empty or NaN
            if 'ID' not in row or pd.isna(row['ID']) or row['ID'] == '':
                print(f"Skipping row {index}: ID is empty or missing")
                continue
                
            # Parse encoder_dims from model_select.csv
            # In model_select.csv, values are formatted as strings like "[128]" or "[512,256,128,64]"
            try:
                if isinstance(row['encoder_dims'], str):
                    # Strip any extra whitespace
                    dims_str = row['encoder_dims'].strip()
                    
                    encoder_dims_value = eval(dims_str)
                    
                    # Ensure it's a list
                    if not isinstance(encoder_dims_value, list):
                        encoder_dims_value = [encoder_dims_value]
                    
                    encoder_dims = [encoder_dims_value]
                else:
                    # If it's not a string (unlikely in CSV), use default
                    encoder_dims = [[128]]
                    
                print(f"Parsed encoder_dims: {encoder_dims} from input: {row['encoder_dims']}")
            except Exception as e:
                print(f"Warning: Could not parse encoder_dims '{row['encoder_dims']}' for ID {row['ID']}, using default [128]. Error: {e}")
                encoder_dims = [[128]]  # Default if parsing fails
            
            # Create parameter dictionary
            try:
                param_dict = {
                    'pretrain_num_epochs': int(row['pretrain_epochs']),
                    'train_num_epochs': int(row['train_epochs']),
                    'pretrain_learning_rate': float(row['pretrain_lr']),
                    'gan_learning_rate': float(row['train_lr']),
                    'dropout_rate': float(row['dropout']),
                    'encoder_dims': encoder_dims
                }
                # Preserve direct experiment folder reference when available (e.g., exp_001).
                if 'result_folder' in row and not pd.isna(row['result_folder']) and str(row['result_folder']).strip():
                    param_dict['result_folder'] = str(row['result_folder']).strip()
                
                # Add lambda_cls if it exists in the dataframe
                if 'lambda_cls' in row and use_lambda_cls:
                    param_dict['lambda_cls'] = int(row['lambda_cls']) if not pd.isna(row['lambda_cls']) else 0
                    
                # Add use_class_weight if it exists in the dataframe
                if 'use_class_weight' in row:
                    # Parse boolean value from string or boolean
                    if isinstance(row['use_class_weight'], bool):
                        param_dict['use_class_weight'] = row['use_class_weight']
                    elif isinstance(row['use_class_weight'], str):
                        param_dict['use_class_weight'] = row['use_class_weight'].lower() == 'true'
                    else:
                        # Default to False if value can't be interpreted
                        param_dict['use_class_weight'] = False
                
                param_combinations.append(param_dict)
                print(f"Added parameters for ID {row['ID']}")
            except Exception as e:
                print(f"Error in row {index} (ID: {row.get('ID', 'unknown')}): {e}")
                print(f"Row data: {row.to_dict()}")
                # Continue to next row instead of stopping completely
                continue
            
        print(f"Created {len(param_combinations)} parameter combinations from model_select.csv")
        
        if len(param_combinations) == 0:
            raise ValueError("No valid parameter combinations could be created from model_select.csv")
            
    except FileNotFoundError:
        print(f"Error: model_select.csv not found at {model_select_path}")
        print("Please ensure the file exists and try again.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: model_select.csv is empty at {model_select_path}")
        raise
    except Exception as e:
        print(f"Error reading model_select.csv: {e}")
        # Try to provide more detailed error information if possible
        if 'model_select_df' in locals():
            # If we have the dataframe but had issues processing it
            print("\nDetailed error information for each problematic row:")
            for index, row in model_select_df.iterrows():
                # Skip rows where ID is empty or NaN in detailed error check
                if 'ID' not in row or pd.isna(row['ID']) or row['ID'] == '':
                    print(f"Row {index}: Skipping (ID is empty or missing)")
                    continue
                    
                try:
                    # Test conversion of each value
                    int(row['pretrain_epochs'])
                    int(row['train_epochs'])
                    float(row['pretrain_lr'])
                    float(row['train_lr']) 
                    float(row['dropout'])
                    # encoder_dims is handled separately above
                    if 'lambda_cls' in row and use_lambda_cls and not pd.isna(row['lambda_cls']):
                        int(row['lambda_cls'])
                except Exception as row_e:
                    print(f"Row {index} (ID: {row['ID']}): Data conversion error: {row_e}")
                    print(f"Row data: {row.to_dict()}")
        
        print("\nPlease fix the errors in model_select.csv and try again.")
        raise
    
    # Generate fine-tuning parameter combinations from config
    ft_param_combinations = [dict(zip(fine_tune_params_grid_dict.keys(), v)) 
                           for v in product(*fine_tune_params_grid_dict.values())]
    
    all_param_results_list = []
    best_overall_tcga_metrics = {'best_overall_auc': float('-inf'), 'params': None, 'metrics': None}
    
    safemakedirs(outfolder)
    with open(os.path.join(outfolder, 'finetune_config_used.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'pretrain_model_root_folder': pretrain_model_root_folder,
            'drug_model_path': drug_model_path,
            'response_data_path': response_data_path,
            'expression_data_path': expression_data_path,
            'drug_smiles_data_path': drug_smiles_data_path,
            'tcga_data_folder': tcga_inference_data_folder,
            'tcga_data_folder_extra': tcga_inference_data_folder_extra,
            'batch_size': batch_size,
            'encoder_type': encoder_type,
            'use_lambda_cls': use_lambda_cls,
            'run_config': run_config
        }, f, indent=2, ensure_ascii=False, default=str)
    
    for ft_params in ft_param_combinations:
        for train_params in param_combinations:
            current_hyperparams = {**train_params, **ft_params}
            print(f"\nTraining with parameters: {current_hyperparams}")
            
            # Determine encoder dimensions for directory naming and model creation
            current_encoder_dims = train_params['encoder_dims']
            if isinstance(current_encoder_dims[0], list):
                current_encoder_dims = current_encoder_dims[0]
            
            # Construct directory name parts exactly like in pretrain_VAEwC.py
            model_name_parts = [
                f"pt_epochs_{train_params['pretrain_num_epochs']}",
                f"t_epochs_{train_params['train_num_epochs']}",
                f"Ptlr_{train_params['pretrain_learning_rate']}",
                f"tlr{train_params['gan_learning_rate']}",
                f"dop{train_params['dropout_rate']}",
                f"enc{len(current_encoder_dims)}"  # Match the naming convention in pretrain_VAEwC.py
            ]
            
            # Add lambda_cls to model name only if we're using it
            if use_lambda_cls:
                model_name_parts.append(f"lambda_cls{train_params.get('lambda_cls', 0)}")
            
            # Add use_class_weight to model name if it exists
            if 'use_class_weight' in train_params and train_params['use_class_weight']:
                model_name_parts.append("use_cw_True")
            
            # Add finetune parameters
            model_name_parts.extend([
                f"ftlr_{ft_params['ftlr']}",
                f"sched_{ft_params['scheduler_flag']}"
            ])
            
            model_name = ",".join(model_name_parts)
            model_folder = os.path.join(outfolder, model_name)
            safemakedirs(model_folder)
            with open(os.path.join(model_folder, 'params_used.json'), 'w', encoding='utf-8') as f:
                json.dump(current_hyperparams, f, indent=2, ensure_ascii=False, default=str)
            
            # Create datasets for train, validation, and test
            train_df = response_df[response_df[id_column].isin(train_samples)]
            val_df = response_df[response_df[id_column].isin(val_samples)]
            test_df = response_df[response_df[id_column].isin(test_samples)]
            
            train_subset = DrugResponseDataset(train_df, expression_df, drug_smiles_df)
            val_subset = DrugResponseDataset(val_df, expression_df, drug_smiles_df)
            test_subset = DrugResponseDataset(test_df, expression_df, drug_smiles_df)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            # Prefer explicit folder from model_select.csv; fallback to legacy naming convention.
            pretrain_model_dir = None
            if train_params.get('result_folder'):
                for root_dir in model_dir_roots:
                    candidate_dir = os.path.join(root_dir, train_params['result_folder'])
                    candidate_encoder = os.path.join(candidate_dir, 'after_traingan_shared_vae.pth')
                    if os.path.exists(candidate_encoder):
                        pretrain_model_dir = candidate_dir
                        break
            if pretrain_model_dir is None:
                if use_lambda_cls:
                    # For models trained with VAEwC (with lambda_cls)
                    pretrain_model_dir_name = (f"pt_epochs_{train_params['pretrain_num_epochs']}"
                                            f",t_epochs_{train_params['train_num_epochs']}"
                                            f",Ptlr_{train_params['pretrain_learning_rate']}"
                                            f",tlr{train_params['gan_learning_rate']}"
                                            f",dop{train_params['dropout_rate']}"
                                            f",enc{len(current_encoder_dims)}"
                                            f",lambda_cls{train_params.get('lambda_cls', 0)}")
                    
                    # Add use_class_weight if it exists
                    if 'use_class_weight' in train_params and train_params['use_class_weight']:
                        pretrain_model_dir_name += ",use_cw_True"
                else:
                    # For models trained with standard VAE (without lambda_cls)
                    pretrain_model_dir_name = (f"pt_epochs_{train_params['pretrain_num_epochs']}"
                                            f",t_epochs_{train_params['train_num_epochs']}"
                                            f",Ptlr_{train_params['pretrain_learning_rate']}"
                                            f",tlr{train_params['gan_learning_rate']}"
                                            f",dop{train_params['dropout_rate']}"
                                            f",enc{len(current_encoder_dims)}")
                for root_dir in model_dir_roots:
                    candidate_dir = os.path.join(root_dir, pretrain_model_dir_name)
                    candidate_encoder = os.path.join(candidate_dir, 'after_traingan_shared_vae.pth')
                    if os.path.exists(candidate_encoder):
                        pretrain_model_dir = candidate_dir
                        break
                if pretrain_model_dir is None:
                    pretrain_model_dir = os.path.join(model_dir_roots[0], pretrain_model_dir_name)
            print(f"Looking for pretrained model in: {pretrain_model_dir}")
            encoder_path = os.path.join(pretrain_model_dir, 'after_traingan_shared_vae.pth')
            classifier_path = os.path.join(pretrain_model_dir, 'after_traingan_classifier.pth')
            
            try:
                encoder_state_dict = torch.load(encoder_path, map_location=device)
                # Check if cancer classifier exists (for VAEwC models)
                has_cancer_classifier = os.path.exists(classifier_path)
                if has_cancer_classifier and use_lambda_cls:
                    print(f"Found cancer classifier model at {classifier_path}")
                elif use_lambda_cls and not has_cancer_classifier:
                    print(f"Warning: Expected cancer classifier but not found at {classifier_path}")
                    print(f"Continuing without cancer classifier...")
            except FileNotFoundError:
                print(f"ERROR: Pretrained encoder not found at {encoder_path}")
                print(f"Skipping this parameter combination: {current_hyperparams}")
                continue
            
            if encoder_type == 'vae':
                encoder = VAE(input_size=GENE_INPUT_DIM, 
                            output_size=GENE_INPUT_DIM, 
                            latent_size=ENCODER_LATENT_DIM,
                            encoder_hidden_dims=current_encoder_dims,
                            decoder_hidden_dims=current_encoder_dims[::-1],
                            dop=train_params['dropout_rate'],
                            act_fn=nn.ReLU).to(device)
            else:
                from tools.model_opt import MLP
                encoder = MLP(input_dim=GENE_INPUT_DIM, 
                            output_dim=ENCODER_LATENT_DIM, 
                            hidden_dims=current_encoder_dims,
                            dop=train_params['dropout_rate'],
                            act_fn=nn.ReLU,
                            gr_flag=False).to(device)
            
            encoder.load_state_dict(encoder_state_dict)
            
            drug_gcnmodel = GINConvNet(input_dim=DRUG_INPUT_DIM, output_dim=DRUG_EMBED_DIM).to(device)
            try:
                drug_gcnmodel.load_state_dict(drug_encoder_dict)
            except RuntimeError as e:
                # Allow architecture mismatch between historical drug encoder checkpoints and current GIN layout.
                print(f"Warning: strict drug encoder load failed ({e}). Retrying with strict=False.")
                drug_gcnmodel.load_state_dict(drug_encoder_dict, strict=False)
            
            classifymodel = Classify(input_dim=ENCODER_LATENT_DIM + CLASSIFIER_INPUT_DIM_EXTENSION,
                                  hidden_dims=[10],
                                  dop=0.0, 
                                  act_fn=nn.ReLU,
                                  out_fn=None,
                                  use_bn=True).to(device)
            classifymodel.apply(init_weights)
            
            # Set bias to small value to prevent early NaN issues
            for name, param in classifymodel.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.01)
                elif 'weight' in name:
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
            
            model_components = {
                'encoder': encoder, 'drug_model': drug_gcnmodel,
                'classifier': classifymodel, 'encoder_type': encoder_type
            }
            
            optimizer = optim.AdamW(classifymodel.parameters(), lr=ft_params['ftlr'], weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, train_params['train_num_epochs']) if ft_params['scheduler_flag'] else None
            loss_fn = nn.BCEWithLogitsLoss()
            
            best_val_auc = float('-inf')
            best_epoch = 0
            patience_counter = 0
            
            metrics_history = defaultdict(list)
            
            num_finetune_epochs = train_params['train_num_epochs']
            
            for epoch in range(num_finetune_epochs):
                train_loss, train_auc, train_auprc = train_one_epoch(model_components, train_loader, optimizer, loss_fn)
                if scheduler: scheduler.step()
                val_loss, val_auc, val_auprc = evaluate_model(model_components, val_loader, loss_fn)
                
                metrics_history['train_loss'].append(train_loss)
                metrics_history['val_loss'].append(val_loss)
                metrics_history['train_auc'].append(train_auc)
                metrics_history['val_auc'].append(val_auc)
                metrics_history['train_auprc'].append(train_auprc)
                metrics_history['val_auprc'].append(val_auprc)
                
                print(f"Epoch {epoch + 1}/{num_finetune_epochs} - Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f} | Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f}")
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save({
                        'classifier_state_dict': classifymodel.state_dict(),
                        'epoch': epoch,
                        'best_val_auc': best_val_auc,
                    }, os.path.join(model_folder, 'best_model.pth'))
                else:
                    patience_counter += 1
                    if patience_counter >= DEFAULT_PATIENCE:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Save learning curves
            curves_folder = os.path.join(model_folder, 'learning_curves')
            safemakedirs(curves_folder)
            pd.DataFrame(metrics_history).to_csv(os.path.join(curves_folder, 'metrics_history.csv'), index_label='Epoch')
            
            # Evaluate on test set
            test_loss, test_auc, test_auprc = evaluate_model(model_components, test_loader, loss_fn)
            print(f"\nTest Results - Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}")
            
            # TCGA inference
            best_model_path = os.path.join(model_folder, 'best_model.pth')
            tcga_results = inference_on_tcga_drugs(
                model_components, tcga_inference_data_folder,
                best_model_path, train_params, ft_params, drug_smiles_df, tcga_tag='TCGA1'
            )
            tcga_results_extra = {}
            if tcga_inference_data_folder_extra:
                tcga_results_extra = inference_on_tcga_drugs(
                    model_components, tcga_inference_data_folder_extra,
                    best_model_path, train_params, ft_params, drug_smiles_df, tcga_tag='TCGA2'
                )
            
            # Calculate overall TCGA metrics and add them to tcga_results
            tcga_results = calculate_overall_tcga_metrics(tcga_results)
            tcga_results_extra = calculate_overall_tcga_metrics(tcga_results_extra) if tcga_results_extra else {}
            
            # Store results for this parameter combination
            current_param_set_results = {
                'Params': current_hyperparams,
                'Train_Metrics': {
                    'Loss': metrics_history['train_loss'][best_epoch],
                    'AUC': metrics_history['train_auc'][best_epoch],
                    'AUPRC': metrics_history['train_auprc'][best_epoch]
                },
                'Val_Metrics': {
                    'Loss': metrics_history['val_loss'][best_epoch],
                    'AUC': metrics_history['val_auc'][best_epoch],
                    'AUPRC': metrics_history['val_auprc'][best_epoch]
                },
                'Test_Metrics': {
                    'Loss': test_loss,
                    'AUC': test_auc,
                    'AUPRC': test_auprc
                },
                'TCGA_Metrics': tcga_results,
                'TCGA_Metrics_Extra': tcga_results_extra,
                'Best_Epoch': best_epoch + 1
            }
            
            all_param_results_list.append(current_param_set_results)
            
            # Save metrics for current parameter set
            save_single_param_set_metrics(model_folder, current_param_set_results)
            print(f"Metrics for current params saved in: {model_folder}")
            
            # Update best overall TCGA metrics
            if tcga_results['Overall_AUC'] > best_overall_tcga_metrics['best_overall_auc']:
                best_overall_tcga_metrics['best_overall_auc'] = tcga_results['Overall_AUC']
                best_overall_tcga_metrics['params'] = current_hyperparams
                best_overall_tcga_metrics['metrics'] = current_param_set_results
    
    # Save all parameter results to a single JSON file
    all_results_path = os.path.join(outfolder, 'all_parameter_results_summary.json')
    with open(all_results_path, 'w') as f:
        json.dump({
            'all_parameter_sets_results': all_param_results_list,
            'best_performing_params_on_overall_tcga_auc': best_overall_tcga_metrics
        }, f, indent=4, cls=NpEncoder)
    
    # Create final comparative summary CSV for all parameter sets
    comparison_df = create_final_parameter_comparison_csv(outfolder, all_param_results_list, use_lambda_cls)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE - SUMMARY OF RESULTS")
    print("="*80)
    
    if best_overall_tcga_metrics['params']:
        print("\nBest Parameter Combination (based on Overall TCGA AUC):")
        print(f"  Pretrain Epochs: {best_overall_tcga_metrics['params']['pretrain_num_epochs']}")
        print(f"  Train Epochs: {best_overall_tcga_metrics['params']['train_num_epochs']}")
        print(f"  Dropout Rate: {best_overall_tcga_metrics['params']['dropout_rate']}")
        
        # Format encoder_dims for readability
        encoder_dims = best_overall_tcga_metrics['params']['encoder_dims']
        if isinstance(encoder_dims, list) and isinstance(encoder_dims[0], list):
            encoder_dims_display = encoder_dims[0]
        else:
            encoder_dims_display = encoder_dims
        print(f"  Encoder Dims: {encoder_dims_display}")
        
        # Only display lambda_cls if we're using it
        if use_lambda_cls and 'lambda_cls' in best_overall_tcga_metrics['params']:
            print(f"  Lambda Cls: {best_overall_tcga_metrics['params'].get('lambda_cls', 0)}")
            
        print(f"  Finetune LR: {best_overall_tcga_metrics['params']['ftlr']}")
        print(f"  Scheduler: {best_overall_tcga_metrics['params']['scheduler_flag']}")
        
        print("\nPerformance Metrics:")
        print(f"  Train AUC: {best_overall_tcga_metrics['metrics']['Train_Metrics']['AUC']:.4f}")
        print(f"  Val AUC: {best_overall_tcga_metrics['metrics']['Val_Metrics']['AUC']:.4f}")
        print(f"  Test AUC: {best_overall_tcga_metrics['metrics']['Test_Metrics']['AUC']:.4f}")
        print(f"  Best Epoch: {best_overall_tcga_metrics['metrics']['Best_Epoch']}")
        
        print("\nTCGA Performance:")
        print(f"  Overall TCGA AUC: {best_overall_tcga_metrics['metrics']['TCGA_Metrics']['Overall_AUC']:.4f}")
        print(f"  Overall TCGA AUPRC: {best_overall_tcga_metrics['metrics']['TCGA_Metrics']['Overall_AUPRC']:.4f}")
        
        print("\nIndividual TCGA Drug Performance:")
        print(f"  {'Drug':<5} | {'AUC':^15} | {'AUPRC':^15}")
        print(f"  {'-'*5}-+-{'-'*15}-+-{'-'*15}")
        for drug_name, metrics in best_overall_tcga_metrics['metrics']['TCGA_Metrics'].items():
            if drug_name not in ['Overall_AUC', 'Overall_AUPRC']:
                print(f"  {drug_name:<5} | {metrics['AUC']:.4f} | {metrics['AUPRC']:.4f}")
    else:
        print("No results found to determine the best parameters.")
    
    print("\nOutput Files:")
    print(f"  - Detailed Parameter Comparison: {os.path.join(outfolder, 'parameter_comparison_detailed.csv')}")
    print(f"  - TCGA-Focused Parameter Comparison: {os.path.join(outfolder, 'parameter_comparison_tcga_focus.csv')}")
    print(f"  - Complete Results JSON: {all_results_path}")
    print("="*80)
    
    return best_overall_tcga_metrics

def calculate_overall_tcga_metrics(tcga_results):
    """Calculate mean metrics across all TCGA drugs"""
    all_drug_aucs = []
    all_drug_auprcs = []
    
    try:
        for drug_name, metrics in tcga_results.items():
            # Skip existing summary metrics
            if drug_name in ['Overall_AUC', 'Overall_AUPRC']:
                continue
                
            # Only add valid metrics (non-NaN)
            if isinstance(metrics, dict):
                if 'AUC' in metrics and not pd.isna(metrics['AUC']):
                    all_drug_aucs.append(metrics['AUC'])
                if 'AUPRC' in metrics and not pd.isna(metrics['AUPRC']):
                    all_drug_auprcs.append(metrics['AUPRC'])
        
        # Calculate means only if we have valid values
        overall_auc = np.mean(all_drug_aucs) if all_drug_aucs else np.nan
        overall_auprc = np.mean(all_drug_auprcs) if all_drug_auprcs else np.nan
        
        # Return a dictionary with the overall metrics
        overall_metrics = {
            'Overall_AUC': overall_auc,
            'Overall_AUPRC': overall_auprc
        }
        
        # Add overall metrics to the tcga_results dictionary
        tcga_results.update(overall_metrics)
        
    except Exception as e:
        print(f"Error calculating overall TCGA metrics: {e}")
        # Provide default values in case of error
        tcga_results.update({
            'Overall_AUC': np.nan,
            'Overall_AUPRC': np.nan
        })
    
    return tcga_results

def save_single_param_set_metrics(model_folder, param_set_results):
    """Save metrics for a single parameter combination"""
    metrics_out_folder = os.path.join(model_folder, 'metrics_summary')
    safemakedirs(metrics_out_folder)
    
    # Save detailed metrics - use params directly without adding lambda_cls again
    metrics_data = {
        # Parameters - already includes lambda_cls if it exists
        **param_set_results['Params'],
        
        # Train metrics
        'Train_Loss': param_set_results['Train_Metrics']['Loss'],
        'Train_AUC': param_set_results['Train_Metrics']['AUC'],
        'Train_AUPRC': param_set_results['Train_Metrics']['AUPRC'],
        
        # Validation metrics
        'Val_Loss': param_set_results['Val_Metrics']['Loss'],
        'Val_AUC': param_set_results['Val_Metrics']['AUC'],
        'Val_AUPRC': param_set_results['Val_Metrics']['AUPRC'],
        
        # Test metrics
        'Test_Loss': param_set_results['Test_Metrics']['Loss'],
        'Test_AUC': param_set_results['Test_Metrics']['AUC'],
        'Test_AUPRC': param_set_results['Test_Metrics']['AUPRC'],
        
        # Best epoch
        'Best_Epoch': param_set_results['Best_Epoch']
    }
    
    # Add TCGA metrics
    for drug, metrics in param_set_results['TCGA_Metrics'].items():
        if drug not in ['Overall_AUC', 'Overall_AUPRC']:
            metrics_data[f'{drug}_TCGA_AUC'] = metrics['AUC']
            metrics_data[f'{drug}_TCGA_AUPRC'] = metrics['AUPRC']
    
    # Add overall TCGA metrics
    metrics_data['Overall_TCGA_AUC'] = param_set_results['TCGA_Metrics']['Overall_AUC']
    metrics_data['Overall_TCGA_AUPRC'] = param_set_results['TCGA_Metrics']['Overall_AUPRC']
    tcga_extra = param_set_results.get('TCGA_Metrics_Extra', {})
    if tcga_extra:
        metrics_data['TCGA2_Overall_TCGA_AUC'] = tcga_extra.get('Overall_AUC', np.nan)
        metrics_data['TCGA2_Overall_TCGA_AUPRC'] = tcga_extra.get('Overall_AUPRC', np.nan)
        for drug, metrics in tcga_extra.items():
            if drug not in ['Overall_AUC', 'Overall_AUPRC']:
                metrics_data[f'TCGA2_{drug}_TCGA_AUC'] = metrics.get('AUC', np.nan)
                metrics_data[f'TCGA2_{drug}_TCGA_AUPRC'] = metrics.get('AUPRC', np.nan)
    
    # Save the metrics
    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(os.path.join(metrics_out_folder, 'metrics_summary.csv'), index=False)

def create_final_parameter_comparison_csv(outfolder, all_param_results_list, use_lambda_cls=False):
    """Create CSV files comparing all hyperparameter combinations"""
    if not all_param_results_list:
        print("No data to create final parameter comparison CSV.")
        return
    
    # Create a streamlined comparison table focusing on parameters and results
    comparison_rows = []
    
    for result in all_param_results_list:
        # Extract key parameters and metrics
        row = {
            # Parameters
            'Pretrain_Epochs': result['Params']['pretrain_num_epochs'],
            'Train_Epochs': result['Params']['train_num_epochs'],
            'Pretrain_LR': result['Params']['pretrain_learning_rate'],
            'GAN_LR': result['Params']['gan_learning_rate'],
            'Dropout_Rate': result['Params']['dropout_rate'],
            # Display encoder_dims in a cleaner format
            'Encoder_Dims': str(result['Params']['encoder_dims'][0] if isinstance(result['Params']['encoder_dims'], list) and 
                            isinstance(result['Params']['encoder_dims'][0], list) else result['Params']['encoder_dims']),
            'Finetune_LR': result['Params']['ftlr'],
            'Scheduler': result['Params']['scheduler_flag'],
            
            # Add lambda_cls only if we're using it
            **({"Lambda_Cls": result['Params'].get('lambda_cls', 0)} if use_lambda_cls else {}),
            
            # Add use_class_weight if it exists in the parameters
            **({"Use_Class_Weight": result['Params'].get('use_class_weight', False)} if 'use_class_weight' in result['Params'] else {}),
            
            # Train/Val/Test metrics
            'Train_AUC': result['Train_Metrics']['AUC'],
            'Val_AUC': result['Val_Metrics']['AUC'],
            'Test_AUC': result['Test_Metrics']['AUC'],
            'Train_AUPRC': result['Train_Metrics']['AUPRC'],
            'Val_AUPRC': result['Val_Metrics']['AUPRC'],
            'Test_AUPRC': result['Test_Metrics']['AUPRC'],
            
            # Overall TCGA metrics
            'Overall_TCGA_AUC': result['TCGA_Metrics']['Overall_AUC'],
            'Overall_TCGA_AUPRC': result['TCGA_Metrics']['Overall_AUPRC'],
            
            # Add Best Epoch information
            'Best_Epoch': result['Best_Epoch']
        }
        
        # Add individual TCGA drug metrics
        for drug_name, metrics in result['TCGA_Metrics'].items():
            if drug_name not in ['Overall_AUC', 'Overall_AUPRC']:
                row[f'{drug_name}_TCGA_AUC'] = metrics['AUC']
                row[f'{drug_name}_TCGA_AUPRC'] = metrics['AUPRC']
        tcga_extra = result.get('TCGA_Metrics_Extra', {})
        if tcga_extra:
            row['TCGA2_Overall_TCGA_AUC'] = tcga_extra.get('Overall_AUC', np.nan)
            row['TCGA2_Overall_TCGA_AUPRC'] = tcga_extra.get('Overall_AUPRC', np.nan)
            for drug_name, metrics in tcga_extra.items():
                if drug_name not in ['Overall_AUC', 'Overall_AUPRC']:
                    row[f'TCGA2_{drug_name}_TCGA_AUC'] = metrics.get('AUC', np.nan)
                    row[f'TCGA2_{drug_name}_TCGA_AUPRC'] = metrics.get('AUPRC', np.nan)
        
        comparison_rows.append(row)
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Handle potential NaN values in the data
    comparison_df = comparison_df.fillna(0)  # Replace NaN with 0 to avoid groupby errors
    
    # Group by parameter combination and keep only the row with the best Overall_TCGA_AUC for each group
    param_columns = ['Pretrain_Epochs', 'Train_Epochs', 'Pretrain_LR', 'GAN_LR', 
                     'Dropout_Rate', 'Encoder_Dims', 'Finetune_LR', 'Scheduler']
    
    # Add lambda_cls to param_columns if we're using it
    if use_lambda_cls:
        param_columns.append('Lambda_Cls')
        
    # Add Use_Class_Weight to param_columns if it exists in the DataFrame
    if 'Use_Class_Weight' in comparison_df.columns:
        param_columns.append('Use_Class_Weight')
    
    # More robust way to find the best row in each group
    def get_best_row(group):
        try:
            # Replace NaN with -inf for finding max
            group_copy = group.copy()
            if 'Overall_TCGA_AUC' in group_copy.columns:
                group_copy['Overall_TCGA_AUC'] = group_copy['Overall_TCGA_AUC'].fillna(-float('inf'))
                # Get the index of the max value
                max_idx = group_copy['Overall_TCGA_AUC'].idxmax()
                if pd.isna(max_idx):
                    # Fallback if still NaN
                    return group.iloc[0]
                return group.loc[max_idx]
            else:
                # Fallback if column doesn't exist
                return group.iloc[0]
        except Exception as e:
            print(f"Error in get_best_row: {e}")
            # Return the first row as a fallback
            return group.iloc[0]
    
    try:
        # Try the groupby operation with error handling
        best_comparison_df = comparison_df.groupby(param_columns, as_index=False, dropna=False).apply(get_best_row)
    except Exception as e:
        print(f"Warning: Error during groupby operation: {e}")
        print("Falling back to using all rows without grouping.")
        best_comparison_df = comparison_df
    
    # Sort by Overall TCGA AUC (descending), handling NaN values correctly
    if 'Overall_TCGA_AUC' in best_comparison_df.columns:
        # Fill NaN with negative infinity for sorting purposes
        sort_column = best_comparison_df['Overall_TCGA_AUC'].copy()
        sort_column = sort_column.fillna(-float('inf'))
        best_comparison_df = best_comparison_df.iloc[sort_column.argsort()[::-1]]
    
    # Save the comparison table
    comparison_path = os.path.join(outfolder, 'parameter_comparison_tcga_focus.csv')
    best_comparison_df.to_csv(comparison_path, index=False)
    
    # Create a more detailed version with all results (not just best per parameter set)
    if 'Overall_TCGA_AUC' in comparison_df.columns:
        # Fill NaN with negative infinity for sorting purposes
        sort_column = comparison_df['Overall_TCGA_AUC'].copy()
        sort_column = sort_column.fillna(-float('inf'))
        detailed_df = comparison_df.iloc[sort_column.argsort()[::-1]]
    else:
        detailed_df = comparison_df
    detailed_path = os.path.join(outfolder, 'parameter_comparison_detailed.csv')
    detailed_df.to_csv(detailed_path, index=False)
    
    print(f"\nParameter comparison tables saved to:")
    print(f"1. {comparison_path} (TCGA-focused, best per parameter set)")
    print(f"2. {detailed_path} (detailed, all runs)")
    
    return best_comparison_df

# Helper for JSON serialization of numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

def collate_fn(batch):
    gene_list, drug_list, target_list = zip(*batch)
    return list(gene_list), list(drug_list), list(target_list)

if __name__ == "__main__":
    import argparse
    
    # Start timer
    start_time = time.time()
    
    parser = argparse.ArgumentParser('finetune_zscore_optimized')
    parser.add_argument('--drug_model_path', dest='drug_model_path', type=str, 
                       default='./result/drug_encoder.pth', 
                       help='Pre-trained drug model GCN/GIN path')
    parser.add_argument('--pretrain_model_root_folder', dest='pretrain_model_root_folder', type=str, 
                       default='./result/pretrain_VAE', 
                       help='Root folder containing pre-trained encoder models and model_select.csv')
    parser.add_argument('--outfolder', dest='outfolder', type=str, 
                       default='./result/classify_optimized',
                       help='folder to save result')
    parser.add_argument('--batch_size', dest='batch_size', type=int, 
                       default=2048, 
                       help='batch size for training and validation')
    parser.add_argument('--encoder_type', dest='encoder_type', type=str,
                       choices=['vae', 'ae'], default='vae',
                       help='type of encoder to use (vae or ae)')
    parser.add_argument('--use_lambda_cls', dest='use_lambda_cls', action='store_true',
                       help='Use models trained with lambda_cls parameter (VAEwC)')
    
    parser.add_argument('--response_data', type=str, default='data/GDSC2_fitted_dose_response_MaxScreen_raw.csv',
                        help='Path to the drug response data CSV file.')
    parser.add_argument('--expression_data', type=str, default='data/pretrain_ccle.csv',
                        help='Path to the gene expression data CSV file.')
    parser.add_argument('--config', type=str, default='config/params_grid_complex.json',
                        help='Path to the configuration file with finetune parameter grids.')

    args = parser.parse_args()
    
    # Only load finetune_params from config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        fine_tune_params_grid = config['finetune_params']
        print(f"Loaded finetune parameters from {args.config}")
    except Exception as e:
        print(f"Error loading finetune parameters from config: {e}")
        print("Using default finetune parameters")
        fine_tune_params_grid = DEFAULT_FINE_TUNE_PARAMS_GRID
    
    results = step_1_finetune_pipeline_zscore(
        pretrain_model_root_folder=args.pretrain_model_root_folder,
        drug_model_path=args.drug_model_path,
        outfolder=args.outfolder,
        response_data_path=args.response_data,
        expression_data_path=args.expression_data,
        drug_smiles_data_path=FIXED_DRUG_SMILES_DATA_PATH,
        tcga_inference_data_folder=FIXED_TCGA_DATA_FOLDER,
        tcga_inference_data_folder_extra=FIXED_TCGA_DATA_FOLDER_EXTRA,
        params_grid_dict={},  # Empty dict as we're only using model_select.csv
        fine_tune_params_grid_dict=fine_tune_params_grid,
        batch_size=args.batch_size,
        encoder_type=args.encoder_type,
        use_lambda_cls=args.use_lambda_cls,
        run_config={
            'config_path': args.config,
            'finetune_params': fine_tune_params_grid
        }
    )
    
    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_str = str(timedelta(seconds=int(execution_time)))
    print("\n" + "="*80)
    print(f"Total execution time: {execution_time_str}")
    print("="*80)
