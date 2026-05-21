"""
Step 1: Finetune Latent Pipeline (5-Fold Cross Validation)

This script fine-tunes a drug response prediction model using gene expression latent representations
and drug SMILES graphs. It performs a 5-fold cross-validation strategy.

Usage Example:
--------------
python step1_finetune_latent_pipeline_5fold_split.py \\
    --outfolder ./result/classify_optimized_5fold \\
    --batch_size 2048 \\
    --response_data "data/GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID678 drug230 samples142188 balanced_high.csv" \\
    --model_select_path ./input/GIN_latent/model_select.csv \\
    --config config/finetune_params_grid.json \\
    --epochs 1000
"""
import os
import torch
import numpy as np
import pandas as pd
import json
import time
import pickle
from datetime import timedelta
from collections import defaultdict
from itertools import product
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                            average_precision_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric import data as DATA
from torch_geometric.data import Batch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from tools.model_opt import Classify, init_weights, FocalLoss
from drugmodels.ginconv import GINConvNet
from tools.dataprocess import smile_to_graph, safemakedirs
from tools.drug_finetune_utils import DrugResponseDataset
from tools.inference_utils import inference_on_tcga_drugs, calculate_comprehensive_metrics, plot_confusion_matrix
from tools.prediction_export import (
    aggregate_fold_prediction_dfs,
    collect_ccle_predictions,
    predictions_from_tcga_inference_result,
    save_prediction_tables,
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. No GPU detected.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

FIXED_DRUG_SMILES_DATA_PATH = "data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv"
FIXED_TCGA_DATA_FOLDER = "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv"
FIXED_TCGA_DATA_FOLDER_EXTRA = "data/TCGA/TCGA_drug_response_from_DAPL.csv"

# Function to load parameters from config file
def load_config(config_path='config/params_grid_latent.json'):
    """Load parameters from config file for latent pipeline"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate that required keys exist
        if 'finetune_params' not in config:
            raise ValueError(f"Config file {config_path} must contain 'finetune_params' key")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config file {config_path}: {e}")



# Define constants for model dimensions and training parameters
DRUG_INPUT_DIM = 78
DRUG_EMBED_DIM = 300  # Confirmed dimension for new GIN latent file
ENCODER_LATENT_DIM = 32  # Expression latent dimension

DEFAULT_PATIENCE = 20
# Learning curve display control:
# skip first N epochs on x-axis when plotting.
CURVE_SKIP_INITIAL_EPOCHS = 1

# find_optimal_threshold and calculate_comprehensive_metrics imported from tools.inference_utils

# plot_confusion_matrix imported from tools.inference_utils

# DrugResponseDataset imported from tools.drug_finetune_utils



def train_combination(model_components,
                      train_loader,
                      val_loader,
                      loss_fn,
                      optimizer,
                      scheduler,
                      num_finetune_epochs,
                      patience_limit,
                      model_folder,
                      param_combination_id,
                      model_params):
    """Run training loop for a single parameter combination and persist best model.

    Returns (metrics_history, best_val_auc, best_epoch).
    """
    best_val_auc = float('-inf')
    best_epoch = 0
    patience_counter = 0
    metrics_history = defaultdict(list)

    pbar = tqdm(range(num_finetune_epochs), desc=f"[Combination {param_combination_id}] Training", leave=False)
    for epoch in pbar:
        train_loss, train_auc, train_auprc = train_one_epoch(model_components, train_loader, optimizer, loss_fn, model_params)
        if scheduler:
            scheduler.step()
        val_loss, val_auc, val_auprc = evaluate_model(model_components, val_loader, loss_fn, model_params)

        # Store basic metrics during training
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['train_auc'].append(train_auc)
        metrics_history['val_auc'].append(val_auc)
        metrics_history['train_auprc'].append(train_auprc)
        metrics_history['val_auprc'].append(val_auprc)

        # Update tqdm postfix with current metrics
        pbar.set_postfix({
            'Train_Loss': f'{train_loss:.4f}',
            'Val_Loss': f'{val_loss:.4f}',
            'Train_AUC': f'{train_auc:.4f}',
            'Val_AUC': f'{val_auc:.4f}'
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            # Save model state dict only if model_folder is provided
            if model_folder is not None:
                classifier = model_components['classifier']
                drug_gcnmodel = model_components['drug_model']
                model_state_dict = {
                    'classifier_state_dict': classifier.state_dict(),
                    'epoch': epoch,
                    'best_val_auc': best_val_auc,
                }
                if drug_gcnmodel is not None:
                    model_state_dict['drug_model_state_dict'] = drug_gcnmodel.state_dict()
                torch.save(model_state_dict, os.path.join(model_folder, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                # Early stopping occurred
                break

    print(f"[Combination {param_combination_id}] Training completed. Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch + 1}")
    print("-" * 80)
    return metrics_history, best_val_auc, best_epoch


def test_combination(model_components,
                     test_loader,
                     loss_fn,
                     model_params,
                     param_combination_id):
    """Evaluate trained components on the test loader and print results."""
    test_loss, test_metrics = evaluate_model(model_components, test_loader, loss_fn, model_params, calculate_detailed_metrics=True)
    # Test completed
    return test_loss, test_metrics


def train_one_epoch(model_components, dataloader, optimizer, loss_fn, model_params=None):
    drug_model = model_components['drug_model']
    classifier = model_components['classifier']
    
    if drug_model is not None:
        drug_model.eval()
    classifier.train()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        gene_list, drug_data_list, target_list, weight_list = batch
        batch_gene = torch.stack(gene_list).to(device)
        batch_target = torch.stack(target_list).to(device)
        batch_weights = torch.stack(weight_list).to(device)
        
        optimizer.zero_grad()
        
        # Use graph-based approach
        drug_batch = Batch.from_data_list(drug_data_list).to(device)
        drug_emb = drug_model(drug_batch)
        combined = torch.cat((batch_gene, drug_emb), dim=1)
        
        pred = classifier(combined).view(-1)
        
        # Check for NaN values in predictions
        if torch.isnan(pred).any():
            print("Warning: NaN values detected in predictions. Replacing with zeros.")
            pred = torch.nan_to_num(pred, nan=0.0)
            
        loss = loss_fn(pred, batch_target, weights=batch_weights)
        
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
    
    # Calculate basic metrics during training (AUC and AUPRC only)
    try:
        # Convert logits to probabilities
        all_preds_proba = 1 / (1 + np.exp(-all_preds_np))  # sigmoid
        auc_score = roc_auc_score(all_targets_np, all_preds_proba)
        auprc_score = average_precision_score(all_targets_np, all_preds_proba)
    except ValueError as e:
        print(f"Error calculating metrics: {e}")
        print("Using default metrics (AUC=0.5, AUPRC=class ratio)")
        auc_score = 0.5  # Default AUC (random classifier)
        # Default AUPRC is the proportion of positive samples
        positive_ratio = np.mean(all_targets_np)
        auprc_score = positive_ratio if not np.isnan(positive_ratio) else 0.5
    
    return avg_loss, auc_score, auprc_score

def evaluate_model(model_components, dataloader, loss_fn, model_params=None, calculate_detailed_metrics=False):
    drug_model = model_components['drug_model']
    classifier = model_components['classifier']
    
    if drug_model is not None:
        drug_model.eval()
    classifier.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            gene_list, drug_data_list, target_list, weight_list = batch
            batch_gene = torch.stack(gene_list).to(device)
            batch_target = torch.stack(target_list).to(device)
            batch_weights = torch.stack(weight_list).to(device)
            
            # Use graph-based approach
            drug_batch = Batch.from_data_list(drug_data_list).to(device)
            drug_emb = drug_model(drug_batch)
            combined = torch.cat((batch_gene, drug_emb), dim=1)
            
            pred = classifier(combined).view(-1)
            
            # Handle NaN values in predictions
            if torch.isnan(pred).any():
                pred = torch.nan_to_num(pred, nan=0.0)
                
            loss = loss_fn(pred, batch_target, weights=batch_weights)
            
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
    
    # Calculate metrics based on the flag
    if calculate_detailed_metrics:
        # Calculate comprehensive metrics for final evaluation
        try:
            # Convert logits to probabilities
            all_preds_proba = 1 / (1 + np.exp(-all_preds_np))  # sigmoid
            metrics = calculate_comprehensive_metrics(all_targets_np, all_preds_proba)
        except ValueError as e:
            print(f"Error calculating validation metrics: {e}")
            print("Using default metrics")
            metrics = {
                'AUC': 0.5,
                'AUPRC': np.mean(all_targets_np) if not np.isnan(np.mean(all_targets_np)) else 0.5,
                'sensitivity': 0.0,
                'specificity': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'optimal_threshold': 0.5,
                'youden_index': 0.0
            }
        return avg_loss, metrics
    else:
        # Calculate basic metrics during training (AUC and AUPRC only)
        try:
            # Convert logits to probabilities
            all_preds_proba = 1 / (1 + np.exp(-all_preds_np))  # sigmoid
            auc_score = roc_auc_score(all_targets_np, all_preds_proba)
            auprc_score = average_precision_score(all_targets_np, all_preds_proba)
        except ValueError as e:
            print(f"Error calculating validation metrics: {e}")
            print("Using default metrics (AUC=0.5, AUPRC=class ratio)")
            auc_score = 0.5
            positive_ratio = np.mean(all_targets_np)
            auprc_score = positive_ratio if not np.isnan(positive_ratio) else 0.5
    return avg_loss, auc_score, auprc_score

# inference_on_tcga_drugs imported from tools.inference_utils
import ast

def construct_folder_name(row):
    """
    Construct model folder name from model_select.csv row.
    Supports both VAE format and GNN AE exp_X format.
    Duplicate of logic in Export_latent2dict.py to avoid circular imports.
    """
    if 'ID' in row and str(row['ID']).startswith('exp_'):
        return str(row['ID'])
        
    try:
        pt_epochs = row['pretrain_epochs']
        t_epochs = row['train_epochs']
        pt_lr = row['pretrain_lr']
        t_lr = row['train_lr']
        dop = row['dropout']
        
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
            if '128' in enc_dims_str and '512' not in enc_dims_str:
                enc_suffix = 1
            else:
                enc_suffix = 4
                
        folder_name = f"pt_epochs_{pt_epochs},t_epochs_{t_epochs},Ptlr_{pt_lr},tlr{t_lr},dop{dop},enc{enc_suffix}"
        return folder_name
    except Exception as e:
        if 'ID' in row:
            return str(row['ID'])
        return str(row.get('ID', 'unknown'))


def resolve_model_folder(model_row, model_select_path):
    """Resolve pretrained experiment folder from model_select row/path."""
    base_model_dir = os.path.dirname(model_select_path)
    folder_candidates = []
    for key in ("result_folder", "ID"):
        value = model_row.get(key)
        if value is not None and str(value).strip():
            folder_candidates.append(str(value).strip())
    constructed = construct_folder_name(model_row)
    if constructed not in folder_candidates:
        folder_candidates.append(constructed)

    parent_dir = os.path.dirname(base_model_dir)
    for folder_name in folder_candidates:
        direct = os.path.join(base_model_dir, folder_name)
        if os.path.isdir(direct):
            return direct
        parent = os.path.join(parent_dir, folder_name)
        if os.path.isdir(parent):
            return parent
    return os.path.join(base_model_dir, folder_candidates[0] if folder_candidates else constructed)

def step_1_finetune_pipeline_zscore(
    outfolder,
    response_data_path,
    model_select_path,  # Changed from expression_latent_root
    drug_smiles_data_path,
    tcga_inference_data_folder,
    tcga_inference_data_folder_extra,
    batch_size,

    config=None,
    epochs=1000):
    
    # Load response data
    print(f"Loading response data from {response_data_path}")
    response_df = pd.read_csv(response_data_path)
    
    # Load model selection
    print(f"Loading model selection from {model_select_path}")
    model_select_df = pd.read_csv(model_select_path)
    
    # Drug latent representations are removed (no pretrain functionality)
    drug_latent_dict = None
    
    # Load drug SMILES data
    drug_smiles_df = pd.read_csv(drug_smiles_data_path, index_col=0)[['SMILES']].dropna()
    
    # For latent pipeline, we use fixed training parameters
    train_num_epochs = epochs
    
    # Generate parameter combinations from config
    if config is None:
        raise ValueError("config parameter is required")
    
    fine_tune_params_grid_dict = config['finetune_params']
    ft_param_combinations = [dict(zip(fine_tune_params_grid_dict.keys(), v)) 
                           for v in product(*fine_tune_params_grid_dict.values())]
    
    classifier_params_grid = config['classifier_params']
    classifier_param_combinations = [dict(zip(classifier_params_grid.keys(), v)) 
                                   for v in product(*classifier_params_grid.values())]
    
    model_params_grid = config['model_params']
    
    # Filter out expression_latent_subdir from grid generation if it exists
    model_params_clean = {k:v for k,v in model_params_grid.items() if k != 'expression_latent_subdir'}
    model_param_combinations = [dict(zip(model_params_clean.keys(), v)) 
                               for v in product(*model_params_clean.values())]

    all_param_results_list = []
    safemakedirs(outfolder)
    run_config_snapshot = {
        'response_data_path': response_data_path,
        'model_select_path': model_select_path,
        'drug_smiles_data_path': drug_smiles_data_path,
        'tcga_data_folder': tcga_inference_data_folder,
        'tcga_data_folder_extra': tcga_inference_data_folder_extra,
        'batch_size': batch_size,
        'epochs': epochs,
        'config': config
    }
    with open(os.path.join(outfolder, 'finetune_config_used.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config_snapshot, f, indent=2, ensure_ascii=False, default=str)
    
    # Iterate over selected models
    total_models = len(model_select_df)
    
    # Initialize best metrics tracker
    best_overall_tcga_metrics = {'best_overall_auc': float('-inf'), 'params': None, 'metrics': None}
    
    # Loop over models
    for idx, model_row in model_select_df.iterrows():
        model_id = model_row.get('ID', f'model_{idx}')
        print(f"\n=== Processing Model {idx+1}/{total_models}: {model_id} ===")
        model_output_root = os.path.join(outfolder, str(model_id))
        safemakedirs(model_output_root)
        with open(os.path.join(model_output_root, 'finetune_config_used.json'), 'w', encoding='utf-8') as f:
            json.dump({
                **run_config_snapshot,
                'model_id': model_id,
                'model_select_row': model_row.to_dict()
            }, f, indent=2, ensure_ascii=False, default=str)
        
        # 1. Locate Latents
        model_folder = resolve_model_folder(model_row, model_select_path)
        
        # Look for latents in the model folder
        if not os.path.isdir(model_folder):
            print(f"  Warning: Model folder not found: {model_folder}")
            print("  Skipping this model.")
            continue
        candidates = [f for f in os.listdir(model_folder) if f.endswith('latent_dict.pkl')]
        
        ccle_latent_path = None
        tcga_latent_path = None
        
        for f in candidates:
            lname = f.lower()
            if 'tcga' in lname:
                tcga_latent_path = os.path.join(model_folder, f)
            elif 'ccle' in lname or 'gdsc' in lname or 'train' in lname or 'pretrain' in lname:
                ccle_latent_path = os.path.join(model_folder, f)
        
        if not ccle_latent_path and candidates:
             potential_train = [f for f in candidates if 'tcga' not in f.lower()]
             if potential_train:
                 ccle_latent_path = os.path.join(model_folder, potential_train[0])
        
        if not ccle_latent_path:
            print(f"  Warning: No suitable training latent file found in {model_folder}. Found: {candidates}")
            print("  Skipping this model.")
            continue
            
        print(f"  Loading training latent from: {ccle_latent_path}")
        with open(ccle_latent_path, 'rb') as f:
            expression_latent_dict = pickle.load(f)
        
        # Detect actual latent dimension from loaded dict
        if expression_latent_dict:
            sample_key = next(iter(expression_latent_dict))
            val = expression_latent_dict[sample_key]
            detected_dim = len(val) if hasattr(val, '__len__') else val.shape[0]
            
            global ENCODER_LATENT_DIM
            ENCODER_LATENT_DIM = detected_dim
            print(f"  Detected encoder latent dim: {ENCODER_LATENT_DIM}")
            
        tcga_latent_dict = None
        if tcga_latent_path:
            print(f"  Loading TCGA latent from: {tcga_latent_path}")
            with open(tcga_latent_path, 'rb') as f:
                tcga_latent_dict = pickle.load(f)
        else:
            print("  Warning: No TCGA latent file found. TCGA evaluation will be limited.")

        # 2. Iterate Hyperparameters
        param_combination_id = 0
        total_combinations = len(ft_param_combinations) * len(classifier_param_combinations) * len(model_param_combinations)
        
        for ft_params in ft_param_combinations:
            for classifier_params in classifier_param_combinations:
                for model_params in model_param_combinations:
                    param_combination_id += 1
                    current_hyperparams = {**ft_params, **classifier_params, **model_params}
                    current_hyperparams['Model_ID'] = model_id
                    
                    print(f"  [Param {param_combination_id}/{total_combinations}] Training params: {current_hyperparams}")
                
                    
                    # Identify drug column
                    drug_col = 'DRUG_ID' # Default fallback
                    if 'drug_name' in response_df.columns: drug_col = 'drug_name'
                    elif 'DRUG_NAME' in response_df.columns: drug_col = 'DRUG_NAME'

                    # Check for missing drug SMILES (keep raise error)
                    valid_indices = set(str(k).lower() for k in drug_smiles_df.index)
                    if 'DRUG_NAME' in drug_smiles_df.columns:
                        valid_indices.update(str(k).lower() for k in drug_smiles_df['DRUG_NAME'])
                    
                    response_drugs = set(response_df[drug_col].astype(str).str.lower())
                    missing_drugs = response_drugs - valid_indices
                    if missing_drugs:
                        raise ValueError(f"Missing SMILES structure for {len(missing_drugs)} drugs. Examples: {list(missing_drugs)[:5]}")

                    # Check for missing expression latent representations:
                    # Instead of raising an error, print the missing IDs and skip those rows.
                    missing_samples = set(response_df['ModelID'].astype(str)) - set(str(k) for k in expression_latent_dict.keys())
                    if missing_samples:
                        print(f"  [WARNING] {len(missing_samples)} ModelID(s) have no latent representation and will be skipped:")
                        print(f"    Missing latent IDs: {sorted(missing_samples)}")
                    
                    # Filter out rows whose ModelID has no latent representation
                    current_response_df = response_df[~response_df['ModelID'].astype(str).isin(missing_samples)]
                    
                    if len(current_response_df) == 0:
                        print("  [WARNING] No rows remain after filtering missing latent IDs. Skipping this combination.")
                        continue
                    
                    # Split data: 10% for testing, 90% for 5-fold CV
                    all_samples = current_response_df['ModelID'].unique()
                    all_labels = []
                    for sample_id in all_samples:
                        sample_data = current_response_df[current_response_df['ModelID'] == sample_id]
                        label = sample_data['Label'].iloc[0]
                        all_labels.append(label)
                    
                    # First split: 10% for testing, 90% for CV
                    from sklearn.model_selection import train_test_split
                    cv_samples, test_samples = train_test_split(all_samples, test_size=0.1, random_state=42, stratify=all_labels)
                    
                    # Create datasets
                    cv_df = current_response_df[current_response_df['ModelID'].isin(cv_samples)]
                    test_df = current_response_df[current_response_df['ModelID'].isin(test_samples)]
                    
                    print(f"  Data split: {len(cv_samples)} samples for CV, {len(test_samples)} samples for testing")
                    
                    # Create model folder for this parameter combination
                    # Note: Need to make it unique per model ID!
                    model_exp_folder = os.path.join(outfolder, str(model_id), f"param_{param_combination_id:03d}")
                    safemakedirs(model_exp_folder)
                    with open(os.path.join(model_exp_folder, 'params_used.json'), 'w', encoding='utf-8') as f:
                        json.dump(current_hyperparams, f, indent=2, ensure_ascii=False, default=str)
                    
                    # Perform 5-fold cross validation
                    fold_results, mean_metrics = perform_5fold_cross_validation(
                        response_df=cv_df,  # Use CV data (90% of data)
                        expression_latent_dict=expression_latent_dict,
                        drug_latent_dict=drug_latent_dict,
                        drug_smiles_df=drug_smiles_df,
                        model_params=model_params,
                        ft_params=ft_params,
                        classifier_params=classifier_params,
                        batch_size=batch_size,
                        num_epochs=train_num_epochs,
                        patience_limit=DEFAULT_PATIENCE,
                        param_combination_id=f"{model_id}_p{param_combination_id}",
                        model_folder=model_exp_folder,
                        tcga_inference_data_folder=tcga_inference_data_folder,
                        tcga_inference_data_folder_extra=tcga_inference_data_folder_extra,
                        tcga_latent_dict=tcga_latent_dict,
                        test_df=test_df  # Pass test data for final evaluation
                    )
                    
                    # Save fold results
                    fold_results_folder = os.path.join(model_exp_folder, 'fold_results')
                    safemakedirs(fold_results_folder)
                
                    # Save detailed fold results
                    with open(os.path.join(fold_results_folder, 'fold_results.json'), 'w') as f:
                        json.dump(fold_results, f, indent=4, cls=NpEncoder)
                    
                    # Save mean metrics
                    with open(os.path.join(fold_results_folder, 'mean_metrics.json'), 'w') as f:
                        json.dump(mean_metrics, f, indent=4, cls=NpEncoder)
                    
                    # Get best fold information from mean_metrics
                    best_fold = mean_metrics['best_fold']
                    best_fold_idx = best_fold['fold_number'] - 1  # Convert to 0-based index
                    
                    # Use best fold's TCGA results (already computed during 5-fold CV)
                    tcga_results = best_fold['tcga_results']
                    
                    # Store results for this parameter combination
                    current_param_set_results = {
                        'Params': current_hyperparams,
                        'Model_ID': model_id,
                        'Fold_Results': fold_results,
                        'Mean_Metrics': mean_metrics,
                        'TCGA_Metrics': tcga_results,
                        'Best_Fold': best_fold_idx + 1,
                        'Best_Fold_AUC': best_fold['best_val_auc']
                    }
                    
                    all_param_results_list.append(current_param_set_results)
                    
                    # Save metrics for current parameter set
                    save_single_param_set_metrics_5fold(model_exp_folder, current_param_set_results)
                    # Metrics saved automatically
                    
                    # Print summary for this combination
                    print(f"[Combination {param_combination_id}] Completed - Mean CV AUC: {mean_metrics['val_AUC_mean']:.4f} ± {mean_metrics['val_AUC_std']:.4f}")
                    print("=" * 80)
                
                    # Update best overall TCGA metrics
                    # Use Global TCGA AUC
                    current_tcga_auc = tcga_results.get('Global_TCGA_AUC', np.nan)
                
                    if not pd.isna(current_tcga_auc) and current_tcga_auc > best_overall_tcga_metrics['best_overall_auc']:
                        best_overall_tcga_metrics['best_overall_auc'] = current_tcga_auc
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
    comparison_df = create_final_parameter_comparison_csv_5fold(outfolder, all_param_results_list, config)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    if best_overall_tcga_metrics['params']:
        print(f"\nBest combination found - Mean CV AUC: {best_overall_tcga_metrics['metrics']['Mean_Metrics']['val_AUC_mean']:.4f} ± {best_overall_tcga_metrics['metrics']['Mean_Metrics']['val_AUC_std']:.4f}")
        print(f"Overall TCGA Global AUC: {best_overall_tcga_metrics['metrics']['TCGA_Metrics'].get('Global_TCGA_AUC', 'N/A')}")
    else:
        print("No results found to determine the best parameters.")
    
    print(f"\nResults saved to: {outfolder}")
    print("="*80)
    
    return best_overall_tcga_metrics

def plot_learning_curves(metrics_dict, save_path):
    """
    Plot and save learning curves for training and validation metrics
    
    Args:
        metrics_dict: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(18, 5))
    start_idx = max(0, int(CURVE_SKIP_INITIAL_EPOCHS))
    total_epochs = len(metrics_dict['train_loss'])
    if start_idx >= total_epochs:
        start_idx = 0
    epochs = range(start_idx + 1, total_epochs + 1)
    train_loss = metrics_dict['train_loss'][start_idx:]
    val_loss = metrics_dict['val_loss'][start_idx:]
    train_auc = metrics_dict['train_auc'][start_idx:]
    val_auc = metrics_dict['val_auc'][start_idx:]
    train_auprc = metrics_dict['train_auprc'][start_idx:]
    val_auprc = metrics_dict['val_auprc'][start_idx:]

    plt.subplot(131)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(132)
    plt.plot(epochs, train_auc, label='Train AUC')
    plt.plot(epochs, val_auc, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(133)
    plt.plot(epochs, train_auprc, label='Train AUPRC')
    plt.plot(epochs, val_auprc, label='Val AUPRC')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.title('Training and Validation AUPRC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_single_param_set_metrics_5fold(model_folder, param_set_results):
    """Save metrics for a single parameter combination with 5-fold cross validation results"""
    metrics_out_folder = os.path.join(model_folder, 'metrics_summary')
    safemakedirs(metrics_out_folder)
    
    # Save detailed metrics
    metrics_data = {
        # Parameters
        **param_set_results['Params'],
        
        # 5-fold cross validation results - only fold number
        'Best_Fold': param_set_results['Best_Fold'],
    }
    
    # Add only mean metrics from cross validation (excluding std)
    mean_metrics = param_set_results['Mean_Metrics']
    for key, value in mean_metrics.items():
        # Only include mean values, skip std values and the full best_fold dict
        if not key.endswith('_std') and key != 'best_fold':
            metrics_data[key] = value
    
    # Add TCGA metrics (flattened)
    tcga_metrics = param_set_results.get('TCGA_Metrics', {})
    for key, value in tcga_metrics.items():
        metrics_data[key] = value
    
    # Save the metrics
    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(os.path.join(metrics_out_folder, 'metrics_summary.csv'), index=False)


def create_final_parameter_comparison_csv_5fold(outfolder, all_param_results_list, config=None):
    """Create CSV files comparing all hyperparameter combinations with 5-fold cross validation results"""
    if not all_param_results_list:
        print("No data to create final parameter comparison CSV.")
        return
    
    # Create a streamlined comparison table focusing on parameters and results
    comparison_rows = []
    
    for i, result in enumerate(all_param_results_list):
        # Extract key parameters and metrics
        if 'Model_ID' in result:
            row = {'Model_ID': result['Model_ID']}
            row['Run_ID'] = i + 1
        else:
            row = {'ID': i + 1}
        
        # Add expression latent subdirectory
        if 'Expression_Latent_Subdir' in result:
            row['Expression_Latent_Subdir'] = result['Expression_Latent_Subdir']
        
        # Dynamically add parameters based on config structure
        if config:
            # Add finetune parameters
            if 'finetune_params' in config:
                for param_name in config['finetune_params'].keys():
                    if param_name in result['Params']:
                        row[f'Finetune_{param_name.upper()}'] = result['Params'][param_name]
            
            # Add classifier parameters
            if 'classifier_params' in config:
                for param_name in config['classifier_params'].keys():
                    if param_name in result['Params']:
                        # Convert list parameters to string for CSV compatibility
                        if isinstance(result['Params'][param_name], list):
                            row[f'Classifier_{param_name.upper()}'] = str(result['Params'][param_name])
                        else:
                            row[f'Classifier_{param_name.upper()}'] = result['Params'][param_name]
            
            # Add model parameters
            if 'model_params' in config:
                for param_name in config['model_params'].keys():
                    if param_name in result['Params']:
                        row[f'Model_{param_name.upper()}'] = result['Params'][param_name]
        else:
            # Fallback to hardcoded parameters if no config provided
            row['Finetune_LR'] = result['Params']['ftlr']
            row['Scheduler'] = result['Params']['scheduler_flag']
            row['GIN_Type'] = result['Params']['gin_type']
            row['Classifier_Hidden_Dims'] = str(result['Params']['hidden_dims'])
            row['Classifier_Dropout'] = result['Params']['dropout_rate']
            row['Classifier_BatchNorm'] = result['Params']['use_batch_norm']
            row['Classifier_Activation'] = result['Params']['activation']
        
        # 5-fold cross validation metrics (mean ± std)
        mean_metrics = result['Mean_Metrics']
        row['CV_Val_AUC_Mean'] = mean_metrics['val_AUC_mean']
        row['CV_Val_AUC_Std'] = mean_metrics['val_AUC_std']
        row['CV_Test_AUC_Mean'] = mean_metrics['test_AUC_mean']
        row['CV_Test_AUC_Std'] = mean_metrics['test_AUC_std']
        row['CV_Val_AUPRC_Mean'] = mean_metrics['val_AUPRC_mean']
        row['CV_Val_AUPRC_Std'] = mean_metrics['val_AUPRC_std']
        row['CV_Test_AUPRC_Mean'] = mean_metrics['test_AUPRC_mean']
        row['CV_Test_AUPRC_Std'] = mean_metrics['test_AUPRC_std']
        
        # Best fold information
        row['Best_Fold'] = result['Best_Fold']
        row['Best_Fold_AUC'] = result['Best_Fold_AUC']
        
        # Add TCGA metrics (flattened)
        if 'TCGA_Metrics' in result:
            for key, value in result['TCGA_Metrics'].items():
                row[key] = value
        
        comparison_rows.append(row)
    
    # Convert to DataFrame
    df_all = pd.DataFrame(comparison_rows)
    
    # Handle potential NaN values in the data
    df_all = df_all.fillna(0)
    
    # --- Create Detailed Table ---
    # Include: Parameters + CCLE + TCGA Global/Avg
    # Exclude: Individual drug columns
    individual_drug_cols = [
        c for c in df_all.columns
        if '_TCGA_' in c
        and not c.startswith('Global_')
        and not c.startswith('Average_')
        and not c.startswith('TCGA2_Global_')
        and not c.startswith('TCGA2_Average_')
    ]
    df_detailed = df_all.drop(columns=individual_drug_cols)
    detailed_path = os.path.join(outfolder, 'parameter_comparison_detailed.csv')
    df_detailed.to_csv(detailed_path, index=False)
    
    # --- Create TCGA Focused Table ---
    # Include: CCLE + TCGA Global/Avg + Individual Drugs
    # Exclude: Detailed Parameters (keep IDs, keep important CV metrics)
    # Identify parameter columns to drop
    # Heuristic: Columns starting with Finetune_, Classifier_, Model_
    param_cols = [c for c in df_all.columns if (c.startswith('Finetune_') or c.startswith('Classifier_') or c.startswith('Model_')) and c != 'Model_ID']
    df_focus = df_all.drop(columns=param_cols)
    
    focus_path = os.path.join(outfolder, 'parameter_comparison_tcga_focus.csv')
    df_focus.to_csv(focus_path, index=False)
    
    print(f"\nParameter comparison tables saved to:")
    print(f"1. {focus_path} (TCGA-focused, with individual scores)")
    print(f"2. {detailed_path} (detailed, no individual scores)")
        
    return df_focus

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
    gene_list, drug_list, target_list, weight_list = zip(*batch)
    return list(gene_list), list(drug_list), list(target_list), list(weight_list)

def perform_5fold_cross_validation(response_df,
                                  expression_latent_dict,
                                  drug_latent_dict,
                                  drug_smiles_df,
                                  model_params,
                                  ft_params,
                                  classifier_params,
                                  batch_size,
                                  num_epochs,
                                  patience_limit,
                                  param_combination_id,
                                  model_folder,
                                  tcga_inference_data_folder,
                                  tcga_inference_data_folder_extra,
                                  tcga_latent_dict,
                                  test_df=None):  # Add test_df parameter
    """
    Perform 5-fold cross validation for a given parameter combination
    
    Returns:
        fold_results: List of dictionaries containing results for each fold
        mean_metrics: Dictionary containing mean and std of all metrics across folds
    """
    # Create 5-fold stratified split
    all_samples = response_df['ModelID'].unique()
    all_labels = []
    
    # Get labels for stratification
    for sample_id in all_samples:
        sample_data = response_df[response_df['ModelID'] == sample_id]
        label = sample_data['Label'].iloc[0]  # Assuming all samples for same ModelID have same label
        all_labels.append(label)
    
    # Create stratified k-fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    all_ccle_pred_dfs = []
    all_tcga_pred_dfs = []
    all_tcga_extra_pred_dfs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, all_labels)):
        # Starting fold
        
        # Create fold-specific model folder
        fold_model_folder = os.path.join(model_folder, f'fold_{fold+1}')
        safemakedirs(fold_model_folder)
        
        # Get train and validation samples
        train_samples = all_samples[train_idx]
        val_samples = all_samples[val_idx]
        
        # Create datasets
        train_df = response_df[response_df['ModelID'].isin(train_samples)]
        val_df = response_df[response_df['ModelID'].isin(val_samples)]
        
        train_dataset = DrugResponseDataset(train_df, expression_latent_dict, drug_latent_dict, drug_smiles_df, model_params['gin_type'])
        val_dataset = DrugResponseDataset(val_df, expression_latent_dict, drug_latent_dict, drug_smiles_df, model_params['gin_type'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Initialize models for this fold
        if model_params['gin_type'] == 'dapl':
            drug_gcnmodel = GINConvNet(
                input_dim=DRUG_INPUT_DIM,
                output_dim=DRUG_EMBED_DIM,
                dropout=0.2,
                num_layers=5,
                jk_mode='last',
                use_batch_norm=True,
                pool_type='max'
            ).to(device)
        elif model_params['gin_type'] == 'drpreter':
            drug_gcnmodel = GINConvNet(
                input_dim=DRUG_INPUT_DIM,
                output_dim=DRUG_EMBED_DIM,
                dropout=0.2,
                num_layers=5,
                jk_mode='cat',
                use_batch_norm=True,
                pool_type='max'
            ).to(device)
        elif model_params['gin_type'] == 'ginpre':
            drug_gcnmodel = GINConvNet(
                input_dim=DRUG_INPUT_DIM,
                output_dim=DRUG_EMBED_DIM,
                dropout=0.0,
                num_layers=5,
                jk_mode='sum',
                use_batch_norm=False,
                pool_type='mean'
            ).to(device)
        else:
            raise ValueError(f"Unknown gin_type: {model_params['gin_type']}")
        
        if drug_gcnmodel is not None:
            drug_gcnmodel.apply(init_weights)
            classifier_input_dim = ENCODER_LATENT_DIM + DRUG_EMBED_DIM
        
        # Get activation function
        activation_map = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU
        }
        act_fn = activation_map[classifier_params['activation']]
        
        classifymodel = Classify(input_dim=classifier_input_dim,
                                hidden_dims=classifier_params['hidden_dims'],
                                dop=classifier_params['dropout_rate'], 
                                act_fn=act_fn,
                                out_fn=None,
                                use_bn=classifier_params['use_batch_norm']).to(device)
        classifymodel.apply(init_weights)
        
        model_components_fold = {
            'drug_model': drug_gcnmodel,
            'classifier': classifymodel
        }
        
        # Create optimizer and loss function
        if drug_gcnmodel is not None:
            optimizer = optim.AdamW([
                {'params': classifymodel.parameters()},
                {'params': drug_gcnmodel.parameters()}
            ], lr=ft_params['ftlr'], weight_decay=1e-5)
        else:
            optimizer = optim.AdamW(classifymodel.parameters(), lr=ft_params['ftlr'], weight_decay=1e-5)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs) if ft_params['scheduler_flag'] else None
        # Configure Loss Function
        loss_type = ft_params.get('loss_type', 'bce')
        gamma = ft_params.get('focal_loss_gamma', 2.0)
        
        if loss_type == 'focal':
            loss_fn = FocalLoss(gamma=gamma, reduction='mean').to(device)
        else:
            # Use FocalLoss with gamma=0 to approximate Weighted BCE
            loss_fn = FocalLoss(gamma=0, reduction='mean').to(device)
        
        # Train the model
        metrics_history, best_val_auc, best_epoch = train_combination(
            model_components=model_components_fold,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            num_finetune_epochs=num_epochs,
            patience_limit=patience_limit,
            model_folder=fold_model_folder,  # Save best model for this fold
            param_combination_id=f"{param_combination_id}_fold_{fold+1}",
            model_params=model_params
        )
        
        # Load best model for testing and inference
        best_model_path = os.path.join(fold_model_folder, 'best_model.pth')
        
        # Load best model weights for this fold
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # Reload the weights to the existing models to get best state
        if drug_gcnmodel is not None:
            drug_gcnmodel.load_state_dict(checkpoint['drug_model_state_dict'])
        classifymodel.load_state_dict(checkpoint['classifier_state_dict'])
        
        # Models now have best weights - use for all evaluations
        
        # Evaluate on validation set using best model with detailed metrics
        val_loss, val_metrics = evaluate_model(model_components_fold, val_loader, loss_fn, model_params, calculate_detailed_metrics=True)
        
        # Evaluate on 10% test set for this fold
        if test_df is not None:
            test_dataset = DrugResponseDataset(test_df, expression_latent_dict, drug_latent_dict, drug_smiles_df, model_params['gin_type'])
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            test_loss, test_metrics = test_combination(
                model_components=model_components_fold,
                test_loader=test_loader,
                loss_fn=loss_fn,
                model_params=model_params,
                param_combination_id=f"{param_combination_id}_fold_{fold+1}_test"
            )
            
            # Save test set confusion matrix for this fold
            test_confusion_matrix_path = os.path.join(fold_model_folder, 'test_confusion_matrix.png')
            plot_confusion_matrix(test_metrics['confusion_matrix'], test_confusion_matrix_path, f"Test Set Confusion Matrix - Fold {fold+1}")

            fold_ccle_pred_df = collect_ccle_predictions(
                model_components=model_components_fold,
                dataset=test_dataset,
                model_params=model_params,
                domain="CCLE",
                fold_id=fold + 1,
                batch_size=batch_size,
                collate_fn=collate_fn,
            )
            all_ccle_pred_dfs.append(fold_ccle_pred_df)
        else:
            test_loss = np.nan
            test_metrics = {'AUC': np.nan, 'AUPRC': np.nan, 'sensitivity': np.nan, 'specificity': np.nan,
                          'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan, 'optimal_threshold': np.nan,
                          'confusion_matrix': None}
            fold_ccle_pred_df = pd.DataFrame()
        
        # TCGA Inference for this fold (weights already loaded, pass None to avoid reloading)
        # TCGA Inference for this fold
        tcga_raw_results = inference_on_tcga_drugs(
            model_components=model_components_fold,
            tcga_data_path=tcga_inference_data_folder, # updated arg name
            best_model_path=None,  # Pass None to avoid reloading weights
            ft_params=ft_params,
            tcga_latent_dict=tcga_latent_dict,
            drug_latent_dict=drug_latent_dict,
            gin_type=model_params['gin_type'],
            fold_model_folder=fold_model_folder,
            drug_smiles_df=drug_smiles_df,
            tcga_tag='TCGA1'
        )
        tcga_raw_results_extra = {}
        if tcga_inference_data_folder_extra:
            tcga_raw_results_extra = inference_on_tcga_drugs(
                model_components=model_components_fold,
                tcga_data_path=tcga_inference_data_folder_extra,
                best_model_path=None,
                ft_params=ft_params,
                tcga_latent_dict=tcga_latent_dict,
                drug_latent_dict=drug_latent_dict,
                gin_type=model_params['gin_type'],
                fold_model_folder=fold_model_folder,
                drug_smiles_df=drug_smiles_df,
                tcga_tag='TCGA2'
            )
        
        if not tcga_raw_results:
            tcga_raw_results = {
                'Global_Metrics': {},
                'Average_Metrics': {},
                'Drug_Metrics': {}
            }
        if not tcga_raw_results_extra:
            tcga_raw_results_extra = {
                'Global_Metrics': {},
                'Average_Metrics': {},
                'Drug_Metrics': {}
            }
        
        # Flatten TCGA results for storage and aggregation
        tcga_results = {}
        # Global
        tcga_results['Global_TCGA_AUC'] = tcga_raw_results['Global_Metrics'].get('AUC', np.nan)
        tcga_results['Global_TCGA_AUPRC'] = tcga_raw_results['Global_Metrics'].get('AUPRC', np.nan)
        # Average
        tcga_results['Average_TCGA_AUC'] = tcga_raw_results['Average_Metrics'].get('AUC', np.nan)
        tcga_results['Average_TCGA_AUPRC'] = tcga_raw_results['Average_Metrics'].get('AUPRC', np.nan)
        # Individual Drugs
        for drug, mets in tcga_raw_results['Drug_Metrics'].items():
            tcga_results[f'{drug}_TCGA_AUC'] = mets['AUC']
            tcga_results[f'{drug}_TCGA_AUPRC'] = mets['AUPRC']
        tcga_results['TCGA2_Global_TCGA_AUC'] = tcga_raw_results_extra['Global_Metrics'].get('AUC', np.nan)
        tcga_results['TCGA2_Global_TCGA_AUPRC'] = tcga_raw_results_extra['Global_Metrics'].get('AUPRC', np.nan)
        tcga_results['TCGA2_Average_TCGA_AUC'] = tcga_raw_results_extra['Average_Metrics'].get('AUC', np.nan)
        tcga_results['TCGA2_Average_TCGA_AUPRC'] = tcga_raw_results_extra['Average_Metrics'].get('AUPRC', np.nan)
        for drug, mets in tcga_raw_results_extra['Drug_Metrics'].items():
            tcga_results[f'TCGA2_{drug}_TCGA_AUC'] = mets.get('AUC', np.nan)
            tcga_results[f'TCGA2_{drug}_TCGA_AUPRC'] = mets.get('AUPRC', np.nan)

        fold_tcga_pred_df = predictions_from_tcga_inference_result(tcga_raw_results, tcga_source="TCGA1")
        fold_tcga_pred_df["fold"] = fold + 1
        fold_tcga_extra_pred_df = predictions_from_tcga_inference_result(tcga_raw_results_extra, tcga_source="TCGA2")
        if not fold_tcga_extra_pred_df.empty:
            fold_tcga_extra_pred_df["fold"] = fold + 1
        all_tcga_pred_dfs.append(fold_tcga_pred_df)
        all_tcga_extra_pred_dfs.append(fold_tcga_extra_pred_df)
        save_prediction_tables(
            fold_model_folder,
            ccle_test_df=fold_ccle_pred_df,
            tcga_eval_df=fold_tcga_pred_df,
            tcga_eval_extra_df=fold_tcga_extra_pred_df,
        )
        
        # Save confusion matrix for validation set
        confusion_matrix_path = os.path.join(fold_model_folder, 'val_confusion_matrix.png')
        plot_confusion_matrix(val_metrics['confusion_matrix'], confusion_matrix_path, f"Confusion Matrix - Fold {fold+1}")
        
        # Save learning curves for this fold
        learning_curves_path = os.path.join(fold_model_folder, 'learning_curves.png')
        plot_learning_curves(metrics_history, learning_curves_path)
        
        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'best_epoch': best_epoch + 1,
            'best_val_auc': best_val_auc,
            'final_val_loss': val_loss,
            'final_val_metrics': val_metrics,
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'tcga_results': tcga_results,
            'metrics_history': metrics_history,
            'best_model_path': best_model_path
        }
        
        fold_results.append(fold_result)
        
        # Fold completed
    
    # Aggregate per-sample predictions across folds (C_prototypical style)
    save_prediction_tables(
        model_folder,
        ccle_test_df=aggregate_fold_prediction_dfs(all_ccle_pred_dfs),
        tcga_eval_df=aggregate_fold_prediction_dfs(all_tcga_pred_dfs),
        tcga_eval_extra_df=aggregate_fold_prediction_dfs(all_tcga_extra_pred_dfs),
    )

    # Calculate mean and std across folds
    mean_metrics = calculate_fold_statistics(fold_results)
    
    return fold_results, mean_metrics

def calculate_fold_statistics(fold_results):
    """
    Calculate mean and standard deviation of metrics across folds and find best fold
    
    Args:
        fold_results: List of fold result dictionaries
        
    Returns:
        mean_metrics: Dictionary with mean and std for each metric and best fold info
    """
    # Validation metrics
    val_metrics_to_aggregate = ['AUC', 'AUPRC', 'sensitivity', 'specificity', 'precision', 'recall', 'f1_score']
    
    mean_metrics = {}
    
    for metric in val_metrics_to_aggregate:
        values = [fold['final_val_metrics'][metric] for fold in fold_results]
        mean_metrics[f'val_{metric}_mean'] = np.mean(values)
        mean_metrics[f'val_{metric}_std'] = np.std(values)
    
    # Test metrics
    test_metrics_to_aggregate = ['AUC', 'AUPRC', 'sensitivity', 'specificity', 'precision', 'recall', 'f1_score']
    
    for metric in test_metrics_to_aggregate:
        values = [fold['test_metrics'][metric] for fold in fold_results]
        mean_metrics[f'test_{metric}_mean'] = np.mean(values)
        mean_metrics[f'test_{metric}_std'] = np.std(values)
    
    # TCGA metrics
    # TCGA metrics
    # We now look for keys in the flattened tcga_results
    # Collect all keys present in the first fold's tcga_results to know what to aggregate
    if len(fold_results) > 0 and 'tcga_results' in fold_results[0]:
        tcga_keys = fold_results[0]['tcga_results'].keys()
        for key in tcga_keys:
            values = [fold['tcga_results'][key] for fold in fold_results if key in fold['tcga_results']]
            if values:
                mean_metrics[f'{key}_mean'] = np.mean(values)
                mean_metrics[f'{key}_std'] = np.std(values)
            else:
                mean_metrics[f'{key}_mean'] = np.nan
                mean_metrics[f'{key}_std'] = np.nan
    
    # Add other aggregated metrics
    best_epochs = [fold['best_epoch'] for fold in fold_results]
    best_val_aucs = [fold['best_val_auc'] for fold in fold_results]
    final_val_losses = [fold['final_val_loss'] for fold in fold_results]
    test_losses = [fold['test_loss'] for fold in fold_results]
    
    mean_metrics['best_epoch_mean'] = np.mean(best_epochs)
    mean_metrics['best_epoch_std'] = np.std(best_epochs)
    mean_metrics['best_val_auc_mean'] = np.mean(best_val_aucs)
    mean_metrics['best_val_auc_std'] = np.std(best_val_aucs)
    mean_metrics['final_val_loss_mean'] = np.mean(final_val_losses)
    mean_metrics['final_val_loss_std'] = np.std(final_val_losses)
    mean_metrics['test_loss_mean'] = np.mean(test_losses)
    mean_metrics['test_loss_std'] = np.std(test_losses)
    
    # Find best fold based on validation AUC
    best_fold_idx = np.argmax(best_val_aucs)
    best_fold = fold_results[best_fold_idx]
    
    mean_metrics['best_fold'] = {
        'fold_number': best_fold['fold'],
        'best_val_auc': best_fold['best_val_auc'],
        'best_model_path': best_fold['best_model_path'],
        'val_metrics': best_fold['final_val_metrics'],
        'test_metrics': best_fold['test_metrics'],
        'tcga_results': best_fold['tcga_results']
    }
    
    return mean_metrics

# TCGA_target_data imported is not needed as it's handled by inference_utils

if __name__ == "__main__":
    import argparse
    
    # Start timer
    start_time = time.time()
    
    parser = argparse.ArgumentParser('finetune_zscore_optimized')

    parser.add_argument('--outfolder', dest='outfolder', type=str, 
                       default='./result/classify_optimized',
                       help='folder to save result')
    parser.add_argument('--batch_size', dest='batch_size', type=int, 
                       default=2048, 
                       help='batch size for training and validation')
                       
    parser.add_argument('--response_data', type=str, default='data/GDSC2_fitted_dose_response_MaxScreen_raw.csv',
                        help='Path to the drug response data CSV file.')
    parser.add_argument('--model_select_path', type=str, required=True,
                        help='Path to model_select.csv for batch processing')
    parser.add_argument('--config', type=str, default='config/finetune_params_grid.json',
                        help='Path to the configuration file with finetune parameter grids.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs.')


    args = parser.parse_args()
    
    # Load config parameters
    config = load_config(args.config)
    
    
    # Process all parameter combinations
    results = step_1_finetune_pipeline_zscore(
        outfolder=args.outfolder,
        response_data_path=args.response_data,
        model_select_path=args.model_select_path,
        drug_smiles_data_path=FIXED_DRUG_SMILES_DATA_PATH,
        tcga_inference_data_folder=FIXED_TCGA_DATA_FOLDER,
        tcga_inference_data_folder_extra=FIXED_TCGA_DATA_FOLDER_EXTRA,
        batch_size=args.batch_size,

        config=config,
        epochs=args.epochs
    )
    
    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_str = str(timedelta(seconds=int(execution_time)))
    print("\n" + "="*80)
    print(f"Total execution time: {execution_time_str}")
    print("="*80)
