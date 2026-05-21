"""
Step 1: Finetune Latent Pipeline (Train/Val/Test Split)

This script fine-tunes a drug response prediction model using gene expression latent representations
and drug SMILES graphs. It uses a single splits strategy (Train/Val/Test).

Usage Example:
--------------
python step1_finetune_latent_pipeline_All_split.py \\
    --outfolder ./result/classify_optimized \\
    --batch_size 2048 \\
    --mini_batch_size 512 \\
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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric import data as DATA
from torch_geometric.data import Batch
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools.model_opt import Classify, init_weights, FocalLoss # Assuming these are in a 'tools' directory
from drugmodels.ginconv import GINConvNet # Assuming this is in a 'drugmodels' directory
from tools.dataprocess import smile_to_graph, safemakedirs # Using safemakedirs from tools.dataprocess
from tools.drug_finetune_utils import DrugResponseDataset
from tools.inference_utils import inference_on_tcga_drugs, calculate_comprehensive_metrics, plot_confusion_matrix
from tools.prediction_export import (
    collect_ccle_predictions,
    predictions_from_tcga_inference_result,
    save_prediction_tables,
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. No GPU detected.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


import ast

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
        return str(row.get('ID', 'unknown'))


def resolve_model_folder(model_row, model_select_path):
    """Resolve pretrained experiment folder from model_select row/path.

    Supports both:
    - model_select placed directly under result root
    - model_select placed under report subfolder (e.g. 00_report/model_select.csv)
    """
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

import ast

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
        return str(row.get('ID', 'unknown'))


# Define constants for model dimensions and training parameters
DRUG_INPUT_DIM = 78
DRUG_EMBED_DIM = 300  # Confirmed dimension for new GIN latent file
ENCODER_LATENT_DIM = 32  # Expression latent dimension

DEFAULT_PATIENCE = 20
# Learning curve display control:
# skip first N epochs on x-axis when plotting.
CURVE_SKIP_INITIAL_EPOCHS = 1

# DrugResponseDataset imported from tools.drug_finetune_utils



def plot_learning_curves(metrics_history, save_path, title_prefix=''):
    """Plot loss and AUC learning curves (train vs validation)."""
    start_idx = max(0, int(CURVE_SKIP_INITIAL_EPOCHS))
    total_epochs = len(metrics_history['train_loss'])
    if start_idx >= total_epochs:
        start_idx = 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(start_idx + 1, total_epochs + 1)
    train_loss = metrics_history['train_loss'][start_idx:]
    val_loss = metrics_history['val_loss'][start_idx:]
    train_auc = metrics_history['train_auc'][start_idx:]
    val_auc = metrics_history['val_auc'][start_idx:]
    train_auprc = metrics_history['train_auprc'][start_idx:]
    val_auprc = metrics_history['val_auprc'][start_idx:]
    
    # Loss
    axes[0].plot(epochs, train_loss, label='Train', linewidth=1.5)
    axes[0].plot(epochs, val_loss, label='Val', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title_prefix}Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC
    axes[1].plot(epochs, train_auc, label='Train', linewidth=1.5)
    axes[1].plot(epochs, val_auc, label='Val', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title(f'{title_prefix}AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUPRC
    axes[2].plot(epochs, train_auprc, label='Train', linewidth=1.5)
    axes[2].plot(epochs, val_auprc, label='Val', linewidth=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUPRC')
    axes[2].set_title(f'{title_prefix}AUPRC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

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

    for epoch in range(num_finetune_epochs):
        train_loss, train_auc, train_auprc = train_one_epoch(model_components, train_loader, optimizer, loss_fn, model_params)
        if scheduler:
            scheduler.step()
        val_loss, val_auc, val_auprc = evaluate_model(model_components, val_loader, loss_fn, model_params)

        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['train_auc'].append(train_auc)
        metrics_history['val_auc'].append(val_auc)
        metrics_history['train_auprc'].append(train_auprc)
        metrics_history['val_auprc'].append(val_auprc)

        print(f"[Combination {param_combination_id}] Epoch {epoch + 1}/{num_finetune_epochs} - Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f} | Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            # Save model state dict
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
                print(f"[Combination {param_combination_id}] Early stopping at epoch {epoch+1}")
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
    test_loss, test_auc, test_auprc = evaluate_model(model_components, test_loader, loss_fn, model_params)
    print(f"[Combination {param_combination_id}] Test Results - Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}")
    return test_loss, test_auc, test_auprc


def train_one_epoch(model_components, dataloader, optimizer, loss_fn, model_params=None):
    """Train for one epoch with optional gradient accumulation.

    Gradient accumulation allows simulating a large logical batch size
    even when the physical mini-batch fits in GPU memory.
    Set model_params['accum_steps'] to control how many mini-batches
    to accumulate before calling optimizer.step() (default=1 = no accumulation).
    """
    drug_model = model_components['drug_model']
    classifier = model_components['classifier']
    accum_steps = model_params.get('accum_steps', 1) if model_params else 1

    if drug_model is not None:
        drug_model.eval()
    classifier.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    optimizer.zero_grad()

    for step_idx, batch in enumerate(tqdm(dataloader, desc="Training Batches", leave=False)):
        batch_gene, batch_drug_data, batch_target, batch_weights = batch
        batch_gene   = batch_gene.to(device)
        batch_target = batch_target.to(device)
        batch_weights = batch_weights.to(device)

        drug_batch = batch_drug_data.to(device)
        drug_emb   = drug_model(drug_batch)
        combined   = torch.cat((batch_gene, drug_emb), dim=1)

        pred = classifier(combined).view(-1)

        if torch.isnan(pred).any():
            print("Warning: NaN values detected in predictions. Replacing with zeros.")
            pred = torch.nan_to_num(pred, nan=0.0)

        loss = loss_fn(pred, batch_target,
                       weights=batch_weights if model_params.get('use_sample_weight', True) else None)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN or Inf loss detected. Skipping this mini-batch.")
            optimizer.zero_grad()
        else:
            # Scale loss for gradient accumulation so the effective gradient
            # magnitude is the same as a single large-batch update.
            (loss / accum_steps).backward()

        total_loss += loss.item() * batch_gene.size(0)
        all_preds.extend(pred.detach().cpu().numpy())
        all_targets.extend(batch_target.cpu().numpy())

        # Update parameters every accum_steps mini-batches (or at the last batch)
        is_last_batch = (step_idx + 1 == len(dataloader))
        if (step_idx + 1) % accum_steps == 0 or is_last_batch:
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    avg_loss = total_loss / max(len(dataloader.dataset), 1)

    all_preds_np   = np.array(all_preds)
    all_targets_np = np.array(all_targets)

    if np.isnan(all_preds_np).any():
        print("Warning: NaN values found in predictions. Replacing with zeros for metric calculation.")
        all_preds_np = np.nan_to_num(all_preds_np, nan=0.0)

    try:
        auc_score   = roc_auc_score(all_targets_np, all_preds_np)
        auprc_score = average_precision_score(all_targets_np, all_preds_np)
    except ValueError as e:
        print(f"Error calculating metrics: {e}")
        print("Using default metrics (AUC=0.5, AUPRC=class ratio)")
        auc_score = 0.5
        positive_ratio = np.mean(all_targets_np)
        auprc_score = positive_ratio if not np.isnan(positive_ratio) else 0.5

    return avg_loss, auc_score, auprc_score

def evaluate_model(model_components, dataloader, loss_fn, model_params=None):
    drug_model = model_components['drug_model']
    classifier = model_components['classifier']
    
    if drug_model is not None:
        drug_model.eval()
    classifier.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval Batches", leave=False):
            batch_gene, batch_drug_data, batch_target, batch_weights = batch
            batch_gene = batch_gene.to(device)
            batch_target = batch_target.to(device)
            batch_weights = batch_weights.to(device)
            
            # Use graph-based approach (already batched by collate_fn)
            drug_batch = batch_drug_data.to(device)
            drug_emb = drug_model(drug_batch)
            combined = torch.cat((batch_gene, drug_emb), dim=1)
            
            pred = classifier(combined).view(-1)
            
            # Handle NaN values in predictions
            if torch.isnan(pred).any():
                pred = torch.nan_to_num(pred, nan=0.0)
                
            loss = loss_fn(pred, batch_target,
                           weights=batch_weights if model_params.get('use_sample_weight', True) else None)
            
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

# inference_on_tcga_drugs imported from tools.inference_utils

def build_and_run_one_combination(
    ft_params,
    classifier_params,
    model_params,
    response_df,
    expression_latent_dict,
    drug_latent_dict,
    drug_smiles_df,
    actual_encoder_latent_dim,
    batch_size,
    train_num_epochs,
    model_exp_folder,
    param_combination_id,
    model_id,
    tcga_inference_data_folder,
    tcga_inference_data_folder_extra,
    tcga_latent_dict,
    current_hyperparams,
):
    """
    執行單一超參數組合的完整 pipeline：
      1. 資料過濾與切分 (train / val / test)
      2. Dataset & DataLoader 建立（num_workers=0，Docker 安全）
      3. 模型初始化（GINConvNet + Classify）
      4. 優化器 / Scheduler / 損失函式設定
      5. 訓練（train_combination）
      6. 測試（test_combination）
      7. TCGA 推論（inference_on_tcga_drugs）
      8. 結果封裝並儲存

    Returns
    -------
    dict or None
        current_param_set_results dict（若無有效樣本則回傳 None）
    """
    import math
    from sklearn.model_selection import train_test_split

    # ── 1. 檢查與過濾 ───────────────────────────────────────────────
    drug_col = 'DRUG_ID'
    if 'mapped_name' in response_df.columns: drug_col = 'mapped_name'
    elif 'drug_name' in response_df.columns: drug_col = 'drug_name'
    elif 'DRUG_NAME' in response_df.columns: drug_col = 'DRUG_NAME'
    elif len(response_df.columns) > 1: drug_col = response_df.columns[1]

    # Check for missing drug SMILES (keep raise error)
    valid_drugs_lower = set(str(d).lower() for d in drug_smiles_df.index)
    if 'DRUG_NAME' in drug_smiles_df.columns:
        valid_drugs_lower.update(str(d).lower() for d in drug_smiles_df['DRUG_NAME'].values)
    
    response_drugs = set(response_df[drug_col].astype(str).str.lower())
    missing_drugs = response_drugs - valid_drugs_lower
    if missing_drugs:
        raise ValueError(f"Missing SMILES structure for {len(missing_drugs)} drugs. Examples: {list(missing_drugs)[:5]}")

    # Check for missing expression latent representations:
    # Instead of raising an error, print the missing IDs and skip those rows.
    missing_samples = set(response_df['ModelID'].astype(str)) - set(str(k) for k in expression_latent_dict.keys())
    if missing_samples:
        print(f"  [WARNING] {len(missing_samples)} ModelID(s) have no latent representation and will be skipped:")
        print(f"    Missing latent IDs: {sorted(missing_samples)}")

    # Filter out rows whose ModelID has no latent representation
    current_df = response_df[~response_df['ModelID'].astype(str).isin(missing_samples)]
    
    if len(current_df) == 0:
        print("    No matching samples/drugs found. Skipping.")
        return None

    # ── 2. Train / Val / Test 切分 ───────────────────────────────────
    all_samples = current_df['ModelID'].unique()
    all_labels = [current_df[current_df['ModelID'] == sid]['Label'].iloc[0] for sid in all_samples]

    try:
        train_val_samples, test_samples = train_test_split(
            all_samples, test_size=0.1, random_state=42, stratify=all_labels)
        train_val_labels = [current_df[current_df['ModelID'] == sid]['Label'].iloc[0]
                            for sid in train_val_samples]
        train_samples, val_samples = train_test_split(
            train_val_samples, test_size=0.2, random_state=42, stratify=train_val_labels)
    except ValueError:
        print("    Warning: Stratification failed, using random split")
        train_val_samples, test_samples = train_test_split(all_samples, test_size=0.1, random_state=42)
        train_samples, val_samples = train_test_split(train_val_samples, test_size=0.2, random_state=42)

    # ── 3. Dataset & DataLoader ──────────────────────────────────────
    train_df = current_df[current_df['ModelID'].isin(train_samples)]
    val_df   = current_df[current_df['ModelID'].isin(val_samples)]
    test_df  = current_df[current_df['ModelID'].isin(test_samples)]

    train_subset = DrugResponseDataset(train_df, expression_latent_dict, None, drug_smiles_df, model_params['gin_type'])
    val_subset   = DrugResponseDataset(val_df,   expression_latent_dict, None, drug_smiles_df, model_params['gin_type'])
    test_subset  = DrugResponseDataset(test_df,  expression_latent_dict, None, drug_smiles_df, model_params['gin_type'])

    def _collate(batch):
        gene_list, drug_list, target_list, weight_list = zip(*batch)
        targets = torch.stack(list(target_list))
        weights = torch.stack(list(weight_list))
        genes = torch.stack(list(gene_list)) if isinstance(gene_list[0], torch.Tensor) \
                else Batch.from_data_list(list(gene_list))
        drugs = torch.stack(list(drug_list)) if isinstance(drug_list[0], torch.Tensor) \
                else Batch.from_data_list(list(drug_list))
        return genes, drugs, targets, weights

    _mini_bs = model_params.get('mini_batch_size', batch_size)
    _safe_train_bs = min(_mini_bs, max(len(train_subset), 1))
    _safe_eval_bs  = min(_mini_bs, max(len(val_subset),   1))
    _safe_test_bs  = min(_mini_bs, max(len(test_subset),  1))
    _accum_steps   = max(1, math.ceil(batch_size / _safe_train_bs))

    print(f"  Batch strategy: logical={batch_size}, mini={_safe_train_bs}, accum_steps={_accum_steps}")

    # num_workers=0: Docker /dev/shm 不足以支援多程序 worker
    _pin_memory = torch.cuda.is_available()
    def _make_loader(ds, bs, shuffle, drop):
        """Creates a single-process DataLoader safe for Docker environments."""
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          collate_fn=_collate, num_workers=0,
                          pin_memory=_pin_memory, drop_last=drop)

    train_loader = _make_loader(train_subset, _safe_train_bs, shuffle=True,  drop=True)
    val_loader   = _make_loader(val_subset,   _safe_eval_bs,  shuffle=False, drop=False)
    test_loader  = _make_loader(test_subset,  _safe_test_bs,  shuffle=False, drop=False)

    # ── 4. 模型初始化 ────────────────────────────────────────────────
    gin_type = model_params['gin_type']
    if gin_type == 'dapl':
        drug_gcnmodel = GINConvNet(input_dim=DRUG_INPUT_DIM, output_dim=DRUG_EMBED_DIM,
                                   dropout=0.2, num_layers=5, jk_mode='last',
                                   use_batch_norm=True, pool_type='max').to(device)
    elif gin_type == 'drpreter':
        drug_gcnmodel = GINConvNet(input_dim=DRUG_INPUT_DIM, output_dim=DRUG_EMBED_DIM,
                                   dropout=0.2, num_layers=5, jk_mode='cat',
                                   use_batch_norm=True, pool_type='max').to(device)
    elif gin_type == 'ginpre':
        drug_gcnmodel = GINConvNet(input_dim=DRUG_INPUT_DIM, output_dim=DRUG_EMBED_DIM,
                                   dropout=0.0, num_layers=5, jk_mode='sum',
                                   use_batch_norm=False, pool_type='mean').to(device)
    else:
        raise ValueError(f"Unknown gin_type: {gin_type}")

    if drug_gcnmodel is not None:
        drug_gcnmodel.apply(init_weights)
        classifier_input_dim = actual_encoder_latent_dim + DRUG_EMBED_DIM

    activation_map = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}
    act_fn = activation_map[classifier_params['activation']]
    classifymodel = Classify(
        input_dim=classifier_input_dim,
        hidden_dims=classifier_params['hidden_dims'],
        dop=classifier_params['dropout_rate'],
        act_fn=act_fn, out_fn=None,
        use_bn=classifier_params['use_batch_norm'],
    ).to(device)
    classifymodel.apply(init_weights)

    model_components = {'drug_model': drug_gcnmodel, 'classifier': classifymodel}

    # ── 5. 優化器 / Scheduler / 損失函式 ────────────────────────────
    param_groups = [{'params': classifymodel.parameters()}]
    if drug_gcnmodel is not None:
        param_groups.append({'params': drug_gcnmodel.parameters()})
    optimizer = optim.AdamW(param_groups, lr=ft_params['ftlr'], weight_decay=1e-5)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, train_num_epochs)
                 if ft_params['scheduler_flag'] else None)

    loss_type = ft_params.get('loss_type', 'bce')
    gamma = ft_params.get('focal_loss_gamma', 2.0)
    use_sample_weight = (loss_type != 'bce_unweighted')
    loss_fn = FocalLoss(gamma=gamma if loss_type == 'focal' else 0, reduction='mean').to(device)

    # 注入 use_sample_weight 和 accum_steps 到 model_params
    model_params = {**model_params, 'use_sample_weight': use_sample_weight,
                    'accum_steps': _accum_steps}

    # ── 6. 訓練 ─────────────────────────────────────────────────────
    metrics_history, best_val_auc, best_epoch = train_combination(
        model_components=model_components,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_finetune_epochs=train_num_epochs,
        patience_limit=DEFAULT_PATIENCE,
        model_folder=model_exp_folder,
        param_combination_id=f"{model_id}_p{param_combination_id}",
        model_params=model_params,
    )

    # 儲存學習曲線
    curves_folder = os.path.join(model_exp_folder, 'learning_curves')
    safemakedirs(curves_folder)
    pd.DataFrame(metrics_history).to_csv(
        os.path.join(curves_folder, 'metrics_history.csv'), index_label='Epoch')
    plot_learning_curves(
        metrics_history,
        os.path.join(curves_folder, 'learning_curves.png'),
        title_prefix=f'{model_id}_p{param_combination_id} ',
    )

    # ── 7. 測試集評估 ────────────────────────────────────────────────
    test_loss, test_auc, test_auprc = test_combination(
        model_components=model_components,
        test_loader=test_loader,
        loss_fn=loss_fn,
        model_params=model_params,
        param_combination_id=f"{model_id}_p{param_combination_id}",
    )

    ccle_pred_df = collect_ccle_predictions(
        model_components=model_components,
        dataset=test_subset,
        model_params=model_params,
        domain="CCLE",
        batch_size=_safe_test_bs,
        collate_fn=_collate,
    )

    # ── 8. TCGA 推論 ─────────────────────────────────────────────────
    best_model_path = os.path.join(model_exp_folder, 'best_model.pth')
    print("\n  Running inference on TCGA drugs...")
    tcga_results = inference_on_tcga_drugs(
        model_components,
        tcga_inference_data_folder,
        best_model_path,
        current_hyperparams,
        tcga_latent_dict,
        drug_latent_dict,
        model_params['gin_type'],
        fold_model_folder=model_exp_folder,
        drug_smiles_df=drug_smiles_df,
        tcga_tag='TCGA1',
    )
    tcga_results_extra = {}
    if tcga_inference_data_folder_extra:
        tcga_results_extra = inference_on_tcga_drugs(
            model_components,
            tcga_inference_data_folder_extra,
            best_model_path,
            current_hyperparams,
            tcga_latent_dict,
            drug_latent_dict,
            model_params['gin_type'],
            fold_model_folder=model_exp_folder,
            drug_smiles_df=drug_smiles_df,
            tcga_tag='TCGA2',
        )

    tcga_pred_df = predictions_from_tcga_inference_result(tcga_results, tcga_source="TCGA1")
    tcga_pred_extra_df = predictions_from_tcga_inference_result(tcga_results_extra, tcga_source="TCGA2")
    save_prediction_tables(
        model_exp_folder,
        ccle_test_df=ccle_pred_df,
        tcga_eval_df=tcga_pred_df,
        tcga_eval_extra_df=tcga_pred_extra_df,
    )

    # ── 9. 結果封裝 ──────────────────────────────────────────────────
    return {
        'Params': current_hyperparams,
        'Model_ID': model_id,
        'Train_Metrics': {
            'Loss': metrics_history['train_loss'][best_epoch],
            'AUC':  metrics_history['train_auc'][best_epoch],
            'AUPRC': metrics_history['train_auprc'][best_epoch],
        },
        'Val_Metrics': {
            'Loss': metrics_history['val_loss'][best_epoch],
            'AUC':  metrics_history['val_auc'][best_epoch],
            'AUPRC': metrics_history['val_auprc'][best_epoch],
        },
        'Test_Metrics': {
            'Loss': test_loss,
            'AUC':  test_auc,
            'AUPRC': test_auprc,
        },
        'TCGA_Metrics': tcga_results,
        'TCGA_Metrics_Extra': tcga_results_extra,
        'Best_Epoch': best_epoch + 1,
        'Classifier_Params': classifier_params,
    }


def step_1_finetune_pipeline_zscore(
    outfolder,
    response_data_path,
    model_select_path,  # Changed from expression_latent_root
    drug_smiles_data_path,
    tcga_inference_data_folder,
    tcga_inference_data_folder_extra,
    batch_size,
    mini_batch_size=None,

    config=None,
    epochs=1000):
    
    # Load response data
    print(f"Loading response data from {response_data_path}")
    response_df = pd.read_csv(response_data_path)

    # Resolve effective mini_batch_size (physical GPU batch)
    # Falls back to logical batch_size when not specified.
    if mini_batch_size is None or mini_batch_size <= 0:
        mini_batch_size = batch_size
    print(f"Batch size strategy: logical={batch_size}, mini={mini_batch_size}")
    
    # Load model selection
    print(f"Loading model selection from {model_select_path}")
    model_select_df = pd.read_csv(model_select_path)
    
    # Drug latent representations are removed (no pretrain functionality)
    drug_latent_dict = None
    
    # Load drug SMILES data
    drug_smiles_df = pd.read_csv(drug_smiles_data_path, index_col=0)[['SMILES']].dropna()
    
    # For latent pipeline, we use fixed training parameters
    train_num_epochs = epochs
    
    # Generate fine-tuning parameter combinations from config
    if config is None:
        raise ValueError("config parameter is required")
    
    fine_tune_params_grid_dict = config['finetune_params']
    ft_param_combinations = [dict(zip(fine_tune_params_grid_dict.keys(), v)) 
                           for v in product(*fine_tune_params_grid_dict.values())]
    
    # Generate classifier parameter combinations from config
    classifier_params_grid = config['classifier_params']
    classifier_param_combinations = [dict(zip(classifier_params_grid.keys(), v)) 
                                   for v in product(*classifier_params_grid.values())]
    
    # Generate model parameter combinations from config
    model_params_grid = config['model_params']
    # Filter out expression_latent_subdir if exists
    model_params_clean = {k:v for k,v in model_params_grid.items() if k != 'expression_latent_subdir'}
    model_param_combinations = [dict(zip(model_params_clean.keys(), v))
                               for v in product(*model_params_clean.values())]
    # Inject mini_batch_size into every model_params combination so DataLoader
    # section can read it without extra bookkeeping.
    model_param_combinations = [{**mp, 'mini_batch_size': mini_batch_size}
                                 for mp in model_param_combinations]
    
    all_param_results_list = []
    best_overall_tcga_metrics = {'best_overall_auc': float('-inf'), 'params': None, 'metrics': None}
    
    safemakedirs(outfolder)
    run_config_snapshot = {
        'response_data_path': response_data_path,
        'model_select_path': model_select_path,
        'drug_smiles_data_path': drug_smiles_data_path,
        'tcga_data_folder': tcga_inference_data_folder,
        'tcga_data_folder_extra': tcga_inference_data_folder_extra,
        'batch_size': batch_size,
        'mini_batch_size': mini_batch_size,
        'epochs': epochs,
        'config': config
    }
    with open(os.path.join(outfolder, 'finetune_config_used.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config_snapshot, f, indent=2, ensure_ascii=False, default=str)
    
    # Iterate over selected models
    total_models = len(model_select_df)
    
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
        candidates = [f for f in os.listdir(model_folder) if f.endswith('.pkl') and 'latent' in f.lower()]
        
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
        sample_latent = next(iter(expression_latent_dict.values()))
        actual_encoder_latent_dim = len(sample_latent) if hasattr(sample_latent, '__len__') else sample_latent.shape[-1]
        if actual_encoder_latent_dim != ENCODER_LATENT_DIM:
            print(f"  Note: Detected latent dim={actual_encoder_latent_dim} (default={ENCODER_LATENT_DIM})")
            
        tcga_latent_dict = None
        if tcga_latent_path:
            print(f"  Loading TCGA latent from: {tcga_latent_path}")
            with open(tcga_latent_path, 'rb') as f:
                tcga_latent_dict = pickle.load(f)
        else:
            print("  Warning: No TCGA latent file found. TCGA evaluation will be limited.")

        param_combination_id = 0
        total_combinations = len(ft_param_combinations) * len(classifier_param_combinations) * len(model_param_combinations)
        
        for ft_params in ft_param_combinations:
            for classifier_params in classifier_param_combinations:
                for model_params in model_param_combinations:
                    param_combination_id += 1
                    current_hyperparams = {**ft_params, **classifier_params, **model_params}
                    current_hyperparams['Model_ID'] = model_id

                    print(f"  [{param_combination_id}/{total_combinations}] Training params: {current_hyperparams}")

                    model_exp_folder = os.path.join(outfolder, str(model_id), f"param_{param_combination_id:03d}")
                    safemakedirs(model_exp_folder)
                    with open(os.path.join(model_exp_folder, 'params_used.json'), 'w', encoding='utf-8') as f:
                        json.dump(current_hyperparams, f, indent=2, ensure_ascii=False, default=str)

                    # ── 執行單一組合（資料過濾、模型訓練、TCGA推論） ────
                    result = build_and_run_one_combination(
                        ft_params=ft_params,
                        classifier_params=classifier_params,
                        model_params=model_params,
                        response_df=response_df,
                        expression_latent_dict=expression_latent_dict,
                        drug_latent_dict=drug_latent_dict,
                        drug_smiles_df=drug_smiles_df,
                        actual_encoder_latent_dim=actual_encoder_latent_dim,
                        batch_size=batch_size,
                        train_num_epochs=train_num_epochs,
                        model_exp_folder=model_exp_folder,
                        param_combination_id=param_combination_id,
                        model_id=model_id,
                        tcga_inference_data_folder=tcga_inference_data_folder,
                        tcga_inference_data_folder_extra=tcga_inference_data_folder_extra,
                        tcga_latent_dict=tcga_latent_dict,
                        current_hyperparams=current_hyperparams,
                    )

                    if result is None:
                        continue  # 無有效樣本，跳過

                    all_param_results_list.append(result)
                    save_single_param_set_metrics(model_exp_folder, result)
                    print(f"  Metrics saved in: {model_exp_folder}")

                    # ── 更新全域最佳 ─────────────────────────
                    current_global_auc = result['TCGA_Metrics']['Global_Metrics'].get('AUC', np.nan)
                    if pd.isna(current_global_auc):
                        current_global_auc = result['Val_Metrics']['AUC']

                    if not pd.isna(current_global_auc) and current_global_auc > best_overall_tcga_metrics['best_overall_auc']:
                        best_overall_tcga_metrics['best_overall_auc'] = current_global_auc
                        best_overall_tcga_metrics['params'] = current_hyperparams
                        best_overall_tcga_metrics['metrics'] = result

    all_results_path = os.path.join(outfolder, 'all_parameter_results_summary.json')
    with open(all_results_path, 'w') as f:
        json.dump({
            'all_parameter_sets_results': all_param_results_list,
            'best_performing_params_on_overall_tcga_auc': best_overall_tcga_metrics
        }, f, indent=4, cls=NpEncoder)
    
    # Create final comparative summary CSV for all parameter sets
    comparison_df = create_final_parameter_comparison_csv(outfolder, all_param_results_list, config)

    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE - SUMMARY OF RESULTS")
    print("="*80)
    
    if best_overall_tcga_metrics['params']:
        print("\nBest Parameter Combination (based on Overall TCGA AUC):")
        print(f"  Train Epochs: {train_num_epochs}")
        print(f"  Finetune LR: {best_overall_tcga_metrics['params']['ftlr']}")
        print(f"  Scheduler: {best_overall_tcga_metrics['params']['scheduler_flag']}")
        print(f"  Classifier Hidden Dims: {best_overall_tcga_metrics['params']['hidden_dims']}")
        print(f"  Classifier Dropout: {best_overall_tcga_metrics['params']['dropout_rate']}")
        print(f"  Classifier Batch Norm: {best_overall_tcga_metrics['params']['use_batch_norm']}")
        print(f"  Classifier Activation: {best_overall_tcga_metrics['params']['activation']}")
        print(f"  GIN Type: {best_overall_tcga_metrics['params']['gin_type']}")
        
        print("\nPerformance Metrics:")
        print(f"  Train AUC: {best_overall_tcga_metrics['metrics']['Train_Metrics']['AUC']:.4f}")
        print(f"  Val AUC: {best_overall_tcga_metrics['metrics']['Val_Metrics']['AUC']:.4f}")
        print(f"  Test AUC: {best_overall_tcga_metrics['metrics']['Test_Metrics']['AUC']:.4f}")
        print(f"  Best Epoch: {best_overall_tcga_metrics['metrics']['Best_Epoch']}")
        
        print("\nTCGA Performance:")
        print(f"  Global TCGA AUC: {best_overall_tcga_metrics['metrics']['TCGA_Metrics']['Global_Metrics']['AUC']:.4f}")
        print(f"  Average TCGA AUC: {best_overall_tcga_metrics['metrics']['TCGA_Metrics']['Average_Metrics']['AUC']:.4f}")
        
        print("\nIndividual TCGA Drug Performance:")
        print(f"  {'Drug':<15} | {'AUC':^15} | {'AUPRC':^15}")
        print(f"  {'-'*15}-+-{'-'*15}-+-{'-'*15}")
        for drug_name, metrics in best_overall_tcga_metrics['metrics']['TCGA_Metrics']['Drug_Metrics'].items():
            print(f"  {drug_name:<15} | {metrics['AUC']:.4f} | {metrics['AUPRC']:.4f}")
    else:
        print("No results found to determine the best parameters.")
    
    print("\nOutput Files:")
    print(f"  - Detailed Parameter Comparison: {os.path.join(outfolder, 'parameter_comparison_detailed.csv')}")
    print(f"  - TCGA-Focused Parameter Comparison: {os.path.join(outfolder, 'parameter_comparison_tcga_focus.csv')}")
    print(f"  - Complete Results JSON: {all_results_path}")
    print("="*80)
    
    return best_overall_tcga_metrics



def save_single_param_set_metrics(model_folder, param_set_results):
    """Save metrics for a single parameter combination"""
    metrics_out_folder = os.path.join(model_folder, 'metrics_summary')
    safemakedirs(metrics_out_folder)
    
    # Save detailed metrics
    metrics_data = {
        # Parameters
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
    tcga_res = param_set_results['TCGA_Metrics']
    
    # Global & Average
    metrics_data['Global_TCGA_AUC'] = tcga_res['Global_Metrics'].get('AUC')
    metrics_data['Global_TCGA_AUPRC'] = tcga_res['Global_Metrics'].get('AUPRC')
    metrics_data['Average_TCGA_AUC'] = tcga_res['Average_Metrics'].get('AUC')
    metrics_data['Average_TCGA_AUPRC'] = tcga_res['Average_Metrics'].get('AUPRC')
    
    # Individual Drugs
    for drug, metrics in tcga_res['Drug_Metrics'].items():
        metrics_data[f'{drug}_TCGA_AUC'] = metrics['AUC']
        metrics_data[f'{drug}_TCGA_AUPRC'] = metrics['AUPRC']
    tcga_res_extra = param_set_results.get('TCGA_Metrics_Extra', {})
    if tcga_res_extra:
        metrics_data['TCGA2_Global_TCGA_AUC'] = tcga_res_extra.get('Global_Metrics', {}).get('AUC')
        metrics_data['TCGA2_Global_TCGA_AUPRC'] = tcga_res_extra.get('Global_Metrics', {}).get('AUPRC')
        metrics_data['TCGA2_Average_TCGA_AUC'] = tcga_res_extra.get('Average_Metrics', {}).get('AUC')
        metrics_data['TCGA2_Average_TCGA_AUPRC'] = tcga_res_extra.get('Average_Metrics', {}).get('AUPRC')
        for drug, metrics in tcga_res_extra.get('Drug_Metrics', {}).items():
            metrics_data[f'TCGA2_{drug}_TCGA_AUC'] = metrics.get('AUC')
            metrics_data[f'TCGA2_{drug}_TCGA_AUPRC'] = metrics.get('AUPRC')
    
    # Save the metrics
    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(os.path.join(metrics_out_folder, 'metrics_summary.csv'), index=False)

def create_final_parameter_comparison_csv(outfolder, all_param_results_list, config=None):
    """Create CSV files comparing all hyperparameter combinations"""
    if not all_param_results_list:
        print("No data to create final parameter comparison CSV.")
        return
    
    # 1. Detailed Result Table (Detailed Parameters + Global/Avg TCGA, NO Individual Drugs)
    detailed_rows = []
    
    # 2. TCGA Focused Table (Minimal Parameters + Global/Avg + Individual Drugs)
    
    all_rows = []
    
    for i, result in enumerate(all_param_results_list):
        row = {'ID': i + 1}
        
        if 'Model_ID' in result:
            row['Model_ID'] = result['Model_ID']
            
        # --- Parameters ---
        if config:
            # Finetune
            if 'finetune_params' in config:
                for param_name in config['finetune_params'].keys():
                    if param_name in result['Params']:
                        row[f'Finetune_{param_name.upper()}'] = result['Params'][param_name]
            # Classifier
            if 'classifier_params' in config:
                for param_name in config['classifier_params'].keys():
                    if param_name in result['Params']:
                        if isinstance(result['Params'][param_name], list):
                             row[f'Classifier_{param_name.upper()}'] = str(result['Params'][param_name])
                        else:
                            row[f'Classifier_{param_name.upper()}'] = result['Params'][param_name]
            # Model
            if 'model_params' in config:
                for param_name in config['model_params'].keys():
                    if param_name in result['Params']:
                        row[f'Model_{param_name.upper()}'] = result['Params'][param_name]
        else:
             # Fallback
            row['Finetune_LR'] = result['Params']['ftlr']
            row['Scheduler'] = result['Params']['scheduler_flag']
            row['GIN_Type'] = result['Params']['gin_type']

        # --- Metrics ---
        # CCLE Train/Val/Test
        row['Train_AUC'] = result['Train_Metrics']['AUC']
        row['Val_AUC'] = result['Val_Metrics']['AUC']
        row['Test_AUC'] = result['Test_Metrics']['AUC']
        row['Train_AUPRC'] = result['Train_Metrics']['AUPRC']
        row['Val_AUPRC'] = result['Val_Metrics']['AUPRC']
        row['Test_AUPRC'] = result['Test_Metrics']['AUPRC']
        
        row['Best_Epoch'] = result['Best_Epoch']

        # TCGA Metrics
        tcga_res = result['TCGA_Metrics']
        row['Global_TCGA_AUC'] = tcga_res['Global_Metrics'].get('AUC')
        row['Global_TCGA_AUPRC'] = tcga_res['Global_Metrics'].get('AUPRC')
        row['Average_TCGA_AUC'] = tcga_res['Average_Metrics'].get('AUC')
        row['Average_TCGA_AUPRC'] = tcga_res['Average_Metrics'].get('AUPRC')
        
        # Individual Drugs (Variable columns)
        for drug, metrics in tcga_res['Drug_Metrics'].items():
            row[f'{drug}_TCGA_AUC'] = metrics['AUC']
            row[f'{drug}_TCGA_AUPRC'] = metrics['AUPRC']
        tcga_res_extra = result.get('TCGA_Metrics_Extra', {})
        if tcga_res_extra:
            row['TCGA2_Global_TCGA_AUC'] = tcga_res_extra.get('Global_Metrics', {}).get('AUC')
            row['TCGA2_Global_TCGA_AUPRC'] = tcga_res_extra.get('Global_Metrics', {}).get('AUPRC')
            row['TCGA2_Average_TCGA_AUC'] = tcga_res_extra.get('Average_Metrics', {}).get('AUC')
            row['TCGA2_Average_TCGA_AUPRC'] = tcga_res_extra.get('Average_Metrics', {}).get('AUPRC')
            for drug, metrics in tcga_res_extra.get('Drug_Metrics', {}).items():
                row[f'TCGA2_{drug}_TCGA_AUC'] = metrics.get('AUC')
                row[f'TCGA2_{drug}_TCGA_AUPRC'] = metrics.get('AUPRC')
            
        all_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    
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
    # Exclude: Detailed Parameters (keep IDs)
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
    gene_list, drug_list, target_list = zip(*batch)
    return list(gene_list), list(drug_list), list(target_list)



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
                       help='Logical (effective) batch size. Gradient accumulation is used '
                            'automatically if mini_batch_size < batch_size.')
    parser.add_argument('--mini_batch_size', dest='mini_batch_size', type=int,
                       default=None,
                       help='Physical mini-batch size loaded to GPU per step. '
                            'Defaults to batch_size (no accumulation). '
                            'Set smaller than batch_size to save GPU memory while '
                            'preserving the same effective batch size via gradient accumulation.')
                       
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
        mini_batch_size=args.mini_batch_size,

        config=config,
        epochs=args.epochs
    )
    
    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_str = str(timedelta(seconds=int(execution_time)))
    print("="*80)
