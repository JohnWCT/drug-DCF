import os
import torch
import numpy as np
import pandas as pd
import json
import time
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
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from tools.model_opt import VAE, Classify, init_weights
from drugmodels.ginconv import GINConvNet
from tools.dataprocess import smile_to_graph, safemakedirs

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. No GPU detected.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

FIXED_DRUG_SMILES_DATA_PATH = "./data/GDSC+214_drug_merge_pubchem.csv"
FIXED_TCGA_DATA_FOLDER = "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv"
FIXED_TCGA_DATA_FOLDER_EXTRA = "data/TCGA/TCGA_drug_response_from_DAPL.csv"

# Function to load parameters from config file
def load_config(config_path='config/params_grid.json'):
    """Load parameters from config file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate that required keys exist
        required_keys = ['pretrain_params', 'finetune_params', 'classifier_params', 'model_params']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config file {config_path} must contain '{key}' key")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config file {config_path}: {e}")



# Define constants for model dimensions and training parameters
GENE_INPUT_DIM = 1426
DRUG_INPUT_DIM = 78
ENCODER_LATENT_DIM = 32
DRUG_EMBED_DIM = 300  # Updated to match GIN latent dimension
DEFAULT_PATIENCE = 20
# Learning curve display control:
# skip first N epochs on x-axis when plotting.
CURVE_SKIP_INITIAL_EPOCHS = 1


def _normalize_drug_name(name):
    if pd.isna(name):
        return ""
    return ''.join(ch for ch in str(name).lower().strip() if ch.isalnum())


def _tcga_patient_key(sample_id):
    sid = str(sample_id).strip()
    if sid.upper().startswith("TCGA-"):
        parts = sid.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
    return sid


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



def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find optimal threshold using Youden's index (sensitivity + specificity - 1)
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        optimal_threshold: Threshold that maximizes Youden's index
        youden_scores: Dictionary with optimal threshold and corresponding metrics
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate Youden's index for each threshold
    youden_scores = tpr - fpr  # sensitivity - (1-specificity) = sensitivity + specificity - 1
    
    # Find threshold that maximizes Youden's index
    optimal_idx = np.argmax(youden_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity  # Same as sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    youden_scores_dict = {
        'optimal_threshold': optimal_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'youden_index': youden_scores[optimal_idx]
    }
    
    return optimal_threshold, youden_scores_dict

def calculate_comprehensive_metrics(y_true, y_pred_proba):
    """
    Calculate comprehensive evaluation metrics including AUC, AUPRC, and binary classification metrics
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        metrics_dict: Dictionary containing all metrics
    """
    # Calculate AUC and AUPRC
    auc_score = roc_auc_score(y_true, y_pred_proba)
    auprc_score = average_precision_score(y_true, y_pred_proba)
    
    # Find optimal threshold and calculate binary metrics
    optimal_threshold, binary_metrics = find_optimal_threshold(y_true, y_pred_proba)
    
    # Combine all metrics
    metrics_dict = {
        'AUC': auc_score,
        'AUPRC': auprc_score,
        'optimal_threshold': optimal_threshold,
        'sensitivity': binary_metrics['sensitivity'],
        'specificity': binary_metrics['specificity'],
        'precision': binary_metrics['precision'],
        'recall': binary_metrics['recall'],
        'f1_score': binary_metrics['f1_score'],
        'confusion_matrix': binary_metrics['confusion_matrix'],
        'youden_index': binary_metrics['youden_index']
    }
    
    return metrics_dict

def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

class DrugResponseDataset(Dataset):
    def __init__(self, df, expression_df, drug_smiles_df):
        self.df = df.reset_index(drop=True).copy()
        self.drug_smiles_df = drug_smiles_df
        
        # Use raw expression data
        self.expr_dict = {}
        for sample_id in expression_df.index:
            expr = expression_df.loc[sample_id].values.astype(np.float32)
            self.expr_dict[sample_id] = torch.tensor(expr, dtype=torch.float32)
        
        # Use SMILES-based drug graphs
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
        
        # Handle different column names
        if 'DepMap_ID' in row:
            sample_id = row['DepMap_ID']
        elif 'ModelID' in row:
            sample_id = row['ModelID']
        else:
            raise ValueError("Neither 'DepMap_ID' nor 'ModelID' column found in data")
        
        if 'DRUG_NAME' in row:
            drug_id = row['DRUG_NAME']
        elif 'drug_name' in row:
            drug_id = row['drug_name']
        else:
            raise ValueError("Neither 'DRUG_NAME' nor 'drug_name' column found in data")
        
        if 'Class' in row:
            target = float(row['Class'])
        elif 'Label' in row:
            target = float(row['Label'])
        else:
            raise ValueError("Neither 'Class' nor 'Label' column found in data")
        
        # Return raw data, let the models handle transformation
        gene_feature = self.expr_dict[sample_id]
        drug_key = self.drug_graph_norm_map.get(_normalize_drug_name(drug_id), drug_id)
        drug_data = self.drug_graph_dict[drug_key]
        target = torch.tensor(target, dtype=torch.float32)
        return gene_feature, drug_data, target

def get_encoder_output(encoder, x):
    """
    Get encoder output from VAE
    """
    _, z, _, _ = encoder(x)
    return z

def train_one_epoch(model_components, dataloader, optimizer, loss_fn):
    encoder = model_components['encoder']
    drug_model = model_components['drug_model']
    classifier = model_components['classifier']
    
    # Set model modes
    # Encoder is always in eval mode (either frozen or trainable)
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
        
        optimizer.zero_grad()
        
        # Get gene features using encoder
        with torch.no_grad():
            z = get_encoder_output(encoder, batch_gene)
        
        # Get drug features using GIN model
        drug_batch = Batch.from_data_list(drug_data_list).to(device)
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

def evaluate_model(model_components, dataloader, loss_fn, calculate_detailed_metrics=False):
    encoder = model_components['encoder']
    drug_model = model_components['drug_model']
    classifier = model_components['classifier']
    
    # Set model modes
    # Encoder is always in eval mode (either frozen or trainable)
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
            
            # Get gene features using encoder
            z = get_encoder_output(encoder, batch_gene)
            
            # Get drug features using GIN model
            drug_batch = Batch.from_data_list(drug_data_list).to(device)
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

def test_combination(model_components, test_loader, loss_fn, param_combination_id):
    """Evaluate trained components on the test loader and print results."""
    test_loss, test_metrics = evaluate_model(model_components, test_loader, loss_fn, calculate_detailed_metrics=True)
    # Test completed
    return test_loss, test_metrics

def inference_on_tcga_drugs(model_components, tcga_data_folder, best_model_path, ft_params, fold_model_folder=None, tcga_tag='TCGA'):
    encoder = model_components['encoder']
    drug_model = model_components['drug_model']
    classifier = model_components['classifier']
    
    if best_model_path is not None:
        try:
            checkpoint = torch.load(best_model_path)
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
        except Exception as e:
            print(f"Error loading best model: {e}")
            print(f"Continuing with current model state...")
    
    encoder.eval()
    drug_model.eval()
    classifier.eval()
    
    drug_list = ['cis', 'sor', 'tem', 'gem', 'fu']
    drug_smiles = ['N.N.Cl[Pt]Cl', 
                   'CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F',
                   'CN1C(=O)N2C=NC(=C2N=N1)C(=O)N', 
                   'C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F', 
                   'C1=C(C(=O)NC(=O)N1)F']
    drug_aliases = {
        'cis': ['cis', 'cisplatin'],
        'sor': ['sor', 'sorafenib'],
        'tem': ['tem', 'temozolomide'],
        'gem': ['gem', 'gemcitabine'],
        'fu': ['fu', '5fu', '5 fluorouracil', '5-fluorouracil', 'fluorouracil'],
    }
    if not (os.path.isfile(tcga_data_folder) and str(tcga_data_folder).lower().endswith('.csv')):
        raise ValueError(
            f"tcga_data_folder must be a TCGA response CSV file, got: {tcga_data_folder}"
        )
    tcga_expr_df = pd.read_csv("data/TCGA/pretrain_tcga.csv", index_col=0)
    tcga_expr_df.index = tcga_expr_df.index.astype(str).map(_tcga_patient_key)
    tcga_expr_df = tcga_expr_df.groupby(tcga_expr_df.index).first()
    tcga_resp_df = pd.read_csv(tcga_data_folder)
    candidate_patient_cols = ['patient', 'Patient_id', 'patient.arr', 'sample', 'Sample']
    candidate_drug_cols = ['drug_name', 'DRUG_NAME', 'drug.name']
    candidate_label_cols = ['Label', 'Class']
    resp_patient_col = next((c for c in candidate_patient_cols if c in tcga_resp_df.columns), None)
    resp_drug_col = next((c for c in candidate_drug_cols if c in tcga_resp_df.columns), None)
    resp_label_col = next((c for c in candidate_label_cols if c in tcga_resp_df.columns), None)
    if not resp_patient_col or not resp_drug_col or not resp_label_col:
        raise ValueError("Missing required columns in TCGA response CSV.")
    tcga_resp_df = tcga_resp_df[[resp_patient_col, resp_drug_col, resp_label_col]].copy()
    tcga_resp_df[resp_patient_col] = tcga_resp_df[resp_patient_col].astype(str).map(_tcga_patient_key)
    tcga_resp_df[resp_drug_col] = tcga_resp_df[resp_drug_col].map(_normalize_drug_name)
    tcga_resp_df[resp_label_col] = pd.to_numeric(tcga_resp_df[resp_label_col], errors='coerce')
    tcga_resp_df = tcga_resp_df.dropna(subset=[resp_patient_col, resp_drug_col, resp_label_col])
    
    results = {}
    
    for drug_name, drug_smile in zip(drug_list, drug_smiles):
        try:
            alias_set = {_normalize_drug_name(x) for x in drug_aliases.get(drug_name, [drug_name])}
            sub = tcga_resp_df[tcga_resp_df[resp_drug_col].isin(alias_set)].copy()
            if sub.empty:
                print(f"Warning: No TCGA rows for drug {drug_name} in CSV mode.")
                target_data = torch.empty(0, GENE_INPUT_DIM).float().to(device)
                target_labels = torch.empty(0).float().squeeze().to(device)
            else:
                sub = sub.drop_duplicates(subset=[resp_patient_col], keep='first')
                common_ids = [pid for pid in sub[resp_patient_col].tolist() if pid in tcga_expr_df.index]
                if not common_ids:
                    print(f"Warning: No overlapping TCGA expression for drug {drug_name} in CSV mode.")
                    target_data = torch.empty(0, GENE_INPUT_DIM).float().to(device)
                    target_labels = torch.empty(0).float().squeeze().to(device)
                else:
                    sub_idx = sub.set_index(resp_patient_col)
                    feat_np = np.nan_to_num(tcga_expr_df.loc[common_ids].values, nan=0.0).astype(np.float32)
                    label_np = np.nan_to_num(sub_idx.loc[common_ids, resp_label_col].values, nan=0.0).astype(np.float32)
                    target_data = torch.from_numpy(feat_np).float().to(device)
                    target_labels = torch.from_numpy(label_np).float().squeeze().to(device)
            
            # Skip if no data is available
            if target_data.shape[0] == 0:
                print(f"Warning: No data available for drug {drug_name}. Skipping.")
                results[drug_name] = {
                    'AUC': np.nan, 'AUPRC': np.nan, 'sensitivity': np.nan, 'specificity': np.nan,
                    'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan, 'optimal_threshold': np.nan
                }
                continue
            
            c_size, atom_features_list, edge_index = smile_to_graph(drug_smile)
            drug_x = torch.tensor(np.array(atom_features_list), dtype=torch.float32)
            if len(edge_index) > 0:
                drug_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                drug_edge_index = torch.empty((2, 0), dtype=torch.long)
            drug_data = DATA.Data(x=drug_x, edge_index=drug_edge_index)
            
            with torch.no_grad():
                # Get gene expression embeddings
                z = get_encoder_output(encoder, target_data)
                num_samples = z.shape[0]
                
                # Get drug embedding and expand to match batch size
                drug_batch = Batch.from_data_list([drug_data]).to(device)
                drug_emb_single = drug_model(drug_batch)
                drug_emb_single = _align_drug_embedding(drug_emb_single, drug_batch, 1)
                drug_emb_batch = drug_emb_single.expand(num_samples, -1)
                
                # Combine embeddings and get predictions
                combined = torch.cat((z, drug_emb_batch), dim=1)
                pred = classifier(combined).view(-1)
                pred_probs = torch.sigmoid(pred)
                
                # Convert to numpy for metric calculation
                all_preds = pred_probs.cpu().numpy()
                all_targets = target_labels.cpu().numpy()
                
                # Handle potential NaN values
                if np.isnan(all_preds).any() or np.isnan(all_targets).any():
                    print(f"Warning: NaN values in predictions or targets for drug {drug_name}")
                    all_preds = np.nan_to_num(all_preds)
                    all_targets = np.nan_to_num(all_targets)
                
                # Calculate comprehensive metrics
                try:
                    comprehensive_metrics = calculate_comprehensive_metrics(all_targets, all_preds)
                    auc_score = comprehensive_metrics['AUC']
                    auprc_score = comprehensive_metrics['AUPRC']
                    sensitivity = comprehensive_metrics['sensitivity']
                    specificity = comprehensive_metrics['specificity']
                    precision = comprehensive_metrics['precision']
                    recall = comprehensive_metrics['recall']
                    f1_score = comprehensive_metrics['f1_score']
                    optimal_threshold = comprehensive_metrics['optimal_threshold']
                    confusion_matrix = comprehensive_metrics['confusion_matrix']
                    
                    # Save confusion matrix for this drug
                    safe_drug_name = ''.join(ch if str(ch).isalnum() else '_' for ch in str(drug_name))
                    if fold_model_folder is not None:
                        # If we're in a fold context, save to fold folder
                        cm_folder = os.path.join(fold_model_folder, 'tcga_confusion_matrices')
                        safemakedirs(cm_folder)
                        cm_path = os.path.join(cm_folder, f'{tcga_tag}_tcga_confusion_matrix_{safe_drug_name}.png')
                    else:
                        # Otherwise save to a general location
                        cm_folder = 'tcga_confusion_matrices'
                        safemakedirs(cm_folder)
                        cm_path = os.path.join(cm_folder, f'{tcga_tag}_tcga_confusion_matrix_{safe_drug_name}.png')
                    
                    plot_confusion_matrix(confusion_matrix, cm_path, f"{tcga_tag} Confusion Matrix - {drug_name}")
                    print(f"{tcga_tag} confusion matrix saved for {drug_name}: {cm_path}")
                    
                except ValueError as e:
                    print(f"Error calculating metrics for drug {drug_name}: {e}")
                    auc_score = np.nan
                    auprc_score = np.nan
                    sensitivity = np.nan
                    specificity = np.nan
                    precision = np.nan
                    recall = np.nan
                    f1_score = np.nan
                    optimal_threshold = np.nan
                    confusion_matrix = None
                    
        except Exception as e:
            print(f"Error processing drug {drug_name}: {e}")
            auc_score = np.nan
            auprc_score = np.nan
            sensitivity = np.nan
            specificity = np.nan
            precision = np.nan
            recall = np.nan
            f1_score = np.nan
            optimal_threshold = np.nan
            
        results[drug_name] = {
            'AUC': auc_score,
            'AUPRC': auprc_score,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'optimal_threshold': optimal_threshold
        }
    
    return results

def step_1_finetune_pipeline_zscore(
    pretrain_model_root_folder, 
    drug_model_path, 
    outfolder,
    response_data_path,
    expression_data_path,
    drug_smiles_data_path,
    tcga_inference_data_folder,
    tcga_inference_data_folder_extra,
    pretrain_params_grid,
    fine_tune_params_grid,
    classifier_params_grid,
    model_params_grid,
    batch_size=2048,
    config=None):
    
    response_df = pd.read_csv(response_data_path)
    expression_df = pd.read_csv(expression_data_path, index_col=0)
    drug_smiles_df = pd.read_csv(drug_smiles_data_path, index_col=0)[['SMILES']].dropna()
    
    valid_ids = expression_df.index
    # Check if DepMap_ID or ModelID column exists
    if 'DepMap_ID' in response_df.columns:
        id_column = 'DepMap_ID'
    elif 'ModelID' in response_df.columns:
        id_column = 'ModelID'
    else:
        raise ValueError("Neither 'DepMap_ID' nor 'ModelID' column found in response data")
    
    response_df = response_df[response_df[id_column].isin(valid_ids)]
    all_samples = response_df[id_column].unique()
    print(f'Total number of unique samples: {len(all_samples)}')
    
    drug_encoder_dict = torch.load(drug_model_path)
    model_dir_roots = [pretrain_model_root_folder]
    parent_model_root = os.path.dirname(pretrain_model_root_folder)
    if parent_model_root not in model_dir_roots:
        model_dir_roots.append(parent_model_root)
    
    pretrain_param_combinations = [dict(zip(pretrain_params_grid.keys(), v)) 
                                 for v in product(*pretrain_params_grid.values())]
    ft_param_combinations = [dict(zip(fine_tune_params_grid.keys(), v)) 
                           for v in product(*fine_tune_params_grid.values())]
    classifier_param_combinations = [dict(zip(classifier_params_grid.keys(), v)) 
                                   for v in product(*classifier_params_grid.values())]
    model_param_combinations = [dict(zip(model_params_grid.keys(), v)) 
                              for v in product(*model_params_grid.values())]
    
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
            'config': config if config is not None else {
                'pretrain_params': pretrain_params_grid,
                'finetune_params': fine_tune_params_grid,
                'classifier_params': classifier_params_grid,
                'model_params': model_params_grid
            }
        }, f, indent=2, ensure_ascii=False, default=str)
    
    param_combination_id = 0
    
    for pretrain_params in pretrain_param_combinations:
        for ft_params in ft_param_combinations:
            for classifier_params in classifier_param_combinations:
                for model_params in model_param_combinations:
                    param_combination_id += 1
                    current_hyperparams = {**pretrain_params, **ft_params, **classifier_params, **model_params}
                    print(f"\n[{param_combination_id}] Training with parameters: {current_hyperparams}")
                    
                    model_folder = os.path.join(outfolder, f"param_{param_combination_id:03d}")
                    safemakedirs(model_folder)
                    with open(os.path.join(model_folder, 'params_used.json'), 'w', encoding='utf-8') as f:
                        json.dump(current_hyperparams, f, indent=2, ensure_ascii=False, default=str)
            
            # Split data: 10% for testing, 90% for 5-fold CV
            # Determine column names
            if 'DepMap_ID' in response_df.columns:
                id_column = 'DepMap_ID'
            elif 'ModelID' in response_df.columns:
                id_column = 'ModelID'
            else:
                raise ValueError("Neither 'DepMap_ID' nor 'ModelID' column found in response data")
            
            if 'Class' in response_df.columns:
                label_column = 'Class'
            elif 'Label' in response_df.columns:
                label_column = 'Label'
            else:
                raise ValueError("Neither 'Class' nor 'Label' column found in response data")
            
            all_samples = response_df[id_column].unique()
            all_labels = []
            for sample_id in all_samples:
                sample_data = response_df[response_df[id_column] == sample_id]
                label = sample_data[label_column].iloc[0]
                all_labels.append(label)
            
            # First split: 10% for testing, 90% for CV
            from sklearn.model_selection import train_test_split
            cv_samples, test_samples = train_test_split(all_samples, test_size=0.1, random_state=42, stratify=all_labels)
            
            # Create datasets
            cv_df = response_df[response_df[id_column].isin(cv_samples)]
            test_df = response_df[response_df[id_column].isin(test_samples)]
            
            print(f"Data split: {len(cv_samples)} samples for CV, {len(test_samples)} samples for testing")
            
            # Load pretrained encoder weights based on encoder_mode
            encoder_state_dict = None
            
            if model_params['encoder_mode'] == 'frozen':
                # Load pretrained encoder weights for frozen mode
                pretrain_model_dir_name = (f"pt_epochs_{pretrain_params['pretrain_num_epochs']}"
                                        f",t_epochs_{pretrain_params['train_num_epochs']}"
                                        f",Ptlr_{pretrain_params['pretrain_learning_rate']}"
                                        f",tlr{pretrain_params['gan_learning_rate']}"
                                        f",dop{pretrain_params['dropout_rate']}"
                                        f",enc{len(pretrain_params['encoder_dims'])}")

                pretrain_model_dir = None
                encoder_path = None
                for root_dir in model_dir_roots:
                    candidate_dir = os.path.join(root_dir, pretrain_model_dir_name)
                    candidate_encoder_path = os.path.join(candidate_dir, 'after_traingan_shared_vae.pth')
                    if os.path.exists(candidate_encoder_path):
                        pretrain_model_dir = candidate_dir
                        encoder_path = candidate_encoder_path
                        break
                if pretrain_model_dir is None:
                    pretrain_model_dir = os.path.join(model_dir_roots[0], pretrain_model_dir_name)
                    encoder_path = os.path.join(pretrain_model_dir, 'after_traingan_shared_vae.pth')
                
                try:
                    encoder_state_dict = torch.load(encoder_path, map_location=device)
                    print(f"Loaded pretrained encoder from {encoder_path}")
                except FileNotFoundError:
                    print(f"ERROR: Pretrained encoder not found at {encoder_path}")
                    print(f"Skipping this parameter combination: {current_hyperparams}")
                    continue
            else:
                # trainable mode: don't load pretrained weights, will train from scratch
                print(f"Encoder mode: {model_params['encoder_mode']} - will train encoder from scratch")
            
            # Perform 5-fold cross validation
            fold_results, mean_metrics = perform_5fold_cross_validation(
                response_df=cv_df,  # Use CV data (90% of data)
                expression_df=expression_df,
                drug_smiles_df=drug_smiles_df,
                pretrain_params=pretrain_params,
                model_params=model_params,
                ft_params=ft_params,
                classifier_params=classifier_params,
                batch_size=batch_size,
                num_epochs=pretrain_params['train_num_epochs'],
                patience_limit=DEFAULT_PATIENCE,
                param_combination_id=param_combination_id,
                model_folder=model_folder,
                tcga_inference_data_folder=tcga_inference_data_folder,
                tcga_inference_data_folder_extra=tcga_inference_data_folder_extra,
                encoder_state_dict=encoder_state_dict,
                test_df=test_df  # Pass test data for final evaluation
            )
            
            # Save fold results
            fold_results_folder = os.path.join(model_folder, 'fold_results')
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
            tcga_results_extra = best_fold.get('tcga_results_extra', {})
            
            # Store results for this parameter combination
            current_param_set_results = {
                'Params': current_hyperparams,
                'Fold_Results': fold_results,
                'Mean_Metrics': mean_metrics,
                'TCGA_Metrics': tcga_results,
                'TCGA_Metrics_Extra': tcga_results_extra,
                'Best_Fold': best_fold_idx + 1,
                'Best_Fold_AUC': best_fold['best_val_auc']
            }
            
            all_param_results_list.append(current_param_set_results)
            
            # Save metrics for current parameter set
            save_single_param_set_metrics_5fold(model_folder, current_param_set_results)
            
            # Print summary for this combination
            print(f"[Combination {param_combination_id}] Completed - Mean CV AUC: {mean_metrics['val_AUC_mean']:.4f} ± {mean_metrics['val_AUC_std']:.4f}")
            print("=" * 80)
            
            # Update best overall TCGA metrics
            if tcga_results['Overall_AUC'] > best_overall_tcga_metrics['best_overall_auc']:
                best_overall_tcga_metrics['best_overall_auc'] = tcga_results['Overall_AUC']
                best_overall_tcga_metrics['params'] = current_hyperparams
                best_overall_tcga_metrics['metrics'] = current_param_set_results
                

    
    # --- After all parameter combinations ---
    # Save all parameter results to a single JSON file
    all_results_path = os.path.join(outfolder, 'all_parameter_results_summary.json')
    with open(all_results_path, 'w') as f:
        json.dump({
            'all_parameter_sets_results': all_param_results_list,
            'best_performing_params_on_overall_tcga_auc': best_overall_tcga_metrics
        }, f, indent=4, cls=NpEncoder)
    
    # Create final comparative summary CSV for all parameter sets
    comparison_df = create_final_parameter_comparison_csv_5fold(outfolder, all_param_results_list)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    if best_overall_tcga_metrics['params']:
        print(f"\nBest combination found - Mean CV AUC: {best_overall_tcga_metrics['metrics']['Mean_Metrics']['val_AUC_mean']:.4f} ± {best_overall_tcga_metrics['metrics']['Mean_Metrics']['val_AUC_std']:.4f}")
        print(f"Overall TCGA AUC: {best_overall_tcga_metrics['metrics']['TCGA_Metrics']['Overall_AUC']:.4f}")
    else:
        print("No results found to determine the best parameters.")
    
    print(f"\nResults saved to: {outfolder}")
    print("="*80)
        
    return best_overall_tcga_metrics


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

def perform_5fold_cross_validation(response_df,
                                  expression_df,
                                  drug_smiles_df,
                                  pretrain_params,
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
                                  encoder_state_dict=None,
                                  test_df=None):
    """
    Perform 5-fold cross validation for a given parameter combination
    
    Returns:
        fold_results: List of dictionaries containing results for each fold
        mean_metrics: Dictionary containing mean and std of all metrics across folds
    """
    # Create 5-fold stratified split
    # Determine column names
    if 'DepMap_ID' in response_df.columns:
        id_column = 'DepMap_ID'
    elif 'ModelID' in response_df.columns:
        id_column = 'ModelID'
    else:
        raise ValueError("Neither 'DepMap_ID' nor 'ModelID' column found in response data")
    
    if 'Class' in response_df.columns:
        label_column = 'Class'
    elif 'Label' in response_df.columns:
        label_column = 'Label'
    else:
        raise ValueError("Neither 'Class' nor 'Label' column found in response data")
    
    all_samples = response_df[id_column].unique()
    all_labels = []
    
    # Get labels for stratification
    for sample_id in all_samples:
        sample_data = response_df[response_df[id_column] == sample_id]
        label = sample_data[label_column].iloc[0]  # Assuming all samples for same ID have same label
        all_labels.append(label)
    
    # Create stratified k-fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, all_labels)):
        # Starting fold
        
        # Create fold-specific model folder
        fold_model_folder = os.path.join(model_folder, f'fold_{fold+1}')
        safemakedirs(fold_model_folder)
        
        # Get train and validation samples
        train_samples = all_samples[train_idx]
        val_samples = all_samples[val_idx]
        
        # Create datasets
        train_df = response_df[response_df[id_column].isin(train_samples)]
        val_df = response_df[response_df[id_column].isin(val_samples)]
        
        train_subset = DrugResponseDataset(train_df, expression_df, drug_smiles_df)
        val_subset = DrugResponseDataset(val_df, expression_df, drug_smiles_df)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Initialize models for this fold
        # Always create encoder (VAE) with specified architecture
        encoder = VAE(input_size=GENE_INPUT_DIM, 
                    output_size=GENE_INPUT_DIM, 
                    latent_size=ENCODER_LATENT_DIM,
                    encoder_hidden_dims=pretrain_params['encoder_dims'],
                    decoder_hidden_dims=pretrain_params['encoder_dims'][::-1],
                    dop=pretrain_params['dropout_rate'],
                    act_fn=nn.ReLU).to(device)
        
        # Load pretrained weights and set training mode based on encoder_mode
        if model_params['encoder_mode'] == 'frozen':
            if encoder_state_dict is not None:
                encoder.load_state_dict(encoder_state_dict)
                print("Loaded pretrained encoder weights for frozen mode")
            # Freeze encoder parameters
            for param in encoder.parameters():
                param.requires_grad = False
            print("Encoder set to frozen mode (no training)")
        else:
            # trainable mode: don't load weights, keep trainable
            print("Encoder set to trainable mode (will train from scratch)")
        
        encoder_latent_dim = ENCODER_LATENT_DIM
        
        # Initialize GIN model based on gin_type
        if model_params['gin_type'] == 'precomputed':
            # For precomputed, still create a GIN model but it will be used differently
            drug_gcnmodel = GINConvNet(
                input_dim=DRUG_INPUT_DIM,
                output_dim=DRUG_EMBED_DIM,
                dropout=0.2,
                num_layers=5,
                jk_mode='last',
                use_batch_norm=True,
                pool_type='max'
            ).to(device)
            drug_embed_dim = DRUG_EMBED_DIM
        elif model_params['gin_type'] == 'dapl':
            drug_gcnmodel = GINConvNet(
                input_dim=DRUG_INPUT_DIM,
                output_dim=DRUG_EMBED_DIM,
                dropout=0.2,
                num_layers=5,
                jk_mode='last',
                use_batch_norm=True,
                pool_type='max'
            ).to(device)
            drug_embed_dim = DRUG_EMBED_DIM
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
            drug_embed_dim = DRUG_EMBED_DIM
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
            drug_embed_dim = DRUG_EMBED_DIM
        else:
            raise ValueError(f"Unknown gin_type: {model_params['gin_type']}")
        
        # Always apply init_weights to GIN model
        drug_gcnmodel.apply(init_weights)
        
        # Get activation function
        activation_map = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU
        }
        act_fn = activation_map[classifier_params['activation']]
        
        classifymodel = Classify(input_dim=encoder_latent_dim + drug_embed_dim,
                                hidden_dims=classifier_params['hidden_dims'],
                                dop=classifier_params['dropout_rate'], 
                                act_fn=act_fn,
                                out_fn=None,
                                use_bn=classifier_params['use_batch_norm']).to(device)
        classifymodel.apply(init_weights)
        
        model_components = {
            'encoder': encoder,
            'drug_model': drug_gcnmodel,
            'classifier': classifymodel
        }
        
        # Create optimizer and loss function
        # Include parameters based on encoder_mode
        optimizer_params = []
        
        if model_params['encoder_mode'] == 'trainable':
            optimizer_params.append({'params': encoder.parameters()})
        
        optimizer_params.extend([
            {'params': drug_gcnmodel.parameters()},
            {'params': classifymodel.parameters()}
        ])
        
        optimizer = optim.AdamW(optimizer_params, lr=ft_params['ftlr'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs) if ft_params['scheduler_flag'] else None
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Train the model
        best_val_auc = float('-inf')
        best_epoch = 0
        patience_counter = 0
        metrics_history = defaultdict(list)
        
        pbar = tqdm(range(num_epochs), desc=f"[Combination {param_combination_id}_fold_{fold+1}] Training", leave=False)
        for epoch in pbar:
            train_loss, train_auc, train_auprc = train_one_epoch(model_components, train_loader, optimizer, loss_fn)
            if scheduler:
                scheduler.step()
            val_loss, val_auc, val_auprc = evaluate_model(model_components, val_loader, loss_fn)
            
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
                # Save model state dict
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'drug_model_state_dict': drug_gcnmodel.state_dict(),
                    'classifier_state_dict': classifymodel.state_dict(),
                    'epoch': epoch,
                    'best_val_auc': best_val_auc,
                }, os.path.join(fold_model_folder, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    # Early stopping occurred
                    break
        
        print(f"[Combination {param_combination_id}_fold_{fold+1}] Training completed. Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch + 1}")
        
        # Load best model for testing and inference
        best_model_path = os.path.join(fold_model_folder, 'best_model.pth')
        checkpoint = torch.load(best_model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        drug_gcnmodel.load_state_dict(checkpoint['drug_model_state_dict'])
        classifymodel.load_state_dict(checkpoint['classifier_state_dict'])
        
        # Evaluate on validation set using best model with detailed metrics
        val_loss, val_metrics = evaluate_model(model_components, val_loader, loss_fn, calculate_detailed_metrics=True)
        
        # Evaluate on test set for this fold
        if test_df is not None:
            test_subset = DrugResponseDataset(test_df, expression_df, drug_smiles_df)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            test_loss, test_metrics = test_combination(
                model_components=model_components,
                test_loader=test_loader,
                loss_fn=loss_fn,
                param_combination_id=f"{param_combination_id}_fold_{fold+1}_test"
            )
            
            # Save test set confusion matrix for this fold
            test_confusion_matrix_path = os.path.join(fold_model_folder, 'test_confusion_matrix.png')
            plot_confusion_matrix(test_metrics['confusion_matrix'], test_confusion_matrix_path, f"Test Set Confusion Matrix - Fold {fold+1}")
        else:
            test_loss = np.nan
            test_metrics = {'AUC': np.nan, 'AUPRC': np.nan, 'sensitivity': np.nan, 'specificity': np.nan,
                          'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan, 'optimal_threshold': np.nan,
                          'confusion_matrix': None}
        
        # TCGA Inference for this fold
        tcga_results = inference_on_tcga_drugs(
            model_components=model_components,
            tcga_data_folder=tcga_inference_data_folder,
            best_model_path=None,  # Pass None to avoid reloading weights
            ft_params=ft_params,
            fold_model_folder=fold_model_folder,  # Pass folder info for confusion matrix saving
            tcga_tag='TCGA1'
        )
        
        # Calculate overall TCGA metrics for this fold
        tcga_results = calculate_overall_tcga_metrics(tcga_results)
        tcga_results_extra = {}
        if tcga_inference_data_folder_extra:
            tcga_results_extra = inference_on_tcga_drugs(
                model_components=model_components,
                tcga_data_folder=tcga_inference_data_folder_extra,
                best_model_path=None,
                ft_params=ft_params,
                fold_model_folder=fold_model_folder,
                tcga_tag='TCGA2'
            )
            tcga_results_extra = calculate_overall_tcga_metrics(tcga_results_extra)
        
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
            'tcga_results_extra': tcga_results_extra,
            'metrics_history': metrics_history,
            'best_model_path': best_model_path
        }
        
        fold_results.append(fold_result)
        
        # Fold completed
    
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
    tcga_metrics_to_aggregate = ['Overall_AUC', 'Overall_AUPRC', 'Overall_Sensitivity', 'Overall_Specificity', 
                                'Overall_Precision', 'Overall_Recall', 'Overall_F1_Score']
    
    for metric in tcga_metrics_to_aggregate:
        values = [fold['tcga_results'][metric] for fold in fold_results]
        mean_metrics[f'tcga_{metric}_mean'] = np.mean(values)
        mean_metrics[f'tcga_{metric}_std'] = np.std(values)
    
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

def calculate_overall_tcga_metrics(tcga_results):
    """Calculate mean metrics across all TCGA drugs"""
    all_drug_aucs = []
    all_drug_auprcs = []
    all_drug_sensitivities = []
    all_drug_specificities = []
    all_drug_precisions = []
    all_drug_recalls = []
    all_drug_f1_scores = []
    
    try:
        for drug_name, metrics in tcga_results.items():
            # Skip existing summary metrics
            if drug_name in ['Overall_AUC', 'Overall_AUPRC', 'Overall_Sensitivity', 'Overall_Specificity', 
                           'Overall_Precision', 'Overall_Recall', 'Overall_F1_Score']:
                continue
                
            # Only add valid metrics (non-NaN)
            if isinstance(metrics, dict):
                if 'AUC' in metrics and not pd.isna(metrics['AUC']):
                    all_drug_aucs.append(metrics['AUC'])
                if 'AUPRC' in metrics and not pd.isna(metrics['AUPRC']):
                    all_drug_auprcs.append(metrics['AUPRC'])
                if 'sensitivity' in metrics and not pd.isna(metrics['sensitivity']):
                    all_drug_sensitivities.append(metrics['sensitivity'])
                if 'specificity' in metrics and not pd.isna(metrics['specificity']):
                    all_drug_specificities.append(metrics['specificity'])
                if 'precision' in metrics and not pd.isna(metrics['precision']):
                    all_drug_precisions.append(metrics['precision'])
                if 'recall' in metrics and not pd.isna(metrics['recall']):
                    all_drug_recalls.append(metrics['recall'])
                if 'f1_score' in metrics and not pd.isna(metrics['f1_score']):
                    all_drug_f1_scores.append(metrics['f1_score'])
        
        # Calculate means only if we have valid values
        overall_auc = np.mean(all_drug_aucs) if all_drug_aucs else np.nan
        overall_auprc = np.mean(all_drug_auprcs) if all_drug_auprcs else np.nan
        overall_sensitivity = np.mean(all_drug_sensitivities) if all_drug_sensitivities else np.nan
        overall_specificity = np.mean(all_drug_specificities) if all_drug_specificities else np.nan
        overall_precision = np.mean(all_drug_precisions) if all_drug_precisions else np.nan
        overall_recall = np.mean(all_drug_recalls) if all_drug_recalls else np.nan
        overall_f1_score = np.mean(all_drug_f1_scores) if all_drug_f1_scores else np.nan
        
        # Return a dictionary with the overall metrics
        overall_metrics = {
            'Overall_AUC': overall_auc,
            'Overall_AUPRC': overall_auprc,
            'Overall_Sensitivity': overall_sensitivity,
            'Overall_Specificity': overall_specificity,
            'Overall_Precision': overall_precision,
            'Overall_Recall': overall_recall,
            'Overall_F1_Score': overall_f1_score
        }
        
        # Add overall metrics to the tcga_results dictionary
        tcga_results.update(overall_metrics)
        
    except Exception as e:
        print(f"Error calculating overall TCGA metrics: {e}")
        # Provide default values in case of error
        tcga_results.update({
            'Overall_AUC': np.nan,
            'Overall_AUPRC': np.nan,
            'Overall_Sensitivity': np.nan,
            'Overall_Specificity': np.nan,
            'Overall_Precision': np.nan,
            'Overall_Recall': np.nan,
            'Overall_F1_Score': np.nan
        })
    
    return tcga_results



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
    
    # Add TCGA metrics
    for drug, metrics in param_set_results['TCGA_Metrics'].items():
        if drug not in ['Overall_AUC', 'Overall_AUPRC', 'Overall_Sensitivity', 'Overall_Specificity', 
                       'Overall_Precision', 'Overall_Recall', 'Overall_F1_Score']:
            metrics_data[f'{drug}_TCGA_AUC'] = metrics['AUC']
            metrics_data[f'{drug}_TCGA_AUPRC'] = metrics['AUPRC']
            metrics_data[f'{drug}_TCGA_Sensitivity'] = metrics['sensitivity']
            metrics_data[f'{drug}_TCGA_Specificity'] = metrics['specificity']
            metrics_data[f'{drug}_TCGA_Precision'] = metrics['precision']
            metrics_data[f'{drug}_TCGA_Recall'] = metrics['recall']
            metrics_data[f'{drug}_TCGA_F1_Score'] = metrics['f1_score']
            metrics_data[f'{drug}_TCGA_Optimal_Threshold'] = metrics['optimal_threshold']
    
    # Add overall TCGA metrics
    metrics_data['Overall_TCGA_AUC'] = param_set_results['TCGA_Metrics']['Overall_AUC']
    metrics_data['Overall_TCGA_AUPRC'] = param_set_results['TCGA_Metrics']['Overall_AUPRC']
    metrics_data['Overall_TCGA_Sensitivity'] = param_set_results['TCGA_Metrics']['Overall_Sensitivity']
    metrics_data['Overall_TCGA_Specificity'] = param_set_results['TCGA_Metrics']['Overall_Specificity']
    metrics_data['Overall_TCGA_Precision'] = param_set_results['TCGA_Metrics']['Overall_Precision']
    metrics_data['Overall_TCGA_Recall'] = param_set_results['TCGA_Metrics']['Overall_Recall']
    metrics_data['Overall_TCGA_F1_Score'] = param_set_results['TCGA_Metrics']['Overall_F1_Score']
    tcga_extra = param_set_results.get('TCGA_Metrics_Extra', {})
    if tcga_extra:
        metrics_data['TCGA2_Overall_TCGA_AUC'] = tcga_extra.get('Overall_AUC', np.nan)
        metrics_data['TCGA2_Overall_TCGA_AUPRC'] = tcga_extra.get('Overall_AUPRC', np.nan)
        metrics_data['TCGA2_Overall_TCGA_Sensitivity'] = tcga_extra.get('Overall_Sensitivity', np.nan)
        metrics_data['TCGA2_Overall_TCGA_Specificity'] = tcga_extra.get('Overall_Specificity', np.nan)
        metrics_data['TCGA2_Overall_TCGA_Precision'] = tcga_extra.get('Overall_Precision', np.nan)
        metrics_data['TCGA2_Overall_TCGA_Recall'] = tcga_extra.get('Overall_Recall', np.nan)
        metrics_data['TCGA2_Overall_TCGA_F1_Score'] = tcga_extra.get('Overall_F1_Score', np.nan)
        for drug, metrics in tcga_extra.items():
            if drug not in ['Overall_AUC', 'Overall_AUPRC', 'Overall_Sensitivity', 'Overall_Specificity',
                           'Overall_Precision', 'Overall_Recall', 'Overall_F1_Score'] and isinstance(metrics, dict):
                metrics_data[f'TCGA2_{drug}_TCGA_AUC'] = metrics.get('AUC', np.nan)
                metrics_data[f'TCGA2_{drug}_TCGA_AUPRC'] = metrics.get('AUPRC', np.nan)
    
    # Save the metrics
    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(os.path.join(metrics_out_folder, 'metrics_summary.csv'), index=False)



def create_final_parameter_comparison_csv_5fold(outfolder, all_param_results_list):
    """Create CSV files comparing all hyperparameter combinations with 5-fold cross validation results"""
    if not all_param_results_list:
        print("No data to create final parameter comparison CSV.")
        return
    
    # Create a streamlined comparison table focusing on parameters and results
    comparison_rows = []
    
    for i, result in enumerate(all_param_results_list):
        # Extract key parameters and metrics
        row = {'ID': i + 1}  # Add ID starting from 1
        
        # Add parameters
        row['Pretrain_Epochs'] = result['Params']['pretrain_num_epochs']
        row['Train_Epochs'] = result['Params']['train_num_epochs']
        row['Pretrain_LR'] = result['Params']['pretrain_learning_rate']
        row['GAN_LR'] = result['Params']['gan_learning_rate']
        row['Dropout_Rate'] = result['Params']['dropout_rate']
        row['Encoder_Dims'] = str(result['Params']['encoder_dims'])
        row['Finetune_LR'] = result['Params']['ftlr']
        row['Scheduler'] = result['Params']['scheduler_flag']
        
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
        row['CV_Val_Sensitivity_Mean'] = mean_metrics['val_sensitivity_mean']
        row['CV_Val_Sensitivity_Std'] = mean_metrics['val_sensitivity_std']
        row['CV_Test_Sensitivity_Mean'] = mean_metrics['test_sensitivity_mean']
        row['CV_Test_Sensitivity_Std'] = mean_metrics['test_sensitivity_std']
        row['CV_Val_Specificity_Mean'] = mean_metrics['val_specificity_mean']
        row['CV_Val_Specificity_Std'] = mean_metrics['val_specificity_std']
        row['CV_Test_Specificity_Mean'] = mean_metrics['test_specificity_mean']
        row['CV_Test_Specificity_Std'] = mean_metrics['test_specificity_std']
        row['CV_Val_Precision_Mean'] = mean_metrics['val_precision_mean']
        row['CV_Val_Precision_Std'] = mean_metrics['val_precision_std']
        row['CV_Test_Precision_Mean'] = mean_metrics['test_precision_mean']
        row['CV_Test_Precision_Std'] = mean_metrics['test_precision_std']
        row['CV_Val_Recall_Mean'] = mean_metrics['val_recall_mean']
        row['CV_Val_Recall_Std'] = mean_metrics['val_recall_std']
        row['CV_Test_Recall_Mean'] = mean_metrics['test_recall_mean']
        row['CV_Test_Recall_Std'] = mean_metrics['test_recall_std']
        row['CV_Val_F1_Mean'] = mean_metrics['val_f1_score_mean']
        row['CV_Val_F1_Std'] = mean_metrics['val_f1_score_std']
        row['CV_Test_F1_Mean'] = mean_metrics['test_f1_score_mean']
        row['CV_Test_F1_Std'] = mean_metrics['test_f1_score_std']
        
        # Best fold information
        row['Best_Fold'] = result['Best_Fold']
        row['Best_Fold_AUC'] = result['Best_Fold_AUC']
        
        # Overall TCGA metrics
        row['Overall_TCGA_AUC'] = result['TCGA_Metrics']['Overall_AUC']
        row['Overall_TCGA_AUPRC'] = result['TCGA_Metrics']['Overall_AUPRC']
        row['Overall_TCGA_Sensitivity'] = result['TCGA_Metrics']['Overall_Sensitivity']
        row['Overall_TCGA_Specificity'] = result['TCGA_Metrics']['Overall_Specificity']
        row['Overall_TCGA_Precision'] = result['TCGA_Metrics']['Overall_Precision']
        row['Overall_TCGA_Recall'] = result['TCGA_Metrics']['Overall_Recall']
        row['Overall_TCGA_F1_Score'] = result['TCGA_Metrics']['Overall_F1_Score']
        
        # Add individual TCGA drug metrics
        for drug_name, metrics in result['TCGA_Metrics'].items():
            if drug_name not in ['Overall_AUC', 'Overall_AUPRC', 'Overall_Sensitivity', 'Overall_Specificity', 
                               'Overall_Precision', 'Overall_Recall', 'Overall_F1_Score']:
                row[f'{drug_name}_TCGA_AUC'] = metrics['AUC']
                row[f'{drug_name}_TCGA_AUPRC'] = metrics['AUPRC']
                row[f'{drug_name}_TCGA_Sensitivity'] = metrics['sensitivity']
                row[f'{drug_name}_TCGA_Specificity'] = metrics['specificity']
                row[f'{drug_name}_TCGA_Precision'] = metrics['precision']
                row[f'{drug_name}_TCGA_Recall'] = metrics['recall']
                row[f'{drug_name}_TCGA_F1_Score'] = metrics['f1_score']
                row[f'{drug_name}_TCGA_Optimal_Threshold'] = metrics['optimal_threshold']
        tcga_extra = result.get('TCGA_Metrics_Extra', {})
        if tcga_extra:
            row['TCGA2_Overall_TCGA_AUC'] = tcga_extra.get('Overall_AUC', np.nan)
            row['TCGA2_Overall_TCGA_AUPRC'] = tcga_extra.get('Overall_AUPRC', np.nan)
            row['TCGA2_Overall_TCGA_Sensitivity'] = tcga_extra.get('Overall_Sensitivity', np.nan)
            row['TCGA2_Overall_TCGA_Specificity'] = tcga_extra.get('Overall_Specificity', np.nan)
            row['TCGA2_Overall_TCGA_Precision'] = tcga_extra.get('Overall_Precision', np.nan)
            row['TCGA2_Overall_TCGA_Recall'] = tcga_extra.get('Overall_Recall', np.nan)
            row['TCGA2_Overall_TCGA_F1_Score'] = tcga_extra.get('Overall_F1_Score', np.nan)
            for drug_name, metrics in tcga_extra.items():
                if drug_name not in ['Overall_AUC', 'Overall_AUPRC', 'Overall_Sensitivity', 'Overall_Specificity',
                                   'Overall_Precision', 'Overall_Recall', 'Overall_F1_Score'] and isinstance(metrics, dict):
                    row[f'TCGA2_{drug_name}_TCGA_AUC'] = metrics.get('AUC', np.nan)
                    row[f'TCGA2_{drug_name}_TCGA_AUPRC'] = metrics.get('AUPRC', np.nan)
                    row[f'TCGA2_{drug_name}_TCGA_Sensitivity'] = metrics.get('sensitivity', np.nan)
                    row[f'TCGA2_{drug_name}_TCGA_Specificity'] = metrics.get('specificity', np.nan)
                    row[f'TCGA2_{drug_name}_TCGA_Precision'] = metrics.get('precision', np.nan)
                    row[f'TCGA2_{drug_name}_TCGA_Recall'] = metrics.get('recall', np.nan)
                    row[f'TCGA2_{drug_name}_TCGA_F1_Score'] = metrics.get('f1_score', np.nan)
                    row[f'TCGA2_{drug_name}_TCGA_Optimal_Threshold'] = metrics.get('optimal_threshold', np.nan)
        
        comparison_rows.append(row)
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Handle potential NaN values in the data
    comparison_df = comparison_df.fillna(0)  # Replace NaN with 0 to avoid groupby errors
    
    # Sort by Overall TCGA AUC (descending), handling NaN values correctly
    if 'Overall_TCGA_AUC' in comparison_df.columns:
        # Fill NaN with negative infinity for sorting purposes
        sort_column = comparison_df['Overall_TCGA_AUC'].copy()
        sort_column = sort_column.fillna(-float('inf'))
        comparison_df = comparison_df.iloc[sort_column.argsort()[::-1]]
    
    # Save the comparison table
    comparison_path = os.path.join(outfolder, 'parameter_comparison_tcga_focus.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
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
    print(f"1. {comparison_path} (TCGA-focused)")
    print(f"2. {detailed_path} (detailed)")
    
    return comparison_df




if __name__ == "__main__":
    import argparse
    
    # Start timer
    start_time = time.time()
    
    parser = argparse.ArgumentParser('finetune_zscore_optimized')
    parser.add_argument('--drug_model_path', dest='drug_model_path', type=str, 
                       default='./result/drug_encoder.pth', 
                       help='Pre-trained drug model GCN/GIN path')
    parser.add_argument('--pretrain_model_root_folder', dest='pretrain_model_root_folder', type=str, 
                       default='./result/pretrain', 
                       help='Root folder containing pre-trained encoder models')
    parser.add_argument('--outfolder', dest='outfolder', type=str, 
                       default='./result/classify_optimized', # Changed default outfolder
                       help='folder to save result')
    parser.add_argument('--batch_size', dest='batch_size', type=int, 
                       default=2048, 
                       help='batch size for training and validation')
    
    parser.add_argument('--response_data', type=str, default='./data/gdsc1+2_ccle_stacked_Z_SCORE.csv',
                        help='Path to the drug response data CSV file.')
    parser.add_argument('--expression_data', type=str, default='./data/ccle_uq1000_feature_sorted_match_pdx_tcga_zscore.csv',
                        help='Path to the gene expression data CSV file.')
    parser.add_argument('--config', type=str, default='config/params_grid_enhanced.json',
                        help='Path to the configuration file with parameter grids.')

    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Get parameter grids from config
    pretrain_params_grid = config['pretrain_params']
    fine_tune_params_grid = config['finetune_params']
    classifier_params_grid = config['classifier_params']
    model_params_grid = config['model_params']

    results = step_1_finetune_pipeline_zscore(
        pretrain_model_root_folder=args.pretrain_model_root_folder,
        drug_model_path=args.drug_model_path,
        outfolder=args.outfolder,
        response_data_path=args.response_data,
        expression_data_path=args.expression_data,
        drug_smiles_data_path=FIXED_DRUG_SMILES_DATA_PATH,
        tcga_inference_data_folder=FIXED_TCGA_DATA_FOLDER,
        tcga_inference_data_folder_extra=FIXED_TCGA_DATA_FOLDER_EXTRA,
        pretrain_params_grid=pretrain_params_grid,
        fine_tune_params_grid=fine_tune_params_grid,
        classifier_params_grid=classifier_params_grid,
        model_params_grid=model_params_grid,
        batch_size=args.batch_size,
        config=config
    )
    
    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_str = str(timedelta(seconds=int(execution_time)))
    print("\n" + "="*80)
    print(f"Total execution time: {execution_time_str}")
    print("="*80)