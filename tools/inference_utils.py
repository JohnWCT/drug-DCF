import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, confusion_matrix
from tools.dataprocess import safemakedirs

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find optimal threshold using Youden's index (sensitivity + specificity - 1)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_scores = tpr - fpr
    optimal_idx = np.argmax(youden_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return optimal_threshold, {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'youden_index': youden_scores[optimal_idx]
    }

def calculate_comprehensive_metrics(y_true, y_pred_proba):
    """
    Calculate comprehensive evaluation metrics including AUC, AUPRC, and binary classification metrics
    """
    try:
        auc_score = roc_auc_score(y_true, y_pred_proba)
        auprc_score = average_precision_score(y_true, y_pred_proba)
    except ValueError:
        auc_score = 0.5
        auprc_score = 0.0
        
    optimal_threshold, binary_metrics = find_optimal_threshold(y_true, y_pred_proba)
    
    return {
        'AUC': auc_score,
        'AUPRC': auprc_score,
        'optimal_threshold': optimal_threshold,
        **binary_metrics
    }

def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
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

from torch_geometric import data as DATA
from tools.dataprocess import smile_to_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TCGA_target_data:
    def __init__(self, drug_name, tcga_response_df, tcga_latent_dict):
        # Initialize attributes
        self.target_data = torch.empty(0, 32).float().to(device)
        self.target_labels = torch.empty(0).float().to(device)
        self.patient_ids = []
        self.drug_name = drug_name
        
        if tcga_latent_dict is None:
             # Just return empty if no latents (will be handled by caller)
            return

        try:
            # Filter for specific drug
            # Case-insensitive match for robustness
            drug_df = tcga_response_df[tcga_response_df['drug_name'].astype(str).str.lower() == drug_name.lower()]
            
            if drug_df.empty:
                # No samples for this drug
                return
            
            # Get valid samples (Patient_id)
            # TCGA latent keys are like 'TCGA-OR-A5J1-01', response IDs are 'TCGA-OR-A5J1'
            # We need to map response IDs to latent keys
            
            # Create mapping from Patient_ID (short) to Latent_Key (long)
            # Warning: A patient might have multiple samples (01, 06 etc). 
            # Logic: Use the first matching latent key for a patient ID.
            
            patient_to_latent_key = {}
            for latent_key in tcga_latent_dict.keys():
                # Extract patient ID (first 3 parts: TCGA-XX-XXXX)
                parts = latent_key.split('-')
                if len(parts) >= 3:
                    patient_id = '-'.join(parts[:3])
                    # Store if not exists (taking first one)
                    if patient_id not in patient_to_latent_key:
                        patient_to_latent_key[patient_id] = latent_key
            
            # Find common patients
            latent_list = []
            labels_list = []
            patient_ids = []
            
            for _, row in drug_df.iterrows():
                patient_id = row['Patient_id']
                label = row['Label']
                
                if patient_id in patient_to_latent_key:
                    latent_key = patient_to_latent_key[patient_id]
                    if latent_key in tcga_latent_dict:
                        latent_list.append(tcga_latent_dict[latent_key])
                        labels_list.append(label)
                        patient_ids.append(str(patient_id))
            
            if len(latent_list) == 0:
                # No common samples
                return
            
            # Convert to tensors
            self.target_data = torch.tensor(np.array(latent_list), dtype=torch.float32).to(device)
            self.target_labels = torch.tensor(np.array(labels_list), dtype=torch.float32).to(device)
            self.patient_ids = patient_ids
            
        except Exception as e:
            print(f"Error loading TCGA data for drug {drug_name}: {e}")
            raise e

def inference_on_tcga_drugs(model_components, tcga_data_path, best_model_path, ft_params, 
                           tcga_latent_dict, drug_latent_dict=None, gin_type='precomputed', fold_model_folder=None,
                           drug_smiles_df=None, tcga_tag='TCGA'):
    """
    Run inference on TCGA drugs using a single response CSV file.
    Calculates Global (pooled) and Average (mean of per-drug) scores.
    """
    # 1. Load TCGA Response Data (CSV only)
    if not (os.path.isfile(tcga_data_path) and str(tcga_data_path).lower().endswith(".csv")):
        raise ValueError(
            f"tcga_data_path must be a TCGA response CSV file, got: {tcga_data_path}"
        )
    from tools.finetune_tcga_eval import load_tcga_response_csv

    try:
        tcga_response_df = load_tcga_response_csv(tcga_data_path)
    except ValueError as exc:
        print(f"Error: {exc}")
        return {}

    # Identify unique drugs dynamically
    if 'drug_name' not in tcga_response_df.columns:
        print("Error: 'drug_name' column not found in TCGA CSV.")
        return {}
        
    unique_drugs = tcga_response_df['drug_name'].unique()
    
    # Initialize model
    if model_components is None:
        print("Warning: model_components is None. Skipping TCGA inference.")
        return {}
    
    drug_model = model_components['drug_model']
    classifier = model_components['classifier']
    
    # Load best model if provided
    if best_model_path is not None and os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            if drug_model is not None and 'drug_model_state_dict' in checkpoint:
                drug_model.load_state_dict(checkpoint['drug_model_state_dict'])
        except Exception as e:
            print(f"Error loading best model: {e}")
            print(f"Continuing with current model state...")
    
    if drug_model is not None:
        drug_model.eval()
    classifier.eval()
    
    from tools.prediction_export import build_tcga_prediction_rows

    results = {
        'Global_Metrics': {},
        'Average_Metrics': {},
        'Drug_Metrics': {},
        'Sample_Predictions': [],
    }
    
    # Accumulators for Global Score
    global_preds = []
    global_targets = []
    
    # Accumulators for Average Score
    per_drug_metrics = {
        'AUC': [], 'AUPRC': [], 
        'sensitivity': [], 'specificity': [], 
        'precision': [], 'recall': [], 'f1_score': []
    }
    
    # Initialize all drugs with NaN metrics to ensure they appear in output
    for drug_name in unique_drugs:
        results['Drug_Metrics'][drug_name] = {
            'AUC': np.nan,
            'AUPRC': np.nan
        }
    
    print(f"Evaluating on {len(unique_drugs)} TCGA drugs...")
    
    # Create master mapping for SMILES lookup if needed (dapl mode)
    # Using the passed drug_smiles_df if available, or predefined list
    drug_smiles_map = {}
    if drug_smiles_df is not None:
        # Create map from various name columns to SMILES
        for idx in drug_smiles_df.index:
            drug_smiles_map[str(idx).lower()] = drug_smiles_df.loc[idx, 'SMILES']
            if 'DRUG_NAME' in drug_smiles_df.columns:
                name = str(drug_smiles_df.loc[idx, 'DRUG_NAME']).lower()
                drug_smiles_map[name] = drug_smiles_df.loc[idx, 'SMILES']
    
    
    # Validation: Ensure we have a map
    if not drug_smiles_map and gin_type != 'precomputed':
        print("Warning: No drug SMILES map available for graph inference.")


    count_valid_drugs = 0

    for drug_name in unique_drugs:
        try:
            data_generator = TCGA_target_data(drug_name, tcga_response_df, tcga_latent_dict)
            target_data, target_labels = data_generator.target_data, data_generator.target_labels
            
            # Skip if no data
            if target_data.shape[0] == 0:
                continue
            
            with torch.no_grad():
                # Prepare Drug Input
                combined_input = None
                
                if gin_type == 'precomputed':
                    # Look up in latent dict
                    # Try exact, then lower, then partial match
                    drug_key = None
                    if drug_latent_dict:
                        if drug_name in drug_latent_dict: drug_key = drug_name
                        elif drug_name.lower() in drug_latent_dict: drug_key = drug_name.lower()
                    
                    if drug_key is None:
                        # Skip this drug if no latent feature
                        continue
                        
                    drug_emb = torch.tensor(drug_latent_dict[drug_key], dtype=torch.float32).to(device)
                    num_samples = target_data.shape[0]
                    drug_emb_batch = drug_emb.expand(num_samples, -1)
                    combined_input = torch.cat((target_data, drug_emb_batch), dim=1)
                    
                else:
                    # SMILES mode (dapl)
                    smile = None
                    # Try map first
                    drug_lower = drug_name.lower()
                    if drug_lower in drug_smiles_map:
                        smile = drug_smiles_map[drug_lower]
                    if drug_lower in drug_smiles_map:
                        smile = drug_smiles_map[drug_lower]
                    
                    if not smile:
                        # Try to find loose match in map keys
                        for k in drug_smiles_map:
                            if drug_lower in k or k in drug_lower:
                                smile = drug_smiles_map[k]
                                break
                    
                    if not smile:
                        print(f"  Skipping {drug_name}: No SMILES found.")
                        continue
                        
                    # Generate Graph
                    try:
                        _, atom_features_list, edge_index = smile_to_graph(smile)
                        drug_x = torch.tensor(np.array(atom_features_list), dtype=torch.float32)
                        if len(edge_index) > 0:
                            drug_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                        else:
                            drug_edge_index = torch.empty((2, 0), dtype=torch.long)
                        
                        drug_data = DATA.Data(x=drug_x, edge_index=drug_edge_index).to(device)
                        
                        # Get embedding
                        drug_emb_single = drug_model(drug_data)
                        if len(drug_emb_single.shape) == 1:
                            drug_emb_single = drug_emb_single.unsqueeze(0)
                        
                        num_samples = target_data.shape[0]
                        drug_emb_batch = drug_emb_single.expand(num_samples, -1)
                        combined_input = torch.cat((target_data, drug_emb_batch), dim=1)
                    except Exception as e:
                        print(f"  Graph error for {drug_name}: {e}")
                        continue

                # Run Inference
                pred = classifier(combined_input).view(-1)
                pred_probs = torch.sigmoid(pred)
                
                # Convert to numpy
                cur_preds = pred_probs.cpu().numpy()
                cur_targets = target_labels.cpu().numpy()
                
                # Check NaNs
                if np.isnan(cur_preds).any():
                    cur_preds = np.nan_to_num(cur_preds)
                
                # 1. Store for Global Metrics
                global_preds.extend(cur_preds)
                global_targets.extend(cur_targets)

                patient_ids = getattr(data_generator, "patient_ids", [])
                if len(patient_ids) == len(cur_preds):
                    results['Sample_Predictions'].extend(
                        build_tcga_prediction_rows(
                            patient_ids=patient_ids,
                            drug_name=drug_name,
                            ground_truth=cur_targets,
                            confidence=cur_preds,
                            tcga_source=tcga_tag,
                        )
                    )
                
                # 2. Calculate Individual Metrics
                try:
                    metrics = calculate_comprehensive_metrics(cur_targets, cur_preds)
                    
                    # Store results for this drug
                    results['Drug_Metrics'][drug_name] = {
                        'AUC': metrics['AUC'],
                        'AUPRC': metrics['AUPRC']
                    }
                    
                    # Accumulate for Average Metrics (only valid scores)
                    if not np.isnan(metrics['AUC']):
                        per_drug_metrics['AUC'].append(metrics['AUC'])
                    if not np.isnan(metrics['AUPRC']):
                        per_drug_metrics['AUPRC'].append(metrics['AUPRC'])
                        
                    count_valid_drugs += 1
                    
                    # Optional: Save Confusion Matrix
                    if fold_model_folder:
                         cm_folder = os.path.join(fold_model_folder, 'tcga_cm')
                         safemakedirs(cm_folder)
                         safe_drug_name = ''.join(ch if str(ch).isalnum() else '_' for ch in str(drug_name))
                         cm_path = os.path.join(cm_folder, f'{tcga_tag}_{safe_drug_name}_cm.png')
                         plot_confusion_matrix(metrics['confusion_matrix'], cm_path, f"{tcga_tag} - {drug_name}")

                except Exception as e_metrics:
                    print(f"Metric error for {drug_name}: {e_metrics}")

        except Exception as e:
            print(f"Error processing drug {drug_name}: {e}")
            continue

    # --- Final Calculations ---
    
    # 1. Global Metrics (Pooled)
    if global_targets:
        try:
            global_metrics = calculate_comprehensive_metrics(np.array(global_targets), np.array(global_preds))
            results['Global_Metrics'] = {
                'AUC': global_metrics['AUC'],
                'AUPRC': global_metrics['AUPRC'],
                'f1_score': global_metrics['f1_score']
            }
        except Exception as e:
            print(f"Global metric calc error: {e}")
            results['Global_Metrics'] = {'AUC': np.nan, 'AUPRC': np.nan}
    else:
        results['Global_Metrics'] = {'AUC': np.nan, 'AUPRC': np.nan}

    # 2. Average Metrics (Mean of Means)
    avg_results = {}
    for key, val_list in per_drug_metrics.items():
        if val_list:
            avg_results[key] = np.mean(val_list)
        else:
            avg_results[key] = np.nan
    
    results['Average_Metrics'] = {
        'AUC': avg_results['AUC'],
        'AUPRC': avg_results['AUPRC']
    }
    
    print(f"TCGA Inference Complete. Valid drugs: {count_valid_drugs}/{len(unique_drugs)}")
    print(f"Global AUC: {results['Global_Metrics']['AUC']:.4f} | Average AUC: {results['Average_Metrics']['AUC']:.4f}")
    
    return results

def calculate_overall_tcga_metrics(tcga_results):
    """
    Calculate average metrics across all TCGA drugs.
    """
    metrics = ['AUC', 'AUPRC', 'sensitivity', 'specificity', 'precision', 'recall', 'f1_score']
    
    # Initialize accumulators
    aggregated = {metric: [] for metric in metrics}
    
    for drug_code, res in tcga_results.items():
        for metric in metrics:
            if metric in res and not np.isnan(res[metric]):
                aggregated[metric].append(res[metric])
    
    # Calculate means
    overall_results = tcga_results.copy()
    
    for metric in metrics:
        if aggregated[metric]:
            mean_val = np.mean(aggregated[metric])
            overall_results[f'Overall_{metric}'] = mean_val
        else:
            overall_results[f'Overall_{metric}'] = np.nan
            
    return overall_results
