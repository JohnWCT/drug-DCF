"""Shared data loading/preprocessing utilities."""

import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('use device:', device)
from torch.utils.data import TensorDataset, DataLoader, Subset
import random

def normalize_gene_expression(data, method='zscore'):
    """
    對每個基因進行標準化
    Args:
        data: 基因表達數據 numpy array 或 pandas DataFrame
        method: 標準化方法 ('zscore', 'none', 'log')
    Returns:
        normalized_data: 標準化後的數據
    """
    if method == 'none':
        return data
    elif method == 'log':
        # Apply log(x + 0.001) transform
        if isinstance(data, pd.DataFrame):
            return np.log(data + 0.001)
        elif isinstance(data, np.ndarray):
            return np.log(data + 0.001)
        else:
            raise ValueError("Data must be pandas DataFrame or numpy array")
    elif method == 'zscore':
        if isinstance(data, pd.DataFrame):
            # 對每個基因進行標準化
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            normalized = (data - mean) / (std + 1e-8)
            return normalized
        elif isinstance(data, np.ndarray):
            # 對每個基因進行標準化
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True)
            normalized = (data - mean) / (std + 1e-8)
            return normalized
        else:
            raise ValueError("Data must be pandas DataFrame or numpy array")
    else:
        raise ValueError(f"Unsupported normalization method: {method}. Supported methods: 'zscore', 'none', 'log'")

def compute_class_weights(labels, device='cuda'):
    """
    Compute class weights based on label distribution.
    
    Args:
        labels (torch.Tensor or numpy.ndarray): Integer labels for which to compute weights
        device (str): Device on which to return the weights tensor
        
    Returns:
        torch.Tensor: Class weights tensor with shape (num_classes,)
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Count occurrences of each class
    class_counts = np.bincount(labels)
    
    # Compute weights as inverse of frequency
    # Add small epsilon to avoid division by zero
    weights = 1.0 / (class_counts + 1e-6)
    
    # Normalize weights to sum to number of classes
    weights = weights * (len(class_counts) / weights.sum())
    
    # Convert to tensor and move to specified device
    weights_tensor = torch.from_numpy(weights).float().to(device)
    
    return weights_tensor

def load_data_with_labels_and_weights(ccle_path, xena_path, batch_size=128, device='cuda', normalization_method='zscore'):
    """
    Load CCLE and TCGA data with cancer type labels and compute class weights
    
    Args:
        ccle_path (str): Path to CCLE data
        xena_path (str): Path to TCGA data
        batch_size (int): Batch size for DataLoader
        device (str): Device to load tensors on
        normalization_method (str): Normalization method ('zscore', 'none', 'log')
        
    Returns:
        tuple: ((source_loader, source_test_tensor, source_test_labels_tensor, source_weights),
                (target_loader, target_test_tensor, target_test_labels_tensor, target_weights),
                mapping_int2str)
    """
    # First get data using the original function
    (source_loader, source_test_tensor, source_test_labels_tensor), \
    (target_loader, target_test_tensor, target_test_labels_tensor), \
    mapping_int2str = load_data_with_labels(ccle_path, xena_path, batch_size, device, normalization_method=normalization_method)
    
    # Extract training labels from the loaders
    source_train_labels = []
    for _, labels in source_loader:
        source_train_labels.append(labels)
    source_train_labels = torch.cat(source_train_labels)
    
    target_train_labels = []
    for _, labels in target_loader:
        target_train_labels.append(labels)
    target_train_labels = torch.cat(target_train_labels)
    
    # Compute class weights for source and target datasets
    source_weights = compute_class_weights(source_train_labels, device)
    target_weights = compute_class_weights(target_train_labels, device)
    
    return ((source_loader, source_test_tensor, source_test_labels_tensor, source_weights),
            (target_loader, target_test_tensor, target_test_labels_tensor, target_weights),
            mapping_int2str)

def pretrain_data():
    ccle_df = pd.read_csv(os.path.join('data', 'pretrain_ccle.csv'), index_col=0, header=0)
    xena_df = pd.read_csv(os.path.join('data', 'pretrain_tcga.csv'), index_col=0, header=0)
    ccle_sample_info_df = pd.read_csv(os.path.join('data', 'ccle_sample_info_df.csv'), index_col=0, header=0)
    xena_sample_info_df = pd.read_csv(os.path.join('data', 'xena_sample_info_df.csv'), index_col=0, header=0)

    excluded_ccle_samples = []
    excluded_ccle_samples.extend(ccle_df.index.difference(ccle_sample_info_df.index))
    excluded_ccle_diseases = ccle_sample_info_df.primary_disease.value_counts()[
        ccle_sample_info_df.primary_disease.value_counts() < 2].index
    excluded_ccle_samples.extend(
        ccle_sample_info_df[ccle_sample_info_df.primary_disease.isin(excluded_ccle_diseases)].index)
    to_split_ccle_df = ccle_df[~ccle_df.index.isin(excluded_ccle_samples)]
    train_ccle_df, test_ccle_df = train_test_split(to_split_ccle_df, test_size=0.1,
                                                stratify=ccle_sample_info_df.loc[
                                                    to_split_ccle_df.index].primary_disease)
    test_ccle_df = test_ccle_df.append(ccle_df.loc[excluded_ccle_samples])
    train_xena_df, test_xena_df = train_test_split(xena_df, test_size=len(test_ccle_df) / len(xena_df),
                                                   stratify=xena_sample_info_df['_primary_disease'],
                                                   random_state=2020)
    # # gene function data:DATA.Data num:1284
    # gene_function_adj_df = pd.read_csv('gene_function_adj.csv', index_col=0, header=0)
    # # gene function adj 1284->1426
    # gene_list = ccle_df.columns.to_list()
    # gene_function_adj_df = gene_function_adj_df.reindex(index=gene_list, columns=gene_list, fill_value=0)
    # gene_function_adj_df = gene_function_adj_df.loc[gene_list, gene_list]
    # gene_function_adj_tensor = torch.from_numpy(gene_function_adj_df.values).type(torch.float32).to(device)
    # row_indices, col_indices = torch.where(gene_function_adj_tensor != 0)
    # edge_index = torch.stack([row_indices, col_indices], dim=0).to(gene_function_adj_tensor.device)
    # create dataloader
    ccle_tensor = torch.from_numpy(ccle_df.values).type(torch.float32).to(device)
    ccle_test_tensor = torch.from_numpy(test_ccle_df.values).type(torch.float32).to(device)
    tcga_tensor = torch.from_numpy(xena_df.values).type(torch.float32).to(device)
    tcga_test_tensor = torch.from_numpy(test_xena_df.values).type(torch.float32).to(device)

    # dataloader
    batch_size = 64
    ccleDataset = TensorDataset(ccle_tensor)
    ccleloader = DataLoader(ccleDataset, batch_size=batch_size, shuffle=True, drop_last=True)

    tcgaDataset = TensorDataset(tcga_tensor)
    tcgaloader = DataLoader(tcgaDataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return (ccleloader, ccle_test_tensor), (tcgaloader, tcga_test_tensor)

def load_data_with_labels(ccle_path, xena_path, batch_size=128, device='cuda', normalization_method='zscore'):
    """
    Load CCLE and TCGA data with cancer type labels for visualization
    
    Args:
        ccle_path (str): Path to CCLE data
        xena_path (str): Path to TCGA data
        batch_size (int): Batch size for DataLoader
        device (str): Device to load tensors on
        normalization_method (str): Normalization method ('zscore', 'none', 'log')
        
    Returns:
        tuple: ((source_loader, source_test_tensor, source_test_labels_tensor),
                (target_loader, target_test_tensor, target_test_labels_tensor),
                mapping_int2str)
    """
    min_count = 10 # 低頻病種：出現次數 <10 in ccle

    # Load CCLE data
    ccle_df = pd.read_csv(ccle_path, index_col=0)
    ccle_sample_info_df = pd.read_csv(os.path.join('data', 'ccle_sample_info_df.csv'), index_col=0, header=0)
    # Load TCGA data
    xena_df = pd.read_csv(xena_path, index_col=0)
    xena_sample_info_df = pd.read_csv(os.path.join('data', 'xena_sample_info_df.csv'), index_col=0, header=0)
    # 針對原始完整資料進行標準化
    print(f"Applying {normalization_method} normalization to complete datasets...")
    ccle_df = normalize_gene_expression(ccle_df, method=normalization_method)
    xena_df = normalize_gene_expression(xena_df, method=normalization_method)
    # Ensure all indices are strings
    ccle_df.index = ccle_df.index.astype(str)
    xena_df.index = xena_df.index.astype(str)
    ccle_sample_info_df.index = ccle_sample_info_df.index.astype(str)
    xena_sample_info_df.index = xena_sample_info_df.index.astype(str)
    
    # Map TCGA cancer types to CCLE cancer types
    target_to_source_map = {
        'acute myeloid leukemia': 'na', #Only solid tumors
        'adrenocortical cancer': 'na',
        'bladder urothelial carcinoma': 'Bladder Cancer',
        'brain lower grade glioma': 'Brain Cancer',
        'breast invasive carcinoma': 'Breast Cancer',
        'cervical & endocervical cancer': 'Cervical Cancer',
        'cholangiocarcinoma': 'Bile Duct Cancer',
        'colon adenocarcinoma': 'Colon/Colorectal Cancer',
        'diffuse large B-cell lymphoma': 'na', #Only solid tumors
        'esophageal carcinoma': 'Esophageal Cancer',
        'glioblastoma multiforme': 'Brain Cancer',
        'head & neck squamous cell carcinoma': 'Head and Neck Cancer',
        'kidney chromophobe': 'Kidney Cancer',
        'kidney clear cell carcinoma': 'Kidney Cancer',
        'kidney papillary cell carcinoma': 'Kidney Cancer',
        'liver hepatocellular carcinoma': 'Liver Cancer',
        'lung adenocarcinoma': 'Lung Cancer',
        'lung squamous cell carcinoma': 'Lung Cancer',
        'mesothelioma': 'na',
        'ovarian serous cystadenocarcinoma': 'Ovarian Cancer',
        'pancreatic adenocarcinoma': 'Pancreatic Cancer',
        'pheochromocytoma & paraganglioma': 'na',
        'prostate adenocarcinoma': 'Prostate Cancer',
        'rectum adenocarcinoma': 'Colon/Colorectal Cancer',
        'sarcoma': 'Sarcoma',
        'skin cutaneous melanoma': 'Skin Cancer',
        'stomach adenocarcinoma': 'Gastric Cancer',
        'testicular germ cell tumor': 'na',
        'thymoma': 'na',
        'thyroid carcinoma': 'Thyroid Cancer',
        'uterine carcinosarcoma': 'Endometrial/Uterine Cancer',
        'uterine corpus endometrioid carcinoma': 'Endometrial/Uterine Cancer',
        'uveal melanoma': 'Eye Cancer'
    }
    
    # Apply mapping to TCGA sample info
    xena_sample_info_df["_primary_disease"] = xena_sample_info_df["_primary_disease"].map(target_to_source_map)
    
    # ----------------------------
    # 來源資料 (CCLE) 處理
    # ----------------------------
    # 1. 過濾掉來源中 sample_info 不存在的樣本
    excluded_ccle_samples = list(set(ccle_df.index) - set(ccle_sample_info_df.index))
    
    # 2. 過濾掉低頻病種 (出現次數 < min_count)
    excluded_ccle_diseases = ccle_sample_info_df.primary_disease.value_counts()[
        ccle_sample_info_df.primary_disease.value_counts() < min_count].index
    excluded_ccle_samples.extend(
        list(ccle_sample_info_df[ccle_sample_info_df.primary_disease.isin(excluded_ccle_diseases)].index)
    )
    
    # 3. 僅保留 sample_info 中存在的資料
    to_split_ccle_df = ccle_df.loc[ccle_df.index.difference(excluded_ccle_samples)]
    
    # 4. 保留 primary_disease 非空
    valid_source_idx = to_split_ccle_df.index.intersection(
        ccle_sample_info_df[ccle_sample_info_df.primary_disease.notna()].index
    )
    filtered_ccle_df = to_split_ccle_df.loc[valid_source_idx]
    source_labels = ccle_sample_info_df.loc[valid_source_idx, "primary_disease"]
    
    # ----------------------------
    # 目標資料 (TCGA) 處理
    # ----------------------------
    # 確保 _primary_disease 非空
    valid_target_idx = xena_df.index.intersection(
        xena_sample_info_df[xena_sample_info_df["_primary_disease"].notna()].index
    )
    filtered_xena_df = xena_df.loc[valid_target_idx]
    target_labels = xena_sample_info_df.loc[valid_target_idx, "_primary_disease"]
    
    # ----------------------------
    # 僅保留交集病種
    # ----------------------------
    common_labels = set(source_labels.unique()) & set(target_labels.unique())
    common_labels = {label for label in common_labels if label != 'na' and not pd.isna(label)}
    
    ccle_mask = source_labels.isin(common_labels)
    xena_mask = target_labels.isin(common_labels)
    
    filtered_ccle_df = filtered_ccle_df[ccle_mask]
    source_labels = source_labels[ccle_mask]
    filtered_xena_df = filtered_xena_df[xena_mask]
    target_labels = target_labels[xena_mask]
    
    # Split data into train/test using stratified split to maintain label distribution
    train_ccle_df, test_ccle_df, source_train_labels, source_test_labels = train_test_split(
        filtered_ccle_df, source_labels, test_size=0.2, 
        stratify=source_labels, random_state=42
    )
    
    train_xena_df, test_xena_df, target_train_labels, target_test_labels = train_test_split(
        filtered_xena_df, target_labels, test_size=0.2, 
        stratify=target_labels, random_state=42
    )
    
    # Create mapping from disease names to integers
    common_labels = sorted(list(common_labels))
    mapping = {d: i for i, d in enumerate(common_labels)}
    mapping_int2str = {i: d for d, i in mapping.items()}
    
    # Convert labels to integers
    source_train_labels_int = np.array([mapping[label] for label in source_train_labels])
    source_test_labels_int = np.array([mapping[label] for label in source_test_labels])
    target_train_labels_int = np.array([mapping[label] for label in target_train_labels])
    target_test_labels_int = np.array([mapping[label] for label in target_test_labels])
    

    
    # Convert to tensors
    source_train_tensor = torch.from_numpy(train_ccle_df.values).type(torch.float32).to(device)
    source_test_tensor = torch.from_numpy(test_ccle_df.values).type(torch.float32).to(device)
    target_train_tensor = torch.from_numpy(train_xena_df.values).type(torch.float32).to(device)
    target_test_tensor = torch.from_numpy(test_xena_df.values).type(torch.float32).to(device)
    
    source_train_labels_tensor = torch.from_numpy(source_train_labels_int).to(device)
    source_test_labels_tensor = torch.from_numpy(source_test_labels_int).to(device)
    target_train_labels_tensor = torch.from_numpy(target_train_labels_int).to(device)
    target_test_labels_tensor = torch.from_numpy(target_test_labels_int).to(device)
    
    # Create DataLoaders
    sourcedataset = TensorDataset(source_train_tensor, source_train_labels_tensor)
    source_loader = DataLoader(sourcedataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    targetdataset = TensorDataset(target_train_tensor, target_train_labels_tensor)
    target_loader = DataLoader(targetdataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return ((source_loader, source_test_tensor, source_test_labels_tensor),
            (target_loader, target_test_tensor, target_test_labels_tensor),
            mapping_int2str)

def pretrain_loader(df:pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_tensor = torch.from_numpy(train_df.values).type(torch.float32).to(device)
    test_tensor = torch.from_numpy(test_df.values).type(torch.float32).to(device)
    batch_size = 64
    train_dataset = TensorDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = TensorDataset(test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader

def PDTC_source_5fold(drug):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [drug.lower()]
    gdsc_target_file1 = os.path.join('data', 'GDSC1_fitted_dose_response_25Feb20.csv')
    gdsc_target_file2 = os.path.join('data', 'GDSC2_fitted_dose_response_25Feb20.csv')
    gdsc1_response = pd.read_csv(gdsc_target_file1)
    gdsc2_response = pd.read_csv(gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')
    ccle_sample_file = os.path.join('data', 'ccle_sample_info.csv')
    ccle_sample_info = pd.read_csv(ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')
    gdsc_sample_file = os.path.join('data', 'gdsc_cell_line_annotation.csv')
    gdsc_sample_info = pd.read_csv(gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    # gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.iloc[:, 8].dropna().index]
    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
        ['DepMap_ID']]
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']
    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]
    ccle_target_df = target_df[drugs_to_keep[0]]
    ccle_target_df.dropna(inplace=True)
    gex_feature_file = os.path.join('data', 'uq1000_feature.csv')
    gex_features_df = pd.read_csv(gex_feature_file, index_col=0)
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)
    s_kfold = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.values[train_index], \
                                                      ccle_labeled_feature_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]
        # df->tensor
        ccle_train_data = torch.from_numpy(train_labeled_ccle_df).type(torch.float32).to(device)
        ccle_train_label = torch.from_numpy(train_ccle_labels).type(torch.float32).squeeze().to(device)
        ccle_test_data = torch.from_numpy(test_labeled_ccle_df).type(torch.float32).to(device)
        ccle_test_label = torch.from_numpy(test_ccle_labels).type(torch.float32).squeeze().to(device)

        yield (ccle_train_data, ccle_train_label), (ccle_test_data, ccle_test_label)

def PDTC_target_data(drug):
    pdtc_gex_file = os.path.join('data', 'pdtc_uq1000_feature.csv')
    pdtc_features_df = pd.read_csv(pdtc_gex_file, index_col=0)
    pdtc_target_file = os.path.join('data', 'DrugResponsesAUCModels.txt')
    target_df = pd.read_csv(pdtc_target_file, index_col=0, sep='\t')
    drug_target_df = target_df.loc[target_df.Drug == drug]
    labeled_samples = drug_target_df.index.intersection(pdtc_features_df.index)
    drug_target_vec = drug_target_df.loc[labeled_samples, 'AUC']
    drug_feature_df = pdtc_features_df.loc[labeled_samples]
    threshold = np.median(drug_target_vec)
    drug_label_vec = (drug_target_vec < threshold).astype('int')
    pdtc_features = torch.from_numpy(drug_feature_df.values).type(torch.float32).to(device)
    pdtc_label = torch.from_numpy(drug_label_vec.values).type(torch.float32).squeeze().to(device)
    return (pdtc_features, pdtc_label)

def PDTC_data_generator(drug):
    drug_mapping_df = pd.read_csv(os.path.join('data', 'pdtc_gdsc_drug_mapping.csv'), index_col=0)
    drug_name = drug_mapping_df.loc[drug, 'drug_name']
    pdtc_data = PDTC_target_data(drug_name)
    gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
    ccle_data_tuple = PDTC_source_5fold(gdsc_drug)
    for ccle_train_data, ccle_eval_data in ccle_data_tuple:
        yield (ccle_train_data, ccle_eval_data, pdtc_data)


def TCGA_source_5fold(drug):
    # data df gene_num:1426
    ccle_features_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'ccledata.csv'), index_col=0, header=0)
    ccle_label_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'cclelabel.csv'), index_col=0, header=0)
    # split 5-fold
    s_kfold = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for train_index, test_index in s_kfold.split(ccle_features_df.values, ccle_label_df.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_features_df.values[train_index], \
                                                    ccle_features_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_label_df.values[train_index], ccle_label_df.values[test_index]
        # df->tensor
        ccle_train_data = torch.from_numpy(train_labeled_ccle_df).type(torch.float32).to(device)
        ccle_train_label = torch.from_numpy(train_ccle_labels).type(torch.float32).squeeze().to(device)
        ccle_test_data = torch.from_numpy(test_labeled_ccle_df).type(torch.float32).to(device)
        ccle_test_label = torch.from_numpy(test_ccle_labels).type(torch.float32).squeeze().to(device)

        yield (ccle_train_data, ccle_train_label), (ccle_test_data, ccle_test_label)


def TCGA_target_data(drug):
    tcga_features_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'tcgadata.csv'), index_col=0, header=0)
    # tcga_features_df = tcga_features_df.reindex(columns=ccle_columns)
    tcga_label_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'tcgalabel.csv'), index_col=0, header=0)
    tcga_features = torch.from_numpy(tcga_features_df.values).type(torch.float32).to(device)
    tcga_label = torch.from_numpy(tcga_label_df.values).type(torch.float32).squeeze().to(device)

    return (tcga_features, tcga_label)


def TCGA_data_generator(drug):
    tcga_data = TCGA_target_data(drug)
    ccle_data_tuple = TCGA_source_5fold(drug)
    for ccle_train_data, ccle_eval_data in ccle_data_tuple:
        yield (ccle_train_data, ccle_eval_data, tcga_data)


def other_data_generator(datafolder):
    os.path.join(datafolder)
    source_features_df = pd.read_csv(os.path.join(datafolder, 'sourcedata.csv'), index_col=0, header=0)
    source_label_df = pd.read_csv(os.path.join(datafolder, 'sourcelabel.csv'), index_col=0, header=0)
    target_features_df = pd.read_csv(os.path.join(datafolder, 'targetdata.csv'), index_col=0, header=0)
    target_label_df = pd.read_csv(os.path.join(datafolder, 'targetlabel.csv'), index_col=0, header=0)
    s_kfold = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in s_kfold.split(source_features_df.values, source_label_df.values):
        train_labeled_source_df, test_labeled_source_df = source_features_df.values[train_index], \
                                                    source_features_df.values[test_index]
        train_source_labels, test_source_labels = source_label_df.values[train_index], source_label_df.values[test_index]
        # df->tensor
        source_train_data = torch.from_numpy(train_labeled_source_df).type(torch.float32).to(device)
        source_train_label = torch.from_numpy(train_source_labels).type(torch.float32).squeeze().to(device)
        source_test_data = torch.from_numpy(test_labeled_source_df).type(torch.float32).to(device)
        source_test_label = torch.from_numpy(test_source_labels).type(torch.float32).squeeze().to(device)
        target_data = torch.from_numpy(target_features_df.values).type(torch.float32).to(device)
        target_label = torch.from_numpy(target_label_df.values).type(torch.float32).squeeze().to(device)

        yield (source_train_data, source_train_label),(source_test_data, source_test_label),(target_data, target_label)

def load_data_with_labels_and_gene_mapping(ccle_path, xena_path, batch_size=128, device='cuda', normalization_method='zscore'):
    """
    Load CCLE and TCGA data with cancer type labels and gene mapping for pathway extraction
    
    Args:
        ccle_path (str): Path to CCLE data
        xena_path (str): Path to TCGA data
        batch_size (int): Batch size for DataLoader
        device (str): Device to load tensors on
        normalization_method (str): Normalization method ('zscore', 'none', 'log')
        
    Returns:
        tuple: ((source_loader, source_test_tensor, source_test_labels_tensor),
                (target_loader, target_test_tensor, target_test_labels_tensor),
                mapping_int2str, gene_to_idx)
    """
    min_count = 10 # 低頻病種：出現次數 <10 in ccle

    # Load CCLE data
    ccle_df = pd.read_csv(ccle_path, index_col=0)
    ccle_sample_info_df = pd.read_csv(os.path.join('data', 'ccle_sample_info_df.csv'), index_col=0, header=0)
    # Load TCGA data
    xena_df = pd.read_csv(xena_path, index_col=0)
    xena_sample_info_df = pd.read_csv(os.path.join('data', 'xena_sample_info_df.csv'), index_col=0, header=0)
    # 針對原始完整資料進行標準化
    print(f"Applying {normalization_method} normalization to complete datasets...")
    ccle_df = normalize_gene_expression(ccle_df, method=normalization_method)
    xena_df = normalize_gene_expression(xena_df, method=normalization_method)
    # Ensure all indices are strings
    ccle_df.index = ccle_df.index.astype(str)
    xena_df.index = xena_df.index.astype(str)
    ccle_sample_info_df.index = ccle_sample_info_df.index.astype(str)
    xena_sample_info_df.index = xena_sample_info_df.index.astype(str)
    
    # Map TCGA cancer types to CCLE cancer types
    target_to_source_map = {
        'acute myeloid leukemia': 'na', #Only solid tumors
        'adrenocortical cancer': 'na',
        'bladder urothelial carcinoma': 'Bladder Cancer',
        'brain lower grade glioma': 'Brain Cancer',
        'breast invasive carcinoma': 'Breast Cancer',
        'cervical & endocervical cancer': 'Cervical Cancer',
        'cholangiocarcinoma': 'Bile Duct Cancer',
        'colon adenocarcinoma': 'Colon/Colorectal Cancer',
        'diffuse large B-cell lymphoma': 'na', #Only solid tumors
        'esophageal carcinoma': 'Esophageal Cancer',
        'glioblastoma multiforme': 'Brain Cancer',
        'head & neck squamous cell carcinoma': 'Head and Neck Cancer',
        'kidney chromophobe': 'Kidney Cancer',
        'kidney clear cell carcinoma': 'Kidney Cancer',
        'kidney papillary cell carcinoma': 'Kidney Cancer',
        'liver hepatocellular carcinoma': 'Liver Cancer',
        'lung adenocarcinoma': 'Lung Cancer',
        'lung squamous cell carcinoma': 'Lung Cancer',
        'mesothelioma': 'na',
        'ovarian serous cystadenocarcinoma': 'Ovarian Cancer',
        'pancreatic adenocarcinoma': 'Pancreatic Cancer',
        'pheochromocytoma & paraganglioma': 'na',
        'prostate adenocarcinoma': 'Prostate Cancer',
        'rectum adenocarcinoma': 'Colon/Colorectal Cancer',
        'sarcoma': 'Sarcoma',
        'skin cutaneous melanoma': 'Skin Cancer',
        'stomach adenocarcinoma': 'Gastric Cancer',
        'testicular germ cell tumor': 'na',
        'thymoma': 'na',
        'thyroid carcinoma': 'Thyroid Cancer',
        'uterine carcinosarcoma': 'Endometrial/Uterine Cancer',
        'uterine corpus endometrioid carcinoma': 'Endometrial/Uterine Cancer',
        'uveal melanoma': 'Eye Cancer'
    }
    
    # Apply mapping to TCGA sample info
    xena_sample_info_df["_primary_disease"] = xena_sample_info_df["_primary_disease"].map(target_to_source_map)
    
    # ----------------------------
    # 來源資料 (CCLE) 處理
    # ----------------------------
    # 1. 過濾掉來源中 sample_info 不存在的樣本
    excluded_ccle_samples = list(set(ccle_df.index) - set(ccle_sample_info_df.index))
    
    # 2. 過濾掉低頻病種 (出現次數 < min_count)
    excluded_ccle_diseases = ccle_sample_info_df.primary_disease.value_counts()[
        ccle_sample_info_df.primary_disease.value_counts() < min_count].index
    excluded_ccle_samples.extend(
        list(ccle_sample_info_df[ccle_sample_info_df.primary_disease.isin(excluded_ccle_diseases)].index)
    )
    
    # 3. 僅保留 sample_info 中存在的資料
    to_split_ccle_df = ccle_df.loc[ccle_df.index.difference(excluded_ccle_samples)]
    
    # 4. 保留 primary_disease 非空
    valid_source_idx = to_split_ccle_df.index.intersection(
        ccle_sample_info_df[ccle_sample_info_df.primary_disease.notna()].index
    )
    filtered_ccle_df = to_split_ccle_df.loc[valid_source_idx]
    source_labels = ccle_sample_info_df.loc[valid_source_idx, "primary_disease"]
    
    # ----------------------------
    # 目標資料 (TCGA) 處理
    # ----------------------------
    # 確保 _primary_disease 非空
    valid_target_idx = xena_df.index.intersection(
        xena_sample_info_df[xena_sample_info_df["_primary_disease"].notna()].index
    )
    filtered_xena_df = xena_df.loc[valid_target_idx]
    target_labels = xena_sample_info_df.loc[valid_target_idx, "_primary_disease"]
    
    # ----------------------------
    # 僅保留交集病種
    # ----------------------------
    common_labels = set(source_labels.unique()) & set(target_labels.unique())
    common_labels = {label for label in common_labels if label != 'na' and not pd.isna(label)}
    
    ccle_mask = source_labels.isin(common_labels)
    xena_mask = target_labels.isin(common_labels)
    
    filtered_ccle_df = filtered_ccle_df[ccle_mask]
    source_labels = source_labels[ccle_mask]
    filtered_xena_df = filtered_xena_df[xena_mask]
    target_labels = target_labels[xena_mask]
    
    # Split data into train/test using stratified split to maintain label distribution
    train_ccle_df, test_ccle_df, source_train_labels, source_test_labels = train_test_split(
        filtered_ccle_df, source_labels, test_size=0.2, 
        stratify=source_labels, random_state=42
    )
    
    train_xena_df, test_xena_df, target_train_labels, target_test_labels = train_test_split(
        filtered_xena_df, target_labels, test_size=0.2, 
        stratify=target_labels, random_state=42
    )
    
    # Create mapping from disease names to integers
    common_labels = sorted(list(common_labels))
    mapping = {d: i for i, d in enumerate(common_labels)}
    mapping_int2str = {i: d for d, i in mapping.items()}
    
    # Convert labels to integers
    source_train_labels_int = np.array([mapping[label] for label in source_train_labels])
    source_test_labels_int = np.array([mapping[label] for label in source_test_labels])
    target_train_labels_int = np.array([mapping[label] for label in target_train_labels])
    target_test_labels_int = np.array([mapping[label] for label in target_test_labels])
    
    # Save gene names for pathway extraction
    gene_names = train_ccle_df.columns.tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    # Convert to tensors
    source_train_tensor = torch.from_numpy(train_ccle_df.values).type(torch.float32).to(device)
    source_test_tensor = torch.from_numpy(test_ccle_df.values).type(torch.float32).to(device)
    target_train_tensor = torch.from_numpy(train_xena_df.values).type(torch.float32).to(device)
    target_test_tensor = torch.from_numpy(test_xena_df.values).type(torch.float32).to(device)
    
    source_train_labels_tensor = torch.from_numpy(source_train_labels_int).to(device)
    source_test_labels_tensor = torch.from_numpy(source_test_labels_int).to(device)
    target_train_labels_tensor = torch.from_numpy(target_train_labels_int).to(device)
    target_test_labels_tensor = torch.from_numpy(target_test_labels_int).to(device)
    
    # Create DataLoaders
    sourcedataset = TensorDataset(source_train_tensor, source_train_labels_tensor)
    source_loader = DataLoader(sourcedataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    targetdataset = TensorDataset(target_train_tensor, target_train_labels_tensor)
    target_loader = DataLoader(targetdataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return ((source_loader, source_test_tensor, source_test_labels_tensor),
            (target_loader, target_test_tensor, target_test_labels_tensor),
            mapping_int2str, gene_to_idx)

def load_data_with_labels_and_weights_and_gene_mapping(ccle_path, xena_path, batch_size=128, device='cuda', normalization_method='zscore'):
    """
    Load CCLE and TCGA data with cancer type labels, class weights, and gene mapping for pathway extraction
    
    Args:
        ccle_path (str): Path to CCLE data
        xena_path (str): Path to TCGA data
        batch_size (int): Batch size for DataLoader
        device (str): Device to load tensors on
        normalization_method (str): Normalization method ('zscore', 'none', 'log')
        
    Returns:
        tuple: ((source_loader, source_test_tensor, source_test_labels_tensor, source_weights),
                (target_loader, target_test_tensor, target_test_labels_tensor, target_weights),
                mapping_int2str, gene_to_idx)
    """
    # First get data using the function with gene mapping
    (source_loader, source_test_tensor, source_test_labels_tensor), \
    (target_loader, target_test_tensor, target_test_labels_tensor), \
    mapping_int2str, gene_to_idx = load_data_with_labels_and_gene_mapping(ccle_path, xena_path, batch_size, device, normalization_method=normalization_method)
    
    # Extract training labels from the loaders
    source_train_labels = []
    for _, labels in source_loader:
        source_train_labels.append(labels)
    source_train_labels = torch.cat(source_train_labels)
    
    target_train_labels = []
    for _, labels in target_loader:
        target_train_labels.append(labels)
    target_train_labels = torch.cat(target_train_labels)
    
    # Compute class weights for source and target datasets
    source_weights = compute_class_weights(source_train_labels, device)
    target_weights = compute_class_weights(target_train_labels, device)
    
    return ((source_loader, source_test_tensor, source_test_labels_tensor, source_weights),
            (target_loader, target_test_tensor, target_test_labels_tensor, target_weights),
            mapping_int2str, gene_to_idx)

def balance_dataset(source_loader, target_loader, balance_strategy, balance_ratio):
    """
    Balance source and target datasets using specified strategy and ratio.
    
    Args:
        source_loader: DataLoader for source dataset
        target_loader: DataLoader for target dataset
        balance_strategy: 'oversample', 'undersample', or 'hybrid'
        balance_ratio: Float ratio for balancing (1.0 = perfect balance)
        
    Returns:
        tuple: (balanced_source_loader, balanced_target_loader)
    """
    # Get original dataset sizes
    source_size = len(source_loader.dataset)
    target_size = len(target_loader.dataset)
    
    print(f"Original dataset sizes - CCLE: {source_size}, TCGA: {target_size}")
    
    # Calculate target sizes based on strategy
    if balance_strategy == 'oversample':
        # Oversample smaller dataset to match larger dataset size
        target_balanced_size = max(source_size, target_size)
        source_balanced_size = target_balanced_size
    elif balance_strategy == 'undersample':
        # Undersample larger dataset to match smaller dataset size
        target_balanced_size = min(source_size, target_size)
        source_balanced_size = target_balanced_size
    elif balance_strategy == 'hybrid':
        # Hybrid strategy: oversample smaller, undersample larger
        avg_size = (source_size + target_size) // 2
        source_balanced_size = avg_size
        target_balanced_size = avg_size
    else:
        raise ValueError(f"Unknown balance strategy: {balance_strategy}")
    
    # Apply balance ratio
    source_balanced_size = int(source_balanced_size * balance_ratio)
    target_balanced_size = int(target_balanced_size * balance_ratio)
    
    print(f"Balanced dataset sizes - CCLE: {source_balanced_size}, TCGA: {target_balanced_size}")
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Balance source dataset
    if source_balanced_size > source_size:
        # Oversample: repeat sampling
        indices = list(range(source_size)) * (source_balanced_size // source_size + 1)
        source_indices = indices[:source_balanced_size]
    else:
        # Undersample: random selection
        source_indices = random.sample(range(source_size), source_balanced_size)
    
    # Balance target dataset
    if target_balanced_size > target_size:
        # Oversample: repeat sampling
        indices = list(range(target_size)) * (target_balanced_size // target_size + 1)
        target_indices = indices[:target_balanced_size]
    else:
        # Undersample: random selection
        target_indices = random.sample(range(target_size), target_balanced_size)
    
    # Create balanced data loaders
    source_subset = Subset(source_loader.dataset, source_indices)
    target_subset = Subset(target_loader.dataset, target_indices)
    
    balanced_source_loader = DataLoader(
        source_subset, 
        batch_size=source_loader.batch_size, 
        shuffle=True, 
        num_workers=source_loader.num_workers,
        pin_memory=source_loader.pin_memory
    )
    
    balanced_target_loader = DataLoader(
        target_subset, 
        batch_size=target_loader.batch_size, 
        shuffle=True, 
        num_workers=target_loader.num_workers,
        pin_memory=target_loader.pin_memory
    )
    
    print(f"Dataset balancing completed. New batch counts - CCLE: {len(balanced_source_loader)}, TCGA: {len(balanced_target_loader)}")
    
    return balanced_source_loader, balanced_target_loader

def load_pathway_data_with_labels(ccle_path, xena_path, pathway_key, batch_size=128, device='cuda', 
                                  normalization_method='zscore', balance_strategy=None, balance_ratio=1.0):
    """
    Load CCLE and TCGA data with cancer type labels for specific pathway
    
    Args:
        ccle_path (str): Path to CCLE data
        xena_path (str): Path to TCGA data
        pathway_key (str): Key to extract gene list from pathway file
        batch_size (int): Batch size for DataLoader
        device (str): Device to load tensors on
        normalization_method (str): Normalization method ('zscore', 'none', 'log')
        balance_strategy (str): 'oversample', 'undersample', 'hybrid', or None
        balance_ratio (float): Ratio for balancing (1.0 = perfect balance)
        
    Returns:
        tuple: ((source_loader, source_test_tensor, source_test_labels_tensor),
                (target_loader, target_test_tensor, target_test_labels_tensor),
                mapping_int2str)
    """
    import pickle
    
    min_count = 10 # 低頻病種：出現次數 <10 in ccle

    # Load pathway data and extract gene list
    pathway_file = os.path.join('data', '34pathway_score990.pkl')
    with open(pathway_file, 'rb') as f:
        pathway_data = pickle.load(f)
    
    if pathway_key not in pathway_data:
        raise ValueError(f"Pathway key '{pathway_key}' not found in pathway file. Available keys: {list(pathway_data.keys())}")
    
    gene_list = pathway_data[pathway_key]
    print(f"Extracted {len(gene_list)} genes for pathway: {pathway_key}")

    # Load CCLE data
    ccle_df = pd.read_csv(ccle_path, index_col=0)
    ccle_sample_info_df = pd.read_csv(os.path.join('data', 'ccle_sample_info_df.csv'), index_col=0, header=0)
    # Load TCGA data
    xena_df = pd.read_csv(xena_path, index_col=0)
    xena_sample_info_df = pd.read_csv(os.path.join('data', 'xena_sample_info_df.csv'), index_col=0, header=0)
    
    # Extract only pathway genes
    available_genes = [gene for gene in gene_list if gene in ccle_df.columns and gene in xena_df.columns]
    if len(available_genes) == 0:
        raise ValueError(f"No genes from pathway '{pathway_key}' found in the datasets")
    
    print(f"Found {len(available_genes)} genes in datasets out of {len(gene_list)} pathway genes")
    
    # Extract pathway-specific data
    ccle_df = ccle_df[available_genes]
    xena_df = xena_df[available_genes]
    
    # 針對原始完整資料進行標準化
    print(f"Applying {normalization_method} normalization to pathway datasets...")
    ccle_df = normalize_gene_expression(ccle_df, method=normalization_method)
    xena_df = normalize_gene_expression(xena_df, method=normalization_method)
    
    # Ensure all indices are strings
    ccle_df.index = ccle_df.index.astype(str)
    xena_df.index = xena_df.index.astype(str)
    ccle_sample_info_df.index = ccle_sample_info_df.index.astype(str)
    xena_sample_info_df.index = xena_sample_info_df.index.astype(str)
    
    # Map TCGA cancer types to CCLE cancer types
    target_to_source_map = {
        'acute myeloid leukemia': 'na', #Only solid tumors
        'adrenocortical cancer': 'na',
        'bladder urothelial carcinoma': 'Bladder Cancer',
        'brain lower grade glioma': 'Brain Cancer',
        'breast invasive carcinoma': 'Breast Cancer',
        'cervical & endocervical cancer': 'Cervical Cancer',
        'cholangiocarcinoma': 'Bile Duct Cancer',
        'colon adenocarcinoma': 'Colon/Colorectal Cancer',
        'diffuse large B-cell lymphoma': 'na', #Only solid tumors
        'esophageal carcinoma': 'Esophageal Cancer',
        'glioblastoma multiforme': 'Brain Cancer',
        'head & neck squamous cell carcinoma': 'Head and Neck Cancer',
        'kidney chromophobe': 'Kidney Cancer',
        'kidney clear cell carcinoma': 'Kidney Cancer',
        'kidney papillary cell carcinoma': 'Kidney Cancer',
        'liver hepatocellular carcinoma': 'Liver Cancer',
        'lung adenocarcinoma': 'Lung Cancer',
        'lung squamous cell carcinoma': 'Lung Cancer',
        'mesothelioma': 'na',
        'ovarian serous cystadenocarcinoma': 'Ovarian Cancer',
        'pancreatic adenocarcinoma': 'Pancreatic Cancer',
        'pheochromocytoma & paraganglioma': 'na',
        'prostate adenocarcinoma': 'Prostate Cancer',
        'rectum adenocarcinoma': 'Colon/Colorectal Cancer',
        'sarcoma': 'Sarcoma',
        'skin cutaneous melanoma': 'Skin Cancer',
        'stomach adenocarcinoma': 'Gastric Cancer',
        'testicular germ cell tumor': 'na',
        'thymoma': 'na',
        'thyroid carcinoma': 'Thyroid Cancer',
        'uterine carcinosarcoma': 'Endometrial/Uterine Cancer',
        'uterine corpus endometrioid carcinoma': 'Endometrial/Uterine Cancer',
        'uveal melanoma': 'Eye Cancer'
    }
    
    # Apply mapping to TCGA sample info
    xena_sample_info_df["_primary_disease"] = xena_sample_info_df["_primary_disease"].map(target_to_source_map)
    
    # ----------------------------
    # 來源資料 (CCLE) 處理
    # ----------------------------
    # 1. 過濾掉來源中 sample_info 不存在的樣本
    excluded_ccle_samples = list(set(ccle_df.index) - set(ccle_sample_info_df.index))
    
    # 2. 過濾掉低頻病種 (出現次數 < min_count)
    excluded_ccle_diseases = ccle_sample_info_df.primary_disease.value_counts()[
        ccle_sample_info_df.primary_disease.value_counts() < min_count].index
    excluded_ccle_samples.extend(
        list(ccle_sample_info_df[ccle_sample_info_df.primary_disease.isin(excluded_ccle_diseases)].index)
    )
    
    # 3. 僅保留 sample_info 中存在的資料
    to_split_ccle_df = ccle_df.loc[ccle_df.index.difference(excluded_ccle_samples)]
    
    # 4. 保留 primary_disease 非空
    valid_source_idx = to_split_ccle_df.index.intersection(
        ccle_sample_info_df[ccle_sample_info_df.primary_disease.notna()].index
    )
    filtered_ccle_df = to_split_ccle_df.loc[valid_source_idx]
    source_labels = ccle_sample_info_df.loc[valid_source_idx, "primary_disease"]
    
    # ----------------------------
    # 目標資料 (TCGA) 處理
    # ----------------------------
    # 確保 _primary_disease 非空
    valid_target_idx = xena_df.index.intersection(
        xena_sample_info_df[xena_sample_info_df["_primary_disease"].notna()].index
    )
    filtered_xena_df = xena_df.loc[valid_target_idx]
    target_labels = xena_sample_info_df.loc[valid_target_idx, "_primary_disease"]
    
    # ----------------------------
    # 僅保留交集病種
    # ----------------------------
    common_labels = set(source_labels.unique()) & set(target_labels.unique())
    common_labels = {label for label in common_labels if label != 'na' and not pd.isna(label)}
    
    ccle_mask = source_labels.isin(common_labels)
    xena_mask = target_labels.isin(common_labels)
    
    filtered_ccle_df = filtered_ccle_df[ccle_mask]
    source_labels = source_labels[ccle_mask]
    filtered_xena_df = filtered_xena_df[xena_mask]
    target_labels = target_labels[xena_mask]
    
    # Split data into train/test using stratified split to maintain label distribution
    train_ccle_df, test_ccle_df, source_train_labels, source_test_labels = train_test_split(
        filtered_ccle_df, source_labels, test_size=0.2, 
        stratify=source_labels, random_state=42
    )
    
    train_xena_df, test_xena_df, target_train_labels, target_test_labels = train_test_split(
        filtered_xena_df, target_labels, test_size=0.2, 
        stratify=target_labels, random_state=42
    )
    
    # Create mapping from disease names to integers
    common_labels = sorted(list(common_labels))
    mapping = {d: i for i, d in enumerate(common_labels)}
    mapping_int2str = {i: d for d, i in mapping.items()}
    
    # Convert labels to integers
    source_train_labels_int = np.array([mapping[label] for label in source_train_labels])
    source_test_labels_int = np.array([mapping[label] for label in source_test_labels])
    target_train_labels_int = np.array([mapping[label] for label in target_train_labels])
    target_test_labels_int = np.array([mapping[label] for label in target_test_labels])
    
    # Convert to tensors
    source_train_tensor = torch.from_numpy(train_ccle_df.values).type(torch.float32).to(device)
    source_test_tensor = torch.from_numpy(test_ccle_df.values).type(torch.float32).to(device)
    target_train_tensor = torch.from_numpy(train_xena_df.values).type(torch.float32).to(device)
    target_test_tensor = torch.from_numpy(test_xena_df.values).type(torch.float32).to(device)
    
    source_train_labels_tensor = torch.from_numpy(source_train_labels_int).to(device)
    source_test_labels_tensor = torch.from_numpy(source_test_labels_int).to(device)
    target_train_labels_tensor = torch.from_numpy(target_train_labels_int).to(device)
    target_test_labels_tensor = torch.from_numpy(target_test_labels_int).to(device)
    
    # Create DataLoaders
    sourcedataset = TensorDataset(source_train_tensor, source_train_labels_tensor)
    source_loader = DataLoader(sourcedataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    targetdataset = TensorDataset(target_train_tensor, target_train_labels_tensor)
    target_loader = DataLoader(targetdataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Apply dataset balancing if specified
    if balance_strategy is not None:
        print(f"Applying {balance_strategy} balancing with ratio {balance_ratio}")
        source_loader, target_loader = balance_dataset(source_loader, target_loader, balance_strategy, balance_ratio)
    
    return ((source_loader, source_test_tensor, source_test_labels_tensor),
            (target_loader, target_test_tensor, target_test_labels_tensor),
            mapping_int2str)

def load_pathway_data_with_labels_and_weights(ccle_path, xena_path, pathway_key, batch_size=128, device='cuda', 
                                             normalization_method='zscore', balance_strategy=None, balance_ratio=1.0):
    """
    Load CCLE and TCGA data with cancer type labels and class weights for specific pathway
    
    Args:
        ccle_path (str): Path to CCLE data
        xena_path (str): Path to TCGA data
        pathway_key (str): Key to extract gene list from pathway file
        batch_size (int): Batch size for DataLoader
        device (str): Device to load tensors on
        normalization_method (str): Normalization method ('zscore', 'none', 'log')
        balance_strategy (str): 'oversample', 'undersample', 'hybrid', or None
        balance_ratio (float): Ratio for balancing (1.0 = perfect balance)
        
    Returns:
        tuple: ((source_loader, source_test_tensor, source_test_labels_tensor, source_weights),
                (target_loader, target_test_tensor, target_test_labels_tensor, target_weights),
                mapping_int2str)
    """
    # First get data using the function without weights
    (source_loader, source_test_tensor, source_test_labels_tensor), \
    (target_loader, target_test_tensor, target_test_labels_tensor), \
    mapping_int2str = load_pathway_data_with_labels(ccle_path, xena_path, pathway_key, batch_size, device, 
                                                   normalization_method, balance_strategy, balance_ratio)
    
    # Extract training labels from the loaders
    source_train_labels = []
    for _, labels in source_loader:
        source_train_labels.append(labels)
    source_train_labels = torch.cat(source_train_labels)
    
    target_train_labels = []
    for _, labels in target_loader:
        target_train_labels.append(labels)
    target_train_labels = torch.cat(target_train_labels)
    
    # Compute class weights for source and target datasets
    source_weights = compute_class_weights(source_train_labels, device)
    target_weights = compute_class_weights(target_train_labels, device)
    
    return ((source_loader, source_test_tensor, source_test_labels_tensor, source_weights),
            (target_loader, target_test_tensor, target_test_labels_tensor, target_weights),
            mapping_int2str)