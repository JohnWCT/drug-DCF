import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import SaltRemover
from tools.dataprocess import smile_to_graph
from torch_geometric import data as DATA
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_class_weight

def desalt_smile(smile):
    """
    Remove salts from a SMILES string using RDKit SaltRemover.
    """
    try:
        remover = SaltRemover.SaltRemover()
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return smile
        res = remover.StripMol(mol)
        if res is None:
            return smile
        return Chem.MolToSmiles(res)
    except Exception as e:
        print(f"Error desalting smile {smile}: {e}")
        return smile

def process_drug_data(drug_df, smile_col='SMILES'):
    """
    Process drug dataframe: desalt SMILES and convert to graph objects.
    Returns a dictionary mapping drug_id to (SMILES, graph_data).
    """
    drug_dict = {}
    print("Processing drug data (Desalting and Graph Conversion)...")
    
    for idx, row in tqdm(drug_df.iterrows(), total=len(drug_df)):
        drug_id = idx
        raw_smile = row[smile_col]
        
        # 1. Desalt
        desalted_smile = desalt_smile(raw_smile)
        
        # 2. Convert to Graph
        try:
            c_size, features, edge_index = smile_to_graph(desalted_smile)
            
            # Create PyG Data object
            x = torch.tensor(np.array(features), dtype=torch.float32)
            if len(edge_index) > 0:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            drug_data = DATA.Data(x=x, edge_index=edge_index)
            drug_dict[drug_id] = {
                'smiles': desalted_smile,
                'graph': drug_data
            }
        except Exception as e:
            print(f"Failed to process drug {drug_id}: {e}")
            
    return drug_dict

def load_tcga_data_deduplicated(expression_path, response_path=None):
    """
    Load TCGA expression and (optional) response data.
    Deduplicate samples based on Patient ID (first 12 chars of barcode).
    Strategy: Average expression for duplicate samples.
    """
    print(f"Loading TCGA data from {expression_path}...")
    expr_df = pd.read_csv(expression_path, index_col=0)
    
    # Extract Patient ID (TCGA-XX-XXXX)
    # Assumes index is the barcode
    expr_df.index = expr_df.index.astype(str)
    
    # Create a mapping from barcode to patient ID
    # Filter for valid TCGA barcodes (at least 12 chars)
    valid_mask = expr_df.index.str.len() >= 12
    expr_df = expr_df[valid_mask]
    
    expr_df['PatientID'] = expr_df.index.str[:12]
    
    # Deduplicate by averaging expression per patient
    print("Deduplicating TCGA samples by Patient ID (averaging)...")
    expr_df_dedup = expr_df.groupby('PatientID').mean()
    
    response_df_dedup = None
    if response_path and os.path.exists(response_path):
        print(f"Loading response data from {response_path}...")
        resp_df = pd.read_csv(response_path, index_col=0)
        resp_df.index = resp_df.index.astype(str)
        
        # Match response to patients
        # Assuming response file also uses barcodes or patient IDs
        # If barcodes, we need to map them too.
        # But usually response data might already be patient-level or have barcodes.
        
        # For safety, let's try to map info if possible.
        # If resp index overlaps with patient IDs, good.
        # If resp index looks like barcodes, truncate.
        
        # Check first index
        if not resp_df.empty:
            first_idx = str(resp_df.index[0])
            if len(first_idx) >= 12 and first_idx.startswith('TCGA'):
                 resp_df['PatientID'] = resp_df.index.str[:12]
                 # For response (clinical outcome), if duplicates exist, we need a strategy.
                 # Categorical? Continuous?
                 # If it's drug response (AUC), average is okay.
                 # If it's class label (Responder/Non-Responder), we might need majority vote or take first.
                 # Let's assume numeric for now or provided labels.
                 
                 # If we are grouping, we need to handle non-numeric.
                 # Assuming numeric response for now or 'Class' column.
                 
                 # Let's inspect columns. If specific columns are needed, we handle them.
                 # For now, just group by PatientID and take mean (if numeric) or first (if object).
                 resp_df_dedup = resp_df.groupby('PatientID').first() # Safe default for labels
            else:
                 resp_df_dedup = resp_df # Assume already patient IDs or other ID
        else:
            resp_df_dedup = resp_df
            
        response_df_dedup = resp_df_dedup
        
    return expr_df_dedup, response_df_dedup

class DrugResponseDataset(Dataset):
    def __init__(self, df, expression_dict, drug_dict=None, drug_smiles_df=None, gin_type='precomputed', edge_index=None):
        """
        Args:
            df: DataFrame containing response data.
            expression_dict: Dictionary or DataFrame. 
                             If gin_type='precomputed', this is expression_latent_dict {sample_id: latent_vector}.
                             If gin_type!='precomputed', this is expr_df [Samples, Genes] (or passed as dict).
            drug_dict: Dictionary.
                       If gin_type='precomputed', this is drug_latent_dict {drug_name: latent_vector}.
                       If gin_type!='precomputed', this is drug_graph_dict {drug_id: graph_data}.
            drug_smiles_df: DataFrame containing SMILES (only if gin_type!='precomputed' and drug_dict not full).
            gin_type: 'precomputed' (latent mode) or 'dapl'/'ginpre' (graph mode).
            edge_index: PPI edge index (only for graph mode).
        """
        self.df = df.reset_index(drop=True)
        self.expression_dict = expression_dict
        self.gin_type = gin_type
        self.edge_index = edge_index
        
        # Standardize column names
        self.col_map = {}
        cols = self.df.columns
        
        # Sample ID
        if 'ModelID' in cols: self.col_map['sample'] = 'ModelID'
        elif 'DepMap_ID' in cols: self.col_map['sample'] = 'DepMap_ID'
        else: self.col_map['sample'] = cols[0] # Fallback
        
        # Drug ID
        if 'drug_name' in cols: self.col_map['drug'] = 'drug_name'
        elif 'DRUG_NAME' in cols: self.col_map['drug'] = 'DRUG_NAME'
        else: self.col_map['drug'] = cols[1] # Fallback
        
        # Label
        if 'Label' in cols: self.col_map['label'] = 'Label'
        elif 'Class' in cols: self.col_map['label'] = 'Class'
        elif 'binary_response' in cols: self.col_map['label'] = 'binary_response'
        elif 'response' in cols: self.col_map['label'] = 'response' # TCGA
        else: self.col_map['label'] = cols[2] # Fallback (original was cols[2], new instruction implies cols[3] but that would be incorrect if 'response' is still checked)
        
        # Calculate Sample Weights (Jointly for Drug & Label)
        # Create a combined category for stratification
        self.df['weight_group'] = self.df[self.col_map['drug']].astype(str) + '_' + self.df[self.col_map['label']].astype(str)
        unique_groups = np.unique(self.df['weight_group'])
        
        # Compute weights for each group
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_groups, y=self.df['weight_group'])
        weight_dict = dict(zip(unique_groups, class_weights))
        
        # Map back to dataframe
        self.df['sample_weight'] = self.df['weight_group'].map(weight_dict).astype('float32')
        print("Sample weights calculated based on (Drug, Label) balancing.")

        if gin_type == 'precomputed':
            # Use pre-computed latent representations
            self.drug_latent_dict = drug_dict
            self.expression_latent_dict = expression_dict
        else:
            # Use original SMILES-based approach with trainable GIN
            self.drug_smiles_df = drug_smiles_df
            self.drug_graph_dict = drug_dict if drug_dict is not None else {}
            
            # If drug_graph_dict is empty/incomplete, populate it from drug_smiles_df
            if drug_smiles_df is not None:
                # Only create graphs for drugs that are actually used in the response data
                unique_drugs = set(self.df[self.col_map['drug']].unique())
                
                # Check if we need to compute anything
                drugs_to_compute = unique_drugs - set(self.drug_graph_dict.keys())
                
                if drugs_to_compute:
                    print(f"Computing graphs for {len(drugs_to_compute)} drugs...")
                    for drug_id in drugs_to_compute:
                        # Find drug in smiles df - try exact match first, then lowercase
                        drug_smile = None
                        if drug_id in self.drug_smiles_df.index:
                            drug_smile = self.drug_smiles_df.loc[drug_id]['SMILES']
                        else:
                            # Case-insensitive fallback: SMILES index is lowercase
                            drug_id_lower = str(drug_id).lower()
                            if drug_id_lower in self.drug_smiles_df.index:
                                drug_smile = self.drug_smiles_df.loc[drug_id_lower]['SMILES']
                            elif 'DRUG_NAME' in self.drug_smiles_df.columns:
                                match = self.drug_smiles_df[self.drug_smiles_df['DRUG_NAME'].str.lower() == drug_id_lower]
                                if not match.empty:
                                    drug_smile = match.iloc[0]['SMILES']
                        
                        if drug_smile is None or (isinstance(drug_smile, float) and np.isnan(drug_smile)):
                            continue # Skip if SMILES not found or empty
                            
                        c_size, atom_features_list, edge_index_graph = smile_to_graph(drug_smile)
                        drug_x = torch.tensor(np.array(atom_features_list), dtype=torch.float32)
                        if len(edge_index_graph) > 0:
                            drug_edge_index = torch.tensor(edge_index_graph, dtype=torch.long).t().contiguous()
                        else:
                            drug_edge_index = torch.empty((2, 0), dtype=torch.long)
                        drug_data = DATA.Data(x=drug_x, edge_index=drug_edge_index)
                        self.drug_graph_dict[drug_id] = drug_data

            
            # For expression, if passing DataFrame, convert to tensor lookup
            if isinstance(expression_dict, pd.DataFrame):
                 self.expr_tensor = torch.tensor(expression_dict.values, dtype=torch.float32)
                 self.sample_to_idx = {sid: i for i, sid in enumerate(expression_dict.index)}
                 self.expression_mode = 'dataframe'
            else:
                 self.expression_mode = 'dict'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row[self.col_map['sample']]
        drug_id = row[self.col_map['drug']]
        target = float(row[self.col_map['label']])
        
        # Get expression representation
        if self.gin_type == 'precomputed':
            if sample_id in self.expression_latent_dict:
                gene_feature = torch.tensor(self.expression_latent_dict[sample_id], dtype=torch.float32)
            else:
                # Try finding without suffix if TCGA
                raise ValueError(f"Sample {sample_id} not found in expression latent dict")
        else:
            # Graph mode expression
            if self.expression_mode == 'dataframe':
                 if sample_id in self.sample_to_idx:
                     expr_idx = self.sample_to_idx[sample_id]
                     x = self.expr_tensor[expr_idx].view(-1, 1)
                     gene_feature = DATA.Data(x=x, edge_index=self.edge_index)
                 else:
                     raise ValueError(f"Sample {sample_id} not found in expression dataframe")
            else:
                  # Dict of latent arrays
                  if sample_id in self.expression_dict:
                      val = self.expression_dict[sample_id]
                      if isinstance(val, torch.Tensor):
                          gene_feature = val
                      else:
                          gene_feature = torch.tensor(np.array(val), dtype=torch.float32)
                  else:
                      raise ValueError(f"Sample {sample_id} not found in expression dict")

        # Get drug representation
        if self.gin_type == 'precomputed':
            drug_id_lower = str(drug_id).lower()
            if drug_id_lower in self.drug_latent_dict:
                drug_feature = torch.tensor(self.drug_latent_dict[drug_id_lower], dtype=torch.float32)
            else:
                raise ValueError(f"Drug {drug_id} not found in drug latent dict")
            return gene_feature, drug_feature, torch.tensor(target, dtype=torch.float32), torch.tensor(row['sample_weight'], dtype=torch.float32)
        else:
            if drug_id in self.drug_graph_dict:
                drug_data = self.drug_graph_dict[drug_id]
                return gene_feature, drug_data, torch.tensor(target, dtype=torch.float32), torch.tensor(row['sample_weight'], dtype=torch.float32)
            else:
                # Try case-insensitive lookup (response data may use mixed case while graphs use lowercase)
                drug_id_lower = str(drug_id).lower()
                if drug_id_lower in self.drug_graph_dict:
                    drug_data = self.drug_graph_dict[drug_id_lower]
                    return gene_feature, drug_data, torch.tensor(target, dtype=torch.float32), torch.tensor(row['sample_weight'], dtype=torch.float32)
                else:
                    raise KeyError(f"Drug '{drug_id}' not found in drug graph data.")


