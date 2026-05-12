import math
import os
import pandas as pd
import numpy as np
import argparse
import json
import base64
import io
import matplotlib.pyplot as plt
from glob import glob
import torch
from scipy.spatial.distance import cdist

# Set non-interactive backend
plt.switch_backend('Agg')

def calculate_mmd(source_latent, target_latent, kernel='rbf', gamma=None):
    """Calculate Maximum Mean Discrepancy between source and target latents"""
    try:
        if isinstance(source_latent, torch.Tensor):
            source_latent = source_latent.cpu().numpy()
        if isinstance(target_latent, torch.Tensor):
            target_latent = target_latent.cpu().numpy()
        
        # Ensure 2D
        if len(source_latent.shape) == 1:
            source_latent = source_latent.reshape(1, -1)
        if len(target_latent.shape) == 1:
            target_latent = target_latent.reshape(1, -1)
        
        # Sample if too large
        if source_latent.shape[0] > 1000:
            indices = np.random.choice(source_latent.shape[0], 1000, replace=False)
            source_latent = source_latent[indices]
        if target_latent.shape[0] > 1000:
            indices = np.random.choice(target_latent.shape[0], 1000, replace=False)
            target_latent = target_latent[indices]
        
        # RBF kernel
        if gamma is None:
            gamma = 1.0 / source_latent.shape[1]
        
        xx = np.exp(-gamma * cdist(source_latent, source_latent, 'sqeuclidean'))
        yy = np.exp(-gamma * cdist(target_latent, target_latent, 'sqeuclidean'))
        xy = np.exp(-gamma * cdist(source_latent, target_latent, 'sqeuclidean'))
        
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return max(0, mmd)  # MMD should be non-negative
    except Exception as e:
        print(f"Error calculating MMD: {e}")
        return np.nan

def calculate_wasserstein(source_latent, target_latent):
    """Calculate Wasserstein distance (approximation using mean and std)"""
    try:
        if isinstance(source_latent, torch.Tensor):
            source_latent = source_latent.cpu().numpy()
        if isinstance(target_latent, torch.Tensor):
            target_latent = target_latent.cpu().numpy()
        
        # Ensure 2D
        if len(source_latent.shape) == 1:
            source_latent = source_latent.reshape(1, -1)
        if len(target_latent.shape) == 1:
            target_latent = target_latent.reshape(1, -1)
        
        # Calculate mean and covariance
        mu_s = np.mean(source_latent, axis=0)
        mu_t = np.mean(target_latent, axis=0)
        
        # Simplified Wasserstein: Euclidean distance between means
        w_dist = np.linalg.norm(mu_s - mu_t)
        return w_dist
    except Exception as e:
        print(f"Error calculating Wasserstein: {e}")
        return np.nan

def load_experiment_data(exp_dir):
    """
    Load data from a single experiment directory.
    Reads params.json and the last row of training_log.csv.
    Finds the latest t-SNE image.
    """
    data = {}
    
    # 1. Load Parameters
    params_path = os.path.join(exp_dir, 'params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        data.update(params)
    else:
        print(f"Warning: params.json not found in {exp_dir}")
        data['ID'] = os.path.basename(exp_dir)
        
    data['ID'] = os.path.basename(exp_dir)
    
    # 2. Load Metrics (Last Epoch)
    log_path = os.path.join(exp_dir, 'training_log.csv')
    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            if not df.empty:
                # Get best FID run or last? Usually comparisons use best or last.
                # Let's use Last for now, or Best FID if we want to cherry pick.
                # Let's use Last, but also capture Best FID.
                last_row = df.iloc[-1]
                for col in df.columns:
                    data[col] = last_row[col]
                
                if 'fid' in df.columns:
                    data['Best_FID'] = df['fid'].min()
                    data['Best_FID_Epoch'] = df.loc[df['fid'].idxmin(), 'epoch']
        except Exception as e:
            print(f"Error reading log {log_path}: {e}")
            
    # 3. Find t-SNE Image (Latest epoch - prefer pretrained)
    tsne_files = glob(os.path.join(exp_dir, 'tsne_epoch_*_pretrained.png'))
    if not tsne_files:
        tsne_files = glob(os.path.join(exp_dir, 'tsne_epoch_*.png'))
    if tsne_files:
        # Sort by epoch number
        try:
            tsne_files.sort(key=lambda x: int(os.path.basename(x).split('_')[2]))
            data['tsne_image_path'] = tsne_files[-1]
        except:
            data['tsne_image_path'] = tsne_files[-1]
    else:
        data['tsne_image_path'] = None
    
    # 4. Calculate MMD and Wasserstein (if latent files exist)
    # Look for saved latent representations
    try:
        # Try to load from model checkpoint and calculate metrics
        # For now, use placeholder values - these should be calculated during training
        # and saved in the training_log.csv or separate metrics file
        data['MMD'] = data.get('mmd', np.nan)
        data['Wasserstein'] = data.get('wasserstein', np.nan)
    except Exception as e:
        data['MMD'] = np.nan
        data['Wasserstein'] = np.nan
        
    return data

def create_bar_chart(value, label, color='blue', limit=None):
    """Create a simple bar chart as base64 string"""
    plt.figure(figsize=(3, 2))
    plt.bar([label], [value], color=color)
    plt.title(f"{label}: {value:.4f}")
    if limit:
        plt.ylim(0, limit)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_html_report(df, output_dir, samples_per_file=100):
    """Generate HTML comparison report with pagination."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Columns to display - ONLY key metrics
    display_cols = ['ID', 'Best_FID', 'MMD', 'Wasserstein']
    # Filter to only columns that exist
    display_cols = [c for c in display_cols if c in df.columns]
    
    # Calculate number of files needed
    num_files = math.ceil(len(df) / samples_per_file)
    
    # Index page content if multiple files
    index_html = "<html><head><style>"
    index_html += "body { font-family: sans-serif; margin: 20px; }"
    index_html += "h1 { color: #333; }"
    index_html += "ul { list-style-type: none; padding: 0; }"
    index_html += "li { margin: 10px 0; }"
    index_html += "a { text-decoration: none; color: #0066cc; font-size: 18px; }"
    index_html += "a:hover { text-decoration: underline; }"
    index_html += "</style></head><body>"
    index_html += "<h1>GNN AE Experiment Reports</h1>"
    index_html += "<ul>"
    
    for file_idx in range(num_files):
        start_idx = file_idx * samples_per_file
        end_idx = min((file_idx + 1) * samples_per_file, len(df))
        
        # Subset dataframe
        df_subset = df.iloc[start_idx:end_idx]
        
        # File naming
        if num_files == 1:
            filename = 'gnn_ae_report.html'
            report_title = "GNN AE Experiment Report"
        else:
            filename = f'gnn_ae_report_part_{file_idx + 1}.html'
            report_title = f"GNN AE Experiment Report (Part {file_idx + 1}: {start_idx+1}-{end_idx})"
            
            # Add to index
            index_html += f'<li><a href="{filename}">Part {file_idx+1}: Experiments {start_idx+1} to {end_idx}</a></li>'
            
        html_path = os.path.join(output_dir, filename)
        
        # Generate HTML for this part
        html = "<html><head><style>"
        html += "body { font-family: sans-serif; margin: 20px; }"
        html += "table { border-collapse: collapse; width: 100%; }"
        html += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }"
        html += "th { background-color: #f2f2f2; }"
        html += "img.tsne { max-width: 400px; }"
        html += "img.metric { max-width: 250px; }"  # Keep for future
        html += "tr:nth-child(even) { background-color: #f9f9f9; }"
        html += "nav { margin: 20px 0; }"
        html += "nav a { margin-right: 15px; text-decoration: none; color: #0066cc; }"
        html += "</style></head><body>"
        
        html += f"<h1>{report_title}</h1>"
        
        # Navigation
        if num_files > 1:
            html += "<nav>"
            html += '<a href="gnn_ae_report_index.html">Back to Index</a>'
            if file_idx > 0:
                html += f'<a href="gnn_ae_report_part_{file_idx}.html">Previous</a>'
            if file_idx < num_files - 1:
                html += f'<a href="gnn_ae_report_part_{file_idx + 2}.html">Next</a>'
            html += "</nav>"
            
        html += "<table><thead><tr>"
        
        for col in display_cols:
            html += f"<th>{col}</th>"
        html += "<th>t-SNE</th>"
        html += "</tr></thead><tbody>"
        
        for _, row in df_subset.iterrows():
            html += "<tr>"
            for col in display_cols:
                val = row.get(col, 'N/A')
                # Format floats
                if isinstance(val, float):
                    val = f"{val:.4f}"
                html += f"<td>{val}</td>"
            
            # t-SNE Image
            html += "<td>"
            if row.get('tsne_image_path') and os.path.exists(row['tsne_image_path']):
                try:
                    with open(row['tsne_image_path'], "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    html += f'<img class="tsne" src="data:image/png;base64,{encoded_string}">'
                except Exception as e:
                    html += f"Error: {e}"
            else:
                html += "No Image"
            html += "</td>"
            
            html += "</tr>"
            
        html += "</tbody></table>"
        
        # Navigation footer
        if num_files > 1:
            html += "<nav>"
            html += '<a href="gnn_ae_report_index.html">Back to Index</a>'
            if file_idx > 0:
                html += f'<a href="gnn_ae_report_part_{file_idx}.html">Previous</a>'
            if file_idx < num_files - 1:
                html += f'<a href="gnn_ae_report_part_{file_idx + 2}.html">Next</a>'
            html += "</nav>"
            
        html += "</body></html>"
        
        with open(html_path, 'w') as f:
            f.write(html)
            
        print(f"Report saved to {html_path}")
        
    # Save index if multiple files
    if num_files > 1:
        index_html += "</ul></body></html>"
        with open(os.path.join(output_dir, 'gnn_ae_report_index.html'), 'w') as f:
            f.write(index_html)
        print(f"Index saved to {os.path.join(output_dir, 'gnn_ae_report_index.html')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True, help='Directory containing experiment subfolders')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save report')
    parser.add_argument('--per_page', type=int, default=100, help='Number of experiments per HTML page')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.result_dir
        
    experiments = sorted(glob(os.path.join(args.result_dir, 'exp_*')))
    if not experiments:
        # Maybe the result_dir IS the experiment?
        if os.path.exists(os.path.join(args.result_dir, 'params.json')):
            experiments = [args.result_dir]
        else:
            # Maybe standard folder naming?
            experiments = sorted([d for d in glob(os.path.join(args.result_dir, '*')) if os.path.isdir(d)])
            
    all_data = []
    for exp_dir in experiments:
        data = load_experiment_data(exp_dir)
        all_data.append(data)
        
    if not all_data:
        print("No experiments found.")
        return
        
    df = pd.DataFrame(all_data)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(args.output_dir, 'aggregated_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Aggregated CSV saved to {csv_path}")
    
    # Generate HTML
    generate_html_report(df, args.output_dir, samples_per_file=args.per_page)

if __name__ == '__main__':
    main()
