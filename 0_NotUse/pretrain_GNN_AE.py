
import os
import argparse
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Import custom modules
from tools.data_utils import load_data_with_labels_and_gene_mapping
from tools.graph_utils import PPIEdgeProcessor, GraphDataset
from drugmodels.gnn_ae_model import GNNAutoencoder, Discriminator, Classifier
from tools.dataprocess import safemakedirs
from tools.metrics import calculate_frechet_distance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import cm

try:
    import seaborn as sns
except ImportError:
    sns = None

plt.switch_backend('Agg')

import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def reconstruction_loss(recon_x, x):
    return nn.MSELoss()(recon_x, x)

def ortho_loss(s_z, p_z):
    s_n = F.normalize(s_z, dim=1)
    p_n = F.normalize(p_z, dim=1)
    return torch.mean((s_n * p_n).sum(dim=1) ** 2)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN-GP"""
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.shape[0], 1).to(real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_mmd_metric(source_latent, target_latent, gamma=None):
    """Calculate Maximum Mean Discrepancy"""
    try:
        from scipy.spatial.distance import cdist
        if isinstance(source_latent, torch.Tensor):
            source_latent = source_latent.cpu().numpy()
        if isinstance(target_latent, torch.Tensor):
            target_latent = target_latent.cpu().numpy()
        if source_latent.shape[0] > 1000:
            idx = np.random.choice(source_latent.shape[0], 1000, replace=False)
            source_latent = source_latent[idx]
        if target_latent.shape[0] > 1000:
            idx = np.random.choice(target_latent.shape[0], 1000, replace=False)
            target_latent = target_latent[idx]
        if gamma is None:
            gamma = 1.0 / source_latent.shape[1]
        xx = np.exp(-gamma * cdist(source_latent, source_latent, 'sqeuclidean'))
        yy = np.exp(-gamma * cdist(target_latent, target_latent, 'sqeuclidean'))
        xy = np.exp(-gamma * cdist(source_latent, target_latent, 'sqeuclidean'))
        return max(0, xx.mean() + yy.mean() - 2 * xy.mean())
    except Exception as e:
        print(f"Error calculating MMD: {e}")
        return 0.0

def calculate_wasserstein_metric(source_latent, target_latent):
    """Calculate Wasserstein distance (mean approximation)"""
    try:
        if isinstance(source_latent, torch.Tensor):
            source_latent = source_latent.cpu().numpy()
        if isinstance(target_latent, torch.Tensor):
            target_latent = target_latent.cpu().numpy()
        return float(np.linalg.norm(np.mean(source_latent, axis=0) - np.mean(target_latent, axis=0)))
    except Exception as e:
        print(f"Error calculating Wasserstein: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_tsne_with_tumor_types(source_z, target_z, source_labels, target_labels,
                                     mapping_int2str, epoch, save_dir, is_pretrained=False):
    try:
        if isinstance(source_z, torch.Tensor): source_z = source_z.cpu().numpy()
        if isinstance(target_z, torch.Tensor): target_z = target_z.cpu().numpy()
        if isinstance(source_labels, torch.Tensor): source_labels = source_labels.cpu().numpy()
        if isinstance(target_labels, torch.Tensor): target_labels = target_labels.cpu().numpy()

        if len(source_z.shape) == 1: source_z = source_z.reshape(1, -1)
        if len(target_z.shape) == 1: target_z = target_z.reshape(1, -1)

        source_z = np.nan_to_num(source_z, nan=0.0, posinf=0.0, neginf=0.0)
        target_z = np.nan_to_num(target_z, nan=0.0, posinf=0.0, neginf=0.0)

        np.random.seed(42)
        source_z += np.random.normal(0, 1e-10, source_z.shape)
        target_z += np.random.normal(0, 1e-10, target_z.shape)

        combined_features = np.vstack([source_z, target_z])

        if combined_features.shape[0] > 3000:
            indices = np.random.choice(combined_features.shape[0], 3000, replace=False)
            combined_features = combined_features[indices]
            source_indices = indices[indices < len(source_z)]
            target_indices = indices[indices >= len(source_z)] - len(source_z)
            source_labels = source_labels[source_indices]
            target_labels = target_labels[target_indices]
            split_point = len(source_indices)
        else:
            split_point = len(source_z)

        tsne = TSNE(
            n_components=2, random_state=42,
            perplexity=min(30, len(combined_features) - 1),
            init='random', learning_rate='auto'
        )
        tsne_result = tsne.fit_transform(combined_features)
        tsne_source = tsne_result[:split_point]
        tsne_target = tsne_result[split_point:]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [3, 1]})

        all_unique_labels = np.unique(np.concatenate([source_labels, target_labels]))
        num_classes = len(all_unique_labels)
        if num_classes <= 10:
            cmap = cm.get_cmap('tab10', max(10, num_classes))
        elif num_classes <= 20:
            cmap = cm.get_cmap('tab20', max(20, num_classes))
        else:
            cmap = cm.get_cmap('hsv', num_classes)

        label_to_color = {label: cmap(i % cmap.N) for i, label in enumerate(all_unique_labels)}

        for cls in np.unique(target_labels):
            idx = np.where(target_labels == cls)[0]
            ax1.scatter(tsne_target[idx, 0], tsne_target[idx, 1],
                        color=label_to_color[cls], marker='^', edgecolors='k',
                        s=10, alpha=0.5, linewidths=0.5)

        for cls in np.unique(source_labels):
            idx = np.where(source_labels == cls)[0]
            ax1.scatter(tsne_source[idx, 0], tsne_source[idx, 1],
                        color=label_to_color[cls], marker='o', edgecolors='k',
                        s=15, alpha=0.9, linewidths=0.5)

        subtitle = "Pretrained (AE)" if is_pretrained else "After GAN"
        ax1.set_aspect('equal')
        x_min, x_max = ax1.get_xlim()
        y_min, y_max = ax1.get_ylim()
        ax1.set_xlim(x_min - (x_max - x_min) * 0.05, x_max + (x_max - x_min) * 0.05)
        ax1.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.05)
        ax1.set_title(f"t-SNE — Epoch {epoch} ({subtitle})")
        ax1.set_xlabel("Dimension 1")
        ax1.set_ylabel("Dimension 2")
        ax1.grid(alpha=0.3)
        ax1.set_box_aspect(1)

        ax2.axis('off')
        marker_handles = [
            mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Source (CCLE)'),
            mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=6, label='Target (TCGA)')
        ]
        color_handles = [
            mpatches.Patch(color=label_to_color[cls], label=mapping_int2str.get(cls, f"Class_{cls}"))
            for cls in all_unique_labels
        ]
        legend = ax2.legend(handles=marker_handles + color_handles,
                            loc='center', title="Legend", fontsize=7)
        ax2.add_artist(legend)

        plt.tight_layout()
        tag = "pretrained" if is_pretrained else "gan"
        fig_path = os.path.join(save_dir, f'tsne_epoch_{epoch}_{tag}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error in t-SNE visualization: {e}")


def plot_learning_curves(save_dir, log_file, title, out_prefix):
    """Generic learning curve plotter from a CSV log file."""
    try:
        log_path = os.path.join(save_dir, log_file)
        if not os.path.exists(log_path):
            return
        df = pd.read_csv(log_path)
        if df.empty:
            return

        loss_cols = [c for c in df.columns if c not in ('epoch', 'gan_epoch')]
        epoch_col = 'gan_epoch' if 'gan_epoch' in df.columns else 'epoch'

        fig, ax = plt.subplots(figsize=(10, 6))
        for col in loss_cols:
            ax.plot(df[epoch_col], df[col], label=col)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss / Score')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{out_prefix}_curve.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting {log_file}: {e}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_params_grid(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    pretrain_grid = config.get('pretrain_params', {})
    gan_grid      = config.get('gan_params', {})
    gnn_grid      = config.get('GNN_params', {})

    def expand_grid(grid):
        if not grid:
            return [{}]
        keys, values = zip(*grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    pretrain_list = expand_grid(pretrain_grid)
    gan_list      = expand_grid(gan_grid)
    gnn_list      = expand_grid(gnn_grid)

    # Each experiment = one pretrain config × one gan config × one gnn config
    experiments = list(itertools.product(pretrain_list, gan_list, gnn_list))
    return experiments  # list of (pretrain_param, gan_param, gnn_param)


# ---------------------------------------------------------------------------
# Shared helper: collect latents from a loader
# ---------------------------------------------------------------------------

def collect_latents(model, loader, domain, device):
    latents, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, s_z, _ = model(data.x, data.edge_index, data.batch, domain=domain)
            latents.append(s_z.cpu())
            labels.append(data.y.cpu())
    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)


def compute_fid(model, s_loader, t_loader, device):
    """Compute FID of source shared-z vs Gaussian prior."""
    z_s, labels_s = collect_latents(model, s_loader, 'source', device)
    z_t, labels_t = collect_latents(model, t_loader, 'target', device)
    z_np = z_s.numpy()
    mu, sigma = np.mean(z_np, axis=0), np.cov(z_np, rowvar=False)
    z_prior = np.random.randn(*z_np.shape)
    mu_p, sigma_p = np.mean(z_prior, axis=0), np.cov(z_prior, rowvar=False)
    fid = calculate_frechet_distance(mu, sigma, mu_p, sigma_p)
    return fid, z_s, z_t, labels_s, labels_t


# ---------------------------------------------------------------------------
# Phase 1 – Pretrain
# ---------------------------------------------------------------------------

def pretrain_phase(model, classifier, source_loader, target_loader,
                   s_test_loader, t_test_loader,
                   pretrain_param, device, save_dir, mapping_int2str):
    """
    Train shared_encoder + private_encoders + decoder + classifier.
    Loss: recon + lambda_ortho * ortho + lambda_cls * cls
    Early stopping on validation total loss.
    Returns: state_dict of best model + classifier, and final epoch.
    """
    lr        = pretrain_param.get('pretrain_learning_rate', 1e-4)
    beta1     = pretrain_param.get('pretrain_beta1', 0.9)
    beta2     = pretrain_param.get('pretrain_beta2', 0.999)
    wd        = pretrain_param.get('pretrain_weight_decay', 1e-4)
    epochs    = pretrain_param.get('pretrain_num_epochs', 200)
    lam_ortho = pretrain_param.get('lambda_ortho_pretrain', 0.0001)
    lam_cls   = pretrain_param.get('lambda_cls', 1.0)
    patience  = pretrain_param.get('pretrain_scheduler_patience', 15)
    factor    = pretrain_param.get('pretrain_scheduler_factor', 0.9)
    grad_clip = pretrain_param.get('pretrain_grad_clip', 1.0)

    early_stop_patience = patience * 3  # stop if no improvement
    cls_criterion = nn.CrossEntropyLoss()

    all_params = list(model.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(all_params, lr=lr, betas=(beta1, beta2), weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience, factor=factor, verbose=False)

    best_val_loss = float('inf')
    best_model_dict = copy.deepcopy(model.state_dict())
    best_cls_dict   = copy.deepcopy(classifier.state_dict())
    best_epoch = 0
    no_improve = 0

    logs = []
    steps = max(len(source_loader), len(target_loader))

    print(f"\n{'='*60}")
    print(f"  Phase 1 — Pretrain  ({epochs} epochs max, early stop pat={early_stop_patience})")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        classifier.train()

        tot_recon = tot_ortho = tot_cls = tot_loss = 0.0

        if len(source_loader) < len(target_loader):
            loader_zip = zip(itertools.cycle(source_loader), target_loader)
        else:
            loader_zip = zip(source_loader, itertools.cycle(target_loader))

        pbar = tqdm(loader_zip, total=steps, desc=f"PT Ep {epoch+1}/{epochs}", leave=False)
        for s_data, t_data in pbar:
            s_data = s_data.to(device)
            t_data = t_data.to(device)
            optimizer.zero_grad()

            s_recon, s_z, p_z_s = model(s_data.x, s_data.edge_index, s_data.batch, domain='source')
            t_recon, t_z, p_z_t = model(t_data.x, t_data.edge_index, t_data.batch, domain='target')

            loss_recon = reconstruction_loss(s_recon, s_data.x) + reconstruction_loss(t_recon, t_data.x)
            loss_ortho = ortho_loss(s_z, p_z_s) + ortho_loss(t_z, p_z_t)

            s_logits = classifier(s_z)
            t_logits = classifier(t_z)
            loss_cls = cls_criterion(s_logits, s_data.y.long()) + cls_criterion(t_logits, t_data.y.long())

            loss = loss_recon + lam_ortho * loss_ortho + lam_cls * loss_cls
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(all_params, grad_clip)
            optimizer.step()

            tot_recon += loss_recon.item()
            tot_ortho += loss_ortho.item()
            tot_cls   += loss_cls.item()
            tot_loss  += loss.item()

        avg_recon = tot_recon / steps
        avg_ortho = tot_ortho / steps
        avg_cls   = tot_cls   / steps
        avg_loss  = tot_loss  / steps

        # --- Validation ---
        model.eval()
        classifier.eval()
        with torch.no_grad():
            val_recon = val_ortho = val_cls = 0.0
            n_val = 0
            for s_data, t_data in zip(s_test_loader, itertools.cycle(t_test_loader)):
                s_data = s_data.to(device)
                t_data = t_data.to(device)
                s_recon, s_z, p_z_s = model(s_data.x, s_data.edge_index, s_data.batch, domain='source')
                t_recon, t_z, p_z_t = model(t_data.x, t_data.edge_index, t_data.batch, domain='target')
                val_recon += (reconstruction_loss(s_recon, s_data.x) + reconstruction_loss(t_recon, t_data.x)).item()
                val_ortho += (ortho_loss(s_z, p_z_s) + ortho_loss(t_z, p_z_t)).item()
                val_cls   += (cls_criterion(classifier(s_z), s_data.y.long()) +
                              cls_criterion(classifier(t_z), t_data.y.long())).item()
                n_val += 1
            if n_val > 0:
                val_recon /= n_val; val_ortho /= n_val; val_cls /= n_val
            val_total = val_recon + lam_ortho * val_ortho + lam_cls * val_cls

        scheduler.step(val_total)

        row = {
            'epoch': epoch + 1,
            'recon_loss': avg_recon, 'ortho_loss': avg_ortho, 'cls_loss': avg_cls, 'train_total': avg_loss,
            'val_recon': val_recon, 'val_ortho': val_ortho, 'val_cls': val_cls, 'val_total': val_total
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(os.path.join(save_dir, 'pretrain_log.csv'), index=False)

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_model_dict = copy.deepcopy(model.state_dict())
            best_cls_dict   = copy.deepcopy(classifier.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
            print(f"  [PT Ep {epoch+1:4d}] val_loss={val_total:.4f}  ← new best")
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            print(f"  Pretrain early stop at epoch {epoch+1} (patience={early_stop_patience})")
            break

    print(f"  Pretrain done. Best epoch={best_epoch}, best val_loss={best_val_loss:.4f}")
    plot_learning_curves(save_dir, 'pretrain_log.csv', 'Pretrain Learning Curve', 'pretrain')
    return best_model_dict, best_cls_dict, best_epoch


# ---------------------------------------------------------------------------
# Phase 2 – GAN
# ---------------------------------------------------------------------------

def gan_phase(model, classifier, discriminator,
              source_loader, target_loader,
              s_test_loader, t_test_loader,
              gan_param, device, save_dir, mapping_int2str):
    """
    WGAN-GP adversarial training on top of pretrained model.
    Discriminator on shared latent z_s vs z_t.
    Early stopping based on FID.
    Returns: state_dict of best model after GAN, and best epoch.
    """
    gan_epochs   = gan_param.get('gan_epochs', 500)
    base_lr      = gan_param.get('gan_learning_rate', 1e-4)
    disc_lr_mul  = gan_param.get('disc_lr_multiplier', 0.2)
    gen_lr_mul   = gan_param.get('gen_lr_multiplier', 0.01)
    n_critic     = gan_param.get('critic_iterations', 1)
    lam_gp       = gan_param.get('gradient_penalty_weight', 10.0)
    lam_gan      = gan_param.get('lambda_gan', 0.01)
    lam_ortho    = gan_param.get('lambda_ortho', 1e-5)
    lam_recon    = gan_param.get('lambda_vae', 0.1)  # reconstruction weight during GAN
    es_patience  = gan_param.get('early_stopping_patience', 50)
    beta1_g      = gan_param.get('beta1', 0.5)
    beta2_g      = gan_param.get('beta2', 0.9)
    disc_dop     = gan_param.get('disc_dropout', 0.1)

    disc_lr = base_lr * disc_lr_mul
    gen_lr  = base_lr * gen_lr_mul

    optimizer_D  = optim.RMSprop(discriminator.parameters(), lr=disc_lr)
    optimizer_G  = optim.RMSprop(
        list(model.parameters()) + list(classifier.parameters()), lr=gen_lr)

    best_fid = float('inf')
    best_model_dict = copy.deepcopy(model.state_dict())
    best_cls_dict   = copy.deepcopy(classifier.state_dict())
    best_epoch = 0
    no_improve = 0

    logs = []
    steps = max(len(source_loader), len(target_loader))
    cls_criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*60}")
    print(f"  Phase 2 — GAN  ({gan_epochs} epochs max, early stop pat={es_patience})")
    print(f"{'='*60}")

    for epoch in range(gan_epochs):
        model.train()
        discriminator.train()
        classifier.train()

        tot_d = tot_g = tot_recon = tot_ortho = tot_cls = 0.0

        if len(source_loader) < len(target_loader):
            loader_zip = zip(itertools.cycle(source_loader), target_loader)
        else:
            loader_zip = zip(source_loader, itertools.cycle(target_loader))

        pbar = tqdm(loader_zip, total=steps, desc=f"GAN Ep {epoch+1}/{gan_epochs}", leave=False)
        for step, (s_data, t_data) in enumerate(pbar):
            s_data = s_data.to(device)
            t_data = t_data.to(device)

            # --- Train Discriminator ---
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                with torch.no_grad():
                    _, s_z, _ = model(s_data.x, s_data.edge_index, s_data.batch, domain='source')
                    _, t_z, _ = model(t_data.x, t_data.edge_index, t_data.batch, domain='target')
                d_real = discriminator(s_z)
                d_fake = discriminator(t_z)
                gp = compute_gradient_penalty(discriminator, s_z.detach(), t_z.detach())
                d_loss = -torch.mean(d_real) + torch.mean(d_fake) + lam_gp * gp
                d_loss.backward()
                optimizer_D.step()
                tot_d += d_loss.item()

            # --- Train Generator (AE + Classifier) ---
            optimizer_G.zero_grad()
            s_recon, s_z, p_z_s = model(s_data.x, s_data.edge_index, s_data.batch, domain='source')
            t_recon, t_z, p_z_t = model(t_data.x, t_data.edge_index, t_data.batch, domain='target')

            loss_recon = reconstruction_loss(s_recon, s_data.x) + reconstruction_loss(t_recon, t_data.x)
            loss_ortho = ortho_loss(s_z, p_z_s) + ortho_loss(t_z, p_z_t)
            g_loss_adv = -torch.mean(discriminator(t_z))
            loss_cls   = cls_criterion(classifier(s_z), s_data.y.long()) + \
                         cls_criterion(classifier(t_z), t_data.y.long())

            g_total = lam_recon * loss_recon + lam_ortho * loss_ortho + lam_gan * g_loss_adv + loss_cls
            g_total.backward()
            optimizer_G.step()

            tot_g     += g_loss_adv.item()
            tot_recon += loss_recon.item()
            tot_ortho += loss_ortho.item()
            tot_cls   += loss_cls.item()

        # --- Evaluation: FID ---
        model.eval()
        with torch.no_grad():
            fid_score, z_s, z_t, labels_s, labels_t = compute_fid(model, s_test_loader, t_test_loader, device)
            mmd = calculate_mmd_metric(z_s, z_t)
            wass = calculate_wasserstein_metric(z_s, z_t)

        row = {
            'gan_epoch': epoch + 1,
            'd_loss': tot_d / (steps * n_critic),
            'g_loss': tot_g / steps,
            'recon_loss': tot_recon / steps,
            'ortho_loss': tot_ortho / steps,
            'cls_loss': tot_cls / steps,
            'fid': fid_score, 'mmd': mmd, 'wasserstein': wass
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(os.path.join(save_dir, 'gan_log.csv'), index=False)

        if fid_score < best_fid:
            best_fid = fid_score
            best_model_dict = copy.deepcopy(model.state_dict())
            best_cls_dict   = copy.deepcopy(classifier.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
            print(f"  [GAN Ep {epoch+1:4d}] FID={fid_score:.3f}  ← new best")
        else:
            no_improve += 1

        if no_improve >= es_patience:
            print(f"  GAN early stop at epoch {epoch+1} (patience={es_patience})")
            break

    print(f"  GAN done. Best epoch={best_epoch}, best FID={best_fid:.4f}")
    plot_learning_curves(save_dir, 'gan_log.csv', 'GAN Learning Curve', 'gan')
    return best_model_dict, best_cls_dict, best_epoch, best_fid, z_s, z_t, labels_s, labels_t


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def train_single_experiment(pretrain_param, gan_param, gnn_param,
                             source_loader, target_loader,
                             s_test_loader, t_test_loader,
                             gene_list, edge_index, edge_attr,
                             device, parent_folder, mapping_int2str, exp_id):

    exp_name = f"exp_{exp_id}"
    save_dir = os.path.join(parent_folder, exp_name)
    safemakedirs(save_dir)

    # --- GNN params (from gnn_param) ---
    latent_dim      = gnn_param.get('latent_dim', 32)
    gnn_hidden_dims = gnn_param.get('gnn_hidden_dims', None)
    if gnn_hidden_dims is None:
        # Fallback: build dims from scalar gnn_hidden_dim × num_layers
        num_layers  = pretrain_param.get('num_layers', 2)
        base_hidden = gnn_param.get('gnn_hidden_dim', 64)
        gnn_hidden_dims = [base_hidden] * num_layers
    decoder_hidden_dims = list(reversed(gnn_hidden_dims))
    gnn_heads       = gnn_param.get('gnn_heads', 2)
    gnn_dropout     = gnn_param.get('gnn_dropout', pretrain_param.get('dropout_rate', 0.1))
    gnn_pool_ratios = gnn_param.get('gnn_pool_ratios', [0.5] * len(gnn_hidden_dims))
    # Ensure pool_ratios length matches hidden_dims length
    if len(gnn_pool_ratios) < len(gnn_hidden_dims):
        gnn_pool_ratios = [gnn_pool_ratios[0]] * len(gnn_hidden_dims)

    num_nodes   = len(gene_list)
    num_classes = len(mapping_int2str)

    model = GNNAutoencoder(
        num_nodes=num_nodes,
        latent_dim=latent_dim,
        gnn_hidden_dims=gnn_hidden_dims,
        gnn_heads=gnn_heads,
        gnn_dropout=gnn_dropout,
        gnn_pool_ratios=gnn_pool_ratios,
        decoder_hidden_dims=decoder_hidden_dims,
        device=device
    ).to(device)

    classifier = Classifier(input_dim=latent_dim, num_classes=num_classes).to(device)

    # --- Print experiment header ---
    print(f"\n{'='*65}")
    print(f"Starting experiment: {exp_name}")
    print(f"{'='*65}")
    for k, v in pretrain_param.items():
        print(f"  [pretrain] {k:<30} = {v}")
    for k, v in gan_param.items():
        print(f"  [gan]      {k:<30} = {v}")
    for k, v in gnn_param.items():
        print(f"  [gnn]      {k:<30} = {v}")
    print(f"{'='*65}")

    # Save combined params
    combined_param = {**{f'pretrain_{k}': v for k, v in pretrain_param.items()},
                      **{f'gan_{k}': v for k, v in gan_param.items()},
                      **{f'gnn_{k}': v for k, v in gnn_param.items()}}
    def _make_serializable(d):
        return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool, list, dict))}
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(_make_serializable(combined_param), f, indent=4)

    # ===================================================================
    # Phase 1 — Pretrain
    # ===================================================================
    best_pt_dict, best_cls_pt_dict, pt_best_epoch = pretrain_phase(
        model, classifier,
        source_loader, target_loader,
        s_test_loader, t_test_loader,
        pretrain_param, device, save_dir, mapping_int2str
    )

    # Save pretrain best weights
    torch.save(best_pt_dict,     os.path.join(save_dir, 'pretrain_best.pth'))
    torch.save(best_cls_pt_dict, os.path.join(save_dir, 'pretrain_best_cls.pth'))

    # t-SNE after pretrain (best model)
    model.load_state_dict(best_pt_dict)
    classifier.load_state_dict(best_cls_pt_dict)
    model.eval()
    with torch.no_grad():
        z_s, labels_s = collect_latents(model, s_test_loader, 'source', device)
        z_t, labels_t = collect_latents(model, t_test_loader, 'target', device)
    print(f"  Drawing t-SNE after pretrain (epoch {pt_best_epoch})...")
    visualize_tsne_with_tumor_types(z_s, z_t, labels_s, labels_t,
                                     mapping_int2str, pt_best_epoch, save_dir, is_pretrained=True)

    # (Pretrain latents not saved — only GAN-phase latents are kept for downstream use)

    # ===================================================================
    # Phase 2 — GAN
    # ===================================================================
    # Reload best pretrain weights into model
    model.load_state_dict(best_pt_dict)
    classifier.load_state_dict(best_cls_pt_dict)

    disc_input_dim = latent_dim  # discriminator on shared z only
    disc_dropout   = gan_param.get('disc_dropout', 0.1)
    discriminator  = Discriminator(input_dim=disc_input_dim,
                                    hidden_dim=[64, 32]).to(device)

    best_gan_dict, best_cls_gan_dict, gan_best_epoch, best_fid, \
        z_s_gan, z_t_gan, labels_s_gan, labels_t_gan = gan_phase(
            model, classifier, discriminator,
            source_loader, target_loader,
            s_test_loader, t_test_loader,
            gan_param, device, save_dir, mapping_int2str
        )

    # Save GAN best weights
    torch.save(best_gan_dict,     os.path.join(save_dir, 'gan_best.pth'))
    torch.save(best_cls_gan_dict, os.path.join(save_dir, 'gan_best_cls.pth'))

    with open(os.path.join(save_dir, 'best_fid.txt'), 'w') as f:
        f.write(f"Epoch: {gan_best_epoch}, FID: {best_fid}\n")

    # t-SNE after GAN (best model)
    model.load_state_dict(best_gan_dict)
    model.eval()
    print(f"  Drawing t-SNE after GAN (epoch {gan_best_epoch})...")
    visualize_tsne_with_tumor_types(z_s_gan, z_t_gan, labels_s_gan, labels_t_gan,
                                     mapping_int2str, gan_best_epoch, save_dir, is_pretrained=False)

    # Save latents for GAN phase
    np.save(os.path.join(save_dir, 'gan_latents_source.npy'), z_s_gan.numpy())
    np.save(os.path.join(save_dir, 'gan_latents_target.npy'), z_t_gan.numpy())
    np.save(os.path.join(save_dir, 'gan_labels_source.npy'),  labels_s_gan.numpy())
    np.save(os.path.join(save_dir, 'gan_labels_target.npy'),  labels_t_gan.numpy())

    # Build summary row
    final_stats = {
        'exp_id': exp_id,
        'pt_best_epoch': pt_best_epoch,
        'gan_best_epoch': gan_best_epoch,
        'best_fid': best_fid,
    }
    for k, v in {**pretrain_param, **gan_param}.items():
        final_stats[k] = v if isinstance(v, (int, float, str, bool)) else str(v)

    return final_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      type=str, default='config/params_grid_pretrain_fix.json')
    parser.add_argument('--outfolder',   type=str, default='result/gnn_ae_tuning')
    parser.add_argument('--source_path', type=str, default='data/pretrain_ccle.csv')
    parser.add_argument('--target_path', type=str, default='data/pretrain_tcga.csv')
    parser.add_argument('--ppi_path',    type=str, default='data/Edge/string_interactions_short.tsv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    experiments = load_params_grid(args.config)
    print(f"Total experiments: {len(experiments)}")

    if len(experiments) == 0:
        print("No experiments found. Exiting.")
        return

    # --- Data loading ---
    print("Loading data...")
    # Use batch_size and normalization from first pretrain_param; default 32
    first_pt = experiments[0][0]
    batch_size = first_pt.get('batch_size', 32)

    (source_loader_orig, s_test, s_test_labels), \
    (target_loader_orig, t_test, t_test_labels), \
    mapping_int2str, gene_to_idx = load_data_with_labels_and_gene_mapping(
        ccle_path=args.source_path,
        xena_path=args.target_path,
        batch_size=batch_size,
        device=device,
        normalization_method=first_pt.get('normalization_method', 'zscore')
    )
    gene_list = list(gene_to_idx.keys())

    # --- Load PPI file once (graph edges built per-experiment inside loop) ---
    ppi_df = pd.read_csv(args.ppi_path, sep='\t' if args.ppi_path.endswith('.tsv') else ',')
    col_map = {'GeneA': '#node1', 'GeneB': 'node2', 'PPI_score': 'combined_score'}

    # Pre-extract tensors (shared across experiments)
    if hasattr(source_loader_orig.dataset, 'tensors'):
        s_train_x, s_train_y = source_loader_orig.dataset.tensors
        t_train_x, t_train_y = target_loader_orig.dataset.tensors
    else:
        s_train_x = torch.cat([x for x, y in source_loader_orig])
        s_train_y = torch.cat([y for x, y in source_loader_orig])
        t_train_x = torch.cat([x for x, y in target_loader_orig])
        t_train_y = torch.cat([y for x, y in target_loader_orig])

    safemakedirs(args.outfolder)
    all_results = []

    for i, (pretrain_param, gan_param, gnn_param) in enumerate(experiments, 1):
        curr_batch = pretrain_param.get('batch_size', 32)

        # --- Build graph for this experiment's GNN_params ---
        edge_processor = PPIEdgeProcessor(
            ppi_df, gene_list,
            mode=gnn_param.get('graph_mode', 'B'),
            threshold=gnn_param.get('threshold', 0.95),
            low_score=gnn_param.get('low_score', 0.1),
            complete_graph=gnn_param.get('complete_graph', False),
            col_map=col_map
        )
        edge_index, edge_attr = edge_processor.process()
        edge_index = edge_index.to(device)
        edge_attr  = edge_attr.to(device)

        source_dataset = GraphDataset(s_train_x, s_train_y, edge_index, edge_attr)
        target_dataset = GraphDataset(t_train_x, t_train_y, edge_index, edge_attr)
        s_loader = DataLoader(source_dataset, batch_size=curr_batch, shuffle=True, drop_last=True)
        t_loader = DataLoader(target_dataset, batch_size=curr_batch, shuffle=True, drop_last=True)

        s_test_dataset = GraphDataset(s_test, s_test_labels, edge_index, edge_attr)
        t_test_dataset = GraphDataset(t_test, t_test_labels, edge_index, edge_attr)
        s_test_loader  = DataLoader(s_test_dataset, batch_size=curr_batch, shuffle=False)
        t_test_loader  = DataLoader(t_test_dataset, batch_size=curr_batch, shuffle=False)

        stats = train_single_experiment(
            pretrain_param, gan_param, gnn_param,
            s_loader, t_loader, s_test_loader, t_test_loader,
            gene_list, edge_index, edge_attr,
            device, args.outfolder, mapping_int2str, exp_id=i
        )
        all_results.append(stats)
        torch.cuda.empty_cache()

    summary_df   = pd.DataFrame(all_results)
    summary_path = os.path.join(args.outfolder, 'summary_results.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAll experiments done. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
