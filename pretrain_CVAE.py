"""
CVAEwC pretraining + GAN alignment pipeline.

This file follows the same workflow as `pretrain_VAEwC.py`:
- parameter grid/combinations
- exp_XXX output folders
- overlap filtering for TCGA
- GAN phase weight export
- latent dict export
- GAN metrics/t-SNE output

The backbone is CVAE (VAE -> CVAE), and the cancer-type classifier head
(`PrimaryClassifier`) is kept for supervised alignment.
"""

import os
import json
import copy
import pickle
import argparse
import itertools
from collections import defaultdict
from itertools import chain, cycle
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.dataprocess import safemakedirs, append_csv_log
from tools.model_opt import MLP, Discriminator, vaeloss, init_weights, ortho_loss, compute_gradient_penalty

import pretrain_VAEwC as core

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. No GPU detected.")

device = core.device


class CVAEEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_dims: List[int],
        num_classes: int,
        cond_embed_dim: int = 16,
        dop: float = 0.1,
        act_fn=nn.ReLU,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cond_embed = nn.Linear(num_classes, cond_embed_dim)
        self.cond_act = nn.ReLU()
        self.mlp = MLP(
            input_dim=input_size + cond_embed_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims,
            dop=dop,
            act_fn=act_fn,
        )
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_size)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_size)

    def _condition_vec(self, labels, batch_size, device_):
        if labels is None:
            c = torch.zeros(batch_size, self.num_classes, device=device_)
        else:
            c = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        return self.cond_act(self.cond_embed(c))

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        c = self._condition_vec(labels, x.shape[0], x.device)
        h = self.mlp(torch.cat([x, c], dim=1))
        return self.mu_layer(h), self.logvar_layer(h)


class CVAEDecoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        output_size: int,
        hidden_dims: List[int],
        num_classes: int,
        cond_embed_dim: int = 16,
        dop: float = 0.1,
        act_fn=nn.ReLU,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cond_embed = nn.Linear(num_classes, cond_embed_dim)
        self.cond_act = nn.ReLU()
        self.mlp = MLP(
            input_dim=latent_size + cond_embed_dim,
            output_dim=output_size,
            hidden_dims=hidden_dims,
            dop=dop,
            act_fn=act_fn,
        )

    def _condition_vec(self, labels, batch_size, device_):
        if labels is None:
            c = torch.zeros(batch_size, self.num_classes, device=device_)
        else:
            c = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        return self.cond_act(self.cond_embed(c))

    def forward(self, z: torch.Tensor, labels: torch.Tensor = None):
        c = self._condition_vec(labels, z.shape[0], z.device)
        return self.mlp(torch.cat([z, c], dim=1))


class CVAE(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        latent_size: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        num_classes: int,
        cond_embed_dim: int = 16,
        dop: float = 0.1,
        act_fn=nn.ReLU,
    ):
        super().__init__()
        self.encoder = CVAEEncoder(
            input_size=input_size,
            latent_size=latent_size,
            hidden_dims=encoder_hidden_dims,
            num_classes=num_classes,
            cond_embed_dim=cond_embed_dim,
            dop=dop,
            act_fn=act_fn,
        )
        self.decoder = CVAEDecoder(
            latent_size=latent_size,
            output_size=output_size,
            hidden_dims=decoder_hidden_dims,
            num_classes=num_classes,
            cond_embed_dim=cond_embed_dim,
            dop=dop,
            act_fn=act_fn,
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        mu, logvar = self.encoder(x, labels)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        re_x = self.decoder(z, labels)
        return re_x, z, mu, logvar


def _encode_latent_dict(model: CVAE, feature_df: pd.DataFrame, batch_size=512):
    model.eval()
    latents = {}
    ids = feature_df.index.astype(str).tolist()
    x = torch.from_numpy(feature_df.values).float().to(device)
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            end = min(len(ids), start + batch_size)
            _, z, _, _ = model(x[start:end], None)
            z_np = z.detach().cpu().numpy()
            for i, sid in enumerate(ids[start:end]):
                latents[sid] = z_np[i].tolist()
    return latents


def train_discrim(s_batch, t_batch, s_labels, t_labels, shared_encoder, sencoder, tencoder, discrim, optimizer, scheduler):
    loss_log = defaultdict(float)
    shared_encoder.zero_grad()
    sencoder.zero_grad()
    tencoder.zero_grad()
    discrim.zero_grad()
    sencoder.eval()
    tencoder.eval()
    shared_encoder.eval()
    discrim.train()
    optimizer.zero_grad()
    with torch.no_grad():
        _, pzs, _, _ = sencoder(s_batch, s_labels)
        _, pzt, _, _ = tencoder(t_batch, t_labels)
        _, zs, _, _ = shared_encoder(s_batch, s_labels)
        _, zt, _, _ = shared_encoder(t_batch, t_labels)
    s = torch.cat((zs, pzs), dim=1)
    t = torch.cat((zt, pzt), dim=1)
    d_loss = torch.mean(t) - torch.mean(s)
    g_p = compute_gradient_penalty(critic=discrim, real_samples=s, fake_samples=t, device=device)
    loss_log.update({"discrim_loss": d_loss, "g_p": g_p})
    d_loss = d_loss + 10 * g_p
    d_loss.backward()
    optimizer.step()
    scheduler.step()
    discrim.eval()
    return loss_log


def train_d_ae(s_batch, t_batch, s_labels, t_labels, shared_encoder, sencoder, tencoder, discrim, classifier, optimizer, scheduler, lambda_cls, source_weights=None, target_weights=None, use_class_weight=False):
    loss_log = defaultdict(float)
    shared_encoder.zero_grad()
    sencoder.zero_grad()
    tencoder.zero_grad()
    discrim.zero_grad()
    classifier.zero_grad()
    sencoder.train()
    tencoder.train()
    shared_encoder.train()
    discrim.eval()
    classifier.train()
    optimizer.zero_grad()
    pccle_re_x, pccle_z, pccle_mu, pccle_sigma = sencoder(s_batch, s_labels)
    pccle_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, s_batch)
    ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = tencoder(t_batch, t_labels)
    ptcga_vae_loss = vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, t_batch)
    ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_encoder(s_batch, s_labels)
    ccle_vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, s_batch)
    tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_encoder(t_batch, t_labels)
    tcga_vae_loss = vaeloss(tcga_mu, tcga_sigma, tcga_re_x, t_batch)
    if use_class_weight and source_weights is not None and target_weights is not None:
        s_cls_criterion = nn.CrossEntropyLoss(weight=source_weights)
        t_cls_criterion = nn.CrossEntropyLoss(weight=target_weights)
        cls_loss = s_cls_criterion(classifier(ccle_z), s_labels) + t_cls_criterion(classifier(tcga_z), t_labels)
    else:
        cls_criterion = nn.CrossEntropyLoss()
        cls_loss = cls_criterion(classifier(ccle_z), s_labels) + cls_criterion(classifier(tcga_z), t_labels)
    pvae_loss = pccle_vae_loss + ptcga_vae_loss
    vae_loss = ccle_vae_loss + tcga_vae_loss
    o_loss = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)
    g_loss = -torch.mean(discrim(torch.cat((tcga_z, ptcga_z), dim=1)))
    loss = o_loss + g_loss + vae_loss + pvae_loss + lambda_cls * cls_loss
    loss_log.update({"ortho_loss": o_loss, "pvae_loss": pvae_loss, "gen_loss": g_loss, "vae_loss": vae_loss, "cls_loss": cls_loss})
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss_log


def run_single_experiment(sourcedata, targetdata, param, exp_name, exp_dir, ccle_df_for_latent, tcga_df_for_latent):
    print(f"start experiment {exp_name}")
    use_class_weight = param.get("use_class_weight", False)
    trainloss_csv = os.path.join(exp_dir, "pretrain_loss.csv")
    evalloss_csv = os.path.join(exp_dir, "pretrain_eval_loss.csv")
    dloss_csv = os.path.join(exp_dir, "d_loss.csv")
    genloss_csv = os.path.join(exp_dir, "g_loss.csv")
    sourcetrainloader, sourcetest, source_test_labels = sourcedata[0], sourcedata[1], sourcedata[2]
    targettrainloader, targettest, target_test_labels = targetdata[0], targetdata[1], targetdata[2]
    if use_class_weight:
        source_weights, target_weights, mapping_int2str = sourcedata[3], targetdata[3], sourcedata[4]
    else:
        source_weights = None
        target_weights = None
        mapping_int2str = sourcedata[3]

    config_payload = {"exp_id": exp_name, "device": str(device), "params": core._json_safe(param), "use_class_weight": use_class_weight}
    with open(os.path.join(exp_dir, "params.json"), "w") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    num_classes = len(mapping_int2str)
    input_size = sourcetest.shape[1]
    latent_size = param.get("latent_size", 32)
    encoder_hidden_dims = param["encoder_dims"]
    decoder_hidden_dims = encoder_hidden_dims[::-1]
    dropout_rate = param["dropout_rate"]
    lambda_cls = param["lambda_cls"]
    cond_embed_dim = int(param.get("cond_embed_dim", 16))

    shared_cvae = CVAE(input_size=input_size, output_size=input_size, latent_size=latent_size, encoder_hidden_dims=encoder_hidden_dims, decoder_hidden_dims=decoder_hidden_dims, num_classes=num_classes, cond_embed_dim=cond_embed_dim, dop=dropout_rate, act_fn=nn.ReLU).to(device)
    source_private_cvae = CVAE(input_size=input_size, output_size=input_size, latent_size=latent_size, encoder_hidden_dims=encoder_hidden_dims, decoder_hidden_dims=decoder_hidden_dims, num_classes=num_classes, cond_embed_dim=cond_embed_dim, dop=dropout_rate, act_fn=nn.ReLU).to(device)
    target_private_cvae = CVAE(input_size=input_size, output_size=input_size, latent_size=latent_size, encoder_hidden_dims=encoder_hidden_dims, decoder_hidden_dims=decoder_hidden_dims, num_classes=num_classes, cond_embed_dim=cond_embed_dim, dop=dropout_rate, act_fn=nn.ReLU).to(device)
    cancer_classifier = core.PrimaryClassifier(input_dim=latent_size, num_classes=num_classes, hidden_dims=[64, 32], dop=0.2, act_fn=nn.ReLU).to(device)
    for m in [shared_cvae, source_private_cvae, target_private_cvae, cancer_classifier]:
        m.apply(init_weights)

    source_dict = copy.deepcopy(source_private_cvae.state_dict())
    shared_dict = copy.deepcopy(shared_cvae.state_dict())
    target_dict = copy.deepcopy(target_private_cvae.state_dict())
    classifier_dict = copy.deepcopy(cancer_classifier.state_dict())

    pretrain_epochs = param["pretrain_num_epochs"]
    pre_lr = param["pretrain_learning_rate"]
    pre_tol = 0
    pre_tol_max = param.get("pretrain_patience", 50)
    min_eval_loss = float("inf")
    models = [shared_cvae, source_private_cvae, target_private_cvae, cancer_classifier]
    optimizer = torch.optim.Adam(chain(*(m.parameters() for m in models)), lr=pre_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, pretrain_epochs))
    if use_class_weight and source_weights is not None and target_weights is not None:
        s_cls_criterion = nn.CrossEntropyLoss(weight=source_weights)
        t_cls_criterion = nn.CrossEntropyLoss(weight=target_weights)
    else:
        cls_criterion = nn.CrossEntropyLoss()

    for epoch in range(pretrain_epochs):
        train_ol, train_pv, train_v, train_c = 0.0, 0.0, 0.0, 0.0
        steps = 0
        target_cycle = cycle(targettrainloader)
        for ccledata, ccle_labels in sourcetrainloader:
            tcgadata, tcga_labels = next(target_cycle)
            optimizer.zero_grad()
            pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_cvae(ccledata, ccle_labels)
            ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_cvae(tcgadata, tcga_labels)
            ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_cvae(ccledata, ccle_labels)
            tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_cvae(tcgadata, tcga_labels)
            p_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, ccledata) + vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, tcgadata)
            vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, ccledata) + vaeloss(tcga_mu, tcga_sigma, tcga_re_x, tcgadata)
            o_loss = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)
            if use_class_weight and source_weights is not None and target_weights is not None:
                cls_loss = s_cls_criterion(cancer_classifier(ccle_z), ccle_labels) + t_cls_criterion(cancer_classifier(tcga_z), tcga_labels)
            else:
                cls_loss = cls_criterion(cancer_classifier(ccle_z), ccle_labels) + cls_criterion(cancer_classifier(tcga_z), tcga_labels)
            loss = o_loss + vae_loss + p_vae_loss + lambda_cls * cls_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_ol += core._to_scalar(o_loss)
            train_pv += core._to_scalar(p_vae_loss)
            train_v += core._to_scalar(vae_loss)
            train_c += core._to_scalar(cls_loss)
            steps += 1

        append_csv_log(trainloss_csv, {
            "epoch": epoch + 1,
            "ortholoss": train_ol / max(1, steps),
            "pCVAE_loss": train_pv / max(1, steps),
            "CVAE_loss": train_v / max(1, steps),
            "cls_loss": train_c / max(1, steps),
        })

        with torch.no_grad():
            pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_cvae(sourcetest, source_test_labels)
            ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_cvae(targettest, target_test_labels)
            ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_cvae(sourcetest, source_test_labels)
            tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_cvae(targettest, target_test_labels)
            eval_p = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, sourcetest) + vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, targettest)
            eval_v = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, sourcetest) + vaeloss(tcga_mu, tcga_sigma, tcga_re_x, targettest)
            eval_o = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)
            if use_class_weight and source_weights is not None and target_weights is not None:
                eval_cls = s_cls_criterion(cancer_classifier(ccle_z), source_test_labels) + t_cls_criterion(cancer_classifier(tcga_z), target_test_labels)
            else:
                eval_cls = cls_criterion(cancer_classifier(ccle_z), source_test_labels) + cls_criterion(cancer_classifier(tcga_z), target_test_labels)
            eval_total = eval_o + eval_p + eval_v + lambda_cls * eval_cls
            append_csv_log(evalloss_csv, {
                "epoch": epoch + 1,
                "ortholoss": core._to_scalar(eval_o),
                "pCVAE_loss": core._to_scalar(eval_p),
                "CVAE_loss": core._to_scalar(eval_v),
                "cls_loss": core._to_scalar(eval_cls),
            })
            if core._to_scalar(eval_total) < min_eval_loss:
                min_eval_loss = core._to_scalar(eval_total)
                pre_tol = 0
                source_dict = copy.deepcopy(source_private_cvae.state_dict())
                target_dict = copy.deepcopy(target_private_cvae.state_dict())
                shared_dict = copy.deepcopy(shared_cvae.state_dict())
                classifier_dict = copy.deepcopy(cancer_classifier.state_dict())
            else:
                pre_tol += 1
                if pre_tol >= pre_tol_max:
                    print(f"pretrain early stop @ epoch {epoch + 1}")
                    break

    core._plot_pretrain_curves(trainloss_csv, evalloss_csv, exp_dir)
    shared_cvae.load_state_dict(shared_dict)
    source_private_cvae.load_state_dict(source_dict)
    target_private_cvae.load_state_dict(target_dict)
    cancer_classifier.load_state_dict(classifier_dict)

    gan_epoch = param["train_num_epochs"]
    gan_lr = param["gan_learning_rate"]
    discrim = Discriminator(input_dim=latent_size * 2, dop=dropout_rate).to(device)
    discrim.apply(init_weights)
    discrim_optimizer = torch.optim.RMSprop(discrim.parameters(), lr=gan_lr)
    discrim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discrim_optimizer, max(1, gan_epoch))
    d_ae_optimizer = torch.optim.RMSprop(chain(shared_cvae.parameters(), source_private_cvae.parameters(), target_private_cvae.parameters(), cancer_classifier.parameters()), lr=gan_lr)
    d_ae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_ae_optimizer, max(1, gan_epoch))

    max_gan_tolerance = param.get("gan_patience", 20)
    gan_tolerance = 0
    gan_best_epoch = 0
    gan_best_loss = float("inf")
    shared_after = copy.deepcopy(shared_cvae.state_dict())
    classifier_after = copy.deepcopy(cancer_classifier.state_dict())
    source_after = copy.deepcopy(source_private_cvae.state_dict())
    target_after = copy.deepcopy(target_private_cvae.state_dict())
    discrim_after = copy.deepcopy(discrim.state_dict())

    for epoch in range(gan_epoch):
        dloss_list = []
        genloss_list = []
        target_cycle = cycle(targettrainloader)
        for step, (ccledata, ccle_labels) in enumerate(sourcetrainloader):
            tcgadata, tcga_labels = next(target_cycle)
            dloss_list.append(train_discrim(ccledata, tcgadata, ccle_labels, tcga_labels, shared_cvae, source_private_cvae, target_private_cvae, discrim, discrim_optimizer, discrim_scheduler))
            if (step + 1) % 5 == 0:
                genloss_list.append(train_d_ae(ccledata, tcgadata, ccle_labels, tcga_labels, shared_cvae, source_private_cvae, target_private_cvae, discrim, cancer_classifier, d_ae_optimizer, d_ae_scheduler, lambda_cls, source_weights, target_weights, use_class_weight))
        if not dloss_list:
            continue
        dloss_mean = defaultdict(float)
        for log in dloss_list:
            for k, v in log.items():
                dloss_mean[k] += core._to_scalar(v)
        for k in dloss_mean:
            dloss_mean[k] /= len(dloss_list)
        genloss_mean = defaultdict(float)
        for log in genloss_list:
            for k, v in log.items():
                genloss_mean[k] += core._to_scalar(v)
        for k in genloss_mean:
            genloss_mean[k] /= max(1, len(genloss_list))
        dloss_mean["epoch"] = epoch + 1
        genloss_mean["epoch"] = epoch + 1
        append_csv_log(dloss_csv, dloss_mean)
        append_csv_log(genloss_csv, genloss_mean)
        temp_loss = sum(v for k, v in dloss_mean.items() if k != "epoch") + sum(v for k, v in genloss_mean.items() if k != "epoch")
        if temp_loss < gan_best_loss:
            gan_best_loss = temp_loss
            gan_tolerance = 0
            gan_best_epoch = epoch + 1
            shared_after = copy.deepcopy(shared_cvae.state_dict())
            classifier_after = copy.deepcopy(cancer_classifier.state_dict())
            source_after = copy.deepcopy(source_private_cvae.state_dict())
            target_after = copy.deepcopy(target_private_cvae.state_dict())
            discrim_after = copy.deepcopy(discrim.state_dict())
        else:
            gan_tolerance += 1
            if gan_tolerance >= max_gan_tolerance:
                print(f"gan early stop @ epoch {epoch + 1}")
                break

    core._plot_gan_curves(dloss_csv, genloss_csv, exp_dir)
    shared_cvae.load_state_dict(shared_after)
    source_private_cvae.load_state_dict(source_after)
    target_private_cvae.load_state_dict(target_after)
    discrim.load_state_dict(discrim_after)
    cancer_classifier.load_state_dict(classifier_after)

    torch.save(shared_after, os.path.join(exp_dir, "after_traingan_shared_cvae.pth"))
    torch.save(source_after, os.path.join(exp_dir, "after_traingan_source_cvae.pth"))
    torch.save(target_after, os.path.join(exp_dir, "after_traingan_target_cvae.pth"))
    torch.save(classifier_after, os.path.join(exp_dir, "after_traingan_classifier.pth"))
    torch.save(discrim_after, os.path.join(exp_dir, "after_traingan_discriminator.pth"))

    ccle_latent_dict = _encode_latent_dict(shared_cvae, ccle_df_for_latent)
    tcga_latent_raw_dict = _encode_latent_dict(shared_cvae, tcga_df_for_latent)
    tcga_latent_dict = core.deduplicate_tcga_latent_dict(tcga_latent_raw_dict)
    with open(os.path.join(exp_dir, "ccle_latent_dict.pkl"), "wb") as f:
        pickle.dump(ccle_latent_dict, f)
    with open(os.path.join(exp_dir, "tcga_latent_dict.pkl"), "wb") as f:
        pickle.dump(tcga_latent_dict, f)

    source_latent = np.asarray(list(ccle_latent_dict.values()), dtype=np.float32)
    target_latent = np.asarray(list(tcga_latent_dict.values()), dtype=np.float32)
    with torch.no_grad():
        _, source_test_z, _, _ = shared_cvae(sourcetest, source_test_labels)
        _, target_test_z, _, _ = shared_cvae(targettest, target_test_labels)
    source_true = source_test_labels.detach().cpu().numpy()
    target_true = target_test_labels.detach().cpu().numpy()
    source_test_latent = source_test_z.detach().cpu().numpy()
    target_test_latent = target_test_z.detach().cpu().numpy()
    cluster_metrics = core._kmeans_combined_metrics(
        source_test_latent,
        target_test_latent,
        source_true,
        target_true,
        len(mapping_int2str),
    )
    metrics = {
        "exp_id": exp_name,
        "best_gan_epoch": gan_best_epoch,
        "best_gan_loss": gan_best_loss,
        "fid": core._compute_fid(source_latent),
        "mmd": core._calculate_mmd(source_latent, target_latent),
        "wasserstein": core._calculate_wasserstein(source_latent, target_latent),
        "tcga_raw_sample_count_for_latent": int(len(tcga_latent_raw_dict)),
        "tcga_patient_count_for_latent": int(len(tcga_latent_dict)),
    }
    metrics.update(cluster_metrics)
    with open(os.path.join(exp_dir, "gan_metrics.json"), "w") as f:
        json.dump(core._json_safe(metrics), f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(exp_dir, "gan_metrics.csv"), index=False)
    core._plot_gan_tsne(
        source_test_z.detach().cpu().numpy(),
        target_test_z.detach().cpu().numpy(),
        source_true,
        target_true,
        mapping_int2str,
        os.path.join(exp_dir, "tsne_gan_best.png"),
    )
    with open(os.path.join(exp_dir, "run_summary.json"), "w") as f:
        json.dump(core._json_safe({
            "exp_id": exp_name,
            "params": param,
            "metrics": metrics,
            "artifacts": {
                "weights": [
                    "after_traingan_shared_cvae.pth",
                    "after_traingan_source_cvae.pth",
                    "after_traingan_target_cvae.pth",
                    "after_traingan_classifier.pth",
                    "after_traingan_discriminator.pth",
                ],
                "latents": ["ccle_latent_dict.pkl", "tcga_latent_dict.pkl"],
                "plots": ["tsne_gan_best.png", "gan_learning_curve.png", "pretrain_learning_curve.png"],
            },
        }), f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser("pretrain_CVAE")
    parser.add_argument("--outfolder", default="./result/pretrain_cvae", type=str, help="output folder")
    parser.add_argument("--target_domain", default="tcga", choices=["tcga", "pdx"], type=str, help="target domain selection")
    parser.add_argument("--target", default=None, type=str, help="target expression csv path (optional, auto by target_domain if not set)")
    parser.add_argument("--target_response", default=None, type=str, help="target response csv path (optional, auto by target_domain if not set)")
    parser.add_argument("--target_cancer_reference", default=None, type=str, help="target cancer reference csv path (optional, auto by target_domain if not set)")
    parser.add_argument("--config", default="config/params_grid_cvae.json", type=str, help="path to params grid")
    parser.add_argument("--overlap_tcga", default=None, type=str, help="overlap patient list to exclude from TCGA training (only used when target_domain=tcga)")
    args = parser.parse_args()

    params_payload, payload_type = core.load_params_grid(args.config)
    if payload_type == "combinations":
        param_list = params_payload
    else:
        keys, values = zip(*params_payload.items())
        param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    domain_cfg = core.TARGET_DOMAIN_CONFIG[args.target_domain]
    resolved_target = args.target or domain_cfg["target_expression"]
    resolved_target_response = args.target_response or domain_cfg["target_response"]
    resolved_target_cancer_ref = args.target_cancer_reference or domain_cfg["target_cancer_reference"]
    source_path = core.DEFAULT_SOURCE_CSV
    overlap_path = args.overlap_tcga if args.target_domain == "tcga" else None

    safemakedirs(args.outfolder)
    all_rows = []
    training_target_path, removed_count = core._prepare_training_target_csv(resolved_target, overlap_path, args.outfolder)
    if overlap_path:
        if removed_count > 0:
            print(f"[TCGA overlap filter] removed {removed_count} rows for training target")
        else:
            print("[TCGA overlap filter] overlap file provided but no rows removed; use original target data")
    else:
        print("[TCGA overlap filter] disabled (no --overlap_tcga provided), use original target data")

    frame_cache = {}
    for param_dict in param_list:
        cache_key = f"{source_path}|{resolved_target}"
        if cache_key not in frame_cache:
            frame_cache[cache_key] = core._load_full_feature_frames(source_path, resolved_target)
        ccle_df_full, tcga_df_full = frame_cache[cache_key]

        sourcedata, targetdata = core._load_labeled_data_patient_aware(
            ccle_path=source_path,
            xena_path=training_target_path,
            batch_size=param_dict.get("batch_size", 128),
            use_class_weight=param_dict.get("use_class_weight", False),
            target_domain=args.target_domain,
            target_cancer_reference_path=resolved_target_cancer_ref,
        )
        exp_name, exp_dir = core._next_experiment_dir(args.outfolder)
        metrics = run_single_experiment(
            sourcedata=sourcedata,
            targetdata=targetdata,
            param=param_dict,
            exp_name=exp_name,
            exp_dir=exp_dir,
            ccle_df_for_latent=ccle_df_full,
            tcga_df_for_latent=tcga_df_full,
        )
        row = {
            "ID": exp_name,
            "NO": "",
            "model_type": "CVAE",
            "pretrain_epochs": param_dict.get("pretrain_num_epochs"),
            "train_epochs": param_dict.get("train_num_epochs"),
            "pretrain_lr": param_dict.get("pretrain_learning_rate"),
            "train_lr": param_dict.get("gan_learning_rate"),
            "dropout": param_dict.get("dropout_rate"),
            "latent_size": param_dict.get("latent_size", 32),
            "encoder_dims": str(param_dict.get("encoder_dims")),
            "lambda_cls": param_dict.get("lambda_cls"),
            "use_class_weight": param_dict.get("use_class_weight", False),
            "FID_AfterGAN": metrics["fid"],
            "MMD_AfterGAN": metrics["mmd"],
            "Wasserstein_AfterGAN": metrics["wasserstein"],
            "best_gan_epoch": metrics["best_gan_epoch"],
            "best_gan_loss": metrics["best_gan_loss"],
            "fid": metrics["fid"],
            "mmd": metrics["mmd"],
            "wasserstein": metrics["wasserstein"],
            "kmeans_k": metrics.get("kmeans_k"),
            "kmeans_ari": metrics.get("kmeans_ari"),
            "kmeans_nmi": metrics.get("kmeans_nmi"),
            "kmeans_silhouette": metrics.get("kmeans_silhouette"),
            "kmeans_calinski_harabasz": metrics.get("kmeans_calinski_harabasz"),
            "kmeans_davies_bouldin": metrics.get("kmeans_davies_bouldin"),
            "result_folder": exp_name,
        }
        core._append_model_select(args.outfolder, row)
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(os.path.join(args.outfolder, "summary_results.csv"), index=False)
    print(
        f"All experiments done. {core.PRETRAIN_MODEL_SELECT_FILENAME} and "
        f"summary_results.csv saved under {args.outfolder}"
    )


if __name__ == "__main__":
    main()
