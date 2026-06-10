# Proposal: Repository-grounded Automated VAEwC Pretrain Optimization with GAN-stage Cancer Prototype InfoNCE

**Project:** `JohnWCT/drug-DCF`  
**Target branch/reference:** latest `main` checked from GitHub during planning  
**Primary model for round 1:** `pretrain_VAEwC.py`  
**GPU constraint:** one A6000 Ada GPU, sequential execution  
**Output intent:** complete executable research plan, not a brainstorming note

---

## 1. Executive Summary

This proposal defines a complete, executable optimization plan for the latest `drug-DCF` repository. The current VAEwC pretraining pipeline already includes several recent stabilizing features: staged `lambda_cls` warm-up, corrected WGAN-GP discriminator scoring, GAN-stage classifier-only preservation steps, configurable generator update interval, and visualization/filtering support. The next optimization should therefore not merely tune `lambda_cls` again. Instead, round 1 introduces a **GAN-stage, source+target, cancer prototype InfoNCE loss** to strengthen deconfounding while preserving tumor-class separability.

The full pipeline will run:

```text
72 VAEwC pretrain runs
→ visualize/filter with config/visualize_vaewc_filter.json
→ Top-10 selection with two lambda_proto=0 controls
→ finetune using config/params_finetune_mini.json only
→ aggregate by Global_TCGA_AUC_mean
→ generate CSV/JSON/Markdown/plot reports
```

Round 1 is intentionally limited to VAEwC full experiments. AEwC and CVAE will not be full-sweep targets in this round. AEwC is a thin wrapper over the VAEwC core and should inherit compatible changes after VAEwC validation; CVAE is an independent script and remains a later synchronization target.

---

## 2. Repository-grounded Starting Point

This plan is based on the current public `main` branch structure and behavior.

### 2.1 Current VAEwC capabilities

`pretrain_VAEwC.py` is the central first-round optimization target. It currently provides:

- `smooth_rampup()` and `get_lambda_cls_eff()` for staged tumor-classifier loss.
- `resolve_gan_training_params()` for resolving GAN-stage parameters such as `gan_gen_update_interval`, `gan_cls_update_every_step`, `gan_cls_learning_rate`, `gan_lambda_cls`, and `gan_gp_weight`.
- WGAN-GP discriminator training using discriminator outputs `d_s = discrim(s)` and `d_t = discrim(t)`.
- A classifier-only GAN step that freezes the shared encoder and updates the tumor classifier on source+target shared latents.
- A generator/encoder step where shared/private encoders and the tumor classifier are updated together.
- Summary rows containing deconfounding metrics and clustering/tumor-preservation metrics.

### 2.2 Current AEwC status

`pretrain_AEwC.py` reuses the VAEwC pipeline by importing `pretrain_VAEwC` as `core`, switching `core.MODEL_BACKBONE = AE`, setting `MODEL_TYPE_NAME = "AE"`, and then calling `core.main()`. Therefore, AEwC does not need a separate full implementation path in round 1; compatibility can be validated with a smoke test after VAEwC changes.

### 2.3 Current CVAE status

`pretrain_CVAE.py` is an independent implementation. It still has older GAN training logic, including a discriminator loss form based on direct latent means rather than the corrected VAEwC WGAN discriminator output pattern. It also does not share the latest VAEwC staged/decoupled GAN implementation. Therefore, CVAE is not included in round 1 full experiments.

### 2.4 Current visualization/filtering flow

`visualize_vaewc_results.py` supports:

- loading `exp_*` folders from a result directory,
- applying `config/visualize_vaewc_filter.json`,
- computing `score_deconfounding`, `score_kmeans`, and `score_total`,
- exporting `aggregated_vaewc_results.csv`,
- exporting `model_select.csv` for downstream finetune,
- selecting Top-K via `--select_top_k`.

`config/visualize_vaewc_filter.json` is enabled and defines hard thresholds for deconfounding and tumor-preservation quality. Current thresholds include:

| Metric | Direction | Threshold |
|---|---|---:|
| `fid` | lower is better | 16.95 |
| `mmd` | lower is better | 0.05 |
| `wasserstein` | lower is better | 0.50 |
| `kmeans_davies_bouldin` | lower is better | 1.50 |
| `kmeans_ari` | higher is better | 0.20 |
| `kmeans_nmi` | higher is better | 0.45 |
| `kmeans_silhouette` | higher is better | 0.15 |
| `kmeans_calinski_harabasz` | higher is better | 370 |

### 2.5 Current finetune flow

`config/params_finetune_mini.json` defines exactly four finetune combinations:

```text
loss_type: bce / focal
classifier dropout: 0.05 / 0.1
hidden_dims: [256, 128]
activation: leaky_relu
use_batch_norm: true
gin_type: dapl
scheduler_flag: true
ftlr: 0.001
```

`step1_finetune_latent_pipeline_All_split.py` consumes `--model_select_path` and runs all parameter combinations from the finetune config for each selected pretrain model.

### 2.6 Current aggregation flow

`aggregate_pretrain_tcga_scores.py` groups finetune rows by `Model_ID`, computes mean/std/count summaries, and sorts primarily by `Global_TCGA_AUC_mean`. The final optimization pipeline will use the same primary downstream ranking metric.

---

## 3. Problem Statement

The current configuration can achieve good tumor classification, but deconfounding remains weaker than desired. The key risk is that global source-target adversarial alignment alone may not align biologically corresponding tumor classes across domains. Stronger deconfounding must therefore be added without washing out tumor class structure.

The desired improvement is:

```text
Better source-target deconfounding
+ retained tumor/cancer-type separability
+ maintained or improved downstream TCGA drug-response performance
```

---

## 4. Main Hypothesis

Adding a **GAN-stage source+target cancer prototype InfoNCE loss** will improve domain deconfounding by explicitly aligning shared latent representations by tumor class, instead of relying only on global WGAN-GP alignment.

The hypothesis is:

> During GAN-stage encoder/generator updates, source+target batch prototypes computed from the shared latent space can encourage same-cancer samples to align across domains while preserving separation between different tumor classes. This should improve deconfounding metrics without sacrificing tumor-preservation metrics or downstream `Global_TCGA_AUC_mean`.

---

## 5. Proposed Method

### 5.1 Current GAN-stage generator/encoder objective

The current generator/encoder step conceptually optimizes:

```text
vae_loss
+ pvae_loss
+ ortho_loss
+ g_loss
+ gan_lambda_cls * cls_loss
```

### 5.2 Proposed round-1 objective

Round 1 adds two explicit, backward-compatible loss controls:

```text
lambda_adv_eff * g_loss
lambda_proto_eff * proto_infonce_loss
```

The proposed loss becomes:

```text
gan_total_loss =
    vae_loss
  + pvae_loss
  + ortho_loss
  + lambda_adv_eff * g_loss
  + gan_lambda_cls * cls_loss
  + lambda_proto_eff * proto_infonce_loss
```

Default behavior must be backward-compatible:

```text
lambda_proto = 0.0 → identical to no-InfoNCE control
lambda_adv = 1.0 → same adversarial scale as current behavior
```

### 5.3 Prototype construction

Use source+target shared latent representations from the same GAN generator/encoder update batch:

```text
z_all = concat(z_source_shared, z_target_shared)
y_all = concat(y_source_cancer, y_target_cancer)
prototype[c] = mean(z_all where y_all == c)
```

This matches the current tumor-classifier label space, which already uses common source/target cancer labels.

### 5.4 InfoNCE formulation

For each valid sample:

```text
logits_i[c] = cosine(normalize(z_i), normalize(prototype[c])) / proto_temperature
proto_infonce_loss = CrossEntropyLoss(logits, y_i)
```

The positive target is the sample's cancer class. Negative classes are all other valid cancer prototypes present in the batch.

### 5.5 Edge-case policy

| Case | Required behavior |
|---|---|
| A class is absent from the batch | Exclude that class from prototype matrix |
| A sample's label has no valid prototype | Exclude that sample from InfoNCE calculation |
| Fewer than two valid classes | Return zero prototype loss and log `proto_valid=false` |
| Non-finite loss or logits | Mark run failed with clear error metadata |
| `lambda_proto=0` | Do not affect total loss; still allow optional metric logging |

### 5.6 Schedule

InfoNCE is active only in GAN stage:

```text
lambda_proto_eff = smooth_rampup(
  gan_epoch,
  proto_start_epoch,
  proto_full_epoch,
  lambda_proto
)
```

---

## 6. Round-1 Sweep Plan

Round 1 runs exactly 72 VAEwC pretrain jobs.

| Parameter | Values |
|---|---|
| `lambda_proto` | `[0, 0.03, 0.1, 0.3]` |
| `proto_temperature` | `[0.1, 0.2, 0.5]` |
| `proto_start_epoch` | `[5, 10, 30]` |
| `proto_full_epoch` | `[30, 50]` |
| `lambda_adv` | `[1.0]` |
| `gan_gen_update_interval` | `[5]` |

Total:

```text
4 × 3 × 3 × 2 × 1 × 1 = 72 pretrain jobs
```

The `lambda_proto=0` group is mandatory and serves as the no-InfoNCE control set.

---

## 7. Execution Strategy

The user has one A6000 Ada GPU. Therefore:

1. Run one job at a time.
2. Use a single-GPU sequential queue.
3. Do not implement multi-GPU scheduling.
4. Do not implement DDP.
5. Use a manifest-based runner so interrupted sweeps can resume.
6. Minimize repeated file reads where practical, but do not introduce a large shared in-memory framework.
7. Retain enough artifacts for monitoring and debugging without keeping all large checkpoints.

---

## 8. Full Pipeline

### Stage 1: Generate configs

Generate 72 configs from a sweep spec and a base config derived from the current good parameter region.

Outputs:

```text
config/generated/vaewc_proto_infonce_round1/*.json
result/optimization_runs/{run_id}/manifests/pretrain_sweep_manifest.csv
```

### Stage 2: Run VAEwC pretrain sweep

Run each generated config through:

```bash
python pretrain_VAEwC.py \
  --config {generated_config_path} \
  --outfolder result/optimization_runs/{run_id}/pretrain \
  --target_domain tcga \
  --overlap_tcga data/TCGA/PMID27354694_DR_OMICS_ad.csv
```

### Stage 3: Visualize and filter

Run:

```bash
python visualize_vaewc_results.py \
  --result_dir result/optimization_runs/{run_id}/pretrain \
  --output_dir result/optimization_runs/{run_id}/selection \
  --filter_config config/visualize_vaewc_filter.json \
  --select_top_k 10
```

The selection wrapper must additionally enforce:

```text
Top-10 = best 8 candidates after filtering/ranking + best 2 lambda_proto=0 controls
```

If two valid no-InfoNCE controls do not pass the filter, the report must explicitly state how many controls are available and why.

### Stage 4: Run downstream finetune mini-grid

Use only:

```text
config/params_finetune_mini.json
```

Expected workload:

```text
10 selected pretrain models × 4 finetune configs = 40 finetune jobs
```

Command pattern:

```bash
python step1_finetune_latent_pipeline_All_split.py \
  --config config/params_finetune_mini.json \
  --model_select_path result/optimization_runs/{run_id}/selection/model_select_top10_with_controls.csv \
  --outfolder result/optimization_runs/{run_id}/finetune \
  --batch_size 2048 \
  --mini_batch_size 512
```

### Stage 5: Aggregate downstream scores

Run:

```bash
python aggregate_pretrain_tcga_scores.py \
  --input result/optimization_runs/{run_id}/finetune/parameter_comparison_tcga_focus.csv \
  --output result/optimization_runs/{run_id}/aggregate/pretrain_tcga_model_summary.csv \
  --top_n 10
```

Primary ranking metric:

```text
Global_TCGA_AUC_mean
```

### Stage 6: Generate reports

Generate:

```text
reports/pretrain_sweep_report.md
reports/pretrain_selection_report.md
reports/finetune_status_report.md
reports/final_selection_report.md
```

Reports must include CSV/JSON/Markdown/plot outputs.

---

## 9. Selection and Ranking Policy

### 9.1 Hard filter

Use `config/visualize_vaewc_filter.json` as the first filter. A run must pass all enabled thresholds before entering normal Top-K ranking.

### 9.2 Ranking

Use the current `visualize_vaewc_results.py` scoring behavior as the base ranking system:

```text
score_total = deconf_weight * score_deconfounding + kmeans_weight * score_kmeans
```

Default CLI weights are:

```text
deconf_weight = 0.7
kmeans_weight = 0.3
```

The report must also display individual metric ranks so the Top-10 is explainable.

### 9.3 Control enforcement

Final selection:

```text
Top-10 = best 8 non-control or mixed candidates + best 2 lambda_proto=0 controls
```

This guarantees direct downstream comparison between InfoNCE and no-InfoNCE settings.

---

## 10. Required Outputs

### 10.1 Per-run pretrain outputs

Do not rename current VAEwC outputs unless necessary. Preserve existing file conventions and add new columns/files as needed.

Current-style outputs to preserve:

```text
params.json
pretrain_loss.csv
pretrain_eval_loss.csv
d_loss.csv
g_loss.csv
tsne_gan_best.png
pretrain_model_select.csv or summary row output if generated by the script
latent .pkl files used by downstream finetune
```

New outputs to add:

```text
run_status.json
resolved_config.json
prototype_metrics.csv
run_report.md
```

New columns in `g_loss.csv`:

```text
lambda_proto_eff
proto_loss
proto_acc
proto_valid_class_count
proto_mean_positive_similarity
proto_mean_negative_similarity
proto_margin
lambda_adv_eff
```

### 10.2 Sweep-level outputs

```text
manifests/pretrain_sweep_manifest.csv
manifests/pretrain_status.csv
selection/aggregated_vaewc_results.csv
selection/model_select.csv
selection/model_select_top10_with_controls.csv
selection/pretrain_selection_report.md
finetune/finetune_status.csv
aggregate/pretrain_tcga_model_summary.csv
reports/final_selection_report.md
```

---

## 11. Artifact Retention Policy

| Run type | Retention |
|---|---|
| Top-10 selected candidates | Keep full artifacts |
| Two selected `lambda_proto=0` controls | Keep full artifacts |
| Non-Top-10 completed runs | Keep config, summary metrics, status, reports; remove large checkpoints/latents after selection |
| Failed runs | Keep config/status/stdout/stderr/partial CSV; delete large checkpoints/latents |

Large artifact deletion must be configurable and must never delete summary CSV, config snapshots, reports, or status files.

---

## 12. Ablation Plan

Round 1 directly supports:

| Ablation | Comparison |
|---|---|
| InfoNCE effect | `lambda_proto=0` vs `lambda_proto>0` |
| InfoNCE strength | `0.03` vs `0.1` vs `0.3` |
| temperature sensitivity | `0.1` vs `0.2` vs `0.5` |
| InfoNCE start timing | `5` vs `10` vs `30` |
| full-weight timing | `30` vs `50` |

---

## 13. Success Criteria

Round 1 is successful if:

1. At least one InfoNCE configuration passes `visualize_vaewc_filter.json`.
2. At least one InfoNCE configuration enters the selected Top-8 non-control group.
3. Selected InfoNCE models improve at least one deconfounding metric relative to selected no-InfoNCE controls.
4. Tumor-preservation metrics remain above the filter thresholds.
5. At least one InfoNCE model matches or improves downstream `Global_TCGA_AUC_mean` compared with selected no-InfoNCE controls.
6. No systematic instability appears in prototype loss, GAN loss, or downstream finetune.

---

## 14. Failure Interpretation

| Failure pattern | Interpretation | Next action |
|---|---|---|
| Better deconfounding but worse tumor preservation | InfoNCE too strong or too early | lower `lambda_proto`, delay `proto_start_epoch` |
| Tumor preservation stable but no deconfounding gain | InfoNCE too weak or too smooth | increase `lambda_proto`, reduce `proto_temperature` |
| Prototype metrics collapse | batch prototypes unstable | test EMA prototype in next round |
| Downstream AUC worsens despite good pretrain metrics | pretrain selection score incomplete | adjust ranking/reporting, add downstream-aware criteria |
| No InfoNCE model beats controls | formulation ineffective | test class-wise MMD or EMA prototype ablation |

---

## 15. Future Extensions

Not included in round 1:

1. EMA prototype InfoNCE.
2. Memory bank contrastive learning.
3. TransDRP-like sigmoid dot-product InfoMax.
4. Class-wise MMD loss.
5. AEwC full sweep.
6. CVAE-specific conditional prototype InfoNCE.
7. VAE reparameterization ablation.
8. Multi-seed downstream robustness study.
9. Larger finetune grid beyond `params_finetune_mini.json`.

---

## 16. Final Decisions

| Topic | Final setting |
|---|---|
| Source repository | latest `JohnWCT/drug-DCF` `main` checked during planning |
| First full model | VAEwC only |
| First optimization method | GAN-stage source+target cancer prototype InfoNCE |
| InfoNCE formulation | prototype cosine logits + cross entropy |
| TransDRP-like InfoMax | excluded |
| Prototype update | batch-based |
| EMA | excluded from round 1 |
| Sweep size | 72 VAEwC pretrain runs |
| GPU strategy | one A6000 Ada, sequential queue |
| Top-K | Top-10 |
| Control policy | Top-10 includes two `lambda_proto=0` controls |
| Finetune grid | `config/params_finetune_mini.json` only |
| Finetune workload | 40 jobs |
| Selection filter | `config/visualize_vaewc_filter.json` |
| Final downstream metric | `Global_TCGA_AUC_mean` |
| Report formats | CSV + JSON + Markdown + plots |
