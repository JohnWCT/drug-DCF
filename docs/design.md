# drug-DCF VAEwC / GAN Alignment Optimization: System Architecture Design

## 1. Scope

This design document describes the architecture changes for the `pretrain_VAEwC.py` training pipeline to improve the balance between:

1. **Deconfounding / domain alignment** between source and target latent distributions.
2. **Tumor class preservation** in the shared latent representation.

The confirmed implementation scope is limited to four changes:

| Priority | Change | Purpose |
|---:|---|---|
| 0 | Fix WGAN discriminator loss | Ensure the discriminator actually learns from critic outputs |
| 1 | Staged training: VAE → VAE+classifier → GAN+classifier | Reduce early loss interference |
| 2 | `lambda_cls` warm-up / schedule | Prevent classification loss from dominating latent learning too early |
| 3 | Decouple GAN generator update from classifier update | Improve adversarial stability while preserving tumor class structure |

The design intentionally avoids adding prototype learning, InfoNCE, conditional alignment, or dynamic loss balancing in this iteration.

---

## 2. Current System Summary

The current pipeline performs labeled pretraining and GAN alignment using the following components:

| Component | Responsibility |
|---|---|
| Shared VAE encoder / decoder | Learns domain-shared representation and reconstruction |
| Source private VAE | Learns source-specific private representation |
| Target private VAE | Learns target-specific private representation |
| Tumor classifier | Predicts cancer class from shared latent representation |
| Discriminator / critic | Distinguishes source and target latent representations |
| Training loop | Coordinates pretraining, GAN training, model selection, logging, and saving |

The current loss structure couples several objectives in the same update path:

```python
loss = o_loss + vae_loss + p_vae_loss + lambda_cls * cls_loss
```

and in GAN generator / autoencoder training:

```python
loss = o_loss + g_loss + vae_loss + pvae_loss + lambda_cls * cls_loss
```

This makes the model sensitive to a single fixed `lambda_cls`, because reconstruction, disentanglement, domain alignment, and tumor-class preservation are optimized simultaneously.

---

## 3. Architecture Principles

The revised architecture follows these principles:

1. **Low coupling between training objectives**  
   Each training objective should have a clearly isolated computation path and update schedule.

2. **Explicit stage control**  
   The system should decide when a loss is active based on epoch and stage configuration, not hard-coded assumptions.

3. **Independent testability**  
   Schedule functions, discriminator loss, classifier-only update, and GAN update should be independently unit-testable.

4. **Backward-compatible configuration**  
   Existing config files should continue to run. Missing new fields must fall back to safe defaults.

5. **Observable training behavior**  
   Logs must include effective loss weights and stage-specific losses so failures can be diagnosed from CSV outputs.

---

## 4. Proposed Module Decomposition

Although the current implementation is mostly contained in `pretrain_VAEwC.py`, the new design separates responsibilities logically. The first implementation can keep these functions in the same file, but the module boundaries should be respected so they can later be split into files.

### 4.1 Configuration Module

**Suggested future file:** `training_config.py`  
**Initial implementation:** helper functions inside `pretrain_VAEwC.py`

Responsibilities:

- Read optional stage and schedule parameters from `param`.
- Provide default values for backward compatibility.
- Validate obvious invalid values.

Main config fields:

| Field | Type | Default | Meaning |
|---|---:|---:|---|
| `lambda_cls` | float | `1.0` | Maximum classifier loss weight during pretraining |
| `cls_start_epoch` | int | `1` | Epoch where classification loss becomes active |
| `cls_full_epoch` | int | `cls_start_epoch` | Epoch where classification loss reaches full strength |
| `gan_gen_update_interval` | int | `5` | Update generator / encoder every N batches |
| `gan_cls_update_every_step` | bool | `true` | Whether classifier-only update runs every GAN batch |
| `gan_cls_learning_rate` | float | `gan_learning_rate` | Learning rate for classifier-only optimizer |
| `gan_lambda_cls` | float | `lambda_cls` | Classifier loss weight during GAN generator step |
| `gan_gp_weight` | float | `10.0` | Gradient penalty coefficient |

Design choice:

- Use permissive defaults rather than strict config requirements.
- This avoids breaking old experiments.

---

### 4.2 Schedule Module

**Suggested future file:** `loss_schedules.py`  
**Initial implementation:** functions inside `pretrain_VAEwC.py`

Responsibilities:

- Compute effective loss weights by epoch.
- Keep schedule logic independent from model and optimizer code.

Public functions:

```python
def smooth_rampup(epoch, start_epoch, end_epoch, max_value):
    ...


def get_lambda_cls_eff(epoch, param):
    ...
```

Behavior:

| Epoch range | Effective classifier weight |
|---|---:|
| `epoch < cls_start_epoch` | `0.0` |
| `cls_start_epoch <= epoch < cls_full_epoch` | Smoothly increases from `0.0` to `lambda_cls` |
| `epoch >= cls_full_epoch` | `lambda_cls` |

Rationale:

- VAE and private/shared disentanglement are allowed to stabilize first.
- Classification pressure is gradually introduced.
- The schedule can be unit-tested without requiring PyTorch model execution.

---

### 4.3 Data Module

**Existing function:** `_load_labeled_data_patient_aware()`

Responsibilities:

- Load source and target expression matrices.
- Align feature columns.
- Map source and target samples to common cancer labels.
- Create train/test tensors and dataloaders.
- Optionally compute class weights.

No architectural change is required for this iteration.

Important interface contract:

```python
sourcedata = (
    source_loader,
    source_test_tensor,
    source_test_label_tensor,
    source_weights,
    mapping_int2str,
)

targetdata = (
    target_loader,
    target_test_tensor,
    target_test_label_tensor,
    target_weights,
    mapping_int2str,
)
```

If `use_class_weight=False`, the weight objects are omitted as in the current implementation.

Testing focus:

- Source and target label maps must remain identical.
- Class weights must stay on the correct device.
- Dataloader batches must return `(data, label)` pairs.

---

### 4.4 Model Module

**Existing classes / imports:**

| Model | Source |
|---|---|
| `VAE` | `tools.model_opt` |
| `Discriminator` | `tools.model_opt` |
| `PrimaryClassifier` | `pretrain_VAEwC.py` |

Responsibilities:

- Define shared encoder/decoder, private encoders/decoders, discriminator, and classifier.
- Keep model definitions independent from training-stage decisions.

No major model architecture change is required.

Design constraint:

- `PrimaryClassifier` should only depend on latent dimension and class count.
- It should not know whether it is being trained in pretrain or GAN stage.

---

### 4.5 Loss Module

**Suggested future file:** `training_losses.py`  
**Initial implementation:** keep loss computation inside training step functions, but with explicit boundaries.

Losses:

| Loss | Symbol | Used in pretrain | Used in GAN stage |
|---|---|---:|---:|
| Shared VAE reconstruction/KL | `vae_loss` | Yes | Yes |
| Private VAE reconstruction/KL | `pvae_loss` | Yes | Yes |
| Orthogonality loss | `o_loss` | Yes | Yes |
| Tumor classifier loss | `cls_loss` | Scheduled | Yes / scheduled by GAN config |
| WGAN generator loss | `g_loss` | No | Yes |
| WGAN discriminator loss | `d_loss` | No | Yes |
| Gradient penalty | `g_p` | No | Yes |

Key correction:

The discriminator loss must be computed from discriminator outputs, not raw latent means.

```python
d_s = discrim(s)
d_t = discrim(t)
d_loss = torch.mean(d_t) - torch.mean(d_s)
```

where:

```python
s = torch.cat((zs, pzs), dim=1)
t = torch.cat((zt, pzt), dim=1)
```

Design choice:

- The discriminator step must use `torch.no_grad()` for encoder outputs so only the discriminator is updated.
- This makes the discriminator update independent from encoder training.

---

### 4.6 Pretraining Module

**Existing responsibility:** pretrain shared/private VAEs and classifier.

Revised responsibility:

- Perform staged VAE/classifier training using `lambda_cls_eff`.
- Keep pretraining independent from GAN discriminator logic.

Pretraining stages:

| Stage | Epoch condition | Active losses |
|---|---|---|
| Stage A: VAE-only | `epoch < cls_start_epoch` | `vae_loss + pvae_loss + o_loss` |
| Stage B: VAE + classifier warm-up | `cls_start_epoch <= epoch < cls_full_epoch` | `vae_loss + pvae_loss + o_loss + lambda_cls_eff * cls_loss` |
| Stage C: VAE + full classifier | `epoch >= cls_full_epoch` | `vae_loss + pvae_loss + o_loss + lambda_cls * cls_loss` |

Revised pretrain loss:

```python
lambda_cls_eff = get_lambda_cls_eff(epoch + 1, param)
loss = o_loss + vae_loss + p_vae_loss + lambda_cls_eff * cls_loss
```

Logging requirements:

| CSV column | Meaning |
|---|---|
| `ortholoss` | Orthogonality loss |
| `pVAE_loss` | Private VAE loss |
| `VAE_loss` | Shared VAE loss |
| `cls_loss` | Raw classifier loss |
| `lambda_cls_eff` | Effective classifier loss weight |

Testing focus:

- For epochs before `cls_start_epoch`, classifier loss should not affect total loss.
- At and after `cls_full_epoch`, `lambda_cls_eff == lambda_cls`.
- Existing configs without schedule fields should behave like the old fixed `lambda_cls` setup.

---

### 4.7 GAN Discriminator Training Module

**Existing function:** `train_discrim()`

Revised responsibility:

- Train only the discriminator / critic.
- Use fixed encoder outputs detached from graph.
- Compute WGAN-GP loss correctly.

Revised update path:

```text
source batch, target batch
        ↓
shared/private encoders in eval mode under torch.no_grad()
        ↓
construct source latent and target latent
        ↓
discriminator(source latent), discriminator(target latent)
        ↓
WGAN discriminator loss + gradient penalty
        ↓
update discriminator only
```

Loss:

```python
d_loss = torch.mean(d_t) - torch.mean(d_s)
total_loss = d_loss + gan_gp_weight * gradient_penalty
```

Returned logs:

| Key | Meaning |
|---|---|
| `discrim_loss` | Raw WGAN discriminator loss |
| `g_p` | Gradient penalty |
| `discrim_total_loss` | Discriminator loss plus gradient penalty |
| `d_source_score` | Mean critic score for source latent |
| `d_target_score` | Mean critic score for target latent |

Testing focus:

- Discriminator parameters should change after this step.
- Encoder and classifier parameters should not change.
- `discrim_loss` must depend on `discrim(s)` and `discrim(t)`.

---

### 4.8 GAN Classifier-Only Training Module

**New function:** `train_classifier_step()`

Responsibility:

- Update only the tumor classifier during GAN stage.
- Keep classifier adapted to current shared latent distribution.
- Avoid updating shared encoder every batch, which could destabilize adversarial alignment.

Update path:

```text
source batch, target batch
        ↓
shared encoder in eval mode under torch.no_grad()
        ↓
classifier(source shared latent), classifier(target shared latent)
        ↓
classification loss
        ↓
update classifier only
```

Loss:

```python
cls_loss = CE(classifier(ccle_z), ccle_labels) + CE(classifier(tcga_z), tcga_labels)
```

If class weights are enabled:

```python
cls_loss = weighted_CE_source + weighted_CE_target
```

Returned logs:

| Key | Meaning |
|---|---|
| `cls_only_loss` | Classifier-only update loss during GAN training |

Design choice:

- The shared encoder is frozen in this function by using `torch.no_grad()`.
- Encoder receives classification gradients only during generator / autoencoder updates.

Testing focus:

- Classifier parameters should change.
- Shared/private encoder parameters should not change.
- Function should work with and without class weights.

---

### 4.9 GAN Generator / Autoencoder Training Module

**Existing function:** `train_d_ae()`

Revised responsibility:

- Update shared VAE, private VAEs, and classifier during scheduled generator steps.
- Include adversarial generator loss.
- Use `gan_lambda_cls` or `lambda_cls_eff` as the effective classifier weight.

Update frequency:

```python
if (step + 1) % gan_gen_update_interval == 0:
    train_d_ae(...)
```

Revised loss:

```python
loss = (
    o_loss
    + g_loss
    + vae_loss
    + pvae_loss
    + gan_lambda_cls_eff * cls_loss
)
```

Returned logs:

| Key | Meaning |
|---|---|
| `ortho_loss` | Orthogonality loss |
| `pvae_loss` | Private VAE loss |
| `gen_loss` | WGAN generator loss |
| `vae_loss` | Shared VAE loss |
| `cls_loss` | Classifier loss during generator step |
| `lambda_cls_eff` | Effective classifier loss weight |

Design choice:

- Generator/encoder update remains less frequent than discriminator update.
- Classifier-only update can run every batch to preserve tumor class decision boundaries.

Testing focus:

- Shared encoder and private encoders should change only on scheduled generator steps.
- Classifier should change on both classifier-only and generator steps, if both are enabled.
- Discriminator should not be updated by this step.

---

### 4.10 Training Orchestrator Module

**Existing responsibility:** main training loop inside `pretrain_VAEwC.py`

Revised responsibility:

- Coordinate pretraining and GAN stages.
- Initialize separate optimizers and schedulers.
- Route each batch to the correct update function.
- Aggregate logs.

Revised GAN loop:

```text
for each GAN epoch:
    for each source batch:
        get target batch

        1. update discriminator every step
        2. optionally update classifier every step
        3. update generator / encoder every N steps
```

Optimizer separation:

| Optimizer | Parameters | Step frequency |
|---|---|---:|
| `discrim_optimizer` | Discriminator only | Every GAN batch |
| `classifier_optimizer` | Classifier only | Every GAN batch if enabled |
| `d_ae_optimizer` | Shared VAE + private VAEs + classifier | Every `gan_gen_update_interval` batches |

Reason for separate optimizers:

- Avoid accidental cross-module updates.
- Make unit tests and debugging easier.
- Allow classifier learning rate to be tuned independently from adversarial generator learning rate.

---

### 4.11 Logging and Visualization Module

**Existing functions:** `_plot_pretrain_curves()`, `_plot_gan_curves()`

Revised logging requirements:

Pretrain CSV should include:

```text
epoch, ortholoss, pVAE_loss, VAE_loss, cls_loss, lambda_cls_eff
```

GAN discriminator CSV should include:

```text
epoch, discrim_loss, discrim_total_loss, g_p, d_source_score, d_target_score
```

GAN generator CSV should include:

```text
epoch, gen_loss, cls_loss, cls_only_loss, lambda_cls_eff, gan_gen_update_interval
```

Visualization updates:

- Plot `lambda_cls_eff` in pretraining curves.
- Plot `cls_only_loss` in GAN curves.
- Plot `discrim_total_loss` and `g_p` in discriminator curves.

Rationale:

- If classifier loss fails during GAN stage, `cls_only_loss` will expose it.
- If WGAN-GP is unstable, `g_p` and `discrim_total_loss` will expose it.
- If schedule is misconfigured, `lambda_cls_eff` will expose it.

---

## 5. Detailed Data Flow

### 5.1 Pretraining Data Flow

```text
source batch + target batch
        ↓
shared VAE encodes source and target
source private VAE encodes source
target private VAE encodes target
        ↓
compute VAE / private VAE reconstruction losses
compute orthogonality loss
compute classifier loss from shared latent
        ↓
get lambda_cls_eff from schedule
        ↓
combined pretrain loss
        ↓
update shared VAE + private VAEs + classifier
```

### 5.2 GAN Discriminator Data Flow

```text
source batch + target batch
        ↓
encoders produce detached latents
        ↓
source latent = concat(source shared, source private)
target latent = concat(target shared, target private)
        ↓
critic scores source and target
        ↓
WGAN discriminator loss + gradient penalty
        ↓
update discriminator only
```

### 5.3 GAN Classifier-Only Data Flow

```text
source batch + target batch
        ↓
shared encoder produces detached shared latents
        ↓
classifier predicts tumor classes
        ↓
classification loss
        ↓
update classifier only
```

### 5.4 GAN Generator / Encoder Data Flow

```text
source batch + target batch
        ↓
shared/private encoders produce trainable latents
        ↓
compute reconstruction, orthogonality, classifier, and generator losses
        ↓
combined GAN generator / autoencoder loss
        ↓
update shared VAE + private VAEs + classifier
```

---

## 6. Configuration Design

Recommended first-run configuration:

```json
{
  "lambda_cls": 1.0,
  "cls_start_epoch": 20,
  "cls_full_epoch": 60,
  "gan_gen_update_interval": 5,
  "gan_cls_update_every_step": true,
  "gan_cls_learning_rate": 0.0001,
  "gan_lambda_cls": 1.0,
  "gan_gp_weight": 10.0
}
```

If `pretrain_num_epochs` differs from 100, use proportional settings:

| `pretrain_num_epochs` | `cls_start_epoch` | `cls_full_epoch` |
|---:|---:|---:|
| 50 | 10 | 30 |
| 100 | 20 | 60 |
| 200 | 40 | 120 |

Backward compatibility rules:

| Missing field | Fallback behavior |
|---|---|
| `cls_start_epoch` | Classification active from epoch 1 |
| `cls_full_epoch` | No warm-up; full `lambda_cls` immediately |
| `gan_gen_update_interval` | Use 5 |
| `gan_cls_update_every_step` | Use true |
| `gan_cls_learning_rate` | Use `gan_learning_rate` |
| `gan_lambda_cls` | Use `lambda_cls` |
| `gan_gp_weight` | Use 10.0 |

---

## 7. Technical Trade-Offs and Decisions

### 7.1 Keep functions in one file first vs split files now

Decision: **keep functions in `pretrain_VAEwC.py` for the first implementation.**

Reason:

- The current project already has a working single-script training pipeline.
- Moving files now increases risk of import-path errors.
- Logical module boundaries are documented here so later refactoring remains straightforward.

### 7.2 Freeze shared encoder in classifier-only GAN step

Decision: **classifier-only update should not update shared encoder.**

Reason:

- The goal of the classifier-only step is to keep the decision boundary adapted.
- Updating the encoder every batch through classifier loss could reintroduce the same loss conflict the staged design is trying to reduce.

### 7.3 Use smooth warm-up instead of linear warm-up

Decision: **use smooth cubic ramp-up.**

Reason:

- It avoids abrupt slope changes at the beginning and end of the schedule.
- It is simple and deterministic.
- It does not require external dependencies.

### 7.4 Separate classifier optimizer during GAN stage

Decision: **add a separate classifier optimizer.**

Reason:

- Classifier can be updated every step without forcing encoder/generator updates every step.
- Classifier learning rate can be tuned independently.
- This reduces coupling between adversarial alignment and tumor-class prediction.

---

## 8. Unit Test Plan

The implementation should be validated with small synthetic tensors before full training.

### 8.1 Schedule Tests

| Test | Expected result |
|---|---|
| `epoch < cls_start_epoch` | `get_lambda_cls_eff() == 0.0` |
| `epoch == cls_full_epoch` | `get_lambda_cls_eff() == lambda_cls` |
| missing schedule fields | old behavior preserved |
| `lambda_cls == 0` | all effective values are 0 |

### 8.2 Discriminator Step Tests

| Test | Expected result |
|---|---|
| Run `train_discrim()` once | discriminator parameters change |
| Run `train_discrim()` once | encoder parameters do not change |
| Mock discriminator outputs | `d_loss` uses critic outputs, not latent means |

### 8.3 Classifier-Only Step Tests

| Test | Expected result |
|---|---|
| Run `train_classifier_step()` once | classifier parameters change |
| Run `train_classifier_step()` once | shared encoder parameters do not change |
| Use class weights | function runs without shape/device errors |

### 8.4 GAN Loop Tests

| Test | Expected result |
|---|---|
| `gan_gen_update_interval = 5` | generator step called every 5 batches |
| `gan_cls_update_every_step = true` | classifier-only step called every batch |
| `gan_cls_update_every_step = false` | classifier-only step skipped |

---

## 9. Integration Test Plan

Run four ablation experiments:

| Experiment | Fix WGAN | Staged pretrain | `lambda_cls` warm-up | Classifier every GAN step |
|---|---:|---:|---:|---:|
| A | No | No | No | No |
| B | Yes | No | No | No |
| C | Yes | Yes | Yes | No |
| D | Yes | Yes | Yes | Yes |

Evaluate:

| Metric | Desired direction |
|---|---|
| MMD | Down |
| Wasserstein distance | Down |
| Classifier macro-F1 | Stable or up |
| Balanced accuracy | Stable or up |
| KMeans ARI | Stable or up |
| KMeans NMI | Stable or up |
| t-SNE / UMAP visualization | Source-target closer, cancer classes not collapsed |

Failure patterns:

| Symptom | Likely cause | Suggested adjustment |
|---|---|---|
| MMD improves but ARI/NMI collapses | Adversarial alignment too strong | Increase `gan_lambda_cls`, increase classifier updates, or increase `gan_gen_update_interval` |
| Classifier loss never decreases | Classifier LR too low or labels misaligned | Increase `gan_cls_learning_rate`, verify label mapping |
| Discriminator loss unstable | GP weight or LR issue | Tune `gan_gp_weight`, lower discriminator LR |
| Pretraining classifier overpowers VAE | Warm-up starts too early | Increase `cls_start_epoch` or reduce `lambda_cls` |

---

## 10. Implementation Checklist

### Phase 1: Minimal safe changes

- [ ] Add `smooth_rampup()`.
- [ ] Add `get_lambda_cls_eff()`.
- [ ] Replace discriminator loss with critic-output WGAN loss.
- [ ] Add `lambda_cls_eff` to pretrain loss.
- [ ] Log `lambda_cls_eff` in pretrain train/eval CSV.

### Phase 2: GAN decoupling

- [ ] Add `train_classifier_step()`.
- [ ] Add `classifier_optimizer`.
- [ ] Add `classifier_scheduler`.
- [ ] Add `gan_gen_update_interval` config.
- [ ] Add `gan_cls_update_every_step` config.
- [ ] Update GAN loop to call discriminator, classifier-only, and generator steps separately.
- [ ] Log `cls_only_loss`.

### Phase 3: Observability

- [ ] Add `discrim_total_loss` to discriminator logs.
- [ ] Add `d_source_score` and `d_target_score` to discriminator logs.
- [ ] Plot `lambda_cls_eff` in pretrain curves.
- [ ] Plot `cls_only_loss` and `g_p` in GAN curves.

### Phase 4: Validation

- [ ] Run synthetic unit tests.
- [ ] Run one short smoke test with 2–3 epochs.
- [ ] Run ablation experiments A–D.
- [ ] Compare MMD/Wasserstein with ARI/NMI and classifier macro-F1.

---

## 11. Expected Final Architecture

```text
                 ┌────────────────────┐
                 │ Config / Parameters │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Schedule Functions  │
                 │ lambda_cls_eff      │
                 └─────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌────────────────┐  ┌────────────────────┐
│ Data Loaders  │  │ Model Factory   │  │ Logging / Plotting  │
└───────┬───────┘  └───────┬────────┘  └─────────▲──────────┘
        │                  │                     │
        ▼                  ▼                     │
┌──────────────────────────────────────┐         │
│ Pretraining Loop                     │─────────┘
│ VAE-only → VAE+cls warm-up → full cls │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ GAN Training Loop                               │
│                                                 │
│  every batch:                                   │
│    1. train_discrim()                           │
│    2. train_classifier_step() optional          │
│                                                 │
│  every N batches:                               │
│    3. train_d_ae()                              │
└─────────────────────────────────────────────────┘
```

---

## 12. Non-Goals for This Iteration

The following are intentionally excluded from this implementation round:

- Cancer-category prototype learning.
- InfoNCE / supervised contrastive loss.
- Conditional discriminator.
- Class-wise MMD / CORAL.
- GradNorm or uncertainty-based dynamic loss balancing.
- Major project file restructuring.
- Replacing the existing VAE or discriminator architecture.

These can be added after the current training-stability changes are validated.
