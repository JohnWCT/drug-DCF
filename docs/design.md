# Design: Executable System Architecture for VAEwC Prototype InfoNCE Optimization

**Project:** `JohnWCT/drug-DCF`  
**Input plan:** `docs/proposal.md`  
**Target implementation:** latest repository `main` checked during planning  
**First full model:** VAEwC  
**GPU target:** one A6000 Ada GPU  
**Design goal:** low-coupling, executable, testable modules

---

## 1. System Goal

Build a repository-compatible optimization system that automates:

```text
config generation
→ VAEwC pretrain sweep
→ visualization/filtering
→ Top-10 selection with controls
→ finetune mini-grid
→ downstream aggregation
→ final reports
```

The first round adds GAN-stage source+target cancer prototype InfoNCE to `pretrain_VAEwC.py` while preserving current repository behavior and output conventions wherever possible.

---

## 2. Repository Compatibility Constraints

The design must start from the latest repository state, not a hypothetical refactor.

### 2.1 `pretrain_VAEwC.py`

Use `pretrain_VAEwC.py` as the primary integration target. It already has:

- staged classifier warm-up via `smooth_rampup()` and `get_lambda_cls_eff()`,
- `resolve_gan_training_params()` for GAN parameters,
- corrected WGAN-GP discriminator score usage,
- `train_classifier_step()` for classifier-only GAN updates,
- `train_d_ae()` for generator/encoder updates,
- existing CSV logging to `pretrain_loss.csv`, `pretrain_eval_loss.csv`, `d_loss.csv`, and `g_loss.csv`,
- existing latent and t-SNE output support.

### 2.2 `pretrain_AEwC.py`

AEwC imports `pretrain_VAEwC` as `core`, switches `core.MODEL_BACKBONE = AE`, sets `MODEL_TYPE_NAME = "AE"`, and calls `core.main()`. Therefore, VAEwC changes that remain backbone-compatible should also be compatible with AEwC. Round 1 still runs only VAEwC full sweeps; AEwC gets smoke tests only.

### 2.3 `pretrain_CVAE.py`

CVAE is an independent implementation and currently does not share the latest VAEwC staged/decoupled GAN training structure. It is not included in round 1 full implementation. CVAE synchronization is a future design task.

### 2.4 `visualize_vaewc_results.py`

Use this existing script for filtering, scoring, HTML/CSV report generation, and `model_select.csv` export. The wrapper may add control-aware Top-10 enforcement, but should not replace the script.

### 2.5 `params_finetune_mini.json`

Use only the four configured finetune combinations from `config/params_finetune_mini.json`.

### 2.6 `aggregate_pretrain_tcga_scores.py`

Use this existing script for downstream aggregation. The primary sorting metric is `Global_TCGA_AUC_mean`.

---

## 3. Architectural Principles

1. **Low coupling:** modules communicate through files, not shared in-memory state.
2. **Minimal invasive changes:** preserve existing script entry points and output filenames.
3. **Backwards compatibility:** `lambda_proto=0` must reproduce no-InfoNCE behavior.
4. **Testability:** loss functions, config generation, runner status, and artifact retention must be unit-testable.
5. **Single-GPU efficiency:** sequential queue for one A6000 Ada; no multi-GPU or DDP.
6. **Resume support:** every long-running stage writes manifest status.
7. **Monitoring:** every stage emits CSV/JSON/Markdown/plot artifacts.
8. **No premature CVAE refactor:** CVAE follows after VAEwC validation.

---

## 4. Target Directory Layout

```text
drug-DCF/
  docs/
    proposal.md
    design.md

  config/
    params_finetune_mini.json
    visualize_vaewc_filter.json
    pretrain_sweeps/
      vaewc_proto_infonce_round1.json
    generated/
      vaewc_proto_infonce_round1/
        exp_proto_000.json
        exp_proto_001.json
        ...

  tools/
    proto_infonce.py
    pretrain_proto_metrics.py
    optimization_config_generator.py
    optimization_runner.py
    optimization_selection.py
    optimization_report.py
    artifact_retention.py

  result/
    optimization_runs/
      {run_id}/
        manifests/
        pretrain/
        selection/
        finetune/
        aggregate/
        reports/
```

---

## 5. Module Overview

| Module | File | Responsibility | Coupling boundary |
|---|---|---|---|
| Prototype InfoNCE loss | `tools/proto_infonce.py` | Compute batch prototype InfoNCE and metrics | pure PyTorch tensors |
| Prototype diagnostics | `tools/pretrain_proto_metrics.py` | Compute report-only global prototype metrics | latent files + labels |
| VAEwC integration | `pretrain_VAEwC.py` | Add loss, schedule, logging | config keys + existing train loop |
| Sweep generator | `tools/optimization_config_generator.py` | Generate 72 configs and manifest | JSON in, JSON/CSV out |
| Sequential runner | `tools/optimization_runner.py` | Run pending jobs one at a time | manifest-driven subprocesses |
| Selection wrapper | `tools/optimization_selection.py` | Enforce Top-10 with two controls | wraps `visualize_vaewc_results.py` outputs |
| Finetune dispatcher | `tools/optimization_runner.py` or separate helper | Run Top-10 × 4 finetune | model_select path + config |
| Aggregation wrapper | runner helper | Call `aggregate_pretrain_tcga_scores.py` | finetune CSV in, aggregate CSV out |
| Artifact retention | `tools/artifact_retention.py` | Remove large non-selected artifacts | filesystem patterns |
| Report generator | `tools/optimization_report.py` | Markdown/CSV/JSON summaries | reads manifests + CSVs |

---

## 6. Data Flow

```text
config/pretrain_sweeps/vaewc_proto_infonce_round1.json
  ↓
optimization_config_generator.py
  ↓
config/generated/vaewc_proto_infonce_round1/*.json
manifests/pretrain_sweep_manifest.csv
  ↓
optimization_runner.py pretrain stage
  ↓
result/optimization_runs/{run_id}/pretrain/exp_*/
  ↓
visualize_vaewc_results.py + optimization_selection.py
  ↓
selection/model_select_top10_with_controls.csv
  ↓
step1_finetune_latent_pipeline_All_split.py
  ↓
finetune/parameter_comparison_tcga_focus.csv
  ↓
aggregate_pretrain_tcga_scores.py
  ↓
aggregate/pretrain_tcga_model_summary.csv
  ↓
optimization_report.py
```

---

## 7. Config Generator Module

### 7.1 Input file

`config/pretrain_sweeps/vaewc_proto_infonce_round1.json`

```json
{
  "base_config": "config/params_proto_base_vaewc.json",
  "output_config_dir": "config/generated/vaewc_proto_infonce_round1",
  "run_id": "vaewc_proto_infonce_round1",
  "sweep": {
    "lambda_proto": [0, 0.03, 0.1, 0.3],
    "proto_temperature": [0.1, 0.2, 0.5],
    "proto_start_epoch": [5, 10, 30],
    "proto_full_epoch": [30, 50],
    "lambda_adv": [1.0],
    "gan_gen_update_interval": [5]
  }
}
```

### 7.2 Output manifest

`result/optimization_runs/{run_id}/manifests/pretrain_sweep_manifest.csv`

Required columns:

| Column | Meaning |
|---|---|
| `job_id` | generated job name |
| `config_path` | generated JSON config |
| `lambda_proto` | InfoNCE weight |
| `proto_temperature` | temperature |
| `proto_start_epoch` | GAN epoch start |
| `proto_full_epoch` | GAN epoch full weight |
| `lambda_adv` | adversarial scale |
| `gan_gen_update_interval` | generator update interval |
| `status` | pending/running/success/failed/skipped |
| `result_dir` | expected output folder |
| `start_time` | runtime metadata |
| `end_time` | runtime metadata |
| `error_message` | failure reason if any |

### 7.3 Unit tests

1. Produces exactly 72 rows.
2. Includes all `lambda_proto=0` controls.
3. Preserves base config values.
4. Writes valid JSON configs.
5. Re-running generator does not overwrite successful manifest rows unless `--force` is passed.

---

## 8. Prototype InfoNCE Module

File:

```text
tools/proto_infonce.py
```

### 8.1 Public API

```python
def compute_batch_prototype_infonce(
    z_source,
    y_source,
    z_target,
    y_target,
    num_classes,
    temperature=0.2,
    min_samples_per_class=1,
):
    """
    Returns:
        loss: torch.Tensor scalar
        metrics: dict[str, float | int | bool]
    """
```

### 8.2 Responsibilities

1. Concatenate source and target latents.
2. Concatenate source and target labels.
3. Build prototypes for valid classes present in the batch.
4. Normalize samples and prototypes.
5. Compute cosine logits divided by temperature.
6. Map class labels to compact valid-prototype indices.
7. Compute cross entropy.
8. Return metrics.

### 8.3 Returned metrics

| Metric | Type |
|---|---|
| `proto_loss` | float |
| `proto_acc` | float |
| `proto_valid` | bool |
| `proto_valid_class_count` | int |
| `proto_valid_sample_count` | int |
| `proto_mean_positive_similarity` | float |
| `proto_mean_negative_similarity` | float |
| `proto_margin` | float |

### 8.4 Edge cases

| Case | Behavior |
|---|---|
| fewer than 2 valid classes | zero loss attached to graph, `proto_valid=false` |
| sample label absent from prototypes | exclude sample |
| invalid temperature | raise `ValueError` |
| NaN/Inf logits | raise `FloatingPointError` |

### 8.5 Unit tests

1. Normal multi-class batch.
2. Missing class in batch.
3. Only one valid class.
4. Temperature changes logit scale.
5. Backward pass updates `z_source` and `z_target`.
6. `lambda_proto=0` total loss remains equal to baseline loss.

---

## 9. VAEwC Integration

File:

```text
pretrain_VAEwC.py
```

### 9.1 New config keys

| Key | Default | Meaning |
|---|---:|---|
| `lambda_proto` | `0.0` | InfoNCE loss weight |
| `proto_temperature` | `0.2` | InfoNCE temperature |
| `proto_start_epoch` | `1` | GAN epoch where InfoNCE starts |
| `proto_full_epoch` | `1` | GAN epoch where full weight is reached |
| `lambda_adv` | `1.0` | multiplier on generator adversarial loss |
| `proto_min_samples_per_class` | `1` | valid prototype threshold |

### 9.2 New helper functions

```python
def get_lambda_proto_eff(gan_epoch: int, param: dict) -> float:
    return smooth_rampup(
        gan_epoch,
        int(param.get("proto_start_epoch", 1)),
        int(param.get("proto_full_epoch", 1)),
        float(param.get("lambda_proto", 0.0)),
    )


def resolve_proto_training_params(param: dict) -> dict:
    return {
        "lambda_proto": float(param.get("lambda_proto", 0.0)),
        "proto_temperature": float(param.get("proto_temperature", 0.2)),
        "proto_start_epoch": int(param.get("proto_start_epoch", 1)),
        "proto_full_epoch": int(param.get("proto_full_epoch", 1)),
        "proto_min_samples_per_class": int(param.get("proto_min_samples_per_class", 1)),
        "lambda_adv": float(param.get("lambda_adv", 1.0)),
    }
```

### 9.3 Integration point

Only integrate InfoNCE into `train_d_ae()` or the equivalent generator/encoder update.

Do **not** add InfoNCE to:

1. pretrain stage,
2. discriminator-only step,
3. classifier-only step.

### 9.4 Required `train_d_ae()` behavior

Add parameters for current GAN epoch or resolved effective weights:

```python
train_d_ae(
    ...,
    gan_lambda_cls,
    lambda_adv_eff=1.0,
    lambda_proto_eff=0.0,
    proto_temperature=0.2,
    proto_min_samples_per_class=1,
    ...
)
```

Inside the function:

```python
proto_loss = zero_tensor
proto_metrics = default_proto_metrics

if lambda_proto_eff > 0:
    proto_loss, proto_metrics = compute_batch_prototype_infonce(
        ccle_z,
        s_labels,
        tcga_z,
        t_labels,
        num_classes=num_classes,
        temperature=proto_temperature,
        min_samples_per_class=proto_min_samples_per_class,
    )

loss = (
    vae_loss
    + pvae_loss
    + o_loss
    + lambda_adv_eff * g_loss
    + gan_lambda_cls * cls_loss
    + lambda_proto_eff * proto_loss
)
```

### 9.5 Logging additions

Add columns to `g_loss.csv` without removing existing columns:

```text
lambda_adv_eff
lambda_proto_eff
proto_loss
proto_acc
proto_valid
proto_valid_class_count
proto_valid_sample_count
proto_mean_positive_similarity
proto_mean_negative_similarity
proto_margin
```

### 9.6 Summary row additions

Extend summary row with:

```text
lambda_proto
proto_temperature
proto_start_epoch
proto_full_epoch
lambda_adv
best_proto_loss
best_proto_margin
best_proto_acc
```

The existing summary fields must remain unchanged for compatibility with `visualize_vaewc_results.py` and finetune model selection.

---

## 10. Prototype Diagnostics Module

File:

```text
tools/pretrain_proto_metrics.py
```

### 10.1 Purpose

Compute global prototype metrics after each pretrain run. These metrics are for monitoring and reporting only in round 1.

### 10.2 Inputs

1. Saved source latent dictionary.
2. Saved target latent dictionary.
3. Source/target labels or mapping exported from pretrain.
4. Existing `params.json` or resolved config.

### 10.3 Metrics

| Metric | Meaning |
|---|---|
| `classwise_mmd_mean` | mean source-target MMD within cancer class |
| `same_class_cross_domain_distance` | source prototype to target prototype distance for same class |
| `inter_class_prototype_distance` | distance among different cancer prototypes |
| `prototype_separation_ratio` | inter-class distance / same-class distance |
| `prototype_coverage` | number of valid classes |

### 10.4 Output

```text
prototype_metrics.csv
prototype_metrics.json
```

---

## 11. Sequential Runner

File:

```text
tools/optimization_runner.py
```

### 11.1 Modes

| Mode | Command |
|---|---|
| generate | create configs and pretrain manifest |
| pretrain | run pending pretrain jobs sequentially |
| select | run visualize/filter and control-aware Top-10 |
| finetune | run selected Top-10 using mini config |
| aggregate | aggregate downstream results |
| report | generate reports |
| all | run the full pipeline |

### 11.2 Single-GPU policy

1. Set `CUDA_VISIBLE_DEVICES=0` by default.
2. Run exactly one subprocess at a time.
3. Mark manifest row `running` before process starts.
4. Mark `success` only if expected output files exist.
5. Mark `failed` with return code and error log path otherwise.
6. Support `--resume` to skip `success` jobs.
7. Support `--dry-run` to print commands without execution.

### 11.3 Pretrain command template

```bash
python pretrain_VAEwC.py \
  --config {config_path} \
  --outfolder {run_dir}/pretrain \
  --target_domain tcga \
  --overlap_tcga data/TCGA/PMID27354694_DR_OMICS_ad.csv
```

### 11.4 Finetune command template

```bash
python step1_finetune_latent_pipeline_All_split.py \
  --config config/params_finetune_mini.json \
  --model_select_path {model_select_top10_with_controls} \
  --outfolder {run_dir}/finetune \
  --batch_size 2048 \
  --mini_batch_size 512
```

### 11.5 Aggregate command template

```bash
python aggregate_pretrain_tcga_scores.py \
  --input {run_dir}/finetune/parameter_comparison_tcga_focus.csv \
  --output {run_dir}/aggregate/pretrain_tcga_model_summary.csv \
  --top_n 10
```

---

## 12. Selection Wrapper

File:

```text
tools/optimization_selection.py
```

### 12.1 Purpose

Wrap existing `visualize_vaewc_results.py` output and enforce the project-specific Top-10 control policy.

### 12.2 Inputs

```text
selection/aggregated_vaewc_results.csv
selection/model_select.csv
config/visualize_vaewc_filter.json
```

### 12.3 Output

```text
selection/model_select_top10_with_controls.csv
selection/pretrain_top10_with_controls.csv
selection/pretrain_selection_report.md
```

### 12.4 Logic

1. Run or consume `visualize_vaewc_results.py` results.
2. Identify valid `lambda_proto=0` controls.
3. Select best two controls by `score_total`.
4. Select best eight remaining candidates by `score_total`.
5. Combine into final Top-10.
6. If fewer than two controls are available, record exception in report and fill remaining slots with best available candidates.

---

## 13. Finetune Dispatch

Use existing `step1_finetune_latent_pipeline_All_split.py`.

### 13.1 Input

```text
selection/model_select_top10_with_controls.csv
config/params_finetune_mini.json
```

### 13.2 Expected workload

```text
10 pretrain candidates × 4 configs = 40 finetune jobs
```

The finetune script itself iterates over selected models and config combinations. The runner should treat this as one finetune stage unless future runtime requires splitting by pretrain model.

### 13.3 Required outputs to verify

```text
finetune/parameter_comparison_detailed.csv
finetune/parameter_comparison_tcga_focus.csv
finetune/all_parameter_results_summary.json
```

---

## 14. Aggregation Wrapper

Use existing `aggregate_pretrain_tcga_scores.py`.

### 14.1 Input

```text
finetune/parameter_comparison_tcga_focus.csv
```

### 14.2 Output

```text
aggregate/pretrain_tcga_model_summary.csv
```

### 14.3 Primary ranking

```text
Global_TCGA_AUC_mean
```

The final report must also include:

```text
Global_TCGA_AUC_std
n_finetune_runs
Average_TCGA_AUC_mean
Test_AUC_mean
```

when available.

---

## 15. Artifact Retention

File:

```text
tools/artifact_retention.py
```

### 15.1 Retention policy

| Run type | Keep | Delete after selection |
|---|---|---|
| selected Top-10 | all artifacts | none |
| selected no-InfoNCE controls | all artifacts | none |
| completed non-selected | config, metrics, status, reports | checkpoints, large latent dumps |
| failed | config, status, logs, partial CSV | checkpoints, latents, large plot cache |

### 15.2 Default large patterns

```text
*.pt
*.pth
*checkpoint*
*latent*.pkl
*latent*.csv
```

Do not delete:

```text
params.json
resolved_config.json
run_status.json
*.csv summary files
*.md reports
stdout/stderr logs
```

---

## 16. Reporting

File:

```text
tools/optimization_report.py
```

### 16.1 Required reports

| Report | Purpose |
|---|---|
| `reports/pretrain_sweep_report.md` | 72-run status and metric summary |
| `selection/pretrain_selection_report.md` | filter, ranking, and control selection rationale |
| `reports/finetune_status_report.md` | downstream execution summary |
| `reports/final_selection_report.md` | final interpretation and best model recommendation |

### 16.2 Final report must answer

1. Which InfoNCE configurations passed the hard filter?
2. Which InfoNCE configurations entered Top-8 non-control selection?
3. Which two no-InfoNCE controls were retained?
4. Did deconfounding metrics improve?
5. Did tumor-preservation metrics remain above thresholds?
6. Did downstream `Global_TCGA_AUC_mean` improve?
7. Which failure patterns appeared?
8. Which next ablation should be run?

---

## 17. Smoke Test Plan

Before 72 full runs:

1. Generate mini sweep:

```text
lambda_proto = [0, 0.1]
proto_temperature = [0.2]
proto_start_epoch = [5]
proto_full_epoch = [30]
```

2. Run with reduced pretrain/GAN epochs.
3. Confirm `g_loss.csv` contains prototype columns.
4. Confirm `lambda_proto=0` contributes zero prototype loss to total loss.
5. Confirm `lambda_proto=0.1` produces nonzero valid prototype metrics when valid classes exist.
6. Confirm `visualize_vaewc_results.py` still produces `aggregated_vaewc_results.csv` and `model_select.csv`.
7. Confirm selection wrapper produces `model_select_top10_with_controls.csv` for available candidates.
8. Confirm one selected model can run through finetune mini config.

---

## 18. Unit Test Plan

### 18.1 `tools/proto_infonce.py`

1. Multi-class normal batch.
2. Missing classes.
3. One valid class returns zero loss.
4. Invalid temperature raises error.
5. Backward pass works.
6. Label remapping to valid prototype indices is correct.

### 18.2 Config generator

1. Exactly 72 configs.
2. All combinations present.
3. All configs valid JSON.
4. `lambda_proto=0` controls included.

### 18.3 Runner

1. Sequential execution only.
2. Resume skips success jobs.
3. Failure status captures return code.
4. Dry-run prints all commands.

### 18.4 Selection wrapper

1. Selects eight best candidates plus two controls.
2. Handles fewer than two valid controls.
3. Preserves columns needed by finetune script.

### 18.5 Artifact retention

1. Keeps selected artifacts.
2. Deletes non-selected large files.
3. Keeps debug metadata for failed runs.

---

## 19. Implementation Order for Cursor

1. Implement `tools/proto_infonce.py`.
2. Add unit tests for prototype InfoNCE.
3. Add `lambda_proto`, `proto_temperature`, `proto_start_epoch`, `proto_full_epoch`, `lambda_adv` config resolution in `pretrain_VAEwC.py`.
4. Integrate InfoNCE into `train_d_ae()` only.
5. Add `lambda_adv_eff` multiplier to `g_loss` with default `1.0`.
6. Add prototype logging columns to `g_loss.csv`.
7. Extend summary row with prototype config and best prototype metrics.
8. Add `tools/optimization_config_generator.py`.
9. Add `tools/optimization_runner.py` sequential queue.
10. Add `tools/optimization_selection.py` for Top-10 with two controls.
11. Add `tools/pretrain_proto_metrics.py` for report-only diagnostics.
12. Add `tools/artifact_retention.py`.
13. Add `tools/optimization_report.py`.
14. Run smoke tests.
15. Run 72 VAEwC pretrain jobs.
16. Run Top-10 × mini finetune.
17. Run aggregation and final report.

---

## 20. Final Technical Decisions

| Topic | Decision |
|---|---|
| Source baseline | latest `JohnWCT/drug-DCF` main, checked during planning |
| First full model | VAEwC only |
| AEwC | smoke test only; inherits VAEwC core when compatible |
| CVAE | future synchronization target |
| InfoNCE stage | GAN generator/encoder update only |
| Prototype source | source+target shared latent |
| Prototype update | batch-based |
| EMA | not used |
| Memory bank | not used |
| TransDRP-like InfoMax | not used |
| Sweep size | 72 pretrain jobs |
| GPU execution | one A6000 Ada, sequential queue |
| Multi-GPU / DDP | not implemented |
| Top-K | Top-10 |
| Control policy | two `lambda_proto=0` controls inside Top-10 |
| Finetune config | `config/params_finetune_mini.json` only |
| Downstream jobs | 40 effective finetune combinations |
| Filter | `config/visualize_vaewc_filter.json` |
| Final ranking | `Global_TCGA_AUC_mean` |
| Reports | CSV + JSON + Markdown + plots |
