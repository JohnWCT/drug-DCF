# Round 20 Stage 20A — C16 vs C32 Repeated Drug-Held-Out

## Status

**LOCKED** — selected context **C32** (`stable_improvement`).

| Metric | C16 | C32 | Δ (C32−C16) |
|--------|-----|-----|-------------|
| mean DrugMacro AUC | 0.7434 | 0.7509 | **+0.00745** |
| mean DrugMacro AUPRC | 0.4764 | 0.4829 | +0.00643 |
| seed Δ AUC (52/62/72) | — | — | +0.0084 / +0.0042 / +0.0098 |
| non-worse seeds | — | — | 3/3 |
| jobs | 15 | 15 | 30/30 complete, 0 failed |

Wall time ≈ 6.2 h (16-way parallel). All guardrails passed; parsimony threshold 0.005 exceeded.

## Research question

Under fixed frozen Z64, D0 GIN32, and the artifact-resolved pooled E3 predictor,
does increasing the prototype-context projection from 16 → 32 dimensions produce
a **stable** unseen-drug improvement of at least `0.005` DrugMacro AUC?

## Design logic

### Why reuse Round 19 training blocks

Stage 20A must isolate a *single* experimental factor: context dimension.
Therefore the runner (`step1_finetune_latent_pipeline_round20_cv.py`) reuses:

- `Round19ResponseDataset` / collate
- D0 GIN encoder builder
- P0 `AdapterMLPFusion` + `Round18ResponseHead`
- Round 19 param groups, Focal loss, AMP train/eval loop
- Robust DrugMacro AUC / AUPRC metrics

The only intentional difference between paired jobs is the feature store:

| Arm | Feature store | Omics dim |
|-----|---------------|-----------|
| `A_C16_E3` | `features/z_plus_context16` | 80 = Z64 + C16 |
| `A_C32_E3` | `features/z_plus_context32` | 96 = Z64 + C32 |

PCA is **not** re-fit. Feature stores are read-only. E3 hyperparameters come
strictly from `stage20_0/resolved_e3.json` (fail-closed resolver).

### Split contract

- Development rows only (`development_rows.csv`, 99 218 rows)
- Drug grouping by `canonical_smiles` (alias/salt-safe)
- Seeds `{52, 62, 72}` × 5 GroupKFold folds
- Model seed fixed at `101`
- Leakage audit: train ∩ val drug identities = ∅ for every fold

C16 and C32 jobs share identical assignment files and SHA256 hashes.

### Decision rule (parsimony-first)

```
if mean_AUC(C32) < mean_AUC(C16)            → lock C16
elif |delta| < 0.005                        → lock C16 (parsimony)
elif delta ≥ 0.005
     and ≥2/3 seeds non-worse
     and AUPRC drop ≤ 0.01
     and no seed delta < −0.02              → lock C32
else                                        → lock C16
```

Selection never reads TCGA / internal / post-hoc metrics.

## Dispatch / GPU strategy

Each job uses ≈20 MB VRAM; the bottleneck is CPU graph collation.
Measured scaling on RTX 6000 Ada (49 GB):

| Concurrent jobs | Wall time / epoch |
|-----------------|-------------------|
| 1               | ~122 s            |
| 4               | ~130 s            |
| 8               | ~133 s            |
| 16              | ~145 s            |

Default: **16-way ProcessPoolExecutor**, micro-batch 256 × accum 4
(= effective batch 1024, matching resolved E3). OOM retry halves micro-batch
and doubles accumulation to keep effective batch constant.

Telegram notifications fire on start / every 5 completions / finish.

## Artifacts

```text
stage20_0/stage20_0_go.json
stage20_0/resolved_e3.json
splits/round20a_drug_heldout_seed{52,62,72}_assignments.csv
splits/round20a_drug_split_audit.json
stage20a_dimension/manifest.jsonl          # 30 jobs
stage20a_dimension/jobs/<job_id>/...
stage20a_dimension/stage20a_dimension_decision.json   # after analyze
```

## Smoke results (1 epoch)

| Job | Omics dim | DrugMacro AUC (epoch 0) |
|-----|-----------|-------------------------|
| C16 ss52 f0 | 80 | 0.585 |
| C32 ss52 f0 | 96 | 0.612 |

Smoke confirms correct input dims, finite loss/metrics, checkpoint save.

## Next

Stage 20B is running on locked **C32**: 15 gated jobs training; 15 E3 baseline
jobs reused from Stage 20A winner arm. Then auto-chain 20C → 20D → 20E.
