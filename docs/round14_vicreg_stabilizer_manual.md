# Round 14 VICReg Stabilizer Manual

**Purpose:** Low-weight VICReg var/cov latent stabilizer on validated Conditional ADV + Prototype Alignment + Response Feature stacks.

**Output dir:** `result/optimization_runs/round14_vicreg_stabilizer`

## References

| Benchmark | Avg TCGA |
|-----------|----------|
| Round 13 best `r13_exp_008_own_plus_summary` | **0.6112** |
| Round 13 `r13_exp_035_none` (z-only) | 0.6059 |
| Round 12 `exp_037` | 0.5972 |

## Quick start

```bash
# Tests
pytest tests/test_round14_config_builder.py \
  tests/test_round14_vicreg_training_flags.py \
  tests/test_round14_selection.py \
  tests/test_analyze_round14_vicreg_stabilizer.py \
  tests/test_round14_pipeline_smoke.py -q

# Config smoke
python tools/round14_config_builder.py \
  --settings config/round14_vicreg_stabilizer_settings.json \
  --outdir result/optimization_runs/round14_vicreg_stabilizer_smoke \
  --force

# Full pipeline
bash tools/run_round14_vicreg_stabilizer_pipeline.sh
```

## Pipeline stages

1. `round14_config_builder.py` — 84 pretrain jobs + reference controls
2. `optimization_runner.py pretrain`
3. `analyze_round14_vicreg_stabilizer.py` + `select --selection-mode round14_vicreg_stabilizer_qc --top-k 16`
4. `round14_config_builder.py --build-finetune-manifest`
5. `extract_round13_proto_features.py` (compact modes only)
6. `optimization_runner.py finetune --round13-mode`
7. aggregate + final report

## Key files

| File | Role |
|------|------|
| `config/round14_vicreg_stabilizer_settings.json` | Routes, VICReg grid, selection |
| `config/params_finetune_round14_proto_features.json` | 4-combo finetune grid |
| `tools/round14_config_builder.py` | Pretrain + finetune manifests |
| `tools/round14_selection.py` | `round14_vicreg_stabilizer_qc` |
| `tools/analyze_round14_vicreg_stabilizer.py` | Pretrain + downstream analysis |
| `tools/run_round14_vicreg_stabilizer_pipeline.sh` | End-to-end runner |

## Success criteria

- Beat Round 13 **0.6112**, or match with lower seed std
- Stretch: **0.6200**
- No latent collapse (active_dims, kmeans_ari, proto gap)

## OOM mitigation

```bash
PRETRAIN_PARALLEL=12 FINETUNE_PARALLEL=8 bash tools/run_round14_vicreg_stabilizer_pipeline.sh
```

Finetune retry pattern (if needed): copy `tools/run_round13_finetune_retry.sh` pattern with `FINETUNE_RETRY_PARALLEL=12`.
