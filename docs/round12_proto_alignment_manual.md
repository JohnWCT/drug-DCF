# Round 12 Prototype Alignment Manual

**Purpose:** Conditional ADV + Source-anchor EMA Prototype Alignment on Round 11 `exp_035` baseline.

## Quick start

```bash
# 1) Tests
pytest tests/test_source_anchor_prototypes.py \
  tests/test_round12_config_builder.py \
  tests/test_round12_proto_alignment_training_flags.py \
  tests/test_round12_selection.py \
  tests/test_analyze_round12_proto_alignment.py -q

# 2) Config smoke (66 jobs)
python tools/round12_config_builder.py \
  --settings config/round12_proto_alignment_settings.json \
  --outdir result/optimization_runs/round12_proto_alignment_smoke \
  --force

# 3) Full pipeline
bash tools/run_round12_proto_alignment_pipeline.sh
```

## Baselines

| Reference | Avg TCGA |
|-----------|----------|
| Round 11 exp_035 | 0.5828 |
| Round 10 exp_111 | 0.5749 |
| R7 exp_048 | 0.5918 |

## Key parameters (12B main)

- `source_anchor_proto_enabled=true`
- `lambda_proto_align`: 0.0001, 0.0003, 0.001, 0.003
- `proto_align_schedules`: (20→60), (40→90), (60→120)
- `proto_ema_momentum`: 0.95
- `proto_align_metric`: cosine (12D: euclidean control)
- Conditional ADV unchanged from exp_035

## Artifacts

- Settings: `config/round12_proto_alignment_settings.json`
- Run root: `result/optimization_runs/round12_proto_alignment/`
- 12A QC: `round12a_baseline_qc/`
- Final report: `final_report/round12_final_report.md`

## Success criteria

1. same-cancer prototype distance < exp_035
2. inter-cancer margin retained
3. `Average_TCGA_AUC_mean` > 0.5828
4. Stretch goal: >= 0.5918 (R7 exp_048)

## Round 13

- **Go:** prototype-distance response features in Step 2 predictor
- **No-go:** Round 12.1 — lower λ, later start, stronger weak global guard
