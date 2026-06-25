# Round 16 Brute-force Manual

Focused downstream hyperparameter brute-force on Round 13 / Round 15 top stacks.

## Quick start

```bash
# Stage 16A (1152 finetune jobs)
FINETUNE_PARALLEL=12 bash tools/run_round16_bruteforce_stage16a.sh

# After 16A analysis recommends confirmation
FINETUNE_PARALLEL=12 bash tools/run_round16_confirmation_stage16b.sh
```

## Stages

| Stage | Purpose | Jobs |
|-------|---------|------|
| 16A | 4 models × 4 feature modes × 24 combos × 3 seeds | 1152 |
| 16B | Top 10 candidates × 10 seeds | 100 |
| 16C | Feature ablation (optional, may overlap 16A) | 432 |
| 16D | Ultra-low VICReg micro-search (disabled by default) | 924 |

## References

- Round 13 peak: **0.6112**
- Round 15 best: **0.6083**
- Stretch target: **0.6200**

See `config/round16_bruteforce_settings.json` for model pool and feature modes.
