# Round 20 Inference Guide

## Preflight

Before inference, verify:

- Model lock SHA256 matches release `configs/final_model_lock.json`
- Selected context: **C32** (96-d O2)
- Checkpoint count: **15** (probability-mean ensemble)
- Drug graph coverage for all drugs in the response file

## Frozen latent inference (official)

```bash
python scripts/round20/round20_cli.py infer \
  --release-dir result/optimization_runs/round20_unseen_drug_closure/stage20e_release \
  --mode frozen_latent \
  --response-file path/to/response.csv \
  --output predictions.csv \
  --strict
```

## Raw omics inference (capability path)

```bash
python scripts/round20/round20_cli.py infer \
  --release-dir result/optimization_runs/round20_unseen_drug_closure/stage20e_release \
  --mode raw_omics \
  --response-file path/to/response.csv \
  --output predictions.csv \
  --strict
```

Encoder unfreezing was **not** validated as a formal Round 20 experiment.
