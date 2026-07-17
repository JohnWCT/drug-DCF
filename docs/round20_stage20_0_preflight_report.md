# Round 20-0 — Preflight & E3 Fail-Closed Audit

## Status

**GO** — E3 uniquely resolved; C16/C32 comparable after Round 20 C32 rebuild.

## Resolved E3

| Field | Value |
|---|---|
| Public alias | E3 |
| Source candidate | `F3_best_pooled_o2` |
| Architecture | pooled_mlp (`AdapterMLPFusion+ResponseHead`) |
| Omics / Drug / Predictor | O2 / D0 / P0 |
| Context dim (locked artifact) | 16 baseline; C32 challenger rebuilt |
| D0 training mode | `end_to_end_finetune` |
| Checkpoints | 15 stage19d folds (seeds 52/62/72 × 5) |
| Reconstructed | false |

## Context audit

- C16: Round 19 O2 copy under Round 20 features, with shared comparability attestation.
- C32: rebuilt from the **same raw context matrix** as C16 (`n_components=32`, `random_state=42`).
- Legacy C16 PCA transform verified against reconstructed raw matrix before C32 fit.
- Audit `comparable=true`.

Artifacts:

```text
result/optimization_runs/round20_unseen_drug_closure/projections/context16/
result/optimization_runs/round20_unseen_drug_closure/projections/context32/
result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context16/
result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context32/
```

## Drug identity

- `drug_group_id = canonical_smiles` (alias merges recorded).

## Locked seeds

`[52, 62, 72]` × 5 folds.

## Next action

Stage 20A: build 30-job C16 vs C32 repeated drug-held-out manifest and train.
