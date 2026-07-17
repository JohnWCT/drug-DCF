# Round 20-0 — Preflight & E3 Fail-Closed Audit

## Status

**NO_GO for Stage 20A training** until a comparable C32 projection exists.

## Resolved E3

| Field | Value |
|---|---|
| Public alias | E3 |
| Source candidate | `F3_best_pooled_o2` |
| Architecture | pooled_mlp (`AdapterMLPFusion+ResponseHead`) |
| Omics / Drug / Predictor | O2 / D0 / P0 |
| Context dim (locked artifact) | 16 (omics dim 80) |
| D0 training mode | `end_to_end_finetune` (encoder_lr>0 + encoder state present) |
| Checkpoints | 15 stage19d folds (seeds 52/62/72 × 5) |
| Reconstructed | false |

Cross-sources: role lock, deployment policy, role proposal identity, factorial settings predictor map, and checkpoint introspection all agree.

## Context audit

- C16 (`z_plus_context16`) OK: latent 64 + context 16, projection PCA source `own_proto_context_projected_16`.
- C32: **missing**. Audit did **not** auto-rebuild.
- Rebuild rule (when approved): same raw context matrix/rows/normalization/algorithm/seed; only `n_components` 16→32.

## Drug identity

- Source: Round 19E `round19e_drug_group_table.csv` (230 drugs).
- Round 20 `drug_group_id` = `canonical_smiles` to prevent alias leakage.
- Alias merges recorded under `audit/drug_identity_exceptions.json`.

## Locked seeds

`[52, 62, 72]` × 5 folds (GroupKFold on `drug_group_id`).

## Artifacts

```text
result/optimization_runs/round20_unseen_drug_closure/audit/
  base_git_sha.txt
  schema_validation.json
  resolved_e3.json
  context_audit.json
  drug_identity_audit.json
  drug_identity_mapping.csv
  drug_identity_exceptions.json
  stage20_0_preflight_status.json
```

## Next action

1. Rebuild comparable C32 under `result/optimization_runs/round20_unseen_drug_closure/projections/context32/`.
2. Re-run context audit until `comparable=true`.
3. Only then generate Stage 20A 30-job manifest.
