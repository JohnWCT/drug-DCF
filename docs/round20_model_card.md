# Round 20 Model Card

## Intended use

Research-oriented prediction of drug response for drugs excluded from response-model
training under the repository's established omics and drug preprocessing contracts.

## Locked configuration

| Field | Value |
|-------|-------|
| Git SHA | `None` |
| Model lock SHA256 | `d20fdbbb900ef4312df1ba36d95a5cef5989d5455f805ec45d432522ffc50ba9` |
| Context | C32 (96-d O2) |
| Drug encoder | D0 GIN32 (graph dim 32) |
| Predictor | AdapterMLPFusion+ResponseHead |
| Candidate | B_E3 |
| Checkpoint policy | five_fold_probability_mean_ensemble |

## Development metrics (selection only)

- Stage 20A ΔAUC (C32−C16): 0.007449929979453596
- Stage 20B ΔAUC (gated−E3): -0.0020064731375666334

## Out of scope

Clinical treatment recommendation; dose/toxicity; combinations; unseen-cancer optimization;
molecules outside the locked graph contract.

## Reproduction

```bash
python scripts/round20/round20_cli.py audit --strict
python scripts/round20/round20_cli.py reproduce --strict
```
