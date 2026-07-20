# Round 20 unseen-drug closure

## Scope

Repeated drug-held-out evaluation of prototype-context dimension and pooled predictor architecture.

## Locked model

- Context: **C32** (96-d O2)
- Drug encoder: D0
- Predictor: AdapterMLPFusion+ResponseHead (`B_E3`)
- Selection reason: gated_failed_guardrails

## Validation (development only)

- Stage 20A: C32 selected (ΔAUC C32−C16 = 0.0074)
- Stage 20B: gated `all_pass=False` (ΔAUC = -0.0020)

## TCGA evaluation

Performed after model lock. See `docs/round20_stage20d_tcga_report.md`.

## Reproduction

See [docs/round20_inference_guide.md](docs/round20_inference_guide.md).

## Limitations

Unseen cancer-type optimization and formal encoder unfreezing were outside Round 20 scope.
