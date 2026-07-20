# BioCDA Architecture Finalization — BioCDA-XA v1

**Status: COMPLETE** — verified in Docker `DAPL` (`/workspace/DAPL`).

## Architecture summary

BioCDA (Biological-Context-Guided Drug Response) uses frozen omics latent **Z** and
prototype biological context **C** to form sample representation **S**, which queries
atom-level GIN node embeddings via multi-head cross-attention. The response head receives
only the attended drug representation — no global graph pooling bypass.

| Component | Specification |
|-----------|---------------|
| Architecture | BioCDA-XA `biocda-xa-v1` |
| Omics latent | Z64 (frozen) |
| Biological context | 32-d prototype context (frozen) |
| Sample representation | concat(Z, C) → 96-d |
| Drug encoder | D0 GIN32 atom node embeddings (no pooling in encoder) |
| Cross-attention | 4 heads, 64-d, query=S, keys/values=atoms |
| Baseline | D0-Pooled (`pooled_baseline` factory) |

## Verification (Docker)

```bash
docker exec DAPL bash -lc '/workspace/DAPL/scripts/biocda/run_architecture_finalization.sh'
```

Expected:

- `pytest`: 38 passed
- `ARCHITECTURE_SMOKE=PASS`
- Telegram: START / DONE (if `.env` configured)

## Output modes (interpretability round ready)

```python
output = model(omics, biological_context, drug_graph, output_mode="attention")
# output.atom_attention, output.atom_mask, output.model_atom_index, ...
```

Modes: `prediction` | `attention` | `full` — switching mode does not change logits.

## Completion checklist

- [x] BioCDA-XA architecture version defined
- [x] Sample representation includes Z + C
- [x] GIN atom-level node embeddings
- [x] Cross-attention Q/K/V sources correct
- [x] Per-head attention logits and probabilities
- [x] Atom mask and index interfaces
- [x] No global pooling bypass in cross-attention model
- [x] output_mode three modes; logits invariant
- [x] eval: valid atom attention sums to 1; padding = 0
- [x] Checkpoint strict roundtrip
- [x] Pooled baseline via model factory
- [x] Unit tests + smoke test PASS
- [x] GPU benchmark on CUDA
- [x] Telegram notifications

**Out of scope this round:** TCGA inference, heatmaps, IG/DeepLIFT, attention aggregation.

## Key paths

```text
biocda/models/          Core architecture
configs/model/          biocda_cross_attention.yaml, pooled_baseline.yaml
scripts/biocda/         run_architecture_finalization.sh
reports/                Committed architecture manifest (portable)
outputs/architecture_finalization/   Local smoke artifacts (gitignored)
```

## Smoke test snapshot

```json
{
  "status": "PASS",
  "architecture_version": "biocda-xa-v1",
  "valid_attention_sum_max_error": 0.0,
  "padding_nonzero_count": 0,
  "gpu_benchmark": "ok"
}
```
