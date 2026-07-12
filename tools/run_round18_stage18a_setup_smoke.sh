#!/usr/bin/env bash
# Round 18 Stage 18A: setup + smoke tests
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SETTINGS="${ROUND18_SETTINGS:-config/round18_architecture_settings.json}"
OUTDIR="${ROUND18_ROOT:-result/optimization_runs/round18_architecture}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" "${OUTDIR}"

echo "========== ROUND18 STAGE 18A SETUP SMOKE START $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

echo "[18A] Round-18 sklearn pin check"
python - <<'PY'
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

print("sklearn:", sklearn.__version__)
print(StratifiedGroupKFold)
major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
assert (major, minor) >= (1, 3), sklearn.__version__
PY

echo "[18A] py_compile"
python -m py_compile \
  tools/cross_attention_switch.py \
  tools/round18_fusion_models.py \
  tools/round18_response_head.py \
  tools/round18_cv_splits.py \
  tools/round18_cv_metrics.py \
  tools/round18_oom_runner.py \
  tools/round18_config_builder.py \
  tools/round18_train_loop.py \
  tools/round18_prediction_ensemble.py \
  tools/round18_eligible_data.py \
  tools/round18_dataset.py \
  tools/round18_sklearn_compat.py \
  step1_finetune_latent_pipeline_round18_cv.py \
  drugmodels/ginconv.py \
  tools/transformer_switch.py

echo "[18A] pytest round18 unit tests"
pytest tests/test_round18_*.py -q

echo "[18A] build eligible + splits"
python tools/round18_config_builder.py \
  --settings "${SETTINGS}" \
  --outdir "${OUTDIR}" \
  --stage 18a

echo "[18A] real-data smoke (1-2 batches)"
python step1_finetune_latent_pipeline_round18_cv.py \
  --mode data_smoke \
  --outdir "${OUTDIR}" \
  --architecture-family pooled_mlp \
  --omics-mode own_plus_summary \
  --fold-id 0 \
  --micro-batch-size 8 \
  --accumulation-steps 2 \
  --max-batches 2 \
  --result-dir "${OUTDIR}/data_smoke"

echo "[18A] GIN / CrossAttention / OOM live smoke"
python - <<'PY'
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from drugmodels.ginconv import GINConvNet
from tools.cross_attention_switch import CrossAttentionSwitch
from tools.round18_fusion_models import build_fusion_and_head
from tools.round18_oom_runner import probe_micro_batch, write_resource_metadata

# GIN API smoke
gin = GINConvNet(input_dim=78, output_dim=32, dropout=0.1, num_layers=5, jk_mode="last", pool_type="max")
gin.train()
assert gin.training

graphs = []
for n in (4, 6, 5):
    x = torch.randn(n, 78)
    src = torch.arange(0, n - 1)
    dst = torch.arange(1, n)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    graphs.append(Data(x=x, edge_index=edge_index))
batch = Batch.from_data_list(graphs)

legacy = gin(batch)
assert legacy.shape == (3, 32), legacy.shape

out = gin(batch, return_node_embeddings=True, return_graph_embedding=True)
assert set(out) >= {"node_embeddings", "batch_index", "graph_embedding"}
assert out["node_embeddings"].shape[0] == batch.x.shape[0]
assert out["graph_embedding"].shape == (3, 32)
print("GIN smoke OK", out["node_embeddings"].shape, out["graph_embedding"].shape)

# Cross-attention smoke
ca = CrossAttentionSwitch(d_model=64, n_heads=4, num_layers=2, dim_feedforward=128)
ca.eval()
q = torch.randn(3, 1, 64)
kv = torch.randn(3, 8, 64)
mask = torch.zeros(3, 8, dtype=torch.bool)
mask[:, 6:] = True
upd, attn = ca(q, kv, key_padding_mask=mask, return_attention=True)
assert upd.shape == (3, 1, 64)
assert attn.shape == (2, 3, 4, 1, 8)
pad_attn = attn[..., 6:].abs().max().item()
assert pad_attn < 1e-5, pad_attn
valid_sum = attn[..., :6].sum(dim=-1)
assert torch.allclose(valid_sum, torch.ones_like(valid_sum), atol=1e-4)
print("CrossAttention smoke OK", upd.shape, attn.shape)

# Fusion families smoke (representation + separate head)
omics = torch.randn(3, 43)
mlp, head = build_fusion_and_head("pooled_mlp", omics_dim=43, graph_dim=32)
logits = head(mlp(omics, legacy))
assert logits.shape == (3,)

tf, head = build_fusion_and_head(
    "pooled_transformer",
    omics_dim=43,
    graph_dim=32,
    transformer_cfg={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128},
)
assert tf.metadata()["effective_use_mask"] is False
logits = head(tf(omics, legacy))
assert logits.shape == (3,)

cross, head = build_fusion_and_head(
    "cross_attention",
    omics_dim=43,
    graph_dim=32,
    node_dim=32,
    residual_mode="pooled_residual",
    cross_attn_cfg={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128},
)
logits = head(cross(omics, out["node_embeddings"], out["batch_index"], graph_embedding=legacy))
assert logits.shape == (3,)
print("Fusion smoke OK")

# OOM probe synthetic
def try_fn(b):
    if b >= 256:
        raise RuntimeError("CUDA out of memory")
    return None

probe = probe_micro_batch([512, 256, 128, 64, 32], target_effective_batch=1024, try_fn=try_fn)
assert probe.successful_micro_batch == 128
assert probe.oom_retry_count == 2
assert probe.gradient_accumulation_steps == 8
assert probe.oom_batch_history == [512, 256]
meta_dir = Path("result/optimization_runs/round18_architecture/smoke_resources")
write_resource_metadata(str(meta_dir), probe, extra={"gpu_name": "smoke"})
print("OOM probe OK", probe)

# Split QC presence
meta = json.loads(Path("result/optimization_runs/round18_architecture/splits/split_metadata.json").read_text())
assert meta["n_development_rows"] > 0
assert meta["n_internal_test_rows"] > 0
assert meta.get("eligible_only") is True
print("Split metadata OK", meta["internal_test_row_fraction"])
elig = json.loads(Path("result/optimization_runs/round18_architecture/data/round18_data_eligibility_summary.json").read_text())
assert elig["n_eligible_rows"] > 0
print("Eligible summary OK", elig["n_eligible_rows"])
print("ALL 18A LIVE SMOKES PASSED")
PY

echo "========== ROUND18 STAGE 18A SETUP SMOKE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
