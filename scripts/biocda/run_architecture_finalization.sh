#!/usr/bin/env bash
# BioCDA architecture finalization — run inside Docker DAPL (/workspace/DAPL).
# Host usage:
#   docker exec DAPL bash -lc '/workspace/DAPL/scripts/biocda/run_architecture_finalization.sh'
set -euo pipefail
cd /workspace/DAPL

notify() {
  python3 tools/biocda_telegram_notify.py --message "$1" || true
}

notify "BioCDA architecture finalization START"

echo "=== pytest (BioCDA unit tests) ==="
python3 -m pytest -q \
  tests/test_drug_node_encoder.py \
  tests/test_cross_attention.py \
  tests/test_biocda_forward.py \
  tests/test_attention_output.py \
  tests/test_padding_mask.py \
  tests/test_checkpoint_roundtrip.py \
  tests/test_model_factory.py \
  tests/test_biocda_context_and_bypass.py

echo "=== architecture smoke (GPU if available) ==="
python3 scripts/biocda/run_architecture_smoke_test.py --device auto

notify "BioCDA architecture finalization DONE — see outputs/architecture_finalization/"

echo "ALL_DONE"
