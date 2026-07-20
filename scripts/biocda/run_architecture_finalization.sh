#!/usr/bin/env bash
# BioCDA architecture finalization — run inside Docker DAPL (/workspace/DAPL).
# Host usage:
#   docker exec DAPL bash -lc '/workspace/DAPL/scripts/biocda/run_architecture_finalization.sh'
set -euo pipefail
cd /workspace/DAPL

notify() {
  python3 tools/biocda_telegram_notify.py --message "$1" || true
}

notify "BioCDA Round 21 validation START"

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

echo "=== repository + architecture audit ==="
python3 scripts/audit_repository_state.py --strict || true
python3 scripts/audit_biocda_architecture.py --config configs/biocda/xa_validation.yaml --strict

echo "=== architecture smoke (GPU if available) ==="
python3 scripts/biocda/run_architecture_smoke_test.py --device auto

echo "=== xa validation smoke pipeline ==="
python3 scripts/run_xa_validation.py --config configs/biocda/xa_validation.yaml smoke

notify "BioCDA Round 21 validation DONE — see reports/ and outputs/xa_validation/"

echo "ALL_DONE"
