"""Round 19 MACCS encoder / coverage tests."""
import numpy as np
import torch

from tools.round19_drug_encoders import MACCSDrugEncoder
from tools.round19_drug_features import (
    EXPECTED_MACCS_BITS,
    assert_no_graph_fields_in_maccs_batch,
    load_maccs_by_drug_name,
    validate_maccs_coverage,
)


def test_maccs_encoder_forward_and_grad():
    enc = MACCSDrugEncoder(input_dim=EXPECTED_MACCS_BITS, output_dim=64)
    x = torch.randn(4, EXPECTED_MACCS_BITS)
    y = enc(x)
    assert y.shape == (4, 64)
    y.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


def test_maccs_loader_sample_and_no_graph_fields():
    path = "data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv"
    m = load_maccs_by_drug_name(path, drug_names=None)
    names = list(m.keys())[:5]
    validate_maccs_coverage(m, names)
    assert all(v.shape == (EXPECTED_MACCS_BITS,) for v in (m[n] for n in names))
    assert_no_graph_fields_in_maccs_batch({"maccs": np.zeros(EXPECTED_MACCS_BITS), "omics": 1})
