import json, os
from tools.optimization_config_generator import expand_sweep_combinations
SWEEP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "pretrain_sweeps")

def _load(name):
    with open(os.path.join(SWEEP_DIR, name)) as f:
        return json.load(f)["sweep"]

def test_round6A_job_count():
    assert len(expand_sweep_combinations(_load("vaewc_round6A_tumor_topology.json"))) == 16

def test_round6B_job_count():
    assert len(expand_sweep_combinations(_load("vaewc_round6B_topology_classgap_combo.json"))) == 18

def test_round6C_tumor_dim_less_than_latent():
    combos = expand_sweep_combinations(_load("vaewc_round6C_tumor_transfer_subspace.json"))
    assert len(combos) == 24
    assert all(c["tumor_dim"] < c["latent_size"] for c in combos)

def test_round6D_supcon_includes_zero():
    combos = expand_sweep_combinations(_load("vaewc_round6D_within_domain_tumor_supcon.json"))
    assert len(combos) == 32
    assert any(c["lambda_tumor_supcon"] == 0 for c in combos)

def test_round6E_vicreg_includes_zero_pair():
    combos = expand_sweep_combinations(_load("vaewc_round6E_tumor_vicreg_stabilizer.json"))
    assert len(combos) == 12
    assert any(c["lambda_tumor_var"] == 0 and c["lambda_tumor_cov"] == 0 for c in combos)

def test_paired_params_latent_encoder_bound():
    for c in expand_sweep_combinations(_load("vaewc_round6A_tumor_topology.json")):
        if c["latent_size"] == 32:
            assert c["encoder_dims"] == [256, 128]
