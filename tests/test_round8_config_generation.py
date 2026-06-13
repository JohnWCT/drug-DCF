import json
import os

from tools.optimization_config_generator import expand_sweep_combinations

SWEEP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "pretrain_sweeps")

ACTIVE_TUMOR_KEYS = (
    "lambda_tumor_topology",
    "lambda_class_gap",
    "lambda_tumor_supcon",
    "lambda_subspace_ortho",
    "lambda_tumor_var",
    "lambda_tumor_cov",
)


def _load(name):
    with open(os.path.join(SWEEP_DIR, name)) as f:
        return json.load(f)["sweep"]


def test_round8A_job_count():
    assert len(expand_sweep_combinations(_load("vaewc_round8A_control_arch_broad.json"))) == 288


def test_round8B_job_count():
    assert len(expand_sweep_combinations(_load("vaewc_round8B_vicreg_arch_broad.json"))) == 224


def test_round8A_all_active_tumor_losses_zero():
    for c in expand_sweep_combinations(_load("vaewc_round8A_control_arch_broad.json")):
        for key in ACTIVE_TUMOR_KEYS:
            assert float(c.get(key, 0)) == 0.0


def test_round8B_only_vicreg_active():
    combos = expand_sweep_combinations(_load("vaewc_round8B_vicreg_arch_broad.json"))
    for c in combos:
        assert float(c["lambda_tumor_topology"]) == 0.0
        assert float(c["lambda_class_gap"]) == 0.0
        assert float(c["lambda_tumor_supcon"]) == 0.0
        assert float(c["lambda_subspace_ortho"]) == 0.0
        var = float(c["lambda_tumor_var"])
        cov = float(c["lambda_tumor_cov"])
        assert var >= 0 and cov >= 0
        assert var > 0 or cov > 0


def test_round8B_paired_var_cov_not_crossed():
    combos = expand_sweep_combinations(_load("vaewc_round8B_vicreg_arch_broad.json"))
    pairs = {(round(c["lambda_tumor_var"], 5), round(c["lambda_tumor_cov"], 5)) for c in combos}
    assert (0.0003, 0.0001) in pairs
    assert (0.0001, 0.0003) in pairs
    assert len(pairs) == 6


def test_round8A_latent_and_encoder_coverage():
    latent_sizes = set()
    encoder_dims = set()
    for c in expand_sweep_combinations(_load("vaewc_round8A_control_arch_broad.json")):
        latent_sizes.add(c["latent_size"])
        encoder_dims.add(tuple(c["encoder_dims"]))
    assert {32, 48, 64, 96, 128}.issubset(latent_sizes)
    assert {(512, 256, 128), (768, 384, 192), (1024, 512, 256)}.issubset(encoder_dims)


def test_round8B_latent_and_encoder_coverage():
    latent_sizes = set()
    encoder_dims = set()
    for c in expand_sweep_combinations(_load("vaewc_round8B_vicreg_arch_broad.json")):
        latent_sizes.add(c["latent_size"])
        encoder_dims.add(tuple(c["encoder_dims"]))
    assert {48, 64, 96, 128}.issubset(latent_sizes)
    assert {(512, 256, 128), (768, 384, 192), (1024, 512, 256)}.issubset(encoder_dims)
