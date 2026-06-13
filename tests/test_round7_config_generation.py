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


def test_round7A_job_count():
    assert len(expand_sweep_combinations(_load("vaewc_round7A_exp010_control_refinement.json"))) == 108


def test_round7B_job_count():
    assert len(expand_sweep_combinations(_load("vaewc_round7B_vicreg_focused_ablation.json"))) == 56


def test_round7A_all_active_tumor_losses_zero():
    for c in expand_sweep_combinations(_load("vaewc_round7A_exp010_control_refinement.json")):
        for key in ACTIVE_TUMOR_KEYS:
            assert float(c.get(key, 0)) == 0.0


def test_round7B_only_vicreg_active():
    combos = expand_sweep_combinations(_load("vaewc_round7B_vicreg_focused_ablation.json"))
    for c in combos:
        assert float(c["lambda_tumor_topology"]) == 0.0
        assert float(c["lambda_class_gap"]) == 0.0
        assert float(c["lambda_tumor_supcon"]) == 0.0
        assert float(c["lambda_subspace_ortho"]) == 0.0
        var = float(c["lambda_tumor_var"])
        cov = float(c["lambda_tumor_cov"])
        assert var >= 0 and cov >= 0


def test_round7B_paired_var_cov_not_crossed():
    combos = expand_sweep_combinations(_load("vaewc_round7B_vicreg_focused_ablation.json"))
    pairs = {(c["lambda_tumor_var"], c["lambda_tumor_cov"]) for c in combos}
    assert (0.0, 0.0) in pairs
    assert (0.0003, 0.0001) in pairs
    assert (0.0001, 0.0003) in pairs
    assert len(pairs) == 7


def test_round7_latent_and_encoder_fixed():
    for name in ("vaewc_round7A_exp010_control_refinement.json", "vaewc_round7B_vicreg_focused_ablation.json"):
        for c in expand_sweep_combinations(_load(name)):
            assert c["latent_size"] == 64
            assert c["encoder_dims"] == [512, 256, 128]
