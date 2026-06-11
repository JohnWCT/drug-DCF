from tools.optimization_config_generator import expand_sweep_combinations


def test_paired_params_latent_encoder_grid():
    sweep = {
        "paired_params": [
            {"latent_size": 32, "encoder_dims": [256, 128]},
            {"latent_size": 64, "encoder_dims": [512, 256, 128]},
        ],
        "lambda_proto": [0, 0.001],
    }
    combos = expand_sweep_combinations(sweep)
    assert len(combos) == 4
    assert combos[0]["latent_size"] == 32
    assert combos[2]["latent_size"] == 64
    assert combos[0]["lambda_proto"] == 0


def test_control_centered_sweep_size():
    import json, os
    path = os.path.join("config/pretrain_sweeps/vaewc_round5_control_centered.json")
    with open(path) as f:
        spec = json.load(f)
    combos = expand_sweep_combinations(spec["sweep"])
    # 3 paired x 2 lambda_cls x 2 cls_start x 2 cls_full x 2 gan_interval = 48
    assert len(combos) == 48
