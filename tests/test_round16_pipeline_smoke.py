import subprocess
import sys


def test_round16_config_builder_importable():
    import tools.round16_bruteforce_config_builder  # noqa: F401
    import tools.analyze_round16_bruteforce  # noqa: F401
    import tools.round16_bruteforce_selection  # noqa: F401


def test_expand_finetune_combos_count():
    from tools.optimization_runner import _expand_finetune_combinations
    combos = _expand_finetune_combinations("config/params_finetune_round16_bruteforce.json")
    assert len(combos) == 24
    assert combos[0]["finetune_params"]["weight_decay"] == 1e-5 or combos[0]["finetune_params"]["weight_decay"] == 0.00001
