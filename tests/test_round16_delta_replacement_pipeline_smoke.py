from tools.optimization_runner import _expand_finetune_combinations

def test_delta_replacement_combos():
    combos = _expand_finetune_combinations("config/params_finetune_round16_delta_replacement.json")
    assert len(combos) == 8

def test_stage16f_builder_import():
    import tools.round16_bruteforce_config_builder
