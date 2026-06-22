import json
import pandas as pd
from tests.round9_test_helpers import write_minimal_checkpoint
from tools.build_round9_reproduction_manifest import build_reproduction_manifest

def test_build_three_seed_configs(tmp_path):
    exp_dir = write_minimal_checkpoint(str(tmp_path), "exp_048", params={"latent_size": 64, "encoder_dims": [8, 4], "random_seed": 42})
    resolved = tmp_path / "resolved.csv"
    pd.DataFrame([{
        "exp_id": "exp_048", "role": "primary", "required": True, "resolved": True,
        "checkpoint_dir": exp_dir,
    }]).to_csv(resolved, index=False)
    baseline_cfg = tmp_path / "round9.json"
    baseline_cfg.write_text(json.dumps({"seeds": [101, 202, 303]}))
    outdir = tmp_path / "repro"
    manifest = build_reproduction_manifest(str(resolved), str(baseline_cfg), str(outdir), force=True)
    df = pd.read_csv(manifest)
    assert len(df) == 3
    assert set(df["reproduction_seed"].astype(int)) == {101, 202, 303}
    cfg = json.load(open(outdir / "configs" / "exp_048_seed101.json"))
    assert cfg["round9_reproduction"] is True
    assert cfg["source_exp_id"] == "exp_048"
    assert cfg["pretrain_param_combinations"][0]["latent_size"] == 64
    assert cfg["pretrain_param_combinations"][0]["random_seed"] == 101
