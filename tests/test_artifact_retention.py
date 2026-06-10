import os

import pandas as pd

from tools.artifact_retention import apply_retention, plan_deletions


def test_top10_files_protected(tmp_path):
    pretrain = tmp_path / "pretrain"
    protected_exp = pretrain / "exp_001"
    other_exp = pretrain / "exp_999"
    protected_exp.mkdir(parents=True)
    other_exp.mkdir(parents=True)
    protected_file = protected_exp / "after_traingan_shared_vae.pth"
    delete_file = other_exp / "ccle_latent_dict.pkl"
    protected_file.write_text("x", encoding="utf-8")
    delete_file.write_text("x", encoding="utf-8")

    top10 = tmp_path / "top10.csv"
    pd.DataFrame({"ID": ["exp_001"]}).to_csv(top10, index=False)

    planned = plan_deletions(str(pretrain), {"exp_001"})
    assert str(delete_file) in planned
    assert str(protected_file) not in planned


def test_dry_run_does_not_delete(tmp_path):
    pretrain = tmp_path / "pretrain" / "exp_999"
    pretrain.mkdir(parents=True)
    latent = pretrain / "tcga_latent_dict.pkl"
    latent.write_text("x", encoding="utf-8")
    top10 = tmp_path / "top10.csv"
    pd.DataFrame({"ID": ["exp_001"]}).to_csv(top10, index=False)
    apply_retention(str(tmp_path), str(top10), apply=False)
    assert latent.exists()


def test_apply_deletes_and_logs(tmp_path):
    pretrain = tmp_path / "pretrain" / "exp_999"
    pretrain.mkdir(parents=True)
    latent = pretrain / "tcga_latent_dict.pkl"
    latent.write_text("x", encoding="utf-8")
    top10 = tmp_path / "top10.csv"
    pd.DataFrame({"ID": ["exp_001"]}).to_csv(top10, index=False)
    log_df = apply_retention(str(tmp_path), str(top10), apply=True)
    assert not latent.exists()
    assert len(log_df) >= 1
