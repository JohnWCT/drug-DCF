
import pytest
from pathlib import Path
import pandas as pd
from tools.analyze_round18 import analyze_round18, filter_complete_screening, select_formal_candidates_policy


def test_write_lock_requires_18c_manifest(tmp_path):
    out = tmp_path / "r18"
    man = out / "manifests"
    man.mkdir(parents=True)
    # 18b present but empty -> lock blocked; also 18c missing
    pd.DataFrame(columns=[
        "job_id","stage","architecture_id","architecture_family","omics_mode",
        "transformer_config_id","residual_mode","fold_id","result_dir",
        "response_data_path","feature_dir","split_assignment","drug_smiles_path",
    ]).to_csv(man / "stage18b_screening_manifest.csv", index=False)
    (out / "splits").mkdir(parents=True)
    pd.DataFrame([{"split_name":"development","n_rows":1}]).to_csv(out/"splits"/"fold_balance_report.csv", index=False)
    with pytest.raises(RuntimeError, match="Cannot write lock"):
        analyze_round18(str(out), write_lock=True)


def test_write_lock_requires_18c_file_when_18b_has_jobs(tmp_path, monkeypatch):
    out = tmp_path / "r18"
    man = out / "manifests"
    man.mkdir(parents=True)
    rd = out / "fake_job"
    rd.mkdir(parents=True)
    (rd / "job_status.json").write_text('{"status": "done"}')
    (rd / "val_metrics.json").write_text('{"DrugMacro_AUC": 0.6, "DrugMacro_AUPRC": 0.4}')
    pd.DataFrame([{
        "job_id": "j0", "stage": "18b", "architecture_id": "a",
        "architecture_family": "pooled_mlp", "omics_mode": "own_plus_summary",
        "transformer_config_id": "", "residual_mode": "", "fold_id": 0,
        "result_dir": str(rd),
    }]).to_csv(man / "stage18b_screening_manifest.csv", index=False)
    (out / "splits").mkdir(parents=True)
    pd.DataFrame([{"split_name":"development","n_rows":1}]).to_csv(out/"splits"/"fold_balance_report.csv", index=False)
    with pytest.raises(RuntimeError, match="missing"):
        analyze_round18(str(out), write_lock=True)


def test_filter_complete_screening():
    df = pd.DataFrame([
        {"architecture_id":"a","architecture_family":"pooled_mlp","omics_mode":"own_plus_summary",
         "transformer_config_id":"","residual_mode":"","n_folds_done":3,"n_folds_with_auc":3,
         "mean_DrugMacro_AUC":0.6,"mean_DrugMacro_AUPRC":0.4,"rank":1},
        {"architecture_id":"b","architecture_family":"pooled_mlp","omics_mode":"none",
         "transformer_config_id":"","residual_mode":"","n_folds_done":2,"n_folds_with_auc":2,
         "mean_DrugMacro_AUC":0.7,"mean_DrugMacro_AUPRC":0.5,"rank":2},
    ])
    out = filter_complete_screening(df)
    assert list(out["architecture_id"]) == ["a"]
