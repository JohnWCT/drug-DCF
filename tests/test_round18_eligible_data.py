import json
from pathlib import Path

import pandas as pd

from tools.round18_eligible_data import build_round18_eligible_response, validate_feature_metadata


def test_validate_feature_metadata_own_plus_summary():
    meta = validate_feature_metadata(
        "result/optimization_runs/round17r_18class/features/r13_exp_008/own_plus_summary"
    )
    assert meta["n_trainable_cancer_types"] == 18
    assert meta["uses_legacy_28class_cache"] is False


def test_build_eligible_writes_artifacts(tmp_path):
    outdir = tmp_path / "elig"
    summary = build_round18_eligible_response(
        "data/GDSC2_fitted_dose_response_MaxScreen_raw.csv",
        feature_dir="result/optimization_runs/round17r_18class/features/r13_exp_008/own_plus_summary",
        drug_smiles_path="data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv",
        outdir=str(outdir),
    )
    assert summary["n_eligible_rows"] > 1000
    elig = pd.read_csv(summary["paths"]["eligible"])
    for col in (
        "_row_id",
        "ModelID",
        "DRUG_NAME",
        "Label",
        "has_latent",
        "has_smiles",
        "graph_valid",
    ):
        assert col in elig.columns
    assert Path(summary["paths"]["summary"]).exists()
    meta = json.loads(Path(summary["paths"]["summary"]).read_text())
    assert meta["feature_metadata"]["n_trainable_cancer_types"] == 18
