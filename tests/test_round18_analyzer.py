from pathlib import Path

import pandas as pd

from tools.analyze_round18 import analyze_round18, rank_screening_architectures, select_formal_candidates


def test_rank_screening_mean():
    df = pd.DataFrame(
        [
            {
                'architecture_id': 'a',
                'architecture_family': 'pooled_mlp',
                'omics_mode': 'own_plus_summary',
                'transformer_config_id': '',
                'residual_mode': '',
                'fold_id': 0,
                'status': 'done',
                'DrugMacro_AUC': 0.6,
                'DrugMacro_AUPRC': 0.4,
                'Global_AUC': 0.55,
            },
            {
                'architecture_id': 'a',
                'architecture_family': 'pooled_mlp',
                'omics_mode': 'own_plus_summary',
                'transformer_config_id': '',
                'residual_mode': '',
                'fold_id': 1,
                'status': 'done',
                'DrugMacro_AUC': 0.7,
                'DrugMacro_AUPRC': 0.5,
                'Global_AUC': 0.6,
            },
            {
                'architecture_id': 'b',
                'architecture_family': 'pooled_transformer',
                'omics_mode': 'own_plus_summary',
                'transformer_config_id': 'P2',
                'residual_mode': '',
                'fold_id': 0,
                'status': 'done',
                'DrugMacro_AUC': 0.5,
                'DrugMacro_AUPRC': 0.3,
                'Global_AUC': 0.5,
            },
        ]
    )
    ranking = rank_screening_architectures(df)
    assert ranking.iloc[0]['architecture_id'] == 'a'
    assert abs(ranking.iloc[0]['mean_DrugMacro_AUC'] - 0.65) < 1e-9
    cands = select_formal_candidates(ranking)
    assert any(c['architecture_id'] == 'a' for c in cands)


def test_analyze_writes_reports(tmp_path):
    out = tmp_path / 'r18'
    man = out / 'manifests'
    man.mkdir(parents=True)
    pd.DataFrame(
        columns=[
            'job_id', 'stage', 'architecture_id', 'architecture_family', 'omics_mode',
            'transformer_config_id', 'residual_mode', 'fold_id', 'result_dir',
            'response_data_path', 'feature_dir', 'split_assignment', 'drug_smiles_path',
        ]
    ).to_csv(man / 'stage18b_screening_manifest.csv', index=False)
    (out / 'splits').mkdir(parents=True)
    pd.DataFrame([{'split_name': 'development', 'n_rows': 1}]).to_csv(
        out / 'splits' / 'fold_balance_report.csv', index=False
    )
    result = analyze_round18(str(out), write_lock=False)
    assert Path(result['jobs']).exists()
    assert Path(result['ranking']).exists()
