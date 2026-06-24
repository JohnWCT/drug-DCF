import os
import pandas as pd
from tools.round15_config_builder import build_round15_all, resolve_round15b_model_pool
from tools.round15_selection import select_round15_repro_rescue_candidates
from tools.round9_diagnostics_common import load_json, resolve_path

SETTINGS = "config/round15_repro_rescue_settings.json"
OUT = "result/optimization_runs/round15_repro_rescue_test_forced"


def test_model_pool_must_contain_exp008():
    settings = load_json(resolve_path(SETTINGS))
    pool, _ = resolve_round15b_model_pool(settings)
    assert any(p["pool_model_id"] == "exp_008" for p in pool)


def test_finetune_manifest_exp008_modes():
    build_round15_all(SETTINGS, OUT, force=True)
    ft = pd.read_csv(os.path.join(OUT, "manifests/finetune_dispatch_manifest.csv"))
    exp008 = ft[ft["source_model_id"] == "exp_008"]
    assert not exp008[exp008["feature_mode"] == "none"].empty
    assert not exp008[exp008["feature_mode"] == "own_plus_summary"].empty


def test_selection_never_drops_all_exp008():
    rows = []
    for i in range(6):
        rows.append(
            {
                "ID": f"exp_{i+1:03d}",
                "result_folder": f"/tmp/exp_{i+1:03d}",
                "route_id": "exp008_proto_response_route" if i < 4 else "exp035_strong_zonly_route",
                "source_model": "exp_008" if i < 4 else "exp_035",
                "lambda_tumor_var": 0.0 if i == 0 else 1e-5,
                "lambda_tumor_cov": 0.0 if i == 0 else 1e-5,
                "kmeans_ari": 0.3 if i == 3 else 0.55,
                "score_total": 0.5,
            }
        )
    pool = pd.DataFrame(rows)
    top, info = select_round15_repro_rescue_candidates(pool, pool, top_k=3)
    assert info["exp008_route_included"]
