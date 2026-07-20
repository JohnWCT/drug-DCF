from tools.round20.result_contracts import portable_path, scan_forbidden_selection


def test_portable_path_rewrites_run_root():
    p = portable_path("result/optimization_runs/round20_unseen_drug_closure/stage20c_lock/final_model_lock.json")
    assert p.startswith("${ROUND20_RELEASE_ROOT}/")


def test_forbidden_selection_detects_tcga_path():
    hits = scan_forbidden_selection({"metrics_path": "result/foo/stage20d_tcga/aggregate_metrics.json"})
    assert hits
