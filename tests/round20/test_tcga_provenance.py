from tools.round20.tcga_provenance import audit_tcga_predictions, recalculate_tcga_metrics


def test_tcga_ensemble_recalculates(synthetic_run_root):
    audit = audit_tcga_predictions(run_root=synthetic_run_root)
    assert audit["status"] == "PASS"


def test_tcga_metrics_recalculate(synthetic_run_root):
    metrics = recalculate_tcga_metrics(run_root=synthetic_run_root)
    assert metrics["status"] == "PASS"
