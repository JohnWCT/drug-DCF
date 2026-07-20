from tools.round20.reproduction import run_reproduction_audit, verify_raw_forward_capability


def test_frozen_golden_predictions(synthetic_run_root):
    report = run_reproduction_audit(run_root=synthetic_run_root, strict=False)
    assert report["status"] == "PASS"


def test_raw_encoder_capability():
    cap = verify_raw_forward_capability()
    assert cap["status"] == "PASS"
