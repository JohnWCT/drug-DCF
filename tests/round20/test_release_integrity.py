from tools.round20.release_integrity import validate_release_directory, validate_release_manifest


def test_release_has_no_placeholders():
    manifest = {
        "release_status": "LOCKED",
        "artifacts": [],
        "selection": {"context": "C32", "predictor": "B_E3", "drug_encoder": "D0"},
    }
    assert validate_release_manifest(manifest) == []


def test_release_checkpoint_count(synthetic_run_root):
    report = validate_release_directory(
        synthetic_run_root / "stage20e_release", strict=False
    )
    assert report["status"] == "PASS"
    assert report["n_checkpoints"] == 15
