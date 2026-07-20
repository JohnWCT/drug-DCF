from tools.round20.lock_consistency import verify_final_lock_consistency


def test_final_lock_matches_stage_decisions(synthetic_run_root):
    check = verify_final_lock_consistency(run_root=synthetic_run_root)
    assert check["ok"]
    assert check["context_matches_stage20a"]
    assert check["model_matches_stage20b"]
