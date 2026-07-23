from biocda.validation.round24_gate import rank_passing_candidates, weighted_score


def test_weighted_cannot_rescue_failed_candidate():
    cands = [
        {
            "candidate_id": "weighted_high_but_fail",
            "per_target_fold_mean_auc": {
                "gdsc_intersect13": 0.40,
                "tcga_only3": 0.90,
                "dapl": 0.90,
                "aacdr_gdsc_intersect": 0.90,
                "aacdr_tcga_only": 0.90,
            },
            "gate": {"status": "NO_LOCK"},
        },
        {
            "candidate_id": "all_pass_lower_weight",
            "per_target_fold_mean_auc": {
                "gdsc_intersect13": 0.53,
                "tcga_only3": 0.56,
                "dapl": 0.54,
                "aacdr_gdsc_intersect": 0.56,
                "aacdr_tcga_only": 0.45,
            },
            "gate": {"status": "PASS"},
        },
    ]
    weights = {
        "gdsc_intersect13": 5,
        "tcga_only3": 4,
        "dapl": 3,
        "aacdr_gdsc_intersect": 2,
        "aacdr_tcga_only": 1,
    }
    ranked = rank_passing_candidates(cands, target_priority=list(weights), target_weights=weights)
    assert len(ranked) == 1
    assert ranked[0]["candidate_id"] == "all_pass_lower_weight"
