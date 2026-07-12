from pathlib import Path
from unittest import mock

import pandas as pd

from tools.round18_oom_runner import OOM_EXIT_CODE, dispatch_manifest, run_single_job_with_oom_retry


def _job(tmp_path, micro=512):
    result_dir = tmp_path / "job0"
    return {
        "job_id": "j0",
        "architecture_family": "pooled_mlp",
        "omics_mode": "own_plus_summary",
        "feature_dir": "feat",
        "split_assignment": "split.csv",
        "fold_id": 0,
        "response_data_path": "resp.csv",
        "drug_smiles_path": "smiles.csv",
        "result_dir": str(result_dir),
        "requested_micro_batch": micro,
        "model_seed": 101,
    }


def test_process_level_oom_retry_halves_batch(tmp_path):
    calls = []

    def fake_run(cmd, env=None, stdout=None, stderr=None, cwd=None):
        mb = int(cmd[cmd.index("--micro-batch-size") + 1])
        calls.append(mb)

        class R:
            returncode = OOM_EXIT_CODE if mb >= 256 else 0

        return R()

    with mock.patch("tools.round18_oom_runner.subprocess.run", side_effect=fake_run):
        status = run_single_job_with_oom_retry(
            pipeline="step1_finetune_latent_pipeline_round18_cv.py",
            job=_job(tmp_path),
            micro_batch_candidates=[512, 256, 128],
            target_effective_batch=1024,
            max_retries=4,
            cuda_device="",
        )
    assert calls == [512, 256, 128]
    assert status["status"] == "done"
    assert status["successful_micro_batch"] == 128
    assert status["oom_batch_history"] == [512, 256]


def test_dispatch_manifest_writes_status(tmp_path):
    manifest = tmp_path / "m.csv"
    job = _job(tmp_path)
    pd.DataFrame([job]).to_csv(manifest, index=False)

    def fake_run(cmd, env=None, stdout=None, stderr=None, cwd=None):
        class R:
            returncode = 0

        return R()

    with mock.patch("tools.round18_oom_runner.subprocess.run", side_effect=fake_run):
        summary = dispatch_manifest(
            manifest_path=str(manifest),
            pipeline="step1_finetune_latent_pipeline_round18_cv.py",
            max_jobs_per_gpu=1,
            micro_batch_candidates=[32],
            status_csv=str(tmp_path / "status.csv"),
        )
    assert summary["n_done"] == 1
    assert Path(summary["status_csv"]).exists()
