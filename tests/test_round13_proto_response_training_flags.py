#!/usr/bin/env python3
import inspect

import numpy as np

from tools.optimization_runner import build_parser, run_round13_finetune_stage
from tools.prototype_response_features import concat_latent_and_proto_features


def test_mode_none_preserves_latent_dim():
    z = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = concat_latent_and_proto_features(z, {"features": np.zeros(0, dtype=np.float32)})
    assert out.shape == (3,)


def test_mode_increases_input_dim():
    z = np.array([1.0, 2.0], dtype=np.float32)
    out = concat_latent_and_proto_features(z, {"features": np.array([0.1, 0.2], dtype=np.float32)})
    assert out.shape == (4,)


def test_optimization_runner_accepts_round13_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "finetune",
            "--manifest",
            "dummy.csv",
            "--run-dir",
            "result/x",
            "--finetune-config",
            "config/params_finetune_proto_features.json",
            "--round13-mode",
        ]
    )
    assert args.round13_mode is True
    assert args.top10 is None


def test_round13_runner_uses_manifest_model_select_path():
    import tools.optimization_runner as runner

    src = inspect.getsource(runner)
    assert "run_round13_finetune_stage" in src
    assert '_run_one_round13_finetune_job' in src
    assert 'job_row["model_select_path"]' in src
    assert "build_model_select_from_top10" not in inspect.getsource(runner._run_one_round13_finetune_job)
