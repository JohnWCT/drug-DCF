"""Round 19 OOM runner wrapper around Round 18 process-level retry."""
from __future__ import annotations

from tools.round18_oom_runner import (  # noqa: F401
    dispatch_manifest,
    probe_micro_batch,
    run_single_job_with_oom_retry,
    write_resource_metadata,
)


REQUIRED_JOB_METADATA = [
    "omics_id",
    "drug_representation_id",
    "predictor_id",
    "node_hidden_dim",
    "graph_output_dim",
    "edge_feature_schema",
    "split_strategy",
    "split_seed",
]


def assert_job_metadata(job: dict) -> None:
    missing = [k for k in REQUIRED_JOB_METADATA if k not in job]
    if missing:
        raise KeyError(f"Round19 job missing metadata: {missing}")
