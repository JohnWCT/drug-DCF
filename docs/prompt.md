# Automated Development Prompt for drug-DCF Optimization Agents

## 0. Highest-priority instruction

This project must be developed as a fully automated implementation task. The development agents must not depend on manual participation during coding, testing, refactoring, or validation.

If any implementation detail is ambiguous, the agents must resolve it by following this priority order:

1. `docs/proposal.md`
2. `docs/design.md`
3. the latest `JohnWCT/drug-DCF` repository behavior
4. backward compatibility with existing command-line interfaces
5. the least invasive implementation that satisfies the design

Agents must not stop and ask the user for clarification during implementation. They must document any automatic decision in the final implementation report.

The only exception is safety or environment failure that makes execution technically impossible, such as missing CUDA, missing data files, or an unavailable Docker/DAPL runtime. In that case, the agent must still complete all code changes, unit tests, static checks, and smoke-test scaffolding, then clearly report the blocked runtime step and the exact command the user should run in the target environment.

---

## 1. Source documents and project context

The implementation must be based on the finalized documents:

```text
docs/proposal.md
docs/design.md
```

The target repository is:

```text
https://github.com/JohnWCT/drug-DCF
```

The implementation must assume the latest repository state as the starting point. The current optimization plan is not a generic machine-learning pipeline. It is specifically designed for the current `drug-DCF` codebase.

The first executable optimization round must implement:

```text
GAN-stage source+target cancer prototype InfoNCE for VAEwC
```

The first round must not implement:

```text
TransDRP-like sigmoid InfoMax
EMA prototypes
memory-bank contrastive learning
DDP
multi-GPU scheduling
full AEwC sweep
full CVAE sweep
VAE reparameterization changes
large framework refactor
```

---

## 2. Final implementation target

The agent team must implement an automated full pipeline:

```text
config generation
→ VAEwC pretrain sweep
→ visualization/filtering
→ Top-10 candidate selection
→ finetune mini-grid dispatch
→ downstream aggregation
→ final reports
```

The first full sweep contains 72 pretrain configurations:

| Parameter | Values |
|---|---|
| `lambda_proto` | `[0, 0.03, 0.1, 0.3]` |
| `proto_temperature` | `[0.1, 0.2, 0.5]` |
| `proto_start_epoch` | `[5, 10, 30]` |
| `proto_full_epoch` | `[30, 50]` |
| `lambda_adv` | `[1.0]` |
| `gan_gen_update_interval` | `[5]` |

Selection rule:

```text
Top-10 = best 8 ranked candidates + best 2 lambda_proto=0 controls
```

Downstream workload:

```text
Top-10 selected pretrain candidates × 4 params_finetune_mini.json combinations = 40 finetune jobs
```

Hardware assumption:

```text
single A6000 Ada GPU
single-GPU sequential queue
no DDP
no multi-GPU dispatcher
```

---

## 3. Role definitions

## 3.1 Main Agent

The Main Agent is responsible for overall coordination and must act as the autonomous technical lead.

Responsibilities:

1. Read `docs/proposal.md` and `docs/design.md` before making changes.
2. Inspect the latest repository structure and existing CLI behavior.
3. Build an implementation task graph from the design.
4. Assign concrete low-coupling modules to Sub Agents.
5. Ensure all changes preserve existing CLI compatibility.
6. Ensure every business-logic change is accompanied by tests.
7. Run or prepare all required validation commands.
8. Maintain a development checklist and mark tasks as complete only after tests pass.
9. Generate the final implementation report.
10. Ensure no Sub Agent leaves unresolved TODOs or manual steps.

The Main Agent must not ask the user to manually decide implementation details. If there is a conflict between documents and repository behavior, it must prefer backward-compatible implementation and document the decision.

## 3.2 Sub Agents

Each Sub Agent is responsible for one concrete module. Sub Agents must implement only the assigned module and its tests.

Required Sub Agent domains:

| Sub Agent | Responsibility |
|---|---|
| Prototype InfoNCE Agent | Implement batch-based source+target prototype InfoNCE and tests |
| VAEwC Integration Agent | Integrate InfoNCE into GAN-stage generator/encoder update |
| Config Generator Agent | Generate the 72 pretrain configs and manifest |
| Sequential Runner Agent | Implement single-GPU resumable queue and status tracking |
| Selection Agent | Wrap visualization/filtering and enforce Top-10 with controls |
| Finetune Dispatcher Agent | Dispatch Top-10 × 4 mini-grid finetune jobs |
| Aggregation Agent | Wrap downstream aggregation and produce ranking outputs |
| Report Agent | Produce CSV, JSON, Markdown, and plot/report summaries |
| Test Agent | Add and run unit tests, smoke tests, lint, format, and compile checks |

Sub Agents must not modify unrelated modules unless necessary for integration. If a change touches shared code, the Sub Agent must add regression tests to prove old behavior remains valid.

---

## 4. Core anti-error mechanism

## 4.1 Test-first/synchronized-test rule

Every Sub Agent must generate tests together with code changes.

For every new or modified business-logic function, class, conditional branch, error-handling path, or CLI behavior, the Sub Agent must add or update corresponding tests.

The user does not require 100% diff coverage. However, the following rules are mandatory:

1. All new core modules must have meaningful unit tests.
2. All new CLI entry points must have smoke tests or argument-parsing tests.
3. All new config parsing logic must have valid-input and invalid-input tests.
4. `lambda_proto=0` must be tested as a no-InfoNCE backward-compatible path.
5. InfoNCE edge cases must be tested.
6. Manifest resume/status behavior must be tested.
7. Selection logic must be tested for mandatory `lambda_proto=0` controls.
8. Tests must be runnable in a lightweight environment without launching the full 72-run experiment.

Recommended coverage targets:

```text
new utility modules: high coverage, preferably >= 85%
new business-critical modules: high coverage, preferably >= 90%
legacy repository code: do not chase global coverage unless required
```

Do not block implementation solely because global repository coverage is below a target. Focus coverage on new and modified logic.

## 4.2 Mandatory validation tools

Default validation tools:

```bash
python -m compileall .
ruff check .
black --check .
pytest tests/ -q
pytest tests/ --cov=. --cov-report=term-missing
```

The agent may add or update:

```text
tests/requirements-dev.txt
pyproject.toml
tests/pytest.ini
```

The exact tool versions and commands may be adjusted for the Docker DAPL environment if needed, but the validation intent must remain unchanged:

```text
syntax check
lint check
format check
unit tests
coverage report
smoke tests
```

Do not require `mypy` in the first implementation unless the existing repository already supports it cleanly. The agent may add type hints where helpful, but must not spend implementation time converting the legacy project into a fully typed codebase.

---

## 5. Required implementation modules

## 5.1 Prototype InfoNCE module

Create:

```text
tools/proto_infonce.py
```

Required public API:

```python
def compute_batch_prototype_infonce(
    z_source,
    y_source,
    z_target,
    y_target,
    num_classes,
    temperature=0.2,
    min_samples_per_class=1,
):
    """
    Compute batch-based source+target cancer prototype InfoNCE.

    Returns:
        loss: scalar torch.Tensor
        metrics: dict
    """
```

Required behavior:

1. Concatenate source and target shared latent vectors.
2. Concatenate source and target cancer labels.
3. Compute prototypes from the current batch.
4. Normalize samples and prototypes.
5. Compute cosine similarity logits divided by temperature.
6. Use cross-entropy against cancer labels.
7. Return scalar loss and diagnostics.

Required metrics:

```text
proto_loss
proto_acc
proto_valid_class_count
proto_valid_sample_count
proto_mean_positive_similarity
proto_mean_negative_similarity
proto_margin
proto_valid
```

Required edge-case behavior:

| Case | Behavior |
|---|---|
| Class absent from batch | Exclude class from prototype matrix |
| Sample label has no valid prototype | Exclude that sample from InfoNCE |
| Fewer than 2 valid classes | Return zero loss with `proto_valid=false` |
| Non-positive temperature | Raise `ValueError` |
| NaN/Inf in inputs or logits | Raise error and mark run failed upstream |
| `lambda_proto=0` | Must not affect training behavior |

Required tests:

1. Standard multi-class source+target case.
2. Missing class in batch.
3. Single valid class returns zero loss.
4. `temperature <= 0` raises.
5. Loss is differentiable when valid.
6. Metrics contain all required keys.
7. CPU-only synthetic tensor test.

---

## 5.2 VAEwC integration

Modify:

```text
pretrain_VAEwC.py
```

Do not rewrite the entire training pipeline. Use minimal integration.

Required additions:

1. Add config parsing for:

```text
lambda_proto
proto_temperature
proto_start_epoch
proto_full_epoch
lambda_adv
min_proto_samples_per_class
```

2. Add schedule helper:

```python
def get_lambda_proto_eff(gan_epoch, param):
    ...
```

3. Add optional `lambda_adv_eff` support while preserving default behavior:

```text
lambda_adv defaults to 1.0
```

4. In GAN-stage generator/encoder update, compute `proto_infonce_loss` only when:

```text
lambda_proto_eff > 0
```

5. Add the new term:

```text
lambda_proto_eff * proto_infonce_loss
```

6. Log new metrics into GAN generator loss CSV/report:

```text
lambda_proto_eff
lambda_adv_eff
proto_loss
proto_acc
proto_valid_class_count
proto_valid_sample_count
proto_mean_positive_similarity
proto_mean_negative_similarity
proto_margin
proto_valid
```

Backward compatibility requirements:

1. If config does not contain new fields, behavior must match current baseline as closely as possible.
2. `lambda_proto=0` must be a no-op for loss and gradients.
3. Existing CLI arguments must remain valid.
4. Existing output files must remain readable by existing visualization tools.

Required tests:

1. `lambda_proto=0` path does not call or affect InfoNCE.
2. `lambda_proto>0` includes InfoNCE term in loss.
3. Schedule returns expected values before/start/full epochs.
4. New log keys appear in synthetic/mock GAN update output.
5. Existing config without prototype keys still resolves successfully.

---

## 5.3 Config generator

Create:

```text
tools/optimization_config_generator.py
```

Required input:

```text
config/pretrain_sweeps/vaewc_proto_infonce_round1.json
```

Required output:

```text
config/generated/vaewc_proto_infonce_round1/*.json
result/optimization_runs/{run_id}/manifests/pretrain_sweep_manifest.csv
```

Required sweep:

```json
{
  "lambda_proto": [0, 0.03, 0.1, 0.3],
  "proto_temperature": [0.1, 0.2, 0.5],
  "proto_start_epoch": [5, 10, 30],
  "proto_full_epoch": [30, 50],
  "lambda_adv": [1.0],
  "gan_gen_update_interval": [5]
}
```

Required behavior:

1. Generate exactly 72 pretrain configs.
2. Preserve all baseline config fields unless explicitly swept.
3. Add metadata fields for traceability:

```text
run_id
job_id
sweep_name
generated_at
source_base_config
```

4. Write a manifest with job status initialized to `pending`.
5. Do not overwrite existing configs unless `--force` is passed.

Required tests:

1. Generates exactly 72 configs.
2. Includes `lambda_proto=0` controls.
3. Does not mutate the input base config object.
4. Manifest schema is correct.
5. `--force` behavior is correct.

---

## 5.4 Single-GPU sequential runner

Create:

```text
tools/optimization_runner.py
```

Required behavior:

1. Run one job at a time on the single configured GPU.
2. Use a manifest-based queue.
3. Support resume:

```text
pending → running → success/failed/skipped
```

4. Write per-job stdout/stderr logs.
5. Write per-job status JSON.
6. Avoid running completed jobs unless `--rerun-completed` is passed.
7. Expose dry-run mode that prints commands but does not execute them.
8. Support smoke-test mode with shortened epochs and synthetic or tiny config where feasible.

Required commands must include at least:

```text
pretrain
select
finetune
aggregate
full
```

Example CLI shape:

```bash
python tools/optimization_runner.py pretrain \
  --manifest result/optimization_runs/{run_id}/manifests/pretrain_sweep_manifest.csv \
  --device cuda:0
```

The agent may adjust exact CLI names if they are documented and tested.

Required tests:

1. Pending job selection.
2. Successful job status transition.
3. Failed job status transition.
4. Resume skips completed jobs.
5. Dry-run produces commands without execution.
6. Missing config path is handled as failed with clear error.

---

## 5.5 Selection wrapper

Use existing:

```text
visualize_vaewc_results.py
config/visualize_vaewc_filter.json
```

Create wrapper logic either inside `optimization_runner.py` or a small helper module.

Required output:

```text
result/optimization_runs/{run_id}/selection/pretrain_filtered_candidates.csv
result/optimization_runs/{run_id}/selection/pretrain_top10.csv
result/optimization_runs/{run_id}/selection/model_select.csv
result/optimization_runs/{run_id}/reports/pretrain_selection_report.md
```

Required selection rule:

```text
Top-10 = best 8 ranked candidates + best 2 lambda_proto=0 controls
```

Required behavior:

1. Apply `visualize_vaewc_filter.json` first.
2. Preserve the existing visualization script's outputs.
3. Rank remaining runs by deconfounding quality as defined by the existing score/ranking outputs.
4. Enforce two no-InfoNCE controls inside Top-10 if valid controls exist.
5. If fewer than two valid controls exist, include all valid controls and document the shortage.

Required tests:

1. Top-10 includes exactly two controls when available.
2. Handles fewer than two controls.
3. Excludes filter-failed rows.
4. Produces stable deterministic ranking for ties.

---

## 5.6 Finetune dispatcher

Use existing:

```text
step1_finetune_latent_pipeline_All_split.py
config/params_finetune_mini.json
```

Required behavior:

1. Read `pretrain_top10.csv`.
2. Generate 40 finetune jobs:

```text
10 selected pretrain candidates × 4 mini-grid configs
```

3. Run one finetune job at a time on the single GPU.
4. Track status in:

```text
result/optimization_runs/{run_id}/manifests/finetune_dispatch_manifest.csv
```

5. Write per-job logs and status JSON.
6. Support resume and dry-run.

Required tests:

1. Top-10 input generates 40 jobs.
2. Missing pretrain candidate path is marked failed.
3. Resume skips completed finetune jobs.
4. Dry-run emits commands only.

---

## 5.7 Aggregation and report generation

Use existing:

```text
aggregate_pretrain_tcga_scores.py
```

Primary final ranking metric:

```text
Global_TCGA_AUC_mean
```

Create or update:

```text
tools/optimization_report.py
```

Required final outputs:

```text
result/optimization_runs/{run_id}/aggregate/aggregate_scores.csv
result/optimization_runs/{run_id}/reports/final_selection_report.md
result/optimization_runs/{run_id}/reports/run_summary.json
```

Final report must include:

1. Best downstream pretrain candidate.
2. Top downstream candidates by `Global_TCGA_AUC_mean`.
3. Comparison of `lambda_proto=0` controls vs `lambda_proto>0` runs.
4. Pretrain deconfounding metrics.
5. Tumor preservation metrics.
6. Prototype metrics.
7. Failed/partial run summary.
8. Artifact retention summary.
9. Exact commands used.
10. Environment summary.

Required tests:

1. Aggregation parser handles expected CSV columns.
2. Missing optional metric columns do not crash report generation.
3. Final report is generated from synthetic aggregate input.
4. Control-vs-InfoNCE comparison is included when both exist.

---

## 5.8 Artifact retention

Create:

```text
tools/artifact_retention.py
```

Required policy:

| Run type | Keep | Remove |
|---|---|---|
| Top-10 | checkpoint, latent, plots, metrics, config, report | none by default |
| Selected `lambda_proto=0` controls | checkpoint, latent, metrics, config, report | none by default |
| Non-Top-10 success | config, metrics, ranking row, report | large checkpoints, large latent files |
| Failed | config, status, stdout/stderr, partial metrics | large checkpoints, large latent files |

Required behavior:

1. Dry-run mode prints planned deletions.
2. Never delete manifests, reports, configs, or CSV summaries.
3. Never delete Top-10 artifacts.
4. Log all deletions to `artifact_retention_log.csv`.

Required tests:

1. Top-10 files are protected.
2. Non-Top-10 large files are selected for deletion.
3. Dry-run does not delete.
4. Deletion log is written.

---

## 6. Development workflow for agents

The Main Agent must execute this workflow:

1. Inspect repository.
2. Read `docs/proposal.md` and `docs/design.md`.
3. Create implementation checklist.
4. Create or update dev dependencies for tests/lint.
5. Assign modules to Sub Agents.
6. For each module:
   - implement code
   - implement tests
   - run targeted tests
   - update documentation if needed
7. Run full code validation checks.
8. Run smoke-test pipeline commands.
9. Generate final implementation report.

Sub Agents must execute this workflow:

1. Read assigned module requirements.
2. Inspect existing related code.
3. Implement minimal low-coupling change.
4. Add tests simultaneously.
5. Run targeted tests.
6. Report changed files, tests added, and validation result.

---

## 7. Validation plan

## 7.1 Code validation

The development must pass, or provide Docker/DAPL-adjusted equivalent commands for:

```bash
python -m compileall .
ruff check .
black --check .
pytest tests/ -q
pytest tests/ --cov=. --cov-report=term-missing
```

If the Docker DAPL environment requires different paths, environment variables, or package versions, the Main Agent must document the adjusted commands.

## 7.2 Smoke tests

The Main Agent must provide smoke-test commands that do not run the full 72 pretrain jobs.

Smoke tests should include:

1. Config generation with a tiny sweep.
2. One mock or short-epoch pretrain command if data/environment supports it.
3. Selection logic on synthetic or small CSV input.
4. Finetune dispatch dry-run for a synthetic Top-10 file.
5. Aggregation/report generation from synthetic output.

## 7.3 Full experiment commands

The Main Agent must provide exact commands for the full intended experiment:

```text
72 VAEwC pretrain runs
Top-10 selection with 2 controls
40 finetune mini-grid jobs
aggregation
final report generation
```

The full 72-run experiment does not need to be executed during code validation unless the user explicitly requests it. The code must still make the full run executable.

---

## 8. Backward compatibility requirements

1. Existing `pretrain_VAEwC.py` CLI must remain valid.
2. Existing configs without prototype fields must still run.
3. `lambda_proto=0` must produce behavior equivalent to no InfoNCE.
4. Existing visualization, finetune, and aggregate scripts should be wrapped, not rewritten.
5. Output files currently consumed by downstream scripts must retain compatible schemas.
6. Any new columns must be additive.
7. Existing result directories must not be destroyed.
8. Artifact deletion must require explicit retention command or explicit `--apply`; default must be dry-run where possible.

---

## 9. Environment and Docker/DAPL note

The default development checks are defined in this prompt, but the final user environment may be a Docker DAPL environment.

Agents may adjust:

1. dependency installation commands
2. CUDA device selection
3. path mounts
4. Python executable name
5. shell activation commands
6. package versions

Agents must not adjust the scientific design unless required by the repository code. Any environment-specific change must be documented in the final report.

---

## 10. Final deliverables

At the end of implementation, the Main Agent must produce:

1. List of modified files.
2. List of added files.
3. Summary of implemented modules.
4. Summary of tests added.
5. Validation command results.
6. Smoke-test command results or exact reason why runtime smoke test was blocked.
7. Full experiment commands.
8. Known limitations.
9. Automatic decisions made from ambiguity.
10. Next-step recommendations for optional future ablations.

Expected important files:

```text
tools/proto_infonce.py
tools/pretrain_proto_metrics.py
tools/optimization_config_generator.py
tools/optimization_runner.py
tools/optimization_report.py
tools/artifact_retention.py
tests/
config/pretrain_sweeps/vaewc_proto_infonce_round1.json
```

Existing files expected to be minimally modified:

```text
pretrain_VAEwC.py
pretrain_AEwC.py, only if smoke-test compatibility requires it
pretrain_CVAE.py, only if smoke-test compatibility requires it
```

---

## 11. Hard prohibitions

The agents must not:

1. Ask the user for implementation clarification during coding.
2. Leave TODOs that block execution.
3. Require manual edits to generated configs.
4. Rewrite the entire training pipeline when a minimal integration is sufficient.
5. Implement TransDRP-like sigmoid InfoMax in round 1.
6. Implement EMA prototypes in round 1.
7. Implement memory bank contrastive learning in round 1.
8. Implement DDP or multi-GPU scheduling in round 1.
9. Change VAE reparameterization behavior in round 1.
10. Break existing CLI behavior.
11. Delete large artifacts unless the retention policy is explicitly invoked.
12. Run destructive cleanup without dry-run support.

---

## 12. Definition of done

Implementation is complete only when:

1. Prototype InfoNCE is implemented and tested.
2. VAEwC GAN-stage integration is implemented and tested.
3. 72-config generation is implemented and tested.
4. Single-GPU sequential runner is implemented and tested.
5. Top-10 selection with two controls is implemented and tested.
6. Finetune dispatch dry-run/scheduling is implemented and tested.
7. Aggregation/report generation is implemented and tested.
8. Artifact retention policy is implemented and tested.
9. Default validation tools pass or Docker/DAPL-compatible alternatives are documented.
10. Smoke-test commands are provided.
11. Full experiment commands are provided.
12. Final implementation report is written.

