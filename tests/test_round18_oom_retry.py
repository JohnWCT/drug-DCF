from tools.round18_oom_runner import compute_accumulation_steps, probe_micro_batch


def test_accumulation_steps():
    assert compute_accumulation_steps(1024, 256) == 4
    assert compute_accumulation_steps(1024, 128) == 8


def test_oom_probe_synthetic_history():
    def try_fn(batch):
        if batch >= 256:
            raise RuntimeError("CUDA out of memory at batch=%s" % batch)

    result = probe_micro_batch(
        [512, 256, 128, 64, 32],
        target_effective_batch=1024,
        max_retries=4,
        try_fn=try_fn,
    )
    assert result.successful_micro_batch == 128
    assert result.oom_retry_count == 2
    assert result.gradient_accumulation_steps == 8
    assert result.oom_batch_history == [512, 256]
    assert result.effective_batch_size == 1024
