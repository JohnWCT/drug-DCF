"""Round 18 OOM-safe micro-batch probing and job dispatch helpers."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence


OOM_EXIT_CODE = 42


@dataclass
class OOMProbeResult:
    requested_micro_batch: int
    successful_micro_batch: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    oom_retry_count: int
    oom_batch_history: List[int] = field(default_factory=list)
    amp_enabled: bool = True
    status: str = "ok"
    detail: str = ""


def compute_accumulation_steps(target_effective_batch: int, micro_batch: int) -> int:
    if micro_batch <= 0:
        raise ValueError("micro_batch must be > 0")
    return int(math.ceil(target_effective_batch / micro_batch))


def probe_micro_batch(
    candidates: Sequence[int],
    *,
    target_effective_batch: int = 1024,
    max_retries: int = 4,
    amp_enabled: bool = True,
    try_fn: Optional[Callable[[int], None]] = None,
) -> OOMProbeResult:
    """
    Try micro-batch sizes in order. try_fn(batch) should raise RuntimeError('CUDA out of memory')
    or a custom exception with 'out of memory' in message to simulate OOM.
    """
    history: List[int] = []
    attempts = 0
    last_requested = int(candidates[0]) if candidates else 0

    for batch in candidates:
        if attempts > max_retries:
            break
        last_requested = int(batch)
        attempts += 1
        if try_fn is None:
            # dry probe: accept first candidate
            accum = compute_accumulation_steps(target_effective_batch, batch)
            return OOMProbeResult(
                requested_micro_batch=int(candidates[0]),
                successful_micro_batch=int(batch),
                gradient_accumulation_steps=accum,
                effective_batch_size=int(batch * accum),
                oom_retry_count=0,
                oom_batch_history=[],
                amp_enabled=amp_enabled,
                status="ok",
                detail="dry_probe_no_try_fn",
            )
        try:
            try_fn(int(batch))
            accum = compute_accumulation_steps(target_effective_batch, batch)
            return OOMProbeResult(
                requested_micro_batch=int(candidates[0]),
                successful_micro_batch=int(batch),
                gradient_accumulation_steps=accum,
                effective_batch_size=int(batch * accum),
                oom_retry_count=len(history),
                oom_batch_history=list(history),
                amp_enabled=amp_enabled,
                status="ok",
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                history.append(int(batch))
                continue
            return OOMProbeResult(
                requested_micro_batch=int(candidates[0]),
                successful_micro_batch=-1,
                gradient_accumulation_steps=-1,
                effective_batch_size=-1,
                oom_retry_count=len(history),
                oom_batch_history=list(history),
                amp_enabled=amp_enabled,
                status="hard_failure",
                detail=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return OOMProbeResult(
                requested_micro_batch=int(candidates[0]),
                successful_micro_batch=-1,
                gradient_accumulation_steps=-1,
                effective_batch_size=-1,
                oom_retry_count=len(history),
                oom_batch_history=list(history),
                amp_enabled=amp_enabled,
                status="hard_failure",
                detail=str(exc),
            )

    return OOMProbeResult(
        requested_micro_batch=last_requested,
        successful_micro_batch=-1,
        gradient_accumulation_steps=-1,
        effective_batch_size=-1,
        oom_retry_count=len(history),
        oom_batch_history=list(history),
        amp_enabled=amp_enabled,
        status="oom_exhausted",
        detail="all micro-batch candidates failed",
    )


def write_resource_metadata(outdir: str, probe: OOMProbeResult, extra: Optional[dict] = None) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    payload = asdict(probe)
    if extra:
        payload.update(extra)
    (out / "runtime_resource_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history = {
        "oom_retry_count": probe.oom_retry_count,
        "oom_batch_history": probe.oom_batch_history,
        "status": probe.status,
    }
    (out / "oom_retry_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return payload


def detect_gpu_job_slots(jobs_per_gpu: int = 1) -> int:
    try:
        import torch

        n = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        n = 0
    if n <= 0:
        return 1
    return max(1, n * max(1, jobs_per_gpu))
