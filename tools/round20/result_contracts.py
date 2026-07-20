"""Round 20 result paths, forbidden selection tokens, and contract helpers."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "result/optimization_runs/round20_unseen_drug_closure"

FORBIDDEN_SELECTION_TOKENS = frozenset({
    "tcga", "tcga_auc", "tcga_auprc", "internal", "internal_auc", "internal_test",
    "external", "external_auc", "integrated5", "integrated5_auc", "posthoc",
})
FORBIDDEN_PATH_FRAGMENTS = frozenset({
    "stage20d_tcga", "internal_test", "integrated5", "posthoc",
})
PLACEHOLDER_PATTERNS = re.compile(
    r"(?i)\b(null|TBD|REPLACE_ME|UNKNOWN|\.\.\.|path/to/)\b"
)

STAGE20A_REQUIRED = (
    "manifest.jsonl",
    "stage20a_dimension_decision.json",
    "reports/stage20a_pairwise.csv",
    "reports/stage20a_candidate_summary.csv",
    "reports/stage20a_seed_summary.csv",
)
STAGE20B_REQUIRED = (
    "manifest.jsonl",
    "stage20b_guardrail_report.json",
    "reports/stage20b_pairwise.csv",
    "reports/stage20b_candidate_summary.csv",
    "reports/stage20b_seed_summary.csv",
)
STAGE20C_REQUIRED = ("final_model_lock.json",)
STAGE20D_REQUIRED_MIN = (
    "stage20d_tcga_preflight.json",
    "stage20d_tcga_summary.json",
    "tcga_metrics.json",
)
STAGE20E_REQUIRED = (
    "RELEASE_MANIFEST.json",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "INFERENCE_GUIDE.md",
    "LIMITATIONS.md",
    "hashes/release_audit.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stage_dir(run_root: Path, stage: str) -> Path:
    mapping = {
        "20-0": run_root / "stage20_0",
        "20A": run_root / "stage20a_dimension",
        "20B": run_root / "stage20b_predictor",
        "20C": run_root / "stage20c_lock",
        "20D": run_root / "stage20d_tcga",
        "20E": run_root / "stage20e_release",
    }
    return mapping[stage]


def count_complete_jobs(jobs_root: Path, *, allow_smoke: bool = False) -> Dict[str, int]:
    complete = failed = pending = 0
    if not jobs_root.is_dir():
        return {"complete": 0, "failed": 0, "pending": 0}
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        status_path = job_dir / "status.json"
        metrics_path = job_dir / "metrics.json"
        if not status_path.is_file():
            pending += 1
            continue
        st = load_json(status_path)
        if st.get("status") == "COMPLETE" and metrics_path.is_file():
            m = load_json(metrics_path)
            if m.get("status") == "COMPLETE" and (allow_smoke or not m.get("smoke")):
                complete += 1
            else:
                failed += 1
        elif st.get("status") == "FAILED":
            failed += 1
        else:
            pending += 1
    return {"complete": complete, "failed": failed, "pending": pending}


def scan_forbidden_selection(payload: Any, *, path: str = "root") -> List[str]:
    hits: List[str] = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            key = str(k).lower()
            if key in FORBIDDEN_SELECTION_TOKENS:
                if not (isinstance(v, (bool, type(None))) and not v):
                    hits.append(f"{path}.{k}")
            if isinstance(v, str) and any(tok in v.lower() for tok in FORBIDDEN_PATH_FRAGMENTS):
                if "stage20d" not in path.lower() and "forbidden_metrics_used" not in key:
                    hits.append(f"{path}.{k}={v}")
            hits.extend(scan_forbidden_selection(v, path=f"{path}.{k}"))
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            hits.extend(scan_forbidden_selection(item, path=f"{path}[{i}]"))
    return hits


def portable_path(path: str, *, run_root_name: str = "round20_unseen_drug_closure") -> str:
    p = str(path).replace("\\", "/")
    if "${ROUND20_RELEASE_ROOT}" in p:
        return p
    marker = f"/{run_root_name}/"
    if marker in p:
        return "${ROUND20_RELEASE_ROOT}/" + p.split(marker, 1)[1]
    if p.startswith("result/optimization_runs/round20_unseen_drug_closure/"):
        return "${ROUND20_RELEASE_ROOT}/" + p.split(run_root_name + "/", 1)[1]
    return p
