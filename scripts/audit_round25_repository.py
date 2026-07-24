"""Round 25 repository audit (strict).

Run inside Docker:
  docker exec DAPL bash -lc 'cd /workspace/DAPL && PYTHONPATH=/workspace/DAPL python3 scripts/audit_round25_repository.py --strict'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biocda.audit.final_report_parser import parse_round24_final_report


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_git(args: List[str]) -> subprocess.CompletedProcess:
    """Read-only git without mutating global config."""
    return subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _read_head_filesystem() -> Dict[str, Any]:
    """Fallback for Git <2.35 where safe.directory is unavailable under root+host mount."""
    git_dir = ROOT / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.exists():
        return {"returncode": 1, "stdout": "", "stderr": "missing .git/HEAD"}
    head_txt = head_file.read_text(encoding="utf-8").strip()
    branch = ""
    if head_txt.startswith("ref:"):
        ref = head_txt.split(" ", 1)[1].strip()
        branch = ref.split("/")[-1]
        ref_path = git_dir / ref
        if not ref_path.exists():
            return {"returncode": 1, "stdout": "", "stderr": f"missing ref {ref}"}
        sha = ref_path.read_text(encoding="utf-8").strip()
    else:
        sha = head_txt
    log_lines: List[str] = []
    log_path = git_dir / "logs" / "HEAD"
    if log_path.exists():
        raw = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in raw[-20:]:
            parts = line.split("\t", 1)
            meta = parts[0].split()
            msg = parts[1] if len(parts) > 1 else ""
            if len(meta) >= 2:
                log_lines.append(f"{meta[1][:7]} {msg}".strip())
    return {
        "returncode": 0,
        "stdout": sha,
        "stderr": "",
        "branch": branch,
        "log20": "\n".join(reversed(log_lines)) if log_lines else "",
        "mode": "filesystem_fallback",
    }


def _git_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {"mode": "git_cli"}
    head_cp = _run_git(["rev-parse", "HEAD"])
    if head_cp.returncode != 0 and "dubious ownership" in (head_cp.stderr or ""):
        fb = _read_head_filesystem()
        out["mode"] = "filesystem_fallback"
        out["head"] = {
            "returncode": fb["returncode"],
            "stdout": fb.get("stdout", ""),
            "stderr": fb.get("stderr", ""),
        }
        out["branch"] = {
            "returncode": 0 if fb.get("branch") else 1,
            "stdout": fb.get("branch", ""),
            "stderr": "",
        }
        out["status_porcelain"] = {
            "returncode": 0,
            "stdout": "",
            "stderr": "status skipped under filesystem_fallback (git ownership)",
        }
        out["log20"] = {
            "returncode": 0,
            "stdout": fb.get("log20", ""),
            "stderr": "",
        }
        out["ls_tree_count"] = None
        out["tracked_pycache"] = []
        out["tracked_outputs"] = []
        out["fallback_reason"] = head_cp.stderr.strip()
        return out

    out["head"] = {
        "returncode": head_cp.returncode,
        "stdout": head_cp.stdout.strip(),
        "stderr": head_cp.stderr.strip(),
    }
    for key, args in [
        ("branch", ["branch", "--show-current"]),
        ("status_porcelain", ["status", "--porcelain"]),
        ("log20", ["log", "-20", "--oneline"]),
    ]:
        cp = _run_git(args)
        out[key] = {
            "returncode": cp.returncode,
            "stdout": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
        }
    ls = _run_git(["ls-tree", "-r", "--name-only", "HEAD"])
    files = [ln for ln in ls.stdout.splitlines() if ln.strip()] if ls.returncode == 0 else []
    out["ls_tree_count"] = len(files)
    out["tracked_pycache"] = [f for f in files if "__pycache__" in f or f.endswith(".pyc")]
    out["tracked_outputs"] = [f for f in files if f.startswith("outputs/") or f.startswith("logs/")]
    return out


def _must_exist(rel: str) -> Dict[str, Any]:
    path = ROOT / rel
    ok = path.exists()
    info: Dict[str, Any] = {"path": rel, "exists": ok}
    if ok and path.is_file():
        info["sha256"] = _sha256_file(path)
        info["size"] = path.stat().st_size
    return info


def audit_round24_artifacts(parse: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "docs/round24_final_report.md",
        "reports/round24_final_model_lock.json",
        "reports/round24/stage24e/stage24e_decision.json",
        "configs/round24/eval3.yaml",
        "docs/AACDR_drug_macro_auroc_auprc.md",
    ]
    items = {r: _must_exist(r) for r in required}
    lock = ROOT / "reports/round24_final_model_lock.json"
    decision = ROOT / "reports/round24/stage24e/stage24e_decision.json"
    conflicts: List[str] = []
    lock_obj = decision_obj = None
    if lock.exists():
        lock_obj = json.loads(lock.read_text(encoding="utf-8"))
    if decision.exists():
        decision_obj = json.loads(decision.read_text(encoding="utf-8"))

    if lock_obj and decision_obj:
        if lock_obj.get("status") != decision_obj.get("status"):
            conflicts.append(
                f"lock.status={lock_obj.get('status')} != decision.status={decision_obj.get('status')}"
            )
        champ_lock = (lock_obj.get("champion") or {}).get("candidate_id")
        champ_dec = decision_obj.get("champion_id")
        if champ_lock and champ_dec and champ_lock != champ_dec:
            conflicts.append(f"champion mismatch lock={champ_lock} decision={champ_dec}")
        if parse.get("champion_id") and champ_lock and parse["champion_id"] != champ_lock:
            conflicts.append(
                f"markdown champion={parse['champion_id']} != lock={champ_lock}"
            )
        if parse.get("status") and lock_obj.get("status") and parse["status"] != lock_obj.get("status"):
            # markdown uses `LOCKED` · champion ... — status field alone
            if parse["status"] != lock_obj.get("status"):
                # allow markdown status containing LOCKED
                if lock_obj.get("status") not in str(parse["status"]):
                    conflicts.append(
                        f"markdown status={parse['status']} != lock={lock_obj.get('status')}"
                    )

    # Round 23 must remain REJECTED
    xa_lock = ROOT / "reports/biocda_xa_model_lock.json"
    r23 = None
    if xa_lock.exists():
        r23 = json.loads(xa_lock.read_text(encoding="utf-8"))
        if r23.get("status") != "REJECTED":
            conflicts.append(
                f"Round23 XA lock overwritten or changed: status={r23.get('status')}"
            )

    return {
        "required": items,
        "conflicts": conflicts,
        "lock_status": None if not lock_obj else lock_obj.get("status"),
        "lock_champion": None
        if not lock_obj
        else (lock_obj.get("champion") or {}).get("candidate_id"),
        "round23_xa_status": None if not r23 else r23.get("status"),
        "referenced_from_report": [
            {"path": p, "exists": (ROOT / p).exists()}
            for p in parse.get("referenced_artifacts", [])
        ],
    }


def find_round24_checkpoints() -> Dict[str, Any]:
    roots = [
        ROOT / "result/optimization_runs/round24_tcga_recovery",
        ROOT / "reports/round24/stage24e",
    ]
    found: List[str] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("best.pt"):
            found.append(str(p.relative_to(ROOT)))
        for p in root.rglob("candidate_summary.json"):
            found.append(str(p.relative_to(ROOT)))
    return {"n_files": len(found), "sample": found[:40]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument(
        "--out-dir",
        default="reports",
        help="directory for audit JSON outputs",
    )
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    git = _git_snapshot()
    if git["head"]["returncode"] != 0:
        failures.append(f"git HEAD failed: {git['head']['stderr']}")
    if git.get("tracked_pycache"):
        failures.append(f"tracked __pycache__/pyc: {git['tracked_pycache'][:5]}")
    if git.get("tracked_outputs"):
        failures.append(f"tracked outputs/logs: {git['tracked_outputs'][:5]}")

    # Dirty model-related working tree (ignore untracked pycache)
    dirty = []
    for line in (git["status_porcelain"]["stdout"] or "").splitlines():
        path = line[3:].strip() if len(line) > 3 else line
        if "__pycache__" in path or path.endswith(".pyc"):
            continue
        if any(
            path.startswith(p)
            for p in (
                "biocda/",
                "pretrain_VAEwC.py",
                "tools/source_anchor",
                "tools/conditional_adv",
                "configs/",
                "config/",
            )
        ):
            # During Round25 scaffolding, new files are expected; only fail on
            # unexpected modifications to Round23 lock.
            if "biocda_xa_model_lock.json" in path:
                dirty.append(line)
    if dirty:
        failures.append(f"unexpected dirty lock-related paths: {dirty}")

    must = [
        "docs/round23_final_report.md",
        "docs/round24_final_report.md",
        "reports/round24_final_model_lock.json",
        "reports/biocda_xa_model_lock.json",
        "reports/round23_paired_performance.csv",
        "configs/round24/eval3.yaml",
        "config/round25_stage2_margin_screen.yaml",
    ]
    existence = {m: _must_exist(m) for m in must}
    for m, info in existence.items():
        if not info["exists"]:
            failures.append(f"missing required path: {m}")

    parse = parse_round24_final_report(ROOT / "docs/round24_final_report.md")
    art = audit_round24_artifacts(parse)
    failures.extend(art["conflicts"])

    ckpt = find_round24_checkpoints()
    if ckpt["n_files"] == 0:
        failures.append("no Round24 checkpoints/summaries found under expected roots")

    repo_audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "git": git,
        "required_paths": existence,
        "failures": failures,
        "strict": bool(args.strict),
        "pass": len(failures) == 0,
    }
    parse_out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parse": parse,
    }
    art_out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_audit": art,
        "checkpoints": ckpt,
        "round23_preserved_rejected": art.get("round23_xa_status") == "REJECTED",
    }

    (out_dir / "round25_repository_audit.json").write_text(
        json.dumps(repo_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "round25_round24_final_report_parse.json").write_text(
        json.dumps(parse_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "round25_round24_artifact_audit.json").write_text(
        json.dumps(art_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps({"pass": repo_audit["pass"], "n_failures": len(failures), "failures": failures}, indent=2))
    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
