#!/usr/bin/env python3
"""Unified Round 20 post-completion CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round20.completion_audit import run_completion_audit
from tools.round20.documentation_builder import generate_all_docs
from tools.round20.predictor_contract import build_gate_summary, build_predictor_contract
from tools.round20.release_integrity import validate_release_directory
from tools.round20.reproduction import run_reproduction_audit
from tools.round20.result_contracts import DEFAULT_RUN_ROOT, load_json, sha256_file


def cmd_audit(args: argparse.Namespace) -> None:
    build_predictor_contract(run_root=args.run_root)
    build_gate_summary(run_root=args.run_root)
    audit = run_completion_audit(run_root=args.run_root, strict=args.strict)
    print(f"ROUND20_COMPLETION_AUDIT={audit['audit_status']}")


def cmd_reproduce(args: argparse.Namespace) -> None:
    run_reproduction_audit(
        run_root=args.run_root,
        release_dir=args.release_dir,
        mode=args.mode,
        strict=args.strict,
    )


def cmd_infer(args: argparse.Namespace) -> None:
  from scripts.round20.infer import run_infer  # noqa: WPS433

  run_infer(
      release_dir=args.release_dir,
      mode=args.mode,
      response_file=args.response_file,
      omics_file=args.omics_file,
      output=args.output,
      strict=args.strict,
  )


def cmd_release_info(args: argparse.Namespace) -> None:
    lock_path = args.run_root / "stage20c_lock/final_model_lock.json"
    lock = load_json(lock_path)
    audit_path = args.run_root / "round20_completion_audit.json"
    audit = load_json(audit_path) if audit_path.is_file() else {}
    rel_audit = load_json(args.release_dir / "hashes/release_audit.json") if (
        args.release_dir / "hashes/release_audit.json"
    ).is_file() else {}
    manifest = load_json(args.release_dir / "RELEASE_MANIFEST.json") if (
        args.release_dir / "RELEASE_MANIFEST.json"
    ).is_file() else {}
    print("Round 20 release status:", manifest.get("release_status", "UNKNOWN"))
    print("git SHA:", (audit.get("git") or {}).get("sha"))
    print("model lock SHA:", sha256_file(lock_path))
    print("selected context:", lock["selected_context"]["id"])
    print("selected predictor:", lock["selected_model"]["candidate_id"])
    print("drug encoder:", lock["selected_model"]["drug_encoder"])
    print("checkpoint policy:", lock["selected_model"]["checkpoint_policy"])
    print("TCGA inference:", audit.get("stages", {}).get("20D", {}).get("status"))
    print("release audit:", rel_audit.get("status"))


def cmd_docs(args: argparse.Namespace) -> None:
    generate_all_docs(run_root=args.run_root, docs_dir=args.docs_dir)


def main() -> None:
    p = argparse.ArgumentParser(prog="round20_cli")
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--release-dir", type=Path, default=DEFAULT_RUN_ROOT / "stage20e_release")
    p.add_argument("--strict", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("audit")
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("reproduce")
    s.add_argument("--mode", choices=["frozen", "raw", "both"], default="both")
    s.set_defaults(func=cmd_reproduce)

    s = sub.add_parser("infer")
    s.add_argument("--mode", choices=["frozen_latent", "raw_omics"], required=True)
    s.add_argument("--response-file", type=Path, required=True)
    s.add_argument("--omics-file", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s.set_defaults(func=cmd_infer)

    s = sub.add_parser("release-info")
    s.set_defaults(func=cmd_release_info)

    s = sub.add_parser("docs")
    s.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    s.set_defaults(func=cmd_docs)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
