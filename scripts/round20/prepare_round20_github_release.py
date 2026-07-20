#!/usr/bin/env python3
"""Prepare GitHub release notes and README section from Round 20 artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.round20.publication_checks import build_github_release_notes, build_readme_section
from tools.round20.result_contracts import DEFAULT_RUN_ROOT, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readme_section = build_readme_section(run_root=args.run_root)
    notes = build_github_release_notes(run_root=args.run_root)
    (args.output_dir / "round20_readme_section.md").write_text(readme_section.strip() + "\n", encoding="utf-8")
    (args.output_dir / "round20_github_release_notes.md").write_text(notes, encoding="utf-8")
    write_json(
        args.output_dir / "round20_publication_manifest.json",
        {
            "readme_section": str(args.output_dir / "round20_readme_section.md"),
            "release_notes": str(args.output_dir / "round20_github_release_notes.md"),
        },
    )
    print(f"Wrote {args.output_dir / 'round20_github_release_notes.md'}")


if __name__ == "__main__":
    main()
