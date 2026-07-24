#!/usr/bin/env python3
"""Lock Round 25 Stage2→XA closure artifacts and notify Telegram once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--notify", action="store_true", help="Send Telegram on full-round completion")
    args = ap.parse_args()

    a25 = json.loads((ROOT / "reports/round25_stage25a_decision.json").read_text(encoding="utf-8"))
    b25 = json.loads((ROOT / "reports/round25_selection_decision.json").read_text(encoding="utf-8"))
    c25 = {}
    cpath = ROOT / "reports/round25_c32_xa_effect.json"
    if cpath.exists():
        c25 = json.loads(cpath.read_text(encoding="utf-8"))

    # Never overwrite Round23 REJECTED lock.
    r23 = json.loads((ROOT / "reports/biocda_xa_model_lock.json").read_text(encoding="utf-8"))
    if r23.get("status") != "REJECTED":
        raise SystemExit("refusing lock: Round23 XA lock is not REJECTED")

    promote = bool(b25.get("promote_stage2"))
    selected_stage2 = a25.get("selected_variant") if promote else "S0"
    status = "LOCKED_STAGE2_PROMOTED" if promote else "LOCKED_KEEP_S0"

    lock = {
        "round": 25,
        "status": status,
        "architecture_version": "biocda-xa-v2",
        "downstream_xa": "SELECTED_WITH_INTERPRETABILITY_TRADEOFF",
        "round23_gdsc_status_preserved": "REJECTED",
        "stage25a": {
            "status": a25.get("status"),
            "selected_variant": a25.get("selected_variant"),
            "run_s3": a25.get("run_s3"),
            "run_s2b": a25.get("run_s2b"),
        },
        "stage25b": b25,
        "stage25c": {
            "c32_predictive_effect": c25.get("c32_predictive_effect"),
            "c32_attention_effect": c25.get("c32_attention_effect"),
            "final_claim": c25.get("final_claim"),
        },
        "promoted_stage2_variant": selected_stage2 if promote else "S0",
        "tcga_used_for_selection": False,
        "artifact_hashes": {
            "stage25a_decision": _sha(ROOT / "reports/round25_stage25a_decision.json"),
            "stage25b_decision": _sha(ROOT / "reports/round25_selection_decision.json"),
            "round23_xa_lock": _sha(ROOT / "reports/biocda_xa_model_lock.json"),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out = ROOT / "reports/biocda_xa_stage2_lock.json"
    out.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Final report
    report = ROOT / "docs/round25_final_report.md"
    report.write_text(
        "\n".join(
            [
                "# Round 25 — Stage2 Margin / AADA → No-Pooling XA Closure",
                "",
                f"**狀態：** `{status}`",
                f"**Stage25A：** `{a25.get('status')}` → selected `{a25.get('selected_variant')}`",
                f"**Stage25B：** `{b25.get('status')}` promote={promote}",
                f"**Stage25C claim：** `{c25.get('final_claim', '')}`",
                "",
                "## 硬性約束",
                "",
                "- Round23 GDSC XA lock 維持 `REJECTED`（未覆寫）",
                "- Downstream XA 固定 fresh no-pooling；未搜尋拓撲",
                "- TCGA 未參與選模",
                "- `reconstruction_margin` / `prototype_upper_margin` 欄位分離",
                "",
                "## 產物",
                "",
                "```text",
                "reports/round25_stage25a_decision.json",
                "reports/round25_selection_decision.json",
                "reports/round25_c32_xa_effect.json",
                "reports/biocda_xa_stage2_lock.json",
                "```",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"lock": str(out), "status": status, "promote": promote}, indent=2))
    if args.notify:
        from tools.biocda_telegram_notify import biocda_notify

        biocda_notify(
            "Round25 COMPLETE\n"
            f"status={status}\n"
            f"stage25a={a25.get('status')} selected={a25.get('selected_variant')}\n"
            f"stage25b={b25.get('status')} promote={promote}\n"
            f"c32_claim={c25.get('final_claim')}\n"
            f"lock=reports/biocda_xa_stage2_lock.json"
        )
        print("telegram_sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
