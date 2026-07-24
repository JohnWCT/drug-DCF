#!/usr/bin/env python3
"""Render Round24 vs AACDR standard comparison tables."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "reports/round24/vs_aacdr_standard.json"
MD_PATH = ROOT / "reports/round24/vs_aacdr_standard.md"
ORDER = [
    "gdsc_intersect13",
    "tcga_only3",
    "dapl",
    "aacdr_gdsc_intersect",
    "aacdr_tcga_only",
]


def cell(score: float, base: float) -> str:
    d = score - base
    mark = "Y" if score > base else "N"
    return f"{mark} {score:.4f} ({d:+.4f})"


def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    std = data["standard"]
    cands = data["candidates"]

    lines: list[str] = []
    lines += [
        "# Round 24 vs AACDR DrugMacro 標準",
        "",
        "**標準來源：** [`docs/AACDR_drug_macro_auroc_auprc.md`](../docs/AACDR_drug_macro_auroc_auprc.md)",
        "",
        "目標：五組評估集的 DrugMacro AUROC / AUPRC **嚴格大於**標準 mean。",
        "`Y` = 已超越；`N` = 未超越；括號 = Δ（ours − standard）。",
        "",
        "## 標準（eval3 × 3 + target_infer × 2）",
        "",
        "| 評估集 | DrugMacro AUROC | DrugMacro AUPRC |",
        "|--------|----------------:|----------------:|",
    ]
    for t in ORDER:
        lines.append(f"| `{t}` | {std[t]['AUROC']:.4f} | {std[t]['AUPRC']:.4f} |")

    lines += [
        "",
        "## 一眼看：哪些指標「曾被任一候選超越」",
        "",
        "| 評估集 | AUROC 標準 | 曾超越？ | 最佳 AUROC（候選） | AUPRC 標準 | 曾超越？ | 最佳 AUPRC（候選） |",
        "|--------|-----------:|:--------:|--------------------|-----------:|:--------:|--------------------|",
    ]
    for t in ORDER:
        best_auc = max(cands, key=lambda c: c["per_target"][t]["AUROC"])
        best_ap = max(cands, key=lambda c: c["per_target"][t]["AUPRC"])
        ya = "**是**" if best_auc["per_target"][t]["beat_AUROC"] else "否"
        yp = "**是**" if best_ap["per_target"][t]["beat_AUPRC"] else "否"
        lines.append(
            f"| `{t}` | {std[t]['AUROC']:.4f} | {ya} | "
            f"{best_auc['per_target'][t]['AUROC']:.4f}（{best_auc['name']}） | "
            f"{std[t]['AUPRC']:.4f} | {yp} | "
            f"{best_ap['per_target'][t]['AUPRC']:.4f}（{best_ap['name']}） |"
        )

    auc_ok = data["targets_ever_beat_AUROC"]
    auc_no = data["targets_never_beat_AUROC"]
    ap_ok = data["targets_ever_beat_AUPRC"]
    ap_no = data["targets_never_beat_AUPRC"]

    def fmt_list(keys: list[str]) -> str:
        return ", ".join(f"`{k}`" for k in keys) if keys else "（無）"

    lines += [
        "",
        "### 結論摘要",
        "",
        f"- **AUROC 已有候選可超越：** {fmt_list(auc_ok)}",
        f"- **AUROC 目前無人超越：** {fmt_list(auc_no)}",
        f"- **AUPRC 已有候選可超越：** {fmt_list(ap_ok)}",
        f"- **AUPRC 目前無人超越：** {fmt_list(ap_no)}",
        "- **五組 AUROC 全過：** 尚無任一候選（含 ablation）。",
        "- **最硬缺口：** `tcga_only3`（AUROC 與 AUPRC 皆無人超越標準）。",
        "",
        "## 各候選 × AUROC（vs 標準）",
        "",
        "| Candidate | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only | AUROC 超越數 |",
        "|-----------|-----------------:|-----------:|-----:|---------------------:|---------------:|-------------:|",
    ]
    for c in cands:
        cells = [cell(c["per_target"][t]["AUROC"], std[t]["AUROC"]) for t in ORDER]
        lines.append(
            f"| {c['name']} | " + " | ".join(cells) + f" | {c['auc_pass']}/5 |"
        )

    lines += [
        "",
        "## 各候選 × AUPRC（vs 標準）",
        "",
        "| Candidate | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only | AUPRC 超越數 |",
        "|-----------|-----------------:|-----------:|-----:|---------------------:|---------------:|-------------:|",
    ]
    for c in cands:
        cells = [cell(c["per_target"][t]["AUPRC"], std[t]["AUPRC"]) for t in ORDER]
        lines.append(
            f"| {c['name']} | " + " | ".join(cells) + f" | {c['auprc_pass']}/5 |"
        )

    lines += [
        "",
        "## 各候選：同時超越 AUROC+AUPRC 的 target 數",
        "",
        "| Candidate | AUROC 超越 | AUPRC 超越 | 同 target 雙指標皆超 |",
        "|-----------|-----------:|-----------:|---------------------:|",
    ]
    for c in cands:
        lines.append(
            f"| {c['name']} | {c['auc_pass']}/5 | {c['auprc_pass']}/5 | {c['both_pass']}/5 |"
        )

    lines += [
        "",
        "## 圖例",
        "",
        "- `Y score (Δ)` = 嚴格超越標準；`N score (Δ)` = 未超越。",
        "- Ablation 僅診斷，非正式 lock。",
        "- Round 24 硬閘：五組 **AUROC** 全過；AUPRC 並列追蹤與 tie-break。",
        "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    main()
