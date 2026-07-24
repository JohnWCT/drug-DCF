#!/usr/bin/env python3
"""Rebuild reports/round24/vs_aacdr_standard.{json,md} from eval3.yaml gates."""
from __future__ import annotations

import json
from pathlib import Path

from biocda.validation.round24_gate import evaluate_all_target_gate
from biocda.validation.round24_protocol import gate_table, load_eval3_config

ROOT = Path(__file__).resolve().parents[2]
ORDER = [
    "gdsc_intersect13",
    "tcga_only3",
    "dapl",
    "aacdr_gdsc_intersect",
    "aacdr_tcga_only",
]


def load_candidates():
    cands = []
    base = json.loads((ROOT / "reports/round24/stage24a/baseline_summary.json").read_text())
    auc = {t: base["targets"][t]["fold_mean_DrugMacro_AUC"] for t in ORDER}
    auprc = {t: base["targets"][t]["fold_mean_DrugMacro_AUPRC"] for t in ORDER}
    cands.append(("B0/Ctrl pooled_mlp x own_plus_summary", auc, auprc))

    for bid, label in [("B1", "B1 predictive x C32"), ("B2", "B2 XA x C32")]:
        d = json.loads((ROOT / f"reports/round24/stage24b/{bid}/candidate_summary.json").read_text())
        cands.append((label, d["per_target_fold_mean_auc"], d["per_target_fold_mean_auprc"]))

    r = json.loads((ROOT / "reports/round24/stage24c/feature_attribution_summary.json").read_text())
    feat = {"F0": "own+sum", "F1": "z_only", "F2": "C16", "F3": "C32", "F4": "C64"}
    by_id = {x["candidate_id"]: x for x in r["ranked"]}
    for fid in ["F2", "F3", "F0", "F1", "F4"]:
        x = by_id[fid]
        cands.append(
            (
                f"{fid} pred x {feat[fid]}",
                x["per_target_fold_mean_auc"],
                x["per_target_fold_mean_auprc"],
            )
        )

    for arm in ["NoHoldout", "AACDR"]:
        d = json.loads(
            (ROOT / f"reports/round24/train_source_ablation/{arm}/candidate_summary.json").read_text()
        )
        cands.append(
            (
                f"Ablation {arm}",
                d["per_target_fold_mean_auc"],
                d["per_target_fold_mean_auprc"],
            )
        )

    # Stage24E NoHoldout confirmation arms
    for eid in ["E-NH0", "E-NH1", "E-NH2"]:
        p = ROOT / f"reports/round24/stage24e/{eid}/candidate_summary.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        cands.append(
            (
                f"{eid} {d.get('architecture','?')} x {d.get('feature','?')} (NoHoldout)",
                d["per_target_fold_mean_auc"],
                d["per_target_fold_mean_auprc"],
            )
        )
    return cands


def main() -> None:
    cfg = load_eval3_config(ROOT / "configs/round24/eval3.yaml")
    gates = gate_table(cfg)
    std = {
        t: {
            "AUROC": float(gates[t]["gate_auroc"]),
            "AUPRC": float(gates[t]["gate_auprc"]),
            "AUROC_std": float(gates[t].get("gate_auroc_std", 0.0)),
        }
        for t in ORDER
    }
    req = list(cfg.get("gate_required_targets", ["aacdr_gdsc_intersect", "aacdr_tcga_only"]))

    out_cands = []
    for name, auc, auprc in load_candidates():
        gate = evaluate_all_target_gate(
            auc,
            gates,
            target_priority=cfg["target_priority"],
            target_weights=cfg["target_weights"],
            gate_required_targets=req,
        )
        per = {}
        for t in ORDER:
            per[t] = {
                "AUROC": float(auc[t]),
                "AUPRC": float(auprc[t]),
                "AUROC_delta": float(auc[t]) - std[t]["AUROC"],
                "AUPRC_delta": float(auprc[t]) - std[t]["AUPRC"],
                "beat_AUROC": float(auc[t]) > std[t]["AUROC"],
                "beat_AUPRC": float(auprc[t]) > std[t]["AUPRC"],
            }
        out_cands.append(
            {
                "name": name,
                "gate_status": gate["status"],
                "n_pass_required": gate["n_pass_required"],
                "n_required": gate["n_required"],
                "auc_pass": sum(1 for t in ORDER if per[t]["beat_AUROC"]),
                "auprc_pass": sum(1 for t in ORDER if per[t]["beat_AUPRC"]),
                "both_pass": sum(
                    1 for t in ORDER if per[t]["beat_AUROC"] and per[t]["beat_AUPRC"]
                ),
                "per_target": per,
            }
        )

    auc_ok = [t for t in ORDER if any(c["per_target"][t]["beat_AUROC"] for c in out_cands)]
    auc_no = [t for t in ORDER if t not in auc_ok]
    ap_ok = [t for t in ORDER if any(c["per_target"][t]["beat_AUPRC"] for c in out_cands)]
    ap_no = [t for t in ORDER if t not in ap_ok]

    payload = {
        "standard_source": "docs/AACDR_drug_macro_auroc_auprc.md",
        "standard_name": cfg.get("standard_name", "aacdr_stest0_no_holdout"),
        "standard": std,
        "gate_required_targets": req,
        "candidates": out_cands,
        "targets_ever_beat_AUROC": auc_ok,
        "targets_never_beat_AUROC": auc_no,
        "targets_ever_beat_AUPRC": ap_ok,
        "targets_never_beat_AUPRC": ap_no,
    }
    jpath = ROOT / "reports/round24/vs_aacdr_standard.json"
    jpath.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", jpath)

    # Render markdown
    def cell(score: float, base: float) -> str:
        d = score - base
        mark = "Y" if score > base else "N"
        return f"{mark} {score:.4f} ({d:+.4f})"

    lines = [
        "# Round 24 vs AACDR DrugMacro 標準（stest0 / 無 10% testset）",
        "",
        "**標準來源：** [`docs/AACDR_drug_macro_auroc_auprc.md`](../docs/AACDR_drug_macro_auroc_auprc.md)（`eval3_stest0` / `target_infer_stest0`）",
        "",
        f"**硬閘 PASS：** `{req[0]}` > {std[req[0]]['AUROC']:.4f} ∧ `{req[1]}` > {std[req[1]]['AUROC']:.4f}。",
        "其餘三組必報、不擋 lock。`Y` = 嚴格超越；括號 = Δ。",
        "",
        "## 標準（stest0）",
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
        best_auc = max(out_cands, key=lambda c: c["per_target"][t]["AUROC"])
        best_ap = max(out_cands, key=lambda c: c["per_target"][t]["AUPRC"])
        ya = "**是**" if best_auc["per_target"][t]["beat_AUROC"] else "否"
        yp = "**是**" if best_ap["per_target"][t]["beat_AUPRC"] else "否"
        lines.append(
            f"| `{t}` | {std[t]['AUROC']:.4f} | {ya} | "
            f"{best_auc['per_target'][t]['AUROC']:.4f}（{best_auc['name']}） | "
            f"{std[t]['AUPRC']:.4f} | {yp} | "
            f"{best_ap['per_target'][t]['AUPRC']:.4f}（{best_ap['name']}） |"
        )

    def fmt_list(keys):
        return ", ".join(f"`{k}`" for k in keys) if keys else "（無）"

    hard_passers = [c["name"] for c in out_cands if c["gate_status"] == "PASS"]
    lines += [
        "",
        "### 結論摘要",
        "",
        f"- **AUROC 已有候選可超越：** {fmt_list(auc_ok)}",
        f"- **AUROC 目前無人超越：** {fmt_list(auc_no)}",
        f"- **AUPRC 已有候選可超越：** {fmt_list(ap_ok)}",
        f"- **AUPRC 目前無人超越：** {fmt_list(ap_no)}",
        f"- **硬閘 PASS 候選：** {fmt_list(hard_passers) if hard_passers else '（尚無）'}",
        "",
        "## 各候選 × AUROC（vs stest0）",
        "",
        "| Candidate | 硬閘 | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only | AUROC 超越數 |",
        "|-----------|:----:|-----------------:|-----------:|-----:|---------------------:|---------------:|-------------:|",
    ]
    for c in out_cands:
        cells = [cell(c["per_target"][t]["AUROC"], std[t]["AUROC"]) for t in ORDER]
        lines.append(
            f"| {c['name']} | {c['gate_status']} | "
            + " | ".join(cells)
            + f" | {c['auc_pass']}/5 |"
        )

    lines += [
        "",
        "## 各候選 × AUPRC（vs stest0）",
        "",
        "| Candidate | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only | AUPRC 超越數 |",
        "|-----------|-----------------:|-----------:|-----:|---------------------:|---------------:|-------------:|",
    ]
    for c in out_cands:
        cells = [cell(c["per_target"][t]["AUPRC"], std[t]["AUPRC"]) for t in ORDER]
        lines.append(f"| {c['name']} | " + " | ".join(cells) + f" | {c['auprc_pass']}/5 |")

    lines += [
        "",
        "## 圖例",
        "",
        "- 標準 = **無 10% testset（stest0）** AACDR 基準。",
        "- **硬閘：** 僅 `aacdr_gdsc_intersect` ∧ `aacdr_tcga_only`。",
        "- Ablation 僅診斷，非正式 lock（除非寫入 24E manifest）。",
        "- PASS 後選模序：`aacdr_gdsc`(5) > `aacdr_tcga_only`(4) > `dapl`(3) > `gdsc13`(2) > `tcga_only3`(1)。",
        "",
    ]
    md = ROOT / "reports/round24/vs_aacdr_standard.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", md)
    print("hard_passers:", hard_passers)


if __name__ == "__main__":
    main()
