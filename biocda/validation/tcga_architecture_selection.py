"""TCGA-priority architecture selection (no GDSC test)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# User-specified TCGA target priority (highest → lowest)
TARGET_PRIORITY = [
    "gdsc_intersect13",
    "tcga_only3",
    "dapl",
    "aacdr_gdsc_intersect",
    "aacdr_tcga_only",
]
TARGET_WEIGHTS = {t: w for t, w in zip(TARGET_PRIORITY, [5, 4, 3, 2, 1])}

PRIMARY_METRIC = "DrugMacro_AUC"
TIE_BREAKERS = ["DrugMacro_AUPRC", "Global_AUC", "Global_AUPRC"]

ARCHITECTURE_FAMILY = {
    "r20_predictive_locked": {
        "family": "predictive_pooled_e3",
        "role": "BioCDA-Predictive",
        "description": "Round20 locked 15-fold pooled E3",
    },
    "r23_biocda_predictive": {
        "family": "predictive_pooled_e3",
        "role": "BioCDA-Predictive",
        "description": "Round23 retrained pooled E3 (P0)",
    },
    "r21_pooled_baseline": {
        "family": "predictive_pooled_e3",
        "role": "BioCDA-Predictive-factory",
        "description": "Round21 pooled baseline (M0)",
    },
    "r21_biocda_xa_z": {
        "family": "xa_v1",
        "role": "BioCDA-XA-Candidate",
        "description": "Round21 XA Z-only (M1)",
    },
    "r21_biocda_xa_zc": {
        "family": "xa_v1",
        "role": "BioCDA-XA-Candidate",
        "description": "Round21 XA Z+C (M2)",
    },
    "r23_biocda_xa_fresh": {
        "family": "xa_v2",
        "role": "BioCDA-XA-Candidate",
        "description": "Round23 no-pooling XA fresh (X0)",
    },
    "r23_biocda_xa_transfer": {
        "family": "xa_v2",
        "role": "BioCDA-XA-Candidate",
        "description": "Round23 XA E3 transfer (X1)",
    },
    "r23_biocda_xa_kd": {
        "family": "xa_v2",
        "role": "BioCDA-XA-Candidate",
        "description": "Round23 XA E3 transfer + KD (X2)",
    },
}


@dataclass
class ModelTcgaScorecard:
    model_id: str
    display_name: str
    round_tag: str
    architecture: str
    family: str
    role: str
    per_target: Dict[str, Dict[str, float]] = field(default_factory=dict)
    weighted_drug_macro_auc: float = 0.0
    weighted_drug_macro_auprc: float = 0.0
    lexicographic_rank: int = 0
    weighted_rank: int = 0

    def metric_vector(self, metric: str) -> List[float]:
        return [self.per_target[t].get(metric, float("nan")) for t in TARGET_PRIORITY]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "round": self.round_tag,
            "architecture": self.architecture,
            "family": self.family,
            "role": self.role,
            "per_target": self.per_target,
            "weighted_DrugMacro_AUC": self.weighted_drug_macro_auc,
            "weighted_DrugMacro_AUPRC": self.weighted_drug_macro_auprc,
            "lexicographic_rank": self.lexicographic_rank,
            "weighted_rank": self.weighted_rank,
        }


def load_tcga_long_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_scorecards(df: pd.DataFrame) -> List[ModelTcgaScorecard]:
    cards: List[ModelTcgaScorecard] = []
    for model_id, g in df.groupby("model_id"):
        meta = ARCHITECTURE_FAMILY.get(model_id, {"family": "unknown", "role": "unknown", "description": ""})
        card = ModelTcgaScorecard(
            model_id=str(model_id),
            display_name=str(g["display_name"].iloc[0]),
            round_tag=str(g["round"].iloc[0]),
            architecture=str(g["architecture"].iloc[0]),
            family=str(meta["family"]),
            role=str(meta["role"]),
        )
        for target in TARGET_PRIORITY:
            sub = g[g["target"] == target]
            if sub.empty:
                continue
            card.per_target[target] = {
                "DrugMacro_AUC": float(sub["DrugMacro_AUC"].iloc[0]),
                "DrugMacro_AUPRC": float(sub["DrugMacro_AUPRC"].iloc[0]),
                "Global_AUC": float(sub["Global_AUC"].iloc[0]),
                "Global_AUPRC": float(sub["Global_AUPRC"].iloc[0]),
            }
        card.weighted_drug_macro_auc = sum(
            card.per_target[t]["DrugMacro_AUC"] * TARGET_WEIGHTS[t] for t in TARGET_PRIORITY if t in card.per_target
        )
        card.weighted_drug_macro_auprc = sum(
            card.per_target[t]["DrugMacro_AUPRC"] * TARGET_WEIGHTS[t] for t in TARGET_PRIORITY if t in card.per_target
        )
        cards.append(card)
    return cards


def rank_lexicographic(cards: List[ModelTcgaScorecard]) -> List[ModelTcgaScorecard]:
    def sort_key(c: ModelTcgaScorecard) -> tuple:
        keys = []
        for metric in [PRIMARY_METRIC] + TIE_BREAKERS:
            for t in TARGET_PRIORITY:
                keys.append(-c.per_target.get(t, {}).get(metric, float("-inf")))
        return tuple(keys)

    ranked = sorted(cards, key=sort_key)
    for i, c in enumerate(ranked, start=1):
        c.lexicographic_rank = i
    return ranked


def rank_weighted(cards: List[ModelTcgaScorecard]) -> List[ModelTcgaScorecard]:
    order = sorted(cards, key=lambda c: (-c.weighted_drug_macro_auc, -c.weighted_drug_macro_auprc))
    for i, c in enumerate(order, start=1):
        c.weighted_rank = i
    return order


def best_per_family(cards: List[ModelTcgaScorecard]) -> Dict[str, ModelTcgaScorecard]:
    out: Dict[str, ModelTcgaScorecard] = {}
    for c in cards:
        if c.family not in out or c.weighted_drug_macro_auc > out[c.family].weighted_drug_macro_auc:
            out[c.family] = c
    return out


def decide_final_architecture(cards: List[ModelTcgaScorecard]) -> Dict[str, Any]:
    """Primary decision: weighted TCGA DrugMacro AUC with user target priority."""
    ranked = rank_weighted(cards)
    lex = rank_lexicographic(cards)
    winner = ranked[0]
    lex_winner = lex[0]
    families = best_per_family(cards)

    predictive_best = max(
        (c for c in cards if c.family == "predictive_pooled_e3"),
        key=lambda c: c.weighted_drug_macro_auc,
    )
    xa_best = max(
        (c for c in cards if c.family in {"xa_v1", "xa_v2"}),
        key=lambda c: c.weighted_drug_macro_auc,
    )

    # Unified verdict: weighted score is primary; lexicographic reported as sensitivity
    final_architecture = {
        "canonical_name": "BioCDA-XA" if winner.family.startswith("xa") else "BioCDA-Predictive",
        "model_id": winner.model_id,
        "display_name": winner.display_name,
        "architecture": winner.architecture,
        "training_recipe": winner.display_name.split("/")[0].strip(),
        "selection_method": "TCGA_weighted_DrugMacro_AUC",
        "weighted_DrugMacro_AUC": winner.weighted_drug_macro_auc,
        "weighted_DrugMacro_AUPRC": winner.weighted_drug_macro_auprc,
    }

    decision = {
        "final_architecture": final_architecture,
        "lexicographic_alternative": {
            "model_id": lex_winner.model_id,
            "display_name": lex_winner.display_name,
            "architecture": lex_winner.architecture,
            "note": (
                "若嚴格以最高優先 target gdsc_intersect13 的 DrugMacro AUC 為唯一標準，"
                f"則選 {lex_winner.display_name}（{lex_winner.per_target['gdsc_intersect13']['DrugMacro_AUC']:.3f}）。"
            ),
        },
        "predictive_family_best": predictive_best.to_dict(),
        "xa_family_best": xa_best.to_dict(),
        "selection_basis": "TCGA_weighted_DrugMacro_AUC",
    }

    return {
        "protocol": {
            "evaluation_domain": "TCGA_external_only",
            "excluded_from_selection": [
                "GDSC_development_unseen_drug",
                "GDSC_validation",
                "GDSC_test",
            ],
            "target_priority": TARGET_PRIORITY,
            "target_weights": TARGET_WEIGHTS,
            "primary_metric": PRIMARY_METRIC,
            "tie_breakers": TIE_BREAKERS,
        },
        "winner_weighted": winner.to_dict(),
        "winner_lexicographic": lex_winner.to_dict(),
        "best_per_family": {k: v.to_dict() for k, v in families.items()},
        "all_rankings_weighted": [c.to_dict() for c in ranked],
        "all_rankings_lexicographic": [c.to_dict() for c in lex],
        "final_recommendation": decision,
    }


def scorecard_table_markdown(cards: List[ModelTcgaScorecard], metric: str) -> str:
    header = "| Model | " + " | ".join(TARGET_PRIORITY) + f" | weighted |"
    sep = "|---|" + "|".join(["---:"] * (len(TARGET_PRIORITY) + 1)) + "|"
    lines = [header, sep]
    for c in sorted(cards, key=lambda x: x.weighted_rank or 999):
        vals = [f"{c.per_target.get(t, {}).get(metric, float('nan')):.3f}" for t in TARGET_PRIORITY]
        w = c.weighted_drug_macro_auc if metric == PRIMARY_METRIC else c.weighted_drug_macro_auprc
        lines.append(f"| {c.display_name} | " + " | ".join(vals) + f" | **{w:.3f}** |")
    return "\n".join(lines)


def write_selection_artifacts(
    *,
    long_csv: Path,
    output_json: Path,
    output_md: Path,
) -> Dict[str, Any]:
    df = load_tcga_long_csv(long_csv)
    cards = build_scorecards(df)
    rank_lexicographic(cards)
    rank_weighted(cards)
    payload = decide_final_architecture(cards)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    winner = payload["winner_weighted"]
    lex = payload["winner_lexicographic"]
    rec = payload["final_recommendation"]
    fin = rec["final_architecture"]
    lex_alt = rec["lexicographic_alternative"]

    md = f"""# BioCDA 最終架構選擇（TCGA 優先順序）

## 選模原則

本決策**不以 GDSC development / unseen-drug validation / test 為依據**。

唯一評估域：**TCGA 五個 external target**，優先順序（高→低）：

```text
gdsc_intersect13 > tcga_only3 > dapl > aacdr_gdsc_intersect > aacdr_tcga_only
```

| 項目 | 設定 |
|------|------|
| 主指標 | DrugMacro AUC |
| 加權 | 5 : 4 : 3 : 2 : 1（對應上述順序） |
| 平手規則 | DrugMacro AUPRC → Global AUC → Global AUPRC |
| 次排序 | 字典序（lexicographic）同 target 順序 |

資料來源：`reports/biocda_tcga_comparison/biocda_tcga_comparison_long.csv`

---

## 加權 DrugMacro AUC 排名（決策主排序）

{scorecard_table_markdown(cards, PRIMARY_METRIC)}

## 加權 DrugMacro AUPRC

{scorecard_table_markdown(cards, "DrugMacro_AUPRC")}

---

## 最終建議（單一架構）

| 項目 | 決策 |
|------|------|
| **正式名稱** | **{fin['canonical_name']}** |
| **選中模型** | **{fin['display_name']}** |
| **架構版本** | `{fin['architecture']}` |
| **選模方法** | TCGA 加權 DrugMacro AUC（target 權重 5:4:3:2:1） |
| **加權 DrugMacro AUC** | **{fin['weighted_DrugMacro_AUC']:.4f}** |
| **加權 DrugMacro AUPRC** | {fin['weighted_DrugMacro_AUPRC']:.4f} |

### 架構定義（若選 XA）

```text
Z64 + C32 → sample query Q0 [B,1,128]
GIN 5×32 → atom nodes（no pooling）
2-layer sample→atom cross-attention（d=128, H=4）
response head(Qfinal[:,0,:]) → logit
```

訓練配方：`biocda_xa_fresh`（fresh GIN，無 KD，無 E3 transfer）

### 敏感性：字典序（gdsc_intersect13 絕對優先）

若改以**最高優先 target 單獨決勝**，則選 **{lex_alt['display_name']}**（gdsc_intersect13 DrugMacro AUC = {payload['winner_lexicographic']['per_target']['gdsc_intersect13']['DrugMacro_AUC']:.3f}）。

{lex_alt['note']}

### 加權排名對照

| 排名 | 模型 | weighted DrugMacro AUC |
|------|------|------------------------|
| 1 | {winner['display_name']} | {winner['weighted_DrugMacro_AUC']:.4f} |
| 2 | {rec['predictive_family_best']['display_name']} | {rec['predictive_family_best']['weighted_DrugMacro_AUC']:.4f} |
| 字典序 #1 | {lex['display_name']} | {lex['weighted_DrugMacro_AUC']:.4f} |

### 架構家族最佳

| 家族 | 最佳模型 | weighted DrugMacro AUC |
|------|----------|------------------------|
"""
    for fam, info in payload["best_per_family"].items():
        md += f"| {fam} | {info['display_name']} | {info['weighted_DrugMacro_AUC']:.4f} |\n"

    md += """
---

## 與 Round 23 GDSC 結論的關係

Round 23 以 GDSC unseen-drug 配對 gate 判定 XA **REJECTED**（performance_failure）。

若正式產品決策改以 **TCGA 優先順序** 為準，則需將本文件視為 **新的選模協議**，
並相應更新 `biocda_xa_model_lock.json` 與論文方法學描述（明確聲明不再以 GDSC test 選模）。

**不得**在未更新 lock manifest 的情況下，同時宣稱 GDSC-REJECTED 與 TCGA-SELECTED。

---

## 重現

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/select_biocda_architecture_tcga.py'
```
"""
    output_md.write_text(md, encoding="utf-8")
    return payload
