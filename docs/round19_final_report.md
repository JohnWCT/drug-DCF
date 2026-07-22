# Round 19 — Factorial + Domain Shift + Role Lock

**狀態：** ALL_DONE（19D–19G + reproducibility archive）

見 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md#round-19--factorial--domain-shift--role-lock)。

## 19D 開發集（5CV mean-of-means）

**最高：** F2（D0×P2×O3）**~0.620**

## 19E per-shift mean DrugMacro AUC

| Shift | 最佳 | AUC |
|-------|------|-----|
| cancer_type_heldout | E2 | 0.5824 |
| drug_heldout | **E3** | **0.7503** |
| scaffold_heldout | E1 | 0.5806 |

## 19F 角色鎖定（single champion = null）

| 角色 | 候選 |
|------|------|
| Historical anchor | E0 |
| Source-performance champion | F2 |
| Parsimonious context | F1 |
| Cancer-shift specialist | E1 |
| Chemical-shift specialist | E3 |
| General recommended | **E3** |

## 19G 可解釋性

**Verdict：** `PARTIALLY_SUPPORTED`

- 模型使用 drug 與 omics/context 訊息。
- 高排名 perturbation 通常大於 matched random。
- Attention 跨 member 穩定度不足，不支持唯一或因果解釋。

## 結論

採 **scenario-aware multi-role policy**，無 single champion。E3（pooled O2）為 general recommended 且 drug_heldout 最佳，為 Round 20 E3 來源。可解釋性 claim 限於 post-lock model-behavior evidence。

模型卡片：[`model_cards/round19_locked_models.md`](model_cards/round19_locked_models.md)
