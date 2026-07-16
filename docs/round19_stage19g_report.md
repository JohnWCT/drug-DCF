# Round 19G Interpretability and Faithfulness Report

## 結論

Round 19G 已完成，verdict 為 `PARTIALLY_SUPPORTED`。這是 final-lock 後的描述性分析，
不參與模型選擇，也沒有改變任何角色。

## 執行與完整性

- Local framework commit：`282895f7d2fe7919cb31efc6a383eb8ef9496481`
- Immutable experiment lock：`round19_stage19g_experiment_lock_v5.json`
- Final role lock SHA-256：`e45df23826b31822e986517311969a5b7a540eed659c1f20e847e1c7b29e24ff`
- Cases：230；包含 representative 120、contrastive 60、patient-conditioned 30、
  TCGA exploratory 20。
- Formal jobs：1,801/1,801，0 failed；所有模型均為 15-member ensemble。
- Routing audit：230/230 規則一致，match rate 100%。
- Finalize：`complete=true`、`roles_changed=false`。

## 注意力穩定性

P2 模型使用最後一層、所有 heads 平均作為 primary attention。跨 member 平均結果：

- `F1_primary_o2`：Spearman 0.3392、top-5 overlap 0.3507、JSD 0.0974。
- `F2_full_omics_o3`：Spearman 0.3773、top-5 overlap 0.3677、JSD 0.1092。
- `F4_source_only_o4`：Spearman 0.2853、top-5 overlap 0.3112、JSD 0.0971。

注意力分布在機率與正規化檢查上有效，但跨 checkpoint 排名僅中低度穩定，故不能把單一
member heatmap 當成唯一機制解釋。

## Faithfulness

Top-1 atom feature masking 的平均絕對 prediction delta 均高於 matched random：

- `F1_primary_o2`：0.10981 vs 0.08001。
- `F2_full_omics_o3`：0.10553 vs 0.07563。
- `F4_source_only_o4`：0.05332 vs 0.03129。

Pooled 模型的單 atom input perturbation 亦呈相同方向，但它不是 attention attribution。
MACCS bit ablation 共 210,510 rows，平均絕對 delta 0.02038；只解讀為 fingerprint-bit
敏感度，不映射成原子 heatmap。

## Omics／context sensitivity

在可適用的 O2/O3 模型中，shuffled context 的平均絕對機率變化為 0.04438–0.06080，
zero context 為 0.03837–0.05261。O1/O4 的 context 結果標示為 not-applicable，
不據此宣稱基因或 pathway 層級機制。

## 限制

- TCGA 20 cases 僅為 exploratory，不可作角色選擇或 confirmatory claim。
- Attention、occlusion 與 context intervention 是模型行為證據，不是生物因果證明。
- 所有角色、checkpoint 與 routing policy 仍由 Round 19F immutable lock 決定。
