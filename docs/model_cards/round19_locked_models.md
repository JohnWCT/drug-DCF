# Round 19 Locked Models

Round 19 採多角色 policy，沒有 single champion。所有 checkpoint 均為 3 split seeds ×
5 folds 的 15-member ensemble，使用 mean probability。

## Locked sources

- `F0_historical_anchor`：historical anchor，GIN + pooled historical predictor + O1。
- `F1_primary_o2`：cancer-shift／parsimonious context roles，GIN + P2 atom
  cross-attention + O2。
- `F2_full_omics_o3`：source-performance role，GIN + P2 atom cross-attention + O3。
- `F3_best_pooled_o2`：chemical-shift／general-recommended roles，GIN + pooled
  predictor + O2。
- `F4_source_only_o4`：source-only domain candidate，GIN + P2 atom cross-attention + O4。
- `F5_maccs_efficient`：efficient role，MACCS + pooled predictor + O2。

## Interpretability status

Round 19G verdict 為 `PARTIALLY_SUPPORTED`。P2 primary attention 僅由最後一層所有
heads 平均取得；pooled 與 MACCS 模型沒有 atom attention。高排名 atom perturbation
平均影響高於 matched random，但跨 member attention rank 僅中低度穩定。

## Intended use

僅依 deployment policy 與 novelty classifier 路由至 locked role，並使用完整 15-member
ensemble。適合研究性 response ranking 與 post-lock audit。

## Prohibited use

- 不可挑選最佳 fold、依 internal/TCGA/interpretability 結果更換角色。
- 不可把 attention、occlusion 或 latent ablation 宣稱為生物因果證明。
- 不可把 exploratory TCGA 結果當作臨床決策依據。
