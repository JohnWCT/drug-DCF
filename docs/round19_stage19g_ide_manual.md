# Round 19G perturbation/routing/analyzer 手冊

19G 是 final-lock 後的描述性分析，不得修改 final lock、experiment lock 或角色。
Executor 只接受 immutable 19G experiment lock 及其 pinned attention/occlusion/omics/routing manifests。

## 安全執行

- `bash tools/run_round19_stage19g_setup_smoke.sh`：synthetic 加單一真實 checkpoint
  strict-load/forward/probability replay；不需要 experiment lock，也不啟動全量。
- `ROUND19G_EXPERIMENT_LOCK=... bash tools/run_round19_stage19g.sh pilot`：10 cases ×
  兩個主要 locked sources × 15 members，並驗證 member completeness、atom mapping、
  attention sum、occlusion、原始機率與 immutable hashes。
- 正式 dry-run：
  `python3 tools/round19_stage19g_dispatch.py --formal --experiment-lock "$ROUND19G_EXPERIMENT_LOCK"`。
- 正式執行：
  `ROUND19G_ALLOW_FORMAL=1 ROUND19G_EXPERIMENT_LOCK=... bash tools/run_round19_stage19g.sh formal`。
- 個別 runner：`run_round19_stage19g_attention_export.sh`、
  `run_round19_stage19g_occlusion.sh`、`run_round19_stage19g_omics_ablation.sh`、
  `run_round19_stage19g_routing_audit.sh`、`run_round19_stage19g_finalize.sh`。
- status sidecar 支援完成後 resume skip 與失敗狀態；開始、完成、失敗均呼叫既有 Telegram notifier。

## 正式輸出 CSV

分析目錄必須完整包含：

1. `round19g_atom_occlusion.csv`
2. `round19g_connected_substructure_masking.csv`
3. `round19g_scaffold_sidechain_ablation.csv`
4. `round19g_bond_occlusion.csv`
5. `round19g_pooled_drug_occlusion.csv`
6. `round19g_maccs_ablation.csv`
7. `round19g_omics_group_ablation.csv`
8. `round19g_context_sensitivity.csv`
9. `round19g_routing_audit.csv`
10. `round19g_routing_counterfactual.csv`
11. `round19g_case_summary.csv`

另需 `experiment_lock.sha256`，內容須等於分析時 experiment lock 的完整檔案 SHA-256。

## 分析 gate

設定 `ROUND19G_OUTPUT_DIR`、`ROUND19G_CASE_MANIFEST`、`ROUND19G_FINAL_LOCK`、
`ROUND19G_EXPERIMENT_LOCK` 與 `ROUND19G_VERDICT` 後執行
`bash tools/run_analyze_round19_stage19g.sh`。Analyzer 強制 `--require-complete`，檢查所有
locked cases、每 case 15 members、20 次 matched random、routing 100%，以及 final/experiment
lock hashes。Verdict 僅可為 `SUPPORTED`、`PARTIALLY_SUPPORTED`、`NOT_SUPPORTED`。

Graph atom/bond 方法保留 topology；pooled 方法只有 input perturbation，沒有 attention。
MACCS 僅為 D4 fingerprint-bit 結果，不產生 atom heatmap。Omics latent dimensions 一律稱
`omics feature blocks`，不得稱為 genes。Counterfactual/regret 僅為 post-lock descriptive。

每個 task 開始前及完成後都重驗 experiment lock、final lock 與 checkpoint hashes。Routing
manifest 仍須完整，但 dispatcher 去重為單一 deterministic all-case job。所有 shard CSV/status
以 atomic replace 寫入；finalize 會檢查 job coverage、duplicates、case coverage，並生成
attention long/ensemble/consistency/context/variance CSV 與上述 11 個 analyzer CSV。
