# Round 24 — 執行狀態報告

**更新時間：** 2026-07-23  
**容器：** `DAPL` · `/workspace/DAPL`

## 總覽

| Stage | 狀態 | 說明 |
|-------|------|------|
| 24A 協議/基準 | **PASS** | eval3 manifest；906→886 miss_latent；B0 baseline 重建 |
| 24B 同協議重建 | **COMPLETE · NO_LOCK** | B0/B1/B2 皆未五 target 全過 → 進入 24C |
| 24C 特徵 attribution | **RUNNING** | F3 重用 B1；F0/F1/F2/F4 formal 5-fold 訓練中 |
| 24D gdsc 診斷 | **DONE（診斷稿）** | `reports/round24/stage24d/` 已有 per-drug / coverage |
| 24E 受限優化 | **BLOCKED** | 待 24C top-2 feature |
| 24F lock | **BLOCKED** | 尚無 all-target PASS 候選 |
| 24G 最終報告 | **PARTIAL** | 本狀態報告 + 程式契約已落地；最終 lock 報告待 gate |

## Stage 24B gate（fold-mean DrugMacro AUROC）

基準：gdsc 0.5184 · tcga_only3 0.5586 · dapl 0.5356 · aacdr_gdsc 0.5582 · aacdr_tcga 0.4394

| Candidate | n_pass | gdsc | tcga_only3 | dapl | aacdr_gdsc | aacdr_tcga |
|-----------|-------:|-----:|-----------:|-----:|----------:|----------:|
| B0 pooled_mlp × own_plus_summary | 2/5 | （見 stage24a baseline） | | | | |
| B1 predictive × C32 | 1/5 | 0.500 | 0.454 | 0.466 | 0.527 | **0.473** |
| B2 XA fresh × C32 | 1/5 | 0.484 | 0.488 | 0.494 | 0.526 | **0.486** |

結論：R23 協議下的 winner 在 **eval3 5-fold** 重訓後未過硬 gate；與歷史 headline / 3-seed 協議不可直接等同。

## 已落地程式

```text
configs/round24/eval3.yaml
scripts/round24/run_round24.py
scripts/round24/train_biocda_on_round18_folds.py
scripts/round24/train_stage24b.py
scripts/round24/train_stage24c.py
scripts/round24/evaluate_stage24b.py
scripts/round24/analyze_features.py
scripts/round24/diagnose_gdsc_intersect13.py
scripts/round24/analyze_objective_alignment.py
scripts/round24/lock_round24_model.py
biocda/validation/round24_protocol.py
biocda/validation/round24_gate.py
tests/test_round24_{protocol,gate}.py
```

## 監控

```bash
docker exec DAPL bash -lc 'tail -f /workspace/DAPL/logs/round24/stage24c_formal.log'
docker exec DAPL bash -lc 'ls /workspace/DAPL/result/optimization_runs/round24_tcga_recovery/stage24c/'
```

## 科學敘事約束（不變）

- TCGA 未進入 loss / early stopping / checkpoint selection。
- GDSC 僅 diagnostic。
- 無 all-target PASS 時最終狀態必須為 `NO_LOCK`，不得以加權分數包裝成功。
