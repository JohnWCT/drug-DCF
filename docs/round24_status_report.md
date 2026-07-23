# Round 24 — 執行狀態報告

**更新時間：** 2026-07-23  
**容器：** `DAPL` · `/workspace/DAPL`

## 總覽

| Stage / 實驗 | 狀態 | 說明 |
|--------------|------|------|
| 24A 協議/基準 | **PASS** | eval3 manifest；906→886 miss_latent；B0 baseline 重建 |
| 24B 同協議重建 | **COMPLETE · NO_LOCK** | B0/B1/B2 皆未五 target 全過 → 進入 24C |
| 24C 特徵 attribution | **RUNNING** | F3 重用 B1；F0 formal 進行中（F1/F2/F4 待跑） |
| **Train-source ablation** | **COMPLETE（診斷）** | NoHoldout / AACDR vs Ctrl；見下節 |
| 24D gdsc 診斷 | **DONE（診斷稿）** | `reports/round24/stage24d/` |
| 24E–F | **BLOCKED** | 待 24C / 尚無 all-target PASS |
| 24G | **PARTIAL** | 狀態報告已更新；最終 lock 待 gate |

## Stage 24B gate（摘要）

| Candidate | n_pass |
|-----------|-------:|
| B0 pooled_mlp × own_plus_summary | 2/5 |
| B1 predictive × C32 | 1/5 |
| B2 XA fresh × C32 | 1/5 |

## Train-source ablation（診斷，非 formal lock）

**假設：** 換 AACDR 訓練集、或取消 GDSC ~10% internal holdout，能否抬高 eval3 TCGA。  
**架構：** 與 B0 相同 `pooled_mlp × own_plus_summary`。  
**報告：** [`reports/round24/train_source_ablation/ablation_report.md`](../reports/round24/train_source_ablation/ablation_report.md)

| Arm | gdsc | tcga_only3 | dapl | aacdr_gdsc | aacdr_tcga | n_pass |
|-----|-----:|-----------:|-----:|-----------:|----------:|-------:|
| Ctrl（R18 + holdout） | 0.530 | 0.544 | 0.508 | 0.528 | 0.486 | 2/5 |
| NoHoldout（全量 GDSC2） | **0.570** | 0.485 | 0.482 | **0.565** | 0.497 | **3/5** |
| AACDR 訓練集 | 0.474 | 0.448 | **0.537** | 0.506 | 0.494 | 2/5 |

**Δ vs Ctrl：** NoHoldout 抬高 gdsc/aacdr_gdsc，但壓低 tcga_only3/dapl；AACDR 訓練集整體不升反降（僅 dapl 小幅上升）。  
**結論：** 資料源 / holdout **不是**五 target 全過的主解；繼續 Stage24C feature attribution。`any_all_target_pass=False`。

### 設計要點

1. **Ctrl**：重用 Stage24A baseline，不重訓。  
2. **NoHoldout**：`development ∪ internal_test` → 重建 formal 5-fold（無 10% holdout）。  
3. **AACDR**：eligible 過濾（miss_latent 為主）→ 全量 5-fold；early-stop 僅 source val DrugMacro。  
4. TCGA 五 target 推論與 eval3 gate 表對照；Telegram 僅完整 round 結束時發送。

## 已落地程式（含 ablation）

```text
scripts/round24/prepare_train_source_ablation.py
scripts/round24/run_train_source_ablation.py
reports/round24/train_source_ablation/
```

## 監控

```bash
docker exec DAPL bash -lc 'tail -f /workspace/DAPL/logs/round24/stage24c_formal.log'
docker exec DAPL bash -lc 'cat /workspace/DAPL/reports/round24/train_source_ablation/ablation_report.md'
```

## 科學敘事約束（不變）

- TCGA 未進入 loss / early stopping / checkpoint selection。
- Ablation 與 GDSC 結果僅 diagnostic，不寫入 formal lock。
- 無 all-target PASS 時最終狀態必須為 `NO_LOCK`。
