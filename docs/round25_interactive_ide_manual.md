# Round 25 — Interactive IDE Manual

**容器：** `DAPL` · `/workspace/DAPL`  
**執行：** 一律 `docker exec DAPL …`；本機不改環境。  
**Telegram：** 僅整輪 25A+25B+25C closure 完成後通知。  
**Git push：** 由使用者處理。

## 0. 核心決策

本輪**不**重新搜尋 downstream XA 拓撲。固定：

`Z64 + C32 → single query → fresh GIN atoms → no pooling → sample-to-atom XA → final query → response head`

本輪只搜尋 Stage 2 alignment / prototype strategy：

| ID | 內容 | 時機 |
|----|------|------|
| **S0** | dual WGAN + always-on prototype（現行基準） | 必跑 |
| **S2** | dual WGAN + **margin-gated** prototype | 第一優先 |
| **S1** | **AADA AE** 取代 global WGAN + always-on prototype | 第二優先 |
| **S3** | AADA + margin-gated | 僅 S1 或 S2 通過 25A |
| **S2b** | prototype distance band | 僅過度重疊證據成立 |

## 1. 權威來源（禁止舊記憶）

優先讀本機 HEAD：

1. `docs/round24_final_report.md` + 其引用的 `reports/*.json|csv`
2. Round 24 checkpoints / configs
3. `docs/round23_final_report.md`
4. `reports/biocda_xa_model_lock.json`（**必須維持 `REJECTED`**）
5. `reports/round23_paired_performance.csv`

### 嚴格禁止

- 覆寫 Round 23 歷史 REJECTED／改寫為 non-inferiority PASS
- 重加 graph pooling／pooled E3 transfer／pooled-teacher KD／summary11→C32
- 用 TCGA 選模
- 只依 domain discrepancy 選 Stage 2
- 混稱 `reconstruction_margin` 與 `prototype_upper_margin` / `prototype_lower_margin`
- 看完結果後更換 seeds 或 selection gate

## 2. 科學問題

- **25A：** always-on prototype 或 symmetric global ADV 是否過度壓縮 patient heterogeneity？margin-gate / AADA 能否保留？
- **25B：** 通過 25A 的 Stage2 重產 Z64/proto/PCA C32 後，固定 fresh XA 是否變好？
- **25C：** C32 對 prediction / attention 是否有獨立作用？

## 3. 三種 margin（欄位不得混用）

| 欄位 | 用途 |
|------|------|
| `prototype_upper_margin` | S2 hinge：距離 > δ 才拉近 |
| `prototype_lower_margin` | S2b band：過近輕推（`lower_weight<1`） |
| `reconstruction_margin` | S1 AADA：target 重建誤差 hinge |

`delta_c` 由 **source** minibatch centroid→EMA anchor 距離的 P90 估計；不足樣本用 global median；**warm-up 後 freeze + SHA256**。

## 4. CLI（皆在 Docker）

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && PYTHONPATH=/workspace/DAPL python3 scripts/audit_round25_repository.py --strict'

docker exec DAPL bash -lc 'cd /workspace/DAPL && PYTHONPATH=/workspace/DAPL python3 scripts/run_stage25a_screen.py --config config/round25_stage2_margin_screen.yaml --variants S0 S2 S1 --strict'

docker exec DAPL bash -lc 'cd /workspace/DAPL && PYTHONPATH=/workspace/DAPL python3 scripts/materialize_stage25a_decision.py --strict'
# 條件式：S3 / S2b
# 之後：export_stage25b_features → train/evaluate XA → stage25c → lock_round25
```

## 5. 互動式決策紀錄

### Stage 25A
- S0 status: PASS
- S2 status: FAIL（`prototype_hinge_active_fraction=0`）
- S1 status: PASS → **PROMOTE_S1**
- S3 required?: yes（因 S1 PASS）；結果 FAILED_MARGIN_INACTIVE
- S2b required?: no
- Selected / Reason / Blocking failures: S1；S2 hinge inactive

### Stage 25B
- B0 mean AUC 0.6303 / AUPRC 0.4151；B1 AUC 0.6241 / AUPRC 0.4258
- ΔAUC −0.0063；noninferior 2/3；worst-seed Δ −0.0599 → **KEEP_S0**

### Stage 25C
- C32 predictive effect: weak；attention effect: weak；final claim: **do_not_emphasize_C32**

## 6. 設計原則（實作）

1. **Parity：** S0/S2/S1 共用 data、cancer mapping、base encoder、conditional critic、EMA、seeds、budget；只允許 variant-specific 模組多參數。
2. **選模：** 必須含 source stability + target geometry + context rank + **固定下游 screen**；禁止只看 domain discrepancy。
3. **GPU：** 單卡 RTX 6000 Ada；Stage25A 以 seed×variant 工作佇列平行化（`--max-parallel` + `cuda mem fraction`），目標高利用率。
4. **XA：** 重用 `biocda/models/xa`（不新建衝突拓撲）；`biocda/stage2` 只負責 alignment 變體。
5. **Telegram：** 僅 `lock_round25.py` 成功後一次通知。
