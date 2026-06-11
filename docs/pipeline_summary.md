# VAEwC / AEwC Prototype InfoNCE — 管線總覽與優化討論

**最後更新：** 2026-06-10  
**文件用途：** 供團隊成員快速理解管線、三輪實驗結論、指標關係，並作為 **Round 4 優化方向討論** 的共用依據。  
**主要程式：** `pretrain_VAEwC.py`、`pretrain_AEwC.py`、`tools/optimization_runner.py`  
**Pretrain 基準：** `result/pretrain_vaewc/exp_746`（嚴格 filter 8/8 通過）  
**下游 TCGA 基準：** `result/pretrain_vaewc_loss/`（exp_746 finetune，Avg TCGA = **0.5462**）  
**最新主線：** `result/optimization_runs/vaewc_proto_infonce_round3_exp746`（Stage 4 完成）  
**目前最佳下游（R4.1 finetune）：** **exp_035**（Avg TCGA = **0.5339**）；Round 3 exp_018 歷史 0.5695（舊 protocol）

### 閱讀導覽

| 章節 | 適合讀者 |
|------|----------|
| §1–2 | 想了解模型與管線流程 |
| §3–4 | 想了解品質門檻與三輪實驗結論 |
| **§4.4** | **三組差異總覽；重點：exp_746 vs Round 3 Control** |
| **§10** | **討論下一輪優化方向（建議從此讀起）** |
| §5–6 | 需要實際執行指令或找輸出檔 |

---

## 1. 模型架構

### 1.1 VAEwC（預設）

兩階段域適應 + GAN 對齊，latent 上疊加癌症分類與（可選）Prototype InfoNCE。

```text
CCLE (source) ──┐
                ├──► [Stage 1 Pretrain] shared + private VAE ×2 + cancer classifier
TCGA (target) ──┘         ortho_loss + VAE_loss + λ_cls·cls_loss (ramp)

[Stage 2 GAN]  WGAN-GP discriminator + generator/encoder update
               + λ_proto·Prototype InfoNCE（GAN epoch ramp）
               + λ_cls·cls（可每 step 更新 classifier）

[Stage 3 Finetune] 凍結 encoder → latent 上 GIN drug-response 分類器
```

| 元件 | 說明 |
|------|------|
| **shared VAE** | 跨域共享 encoder/decoder，latent 32-d |
| **private VAE ×2** | CCLE / TCGA 各一，與 shared latent 正交 |
| **cancer classifier** | shared latent → 18 類癌症（`PrimaryClassifier`） |
| **discriminator** | WGAN-GP，對齊 TCGA latent 分佈 |
| **Prototype InfoNCE** | batch 內 source+target 同癌別 prototype 對比（`tools/proto_infonce.py`） |

**骨幹：** `tools/model_opt.py` → `VAE`（`MODEL_BACKBONE = VAE`）

### 1.2 AEwC（對照）

`pretrain_AEwC.py` 與 VAEwC **共用同一套** `pretrain_VAEwC.py` 管線，僅將骨幹換成 **AE**（無 KL）：

```python
# pretrain_AEwC.py
core.MODEL_BACKBONE = AE
core.MODEL_TYPE_NAME = "AE"
```

用於驗證 VAE 的 KL 是否導致 fid/wasserstein 與 K-means 難以同時通過嚴格 filter。  
對照輸出：`result/benchmark_ae_vs_vae_exp746/`。

---

## 2. 優化管線（四階段）

```text
config/pretrain_sweeps/*.json
    → tools/optimization_config_generator.py
    → manifests/pretrain_sweep_manifest.csv

Stage 1  Pretrain   optimization_runner.py pretrain
Stage 2  Selection  optimization_runner.py select  (+ visualize_vaewc_filter.json)
Stage 3  Finetune   optimization_runner.py finetune
Stage 4  Aggregate  optimization_runner.py aggregate + report
```

| 階段 | 平行化 | 建議 GPU 參數 |
|------|--------|----------------|
| Pretrain | `max_parallel` 個子進程，各跑一組超參 | `batch_size=128`（CCLE≈1128 列，≥1128 會使 GAN 0 batch）, `max_parallel=20` |
| Finetune | `max_parallel` 個子進程，各跑一組 (model×combo) | `batch=4096`, `mini=1024`, `max_parallel=26` |

**重要：** Pretrain 的 `batch_size` 不可用 finetune 的 4096；`drop_last=True` 且 CCLE 僅 ~1128 樣本時，batch≥1128 會跳過整個 GAN。

---

## 3. 品質篩選（`config/visualize_vaewc_filter.json`）

| 類型 | 指標 | 方向 |
|------|------|------|
| Deconfounding | fid, mmd, wasserstein, kmeans_davies_bouldin | 越低越好 |
| 腫瘤保留 (K-means) | kmeans_ari, nmi, silhouette, calinski_harabasz | 越高越好 |

| 版本 | fid | mmd | wasserstein | 說明 |
|------|-----|-----|-------------|------|
| **嚴格**（exp_746 基準） | ≤ 16.95 | ≤ 0.019 | ≤ 0.50 | 8 項全過才進候選 |
| **寬鬆**（Round 3 finetune） | ≤ 30.0 | ≤ 0.05 | ≤ 0.70 | K-means 門檻不變 |
| **Round 4.1**（2026-06-11） | ≤ **35.0** | ≤ **0.06** | ≤ **1.05** | 再放寬 deconfounding；K-means 不變；60 池中 **6/60** 通過 |

Selection 門檻：`min_passing=10`、`require_controls=2`（至少 2 個 `lambda_proto=0` control）。

**下游主指標（使用者偏好）：** `Average_TCGA_AUC_mean`（各藥物 TCGA AUC 平均）。

---

## 4. Round 1–3 階段性結論

### Round 1 — `vaewc_proto_infonce_round1`

| 項目 | 結果 |
|------|------|
| Pretrain | 72/72，`lambda_proto`×`temperature`×`start`×`full` sweep |
| Selection | `--no-filter`（嚴格 filter 僅 1/82 通過） |
| Finetune | 40/40，`max_parallel=26` |
| **Average_TCGA_AUC 第 1** | **exp_031**（InfoNCE, λ=0.03）0.544 |
| Control 組平均 | 0.462 vs InfoNCE 0.488 |
| 限制 | 未啟用 K-means filter；pretrain 曾用 batch=4096 等問題已於後續修正 |

### Round 2 — `vaewc_proto_infonce_round2_kmeans`

| 項目 | 結果 |
|------|------|
| 目標 | 啟用嚴格 filter + 180 組弱 InfoNCE sweep |
| 失敗原因 1 | `batch_size=4096` → GAN 未訓練（`best_gan_epoch=0`） |
| 失敗原因 2 | 平行 pretrain 競爭 `_tmp_target_for_training.csv`（已修復） |
| Filter | 58 成功 run，**0/58 通過** |
| 結論 | 架構/參數問題，非 filter 本身；促成 Round 3 |

### Round 3 — `vaewc_proto_infonce_round3_exp746`（**目前主線，已完成**）

| 項目 | 結果 |
|------|------|
| 基準 | 對齊 **exp_746**：`lambda_cls=20`, `use_class_weight=true`, `batch=128` |
| Pretrain | **120/120** 成功，`max_parallel=20` |
| 嚴格 filter | **0/124** 通過（K-means 優於 exp_746，但 fid/wasserstein 偏高） |
| 寬鬆 filter | **13/124**（11 control + 2 InfoNCE）→ Top-10 finetune |
| 最佳 pretrain K-means | **exp_005**：ari=0.802, fid=26.9, wass=0.65 |
| Finetune | **40/40** 成功（Top-10 × 4 combo） |
| Aggregate | 見 `aggregate/aggregate_scores.csv`、`reports/final_selection_report.md` |

**Round 3 關鍵發現：**

1. **下游最佳為 exp_018**（control，λ_proto=0），超越 `pretrain_vaewc_loss` 基準 exp_746。
2. **InfoNCE 無明顯下游增益**：control 組平均 Avg TCGA 0.511 vs InfoNCE 0.511。
3. **Pretrain `score_total` 與下游幾乎無關**（r≈0.03）；不宜單靠現行 selection 分數預測下游。
4. **K-means 與下游 TCGA AUC 有部分正相關**（見 §4.3），值得在下一輪納入 selection 或 sweep 設計。
5. 高 `lambda_cls` 可拉高 K-means，但與 fid/wasserstein 存在取捨。

### 4.3 Pretrain 指標與下游 TCGA AUC 的關係（Round 3 Top-10）

以通過寬鬆 filter 並完成 finetune 的 **10 個模型** 為樣本，計算 pretrain 指標與 `Average_TCGA_AUC_mean` 的 Pearson 相關：

| Pretrain 指標 | vs Average_TCGA_AUC | 解讀 |
|---------------|---------------------|------|
| **score_kmeans**（K-means 子分數） | **r ≈ 0.52** | 目前與下游最相關的單一 pretrain 分數 |
| kmeans_silhouette | r ≈ 0.43 | 分群緊密度與下游有部分關聯 |
| kmeans_ari | r ≈ 0.32 | 中等正相關（Spearman r≈0.42, n=10） |
| kmeans_nmi | r ≈ 0.25 | 弱～中等正相關 |
| fid | r ≈ 0.25 | 單獨看 fid 與下游關係不明確 |
| wasserstein | r ≈ −0.09 | 幾乎無線性相關 |
| **score_total**（現行 selection 總分） | **r ≈ 0.03** | **幾乎無法預測下游** |

**典型案例（說明「部分相關」而非完全一致）：**

| Model | kmeans_ari | Avg TCGA | 備註 |
|-------|------------|----------|------|
| exp_005 | **0.802**（最高） | 0.538（第 3） | 高 K-means → 中等偏上下游 |
| exp_018 | 0.754 | **0.570**（最高） | 均衡：K-means 好 + wass 低 |
| exp_009 | 0.761 | 0.563（第 2） | 類似 exp_018 |
| exp_028 | **0.428**（最低） | 0.458（第 9） | 低 K-means → 低下游 |
| exp_004 | 0.758 | 0.474（第 8） | **反例**：K-means 高但下游差 |

**小結（供討論）：** K-means 可能是下游的必要條件之一（過低如 exp_028 時下游差），但非充分條件；**wasserstein 較低**（exp_018: 0.38）可能與高下游共同出現。下一輪可考慮「K-means + deconfounding」雙目標 selection，而非現行 8 項等權 `score_total`。

### 4.1 下游 TCGA 比較（vs `pretrain_vaewc_loss` 基準）

**比較方式：** 各模型 4 組 finetune combo 的平均；主指標 `Average_TCGA_AUC_mean`。

**基準 exp_746**（`result/pretrain_vaewc_loss/parameter_comparison_tcga_focus.csv`）：

| 指標 | exp_746 基準（4 combo 平均） |
|------|------------------------------|
| **Average_TCGA_AUC** | **0.5462** |
| Global_TCGA_AUC | 0.6168 |
| Test_AUC | 0.7974 |
| TCGA2_Average_TCGA_AUC | 0.5349 |

在 `pretrain_vaewc_loss` 所有歷史模型中，exp_746 的 Average_TCGA_AUC 仍排名第 1（其次 exp_104：0.5439）。

**Round 3 Top-10 下游排名：**

| 排名 | Model | Avg TCGA | Δ vs exp_746 | Global TCGA | λ_proto | 類型 |
|------|-------|----------|--------------|-------------|---------|------|
| **1** | **exp_018** | **0.5695** | **+0.023** | 0.6358 (+0.019) | 0 | control |
| 2 | exp_009 | 0.5631 | +0.017 | 0.5794 (−0.037) | 0 | control |
| 3 | exp_005 | 0.5376 | −0.009 | 0.5967 (−0.020) | 0 | control |
| 4 | exp_026 | 0.5256 | −0.021 | 0.5383 (−0.079) | 0 | control |
| 5 | exp_100 | 0.5153 | −0.031 | 0.6053 (−0.012) | 0.02 | InfoNCE |
| 6 | exp_020 | 0.5073 | −0.039 | 0.5704 (−0.046) | 0 | control |
| 7 | exp_083 | 0.5070 | −0.039 | 0.5848 (−0.032) | 0.02 | InfoNCE |
| 8 | exp_004 | 0.4739 | −0.072 | 0.5782 (−0.039) | 0 | control |
| 9 | exp_028 | 0.4583 | −0.088 | 0.5465 (−0.070) | 0 | control |
| 10 | exp_019 | 0.4507 | −0.096 | 0.5789 (−0.038) | 0 | control |

**exp_018 pretrain 特徵**（相對 exp_746）：`fid=20.3`（較差）、`wasserstein=0.38`（較好）、`kmeans_ari=0.754`（較好）、`proto_min_samples=2`、`proto_start_epoch=50`。

### 4.4 最佳結果三方對照（exp_746 vs Round 3 Control vs Round 3 InfoNCE）

> **建議閱讀順序：** ① 三組差異總覽 → ② **exp_746 vs Control（本章重點）** → ③ InfoNCE 簡述 → ④ 附錄數據表  
> 資料來源：`result/pretrain_vaewc/exp_746/`、`pretrain_vaewc_loss/`、`optimization_runs/vaewc_proto_infonce_round3_exp746/`

**代表模型：**

| 角色 | Model | 一句話 |
|------|-------|--------|
| **歷史基準** | **exp_746** | 嚴格 filter 全過的 control；下游 Avg TCGA = 0.5462 |
| **Round 3 Control** | **exp_018** | 同為 λ_proto=0，Round 3 下游最佳（Avg TCGA = 0.5695） |
| **Round 3 InfoNCE** | **exp_100** | λ_proto=0.02；InfoNCE 組下游最佳但仍低於 exp_746 |

---

#### 4.4.0 三組主要差異（先讀這段）

三組的**本質差異**不在 finetune，而在 **pretrain 階段是否啟用 Prototype InfoNCE**，以及由此產生的 **latent 幾何取捨**：

| 維度 | exp_746（基準） | Round 3 Control（exp_018） | Round 3 InfoNCE（exp_100） |
|------|-----------------|---------------------------|---------------------------|
| **λ_proto** | 0 | 0 | **0.02** |
| **訓練配方** | 歷史單次 pretrain | Round 3 大規模 sweep 中的 control 分支 | 同上，但 GAN 階段加 InfoNCE loss |
| **Pretrain 強項** | **fid / mmd 最佳**（deconfounding） | **wasserstein 最低 + K-means 較好**（腫瘤結構） | 有 proto 訓練訊號，但 K-means 變差 |
| **嚴格 filter** | ✅ 8/8 | ❌（fid 超標） | ❌ |
| **下游 TCGA** | 歷史基準 | **Avg TCGA 最高（+4.3%）** | 未超越 exp_746 |
| **一句話** | deconfounding 與 filter 的「均衡基準」 | 用較差 fid **換** 更好腫瘤結構 → TCGA 提升 | InfoNCE **未帶來**下游增益 |

```text
  exp_746 ──────────►  deconfounding 優先  ──►  嚴格 filter ✅，TCGA 基準
       │
       │  Round 3 重跑（λ_proto=0，超參微調 + 不同隨機軌跡）
       ▼
  exp_018 ──────────►  K-means + 低 wass 優先  ──►  filter 較差，TCGA ↑
       │
       │  同一 sweep 加上 λ_proto>0
       ▼
  exp_100 ──────────►  InfoNCE 約束 latent  ──►  K-means ↓，TCGA 無提升
```

**與你主要疑問的關係：** exp_746 與 Round 3 Control（exp_018）**都沒有啟用 InfoNCE**；差異來自 Round 3 重訓後 latent 落在不同的品質取捨點，而非「有無 InfoNCE」。InfoNCE 是第三組的獨立對照，用來驗證「加 contrastive loss 是否有幫助」——目前答案是否定的。

---

#### 4.4.1 【重點】exp_746 vs Round 3 Control（exp_018）差在哪？

##### 相同點：訓練配方幾乎一致

兩者皆為 **純 control（λ_proto=0）**，核心超參對齊 exp_746 基準：

| 超參 | exp_746 | exp_018 |
|------|---------|---------|
| λ_cls | 20 | 20 |
| use_class_weight | true | true |
| cls_start / cls_full | 40 / 90 | 40 / 90 |
| batch_size | 128 | 128 |
| encoder_dims / latent | [256,128] / 32 | 同左 |
| gan_patience / early_stop | 30 / loss | 同左 |
| gan_gen_update_interval | 5 | 5 |
| **lambda_proto（實際生效）** | **0** | **0** |

exp_018 的 config 雖含 `proto_min_samples=2`、`proto_start_epoch=50` 等 sweep 欄位，但因 **λ_proto=0**，InfoNCE 相關 loss **不會生效**。  
因此兩組差異**不是**「有無加 InfoNCE」，而是 **同一套配方下，不同訓練 run 收斂到不同的 latent 品質平衡點**。

##### 不同點 1：Pretrain 品質的取捨（核心差異）

| 指標 | 方向 | exp_746 | exp_018 | 誰較好 | 解讀 |
|------|------|---------|---------|--------|------|
| **fid** | ↓ | **16.9** | 20.3 | 746 | 018 域分佈重疊較差（deconfounding 較弱） |
| **mmd** | ↓ | **0.019** | 0.036 | 746 | 同上 |
| **wasserstein** | ↓ | 0.48 | **0.38** | **018** | 018 跨域距離更低 |
| **kmeans_ari** | ↑ | 0.679 | **0.754** | **018** | 018 癌別分群更準（+11%） |
| **kmeans_nmi** | ↑ | **0.824** | 0.817 | 746 | 差距很小 |
| **kmeans_silhouette** | ↑ | 0.386 | **0.387** | 平手 | 幾乎相同 |
| **kmeans_calinski** | ↑ | 467 | **555** | **018** | 018 類間分離更清楚 |
| **kmeans_davies_bouldin** | ↓ | 1.02 | **0.94** | **018** | 018 分群更緊湊 |
| **嚴格 filter** | — | **✅ 8/8** | ❌ | 746 | 018 因 fid>16.95 未通過 |

**一句話：** exp_746 是「deconfounding 達標」的均衡解；exp_018 是「犧牲 fid、換取更低 wass + 更好 K-means」的解。這也解釋為何 018 能過**寬鬆** filter 卻过不了**嚴格** filter。

##### 不同點 2：下游表現（finetune 後）

| 指標 | exp_746 | exp_018 | Δ018−746 | 誰較好 |
|------|---------|---------|----------|--------|
| **Average_TCGA_AUC**（主指標） | 0.5462 | **0.5695** | **+0.023 (+4.3%)** | **018** |
| **Global_TCGA_AUC** | 0.6168 | **0.6358** | +0.019 | **018** |
| **Average_TCGA_AUPRC** | 0.6398 | **0.6502** | +0.010 | **018** |
| Global_TCGA_AUPRC | **0.6978** | 0.6902 | −0.008 | 746 |
| TCGA2_Average_TCGA_AUC | **0.5349** | 0.5198 | −0.015 | 746 |
| Test_AUC（CCLE hold-out） | **0.7974** | 0.7072 | −0.090 | 746 |
| Test_AUPRC（CCLE） | **0.6883** | 0.4370 | −0.251 | 746 |

**一句話：** exp_018 在 **TCGA 藥物預測（主指標）全面勝出**，但 **CCLE hold-out 明顯退步**，且 TCGA2 / Global AUPRC 略輸 746。這是典型的 **源域↔目標域取捨**：latent 更貼近 TCGA 藥效任務，但源域（CCLE）泛化變差。

##### 不同點 3：代表藥物層級（為何 TCGA 會升）

| 藥物 | exp_746 AUC | exp_018 AUC | 變化 | 說明 |
|------|-------------|-------------|------|------|
| **Doxorubicin** | 0.337 | **0.680** | **+0.34** | 018 最大單藥增益 |
| **Sorafenib** | 0.452 | **0.594** | +0.14 | |
| **Gemcitabine** | 0.496 | **0.570** | +0.07 | |
| **Temozolomide** | 0.485 | **0.563** | +0.08 | |
| Etoposide | **0.664** | 0.588 | −0.08 | 018 反而下降 |
| Cisplatin | 0.624 | 0.564 | −0.06 | |

018 的提升**不是每種藥都漲**，而是少數藥物（尤其 Doxorubicin）大幅拉抬平均；Etoposide 等藥物 746 仍較好。

##### 直接回答：746 vs Control，我該怎麼理解？

| 你的疑問 | 答案 |
|----------|------|
| 配方有大幅改動嗎？ | **沒有**。兩者都是 λ_proto=0、λ_cls=20 的 control；差在 Round 3 重訓後收斂點不同。 |
| 為何 pretrain filter 較差，下游反而更好？ | 嚴格 filter 重視 **fid**；下游與 **K-means + wasserstein** 較相關（§4.3）。018 走了「腫瘤結構優先」路線。 |
| 018 是全面碾壓 746 嗎？ | **否**。TCGA AUC 贏、CCLE Test 輸、Global AUPRC / TCGA2 略輸；是取捨而非全面升級。 |
| Round 3 優化有效嗎？ | **有效**——在 TCGA 主指標上 +4.3%；代價是放棄嚴格 filter 數字與部分 CCLE 表現。 |
| 下一輪該錨定誰？ | 以 **exp_018 的收斂區域**（低 wass + 高 K-means）為目標，同時監控 fid 與 CCLE Test。 |

---

#### 4.4.2 Round 3 InfoNCE（exp_100）— 簡要對照

InfoNCE 組與上述兩組的差異**只有 pretrain 多了一項 λ_proto=0.02 的 contrastive loss**；finetune protocol 相同。

| 對比 | exp_746 | exp_018 | exp_100 |
|------|---------|---------|---------|
| Average_TCGA_AUC | 0.546 | **0.570** | 0.515 |
| kmeans_ari | 0.679 | **0.754** | 0.621 |
| wasserstein | 0.48 | **0.38** | 0.56 |

**結論：** InfoNCE 既未改善 K-means，也未提升下游；**不建議作為 Round 4 主線**（可放附錄）。Round 3 的真正收穫來自 **control 分支的 re-sweep（exp_018）**，而非 InfoNCE。

---

#### 4.4.3 附錄：完整數據表（Pretrain + 下游）

| 指標 | exp_746（基準） | exp_018（R3 Control） | exp_100（R3 InfoNCE） | Δ018 vs 746 | Δ100 vs 746 |
|------|-----------------|----------------------|----------------------|-------------|-------------|
| **λ_proto** | 0 | 0 | **0.02** | — | +InfoNCE |
| **嚴格 filter** | ✅ 8/8 | ❌（fid 超標） | ❌（fid 超標） | — | — |
| **寬鬆 filter** | ✅ | ✅ | ✅ | — | — |
| | | | | | |
| **fid** ↓ | **16.9** | 20.3 | 19.5 | +3.4 ↑ | +2.6 ↑ |
| **mmd** ↓ | **0.019** | 0.036 | 0.039 | +0.017 ↑ | +0.020 ↑ |
| **wasserstein** ↓ | 0.48 | **0.38** | 0.56 | **−0.10 ↓** | +0.08 ↑ |
| **kmeans_ari** ↑ | 0.679 | **0.754** | 0.621 | **+0.075 ↑** | −0.058 ↓ |
| **kmeans_nmi** ↑ | 0.824 | 0.817 | 0.773 | −0.007 | −0.051 |
| **kmeans_silhouette** ↑ | 0.386 | **0.387** | 0.288 | +0.001 | −0.098 ↓ |
| **kmeans_calinski** ↑ | 467 | **555** | 386 | +88 | −81 |
| **kmeans_davies_bouldin** ↓ | 1.02 | **0.94** | 1.39 | −0.08 | +0.37 ↑ |
| **score_total** | — | **0.794** | 0.516 | — | — |
| **score_kmeans** | — | **0.833** | 0.100 | — | — |
| **score_deconfounding** | — | **0.778** | 0.694 | — | — |
| | | | | | |
| **Average_TCGA_AUC** ↑ | 0.5462 | **0.5695** | 0.5153 | **+0.023 (+4.3%)** | −0.031 (−5.7%) |
| **Global_TCGA_AUC** ↑ | 0.6168 | **0.6358** | 0.6053 | **+0.019 (+3.0%)** | −0.012 (−1.9%) |
| **Average_TCGA_AUPRC** ↑ | 0.6398 | **0.6502** | 0.6377 | **+0.010 (+1.6%)** | −0.002 (−0.3%) |
| **Global_TCGA_AUPRC** ↑ | 0.6978 | 0.6902 | 0.6671 | −0.008 | −0.031 |
| **TCGA2_Average_TCGA_AUC** ↑ | 0.5349 | 0.5198 | 0.5317 | −0.015 | −0.003 |
| **Test_AUC**（CCLE）↑ | 0.7974 | 0.7072 | 0.7798 | −0.090 | −0.018 |
| **Test_AUPRC**（CCLE）↑ | 0.6883 | 0.4370 | 0.5381 | −0.251 | −0.150 |

**Finetune 聚合方式：** 各模型 4 組 combo 平均（與 `aggregate_scores.csv` 一致）。

---

#### 4.4.4 Pretrain 品質細項（附錄）

**Deconfounding（CCLE vs TCGA latent 分佈）：**

| 指標 | exp_746 | exp_018 | exp_100 | 解讀 |
|------|---------|---------|---------|------|
| fid | **16.9** ✅ | 20.3 ❌ | 19.5 ❌ | 基準 deconfounding 仍最佳；R3 模型 K-means 較好但 fid 偏高 |
| wasserstein | 0.48 ✅ | **0.38** ✅ | 0.56 ❌ | exp_018 wass 最低，可能與其高下游有關 |
| mmd | **0.019** | 0.036 | 0.039 | 基準 mmd 最低 |
| best_gan_epoch | 247 | 216 | 47 | InfoNCE 較早停止 GAN（epoch 47） |

**K-means 腫瘤保留（18 類癌症分群 vs 真實標籤）：**

| 指標 | exp_746 | exp_018 | exp_100 | 解讀 |
|------|---------|---------|---------|------|
| kmeans_ari | 0.679 | **0.754** | 0.621 | exp_018 ARI 比基準高 11%；InfoNCE 反而下降 |
| kmeans_nmi | **0.824** | 0.817 | 0.773 | 基準 NMI 仍略高 |
| silhouette | 0.386 | **0.387** | 0.288 | InfoNCE 分群緊密度明顯較差 |
| calinski_harabasz | 467 | **555** | 386 | exp_018 類間分離度最佳 |
| davies_bouldin | **1.02** | **0.94** | 1.39 | 越低越好；InfoNCE 分群品質最差 |

**Selection 子分數（僅 Round 3 候選有；exp_746 無此欄）：**

| 子分數 | exp_018 | exp_100 | 說明 |
|--------|---------|---------|------|
| score_kmeans | **0.833** | 0.100 | exp_018 K-means 子分極高；InfoNCE 極低 |
| score_deconfounding | **0.778** | 0.694 | exp_018 在 deconfounding 也優於 InfoNCE |
| score_total | **0.794**（Top-10 第 2） | 0.516（第 7） | 總分與下游排名大致一致（exp_018 高、exp_100 中） |

**關鍵超參差異（三者共同：λ_cls=20, use_class_weight=true, batch=128）：**

| 超參 | exp_746 | exp_018 | exp_100 |
|------|---------|---------|---------|
| lambda_proto | 0 | 0 | **0.02** |
| proto_temperature | — | 0.7 | **1.0** |
| proto_start_epoch | — | 50 | 50 |
| proto_full_epoch | — | 80 | **120** |
| proto_min_samples | — | **2** | 2 |
| cls_start / cls_full | 40 / 90 | 40 / 90 | 40 / 90 |

**InfoNCE 訓練訊號（exp_100，GAN 最後一步）：** proto_loss≈2.19、proto_acc≈0.85、proto_valid_class_count=18；對比 exp_018 全為 0（純 control）。

---

#### 4.4.5 下游 TCGA：整體與代表藥物（附錄）

**整體 TCGA 指標：**

| 指標 | exp_746 | exp_018 | exp_100 | 最佳 |
|------|---------|---------|---------|------|
| Average_TCGA_AUC | 0.546 | **0.570** | 0.515 | **exp_018** |
| Global_TCGA_AUC | 0.617 | **0.636** | 0.605 | **exp_018** |
| Average_TCGA_AUPRC | 0.640 | **0.650** | 0.638 | **exp_018** |
| Global_TCGA_AUPRC | **0.698** | 0.690 | 0.667 | **exp_746** |
| TCGA2_Average_TCGA_AUC | **0.535** | 0.520 | 0.532 | **exp_746** |

**代表藥物 AUC / AUPRC（4 combo 平均）：**

| 藥物 | exp_746 AUC | exp_018 AUC | exp_100 AUC | exp_746 AUPRC | exp_018 AUPRC | exp_100 AUPRC |
|------|-------------|-------------|-------------|---------------|---------------|---------------|
| Etoposide | **0.664** | 0.588 | 0.417 | **0.932** | 0.779 | 0.690 |
| Cisplatin | 0.624 | 0.564 | 0.529 | 0.848 | **0.813** | **0.807** |
| Doxorubicin | 0.337 | **0.680** | **0.588** | 0.543 | **0.755** | **0.729** |
| Paclitaxel | 0.551 | **0.554** | **0.594** | 0.719 | 0.695 | **0.754** |
| Sorafenib | 0.452 | **0.594** | 0.313 | 0.180 | **0.264** | 0.257 |
| Gemcitabine | 0.496 | **0.570** | 0.533 | 0.467 | **0.572** | 0.536 |
| Temozolomide | 0.485 | **0.563** | 0.545 | 0.118 | 0.135 | **0.168** |

**CCLE hold-out（參考，非主指標）：**

| 指標 | exp_746 | exp_018 | exp_100 |
|------|---------|---------|---------|
| Test_AUC | **0.797** | 0.707 | 0.780 |
| Test_AUPRC | **0.688** | 0.437 | 0.538 |

exp_018 的 TCGA 提升伴隨 CCLE Test AUC 明顯下降（0.797→0.707），顯示 **域轉移取捨**；InfoNCE（exp_100）CCLE 表現介於兩者之間，但 TCGA 未超越基準。

---

#### 4.4.6 輸出檔案快速連結

| 內容 | 路徑 |
|------|------|
| exp_746 pretrain | `result/pretrain_vaewc/exp_746/gan_metrics.json` |
| exp_746 下游 | `result/pretrain_vaewc_loss/pretrain_tcga_model_summary.csv` |
| exp_018 / exp_100 pretrain | `.../round3_exp746/selection/pretrain_top10.csv` |
| 下游聚合 | `.../round3_exp746/aggregate/aggregate_scores.csv` |
| t-SNE 圖 | `.../pretrain/exp_018/tsne_gan_best.png`、`exp_100/tsne_gan_best.png` |

### 4.2 AE vs VAE Pretrain 對照（`result/benchmark_ae_vs_vae_exp746/`）

| Model | 骨幹 | λ_proto | fid | wasserstein | kmeans_ari | 嚴格 filter | 寬鬆 filter |
|-------|------|---------|-----|-------------|------------|-------------|-------------|
| exp_746 | VAE | 0 | 16.9 | 0.48 | 0.679 | ✅ | ✅ |
| exp_005 | VAE | 0 | 26.9 | 0.65 | 0.802 | ❌ | ✅ |
| AE exp_001 | AE | 0 | 25.8 | 0.67 | **0.158** | ❌ | ❌ |
| AE exp_002 | AE | 0.02 | 25.8 | 0.73 | **0.204** | ❌ | ❌ |
| AE exp_003 | AE | 0.01 | 27.5 | **0.82** | 0.714 | ❌ | ❌ |

**AE 結論：** 不建議以 AE 取代 VAE。無 InfoNCE 的 AE K-means 崩潰；AE + 弱 InfoNCE（exp_003）可恢復 K-means，但 wasserstein 惡化，寬鬆 filter 也未通過。

### 基準 exp_746

| 角色 | 路徑 | 說明 |
|------|------|------|
| Pretrain 權重 | `result/pretrain_vaewc/exp_746/` | 嚴格 filter 8/8 通過 |
| 下游 finetune | `result/pretrain_vaewc_loss/` | TCGA eval 對照基準（Avg TCGA = 0.5462） |

| Pretrain 指標 | 值 | 嚴格 filter |
|---------------|-----|-------------|
| lambda_cls | 20 | — |
| fid | 16.9 | ✅ |
| wasserstein | 0.48 | ✅ |
| kmeans_ari | 0.679 | ✅ |
| InfoNCE | 無（λ_proto=0） | 8/8 通過 |

---

## 5. 程式執行方式

### 5.1 Docker 環境

```bash
# 容器名稱：DAPL，工作目錄：/workspace/DAPL
docker exec -w /workspace/DAPL DAPL <command>
```

### 5.2 單次 Pretrain（VAEwC）

```bash
docker exec -w /workspace/DAPL DAPL python3 pretrain_VAEwC.py \
  --config config/params_proto_base_exp746_vaewc.json \
  --outfolder result/pretrain_vaewc \
  --target_domain tcga \
  --overlap_tcga data/TCGA/PMID27354694_DR_OMICS_ad.csv \
  --batch-size 128
```

### 5.3 單次 Pretrain（AEwC）

```bash
docker exec -w /workspace/DAPL DAPL python3 pretrain_AEwC.py \
  --config config/params_benchmark_aewc_exp746.json \
  --outfolder result/benchmark_ae_vs_vae_exp746/aewc \
  --target_domain tcga \
  --overlap_tcga data/TCGA/PMID27354694_DR_OMICS_ad.csv \
  --batch-size 128
```

### 5.4 完整優化管線（Round 3 範本）

```bash
# 1) 產生 manifest
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_config_generator.py \
  --sweep-spec config/pretrain_sweeps/vaewc_proto_infonce_round3_exp746.json --force

# 2) Pretrain
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py pretrain \
  --manifest result/optimization_runs/vaewc_proto_infonce_round3_exp746/manifests/pretrain_sweep_manifest.csv \
  --run-dir result/optimization_runs/vaewc_proto_infonce_round3_exp746 \
  --batch-size 128 --max-parallel 20

# 3) Selection（寬鬆 filter 範例）
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py select \
  --run-dir result/optimization_runs/vaewc_proto_infonce_round3_exp746 \
  --filter-config config/visualize_vaewc_filter.json \
  --min-passing 10 --require-controls 2

# 4) Finetune
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py finetune \
  --manifest result/optimization_runs/vaewc_proto_infonce_round3_exp746/manifests/finetune_dispatch_manifest.csv \
  --run-dir result/optimization_runs/vaewc_proto_infonce_round3_exp746 \
  --top10 result/optimization_runs/vaewc_proto_infonce_round3_exp746/selection/pretrain_top10.csv \
  --epochs 1000 --batch-size 4096 --mini-batch-size 1024 --max-parallel 26

# 5) 聚合與報告
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py aggregate \
  --run-dir result/optimization_runs/vaewc_proto_infonce_round3_exp746
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py report \
  --run-dir result/optimization_runs/vaewc_proto_infonce_round3_exp746
```

### 5.5 一鍵腳本

| 腳本 | 用途 |
|------|------|
| `tools/run_round3_exp746_infonce.sh` | Round 3 全流程 |
| `tools/run_round3_finetune_relaxed.sh` | 寬鬆 filter + finetune + aggregate |
| `tools/run_ae_vs_vae_benchmark.sh` | AE vs VAE 嚴格 filter 對照 |
| `tools/run_round3_after_pretrain.sh` | Pretrain 完成後接 select/finetune |

### 5.6 輔助工具

```bash
# 更新即時報告
python3 tools/update_running_report.py --run-dir result/optimization_runs/<run_id>

# 從 latent 重繪 t-SNE
python3 plot_tsne_from_latent.py --exp_dir result/.../pretrain/exp_XXX

# Pretrain 品質對照
python3 tools/compare_pretrain_filter_metrics.py --aewc-dir ... --vaewc-ids exp_746
```

---

## 6. 輸出目錄結構

### 6.1 單次 Pretrain（`exp_XXX/`）

```text
result/<outfolder>/exp_XXX/
├── params.json                 # 完整超參
├── gan_metrics.json / .csv     # fid, mmd, wasserstein, kmeans_*
├── pretrain_loss.csv
├── pretrain_eval_loss.csv
├── d_loss.csv / g_loss.csv     # GAN 訓練曲線
├── after_traingan_*.pth        # GAN 最佳權重
├── ccle_latent_dict.pkl
├── tcga_latent_dict.pkl
├── tsne_gan_best.png           # 雙面板 t-SNE（domain + cancer type）
├── gan_learning_curve.png
└── run_summary.json
```

### 6.2 優化 Run（`result/optimization_runs/<run_id>/`）

```text
<run_id>/
├── manifests/
│   ├── pretrain_sweep_manifest.csv
│   └── finetune_dispatch_manifest.csv
├── pretrain/
│   └── exp_XXX/                # 同上
├── selection/
│   ├── pretrain_all_candidates.csv
│   ├── pretrain_filtered_candidates.csv
│   ├── filter_threshold_report.csv
│   ├── pretrain_top10.csv
│   └── model_select.csv
├── finetune/
│   └── exp_XXX/combo_YY/
│       ├── finetune_config_used.json
│       ├── parameter_comparison_tcga_focus.csv
│       └── exp_XXX/param_001/
│           ├── tcga_eval_predictions.csv
│           └── metrics_summary/
├── aggregate/
│   ├── aggregate_scores.csv    # 下游主表（含 Average_TCGA_AUC_mean）
│   └── merged_finetune_tcga_focus.csv
├── reports/
│   ├── final_selection_report.md
│   └── run_summary.json
├── logs/
│   ├── pretrain/
│   └── finetune/
└── running_report.md           # 即時進度
```

### 6.3 設定檔

```text
config/
├── visualize_vaewc_filter.json           # 品質門檻
├── params_proto_base_exp746_vaewc.json   # Round 3 VAE 基準
├── params_benchmark_aewc_exp746.json     # AE 對照
├── pretrain_sweeps/
│   └── vaewc_proto_infonce_round3_exp746.json
└── generated/
    └── vaewc_proto_infonce_round3_exp746/  # 自動產生的 job config
```

### 6.4 保留的結果目錄（精簡後）

| 路徑 | 說明 |
|------|------|
| `result/pretrain_vaewc/exp_746/` | Pretrain 基準（嚴格 filter 通過） |
| `result/pretrain_vaewc_loss/` | **下游 TCGA 對照基準**（exp_746 finetune） |
| `result/optimization_runs/vaewc_proto_infonce_round3_exp746/` | **最新主線**（Stage 4 完成） |
| `result/benchmark_ae_vs_vae_exp746/` | AE vs VAE 架構對照（3/3 完成） |

---

## 7. 核心模組對照

| 模組 | 路徑 |
|------|------|
| Prototype InfoNCE | `tools/proto_infonce.py` |
| λ_proto ramp | `tools/pretrain_proto_schedule.py` |
| Config / manifest | `tools/optimization_config_generator.py` |
| 執行器 | `tools/optimization_runner.py` |
| Selection + filter | `tools/optimization_selection.py`, `visualize_vaewc_results.py` |
| 下游聚合 | `tools/optimization_report.py` |
| Finetune | `step1_finetune_latent_pipeline_All_split.py` |

---

## 8. 相關文件

- `docs/design.md` — 系統設計
- `docs/proposal.md` — 提案與 sweep 定義
- `docs/prompt.md` — 實作規格與驗收

---

## 9. 名詞簡釋（給未參與實作者）

| 名詞 | 一句話 |
|------|--------|
| **fid / wasserstein** | 衡量 CCLE vs TCGA latent 分佈有多像；越低表示批次效應越小（deconfounding） |
| **K-means ARI/NMI** | 在 latent 上做 K-means，看分群與真實癌別是否一致；越高表示腫瘤類型結構保留越好 |
| **Prototype InfoNCE** | GAN 階段額外 loss：同癌別跨 domain 拉近、異癌別推開；Round 3 下游無明顯增益 |
| **Average_TCGA_AUC** | 各藥物 TCGA hold-out AUC 的平均；**目前下游主指標** |
| **嚴格 / 寬鬆 filter** | 嚴格：fid≤16.95、wass≤0.50 等 8 項全過；寬鬆：fid≤30、wass≤0.70，K-means 門檻不變 |
| **exp_746** | 歷史 pretrain 基準（嚴格 filter 通過）；其 finetune 結果在 `pretrain_vaewc_loss` |
| **exp_018** | Round 3 下游最佳 pretrain（Avg TCGA 0.5695） |

---

## 10. Round 4 優化方向討論（共用草案）

> 本章供與合作者討論，尚未定案。歡迎在會議中直接修改取捨或補充假設。

### 10.1 已達成的方向共識（2026-06-10）

| 項目 | 共識 |
|------|------|
| **論文敘事** | 需兼顧 deconfounding、腫瘤保留、下游預測，但可接受在**嚴格 filter 數字**上妥協 |
| **主比較對象** | vs **exp_746**（證明優化有效） |
| **下游成功標準** | `Average_TCGA_AUC_mean` 超越基準 0.5462；目標 > **0.57** |
| **InfoNCE** | **可選**；若無增益可放附錄，不作 Round 4 主線 |
| **骨幹** | 繼續 **VAEwC**；AEwC 對照已結束，不建議切換 |
| **算力與規模** | 現行平行速度下，**1–2 週內可承受 Round 3 等級大規模搜尋**（~120 pretrain + 40 finetune） |
| **新假設** | **K-means 與 TCGA AUC 有部分相關**（§4.3）→ selection / sweep 應顯式納入 K-means，而非僅靠 `score_total` |
| **搜尋規模** | 傾向 **Round 3 等級大規模**（~120 pretrain），非小範圍局部搜尋 |

### 10.2 待團隊討論的開放問題

1. **Selection 公式是否改版？**  
   - 現況：`score_total` 與下游 r≈0.03。  
   - 提案：`score_new = α·score_kmeans + β·score_deconfounding`（例如 α=0.6, β=0.4），或兩階段 filter（先 K-means 門檻，再按 wasserstein 排序）。

2. **嚴格 filter 在論文中如何呈現？**  
   - 選項 A：主線用寬鬆 filter 選模，Discussion 報告嚴格通過率。  
   - 選項 B：主表 + 附表分別呈現兩套門檻。  
   - 選項 C：自定義「論文門檻」（例如 fid≤22、wass≤0.55、ari≥0.70）。

3. **Round 4 sweep 範圍？**（目前傾向大規模）  
   - **預設方案**：Round 3 等級 **~100–120 jobs**，擴展 `lambda_cls`、`cls_start/full`、`proto_min_samples`、GAN patience；錨點參考 exp_018。  
   - 備選：若會議決議縮減，可改 ~60 jobs + 3 seed。  
   - InfoNCE 附錄支線（可選）：Top-5 control × λ∈{0.005, 0.01}（~10 jobs）。

4. **相關性是否要擴大樣本驗證？**  
   - 目前 r 值僅基於 Top-10（n=10），統計功效不足。  
   - 可討論：是否對 Round 3 全部 124 個 pretrain 做 **proxy 下游**（例如 frozen latent + 線性 probe）以驗證 K-means–AUC 關係，再決定 selection 權重。

### 10.3 Round 4 草案（大規模搜尋版）

```text
Run ID（建議）: vaewc_round4_exp018_kmeans

Phase 1 — Pretrain（~100–120 jobs, max_parallel=20, ~2–3 天）
  錨點參考: exp_018 (lambda_cls=20, proto_min_samples=2, proto_start=50)
  Sweep 主軸:
    - lambda_cls:        [15, 20, 25]          # 影響 K-means 與 deconfounding 取捨
    - proto_min_samples: [1, 2, 3]
    - cls_start_epoch:   [30, 40, 50]
    - cls_full_epoch:    [80, 90]
    - gan_patience:      [20, 30]
    - lambda_proto:      [0]                   # 主線 control
  固定: batch=128, use_class_weight=true, backbone=VAE

Phase 2 — Selection
  Filter: 寬鬆版（fid≤30, wass≤0.70, K-means 門檻不變）
  Ranking: 提案改用 score_kmeans 加權（待 §10.2 討論定案）
  輸出: Top-10 → finetune

Phase 3 — Finetune（40 jobs, max_parallel=26, ~1 天）
  與 Round 3 相同 protocol（4 combo × Top-10）

Phase 4 — 評估
  主指標: Average_TCGA_AUC_mean vs pretrain_vaewc_loss/exp_746
  次指標: Global_TCGA_AUC, 各藥物 AUC, K-means vs 下游散點圖

附錄支線（可選, ~10 jobs）
  Round 4 最佳 control × λ_proto ∈ {0.005, 0.01}
```

### 10.4 時程估計（Docker 平行，與 Round 3 相同設定）

| 階段 | Jobs | max_parallel | 預估牆鐘時間 |
|------|------|--------------|--------------|
| Pretrain | 100–120 | 20 | 2–3 天 |
| Selection + 報告 | — | — | 數小時 |
| Finetune | 40 | 26 | ~1 天 |
| Aggregate + 圖表 | — | — | 1 天 |
| **合計** | | | **約 4–6 天計算 + 2–3 天分析撰寫** |

在 1–2 週交付窗口內，可保留 **1 次迭代**（若第一輪 Top-1 不穩，縮小 sweep 重跑）。

### 10.5 決策檢核表（會議用）

討論結束後請勾選：

- [ ] Selection 公式：維持現行 / 改為 K-means 加權 / 其他：_______
- [ ] Filter 門檻：寬鬆 / 自定義論文門檻 / 嚴格
- [ ] Sweep 規模：~60 / ~120 / 其他：_______
- [ ] InfoNCE 附錄支線：跑 / 不跑
- [ ] 下游主報指標：Average_TCGA_AUC / Global_TCGA_AUC / 特定藥物：_______
- [ ] 是否做 124 模型 proxy 下游驗證 K-means 相關性：是 / 否

### 10.6 已知風險與緩解

| 風險 | 緩解 |
|------|------|
| K-means–AUC 相關僅 n=10，過擬合解讀 | 擴大 finetune 樣本或做 proxy 驗證 |
| 高 lambda_cls → fid 升高 | sweep 含 λ_cls=15 對照；selection 保留 wasserstein 硬門檻 |
| 高 parallel 偶發缺 tsne / 路徑競爭 | 已修 temp CSV；建議跑完做 manifest 稽核 |
| exp_018 提升不可重現 | sweep 含 2–3 random seed |

---

## 11. Round 4 設計（cross-domain InfoNCE + K-means selection）

**Branch：** `round4-cross-domain-infonce-selection`  
**主 Run ID：** `vaewc_round4_cross_domain_infonce`（~48 pretrain jobs）

### 11.1 為何 Round 3 combined InfoNCE 沒有帶來下游增益

Round 3 使用 **combined prototype**（source+target 同 class 混合均值）。Anchor 可能參與自己的 prototype，較像 batch 內 supervised prototype 分類，**未強制 source↔target 跨域對齊**。下游 AUC：control 0.511 vs InfoNCE 0.511（n=10 finetune）。

### 11.2 為何改成 cross-domain prototype InfoNCE

Round 4 新增 `proto_mode=cross_domain`：

```text
target anchor → source same-class prototype（target_to_source）
source anchor → target same-class prototype（source_to_target）
symmetric = 0.5 × (t2s + s2t)
detach_prototypes=True（避免 anchor/prototype 互相追逐）
```

**不提高 `lambda_cls`**，僅改 latent geometry / contrastive 對齊方式。預設 `proto_mode=combined` 維持 Round 3 行為。

### 11.3 為何 selection 要改成 K-means-aware

Round 3：`score_total` 與 `Average_TCGA_AUC_mean` 的 r≈0.03；`score_kmeans` 的 r≈0.52。

新增 selection modes（`tools/optimization_selection.py`）：

| mode | 說明 |
|------|------|
| `score_total` | 舊行為（backward compatible） |
| `round4_kmeans_first` | K-means → wasserstein → fid → mmd |
| `round4_weighted` | `0.6×score_kmeans + 0.4×score_deconfounding` |

CLI：`--selection-mode round4_kmeans_first --exclude-proto-ineffective`

### 11.4 class-wise MMD 支線

獨立 sweep：`vaewc_round4_cmmd_branch`（`lambda_cmmd`，**不與 InfoNCE 主 grid 混跑**）。  
實作：`tools/classwise_alignment.py` → 每 cancer class 分別算 RBF MMD 再平均。

### 11.5 latent size 支線

`sweep`：`latent_size ∈ {32, 64}` × `encoder_dims` 兩組，control only（λ_proto=0）。  
Run ID：`vaewc_round4_latent_ablation`（4 jobs）。

### 11.6 Smoke test 與正式執行

詳見 [`docs/round4_smoke_test.md`](round4_smoke_test.md)。

**新增 config keys（pretrain）：**

```json
{
  "proto_mode": "combined",
  "proto_direction": "symmetric",
  "proto_detach": true,
  "proto_min_samples_per_domain": 1,
  "lambda_cmmd": 0.0,
  "cmmd_start_epoch": 10,
  "cmmd_full_epoch": 40
}
```

**Proto checkpoint guard：** 若 `lambda_proto>0` 且 `best_gan_epoch < proto_start_epoch`，標記 `proto_not_effective_checkpoint=true`（可於 selection 排除）。

**Report 新增欄位：** `selection_mode`, `proto_mode`, `proto_direction`, `proto_detach`, `proto_not_effective_checkpoint`, `lambda_cmmd`, `latent_size`, `encoder_dims`。

---

## 12. Round 4 初步結果與 collapse 診斷（Round 4.1 計畫）

**Run：** `vaewc_round4_cross_domain_infonce`（48 pretrain jobs，symmetric cross-domain InfoNCE）  
**狀態：** Pretrain 完成；selection / finetune 尚未執行（數據已足以診斷 latent geometry 問題）。

### 12.1 Control vs InfoNCE pretrain 品質

| 群組 | n | mean kmeans_ari | mean fid | mean wasserstein |
|------|---|-----------------|----------|------------------|
| Control（`lambda_proto=0`） | 12 | **0.703** | 26.4 | 0.66 |
| InfoNCE（`lambda_proto>0`） | 36 | **0.225** | **15.0** | 0.80 |

**結論：** symmetric cross-domain InfoNCE 在現有 sweep 下顯著降低 FID，但嚴重破壞 K-means tumor structure；Wasserstein 平均反而較差。

### 12.2 為何 FID 改善不足以選模

FID 衡量 source/target latent 分佈距離，**不保證**保留 cancer-type 拓撲。Round 4 出現大量「domain gap 縮小但 class cluster 崩潰」案例；下游 TCGA AUC 與 `score_kmeans` 相關性（Round 3 r≈0.52）高於 `score_total`，故 Round 4.1 **不以 weighted score 作主選模**。

### 12.3 Alignment collapse 典型案例

- **exp_142**（`lambda_proto=0.01`）：`wasserstein=0.37`，`kmeans_ari=0.05` → global alignment 犧牲 tumor topology。
- 診斷規則（`tools/collapse_detection.py`）：`wasserstein≤0.50` 且 `kmeans_ari<0.30` → `alignment_collapse=true`，`collapse_reason=global_alignment_destroyed_tumor_structure`。

### 12.4 `proto_not_effective_checkpoint` 問題

36 個 InfoNCE jobs 中 **17 個** `proto_not_effective_checkpoint=true`（`best_gan_epoch < proto_start_epoch`），代表 best checkpoint 幾乎未經 InfoNCE。

**Round 4.1 修復：**

- `lambda_proto>0` 時同時追蹤 `best_gan_checkpoint_overall` 與 `best_gan_checkpoint_post_proto`（epoch ≥ `max(proto_start+5, proto_full)`）。
- Selection 對 InfoNCE **必須**使用 post-proto checkpoint；無則 `proto_invalid`，排除 Top-K。

### 12.5 Round 4.1 target-to-source InfoNCE 主線

不再預設 symmetric；改為：

```json
{
  "proto_mode": "cross_domain",
  "proto_direction": "target_to_source",
  "proto_detach": true,
  "proto_pair_align": false
}
```

- **target anchor → source same-class prototype**（source prototype detach，target 保留梯度）。
- **暫停** source→target 與 prototype-pair InfoNCE 作為主 loss。

**Sweep：** `config/pretrain_sweeps/vaewc_round4_1_t2s_infonce_collapse_guard.json`  
`lambda_proto ∈ {0, 0.0001, 0.0003, 0.001, 0.002}`，`proto_temperature ∈ {1.5, 2.0, 3.0}`，較早 `proto_start` / `proto_full`。

### 12.6 更新 selection 規則（`round4_1_structure_first`）

**Stage 1 硬篩選：**

- `structure_pass`：`kmeans_ari ≥ 0.65`（或 ≥ 0.90×control mean）
- `deconfounding_relaxed_pass`：`wasserstein ≤ 0.70`（FID 不作 hard fail）
- 排除 `alignment_collapse` 與 `proto_invalid`

**Stage 2 排序：** wasserstein ↑ → kmeans_ari ↓ → fid ↑ → mmd ↑

CLI：`--selection-mode round4_1_structure_first --exclude-proto-ineffective`

Top-10 仍為 8 ranked + 2 `lambda_proto=0` control；若 InfoNCE 全數 structure fail，不強行選入 collapse 模型。

### 12.7 Class-wise MMD 保守支線

`vaewc_round4_1_cmmd_branch`：`lambda_cmmd ∈ {0, 0.0005, 0.001, 0.003, 0.005}`，`lambda_proto=0`，避免與 InfoNCE 交叉造成結構壓縮。

### 12.8 `latent_size` 64 小型支線

`vaewc_round4_1_latent_ablation`：僅 `latent_size ∈ {32, 64}` × control / 單一 t2s InfoNCE 設定，不與大 grid 交叉。報告比較：`kmeans_ari`、`fid`、`wasserstein`、`classwise_domain_gap_mean`、下游 `Average_TCGA_AUC_mean`。

**診斷工具：** `python tools/analyze_round4_pretrain.py --result-dir <pretrain_dir> --out-dir <run_dir>/reports` → `round4_1_pretrain_diagnostics.{csv,md}`
### 12.9 Round 4.1 執行狀態與結論（2026-06-10）

#### 完成度

| 項目 | Run ID | 狀態 |
|------|--------|------|
| **主線 pretrain** | `vaewc_round4_1_t2s_infonce_collapse_guard` | **60/60 success**（manifest 全數完成） |
| Selection（relaxed filter） | 同上 | **完成** — 6/60 通過 → Top-6 + 強制加入 exp_045 / exp_018 / exp_746 |
| 下游 finetune | 同上 | **完成**；保留 top-6（24 jobs），已清理其餘測試輸出 |
| cMMD 支線 | `vaewc_round4_1_cmmd_branch` | **未啟動** |
| latent 支線 | `vaewc_round4_1_latent_ablation` | **未啟動** |

Sweep 規模：`lambda_proto(5) × proto_temperature(3) × proto_start(2) × proto_full(2) = 60` jobs（非 72）。

診斷報告：`result/optimization_runs/vaewc_round4_1_t2s_infonce_collapse_guard/reports/round4_1_pretrain_diagnostics.{md,csv}`

#### Round 4 vs Round 4.1 pretrain 對照

| 群組 | Run | n | mean kmeans_ari | mean fid | mean wasserstein |
|------|-----|---|-----------------|----------|------------------|
| Control | R4 symmetric | 12 | 0.703 | 26.4 | 0.66 |
| Control | **R4.1 t2s** | 12 | 0.694 | 27.8 | 0.69 |
| InfoNCE | R4 symmetric | 36 | **0.225** | **15.0** | 0.80 |
| InfoNCE | **R4.1 t2s** | 48 | **0.524** | 26.1 | **1.00** |

#### 主要結論

1. **t2s InfoNCE 大幅緩解結構崩潰**：相較 Round 4 symmetric，mean `kmeans_ari` 由 0.225 升至 0.524（約 +0.30），不再出現「FID 極低但 tumor cluster 全毀」的極端案例；`alignment_collapse` 在 60 jobs 中為 **0**。
2. **代價是 domain gap 改善變弱**：mean `wasserstein` 由 0.80 升至 1.00；t2s 方向把 trade-off 從「犧牲結構換 FID」轉為「保留較多結構、較難壓平 domain gap」。
3. **Control 仍是最穩 baseline**：control 的 `kmeans_ari`（0.69）與 structure pass rate（10/12）仍優於 InfoNCE 群（structure pass 5/48）。
4. **`proto_not_effective_checkpoint` 仍偏高**：24/60（40%）；InfoNCE 子集 24/48。post-proto dual checkpoint 已實作，但 sweep 內仍有大量 best epoch 落在 proto 啟動前——selection 應 `--exclude-proto-ineffective`。
5. **`round4_1_structure_first` 硬篩下 InfoNCE 尚無 stage-1 全通過者**：`structure_pass ∧ deconfounding_pass` 僅 **7/60**，且 **0/48** 來自 InfoNCE；最佳 InfoNCE 單點為 **exp_045**（`lambda_proto=0.001`, `T=3.0`, `kmeans_ari=0.707`, `wasserstein=0.89`）— structure pass 但 deconfounding 未過。
6. **下游 finetune 已完成**（見 §12.10）：R4.1 最佳 **exp_035**（Avg TCGA=0.5339）未超越 Round 3 exp_018 歷史 0.5695；pretrain 最佳 InfoNCE **exp_045** 下游僅 0.502。

#### 建議下一步

```bash
# 1) 視需要啟動 cmmd / latent 支線（尚未跑）
# 2) 以 exp_035 / exp_018 為候選進入 Round 5 或論文主線對照
```

### 12.10 Round 4.1 下游 Finetune 結果（relaxed filter + 基準對照，2026-06-11）

**狀態：** **已完成** — 保留排名前 **6** 模型（各 4 combo = **24** finetune jobs）；已刪除 exp_006 / exp_008 / exp_015 等非前列測試輸出。

**TCGA 比較檔（優先閱讀）：**

| 用途 | 路徑 |
|------|------|
| 跨 model 排名 | `result/optimization_runs/vaewc_round4_1_t2s_infonce_collapse_guard/aggregate/aggregate_scores.csv` |
| 原始合併（per combo） | `.../aggregate/merged_finetune_tcga_focus.csv` |
| 單 model 明細 | `.../finetune/<Model_ID>/combo_XX/parameter_comparison_tcga_focus.csv` |
| 整合 TCGA 精簡 | `.../finetune/<Model_ID>/combo_XX/eval_metrics_integrated_summary.csv` |

#### 保留模型下游排名（`Average_TCGA_AUC_mean`，gdsc_intersect13）

| 排名 | Model | Avg TCGA | Global TCGA | Integrated Avg | λ_proto | 角色 |
|------|-------|----------|-------------|----------------|---------|------|
| **1** | **exp_035** | **0.5339** | 0.5897 | 0.5088 | 0.0003 | R4.1 InfoNCE（filter 通過） |
| 2 | **exp_018** | 0.5325 | 0.6009 | **0.5545** | 0 | Round 3 Control 基準 |
| 3 | **exp_746** | 0.5265 | **0.6097** | 0.5202 | 0 | 歷史基準 |
| 4 | exp_016 | 0.5255 | 0.5800 | 0.5160 | 0 | R4.1 control |
| 5 | exp_010 | 0.5103 | 0.5752 | 0.5024 | 0 | R4.1 control |
| 6 | **exp_045** | 0.5022 | 0.5581 | 0.5074 | 0.001 | R4.1 最佳 InfoNCE（pretrain） |

#### 與歷史基準對照

| Model | 本輪 Avg TCGA | 歷史 Avg TCGA | 說明 |
|-------|---------------|---------------|------|
| exp_035 | **0.5339** | — | **R4.1 下游最佳** |
| exp_018 | 0.5325 | 0.5695（Round 3，舊 eval） | 新版三份 TCGA eval 重跑 |
| exp_746 | 0.5265 | 0.5462（`pretrain_vaewc_loss`） | 同上 |

#### 主要結論

1. **R4.1 下游最佳為 exp_035**（`lambda_proto=0.0003` t2s InfoNCE）。
2. **exp_045** pretrain 最佳但下游 Avg=0.502，未超越 control 基準。
3. **整合 TCGA** 最高為 **exp_018**（Integrated Avg=0.5545）。
4. Finetune 輸出含 `target_eval_gdsc_intersect13/`、`tcga_only3/`、`dapl/`、`target_eval_integrated/`。
