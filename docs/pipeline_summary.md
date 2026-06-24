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
| Pretrain | `max_parallel` 個子進程，各跑一組超參 | `batch_size=128`；**`max_parallel=33`**（RTX 6000 Ada 49GB，~80% VRAM；見 `config/gpu_parallel_profile.json`） |
| Finetune | `max_parallel` 個子進程，各跑一組 (model×combo) | `batch=12288`, `mini=3072`, **`max_parallel=42`** |

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

---

## 13. Round 5 Control-centered + Class-gap optimization（2026-06-11）

**狀態：** Pretrain **102/102**、Selection **16 模型**、Finetune **64/64 success**、Aggregate **已完成**（2026-06-11）。

完整操作手冊：`docs/round5_optimization_manual.md`  
診斷報告：`result/optimization_runs/round5_combined_reports/round5_pretrain_diagnostics.{md,csv}`

### 13.0 Pretrain 完成度（2026-06-11）

| 分支 | Run ID | Manifest | 有效 `gan_metrics` | 狀態 |
|------|--------|----------|-------------------|------|
| **A Control** | `vaewc_round5_control_centered` | **48/48 success** | 68（含重啟殘留目錄） | 完成 |
| **B Class-gap** | `vaewc_round5_class_gap_branch` | **30/30 success** | 31（含 1 殘留） | 完成 |
| **C t2s appendix** | `vaewc_round5_t2s_infonce_appendix` | **24/24 success** | 24 | 完成 |

> 合計 **102** sweep jobs 全部 manifest `success`。`gan_metrics` 目錄數略多於 manifest（重啟殘留），selection 以 manifest / 最新 metrics 為準。

### 13.1 為何不再以 InfoNCE 作主線

Round 4 symmetric InfoNCE 破壞 K-means；R4.1 t2s InfoNCE 緩解 collapse 但下游最佳 **exp_035**（Avg TCGA=0.5339）未明顯超越 control。**Round 5 主線改為 control-centered latent 優化 + class-wise prototype gap**；t2s InfoNCE 僅保留 appendix 支線。

### 13.2 三條分支（分開跑）

| 分支 | Run ID | 目的 |
|------|--------|------|
| **A Control-centered** | `vaewc_round5_control_centered` | `lambda_proto=0`，測 latent 32/64/128 + GAN 策略（48 jobs） |
| **B Class-gap** | `vaewc_round5_class_gap_branch` | 同 class prototype 靠近（cosine + L2） |
| **C t2s appendix** | `vaewc_round5_t2s_infonce_appendix` | 極低 λ_proto t2s InfoNCE 確認 |

### 13.3 新增程式能力

- `tools/classwise_alignment.py` → `compute_classwise_prototype_gap()`
- `pretrain_VAEwC.py` → `lambda_class_gap` schedule + GAN loss
- `tools/optimization_config_generator.py` → `paired_params`（latent_size ↔ encoder_dims）
- `tools/optimization_selection.py` → `round5_structure_first` + `--force-baseline-models`
- `tools/analyze_round5_pretrain.py` → 跨分支診斷報告

### 13.4 Selection 原則

- Stage 1：`structure_pass` + 無 `alignment_collapse` + 無 `proto_invalid`
- Stage 2：wasserstein ↑ → kmeans_ari ↓ → fid ↑ → mmd ↑
- 強制納入：**exp_018**、**exp_746**、當輪最佳 control
- 下游主指標：**Average_TCGA_AUC_mean**（gdsc_intersect13）

### 13.5 執行指令（摘要）

```bash
# Branch A
python tools/optimization_runner.py generate \
  --sweep-spec config/pretrain_sweeps/vaewc_round5_control_centered.json \
  --run-dir result/optimization_runs/vaewc_round5_control_centered

python tools/optimization_runner.py pretrain \
  --manifest result/optimization_runs/vaewc_round5_control_centered/manifests/pretrain_sweep_manifest.csv \
  --run-dir result/optimization_runs/vaewc_round5_control_centered \
  --device cuda --max-parallel 30

# 診斷（三分支完成後）
python tools/analyze_round5_pretrain.py \
  --run-dirs \
    result/optimization_runs/vaewc_round5_control_centered \
    result/optimization_runs/vaewc_round5_class_gap_branch \
    result/optimization_runs/vaewc_round5_t2s_infonce_appendix

# Selection（合併多 run）
python tools/optimization_runner.py select \
  --run-dir result/optimization_runs/round5_combined \
  --result-dir result/optimization_runs/vaewc_round5_control_centered/pretrain \
  --result-dirs \
    result/optimization_runs/vaewc_round5_class_gap_branch/pretrain,\
    result/optimization_runs/vaewc_round5_t2s_infonce_appendix/pretrain \
  --selection-mode round5_structure_first \
  --exclude-proto-ineffective \
  --force-baseline-models exp_018,exp_746 \
  --top-k 15
```

### 13.6 成功標準

1. `Average_TCGA_AUC_mean` 超越 R4.1 **exp_035**（0.5339）
2. `kmeans_ari` ≥ 0.65（或 ≥ control mean 的 90%）
3. 不選入 wasserstein 好但 structure collapse 的模型
4. class-gap 分支至少一個 non-collapse 且下游接近 control

### 13.7 Pretrain 結果摘要（三分支完成）

#### 跨分支對照（pretrain 指標）

| 群組 | n | mean kmeans_ari | structure pass rate | mean wasserstein | collapse rate |
|------|---|-----------------|---------------------|------------------|---------------|
| Control-centered（A） | 68 | 0.372 | 41% | 0.429 | **50%** |
| Class-gap（B） | 31 | 0.420 | 39% | 0.721 | 13% |
| t2s appendix（C） | 24 | **0.633** | **54%** | 0.902 | **0%** |

#### `latent_size` 分組（三分支合併）

| latent_size | n | mean kmeans_ari | structure pass rate |
|-------------|---|-----------------|---------------------|
| 32 | 50 | 0.416 | 40% |
| **64** | **38** | **0.451** | **50%** |
| 128 | 35 | 0.444 | 40% |

#### Class-gap 子分析（B 分支，`lambda_class_gap > 0`）

| class_gap_metric | n | mean kmeans_ari | structure pass rate |
|------------------|---|-----------------|---------------------|
| **cosine** | 12 | **0.452** | **33%** |
| l2 | 12 | 0.278 | 25% |

#### 單點最佳

| 分支 | Model | kmeans_ari | wasserstein | latent | 備註 |
|------|-------|------------|-------------|--------|------|
| Control（A） | exp_059 | 0.686 | **0.330** | 32 | structure pass 內 deconfounding 最佳 |
| Control（A） | exp_073 | **0.819** | 0.678 | 64 | 全分支 kmeans 最高 |
| Class-gap（B） | exp_035 | **0.807** | 0.582 | 32 | B 分支 kmeans 最高 |
| Class-gap（B） | exp_053 | 0.704 | 0.535 | — | structure pass 內較佳 |
| t2s appendix（C） | exp_016 | 0.702 | 0.498 | 128 | C 分支 structure+deconfounding 平衡 |

#### 主要結論（pretrain 階段）

1. **三分支 manifest 全數完成**，但 **Control（A）collapse 率仍高（~50%）**；擴 latent / 調 GAN 未根除 structure–deconfounding trade-off。
2. **Class-gap（B）已可評估**：cosine 優於 L2（mean ari 0.45 vs 0.28）；整體 mean ari 略優於 A，但 structure pass 仍僅 ~39%。
3. **t2s appendix（C）pretrain 幾何最穩**：mean kmeans_ari 0.63、collapse 0%，但 wasserstein ~0.90 偏高。
4. **`latent_size=64` 仍是最穩容量**（structure pass ~50%）。
5. **下游已驗證**（見 §13.8–§13.9）：Round 5 最佳 **exp_001**（Avg TCGA=**0.5403**）已超越 R4.1 **exp_035**（0.5339）；但入選模型 **全部 `lambda_proto=0`**，active InfoNCE 仍未進 Top-16。

### 13.8 Selection + Finetune 執行紀錄（已完成）

#### Selection（`round5_combined`）

| 項目 | 設定 / 結果 |
|------|-------------|
| 模式 | `round5_structure_first`，`--exclude-proto-ineffective` |
| Top-K | 15 ranked + 強制 **exp_018**、**exp_746**（`exp_018` 已在 Top-15）→ **16 模型** |
| `--min-passing` | 5（stage-1 通過者足夠時才補滿 Top-K） |
| 輸出 | `result/optimization_runs/round5_combined/selection/pretrain_top10.csv` |

**程式修復（finetune 前）：** 多分支合併時 `pretrain_top10.csv` 須帶 `pretrain_result_dir` / `result_folder`，否則 finetune 會誤指向 `pretrain/exp_XXX` 根目錄。已修 `tools/optimization_selection.py`、`visualize_vaewc_results.py`。

#### Finetune + Aggregate

| 項目 | 設定 |
|------|------|
| 模型數 × combo | **16 × 4 = 64** jobs（`config/params_finetune_mini.json`：bce/focal × dropout 0.05/0.1） |
| 訓練 | `epochs=1000`，`batch_size=4096`，`mini_batch_size=1024` |
| 平行度 | `max_parallel=26`（當次實跑；腳本預設 42 供後續重跑） |
| 狀態 | **64/64 success**（`manifests/finetune_dispatch_manifest.csv`） |
| Log | `result/optimization_runs/round5_combined/logs/round5_finetune_aggregate.log` |

**TCGA 比較檔：**

| 用途 | 路徑 |
|------|------|
| 跨 model 排名 | `result/optimization_runs/round5_combined/aggregate/aggregate_scores.csv` |
| 原始合併（per combo） | `.../aggregate/merged_finetune_tcga_focus.csv` |
| 單 model 明細 | `.../finetune/<Model_ID>/combo_XX/parameter_comparison_tcga_focus.csv` |

後續重跑 finetune：`bash tools/run_round5_finetune_aggregate.sh`（預設 `FINETUNE_MAX_PARALLEL=42`）。

### 13.9 Round 5 下游 Finetune 結果與參數–結果對照（2026-06-11）

**主指標：** `Average_TCGA_AUC_mean`（gdsc_intersect13，4 combo 平均）。  
**R4.1 基準：** `result/optimization_runs/vaewc_round4_1_t2s_infonce_collapse_guard/aggregate/aggregate_scores.csv`。

#### 13.9.1 成功標準達成（對照 §13.6）

| # | 標準 | 結果 |
|---|------|------|
| 1 | Avg TCGA 超越 R4.1 exp_035（0.5339） | **達成** — **exp_001 = 0.5403**（+0.0064） |
| 2 | `kmeans_ari` ≥ 0.65 | **部分** — exp_001（0.669）、exp_005（0.754）、exp_035（0.807）等達標；非全數 |
| 3 | 不選 collapse 模型 | **達成** — structure-first 已過濾；入選 16 模型無 `alignment_collapse` |
| 4 | class-gap 下游接近 control | **部分** — 最佳 class-gap **exp_035**（0.5195）< 最佳 control **exp_005**（0.5301），差距 ~0.011 |

#### 13.9.2 下游排名（16 模型，依 `Average_TCGA_AUC_mean`）

| 排名 | Model | Avg TCGA | Global TCGA | Integrated Avg | Test AUC | Sel# | 分支 | 關鍵 pretrain 參數 |
|------|-------|----------|-------------|----------------|----------|------|------|-------------------|
| **1** | **exp_001** | **0.5403** | 0.6031 | 0.5093 | 0.7474 | 13 | C t2s appendix | `latent=32`, `[256,128]`, **`λ_proto=0`**, `proto_dir=t2s`（未啟用）, `kmeans_ari=0.669`, `wass=0.629` |
| 2 | exp_005 | 0.5301 | 0.5864 | **0.5252** | 0.7816 | 14 | A control | `latent=32`, `[256,128]`, pure control, `kmeans_ari=0.754`, `wass=0.643` |
| 3 | exp_746 | 0.5214 | **0.6053** | 0.5145 | 0.7817 | 16 | 外部 baseline | 歷史 `pretrain_vaewc/exp_746` |
| 4 | exp_035 | 0.5195 | 0.6025 | 0.5098 | 0.7381 | 7 | B class-gap | `latent=32`, **`λ_class_gap=0.001`**, `metric=cosine`, `kmeans_ari=0.807`, `wass=0.582` |
| 5 | exp_057 | 0.5193 | 0.5936 | 0.4930 | 0.7970 | 11 | A control | `latent=64`, `[512,256,128]`, pure control |
| 6 | exp_015 | 0.5119 | 0.5703 | 0.5187 | 0.8128 | 8 | A control | `latent=64` |
| 7 | exp_059 | 0.5051 | 0.5800 | 0.4635 | 0.7185 | **1** | A control | `latent=32`, **selection #1**（`wass=0.330` 最佳 deconfounding） |
| 8 | exp_053 | 0.5004 | 0.5952 | **0.5507** | 0.7603 | 5 | B class-gap | `λ_class_gap=0`（僅 metric=cosine 槽位）, `kmeans_ari=0.704` |
| 9 | exp_063 | 0.4923 | 0.5417 | 0.4899 | 0.7863 | 4 | A control | `latent=64` |
| 10 | exp_018 | 0.4909 | 0.5725 | 0.4561 | 0.7984 | 3 | A control（強制） | R4.1 曾 0.5325（新版 eval）；本輪 pretrain 重跑後下游下降 |
| 11 | exp_016 | 0.4797 | 0.5857 | 0.5024 | 0.7927 | 2 | C t2s appendix | `latent=128`, `[1024,512,256]`, `λ_proto=0`, `T=3.0` |
| 12 | exp_033 | 0.4699 | 0.5617 | 0.4637 | 0.8069 | 12 | B class-gap | `λ_class_gap=0`, `metric=l2` |
| 13 | exp_038 | 0.4639 | 0.5505 | 0.4778 | 0.7958 | 15 | B class-gap | `latent=128`, `λ_class_gap=0.001`, cosine |
| 14 | exp_007 | 0.4627 | 0.6008 | **0.5409** | 0.7603 | 6 | A control | `latent=64` |
| 15 | exp_071 | 0.4527 | 0.5301 | 0.4704 | 0.7883 | 10 | A control | `latent=32`, 高 `kmeans_ari=0.780` 但下游偏弱 |
| 16 | exp_051 | 0.4416 | 0.5755 | 0.4778 | **0.8270** | 9 | A control | Test AUC 最高但 TCGA 最低之一 |

> **Global TCGA** 排名與 Avg 不同：exp_746（0.605）> exp_035（0.603）> exp_001（0.603）。  
> **Integrated Avg** 最高為 exp_053（0.551）、exp_007（0.541），與 gdsc_intersect13 主指標不一致時以 Avg TCGA 為準。

#### 13.9.3 與 R4.1 / 歷史基準對照

| Model | Round | Avg TCGA | Global TCGA | 備註 |
|-------|-------|----------|-------------|------|
| **exp_001** | **R5** | **0.5403** | 0.6031 | **全專案本輪最佳（新版 eval）** |
| exp_035 | R4.1 | 0.5339 | 0.5897 | R4.1 下游最佳；R5 同 ID 為 class-gap 分支（0.5195） |
| exp_018 | R4.1 | 0.5325 | 0.6009 | R5 control 重跑後 0.4909 |
| exp_746 | R4.1 | 0.5265 | 0.6097 | R5 0.5214，略降 |

#### 13.9.4 參數 ↔ 結果：系統性觀察

**（1）Pretrain selection 排名 ≠ 下游排名**

| 現象 | 代表案例 | 解讀 |
|------|----------|------|
| Selection 依 wasserstein 排 #1，下游僅中游 | **exp_059**（`wass=0.330` → Avg 0.505） | 過度 deconfounding 可能損失對 TCGA 有用的 domain/tumor 訊號 |
| 高 `kmeans_ari` 不保證高 TCGA | **exp_035**（ari 0.807 → Avg 0.5195） | 結構保留佳 ≠ 藥物反應可轉移 |
| 下游最佳非 selection 前列 | **exp_001**（sel #13 → Avg **0.5403**） | structure-first 排序偏 deconfounding，與下游最優解錯位 |

**（2）分支（A/B/C）與 loss 類型**

| 分支 | 入選 n | Avg TCGA 最佳 | mean Avg TCGA | 重點 |
|------|--------|---------------|---------------|------|
| A Control-centered | 9 | exp_005 **0.5301** | ~0.489 | 最穩；純 control 仍具競爭力 |
| B Class-gap | 4 | exp_035 0.5195 | ~0.489 | pretrain ari 高，下游未超越 A 最佳 control |
| C t2s appendix | 2 | exp_001 **0.5403** | ~0.510 | 下游冠軍來自 C，但 **`λ_proto=0`（無 active InfoNCE）** |
| 外部 | 1 | exp_746 0.5214 | — | Global 仍強 |

**重要：** 入選 16 模型 **`lambda_proto` 均為 0**；C 分支雖掛 t2s InfoNCE sweep，通過 structure-first 且進 downstream 者皆為 control 槽位。Active InfoNCE 仍未證明下游優勢。

**（3）`latent_size`**

| latent_size | 入選數 | 下游最佳 | mean Avg TCGA（入選子集） |
|-------------|--------|----------|---------------------------|
| **32** | 8 | **exp_001 0.5403** | ~0.501 |
| 64 | 5 | exp_057 0.5193 | ~0.497 |
| 128 | 2 | exp_016 0.4797 | ~0.472 |

Top-4 皆為 **latent=32**；128 在下游明顯偏弱（exp_016、exp_038）。

**（4）Class-gap 參數**

| 設定 | Model | `kmeans_ari` | Avg TCGA | 解讀 |
|------|-------|--------------|----------|------|
| `λ_class_gap=0.001`, cosine | exp_035 | 0.807 | 0.5195 | B 分支最佳；略低於 pure control exp_005 |
| `λ_class_gap=0.001`, cosine, latent=128 | exp_038 | 0.673 | 0.4639 | 大 latent + class-gap 組合下游差 |
| `λ_class_gap=0`, metric=l2 | exp_033 | 0.712 | 0.4699 | L2 槽位無 active loss，下游偏弱 |
| `λ_class_gap=0`, metric=cosine | exp_053 | 0.704 | 0.5004 | Integrated Avg 高（0.551）但 gdsc_intersect13 主指標一般 |

**（5）Pretrain 幾何指標 vs 下游（入選子集趨勢）**

- **`wasserstein` 與 Avg TCGA：弱負相關** — 越低 wasserstein（deconfounding 越好）未必越高 TCGA（exp_059 vs exp_001）。
- **`kmeans_ari` 與 Avg TCGA：無單調正相關** — exp_071（ari 0.78）下游 0.453；exp_001（ari 0.67）下游 0.540。
- **Sweet spot 假說：** 中等 `kmeans_ari`（0.65–0.75）+ 中等 `wasserstein`（0.58–0.65）+ `latent=32` + pure control → 較利下游（exp_001、exp_005）。

**（6）Finetune 側（固定 grid，非 sweep 變因）**

所有模型共用：`params_finetune_mini.json`（4 combo）、`ftlr=0.001`、classifier `[256,128]`、GIN `dapl`。  
下游差異主要來自 **pretrain checkpoint / latent**，非 finetune 超參 grid（每模型 4 combo 平均）。

#### 13.9.5 主要結論

1. **Round 5 達成首要目標：** **exp_001**（Avg TCGA **0.5403**）超越 R4.1 **exp_035**（0.5339），為目前新版 TCGA eval 最佳。
2. **獲勝配方實質為 pure control：** exp_001 在 C 分支 sweep 中 `lambda_proto=0`，僅保留 t2s 相關 config 槽位；**非 active InfoNCE 勝出**。
3. **Class-gap 未證明下游優於 control：** exp_035 pretrain 結構最佳（ari 0.81），下游仍低於 exp_005 / exp_001。
4. **Selection 指標與下游錯位：** wasserstein-first 選出的 exp_059 下游一般；建議下一輪 selection 加權或加入下游 proxy（若可負擔）。
5. **`latent_size=32` 仍是下游首選**；64 可接受，128 在本輪入選者中偏弱。
6. **exp_018 強制 baseline 在本輪 pretrain 重跑後下游降至 0.491** — 論文對照宜註明「同 ID、不同 pretrain run」。

#### 13.9.6 建議下一步

| 方向 | 建議 |
|------|------|
| **論文 / 主線 checkpoint** | 採用 **exp_001**（R5 C 分支、`result/.../vaewc_round5_t2s_infonce_appendix/pretrain/exp_001`） |
| **對照組** | exp_005（pure A control）、exp_035（class-gap）、exp_746（歷史） |
| **Selection 調整** | 考慮降低 wasserstein 權重，或 Top-K 內強制保留「中等 wass + 高 ari」候選 |
| **InfoNCE** | 本輪無 `λ_proto>0` 入選；若再試，需放寬 deconfounding 或改 stage-2 排序 |
| **Finetune 重跑** | `bash tools/run_round5_finetune_aggregate.sh`（`max_parallel=42`） |

**決策要點（更新後）：**

| 問題 | Round 5 下游判讀 |
|------|------------------|
| Class-gap 是否值得繼續？ | pretrain 有亮點，下游未贏 control；可縮小 sweep 或僅作 ablation |
| InfoNCE 是否保留？ | 入選者全 `λ_proto=0`；appendix 可保留但非主線 |
| 主線收斂方向？ | **`latent_size=32` pure control**（exp_001 / exp_005 族）+ 以 **exp_001** 為 production checkpoint |

---

## 14. Round 6 Tumor-topology-aware latent representation（2026-06-12）

**狀態：** pretrain / selection / finetune / aggregate **全部完成**（finetune **64/64** success；2026-06-13 重跑 23 OOM jobs 後定案）。

完整操作手冊：`docs/round6_optimization_manual.md`  
一鍵腳本：`tools/run_round6_full_pipeline.sh`（pretrain→aggregate）；selection 修復後續跑：`tools/run_round6_post_pretrain.sh`

### 14.1 Motivation from Round 5

Round 5 下游最佳 **exp_001**（Avg TCGA=0.5403）實質為 **`lambda_proto=0` pure control**；wasserstein-first selection 與下游最佳錯位。Round 6 主題改為 **tumor latent geometry**（topology / subspace / within-domain SupCon / VICReg），不以提高 `lambda_cls` 或 cross-domain InfoNCE 為主線。

### 14.2 分支總覽

| 分支 | Run ID | Jobs | 目的 |
|------|--------|------|------|
| **6A** | `vaewc_round6A_tumor_topology` | 16 | class prototype **topology** 保留 |
| **6B** | `vaewc_round6B_topology_classgap_combo` | 18 | topology + class-gap 組合 |
| **6C** | `vaewc_round6C_tumor_transfer_subspace` | 24 | tumor / transfer subspace split |
| **6D** | `vaewc_round6D_within_domain_tumor_supcon` | 32 | domain 內 SupCon |
| **6E** | `vaewc_round6E_tumor_vicreg_stabilizer` | 12 | VICReg anti-collapse |
| **6S** | selection | — | `round6_sweetspot`（非獨立 pretrain sweep） |

Baseline config：`config/params_proto_base_exp001_vaewc.json`（對齊 R5 **exp_001**）。

### 14.3 新增程式

| 模組 | 用途 |
|------|------|
| `tools/tumor_geometry.py` | `compute_tumor_topology_loss` |
| `tools/tumor_subspace.py` | latent view / orthogonality |
| `tools/tumor_supcon.py` | within-domain SupCon |
| `tools/tumor_vicreg.py` | variance + covariance regularization |
| `tools/round6_selection.py` | sweet-spot score + `rank_round6_sweetspot` |
| `tools/analyze_round6_pretrain.py` | 跨分支 pretrain 診斷 |

`pretrain_VAEwC.py` 已接入上述 loss（僅 GAN generator step）；`optimization_selection.py` 新增 **`round6_sweetspot`**。

### 14.4 執行（摘要）

```bash
bash tools/run_round6_pretrain.sh

python tools/optimization_runner.py select \
  --run-dir result/optimization_runs/round6_combined \
  --result-dirs \
    result/optimization_runs/vaewc_round6A_tumor_topology/pretrain,\
    result/optimization_runs/vaewc_round6B_topology_classgap_combo/pretrain,\
    result/optimization_runs/vaewc_round6C_tumor_transfer_subspace/pretrain,\
    result/optimization_runs/vaewc_round6D_within_domain_tumor_supcon/pretrain,\
    result/optimization_runs/vaewc_round6E_tumor_vicreg_stabilizer/pretrain \
  --selection-mode round6_sweetspot \
  --force-baseline-models exp_001,exp_005,exp_746 \
  --top-k 30
```

Finetune 仍用 `config/params_finetune_mini.json`（不變 grid）。首輪 `max_parallel=42` 有 23 次 OOM；重跑見 `tools/run_round6_finetune_retry.sh`（`batch=12288`, `parallel=42`）。

### 14.5 成功標準（設計時）

1. `Average_TCGA_AUC_mean` **> 0.5403**（R5 exp_001）
2. `kmeans_ari` ≥ 0.65；無 alignment collapse
3. 至少一個 **active tumor loss** 模型進 downstream Top-5
4. `round6_sweetspot` 選模較 wasserstein-first 更接近下游最佳

### 14.6 Pretrain 結果摘要（102/102 完成）

| 分支 | n | mean kmeans_ari | structure pass | collapse rate | mean sweetspot | best sweetspot |
|------|---|-----------------|----------------|---------------|----------------|----------------|
| **6A** topology | 16 | 0.297 | 25% | 13% | 0.579 | exp_001 |
| **6B** topology+class-gap | 18 | 0.353 | 28% | 6% | 0.604 | exp_017 |
| **6C** subspace | 24 | **0.112** | **0%** | **38%** | 0.520 | exp_015 |
| **6D** within-domain SupCon | 32 | 0.218 | 13% | 19% | 0.547 | exp_004 |
| **6E** VICReg | 12 | 0.379 | 33% | 8% | **0.640** | exp_012 |

**Pretrain 重點：**

1. **6C subspace 整體失敗**：structure pass 0%、collapse 38%，不建議再擴 sweep。
2. **6E VICReg** pretrain sweetspot 最高（mean 0.640），但多為 **`lambda=0` control 槽位** 或極小 λ。
3. **6B class-gap 組合** pretrain 最均衡（collapse 6%、sweetspot pass ~28%）。
4. 全體僅 **17/102** structure pass、**19/102** sweetspot pass → selection 僅能補滿 **16** 模型（目標 top-k=30）。

診斷輸出：`result/optimization_runs/round6_combined/reports/round6_pretrain_diagnostics.md`

### 14.7 Selection + Finetune 執行紀錄

#### Selection（`round6_combined`）

| 項目 | 設定 / 結果 |
|------|-------------|
| 模式 | `round6_sweetspot`，`--exclude-proto-ineffective` |
| Top-K | 30（實際入選 **16**；sweetspot/structure gate 通過者不足） |
| 強制 baseline | **exp_001**（R5 最佳）、**exp_005**（R5 control）、**exp_746** |
| 輸出 | `result/optimization_runs/round6_combined/selection/pretrain_top10.csv` |

**程式修復（selection 階段）：** `round6_selection.py` 修正缺失 lambda 欄位時 `.fillna` 失敗、以及 sweetspot 欄位重複導致排序 crash；`optimization_runner.py` CLI 補上 `round6_sweetspot`；`optimization_selection.py` 讓 round6 走 `select_top_k_with_baselines`。

#### Finetune + Aggregate

| 項目 | 設定 |
|------|------|
| 模型 × combo | **16 × 4 = 64** jobs |
| 首輪 | `max_parallel=42`, `batch=4096` → **41/64** success（23× OOM `-9`） |
| 重跑（2026-06-13） | `bash tools/run_round6_finetune_retry.sh`（`batch=12288`, `mini=3072`, `parallel=42`） |
| 最終狀態 | **64/64 success**（2026-06-13T02:19:55Z） |
| Log | `.../logs/round6_finetune_retry.log` |

**TCGA 比較檔：**

| 用途 | 路徑 |
|------|------|
| 跨 model 排名 | `result/optimization_runs/round6_combined/aggregate/aggregate_scores.csv` |
| 原始合併（per combo） | `.../aggregate/merged_finetune_tcga_focus.csv` |
| Finetune manifest | `.../manifests/finetune_dispatch_manifest.csv` |

### 14.8 Round 6 下游 Finetune 結果（定案，2026-06-13）

**主指標：** `Average_TCGA_AUC_mean`（gdsc_intersect13，4 combo 平均）。  
**R5 基準：** **exp_001 = 0.5403**。

#### 14.8.1 下游排名（16 模型，依 Avg TCGA，64/64 finetune 完整）

| 排名 | Model | Avg TCGA | Global TCGA | Integrated Avg | Test AUC | Sel# | 分支 | 關鍵 pretrain |
|------|-------|----------|-------------|----------------|----------|------|------|---------------|
| **1** | **exp_010** | **0.5569** | 0.6087 | 0.5130 | 0.8088 | 6 | 6E VICReg | `latent=64`, **λ=0**（6E control 槽）, ari=0.744, wass=0.631 |
| 2 | exp_746 | 0.5225 | 0.5981 | 0.5091 | 0.8104 | 16 | 外部 baseline | 歷史 `pretrain_vaewc/exp_746` |
| 3 | exp_009 | 0.5201 | 0.6030 | 0.4986 | 0.8105 | 14 | 6A topology | `latent=64`, λ=0, ari=0.763, wass=0.806 |
| 4 | exp_012 | 0.5182 | 0.6014 | **0.5620** | 0.7938 | 5 | 6E VICReg | `latent=64`, **`λ_var=λ_cov=0.0003`**, ari=0.751 |
| 5 | exp_015 | 0.5129 | 0.5806 | 0.5247 | 0.8288 | 9 | 6D SupCon | `latent=64`, λ=0, ari=0.666 |
| 6 | exp_005 | 0.5079 | 0.5837 | 0.4942 | 0.8151 | 7 | 6B topo+gap | `latent=64`, **`λ_class_gap=0.0003`**, ari=0.737 |
| 7 | exp_016 | 0.5048 | 0.5739 | 0.4987 | 0.7574 | 8 | 6D SupCon | `latent=32`, λ=0, ari=0.692 |
| 8 | exp_004 | 0.5039 | 0.5894 | 0.4679 | 0.7696 | **1** | 6D SupCon | `latent=32`, λ=0, ari=0.701, wass=0.545 |
| 9 | exp_017 | 0.4879 | 0.6042 | 0.4666 | 0.8035 | 2 | 6B topo+gap | **`λ_class_gap=0.0003`**, ari=0.777 |
| 10 | exp_002 | 0.4856 | 0.5795 | 0.4937 | 0.8048 | 11 | 6A topology | **`λ_topology=0.0003`**, ari=0.763 |
| 11 | exp_011 | 0.4823 | 0.5691 | 0.4620 | 0.8018 | 13 | 6E VICReg | `λ_var=λ_cov=0.0001`, ari=0.791 |
| 12 | exp_003 | 0.4809 | 0.5901 | 0.4911 | 0.8032 | 15 | 6A topology | `latent=64`, λ=0 |
| 13 | exp_007 | 0.4784 | 0.5609 | 0.4813 | 0.7756 | 3 | 6E VICReg | `λ_var=λ_cov=0.0003`, ari=0.793 |
| 14 | exp_006 | 0.4762 | 0.5650 | 0.5026 | 0.8049 | 12 | 6B | λ=0, ari=0.767 |
| 15 | exp_001 | 0.4760 | 0.5799 | 0.5010 | 0.7700 | 4 | 6A（R5 ID 重跑） | R5 最佳 ID 在 6A 重 pretrain 後下游下降 |
| 16 | exp_013 | 0.4632 | 0.5448 | 0.4920 | 0.8157 | 10 | 6B | `λ_class_gap=0.0001`, ari=0.726 |

**分支 Avg TCGA（入選子集）：** 6E mean **0.509**（max **0.557**）> 6D 0.507 > 6A 0.491 > 6B 0.484。

#### 14.8.2 與 R5 / 歷史基準對照

| Model | Round | Avg TCGA | Global TCGA | 備註 |
|-------|-------|----------|-------------|------|
| **exp_010** | **R6** | **0.5569** | 0.6087 | **全專案新版 eval 最佳**；6E、λ=0、latent=64 |
| exp_001 | R5 | 0.5403 | 0.6031 | 前主線 checkpoint |
| exp_012 | R6 | 0.5182 | 0.6014 | Top-5 內唯一 **active VICReg**（Integrated 最高 0.562） |
| exp_746 | R5/R6 | 0.5225 | 0.5981 | 外部 baseline |
| exp_004 | R6 | 0.5039 | 0.5894 | sweetspot **#1**，下游 **#8** |

#### 14.8.3 參數 ↔ 結果觀察

1. **Round 6 下游超越 R5：** **exp_010**（+0.0166 vs exp_001）但配方仍為 **λ=0 pure control**（6E sweep 槽位）。
2. **Active tumor loss 有進 Top-5：** **exp_012**（#4，`λ_var=λ_cov=0.0003`）Integrated Avg **0.562** 為全場最高，但 gdsc_intersect13 主指標仍低於 exp_010。
3. **Sweetspot #1（exp_004）≠ 下游最佳**：pretrain 幾何優（sel #1）→ 下游 #8（0.5039）；**#6 exp_010** 成下游冠軍。
4. **6C 無入選**；**6E VICReg** 分支包辦下游冠軍（exp_010）與 active-loss 代表（exp_012）。
5. **`latent_size=64`** 主導 Top-6（exp_010/009/012/015/005/003）；R5 偏好的 latent=32 在本輪入選者中下游居中。

### 14.9 成功標準達成（定案）

| # | 標準 | 結果 |
|---|------|------|
| 1 | Avg TCGA > 0.5403 | **達成** — **exp_010 = 0.5569**（64/64 finetune 完整） |
| 2 | kmeans_ari ≥ 0.65、無 collapse | **達成**（入選 16 皆通過 structure gate） |
| 3 | active tumor loss 進 Top-5 | **部分達成** — **exp_012**（#4，VICReg λ=0.0003）；Top-3 仍為 λ=0 |
| 4 | sweetspot 更接近下游 | **部分** — sweetspot #1 下游 #8；#6 exp_010 成下游 #1 |

### 14.10 主要結論與建議

1. **Round 6 pipeline 完整結束**：pretrain 102/102 → selection 16 → finetune **64/64** → aggregate。
2. **首要目標達成：** 下游 **exp_010**（Avg TCGA **0.5569**）超越 R5 **exp_001**（0.5403）。
3. **獲勝配方仍為 control：** exp_010 `lambda=0`；tumor loss 未證明優於 pure control，但 **VICReg active（exp_012）** 在 Integrated 指標亮眼。
4. **6C subspace 建議停用**；6E 可保留 ablation；6B class-gap 下游仍弱於 6E/6A。
5. **Checkpoint 建議（更新）：**
   - **論文 / production 主線：** **exp_010**（`vaewc_round6E_tumor_vicreg_stabilizer/pretrain/exp_010`）
   - **Active-loss 對照：** **exp_012**（VICReg λ=0.0003）
   - **歷史對照：** R5 **exp_001**、**exp_746**

**決策要點：**

| 問題 | Round 6 定案 |
|------|----------------|
| 是否超越 R5？ | **是** — exp_010 +0.0166 Avg TCGA |
| Tumor loss 主線？ | **否** — 最佳仍 λ=0；VICReg 可作 ablation |
| Subspace（6C）？ | **否** |
| 主線 checkpoint？ | **exp_010**（R6 6E）；Integrated 最佳可看 **exp_012** |

---

## 15. Round 7 exp_010 neighborhood refinement and VICReg ablation（2026-06-10）

> 操作手冊：`docs/round7_optimization_manual.md`

### 15.1 Motivation from Round 6

Round 6 定案：**exp_010** Avg TCGA **0.5569**（λ=0 control-like）；**exp_012** Integrated Avg 突出但 Avg TCGA 未超 exp_010。topology / class-gap / SupCon / subspace 未穩定超越 control；selection 與 downstream 仍有錯位。

Round 7 主軸：**7A** exp_010 鄰域 control refinement；**7B** VICReg-only ablation；**7C** downstream-aware diverse selection；**7D** 小範圍 finetune sensitivity。

### 15.2 Round 7A control refinement

- Sweep：`vaewc_round7A_exp010_control_refinement.json`（**108 jobs**）
- 固定 latent=64、encoder `[512,256,128]`；active tumor loss 全 0
- 掃描：`lambda_cls`、cls schedule、GAN patience / gen interval

### 15.3 Round 7B VICReg focused ablation

- Sweep：`vaewc_round7B_vicreg_focused_ablation.json`（**56 jobs**）
- 僅 VICReg（paired / asymmetric var·cov）；其餘 tumor loss = 0

### 15.4 Round 7C downstream-aware selection

- Mode：`round7_diverse_downstream_probe`（`tools/round7_selection.py`）
- Diverse groups G1–G7 + forced baselines **exp_010, exp_012, exp_001, exp_005, exp_746**
- Combined run：`result/optimization_runs/round7_combined`

### 15.5 Round 7D finetune sensitivity

- Config：`config/finetune_sweeps/round7_finetune_sensitivity.json`（8 combos/checkpoint）
- 第二輪：少數 checkpoint 上測 classifier loss / hidden_dims / dropout
- 平行度：見 **`config/gpu_parallel_profile.json`**（2026-06-13：parallel=20→49% VRAM；**33** 目標 ~80% VRAM；36 會 OOM）

### 15.6 Pretrain 結果摘要（164/164 完成，2026-06-13）

| 階段 | 結果 |
|------|------|
| Pretrain 7A | **108/108 success**（`vaewc_round7A_exp010_control_refinement`） |
| Pretrain 7B | **56/56 success**（`vaewc_round7B_vicreg_focused_ablation`） |
| GPU 平行度 | **`config/gpu_parallel_profile.json`**：pretrain **33**、finetune **42**（parallel=20→49% VRAM；36→OOM；33→~99% SM / ~36GB VRAM） |
| 診斷 | `round7_combined/reports/round7_pretrain_diagnostics.csv` |

**Pretrain 指標（diagnostics）：**

| Branch | n | mean kmeans_ari | mean wasserstein | mean exp010-sim | best control | best VICReg | collapse rate |
|--------|---|-----------------|------------------|-----------------|--------------|-------------|---------------|
| 7A | 126* | 0.384 | 0.387 | 0.666 | exp_124 | — | 51% |
| 7B | 56 | 0.317 | 0.463 | 0.576 | exp_003 | exp_041 | 50% |
| combined | 182 | 0.363 | 0.410 | 0.638 | exp_128 | exp_041 | 51% |

\*7A 載入含 OOM 重試產生之額外 checkpoint；sweep 以 manifest **108 success** 為準。

### 15.7 Selection + Finetune 結果（2026-06-13 完成）

| 階段 | 結果 |
|------|------|
| Selection（7C） | **30 模型**（`round7_diverse_downstream_probe`，含 exp_010 / exp_012 / exp_001 / exp_005 / exp_746） |
| Finetune 首輪 | **120/120 success**（30×4 mini config）；首輪 parallel=42 因 CUBLAS 競爭失敗 → **重跑 parallel=26** 成功 |
| Aggregate / report | `round7_combined/aggregate/aggregate_scores.csv`、`reports/final_selection_report.md`、`reports/run_summary.json` |
| Finetune sensitivity（7D） | _pending_ |
| 30 模型 mean Avg TCGA | **0.5109** |
| 超越 R6 exp_010（0.5569） | **2 / 30**（exp_048、exp_021） |

**Downstream Top-5（Average_TCGA_AUC_mean，4 finetune runs 平均）：**

| Rank | Model | Avg TCGA | Global TCGA | Branch | Selection group |
|------|-------|----------|-------------|--------|-----------------|
| 1 | **exp_048** | **0.5918** | 0.5836 | 7B VICReg | G2_vicreg_active |
| 2 | **exp_021** | **0.5723** | 0.6035 | 7B VICReg | G6_high_integrated_proxy |
| 3 | exp_178 | 0.5444 | 0.5843 | 7A control | G8_fill_ranked |
| 4 | exp_127 | 0.5385 | 0.5778 | 7A control | G8_fill_ranked |
| 5 | exp_041 | 0.5328 | 0.6022 | 7B VICReg | G2_vicreg_active |

**Forced baseline 對照（R7 finetune）：**

| Model | Avg TCGA | 備註 |
|-------|----------|------|
| exp_010（7A retrain） | 0.4835 | 低於 R6 exp_010 **0.5569**（不同 checkpoint / sweep 設定） |
| exp_012 | 0.4839 | |
| exp_005 | 0.5197 | historical baseline |
| exp_034（selection #1 by exp010-sim） | 0.5071 | pretrain proxy 與 downstream 仍錯位 |

**觀察：**

1. **7B VICReg** 產出 Round 7 最佳下游：**exp_048**（+0.035 vs R6 exp_010）。
2. **7A control refinement** 未在 Avg TCGA 上超越 R6 exp_010；selection 偏 exp010-like 的 G1 模型下游普遍偏低。
3. exp_048 / exp_021 的 Integrated Avg TCGA（0.544 / 0.538）仍低於 primary Avg TCGA，但 primary 指標已達成功標準。
4. Finetune 重跑耗時約 **4 h**（parallel=26，~30 min/batch）。

```bash
# 重現 aggregate（finetune 已完成時）
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py aggregate \
  --run-dir result/optimization_runs/round7_combined
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py report \
  --run-dir result/optimization_runs/round7_combined
```

### 15.8 Final recommendation

| 決策 | 建議 |
|------|------|
| **Round 7 主線 checkpoint** | **exp_048**（7B VICReg，Avg TCGA **0.5918**） |
| 次選 | **exp_021**（7B VICReg，Avg TCGA **0.5723**） |
| R6 exp_010 地位 | 仍為 R6 定案基準；R7 7A 重訓 exp_010 未復現 0.5569 |
| 7D finetune sensitivity | 可選：對 exp_048 / exp_021 跑 8 combos/checkpoint，驗證 finetune 是否還有 headroom |
| Pretrain 結論 | 7A 鄰域掃描未穩定提升 downstream；**VICReg-only 7B** 為 R7 主要增益來源 |

---

## 16. Round 8 broad architecture and hyperparameter confirmation（2026-06-10）

> 操作手冊：`docs/round8_optimization_manual.md`

### 16.1 Motivation from Round 7

Round 7 定案：**exp_048** Avg TCGA **0.5918**（7B VICReg）；**exp_021** **0.5723**。7A control refinement 未穩定超越 R6 exp_010。Round 8 **不新增 loss / 方法**，在 VAEwC + VICReg-only / control-like 架構內廣泛確認 latent、encoder、GAN schedule、VICReg strength 與 two-stage finetune。

### 16.2 Round 8A control architecture broad sweep

- Sweep：`vaewc_round8A_control_arch_broad.json`（**288 jobs**）
- Baseline：`params_proto_base_exp048_context_broad_control.json`
- 掃描：latent 32/48/64/96/128、encoder 多 family、dropout、lambda_cls、GAN schedule
- 無 active tumor loss

### 16.3 Round 8B VICReg architecture broad sweep

- Sweep：`vaewc_round8B_vicreg_arch_broad.json`（**224 jobs**）
- Baseline：`params_proto_base_exp048_exp021_vicreg_broad.json`
- 主線：VICReg var/cov + architecture + schedule
- 無 topology / class-gap / SupCon / subspace

### 16.4 Round 8C architecture-diverse selection

- Mode：`round8_architecture_broad_probe`（`tools/round8_selection.py`）
- Groups：G1–G8 diversity + G9 forced baseline + G10 fill
- Forced baselines：**exp_048, exp_021, exp_010, exp_012, exp_005, exp_746**
- Combined run：`result/optimization_runs/round8_combined`
- Top-K：**50**（first-pass finetune **200 jobs** = 50×4 mini）

### 16.5 Round 8D two-stage finetune

| 階段 | Config | Jobs |
|------|--------|------|
| First-pass | `config/params_finetune_mini.json` | 50×4 = **200** |
| Second-pass sensitivity | `config/finetune_sweeps/round8_finetune_sensitivity_broad.json` | **9×24 = 216**（實際選 9 模型） |
| 選模工具 | `tools/build_round8_finetune_sensitivity_select.py` | |
| Run dir | `result/optimization_runs/round8_finetune_sensitivity` | |

GPU：`pretrain parallel=33`、`finetune parallel=26`（見 `config/gpu_parallel_profile.json`）。

### 16.6 Results and interpretation（2026-06-14 完成；pretrain retry 2026-06-14 補完）

| 階段 | 結果 |
|------|------|
| Pretrain 8A | **288/288 success**（初跑 284/288；4 failed 已 retry 成功） |
| Pretrain 8B | **224/224 success**（初跑 222/224；2 failed 已 repair/retry 成功） |
| Pretrain 合計 | **512/512 success**（100%） |
| Selection（8C） | **50 模型**（`round8_architecture_broad_probe`；基於初跑 **506** checkpoint） |
| First-pass finetune | **200/200 success** |
| Second-pass sensitivity | **216/216 success**（**9 模型** × 24 combos） |
| 執行時間 | ~**18.4 h**（2026-06-13 16:37 → 2026-06-14 11:02 UTC）；pretrain retry 另 ~**3 min** |
| 主成功標準（> 0.5918） | **0 / 50** 未達成 |
| 次標準（> 0.5569 R6 exp_010） | **5 / 50** |

**初跑失敗與補跑（pretrain retry）：**

| Branch | job_id | 原因 | 處理 |
|--------|--------|------|------|
| 8A | exp_proto_274/277/284/285 | CUBLAS / CUDA OOM（parallel=33） | `tools/run_round8_pretrain_retry.sh`（parallel=12）→ **exp_289–292** |
| 8B | exp_proto_015 | 訓練完成但 `pretrain_model_select.csv` 並行空檔 | 直接標記 success（**exp_029** 已完整） |
| 8B | exp_proto_223 | CUDA OOM | retry → **exp_224** |

修復：`pretrain_VAEwC.py` 空 CSV append、`optimization_runner.py` 從 log 解析 `result_dir`、`repair_pretrain_manifest.py --force-from-logs`。下游 finetune / selection **未重跑**（6 個新 checkpoint 未納入 50-model selection）。

**First-pass Downstream Top-10（Average_TCGA_AUC_mean）：**

| Rank | Model | Avg TCGA | Global TCGA | Branch | 備註 |
|------|-------|----------|-------------|--------|------|
| 1 | **exp_188** | **0.5777** | 0.6147 | 8A control | latent=64, encoder wide_768 |
| 2 | exp_021 | 0.5723 | 0.6041 | forced（R7 7B） | historical baseline |
| 3 | exp_010 | 0.5644 | 0.6236 | forced | historical baseline |
| 4 | exp_048 | 0.5630 | 0.5813 | forced（R7 最佳） | 未復現 R7 **0.5918** |
| 5 | exp_155 | 0.5610 | 0.5726 | 8A control | low dropout probe |
| 6 | exp_080 | 0.5543 | 0.6154 | 8B VICReg | latent64 vicreg |
| 7 | exp_012 | 0.5514 | 0.6120 | forced | |
| 8 | exp_217 | 0.5438 | 0.5977 | 8A control | latent=96 |
| 9 | exp_129 | 0.5405 | 0.5871 | — | |
| 10 | exp_177 | 0.5403 | 0.5721 | — | |

**Forced baseline 對照（first-pass）：** exp_048 **0.5630**、exp_021 **0.5723**、exp_010 **0.5644**、exp_012 **0.5514**、exp_005 **0.5371**、exp_746 **0.5225**。

**Second-pass sensitivity：** 最佳仍為 exp_188 **0.5479**（低於 first-pass **0.5777**，Δ ≈ **−0.030**）；broad classifier grid **未帶來提升**。

**Pretrain 診斷（combined 512 loaded）：**

| Branch | n | mean kmeans_ari | mean wasserstein | collapse rate | vicreg rate | best model |
|--------|---|-----------------|------------------|---------------|-------------|------------|
| 8A | 288 | 0.362 | 0.461 | 50% | 0% | control **exp_171** |
| 8B | 224 | 0.206 | 0.547 | 51% | 100% | VICReg **exp_123** |
| combined | 512 | 0.294 | 0.498 | 51% | 44% | — |

**架構掃描觀察（50 模型 finetune 子集）：**

| 維度 | 觀察 |
|------|------|
| latent_size | 32 mean 最高（n=4）；**64 出現全局最佳** exp_188 |
| encoder | **wide_768** max 最佳（0.5777）；standard_512 樣本最多 |
| dropout | 0.10 max 略優於 0.05 |
| lambda_cls | 25 mean 略優於 20/15 |
| VICReg vs control | 在 selected 50 內 mean 接近（0.514 vs 0.514） |
| sensitivity | first-pass **優於** second-pass（無 ≥0.005 提升） |

**結論：** Round 8 廣泛架構掃描**未超越 Round 7 exp_048（0.5918）**。R8 最佳 **exp_188**（8A control、latent64、wide encoder）為新候選但仍低 **0.014**。R7 forced baseline 在 R8 finetune 下亦未復現 R7 水準，暗示 sweep 內 checkpoint 與 R7 選模 checkpoint 不完全可比，或 downstream 對 architecture 變化敏感。

### 16.7 Final recommendation

| 決策 | 建議 |
|------|------|
| **全專案主線 checkpoint** | 仍採 **R7 exp_048**（Avg TCGA **0.5918**） |
| Round 8 最佳 | **exp_188**（8A control，Avg TCGA **0.5777**）作次選 / ablation reference |
| 主成功標準 | **未達成**（0/50 > 0.5918） |
| 架構結論 | latent64 + wide_768 encoder 在 R8 略優，但未穩定超越 R7 VICReg 定案 |
| Finetune sensitivity | second-pass **無增益**；不宜再放大 classifier grid |
| 下一輪方向 | 固定 **R7 exp_048** pretrain；若繼續優化，優先 **finetune / 資料協議** 而非再擴 pretrain architecture grid |

---

## 17. Round 9 deconfounding QC and baseline reproduction

> 操作手冊：`docs/round9_diagnostics_manual.md`

### 17.1 Motivation

Round 8 定案：全專案主線仍為 **R7 exp_048**（0.5918）；R8 廣泛架構掃描未超越主線。Round 9 **不新增 training loss**，改為對既有 deconfounding / domain adaptation 做 **quality control**：驗證 global source/target 對齊是否代表 **cancer-type conditional** 對齊，並保留 cancer biology 與 response-relevant heterogeneity。

### 17.2 Baseline resolution

- Config：`config/round9_baselines.json`
- Tool：`tools/round9_baseline_resolver.py`
- Required baseline：**exp_048**（找不到則 fail fast）
- Optional baselines：**exp_021, exp_188, exp_010, exp_012, exp_746**
- 輸出：`result/optimization_runs/round9_diagnostics/baselines/resolved_baselines.csv`

### 17.3 Reproduction setup

- Tool：`tools/build_round9_reproduction_manifest.py`
- 每個 resolved baseline × **3 seeds**（101 / 202 / 303）
- Hyperparameters 從 checkpoint `params.json` 還原（不手動猜測）
- Run dir：`result/optimization_runs/round9_reproduction`
- `pretrain_VAEwC.py` 支援 `random_seed` 參數

### 17.4 Global alignment diagnostics

- Tool：`tools/analyze_deconfounding_qc.py`
- 指標：FID、Wasserstein、global domain AUC、kmeans_ari/nmi、silhouette、Davies-Bouldin
- 整合既有 pretrain 輸出（不重新產生 t-SNE / UMAP）

### 17.5 Conditional domain leakage diagnostics

- Tool：`tools/analyze_conditional_domain_leakage.py`
- 固定 cancer type 後評估 source/target domain classifier AUC
- `leakage_strength = abs(AUC - 0.5)`（含 AUC < 0.5 反轉情形）
- 輸出：`conditional_domain_auc_by_cancer.csv`、`conditional_domain_auc_summary.csv`

### 17.6 Cancer prototype diagnostics

- Tool：`tools/analyze_cancer_prototypes.py`
- 同癌別 source/target prototype 距離 + inter-cancer margin
- 區分 good deconfounding vs biology collapse

### 17.7 Latent stability diagnostics

- Tool：`tools/analyze_latent_stability.py`
- active dimensions、effective rank、off-diagonal covariance、collapse / redundancy flags

### 17.8 Finetune and aggregate

- Tool：`tools/build_round9_finetune_select.py`
- 選入規則：每個 reproduction checkpoint 全選（預期 **6×3 = 18** models；missing baseline 則更少）
- Finetune：`config/params_finetune_mini.json`（4 combos / model）
- Run dir：`result/optimization_runs/round9_diagnostics`
- Pipeline：`tools/run_round9_diagnostics_pipeline.sh`

### 17.9 Deconfounding QC interpretation

QC 狀態（`deconfounding_qc_status`）：

| 狀態 | 意義 |
|------|------|
| `good_conditional_deconfounding` | global + conditional 對齊良好，cancer retention 佳 |
| `global_only_alignment` | global 佳但 conditional leakage 高 |
| `biology_collapse_risk` | global 佳但 cancer retention / margin 差 |
| `insufficient_evidence` | 樣本不足或 latent 缺失 |

Final report：`tools/analyze_round9_diagnostics.py` → `round9_final_report.md`

### 17.10 Round 10 recommendations

- Template only：`config/pretrain_sweeps/vaewc_round10_cond_adv_template.json`（**Round 9 不執行**）
- Round 10 Conditional ADV 起點由 Round 9 `round9_per_cancer_problem_list.csv` 與 seed reproducibility 報告決定
- Go 條件：exp_048 可重現、conditional leakage 報告可用、finetune aggregate 完成

### 17.11 Results and interpretation（2026-06-22 完成）

| 階段 | 結果 |
|------|------|
| Baseline 解析 | **6/6** resolved（exp_048 required 成功） |
| Pretrain reproduction | **18/18 success**（6 baselines × 3 seeds） |
| Finetune mini | **72/72 success**（18 models × 4 combos） |
| GPU 設定 | pretrain **parallel=33**、finetune **parallel=26** |
| 執行時間 | ~**2.9 h**（2026-06-22 02:18 → 05:09 UTC） |
| Post-hoc 修復 | TCGA patient-key 對應修正後重跑 diagnostics |

**3-seed 下游重現（Avg TCGA mean ± std）：**

| source_exp_id | Avg TCGA mean | std | macro_cond AUC mean | leakage mean | repro flag |
|---------------|---------------|-----|---------------------|--------------|------------|
| **exp_048** | **0.5349** | 0.026 | 0.866 | 0.366 | variable |
| exp_021 | 0.5059 | 0.034 | 0.875 | 0.375 | variable |
| exp_188 | 0.5117 | 0.013 | 0.873 | 0.376 | variable |
| exp_010 | 0.4991 | 0.022 | 0.875 | 0.375 | variable |
| exp_012 | 0.5034 | 0.010 | 0.879 | 0.379 | variable |
| exp_746 | 0.4928 | 0.044 | 0.881 | 0.381 | variable |

**各 baseline 最佳 reproduction（finetune 後）：**

| source | best model | Avg TCGA | seed |
|--------|------------|----------|------|
| **exp_048** | exp_010 | **0.5671** | 303 |
| exp_746 | exp_012 | 0.5524 | 101 |
| exp_021 | exp_011 | 0.5344 | 202 |
| exp_188 | exp_018 | 0.5291 | 101 |
| exp_010 | exp_014 | 0.5188 | 202 |
| exp_012 | exp_002 | 0.5182 | 303 |

**Deconfounding QC（18 models）：**

| 狀態 | count | 解讀 |
|------|-------|------|
| `global_only_alignment` | **8** | global domain 可分性中等，但 conditional leakage 偏高 |
| `insufficient_evidence` | **10** | 部分模型 cancer retention / margin 證據不足 |
| `good_conditional_deconfounding` | **0** | 無模型通過完整 conditional QC |

**exp_048 診斷（3 seeds）：** global domain AUC **0.65–0.71**；macro conditional domain AUC **0.84–0.88**（leakage strength **0.34–0.38**）→ 屬 **global_only_alignment** 或 **insufficient_evidence**，**非** good conditional deconfounding。

**Diagnostics ↔ downstream（exploratory, n=18）：** conditional leakage 與 Avg TCGA 呈負相關（spearman **−0.49**）；macro conditional domain AUC 與 Avg TCGA **−0.47**。global FID/Wasserstein 與 downstream 相關弱。

**Round 10 高優先癌別（conditional leakage + prototype distance）：** Brain、Esophageal、Liver、Lung、Ovarian。

**結論：**

1. **exp_048 可重現但下游未復現 R7 0.5918**（reproduction 最佳 **0.5671**，mean **0.5349**）。
2. 現有 deconfounding 呈現 **global 改善、conditional leakage 仍高**（多數模型 `global_only_alignment`）。
3. Round 9 **達成** pipeline / diagnostics 基礎建設；**支持** Round 10 Conditional ADV 從 **exp_048** 出發，優先改善上述高 leakage 癌別。
4. 全專案主線仍建議 **R7 exp_048 原始 checkpoint**（0.5918），reproduction 變異需進一步調查。

**報告路徑：**
- `result/optimization_runs/round9_diagnostics/final_report/round9_final_report.md`
- `result/optimization_runs/round9_diagnostics/aggregate/aggregate_scores.csv`
- `result/optimization_runs/round9_diagnostics/reports/deconfounding_qc_model_summary.csv`

---

## 18. Round 10 Conditional Adversarial Deconfounding

### 18.1 Motivation from Round 9

Round 9 證明 global deconfounding 有效，但 **同一 cancer type 內** source/target 仍可分（exp_048 macro conditional domain AUC ~0.84–0.88）。Round 10 將 global discriminator `D(z)` 升級為 conditional critic `D_cond(z, cancer_type)`。

### 18.2 Conditional critic design

- `tools/conditional_adv.py`：`CancerConditionEncoder`、`ConditionalDomainCritic`、conditional WGAN-GP、λ schedule。
- Cancer type mapping 寫入 `<result_dir>/metadata/cancer_type_mapping.json`。

### 18.3 10A / 10B / 10C branch design

| Branch | global_adv_mode | Jobs |
|--------|-----------------|------|
| 10A | `baseline_global_only` | 3 |
| 10B | `conditional_replacement` | 108 |
| 10C | `conditional_plus_weak_global` (×0.25) | 12 |

Primary baseline：**exp_048** only。10B/10C 關閉 proto / SupCon / topology / subspace；10B 另清零 tumor VICReg。

### 18.4 Config generation

```bash
python tools/round10_config_builder.py \
  --settings config/round10_cond_adv_settings.json \
  --outdir result/optimization_runs/round10_cond_adv \
  --force
```

### 18.5 Training behavior

- `conditional_adv_enabled=false`：與舊版 pretrain 完全一致。
- `conditional_replacement`：只訓練 conditional critic；generator 使用 conditional adversarial loss。
- `conditional_plus_weak_global`：conditional + 弱 global guard。

### 18.6 Selection strategy

`round10_cond_adv_qc`：綜合 conditional leakage 改善、cancer retention、global alignment 安全度；Top-K=24。詳見 `tools/round10_selection.py`。

### 18.7 Finetune and aggregate

```bash
bash tools/run_round10_cond_adv_pipeline.sh
```

### 18.8 Results and interpretation

**執行日期：** 2026-06-22（Docker `dapl:5.1`）。輸出：`result/optimization_runs/round10_cond_adv/`。

| 階段 | 結果 |
|------|------|
| Pretrain | **115/123** 成功（8 失敗，皆 10B；7× `λ=0.001`，2× `λ=0.0003 dim=16`） |
| Selection | **24** 模型（20×10B，4×10C；10A 未進 Top-24） |
| Finetune | **96/96** 成功 |
| `round10_success_status` | **`no_conditional_improvement`** |

**Pretrain 分支（成功模型）：**

| Branch | n | mean wasserstein | mean kmeans_ari |
|--------|---|------------------|-----------------|
| 10A | 3 | 0.51 | 0.16 |
| 10B | 100 | 1.63 | 0.58 |
| 10C | 12 | 1.30 | 0.61 |

Conditional ADV 已實際訓練（`gan_metrics.json` 含 `conditional_adv_enabled`、`cond_critic_loss_mean` 等）。**但未重跑 Round 9 式 conditional leakage diagnostics**；pretrain summary 中 `mean_conditional_leakage_strength` 為 NaN，QC 狀態依 wasserstein / structure 代理判定。

**Downstream（24 模型 × 4 finetune combos）：**

| 參考 | Avg TCGA |
|------|----------|
| **Round 10 最佳 `exp_111`**（10C，`λ=0.001`，dim=16） | **0.5749** |
| Round 9 exp_048 reproduction 最佳 | 0.5671 |
| R7 原始 exp_048 | 0.5918 |
| Round 10 Top-24 mean | 0.5193 |

**結論：**

1. Downstream **優於 Round 9 reproduction**（+0.0078），但仍 **低於 R7 原始 0.5918**。
2. 最佳單模型為 **10C weak global guard**（非純 10B replacement）。
3. **無法宣稱 conditional leakage 已改善** — 需補跑 Round 9 diagnostics 後再定案。
4. `λ=0.001` 在 10B 上 pretrain 失敗率高，後續 sweep 應避開或加穩定化。

分析工具：`tools/analyze_round10_cond_adv.py`。  
完整報告：`docs/round10_final_report.md`（runtime 副本：`result/optimization_runs/round10_cond_adv/final_report/round10_final_report.md`）

### 18.9 Round 11 decision

Round 11 已完成（見 §19.7–19.8）。最佳 downstream **exp_035** Avg TCGA **0.5828**，超越 Round 10 exp_111。

**手冊：** `docs/round10_conditional_adv_manual.md` · `docs/round11_optimization_manual.md`

---

## 19. Round 11 Stabilized Conditional ADV and SmoothL1 Reconstruction Ablation

### 19.1 Motivation from Round 10

Round 10 最佳 `exp_111`（10C weak global guard）Average_TCGA_AUC_mean = **0.5749**，略高於 Round 9 reproduction **0.5671**，但 `round10_success_status=no_conditional_improvement`——尚未重跑 Round 9 式 conditional leakage diagnostics。Round 11 **不**直接進入 Prototype Alignment，而是：

```text
Round 11 = Round 10 post-hoc conditional QC
         + 10C weak global guard stabilization
         + SmoothL1 reconstruction loss ablation
```

### 19.2 Round 11A post-hoc conditional QC

對 Round 10 Top-24（含 `exp_111`）重跑 Round 9 式 diagnostics：

```bash
python tools/run_round11a_round10_qc.py \
  --round10-root result/optimization_runs/round10_cond_adv \
  --round9-diagnostics result/optimization_runs/round9_diagnostics/final_report \
  --outdir result/optimization_runs/round11_stability_recon/round11a_qc
```

輸出：`round11a_round10_conditional_qc.csv`、`round11a_go_no_go.md` 等。

### 19.3 Round 11B 10C stabilization

主掃 `global_adv_mode=conditional_plus_weak_global`，`λ_global_mult ∈ {0.25, 0.5}`，`λ_cond_adv` 五檔，四種 schedule，3 seeds → **120** jobs；另 **12** 個 10B 小型對照。

### 19.4 Round 11C SmoothL1 reconstruction ablation

`reconstruction_loss_type ∈ {mse, smooth_l1, hybrid_mse_smooth_l1}` 僅用於 **VAE/AE reconstruction**，不替換 classification / domain / response loss。

- **11C-1** global control：27 jobs  
- **11C-2** 10C + reconstruction：36 jobs  

實作：`tools/reconstruction_losses.py` → `tools/model_opt.vaeloss` → `pretrain_VAEwC.py`。

### 19.5 Round 11D small combination

Optional；`config/round11_settings.json` 預設 `round11d_combination.enabled=false`。僅在 11C 無 reconstruction collapse 時啟用。

### 19.6 Selection and finetune

```bash
bash tools/run_round11_pipeline.sh
```

或分步：

```bash
python tools/round11_config_builder.py \
  --settings config/round11_settings.json \
  --outdir result/optimization_runs/round11_stability_recon \
  --force

python tools/optimization_runner.py select \
  --selection-mode round11_stability_qc \
  --top-k 30 \
  --force-baseline-models exp_111
```

Selection mode `round11_stability_qc`（`tools/round11_selection.py`）：保留 10C 穩定化、SmoothL1、MSE control、`exp_111` forced reference。

預期 pretrain jobs：**195**（11B 132 + 11C 63）；finetune 30×4 = 120。

### 19.7 Results (2026-06-22)

| Stage | Result |
|-------|--------|
| Pretrain | **195/195** success |
| Finetune | **120/120** success |
| Best downstream | **exp_035** Avg TCGA **0.5828** (+0.0079 vs Round 10 exp_111) |
| Round 11A QC | exp_111 leakage 0.400 vs exp_048 0.409 (improved) |

**Pretrain latent proxy (mean):**

| reconstruction_loss_type | kmeans_ari | wasserstein |
|--------------------------|------------|-------------|
| hybrid_mse_smooth_l1 | 0.770 | 0.635 |
| smooth_l1 | 0.705 | 0.944 |
| mse | 0.539 | 1.109 |

Best finetune model **exp_035** is 11B 10C stabilization (MSE recon, `λ_cond_adv=0.0001`, weak global guard). SmoothL1 improved latent stability; best downstream came from stabilized 10C rather than pure SmoothL1 finetune top rank.

完整報告：`docs/round11_final_report.md`（runtime：`result/optimization_runs/round11_stability_recon/final_report/round11_final_report.md`）

### 19.8 Round 12 decision

**Recommendation:** `go_prototype_alignment`

條件已滿足：11A 量測到 conditional leakage 下降、pretrain 無 collapse、downstream 超越 Round 10。下一步：以 Top-10C 穩定化候選（如 exp_035）為基底，進入 Conditional ADV + Source-anchor EMA Prototype Alignment。

**手冊：** `docs/round11_optimization_manual.md`

---

## 20. Round 12 Source-anchor EMA Prototype Alignment

### 20.1 Motivation from Round 11

Round 11 best **exp_035**（Avg TCGA **0.5828**）已穩定 10C weak global guard + conditional ADV。Round 12 在相同架構上加入 **source-anchor EMA prototype alignment**，降低 same-cancer source/target prototype gap，不取代 Conditional ADV。

### 20.2 Source-anchor prototype design

- Source EMA anchor：`P_source_ema[c] ← m·P + (1-m)·mean(z_source|c)`，stop-gradient
- Alignment loss：只拉 **target per-cancer prototype** 靠近 source anchor（cosine 主線）
- 保留 `λ_cond_adv=0.0001`、`conditional_plus_weak_global`、`λ_global_mult=0.25`

### 20.3 Round 12A baseline prototype gap diagnostics

```bash
python tools/analyze_round12_baseline_prototype_gaps.py \
  --round11-root result/optimization_runs/round11_stability_recon \
  --outdir result/optimization_runs/round12_proto_alignment/round12a_baseline_qc
```

### 20.4–20.6 Branches

| Branch | Jobs | 說明 |
|--------|------|------|
| 12B main | 36 + 3 no-proto | `λ_proto_align` × schedule × seed；MSE |
| 12C recon | 24 | hybrid / smooth_l1 + proto |
| 12D control | 3 | euclidean metric 小型對照 |
| **Total** | **66** | |

### 20.7 Selection and finetune

- Selection mode：`round12_proto_alignment_qc`
- 強制保留：`exp_035`、`exp_111` reference
- Finetune：30×4 = 120 jobs

```bash
bash tools/run_round12_proto_alignment_pipeline.sh
```

### 20.8 Results

| Stage | Result |
|-------|--------|
| Pretrain | **66/66** success |
| Finetune | **120/120** success |
| Best downstream | **exp_037** Avg TCGA **0.5972** (+0.0144 vs Round 11 exp_035) |
| Baseline gap | exp_035 distance = 0.06041 |
| Active proto configs | target→source anchor distance reduced = **True** |

完整報告：`docs/round12_final_report.md`（runtime：`result/optimization_runs/round12_proto_alignment/final_report/round12_final_report.md`）

### 20.9 Round 13 decision

**Recommendation:** `go_response_features`

Round 12 best `exp_037` 已超越 Round 11 exp_035（0.5828）與 R7 exp_048（0.5918），且 active prototype configs 顯示 target→source anchor distance 下降。

**手冊：** `docs/round12_proto_alignment_manual.md`

---

## 21. Round 13 Prototype-distance Response Features

### 21.1 Motivation from Round 12

Round 12 best **exp_037**（Avg TCGA **0.5972**）已同時改善 prototype gap 與 downstream。Round 13 **不再改 Step 1 latent learning**，只把 prototype geometry 轉成 Step 2 response predictor 輔助 features：

```text
response_input = concat(z, prototype_distance_features)
```

### 21.2 Feature modes

| Mode | 說明 |
|------|------|
| `none` | z-only baseline |
| `own_cancer` | 主分支（own-cancer prototype distances） |
| `all_source_anchors` | 全 cancer source-anchor distance vector |
| `all_source_and_target` | 小型 source+target vector 測試 |
| `own_plus_summary` | 可選 global summary branch |

### 21.3 Pipeline

```bash
bash tools/run_round13_proto_response_pipeline.sh
```

流程：config builder → prototype feature extraction → finetune（80–128 jobs）→ aggregate → final report。

### 21.4 Results (2026-06-24, 120/120 complete)

| Stage | Result |
|-------|--------|
| Prototype feature extraction | **30/30** success |
| Finetune (initial) | 81/120（39 OOM，`parallel=26` GPU 搶占） |
| Finetune (retry) | **39/39** → **120/120** total（`FINETUNE_RETRY_PARALLEL=12`） |
| Best downstream | **r13_exp_008_own_plus_summary** Avg TCGA **0.6112** (+0.0141 vs Round 12 exp_037) |
| Stretch goal 0.6000 | **met** |

**Feature mode 結論：** `own_plus_summary` peak best；`none` (z-only) 對 `exp_035` 仍強（0.6059）。全 anchor vector 仍弱於精簡特徵。

**z-only vs proto：** 6 個 source model 中 **4/6** proto 優於 z-only（exp_008、exp_051、exp_057、exp_018）；`exp_035` / `exp_037` 未受益。

**Retry：** `bash tools/run_round13_finetune_retry.sh`（見 `logs/round13_finetune_retry.log`）

完整報告：`docs/round13_final_report.md`（runtime：`result/optimization_runs/round13_proto_response/final_report/round13_final_report.md`）

### 21.5 Round 14 decision

**Recommendation:** `go_vicreg_stabilizer`

條件已滿足：Best Round 13 **0.6112** > Round 12 **0.5972**。最大 proto 增益在 `exp_008`；Round 14 應在已驗證 stack（`exp_008` proto-response / `exp_035` z-only）上整合低權重 VICReg stabilizer。

**手冊：** `docs/round13_proto_response_manual.md`

