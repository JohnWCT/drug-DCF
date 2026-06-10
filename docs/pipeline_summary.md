# VAEwC / AEwC Prototype InfoNCE — 管線總覽與優化討論

**最後更新：** 2026-06-10  
**文件用途：** 供團隊成員快速理解管線、三輪實驗結論、指標關係，並作為 **Round 4 優化方向討論** 的共用依據。  
**主要程式：** `pretrain_VAEwC.py`、`pretrain_AEwC.py`、`tools/optimization_runner.py`  
**Pretrain 基準：** `result/pretrain_vaewc/exp_746`（嚴格 filter 8/8 通過）  
**下游 TCGA 基準：** `result/pretrain_vaewc_loss/`（exp_746 finetune，Avg TCGA = **0.5462**）  
**最新主線：** `result/optimization_runs/vaewc_proto_infonce_round3_exp746`（Stage 4 完成）  
**目前最佳下游：** Round 3 **exp_018**（Avg TCGA = **0.5695**，+4.3% vs 基準）

### 閱讀導覽

| 章節 | 適合讀者 |
|------|----------|
| §1–2 | 想了解模型與管線流程 |
| §3–4 | 想了解品質門檻與三輪實驗結論 |
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

| 版本 | fid | wasserstein | 說明 |
|------|-----|-------------|------|
| **嚴格**（exp_746 基準） | ≤ 16.95 | ≤ 0.50 | 8 項全過才進候選 |
| **寬鬆**（Round 3 finetune） | ≤ 30.0 | ≤ 0.70 | K-means 門檻不變；用於下游驗證 |

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
