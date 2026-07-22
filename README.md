# Deconfounding 實驗

本儲存庫用於進行 **deconfounding**（混淆因子控制／去除）相關之建模、訓練與評估；程式與設定分散於 `config/`、`tools/`、`data/`、`input/`、`result/` 等目錄。

## Current recommended drug-response model

**Canonical name: BioCDA-Predictive** (experiment label `E3` / candidate `B_E3`).

The current locked model was selected under repeated drug-held-out validation in Round 20.

- Omics representation: **C32** O2 (96-d = Z64 + context32)
- Drug encoder: **D0** GIN32 / graph32 (global max pool)
- Predictor: **pooled_e3** (`B_E3` = AdapterMLPFusion + Round18ResponseHead)
- Cross-attention: **none**
- Primary use case: unseen-drug prediction
- Model selection: development-only repeated drug-held-out (seeds 52/62/72 × 5 folds)
- Final TCGA evaluation: performed **after** model lock
- Expanded architecture spec: [biocda_predictive_e3_architecture_spec.json](reports/biocda_predictive_e3_architecture_spec.json)

Context selection (Stage 20A): C32 — ΔAUC(C32−C16) = +0.00745

## Project reports

- [Round 20 final report](docs/round20_final_report.md)
- [Round 20 model card](docs/round20_model_card.md)
- [Round 20 inference guide](docs/round20_inference_guide.md)
- [Round 19 final report](docs/round19_final_report.md)
- [BioCDA architecture finalization](docs/biocda_architecture_finalization.md)
- [Round 23 XA no-pooling closure](docs/round23_xa_validation_report.md)

## Scope and limitations

The Round 20 model focuses on **unseen-drug transfer**. Unseen cancer-type transfer was not
optimized in this round. The omics encoder was frozen during formal model selection. The
repository retains an end-to-end-capable path, but encoder unfreezing was not validated as a
formal Round 20 experiment.

## Round 20 post-completion audit

在 Docker `DAPL` 容器內執行（路徑 `/workspace/DAPL`）：

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/round20/round20_cli.py audit --strict'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/round20/round20_cli.py reproduce --strict'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/round20/round20_cli.py release-info'
```

## BioCDA architecture (naming)

Do **not** use bare “BioCDA” for both rounds:

| Name | Meaning | Status |
|------|---------|--------|
| **BioCDA-Predictive** | Round 20 locked C32 + D0 pooled E3 (`B_E3`) | **LOCKED** |
| **BioCDA-XA-Candidate** | No-pooling sample→atom XA (`biocda-xa-v2`, Round 23) | **REJECTED** (performance vs Predictive) |

Architecture diff: [round20_round21_architecture_diff.json](reports/round20_round21_architecture_diff.json)  
Scientific audit (Q1–Q8): [round20_round21_scientific_audit.md](docs/round20_round21_scientific_audit.md)  
E3 expanded spec: [biocda_predictive_e3_architecture_spec.json](reports/biocda_predictive_e3_architecture_spec.json)  
XA v2 spec: [biocda_xa_v2_architecture_spec.json](reports/biocda_xa_v2_architecture_spec.json)

## BioCDA-XA (Round 23 no-pooling closure)

Omics-conditioned atom-level cross-attention **candidate** (`biocda-xa-v2`): Z64+C32 → sample query; GIN atom nodes as K/V; response head on final query only; **no graph pooling**.

- Validation status: **Round 23 complete** — see [Round 23 report](docs/round23_xa_validation_report.md)
- Model lock: `reports/biocda_xa_model_lock.json` status **REJECTED** (`performance_failure`)
- Closest candidate: fresh XA (mean ΔAUC ≈ −0.0043) still failed ≥2/3-seed non-worse rule; transfer/KD worse
- Retain **BioCDA-Predictive** as the only formal prediction model; do **not** use rejected XA attention to explain Predictive
- Round 21 history: [Round 21 report](docs/round21_xa_validation_report.md)
- TCGA comparison (post-hoc, all BioCDA models): [biocda_tcga_comparison.md](docs/biocda_tcga_comparison.md)

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/compare_biocda_tcga.py'
```

## 環境

建議使用專案內 **`Dockerfile`** 建立含 PyTorch（CUDA）與相依套件之映像；細部套件版本請對照 Dockerfile 內 `pip install` 區段。

## 使用範例（預訓練）

以下為 `pretrain_VAEwC.py` 之指令範例（路徑請依實際資料調整）：

```bash
python pretrain_VAEwC.py \
  --config config/params_from_model_select_fulltest_A_loss_earlystop.json \
  --outfolder result/pretrain_vaewc \
  --target_domain tcga \
  --overlap_tcga data/TCGA/PMID27354694_DR_OMICS_ad.csv
```

> 該腳本預設需要 **CUDA GPU**。其餘管線與分析腳本請見專案根目錄內對應之 `.py` 檔。
