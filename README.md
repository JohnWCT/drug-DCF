# Deconfounding 實驗

本儲存庫用於進行 **deconfounding**（混淆因子控制／去除）相關之建模、訓練與評估；程式與設定分散於 `config/`、`tools/`、`data/`、`input/`、`result/` 等目錄。

## Current recommended drug-response model

The current locked model was selected under repeated drug-held-out validation in Round 20.

- Omics representation: **C32** O2 (96-d = Z64 + context32)
- Drug encoder: **D0** GIN32 / graph32
- Predictor: **pooled_e3** (`B_E3`)
- Primary use case: unseen-drug prediction
- Model selection: development-only repeated drug-held-out (seeds 52/62/72 × 5 folds)
- Final TCGA evaluation: performed **after** model lock

Context selection (Stage 20A): C32 — ΔAUC(C32−C16) = +0.00745

## Project reports

- [Round 20 final report](docs/round20_final_report.md)
- [Round 20 model card](docs/round20_model_card.md)
- [Round 20 inference guide](docs/round20_inference_guide.md)
- [Round 19 final report](docs/round19_final_report.md)
- [BioCDA architecture finalization](docs/biocda_architecture_finalization.md)

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

## BioCDA architecture (BioCDA-XA v1)

Patient-conditioned atom cross-attention model for interpretable drug response prediction.
Architecture finalization is **complete**; interpretability outputs (heatmaps, aggregation)
are deferred to the next round.

- Architecture: **BioCDA-XA** (`biocda-xa-v1`) — sample `[Z;C]` queries GIN atom nodes
- Baseline: **D0-Pooled** (factory `pooled_baseline`)
- Report: [BioCDA architecture finalization](docs/biocda_architecture_finalization.md)

```bash
docker exec DAPL bash -lc '/workspace/DAPL/scripts/biocda/run_architecture_finalization.sh'
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
