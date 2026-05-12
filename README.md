# Deconfounding 實驗

本儲存庫用於進行 **deconfounding**（混淆因子控制／去除）相關之建模、訓練與評估；程式與設定分散於 `config/`、`tools/`、`data/`、`input/`、`result/` 等目錄。

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
