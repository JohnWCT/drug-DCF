# Round 19 Dataset Card

## 用途

Round 19 使用 GDSC development data 建立固定 splits、訓練與角色選擇；internal test 與
五個 TCGA datasets 僅在 final role lock 後使用。TCGA 與 19G TCGA cases 一律標示為
exploratory。

## 外部資料快照

- `gdsc_intersect13`：906 rows
- `tcga_only3`：129 rows
- `DAPL`：178 rows
- `AACDR tcga_only`：97 rows
- `AACDR gdsc_intersect`：425 rows

原始檔案 SHA-256、schema hash、欄位與大小記錄於
`stage19h_reproducibility/dataset_card.json`。

## Case selection

19G case manifest 固定 230 cases：representative 120、contrastive 60、
patient-conditioned 30、TCGA exploratory 20。選案 seed 為 19091，case manifest
SHA-256 為 `0793b91faf4da37ed41936b180aebfc9cb191adff2054af7555a4dd6e6a12741`。

## 限制

- Omics latent dimensions 是 feature blocks，不可直接命名為 genes。
- Patient-conditioned cases 是模型敏感度分析，不等同臨床療效驗證。
- 資料不可用來回溯修改 Round 19F locked roles。
