# drug-DCF 下一輪（建議 Round 20）Unseen-Drug Closure IDE 操作手冊

> **文件定位**：本手冊提供給 VS Code、Cursor、PyCharm、GitHub Copilot 或其他 IDE coding agent，執行 drug-DCF 專案收尾前的最後一個受控研究 Round。
>
> **研究範圍**：只處理 O2 context dimension、D0 pooled predictor、drug-held-out generalization、locked TCGA response inference，以及 frozen／end-to-end-capable artifact 封存。
>
> **不處理**：unseen cancer type、重新搜尋 omics encoder、GIN64／GINE／MACCS、atom cross-attention、Transformer depth、summary feature、O0／O1／O3／O4、MACCS hybrid、TCGA 驅動選模。

| 欄位 | 本輪規格 |
|---|---|
| Repository | `https://github.com/JohnWCT/drug-DCF` |
| 公開程式基線 | 公開 `main` 可見 Round 18 基建；最新可見提交 `94d7375`（2026-07-12） |
| 建議分支 | `feature/round20-unseen-drug-closure` |
| 建議文件路徑 | `docs/round20_unseen_drug_closure_ide_manual.md` |
| 建議結果根目錄 | `result/optimization_runs/round20_unseen_drug_closure/` |
| 核心問題 | C16 vs C32；locked pooled E3 vs gated pooled fusion |
| 主驗證 | repeated drug-held-out：3 split seeds × 5 folds |
| 主指標 | DrugMacro AUC |
| tie-breaker | DrugMacro AUPRC |
| frozen 部分 | omics encoder／Z generation；context projection artifacts |
| 必須保留 | raw-omics end-to-end-capable forward path 與 encoder 全權重 |
| TCGA 用途 | architecture lock 後執行 inference／evaluation；不得回頭選模 |

> **本機對齊狀態**：**Round 20 ALL_DONE**（Stage 20-0 → 20E + post-completion audit）。
> 最終鎖定：`C32` + `D0` + pooled E3（`B_E3`）；gated fusion 未過 guardrails。
> TCGA 為 post-selection evaluation；`ROUND20_RELEASE_AUDIT=PASS`；`ROUND20_COMPLETION_AUDIT=PASS`。
> 詳見 [`docs/round20_final_report.md`](round20_final_report.md) 與 `stage20c_lock/final_model_lock.json`。
> E3 對應 `F3_best_pooled_o2`（`O2×D0×P0`），經 `tools/round20_e3_resolver.py` fail-closed 解析。

---

## 0. 最終目標與停止條件

本輪完成後，專案必須能對第三者清楚回答：

1. 在固定 frozen `Z64`、D0 `GIN32/graph32` 與既有 pooled E3 predictor 下，`context16` 或 `context32` 哪一個對 unseen-drug 較穩？
2. 在勝出的 context dimension 下，gated pooled fusion 是否穩定優於或至少不劣於 pooled E3？
3. 若 gated model 無穩定增益，是否有足夠證據保留較簡單的 pooled E3 並正式收尾？
4. 鎖定模型是否可沿用既有 TCGA response inference，產生完整 per-sample、per-drug 與 aggregate outputs？
5. 是否已封存 frozen latent path 與 raw-omics end-to-end-capable path，使未來可重新訓練而不破壞本輪結果？

### 0.1 本輪正式停止條件

候選模型只有在以下條件全部成立時，才可取代 baseline：

```text
G1. candidate mean DrugMacro AUC >= baseline mean DrugMacro AUC
G2. 3 個 split seeds 中至少 2 個 seed 的 AUC delta >= 0
G3. candidate mean DrugMacro AUPRC >= baseline mean AUPRC - 0.01
G4. 任一 split seed 的 AUC delta 不得 < -0.02
G5. 所有正式 jobs 完成，無 missing fold、NaN aggregate 或 leakage audit failure
```

簡約性規則：

```text
若 |mean AUC delta| < 0.005，選擇：
1. variance 較低者；
2. worst-seed 較高者；
3. 參數較少者；
4. inference 較快者；
5. artifact dependency 較少者。
```

### 0.2 合法的收尾結果

| 結果 | 最終決策 |
|---|---|
| C32 穩定優於 C16 | 鎖定 C32 |
| C32 與 C16 差異小於 0.005 | 基於 parsimony 鎖定 C16 |
| gated 穩定通過全部 guardrails | 鎖定 gated pooled fusion |
| gated 未通過或與 E3 差異小 | 保留 pooled E3 |
| 兩者皆不穩 | 不宣稱提升；保留歷史 pooled E3，將 instability 列為限制並收尾 |

---

## 1. 給 IDE Agent 的最高層指令

以下可直接貼入 Cursor／Copilot Chat：

```text
你正在修改 JohnWCT/drug-DCF。
目標是建立 Round 20 unseen-drug closure pipeline，完成以下五件事：
1. C16 vs C32，在 fixed O2 + D0 + pooled E3 下做 3 split seeds × 5 folds drug-held-out。
2. 在勝出 dimension 下，比較 pooled E3 vs gated pooled fusion。
3. 依預先鎖定 guardrails 產生 final model lock。
4. model lock 後沿用既有 TCGA response targets 執行 inference／evaluation。
5. 封存 frozen latent 與 raw-omics end-to-end-capable artifacts。

硬性規則：
- 不覆寫 Round 17、18、19 的 config、result、checkpoint、split、report。
- 不猜測 E3 架構；先從本機 Round 19 role lock、deployment policy、checkpoint config、manifest 解析。
- 若 E3 exact config 無法唯一解析，程式必須 fail closed。
- 只有使用者先前核准的 canonical fallback（O2 + D0 + pooled MLP）可在明確 CLI flag 下重建，且必須標記為 reconstructed baseline，不得冒充 original E3。
- 不使用 TCGA、internal 或 post-hoc external metrics 選擇 context dimension、predictor 或 hyperparameter。
- omics encoder 本輪 freeze；但保留 full weights、raw input preprocessing 與 end-to-end forward path。
- D0 的訓練模式、optimizer parameter groups 與 pooling 必須繼承 E3 lock，不得自行改變。
- context16 與 context32 必須使用相同 raw context definition、fit population、normalization、feature order 與 projection method。
- 正式比較必須 paired：相同 split seed、fold、model seed、eligible rows、drug groups與 training budget。
- 每一 stage 先跑 unit tests、synthetic smoke、real-data smoke，通過後才能生成 full manifest。
- 所有 artifact 寫入 Round 20 新目錄，並附 SHA256、git SHA、config hash、seed、fold、environment metadata。
- 不新增 residual MLP、pooled Transformer 搜尋、atom cross-attention、GIN family 搜尋或 context8 主實驗。
```

---

## 2. 證據邊界與既有程式利用原則

### 2.1 公開 repository 已有可沿用基建

公開 `main` 已包含：

```text
tools/round18_config_builder.py
tools/round18_cv_splits.py
tools/round18_dataset.py
tools/round18_eligible_data.py
tools/round18_feature_coverage.py
tools/round18_fusion_models.py
tools/round18_cv_metrics.py
tools/round18_oom_runner.py
tools/round18_prediction_ensemble.py
tools/round18_response_head.py
step1_finetune_latent_pipeline_round18_cv.py
```

同時已有對應測試：

```text
tests/test_round18_config_builder.py
tests/test_round18_cross_attention.py
tests/test_round18_cv_splits.py
tests/test_round18_fusion_models.py
tests/test_round18_gin_node_output.py
tests/test_round18_oom_retry.py
tests/test_round18_pipeline_smoke.py
tests/test_round18_robust_drug_macro.py
tests/test_round18_tcga_ensemble.py
tests/test_round18_train_loop.py
```

TCGA 舊流程可利用：

```text
tools/finetune_tcga_eval.py
tools/inference_utils.py
tools/prediction_export.py
tools/collect_integrated_tcga_eval.py
```

### 2.2 本機 Round 19 資產優先於公開 Round 18

本輪 task 中的 `pooled E3`、drug-held-out splits、O2 artifacts 與 role policy 來自後續 Round 19。

因此解析優先順序為：

```text
1. reports/round19_final_role_lock.json
2. reports/round19_deployment_policy.json
3. docs/round19_stage19f_report.md
4. Round 19 candidate manifest／checkpoint config snapshot
5. Round 19 stage19e drug-held-out summary
6. 使用者核准的 reconstructed pooled MLP fallback
```

**不得**僅依候選名稱 `E3` 猜測其 predictor、dropout、optimizer、checkpoint 或 D0 training mode。

### 2.3 不應修改的歷史檔案

禁止直接改動：

```text
config/round18_architecture_settings.json
config/params_round18_screening.json
tools/round18_*.py
step1_finetune_latent_pipeline_round18_cv.py
result/optimization_runs/round18_architecture/**
任何 round19 已鎖定的 reports／manifests／checkpoints
```

若發現共用 bug：

1. 先新增 regression test；
2. 在 Round 20 adapter 修正；
3. 說明是否影響歷史結果；
4. 不靜默重算舊報告。

---

## 3. 本輪不可變研究契約

### 3.1 Omics 契約

```text
raw omics [B, G]
→ frozen omics encoder
→ Z [B, 64]

raw prototype geometry
→ locked projection
→ context [B, C]

concat(Z, context)
→ O2-C [B, 64 + C]
```

正式比較：

| ID | 組成 | omics dim |
|---|---|---:|
| C16 | `Z64 + context16` | 80 |
| C32 | `Z64 + context32` | 96 |

禁止加入：

```text
summary11
O0 / O1 / O3 / O4
context8 主矩陣
新的 prototype definition
重新訓練 omics encoder
```

### 3.2 Drug 契約

```text
SMILES
→ atom features [N_atoms, 78]
→ 5-layer GIN
→ atom hidden [N_atoms, 32]
→ global max pooling
→ graph embedding [32]
```

固定為：

```text
D0 = GIN32 / graph32 / JK last / max pooling
```

**D0 的權重是否 end-to-end 更新，不在手冊中猜測。**
它必須由 E3 artifact 的 `drug_encoder_training_mode`、optimizer param groups 或 checkpoint metadata 解析並鎖定。

### 3.3 Pooled E3 契約

`pooled E3` 是一個 artifact-resolved alias，不是硬編碼架構名稱。

必須解析：

```text
architecture_family
predictor_class
omics_mode
context_dim
drug_encoder_id
drug_encoder_training_mode
pooling
hidden_dims
activation
dropout
normalization
optimizer param groups
learning rates
weight decay
epoch / early-stop settings
model seed
canonical deployment split seed
checkpoint state-dict schema
```

若 role lock 明確指出 E3 為 pooled MLP，則 input 維度為：

```text
C16: omics80 + drug32 = 112
C32: omics96 + drug32 = 128
```

若 E3 是其他 pooled predictor，僅允許調整 omics input adapter；其核心架構保持不變。

### 3.4 Selection 契約

選模資料只允許：

```text
development source response
repeated drug-held-out CV metrics
```

Selection 程式必須拒絕包含以下欄位的輸入：

```text
tcga
tcga_only
internal_test
external
integrated5
posthoc
cancer_held_out
```

---

## 4. 建議新增檔案與目錄

### 4.1 程式與設定

```text
config/round20_unseen_drug_closure_settings.json
config/round20_guardrails.json

tools/round20_schema.py
tools/round20_e3_resolver.py
tools/round20_context_adapter.py
tools/round20_drug_splits.py
tools/round20_predictors.py
tools/round20_dataset_adapter.py
tools/round20_config_builder.py
tools/round20_metrics.py
tools/round20_selection.py
tools/round20_tcga_inference.py
tools/round20_release_audit.py
tools/analyze_round20.py

step1_finetune_latent_pipeline_round20_cv.py
scripts/run_round20_stage20a.sh
scripts/run_round20_stage20b.sh
scripts/run_round20_stage20c.sh
scripts/run_round20_stage20d.sh
scripts/run_round20_stage20e.sh
```

### 4.2 測試

```text
tests/test_round20_schema.py
tests/test_round20_e3_resolver.py
tests/test_round20_context_adapter.py
tests/test_round20_drug_splits.py
tests/test_round20_predictors.py
tests/test_round20_dataset_adapter.py
tests/test_round20_config_builder.py
tests/test_round20_metrics.py
tests/test_round20_selection.py
tests/test_round20_tcga_inference.py
tests/test_round20_end_to_end_equivalence.py
tests/test_round20_release_audit.py
tests/test_round20_pipeline_smoke.py
```

### 4.3 文件與最終報告

```text
docs/round20_unseen_drug_closure_ide_manual.md
docs/round20_stage20a_report.md
docs/round20_stage20b_report.md
docs/round20_stage20c_report.md
docs/round20_stage20d_report.md
docs/round20_stage20e_final_report.md
```

### 4.4 結果目錄

```text
result/optimization_runs/round20_unseen_drug_closure/
├── audit/
├── data/
├── features/
├── projections/
├── splits/
├── manifests/
├── stage20a_dimension/
├── stage20b_predictor/
├── stage20c_lock/
├── stage20d_tcga/
├── stage20e_release/
├── reports/
└── logs/
```

---

## 5. Stage 20-0：Preflight 與 E3 Fail-Closed Audit

> **目的**：在任何新訓練前，唯一化 pooled E3、確認 C16／C32 可比較、鎖定 source data 與 drug-held-out split policy。

### 5.1 建立分支

```bash
git checkout main
git pull --ff-only
git checkout -b feature/round20-unseen-drug-closure
```

記錄基準：

```bash
git rev-parse HEAD > result/optimization_runs/round20_unseen_drug_closure/audit/base_git_sha.txt
git status --short > result/optimization_runs/round20_unseen_drug_closure/audit/base_git_status.txt
```

### 5.2 E3 resolver

新增 `tools/round20_e3_resolver.py`。

建議 API：

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

@dataclass(frozen=True)
class ResolvedE3:
    source: str
    reconstructed: bool
    architecture_family: str
    predictor_class: str
    drug_encoder_id: str
    drug_encoder_training_mode: str
    pooling: str
    context_dim: int
    hidden_dims: tuple[int, ...]
    dropout: float
    activation: str
    optimizer: Mapping[str, Any]
    training: Mapping[str, Any]
    checkpoint_paths: tuple[str, ...]
    config_hash: str

def resolve_e3(
    repo_root: Path,
    *,
    allow_approved_reconstruction: bool = False,
) -> ResolvedE3:
    ...
```

Resolver 行為：

1. 依既定優先順序尋找 role lock／policy／manifest／checkpoint config。
2. 至少兩個來源相互核對 architecture ID。
3. 驗證 D0、O2、pooled family。
4. 驗證 checkpoint 存在並可讀。
5. 若資訊衝突，拋出 `E3ResolutionError`。
6. 若找不到 exact E3，預設 fail。
7. 只有 `--allow-approved-e3-reconstruction` 才建立 canonical pooled MLP fallback。
8. reconstructed baseline metadata 必須包含：

```json
{
  "baseline_label": "approved_reconstructed_pooled_mlp",
  "is_original_e3": false,
  "reason": "exact E3 artifact unavailable",
  "user_approval_reference": "Round 20 planning decision"
}
```

### 5.3 禁止 E3 resolver 依名稱猜測

以下做法禁止：

```python
if candidate_id == "E3":
    predictor = "pooled_mlp"  # 禁止
```

必須從 artifact 欄位解析。

### 5.4 Context artifact audit

新增：

```bash
python -m tools.round20_context_adapter audit \
  --c16-dir <EXISTING_C16_FEATURE_DIR> \
  --c32-dir <EXISTING_C32_FEATURE_DIR> \
  --out result/optimization_runs/round20_unseen_drug_closure/audit/context_audit.json
```

必查欄位：

```text
latent_dim == 64
context_dim == 16 / 32
raw_context_definition_hash 相同
fit_population_hash 相同
feature_order_hash 相同
normalization_hash 相同
projection_method 相同
source_encoder_checkpoint_hash 相同
ModelID coverage 相同
NaN / Inf count == 0
```

若舊 C32 與 C16 的 fit population 不一致，不可直接比較。

### 5.5 C32 重建規則

如果需要重建 C32：

```text
沿用既有 C16：
- 同一 raw context matrix
- 同一 rows
- 同一 normalization
- 同一 projection algorithm
- 同一 random seed
- 只改 n_components: 16 → 32
```

產物：

```text
projections/context16/
├── projection.pkl
├── metadata.json
└── sha256.txt

projections/context32/
├── projection.pkl
├── metadata.json
└── sha256.txt
```

### 5.6 Drug identity audit

unseen-drug split 必須在 canonical drug identity 上執行，不應只靠原始字串。

建議輸出欄位：

```text
raw_drug_name
normalized_drug_name
raw_smiles
canonical_smiles
inchikey_if_available
drug_group_id
scaffold_id
```

必查：

```text
train drug_group_id ∩ val drug_group_id = ∅
同 canonical SMILES 不得跨 train / val
drug alias 不得跨 train / val
無法 canonicalize 的藥物需列入 exception report
```

### 5.7 Preflight 完成條件

```text
[ ] E3 唯一解析或明確 reconstructed
[ ] C16 / C32 metadata 可比較
[ ] eligible response hash 已鎖定
[ ] drug identity mapping 已鎖定
[ ] 3 個 split seed 已鎖定
[ ] TCGA paths 未出現在 selection config
[ ] baseline smoke checkpoint 可 load
```

---

## 6. 共用設定檔

建立 `config/round20_unseen_drug_closure_settings.json`：

```json
{
  "round": "round20",
  "purpose": "unseen-drug closure: C16 vs C32 and pooled E3 vs gated fusion",
  "base_round18_settings": "config/round18_architecture_settings.json",
  "round19_role_lock": "reports/round19_final_role_lock.json",
  "round19_deployment_policy": "reports/round19_deployment_policy.json",
  "result_root": "result/optimization_runs/round20_unseen_drug_closure",
  "selection_data_scope": "development_drug_held_out_only",
  "omics": {
    "latent_dim": 64,
    "encoder_mode": "frozen",
    "retain_end_to_end_path": true,
    "dimensions": [16, 32],
    "feature_dirs": {
      "16": "<C16_FEATURE_DIR>",
      "32": "<C32_FEATURE_DIR>"
    },
    "encoder_checkpoint": "<OMICS_ENCODER_CHECKPOINT>",
    "gene_order": "<GENE_ORDER_ARTIFACT>",
    "normalization": "<OMICS_NORMALIZATION_ARTIFACT>"
  },
  "drug": {
    "encoder_id": "D0",
    "input_dim": 78,
    "hidden_dim": 32,
    "num_layers": 5,
    "jk_mode": "last",
    "pool": "max",
    "training_mode": "inherit_from_e3"
  },
  "predictors": [
    "resolved_e3",
    "gated_pooled_fusion"
  ],
  "validation": {
    "split_type": "drug_held_out",
    "split_seeds": [52, 62, 72],
    "n_splits": 5,
    "model_seed": 101,
    "group_column": "drug_group_id",
    "label_column": "Label"
  },
  "selection": {
    "primary_metric": "DrugMacro_AUC",
    "tie_breaker_metric": "DrugMacro_AUPRC",
    "parsimony_delta": 0.005,
    "auprc_max_drop": 0.01,
    "major_fail_auc_delta": -0.02,
    "min_nonworse_seeds": 2
  },
  "tcga": {
    "run_only_after_lock": true,
    "targets_source": "inherit_round18_or_round19",
    "ensemble_method": "inherit_from_e3",
    "canonical_deployment_split_seed": "inherit_from_e3",
    "require_complete_ensemble": true
  }
}
```

### 6.1 不得保留未解析 placeholder

正式 manifest 建立前，下列值不得仍為 placeholder：

```text
<C16_FEATURE_DIR>
<C32_FEATURE_DIR>
<OMICS_ENCODER_CHECKPOINT>
<GENE_ORDER_ARTIFACT>
<OMICS_NORMALIZATION_ARTIFACT>
inherit_from_e3 尚未解析的欄位
```

### 6.2 設定 schema

`tools/round20_schema.py` 必須驗證：

```text
omics.latent_dim == 64
omics.dimensions == [16, 32]，順序可正規化
D0 fields 固定為 78 / 32 / 5 / last / max
split seeds 恰為三個且不重複
n_splits == 5
selection guardrails 與核准值一致
TCGA run_only_after_lock == true
```

---

## 7. Stage 20A：C16 vs C32 Repeated Drug-Held-Out

### 7.1 科學問題

在完全相同的：

```text
frozen Z64
D0 architecture
resolved pooled E3
eligible response rows
drug identity groups
training budget
optimizer
model seed
split seeds / folds
```

下，僅比較：

```text
C16: O2 = Z64 + context16 = 80
C32: O2 = Z64 + context32 = 96
```

### 7.2 Manifest 數量

```text
2 dimensions × 3 split seeds × 5 folds = 30 jobs
```

### 7.3 split 生成

新增 `tools/round20_drug_splits.py`。

建議 API：

```python
def build_repeated_drug_held_out_splits(
    eligible_df,
    *,
    drug_group_column: str,
    label_column: str,
    split_seeds: list[int],
    n_splits: int,
    outdir: str,
) -> dict:
    ...
```

每個 seed 輸出：

```text
splits/drug_held_out_seed52_assignments.csv
splits/drug_held_out_seed62_assignments.csv
splits/drug_held_out_seed72_assignments.csv
splits/drug_held_out_metadata.json
splits/drug_identity_mapping.csv
splits/leakage_audit.json
```

Assignment schema：

```text
row_id
drug_group_id
split_seed
fold_id
split_role
```

### 7.4 fold validity

每個 fold 必須檢查：

```text
train rows > 0
val rows > 0
train drugs > 0
val drugs > 0
train / val drug groups disjoint
val 至少包含可計算 AUC 的 drug groups
全域 label 不只單一類別
```

無效 fold 不得靜默跳過或改 seed。

### 7.5 Context adapter

新增 `tools/round20_context_adapter.py`：

```python
class Round20O2FeatureStore:
    def __init__(self, feature_dir: str, context_dim: int):
        ...

    def get(self, model_id: str) -> "np.ndarray":
        """Return float32 [64 + context_dim]."""
        ...
```

強制 assertion：

```python
expected = 64 + context_dim
if vector.shape != (expected,):
    raise FeatureShapeError(...)
```

metadata 必須提供 component slices：

```json
{
  "feature_mode": "O2",
  "latent_slice": [0, 64],
  "context_slice": [64, 80],
  "context_dim": 16,
  "output_dim": 80
}
```

C32 對應 `context_slice: [64, 96]`。

### 7.6 Baseline input adapter

E3 核心 predictor 不變，只允許建立 dimension-aware input layer。

```python
def build_resolved_e3(*, omics_dim: int, graph_dim: int, resolved: ResolvedE3):
    ...
```

禁止：

```text
C16 與 C32 使用不同 hidden width
C16 與 C32 使用不同 dropout
C16 與 C32 使用不同 optimizer
C16 與 C32 使用不同 epoch / patience
C16 與 C32 使用不同 GIN initialization policy
```

### 7.7 Stage 20A manifest schema

```text
job_id
stage
candidate_id
context_dim
omics_dim
predictor_id
drug_encoder_id
drug_encoder_training_mode
split_seed
fold_id
model_seed
response_path
feature_dir
projection_artifact
split_assignment
checkpoint_init
optimizer_config_hash
result_dir
status
```

Job ID：

```text
20a_e3_c16_s52_f0
20a_e3_c32_s52_f0
...
```

### 7.8 建立 manifest

```bash
python -m tools.round20_config_builder \
  --settings config/round20_unseen_drug_closure_settings.json \
  --stage 20a \
  --outdir result/optimization_runs/round20_unseen_drug_closure
```

必須輸出：

```text
manifests/stage20a_dimension_manifest.csv
audit/stage20a_manifest_validation.json
```

### 7.9 訓練 runner

新增 `step1_finetune_latent_pipeline_round20_cv.py`。

支援模式：

```text
--mode smoke
--mode data_smoke
--mode train
--mode eval
--mode infer
--mode verify_checkpoint
```

CLI：

```bash
python step1_finetune_latent_pipeline_round20_cv.py \
  --mode train \
  --job-row-json '<ROW_JSON>' \
  --settings config/round20_unseen_drug_closure_settings.json
```

### 7.10 訓練輸出

每 job：

```text
result_dir/
├── resolved_config.json
├── environment.json
├── resource_metadata.json
├── train_history.csv
├── best_checkpoint.pt
├── val_predictions.csv
├── metrics.json
├── drug_metrics.csv
├── status.json
└── logs.txt
```

`val_predictions.csv` 至少包含：

```text
row_id
ModelID
drug_name
drug_group_id
label
probability
split_seed
fold_id
candidate_id
```

### 7.11 Stage 20A aggregate

```bash
python -m tools.analyze_round20 \
  --stage 20a \
  --root result/optimization_runs/round20_unseen_drug_closure
```

輸出：

```text
reports/stage20a_per_fold.csv
reports/stage20a_per_seed.csv
reports/stage20a_dimension_summary.csv
reports/stage20a_paired_deltas.csv
reports/stage20a_dimension_decision.json
docs/round20_stage20a_report.md
```

### 7.12 Dimension selection 規則

C32 取代 C16 必須：

```text
mean AUC(C32) >= mean AUC(C16)
至少 2/3 seeds：AUC(C32) >= AUC(C16)
mean AUPRC(C32) >= mean AUPRC(C16) - 0.01
worst seed AUC delta >= -0.02
```

如果：

```text
abs(mean AUC delta) < 0.005
```

鎖定 C16。

Decision JSON：

```json
{
  "stage": "20a",
  "winner_context_dim": 16,
  "baseline_context_dim": 16,
  "challenger_context_dim": 32,
  "mean_auc_delta_c32_minus_c16": 0.0,
  "nonworse_seed_count": 0,
  "guardrails": {},
  "decision_reason": "parsimony",
  "selection_inputs_hash": "..."
}
```

---

## 8. Stage 20B：Pooled E3 vs Gated Pooled Fusion

### 8.1 啟動條件

必須存在且通過 schema：

```text
reports/stage20a_dimension_decision.json
```

不得由 CLI 任意指定另一個 context dimension，除非 `--override-lock`，且 override 只能用於 smoke，正式 run 禁止。

### 8.2 Gated predictor 定義

新增 `tools/round20_predictors.py`。

建議 canonical gated model：

```python
class GatedPooledFusion(nn.Module):
    def __init__(
        self,
        omics_dim: int,
        graph_dim: int = 32,
        fusion_dim: int = 128,
        head_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.omics_proj = nn.Linear(omics_dim, fusion_dim)
        self.drug_proj = nn.Linear(graph_dim, fusion_dim)
        self.omics_norm = nn.LayerNorm(fusion_dim)
        self.drug_norm = nn.LayerNorm(fusion_dim)
        self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, omics: Tensor, drug_graph: Tensor) -> Tensor:
        o = self.omics_norm(self.omics_proj(omics))
        d = self.drug_norm(self.drug_proj(drug_graph))
        g = torch.sigmoid(self.gate(torch.cat([o, d], dim=-1)))
        fused = g * d + (1.0 - g) * o
        return self.head(fused)
```

### 8.3 設計約束

必須：

```text
input omics = Stage 20A winner
input drug = D0 pooled graph32
output = logits [B, 1]
loss 與 E3 相同
D0 training mode 與 E3 相同
training budget 與 E3 相同
```

禁止在同一 Stage 額外加入：

```text
bilinear layer
FiLM 第二版本
residual MLP
Transformer
gate regularization search
不同 fusion dimensions grid
```

### 8.4 超參數策略

第一輪只使用 E3 繼承 preset。

```text
P0 = inherit E3 optimizer / dropout / epoch / early stop
```

只有 gated model 接近但未穩定時，才允許兩組保守 preset：

```yaml
P1:
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  dropout: 0.20

P2:
  learning_rate: 3.0e-4
  weight_decay: 3.0e-5
  dropout: 0.20
```

不做 Cartesian grid。

### 8.5 正式比較數量

首選直接正式比較：

```text
2 predictors × 3 split seeds × 5 folds = 30 jobs
```

若補 P1／P2，只新增 gated candidate，不重跑 E3：

```text
2 gated presets × 3 seeds × 5 folds = 30 additional jobs
```

新增 preset 的決策必須在未查看 TCGA metrics 的狀態下記錄於：

```text
reports/stage20b_preset_amendment.json
```

### 8.6 manifest

```bash
python -m tools.round20_config_builder \
  --settings config/round20_unseen_drug_closure_settings.json \
  --stage 20b \
  --dimension-lock result/optimization_runs/round20_unseen_drug_closure/reports/stage20a_dimension_decision.json \
  --outdir result/optimization_runs/round20_unseen_drug_closure
```

Job IDs：

```text
20b_e3_c16_s52_f0
20b_gated_p0_c16_s52_f0
```

### 8.7 paired comparison

每個 `(split_seed, fold_id)` 必須同時存在 baseline 與 candidate。

Paired table：

```text
split_seed
fold_id
baseline_auc
candidate_auc
delta_auc
baseline_auprc
candidate_auprc
delta_auprc
```

### 8.8 需要額外報告的 gated diagnostics

不將 gate 當作 causal explanation，但應輸出：

```text
gate mean / std
gate saturation proportion (<0.05, >0.95)
per-drug gate distribution
NaN / Inf
```

若 gate 全部接近 0 或 1，標記 `GATE_COLLAPSE_WARNING`，但不單獨作 selection 指標。

---

## 9. Stage 20C：Guardrail、Final Lock 與 Deployment Contract

### 9.1 Selection 程式

新增 `tools/round20_selection.py`。

建議 API：

```python
def select_final_model(
    baseline_summary,
    candidate_summary,
    guardrails,
    *,
    reject_external_columns: bool = True,
) -> dict:
    ...
```

### 9.2 外部欄位拒絕

```python
FORBIDDEN_SELECTION_TOKENS = {
    "tcga", "internal", "external", "integrated5", "posthoc",
    "cancer_held_out", "patient", "target_cohort"
}
```

欄名、JSON keys、path 任何位置出現即 fail。

### 9.3 Predictor selection guardrails

```text
G1 mean AUC candidate >= baseline
G2 >= 2/3 seed non-worse
G3 mean AUPRC drop <= 0.01
G4 worst-seed AUC delta >= -0.02
G5 complete jobs / no NaN / no leakage fail
```

### 9.4 Final parsimony

若 gated 通過 guardrail，但：

```text
mean AUC delta < 0.005
```

預設保留 pooled E3，除非 gated 同時：

```text
variance 明顯更低
且 worst seed 明顯更高
```

此例外須明列數值與 rationale，不可只寫「較穩」。

### 9.5 Lock 文件

輸出：

```text
reports/round20_final_model_lock.json
reports/round20_deployment_contract.json
reports/round20_selection_audit.json
docs/round20_stage20c_report.md
```

Lock schema：

```json
{
  "round": "round20",
  "status": "locked",
  "selected_model_id": "...",
  "selected_context_dim": 16,
  "omics_mode": "O2",
  "omics_dim": 80,
  "drug_encoder_id": "D0",
  "predictor_id": "resolved_e3",
  "drug_encoder_training_mode": "...",
  "selection_scope": "development_repeated_drug_held_out",
  "selection_metrics": {},
  "guardrail_results": {},
  "canonical_deployment_split_seed": 0,
  "checkpoint_contract": {},
  "source_artifact_hashes": {},
  "tcga_metrics_used_for_selection": false,
  "locked_at": "..."
}
```

### 9.6 Lock 不可變性

Lock 建立後：

- 不得修改 architecture／context／hyperparameters；
- TCGA 結果不能回頭改 lock；
- 若發現程式 bug，必須產生新 lock version，保留舊版；
- 所有 inference outputs 記錄 lock SHA256。

---

## 10. Stage 20D：Locked TCGA Response Inference

### 10.1 原則

TCGA response 已有 labels 且已執行過 inference，因此本輪定位是：

```text
locked post-selection evaluation / reproducible inference
```

不是 untouched external validation。

### 10.2 不直接硬套舊 classifier-only function

舊 `tools/inference_utils.py` 主要假設：

```text
latent + drug embedding → classifier
```

若 final model 是 gated predictor，必須使用與 CV 完全相同的 model factory／forward contract。

因此建立 `tools/round20_tcga_inference.py`，但重用：

```text
TCGA response CSV loader
patient ID mapping
SMILES lookup
prediction export
metric helpers
```

### 10.3 Canonical model loading

```python
def load_locked_round20_model(lock_path: str, checkpoint_path: str, device):
    """Build exact model from lock; strict state_dict load only."""
    ...
```

要求：

```text
strict=True load
config hash match
context projection hash match
omics encoder hash match
D0 config match
predictor class match
```

### 10.4 Ensemble policy

主要 inference ensemble 必須繼承 E3 的既有正式 policy。

推薦流程：

```text
1. 從 E3 lock 解析 canonical deployment split seed。
2. 使用該 seed 的完整 5-fold checkpoints。
3. probability mean ensemble。
4. require all five folds。
```

3×5 的 15-checkpoint ensemble 可作 secondary sensitivity output，但不得取代 primary，除非部署 policy 已事先明確鎖定。

### 10.5 TCGA targets

預設沿用 repository 已有 targets source，不重新定義格式。

常見 keys：

```text
gdsc_intersect13
tcga_only3
dapl
aacdr_tcga_only
aacdr_gdsc_intersect
```

實際 target paths 由 config／既有 Round 19 policy 解析。

### 10.6 建立 inference manifest

```bash
python -m tools.round20_config_builder \
  --settings config/round20_unseen_drug_closure_settings.json \
  --stage 20d \
  --model-lock result/optimization_runs/round20_unseen_drug_closure/reports/round20_final_model_lock.json \
  --outdir result/optimization_runs/round20_unseen_drug_closure
```

Manifest 欄位：

```text
job_id
target_key
target_path
fold_id
checkpoint_path
model_lock_hash
ensemble_group
result_dir
status
```

### 10.7 執行 inference

```bash
python -m tools.round20_tcga_inference \
  --manifest result/optimization_runs/round20_unseen_drug_closure/manifests/stage20d_tcga_manifest.csv \
  --model-lock result/optimization_runs/round20_unseen_drug_closure/reports/round20_final_model_lock.json \
  --outdir result/optimization_runs/round20_unseen_drug_closure/stage20d_tcga
```

### 10.8 每個 target 輸出

```text
target_key/
├── fold_predictions/
├── ensemble_predictions.csv
├── sample_predictions.csv
├── per_drug_metrics.csv
├── aggregate_metrics.json
├── coverage_report.json
├── failed_drugs.csv
└── inference_metadata.json
```

### 10.9 Metrics

至少包含：

```text
Global AUC
Global AUPRC
DrugMacro AUC
DrugMacro AUPRC
valid drug count
total drug count
sample coverage
drug coverage
bootstrap CI（如既有流程已有）
```

DrugMacro 必須遵守最低樣本／正負樣本要求；無效 drug 保留在表格中並標記原因，不得當成 0.5 靜默平均。

### 10.10 TCGA 結果解讀

報告必須標記：

```text
- architecture 在 TCGA 前已鎖定；
- TCGA metrics 未參與 Stage 20A／20B／20C selection；
- TCGA 已被歷史 Round 查看，結果屬 post-selection／post-hoc evaluation；
- 不因 TCGA 表現回頭改模型。
```

---

## 11. Stage 20E：Frozen 與 End-to-End-Capable Artifact 封存

### 11.1 最終 release 目錄

```text
stage20e_release/
├── model/
│   ├── final_model_lock.json
│   ├── fold_checkpoints/
│   ├── model_card.md
│   └── state_dict_schema.json
├── omics/
│   ├── encoder_checkpoint.pt
│   ├── encoder_config.json
│   ├── gene_order.json
│   ├── normalization_artifact.*
│   ├── frozen_latent_manifest.json
│   └── latent_feature_metadata.json
├── context/
│   ├── selected_projection.*
│   ├── selected_projection_metadata.json
│   ├── c16_reference/
│   └── c32_reference/
├── drug/
│   ├── d0_config.json
│   ├── smiles_source_metadata.json
│   ├── graph_preprocessing_version.json
│   └── drug_identity_mapping.csv
├── splits/
│   ├── repeated_drug_held_out/
│   └── canonical_deployment_split/
├── inference/
│   └── tcga/
├── environment/
│   ├── requirements_freeze.txt
│   ├── conda_environment.yml
│   ├── docker_metadata.json
│   ├── cuda_metadata.json
│   └── git_metadata.json
├── manifests/
├── reports/
├── checksums.sha256
└── release_manifest.json
```

### 11.2 Frozen path 必須可重現

```text
precomputed Z64
+ selected context
→ O2
→ D0 + predictor
→ prediction
```

需要封存：

```text
latent vector dtype
sample key format
feature order
component slices
missing-sample policy
projection transform API
```

### 11.3 End-to-end-capable path

雖然本輪不解凍 encoder，程式仍需支援：

```text
raw omics
→ preprocessing
→ encoder
→ Z64
→ context transform
→ O2
→ predictor
```

設定介面：

```yaml
omics:
  mode: frozen_latent  # final Round 20
  freeze_encoder: true
  allow_raw_omics_forward: true
```

未來可改：

```yaml
omics:
  mode: raw_omics
  freeze_encoder: false
```

但該未來設定不屬於 Round 20 正式結果。

### 11.4 Frozen vs raw-path equivalence test

新增 `tests/test_round20_end_to_end_equivalence.py`。

對固定 batch：

```text
path A: load precomputed Z
path B: raw omics → frozen encoder → Z
```

要求：

```python
torch.testing.assert_close(z_a, z_b, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(logit_a, logit_b, rtol=1e-5, atol=1e-6)
```

若 preprocessing 有非確定性，必須先移除或鎖定；不得放寬到無意義 tolerance。

### 11.5 Checksum

```bash
find stage20e_release -type f ! -name checksums.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > stage20e_release/checksums.sha256
```

### 11.6 Release audit

```bash
python -m tools.round20_release_audit \
  --release-dir result/optimization_runs/round20_unseen_drug_closure/stage20e_release \
  --strict
```

Audit 必查：

```text
[ ] final lock 存在且 hash 正確
[ ] 所有 primary ensemble checkpoints 存在
[ ] checkpoint 可 strict load
[ ] selected projection 與 metadata 一致
[ ] encoder full weights 存在
[ ] gene order / normalization 存在
[ ] split manifests 存在
[ ] TCGA prediction coverage report 存在
[ ] environment metadata 存在
[ ] checksums 全數通過
[ ] frozen/raw equivalence test 通過
[ ] TCGA 未出現在 selection audit
[ ] git working tree 狀態已記錄
```

---

## 12. 詳細程式修改建議

### 12.1 `tools/round20_e3_resolver.py`

責任：

- discovery；
- cross-source consistency；
- checkpoint introspection；
- approved fallback；
- resolved config snapshot。

不得負責 training 或 metrics。

### 12.2 `tools/round20_context_adapter.py`

責任：

- feature metadata validation；
- C16/C32 shape；
- projection metadata；
- sample coverage；
- frozen/raw O2 composition。

不得在 dataset `__getitem__` 中偷偷 fit PCA。

### 12.3 `tools/round20_drug_splits.py`

責任：

- canonical drug groups；
- repeated grouped split；
- leakage audit；
- split hashes。

不得使用 sample-level random split。

### 12.4 `tools/round20_predictors.py`

責任：

- resolved E3 wrapper；
- gated pooled fusion；
- consistent forward contract；
- model metadata export。

統一 contract：

```python
logits = predictor(
    omics_features,       # [B, 80] or [B, 96]
    pooled_drug_features, # [B, 32]
)
# logits: [B, 1]
```

### 12.5 `tools/round20_dataset_adapter.py`

優先 wrap Round 18／19 dataset，不重寫 SMILES graph cache。

新增內容只限：

```text
context dimension switch
drug_group_id attachment
row_id preservation
metadata propagation
```

### 12.6 `tools/round20_metrics.py`

優先重用 Round 18 robust drug macro。

新增：

```text
per-seed aggregate
paired deltas
worst-seed
non-worse seed count
major-fail flags
job completeness
```

### 12.7 `tools/round20_selection.py`

只接受 aggregate artifacts；不自行掃描所有 result 目錄，以降低誤讀 TCGA 的風險。

### 12.8 `tools/round20_tcga_inference.py`

使用 final model factory，而不是假設 classifier 是 concat MLP。

### 12.9 `tools/round20_release_audit.py`

只做驗證，不修復 artifact；修復必須由獨立命令完成並留下紀錄。

---

## 13. 測試規格

### 13.1 Unit tests

#### E3 resolver

```text
- exact role lock 可解析
- role lock / checkpoint config 衝突時 fail
- 無 artifact 預設 fail
- approved fallback 需明確 flag
- reconstructed 不得標成 original E3
```

#### Context

```text
- C16 output dim 80
- C32 output dim 96
- latent slice 0:64
- metadata mismatch fail
- NaN fail
- coverage mismatch fail
```

#### Splits

```text
- train / val drug disjoint
- alias / canonical SMILES 不跨 split
- deterministic by seed
- 不同 seeds 不應完全相同（除非資料限制並明確報告）
```

#### Predictors

```text
- E3 C16 forward [B,80]+[B,32]→[B,1]
- E3 C32 forward [B,96]+[B,32]→[B,1]
- gated forward shape
- gate in [0,1]
- backward 有 gradients
- CPU smoke
- CUDA smoke（可用時）
```

#### Selection

```text
- 全 guardrails pass → candidate
- AUPRC fail → baseline
- 1/3 non-worse → baseline
- major fail → baseline
- delta < 0.005 → simpler model
- TCGA key → hard fail
```

#### TCGA

```text
- model lock strict load
- all 5 folds required
- probability mean 正確
- invalid drug metrics 不被當 0.5 平均
- coverage report 正確
```

### 13.2 Regression tests

執行既有 Round 18 tests：

```bash
pytest -q \
  tests/test_round18_config_builder.py \
  tests/test_round18_cv_splits.py \
  tests/test_round18_fusion_models.py \
  tests/test_round18_gin_node_output.py \
  tests/test_round18_robust_drug_macro.py \
  tests/test_round18_tcga_ensemble.py \
  tests/test_round18_train_loop.py
```

### 13.3 Round 20 tests

```bash
pytest -q tests/test_round20_*.py
```

### 13.4 Full tests

```bash
pytest -q
```

---

## 14. Smoke 執行順序

### 14.1 Schema smoke

```bash
python -m tools.round20_schema \
  --settings config/round20_unseen_drug_closure_settings.json
```

### 14.2 E3 resolution smoke

```bash
python -m tools.round20_e3_resolver \
  --repo-root . \
  --out result/optimization_runs/round20_unseen_drug_closure/audit/resolved_e3.json
```

若 exact artifact 缺失且要使用核准 fallback：

```bash
python -m tools.round20_e3_resolver \
  --repo-root . \
  --allow-approved-e3-reconstruction \
  --out result/optimization_runs/round20_unseen_drug_closure/audit/resolved_e3.json
```

### 14.3 Synthetic smoke

```bash
python step1_finetune_latent_pipeline_round20_cv.py \
  --mode smoke \
  --outdir result/optimization_runs/round20_unseen_drug_closure/smoke
```

### 14.4 Real-data smoke

每個 context × predictor 至少一個 batch：

```bash
python step1_finetune_latent_pipeline_round20_cv.py \
  --mode data_smoke \
  --settings config/round20_unseen_drug_closure_settings.json \
  --context-dim 16 \
  --predictor resolved_e3

python step1_finetune_latent_pipeline_round20_cv.py \
  --mode data_smoke \
  --settings config/round20_unseen_drug_closure_settings.json \
  --context-dim 32 \
  --predictor resolved_e3

python step1_finetune_latent_pipeline_round20_cv.py \
  --mode data_smoke \
  --settings config/round20_unseen_drug_closure_settings.json \
  --context-dim 16 \
  --predictor gated_pooled_fusion
```

### 14.5 Smoke assertions

```text
loss finite
parameter update occurred
frozen omics encoder gradients absent
D0 gradient policy equals E3
prediction shape correct
checkpoint strict reload succeeds
```

---

## 15. Full Run 操作順序

### 15.1 Stage 20A

```bash
bash scripts/run_round20_stage20a.sh
```

腳本應：

1. validate settings；
2. resolve E3；
3. audit context；
4. build splits；
5. build 30-job manifest；
6. execute/resume；
7. verify 30/30；
8. aggregate；
9. write dimension lock。

### 15.2 Stage 20B

```bash
bash scripts/run_round20_stage20b.sh
```

腳本應：

1. validate Stage 20A lock；
2. build E3 vs gated manifest；
3. execute 30 jobs；
4. verify paired completeness；
5. aggregate；
6. evaluate guardrails。

### 15.3 Stage 20C

```bash
bash scripts/run_round20_stage20c.sh
```

腳本應：

1. reject external fields；
2. write final lock；
3. strict model reconstruction smoke；
4. freeze lock hash。

### 15.4 Stage 20D

```bash
bash scripts/run_round20_stage20d.sh
```

腳本應：

1. require final lock；
2. resolve canonical deployment ensemble；
3. run all configured TCGA targets；
4. require complete folds；
5. export metrics／predictions／coverage；
6. never modify lock。

### 15.5 Stage 20E

```bash
bash scripts/run_round20_stage20e.sh
```

腳本應：

1. collect artifacts；
2. run frozen/raw equivalence；
3. freeze environment；
4. generate checksums；
5. strict release audit；
6. write final report。

---

## 16. Job Resume、OOM 與失敗處理

### 16.1 Status schema

```json
{
  "job_id": "...",
  "status": "pending|running|done|failed|oom_retry|invalid",
  "attempt": 1,
  "started_at": "...",
  "finished_at": "...",
  "exit_code": 0,
  "error_type": null,
  "error_message": null,
  "checkpoint_path": "...",
  "metrics_path": "..."
}
```

### 16.2 Resume

只跳過：

```text
status == done
且 metrics / predictions / checkpoint hashes 均有效
```

僅看到 checkpoint 不代表完成。

### 16.3 OOM

沿用 Round 18 OOM retry：

```text
512 → 256 → 128 → 64 → 32
```

保持 target effective batch，透過 gradient accumulation 補足。

### 16.4 不可自動修復

禁止自動：

```text
換 split seed
刪除困難 drug
降低 model size
改 optimizer
改 context dimension
```

任何研究契約變更需新 manifest version。

---

## 17. Analysis 與報告模板

### 17.1 Stage 20A 報告

```markdown
# Round 20A — Context Dimension Confirmation

## Objective
## Fixed components
## C16 / C32 artifact equivalence audit
## Repeated drug-held-out design
## Per-fold results
## Per-seed results
## Paired deltas
## Guardrail assessment
## Dimension lock
## Limitations
```

### 17.2 Stage 20B 報告

```markdown
# Round 20B — Pooled Predictor Confirmation

## Objective
## Locked context dimension
## Resolved E3 architecture
## Gated architecture
## Fairness controls
## Repeated CV results
## Gate diagnostics
## Guardrails
## Predictor decision
```

### 17.3 Final report

```markdown
# Round 20 Final — Unseen-Drug Closure

## Executive summary
## Historical rationale
## Locked O2 / D0 contract
## Dimension result
## Predictor result
## Stability result
## Locked TCGA evaluation
## Negative results
## Final recommended model
## Frozen deployment path
## End-to-end-capable path
## Reproducibility manifest
## Claims allowed / not allowed
## Project closure status
```

---

## 18. Claims 規範

### 18.1 可以寫

```text
Under repeated drug-held-out validation, the locked model was non-inferior
to the pooled baseline under the predefined stability guardrails.
```

或：

```text
Increasing projected prototype context from 16 to 32 dimensions did not
provide sufficient stable gain to justify the added representation size.
```

或：

```text
Gated pooled fusion did not consistently outperform the simpler pooled E3
baseline, supporting the parsimonious deployment choice.
```

### 18.2 不可以寫

```text
The model generalizes to all unseen drugs.
The TCGA result is untouched external validation.
C32 contains more biological information because its AUC is slightly higher.
Gating proves causal drug–omics interaction.
The final model is universally optimal.
```

---

## 19. IDE 分階段操作清單

### 19.1 開發者在 IDE 中先做

```text
[ ] 開啟 repository root
[ ] 建立 Round 20 branch
[ ] 搜尋 round19 role lock / deployment policy / E3
[ ] 搜尋 context16 / context32 artifact metadata
[ ] 搜尋 drug-held-out split builder
[ ] 搜尋 TCGA inference entry points
[ ] 搜尋 checkpoint state dict loader
[ ] 建立 TODO.md，不直接先改 train loop
```

### 19.2 第一個 commit

建議 commit：

```text
Add Round 20 schemas, E3 resolver, and preflight audits
```

內容：

```text
schema
resolver
context metadata audit
drug identity audit
unit tests
```

### 19.3 第二個 commit

```text
Add repeated drug-held-out dimension confirmation pipeline
```

### 19.4 第三個 commit

```text
Add gated pooled fusion and paired predictor confirmation
```

### 19.5 第四個 commit

```text
Add guardrail selection and immutable final lock
```

### 19.6 第五個 commit

```text
Add locked TCGA inference and closure release audit
```

---

## 20. 可直接貼入 IDE 的分階段 Prompt

### 20.1 Prompt A：Preflight

```text
請先不要修改訓練模型。掃描本機 repository 中 Round 19 的 role lock、deployment policy、
drug-held-out summary、candidate manifests 與 checkpoints，實作 fail-closed E3 resolver。
不得依 E3 名稱猜架構。列出來源、衝突與缺失欄位，新增 unit tests。
接著 audit C16/C32 的 fit population、raw context definition、normalization、feature order、
encoder checkpoint 與 ModelID coverage。若不一致，只報告，不自動重建。
```

### 20.2 Prompt B：Stage 20A

```text
使用既有 eligible response 與 drug identity mapping，建立 52/62/72 三個 split seeds、
每個 5 folds 的 drug-held-out assignments。確保 canonical drug 不跨 train/val。
在 fixed resolved E3 + D0 下，只比較 C16 與 C32，建立 30-job manifest。
C16/C32 除 omics input dim 外，所有訓練設定完全相同。新增 paired aggregate 與 dimension lock。
```

### 20.3 Prompt C：Stage 20B

```text
讀取 Stage 20A dimension lock。保留 resolved pooled E3，新增唯一一個 GatedPooledFusion。
使用相同 D0、split、seed、training budget，建立 E3 vs gated 的 30-job repeated CV manifest。
輸出 paired AUC/AUPRC delta、per-seed summary、gate saturation diagnostics。
不要加入 residual MLP、Transformer 或新的 GNN。
```

### 20.4 Prompt D：Selection

```text
依固定 guardrails選模。Selection API 只接受 development drug-held-out aggregate，
遇到 tcga/internal/external/posthoc 欄位必須 hard fail。
若 gated AUC delta < 0.005，依 parsimony 保留 E3，除非 variance與worst-seed有明確優勢。
產生 immutable final model lock、deployment contract與selection audit。
```

### 20.5 Prompt E：TCGA 與 release

```text
final lock 後，使用相同 model factory 與 forward contract 執行既有 TCGA response targets。
不要使用只支援 concat classifier 的假設來載入 gated model。
輸出 per-sample predictions、per-drug metrics、DrugMacro/Global metrics與coverage。
最後封存 encoder full weights、frozen latents、context projection、D0／predictor checkpoints、
splits、configs、environment、checksums，並測試 raw-omics frozen encoder path 與 precomputed Z path 等價。
```

---

## 21. 常見問題與排錯

### 21.1 E3 無法解析

**症狀**：role lock 只有 `E3`，沒有 predictor class。

**處理**：

1. 查 manifest row；
2. 查 checkpoint `config`；
3. 查 model state dict keys；
4. 查 stage19e report；
5. 仍無法唯一化則 fail；
6. 使用者已核准時，才啟用 reconstructed pooled MLP fallback。

### 21.2 C32 coverage 少於 C16

不得直接以 inner join 縮小 C16。

先修復 C32 artifact；若無法修復，Stage 20A 不成立。

### 21.3 C32 checkpoint input shape mismatch

只允許重新初始化 input adapter；核心 E3 hidden architecture 不變。
不得用 padding 16 個零假裝 C32。

### 21.4 Drug split leakage

先檢查：

```text
alias
salt form
stereochemistry
canonical SMILES
InChIKey
```

有疑義時合併為同一 group，並重新生成所有三個 seeds。

### 21.5 某 fold 無足夠有效 drug metric

該 split 設計 invalid，不可填 0.5。
必須重新設計 deterministic splitting constraints，並對 C16/C32 同時重跑。

### 21.6 Gated model collapse

若 gate 幾乎全 0 或 1：

- 先確認 normalization 與 scale；
- 確認 sigmoid 前 logits；
- 確認 omics/drug projection 梯度；
- 不立即加入更多 gate loss；
- 可使用核准 P1/P2 preset；
- 仍 collapse 則視為負面結果。

### 21.7 TCGA fold 缺 checkpoint

`require_complete_ensemble=true` 時整個 target fail，不得用 4/5 folds 並與歷史 5-fold比較。

### 21.8 Frozen/raw path 不一致

檢查：

```text
gene order
normalization
encoder eval mode
dropout
batch norm
float precision
sample key mapping
```

---

## 22. 最終驗收清單

### Stage 20-0

```text
[ ] E3 exact resolution 或明確 reconstructed
[ ] C16/C32 comparable
[ ] D0 training mode locked
[ ] drug identity audit passed
[ ] settings schema passed
```

### Stage 20A

```text
[ ] 30/30 jobs done
[ ] 3 seeds × 5 folds complete
[ ] no drug leakage
[ ] paired deltas complete
[ ] dimension decision locked
```

### Stage 20B

```text
[ ] E3 vs gated fair contract
[ ] 30/30 jobs done
[ ] gate diagnostics exported
[ ] no extra architecture search
```

### Stage 20C

```text
[ ] guardrails evaluated
[ ] external columns rejected
[ ] final model lock immutable
[ ] deployment contract written
```

### Stage 20D

```text
[ ] lock existed before inference
[ ] canonical 5-fold ensemble complete
[ ] all configured TCGA targets attempted
[ ] per-sample / per-drug / aggregate outputs exist
[ ] coverage and failed-drug reports exist
[ ] lock not changed after results
```

### Stage 20E

```text
[ ] encoder full weights archived
[ ] frozen latent artifacts archived
[ ] selected and comparison projections archived
[ ] D0/predictor checkpoints archived
[ ] raw/end-to-end-capable code path retained
[ ] frozen/raw equivalence passed
[ ] environment frozen
[ ] checksums passed
[ ] model card / dataset card / final report complete
```

---

## 23. 本輪最小總 job 數

```text
Stage 20A: 2 dimensions × 3 seeds × 5 folds = 30
Stage 20B: 2 predictors × 3 seeds × 5 folds = 30
-------------------------------------------------
正式最低訓練 jobs = 60
```

另加：

```text
smoke jobs
canonical deployment 5-fold checkpoints（若不能直接沿用）
TCGA inference jobs
optional gated P1/P2 presets（只有預先記錄 amendment 後）
```

---

## 24. 最終建議時間線（以 Stage 為單位，不是工時承諾）

```text
Stage 20-0：artifact audit 與 contracts
Stage 20A：C16 vs C32
Stage 20B：E3 vs gated
Stage 20C：selection lock
Stage 20D：TCGA inference
Stage 20E：release closure
```

任何 Stage 未通過其 completion gate，不進入下一 Stage。

---

## 25. 本輪最終決策樹

```text
C32 是否通過相對 C16 guardrails？
├─ 否 → 鎖定 C16
└─ 是
   ├─ mean delta < 0.005 → 鎖定 C16
   └─ mean delta >= 0.005 → 鎖定 C32

在 locked context 下，gated 是否通過 E3 guardrails？
├─ 否 → 鎖定 E3
└─ 是
   ├─ mean delta < 0.005 且無明確 stability 優勢 → 鎖定 E3
   └─ 有穩定且有意義增益 → 鎖定 gated

模型鎖定
→ canonical deployment ensemble
→ TCGA response inference
→ frozen / end-to-end-capable release
→ 專案收尾
```

---

## 26. 文件狀態

```text
Status: Implementation specification
Scope: Final controlled unseen-drug optimization and project closure
Selection target: repeated drug-held-out only
TCGA role: locked post-selection evaluation
Omics training: frozen in this Round
Future capability: end-to-end path retained, not claimed as tested here
```
