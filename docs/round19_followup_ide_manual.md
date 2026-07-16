# drug-DCF Round 19 後續 IDE 實作手冊

> **文件定位：** 本手冊提供給使用 VS Code、Cursor、PyCharm、GitHub Copilot 或其他 coding agent 的開發者，作為從公開 `main` 分支的 Round 18 基線，補齊、重建、稽核與封存 Round 19A–19F 的實作規格。
>
> **核心原則：** 不修改或覆寫 Round 18 的歷史結果；Round 19 以 adapter、registry、獨立 config、獨立 runner 與獨立 artifacts 實作，並透過 regression tests 保證舊流程可重現。

| 文件欄位 | 內容 |
|---|---|
| Repository | <https://github.com/JohnWCT/drug-DCF> |
| 建議文件路徑 | `docs/round19_followup_ide_manual.md` |
| 建議實作分支 | `feature/round19-factorial-and-role-policy` |
| 對應階段 | Round 19A–19F |
| 基準日期 | 2026-07-16 |
| 主要選模指標 | ModelID-grouped DrugMacro AUC；DrugMacro AUPRC 作 tie-breaker |
| 必須保留的基線 | Round 18 eligible population、internal split、frozen omics encoder、historical GIN32／MLP／cross-attention |
| 禁止用於 selection | Internal test、TCGA response、任何 post-hoc external score |

> **本機對齊狀態（2026-07-16）：** DAPL 工作區已以薄相容層對齊本手冊公開介面，**不重跑** 19A–19F 實驗。實作別名見下方「本機 adapter 對照」。

---

## 本機 adapter 對照（DAPL / round19_factorial）

| 手冊公開介面 | 本機實作 |
|---|---|
| `tools/round19_schema.py` | 已新增；selection 欄位 case-insensitive guard |
| `tools/round19_registry.py` | 已新增；以 factorial settings + `fusion_models.COMPATIBLE_CELLS` 為真相 |
| `tools/round19_role_lock.py` | 已新增；包裝 immutable `round19_stage19f_final_lock` |
| `tools/round19_selection.py` | 已新增；只驗證既有 lock，禁止 re-select |
| `tools/round19_release_audit.py` | 已新增；包裝 19H reproducibility + policy／leakage audit |
| `reports/round19_deployment_policy.json` | 由 `tools/round19_deployment_policy_export.py` 匯出；canonical 亦在 `result/.../reports/` |
| `reports/round19_final_role_lock.json` | repo `reports/` 僅為 pointer；immutable lock 仍在 `result/.../reports/` |
| `config/params_round19_*.json` | 已新增為 settings 投影，不改訓練路徑 |
| 13-cell matrix | **本機已執行** 含 `D4×P1`、不含 `D1×P2`；手冊理想為 `D4×P0-only` + 全 `D0–D3×P0–P2`。registry 同時記錄兩者，不改歷史 jobs |

稽核命令：

```bash
python3 tools/round19_deployment_policy_export.py --smoke-route
python3 tools/round19_release_audit.py \
  --role-lock result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json \
  --policy result/optimization_runs/round19_factorial/reports/round19_deployment_policy.json \
  --repository-attestation result/optimization_runs/round19_factorial/metadata/round19_stage19h_repository_attestation.json \
  --strict
pytest -q test_round19_public_reconstruction.py
```

---

## 0. 給 IDE Agent 的主指令

以下內容可直接貼入 Cursor／Copilot Chat，作為本輪的最高層實作指令：

```text
你正在修改 JohnWCT/drug-DCF 專案，目標是以公開 Round 18 程式為基礎，
建立可重現、可測試、可稽核的 Round 19A–19F pipeline。

硬性要求：
1. 不覆寫或改名任何 Round 18 artifact、config、report 或 result path。
2. 優先重用 Round 18 的 eligible data、CV splits、metrics、OOM retry、train loop；
   透過 Round 19 adapter 擴充，不在原模組加入大量 Round 19 特例。
3. Omics encoder 保持 frozen。不得在 Round 19 解凍或重新訓練。
4. Selection 程式禁止讀取 internal／TCGA 欄位與檔案；測試中必須驗證此規則。
5. 所有模型、資料、split、checkpoint、prediction、metric 與 selection 皆需可追溯到：
   config hash、manifest row、git SHA、environment、seed、fold、feature metadata。
6. 每完成一個 stage，先執行 unit tests、synthetic smoke、real-data smoke，
   再允許產生 full manifest。
7. 所有新增 Python API 要有 type hints、清楚錯誤訊息與 deterministic seed handling。
8. 不引入 GIN+MACCS、GINE+MACCS 或 MACCS residual hybrid。
9. 不把 cancer／drug／scaffold shift 分數平均成單一總分。
10. 不根據 attention 圖挑選模型。

實作順序：
A. schema／registry／config builder
B. omics feature adapter
C. drug encoder factory與cache
D. predictor factory與forward contract
E. dataset／collate／manifest／smoke
F. Stage 19B screening
G. Stage 19C controls
H. Stage 19D repeated grouped CV
I. Stage 19E domain shifts
J. Stage 19F role lock、deployment policy、release audit

每次修改後：
- 列出變更檔案；
- 說明 API contract；
- 提供新增測試；
- 執行最小測試命令；
- 不宣稱 full experiment 已完成，除非所有 manifest jobs 與 aggregate checks 都通過。
```

---

## 1. 本手冊要完成的最終狀態

完成後，第三者應能從一個乾淨環境依序執行：

```text
Round 19A：建立 feature／encoder／split／manifest 基建
Round 19B：執行 Omics × Drug × Predictor factorial screening
Round 19C：補 O0／O4 與 shuffled-context faithfulness control
Round 19D：以 3 個 split seed 執行 repeated grouped 5CV
Round 19E：執行 cancer-type／drug／scaffold held-out validation
Round 19F：產生 role lock、deployment policy、model cards 與 release manifest
```

最終至少應存在：

```text
config/round19_factorial_settings.json
config/params_round19_screening.json
config/params_round19_confirmation.json
config/params_round19_domain_shift.json

tools/round19_schema.py
tools/round19_registry.py
tools/round19_feature_adapter.py
tools/round19_drug_encoders.py
tools/round19_fusion_models.py
tools/round19_dataset.py
tools/round19_config_builder.py
tools/round19_splits.py
tools/round19_metrics.py
tools/round19_selection.py
tools/round19_role_lock.py
tools/round19_release_audit.py
tools/analyze_round19.py

step1_finetune_latent_pipeline_round19_cv.py

tests/test_round19_schema.py
tests/test_round19_registry.py
tests/test_round19_features.py
tests/test_round19_drug_encoders.py
tests/test_round19_fusion_models.py
tests/test_round19_dataset.py
tests/test_round19_config_builder.py
tests/test_round19_splits.py
tests/test_round19_pipeline_smoke.py
tests/test_round19_selection.py
tests/test_round19_role_lock.py
tests/test_round19_release_audit.py

docs/round19_stage19a_report.md
docs/round19_stage19b_report.md
docs/round19_stage19c_report.md
docs/round19_stage19d_report.md
docs/round19_stage19e_report.md
docs/round19_stage19f_report.md

reports/round19_final_role_lock.json
reports/round19_deployment_policy.json
reports/round19_release_manifest.json
```

---

## 2. 證據邊界與重建原則

### 2.1 公開 repository 可直接利用的 Round 18 元件

公開 `main` 已有下列 Round 18 模組，可作為 Round 19 底座：

- `tools/round18_config_builder.py`
- `tools/round18_cv_splits.py`
- `tools/round18_dataset.py`
- `tools/round18_eligible_data.py`
- `tools/round18_feature_coverage.py`
- `tools/round18_fusion_models.py`
- `tools/round18_cv_metrics.py`
- `tools/round18_oom_runner.py`
- `tools/round18_prediction_ensemble.py`
- `tools/round18_response_head.py`
- `step1_finetune_latent_pipeline_round18_cv.py`
- `tests/test_round18_*.py`

### 2.2 不應直接做的修改

禁止：

1. 將 `round18_*` 檔案直接改名成 `round19_*`。
2. 在 Round 18 settings 中加入 O0–O4、D0–D4、P0–P2 後覆蓋歷史 config。
3. 讓 Round 18 pipeline 根據 Round 19 manifest 自動切換行為。
4. 改變 Round 18 checkpoint state dict key。
5. 以新的 graph preprocessing 重新計算 Round 18 正式結果。
6. 讓 TCGA 或 internal score 出現在 19B–19F selection input。

### 2.3 建議策略

採取以下相容層：

```text
Round 18 historical modules
        ↓ import／wrapper
Round 19 adapters and factories
        ↓ stable contracts
Round 19 stage runners
        ↓ immutable artifacts
Round 19 analysis／role lock
```

只有在共用 bug 已被證明同時影響 Round 18 與 Round 19 時，才修改共用模組；該修正必須：

- 有 regression test；
- 說明是否需要重跑 Round 18；
- 不靜默改變既有結果；
- 以 changelog 記錄舊／新行為。

---

## 3. 不可變研究契約

### 3.1 Omics 契約

| ID | canonical name | 組成 | 建議維度 | 用途 |
|---|---|---|---:|---|
| O0 | `z_only` | frozen Z | 64 | latent baseline |
| O1 | `z_plus_summary` | Z + own-summary | 75 | summary baseline |
| O2 | `z_plus_context16` | Z + projected context16 | 80 | parsimonious context |
| O3 | `z_plus_summary_context16` | Z + summary + context16 | 91 | full context comparison |
| O4 | `z_plus_source_proto_features` | Z + source-only prototype features | implementation-defined | inductive／source-only control |

硬性規則：

- O2 與 O3 必須使用**相同已 fit projection artifact**。
- 不得為 O2 重新 fit PCA／projection。
- O4 不得讀取 target response label。
- feature metadata 必須包含 component slice 與 artifact hash。
- omics encoder checkpoint 必須相同且 frozen。

### 3.2 Drug representation 契約

| ID | encoder | node dim | graph dim | edge／bond aware | predictor compatibility |
|---|---|---:|---:|---|---|
| D0 | GIN baseline | 32 | 32 | 否 | P0、P1、P2 |
| D1 | GIN wider-node | 64 | 32 | 否 | P0、P1、P2 |
| D2 | GIN wider-node/graph | 64 | 64 | 否 | P0、P1、P2 |
| D3 | GINE | 64 | 64 | 是 | P0、P1、P2 |
| D4 | MACCS-only | N/A | fixed projection | N/A | P0 only |

D4 僅與 P0 相容，使 screening 維持 13 個 Drug × Predictor cells：

```text
4 graph encoders × 3 predictors + 1 fingerprint baseline = 13 cells
```

禁止建立：

```text
GIN + MACCS
GINE + MACCS
MACCS residual
MACCS as extra graph token
```

### 3.3 Predictor 契約

| ID | canonical name | input | 角色 |
|---|---|---|---|
| P0 | `pooled_mlp` | omics vector + pooled drug vector | historical／efficient baseline |
| P1 | `pooled_transformer` | omics token + pooled drug token | pooled interaction comparator |
| P2 | `atom_cross_attention` | omics query + atom K/V | atom-level interaction |

P2 的核心 contract：

```text
Q = projected omics representation
K,V = valid atom node embeddings
mask = atom padding mask
output = attended omics-conditioned drug representation
```

不得在 P2 預設加入 pooled residual。若保留 residual ablation，應使用獨立 predictor ID，例如 `P2R`，不得與 P2 混寫。

---

## 4. 建議目錄與 artifact 規格

```text
result/optimization_runs/round19/
├── stage19a/
│   ├── data/
│   ├── features/
│   ├── graph_cache/
│   ├── fingerprint_cache/
│   ├── splits/
│   ├── manifests/
│   └── smoke/
├── stage19b/
│   ├── jobs/<job_id>/
│   ├── aggregate/
│   └── reports/
├── stage19c/
│   ├── jobs/<job_id>/
│   ├── controls/
│   ├── aggregate/
│   └── reports/
├── stage19d/
│   ├── seed_52/
│   ├── seed_62/
│   ├── seed_72/
│   ├── aggregate/
│   └── reports/
├── stage19e/
│   ├── cancer_type_held_out/
│   ├── drug_held_out/
│   ├── scaffold_held_out/
│   ├── aggregate/
│   └── reports/
└── stage19f/
    ├── role_lock/
    ├── deployment/
    ├── model_cards/
    ├── posthoc/
    └── release/
```

每個 job directory 至少包含：

```text
resolved_config.json
run_metadata.json
checkpoint.pt
best_epoch.json
train_history.csv
val_predictions.csv
val_metrics.json
runtime_resource_summary.json
stdout.log
stderr.log
DONE.json 或 FAILED.json
```

### 4.1 `run_metadata.json` 必要欄位

```json
{
  "round": "19",
  "stage": "19b",
  "job_id": "19b__D0__P2__O2__seed101__fold0",
  "git_sha": "<sha>",
  "config_sha256": "<sha256>",
  "manifest_sha256": "<sha256>",
  "feature_metadata_sha256": "<sha256>",
  "split_sha256": "<sha256>",
  "model_seed": 101,
  "split_seed": 42,
  "fold_id": 0,
  "omics_id": "O2",
  "drug_id": "D0",
  "predictor_id": "P2",
  "device": "cuda:0",
  "started_at_utc": "...",
  "finished_at_utc": "..."
}
```

---

## 5. 建議新增的核心 Schema

建立 `tools/round19_schema.py`，集中所有 ID、enum、manifest 欄位與 validation。

### 5.1 建議資料類別

```python
from dataclasses import dataclass
from typing import Literal, Optional

OmicsID = Literal["O0", "O1", "O2", "O3", "O4"]
DrugID = Literal["D0", "D1", "D2", "D3", "D4"]
PredictorID = Literal["P0", "P1", "P2"]
ShiftID = Literal[
    "modelid_grouped",
    "cancer_type_held_out",
    "drug_held_out",
    "scaffold_held_out",
]

@dataclass(frozen=True)
class Round19ModelSpec:
    omics_id: OmicsID
    drug_id: DrugID
    predictor_id: PredictorID
    residual_mode: str = "pure"

@dataclass(frozen=True)
class Round19JobSpec:
    stage: str
    model: Round19ModelSpec
    fold_id: int
    model_seed: int
    split_seed: int
    shift_id: ShiftID = "modelid_grouped"
    control_id: Optional[str] = None
```

### 5.2 Validation 函式

至少實作：

```python
validate_model_spec(spec)
validate_job_spec(job)
validate_manifest_columns(df)
validate_selection_input_columns(df)
canonical_model_id(spec)
canonical_job_id(job)
```

`validate_selection_input_columns` 必須拒絕：

```text
internal
internal_test
TCGA
tcga
external
Integrated5
posthoc
```

建議拒絕邏輯對欄名使用 case-insensitive substring matching，並在錯誤訊息列出違規欄位。

---

## 6. Registry 與 Compatibility Matrix

建立 `tools/round19_registry.py`，不得在 runner 內散落大量 `if D0 ... elif D1 ...`。

### 6.1 Omics registry

```python
OMICS_REGISTRY = {
    "O0": {"name": "z_only", "expected_dim": 64},
    "O1": {"name": "z_plus_summary", "expected_dim": 75},
    "O2": {"name": "z_plus_context16", "expected_dim": 80},
    "O3": {"name": "z_plus_summary_context16", "expected_dim": 91},
    "O4": {"name": "z_plus_source_proto_features", "expected_dim": None},
}
```

### 6.2 Drug registry

```python
DRUG_REGISTRY = {
    "D0": {"family": "gin", "node_dim": 32, "graph_dim": 32, "bond_aware": False},
    "D1": {"family": "gin", "node_dim": 64, "graph_dim": 32, "bond_aware": False},
    "D2": {"family": "gin", "node_dim": 64, "graph_dim": 64, "bond_aware": False},
    "D3": {"family": "gine", "node_dim": 64, "graph_dim": 64, "bond_aware": True},
    "D4": {"family": "maccs", "node_dim": None, "graph_dim": 167, "bond_aware": False},
}
```

> MACCS 實際 bit 數與 RDKit representation 必須由程式讀取並記錄，不應只依人工記憶。若移除固定常數，`graph_dim` 可由 cache metadata 提供。

### 6.3 Predictor registry

```python
PREDICTOR_REGISTRY = {
    "P0": {"family": "pooled_mlp", "requires_nodes": False},
    "P1": {"family": "pooled_transformer", "requires_nodes": False},
    "P2": {"family": "atom_cross_attention", "requires_nodes": True},
}
```

### 6.4 Compatibility

```python
COMPATIBLE_CELLS = [
    (drug_id, predictor_id)
    for drug_id in ("D0", "D1", "D2", "D3")
    for predictor_id in ("P0", "P1", "P2")
] + [("D4", "P0")]
```

測試必須確認：

- 共 13 cells；
- D4×P1、D4×P2 會被拒絕；
- 所有 P2 cell 都能取得 atom node embeddings；
- 所有 P0/P1 cell 都能取得 fixed-shape pooled embedding。

---

# Part I — Round 19A：Factorial 基建

## 7. Round 19A 完成定義

Round 19A 只建立與驗證基建，不進行大規模 model selection。完成定義：

- O0–O4 feature directories 可被一致載入；
- O2／O3 projection artifact hash 一致；
- D0–D4 forward contracts 通過；
- D3 edge attributes 完整；
- graph／fingerprint cache 可重用且有 metadata；
- Round 18 eligible population 可無損複用；
- ModelID split 與 internal test 完全一致；
- drug／scaffold／cancer-type shift split 可產生；
- 19B manifest 恰為 117 jobs；
- selection lock 能拒絕 internal／TCGA 欄位；
- synthetic smoke 與真實 eligible CUDA smoke 通過。

---

## 8. Omics feature adapter 修改建議

新增 `tools/round19_feature_adapter.py`，不要讓 Round 19 dataset 直接理解所有歷史 feature folder 命名。

### 8.1 API

```python
class Round19FeatureStore:
    def __init__(self, feature_root: str, model_key: str): ...

    def resolve(self, omics_id: str) -> Path: ...

    def load_metadata(self, omics_id: str) -> dict: ...

    def load_vectors(self, omics_id: str) -> dict[str, np.ndarray]: ...

    def validate_cross_mode_coverage(self, omics_ids: list[str]) -> dict: ...

    def validate_projection_lineage(self) -> dict: ...
```

### 8.2 Feature metadata 建議格式

```json
{
  "feature_id": "O2",
  "canonical_name": "z_plus_context16",
  "response_input_dim": 80,
  "components": [
    {"name": "z", "slice": [0, 64]},
    {"name": "context16", "slice": [64, 80]}
  ],
  "encoder_checkpoint": "...",
  "encoder_checkpoint_sha256": "...",
  "projection_artifact": "...",
  "projection_artifact_sha256": "...",
  "sample_count": 937,
  "model_ids_sha256": "...",
  "created_by": "...",
  "git_sha": "..."
}
```

### 8.3 O2 建構規則

若現有 feature exporter 只會輸出 O3，請新增 deterministic slicing exporter：

```text
O3 = [Z | summary | context16]
O2 = [Z | context16]
```

但必須依 metadata slice，不可硬寫 `[:, :64]` 與 `[:, -16:]` 而不檢查欄位。

### 8.4 O4 建構規則

O4 只使用 source-domain 可取得的 prototype features。實作時需記錄：

- prototype 的訓練資料範圍；
- 是否使用 target unlabeled samples；
- projection fit population；
- 每個 sample 的 mapping 規則；
- 缺少 cancer type／prototype 時的 fallback。

建議 fallback：

```text
missing source prototype
→ fail closed in training manifest generation
→ inference 時才允許明確的 global-source fallback
```

不要在訓練資料缺值時靜默填零。

### 8.5 Shuffled context

建立：

```python
build_modelid_level_context_permutation(
    model_ids: Sequence[str],
    seed: int,
    stratify_columns: Optional[list[str]] = None,
) -> dict[str, str]
```

必要條件：

- permutation 單位是 ModelID；
- 同一 ModelID 的所有 drug rows 使用相同 shuffled context；
- 固定 seed 可重現；
- 無 fixed point 或記錄 fixed point 數；
- 產生 mapping CSV 與 hash。

---

## 9. Drug encoder 修改建議

### 9.1 不建議直接塞入 `drugmodels/ginconv.py` 的內容

避免把 D0–D4 全部塞進單一歷史 class。建議：

```text
drugmodels/ginconv.py        保留 historical GIN
新增 drugmodels/gineconv.py  實作 bond-aware GINE
新增 tools/round19_drug_encoders.py 作統一 factory／adapter
```

### 9.2 統一輸出 contract

```python
@dataclass
class DrugEncoderOutput:
    node_embeddings: Optional[torch.Tensor]
    graph_embeddings: torch.Tensor
    node_batch: Optional[torch.Tensor]
    node_mask: Optional[torch.Tensor]
    metadata: dict[str, Any]
```

所有 graph encoder 必須輸出：

- `node_embeddings`: `[num_nodes, node_dim]`；
- `graph_embeddings`: `[batch_size, graph_dim]`；
- `node_batch`: 每個 node 的 graph index；
- `metadata`: node_dim、graph_dim、pooling、layers、JK、bond-aware。

MACCS：

- `node_embeddings=None`；
- `graph_embeddings=[batch, fp_dim]`；
- 只允許 P0。

### 9.3 D0 historical parity test

新增測試，確認在固定 input graph 與固定 state dict 下：

```text
Round19 D0 adapter output == historical GIN output
```

容許誤差建議：

```python
rtol=1e-6
atol=1e-7
```

### 9.4 D1／D2

D1 與 D2 的差異必須只在 pooled graph bottleneck：

```text
D1：node_dim 64 → graph projection 32
D2：node_dim 64 → graph projection 64
```

避免同時改變：

- layer 數；
- dropout；
- JK；
- pooling；
- atom features；
- optimizer。

否則無法歸因 graph bottleneck。

### 9.5 D3 GINE

D3 必須使用 bond／edge attributes。需要修改或新增：

- `tools/graph_utils.py`：產生 `edge_attr`；
- `tools/round19_drug_encoders.py`：驗證 `edge_attr` shape；
- graph cache metadata：記錄 RDKit version、atom schema、bond schema；
- collate：確保 PyG Batch 保留 edge attributes。

edge schema 至少考慮：

```text
bond type
conjugated
in ring
stereo
```

若使用 one-hot，metadata 必須記錄每一段 slice。

### 9.6 D4 MACCS

新增 fingerprint cache：

```text
result/optimization_runs/round19/stage19a/fingerprint_cache/maccs.npz
result/optimization_runs/round19/stage19a/fingerprint_cache/metadata.json
```

metadata 至少包含：

- canonical SMILES hash；
- RDKit version；
- fingerprint type；
- bit dimension；
- failed molecules；
- duplicate drug mapping；
- cache SHA256。

建議對 bit vector 使用 float32，不要先做 PCA；若要 projection，projection 必須在 model 內 end-to-end 訓練並記錄 input dim。

---

## 10. Fusion／Predictor 修改建議

新增 `tools/round19_fusion_models.py`，但可重用 Round 18 已驗證的 modules。

### 10.1 Factory

```python
def build_round19_predictor(
    *,
    predictor_id: str,
    omics_dim: int,
    drug_node_dim: int | None,
    drug_graph_dim: int,
    config: dict,
) -> nn.Module:
    ...
```

### 10.2 Forward contract

```python
class Round19Predictor(nn.Module):
    def forward(
        self,
        *,
        omics: torch.Tensor,
        drug: DrugEncoderOutput,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor | dict]:
        ...
```

回傳：

```python
{
    "logits": logits,
    "aux": {
        "attention_weights": optional,
        "pooled_drug": optional,
        "attended_drug": optional,
    },
}
```

訓練預設 `return_aux=False`，避免儲存大量 attention。

### 10.3 P0

建議結構：

```text
omics_norm → omics_projection
pooled_drug_norm → drug_projection
concat → MLP response head
```

歷史 anchor 必須有一個 config 盡量完全重現 Round 18 MLP。

### 10.4 P1

只使用兩個 pooled tokens：

```text
[omics token, drug token]
→ positional/type embedding
→ compact Transformer encoder
→ response head
```

不要因 node_dim 增加而自動增加 transformer depth。

### 10.5 P2

必須檢查：

- node padding mask 方向正確；
- query shape `[batch, 1, d_model]`；
- K/V shape `[batch, max_atoms, d_model]`；
- padded atoms 不取得 attention mass；
- 每個樣本 attention sum 近似 1；
- 單 atom molecule 不出現 NaN；
- batch 中不同 atom 數可混合。

### 10.6 禁止 shortcut

P2 預設輸出不得 concat pooled GIN embedding。若需要重跑 residual control，必須另列 manifest：

```text
predictor_id=P2R
residual_mode=pooled_residual
```

並且不與 P2 headline result 混為同一模型。

---

## 11. Dataset／Collate 修改建議

新增 `tools/round19_dataset.py`，可包裝 `Round18ResponseDataset`，但不要讓 dataset 根據 model ID 建立模型。

### 11.1 Batch contract

```python
@dataclass
class Round19Batch:
    row_id: torch.Tensor
    model_id: list[str]
    drug_name: list[str]
    label: torch.Tensor
    omics: torch.Tensor
    graph: Optional[Batch]
    fingerprint: Optional[torch.Tensor]
    cancer_type: Optional[list[str]]
    scaffold_id: Optional[list[str]]
```

### 11.2 Dataset mode

```python
Round19ResponseDataset(
    response_df,
    omics_store,
    omics_id,
    drug_id,
    graph_cache,
    fingerprint_cache,
)
```

資料層只負責資料，不判斷 P0/P1/P2。

### 11.3 必要 assertions

- response row count 與 assignment merge 後不改變；
- `_row_id` 唯一；
- ModelID coverage 100%；
- drug coverage 100%，或 manifest 建立時 fail；
- label 只為 0/1；
- graph node count > 0；
- D3 graph 具有 edge_attr；
- D4 不載入 graph；
- O2/O3 context coverage 相同；
- shuffled control 只替換 context component，不替換 Z。

---

## 12. Split 系統修改建議

新增 `tools/round19_splits.py`，重用 Round 18 ModelID grouped split，但將 shift 規則獨立。

### 12.1 ModelID grouped

Round 19B 應複用 Round 18 split seed 42 與 internal test，不重新抽樣。

輸出 metadata：

```json
{
  "split_type": "modelid_grouped",
  "split_seed": 42,
  "reused_from_round18": true,
  "source_assignment": "...",
  "source_assignment_sha256": "..."
}
```

### 12.2 Repeated grouped 5CV

Round 19D 使用：

```text
split seeds = 52, 62, 72
folds = 5
```

每個 seed 都需檢查：

- 每個 ModelID 只出現在單一 fold 的 validation；
- 所有 development ModelID 恰好被驗證一次；
- internal test 永不進入 train/val；
- label balance 與 drug coverage 有 summary。

### 12.3 Drug-held-out

group key 應使用 canonical drug identity，而不是原始名稱字串。

建議新增 canonicalization table：

```text
raw drug name
canonical drug ID
canonical SMILES
InChIKey（可用時）
```

### 12.4 Scaffold-held-out

建議使用 Bemis–Murcko scaffold，並記錄：

- RDKit version；
- salt stripping／canonicalization；
- acyclic molecules policy；
- empty scaffold policy；
- scaffold group size distribution。

### 12.5 Cancer-type-held-out

以 ModelID 所屬 cancer type 作 group，禁止同一 ModelID 跨 cancer type。若資料有多標籤或 unknown：

- 先產生 QC report；
- unknown 不得靜默混入其他類別；
- 少樣本 cancer type 的合併規則必須在 config 中明示。

### 12.6 Leakage tests

每種 split 都需產生：

```text
split_leakage_audit.json
split_group_counts.csv
split_label_balance.csv
split_drug_coverage.csv
```

若發現 group overlap，builder 立即退出非零狀態。

---

## 13. Config Builder 修改建議

建立 `tools/round19_config_builder.py`，CLI 建議：

```bash
python -m tools.round19_config_builder \
  --settings config/round19_factorial_settings.json \
  --stage 19a \
  --outdir result/optimization_runs/round19/stage19a
```

支援 stages：

```text
19a
19b
19c
19d
19e
19f
all_manifests
```

### 13.1 Builder 責任

Builder 只做：

- resolve settings；
- validate paths／metadata；
- 建立 split；
- 建立 manifest；
- 建立 immutable resolved config；
- 計算 job IDs 與 hashes。

Builder 不做：

- 訓練；
- 選模；
- 讀 external score；
- 自動修改 settings；
- 根據已有結果刪除不喜歡的 jobs。

### 13.2 Manifest 欄位

```text
job_id
stage
omics_id
drug_id
predictor_id
control_id
shift_id
fold_id
model_seed
split_seed
response_path
feature_dir
graph_cache_path
fingerprint_cache_path
split_assignment
result_dir
settings_path
resolved_config_path
status
```

### 13.3 Idempotency

重跑 builder 時：

- 同 settings 產生同 manifest hash；
- 若輸出已存在且內容不同，fail；
- 支援 `--force-new-outdir`，但不覆寫；
- manifest row order 固定排序。

---

## 14. Round 19A Smoke 與 Gate

### 14.1 Synthetic smoke

覆蓋 13 cells × 至少一個 O mode，但不需完整組合。建議 smoke set：

```text
D0×P0×O1
D0×P1×O2
D0×P2×O3
D1×P2×O2
D2×P0×O3
D3×P2×O4
D4×P0×O1
```

每個執行 2–3 optimization steps，檢查：

- loss finite；
- 至少一個 encoder parameter 有 gradient；
- logits shape 正確；
- checkpoint save/load parity；
- AMP on/off 可執行。

### 14.2 Real eligible data smoke

每個 encoder family 至少跑：

- 2 train batches；
- 1 validation batch；
- 一次 backward；
- checkpoint round trip；
- metric export。

### 14.3 Stage 19A Gate

必須全部通過：

```text
pytest tests/test_round19_schema.py
pytest tests/test_round19_registry.py
pytest tests/test_round19_features.py
pytest tests/test_round19_drug_encoders.py
pytest tests/test_round19_fusion_models.py
pytest tests/test_round19_dataset.py
pytest tests/test_round19_splits.py
pytest tests/test_round19_config_builder.py
pytest tests/test_round19_pipeline_smoke.py
```

並確認：

```text
19B manifest rows == 117
duplicate job_id == 0
invalid compatibility == 0
missing feature coverage == 0
split leakage == 0
```

---

# Part II — Round 19B：Factorial Screening

## 15. Round 19B 設計

```text
13 Drug × Predictor cells
× 3 omics modes（O1、O2、O3）
× 3 folds
= 117 jobs
```

### 15.1 不變項

- eligible population；
- split seed 42；
- folds；
- model seed 101；
- optimizer family；
- epoch cap；
- early stopping policy；
- label definition；
- primary metric implementation。

### 15.2 可變項

僅：

- O1/O2/O3；
- D0–D4；
- P0–P2 compatible cell。

### 15.3 Job ID

```text
19b__D0__P2__O2__split42__seed101__fold0
```

### 15.4 Runner 修改建議

新增 `step1_finetune_latent_pipeline_round19_cv.py`，可重用 Round 18：

- OOM retry；
- seed；
- optimizer group；
- evaluation／prediction export；
- checkpoint lifecycle。

但需新增 CLI：

```text
--omics-id
--drug-id
--predictor-id
--shift-id
--control-id
--resolved-config
--manifest-row-json
```

避免保留模糊的：

```text
--architecture-family
--transformer-config-id
--omics-mode
```

Round 19 應以 canonical IDs 作唯一輸入。

### 15.5 結果 aggregation

新增 `tools/analyze_round19.py --stage 19b`，輸出：

```text
round19b_per_fold_metrics.csv
round19b_cell_omics_summary.csv
round19b_architecture_ranking.csv
round19b_omics_effects.csv
round19b_predictor_effects.csv
round19b_missing_jobs.csv
round19b_failure_summary.csv
```

### 15.6 Pairing 規則

計算 O2−O1、O3−O2 時必須 pair by：

```text
drug_id
predictor_id
fold_id
model_seed
split_seed
```

若任一 pair 缺 job，不可用不成對 mean 代替；應輸出 incomplete pair report。

### 15.7 19B selection gate

不直接鎖最終模型，只選進 19C 的 cells。建議規則：

1. 依 O2/O3 中最佳 mean DrugMacro AUC 排名；
2. 保留 P0 historical anchor；
3. 保留最佳 pooled comparator；
4. 保留最佳 D3 P2；
5. 保留至少一個 D1/D2 capacity comparator；
6. 總數固定 7 cells；
7. selection input 不含 internal／TCGA。

若第 7 名有 tie，使用：

```text
DrugMacro AUPRC
→ AUC std 較低
→ 參數較少
→ canonical ID lexicographic
```

---

# Part III — Round 19C：Omics Completion 與 Context Faithfulness

## 16. Round 19C 設計

```text
7 selected cells × O0/O4 × 3 folds = 42 jobs
+ 12 shuffled-context controls
= 54 jobs
```

### 16.1 Selected cells 的來源

必須由 `round19b_selected_cells.json` 產生，不手動複製到新 config。

格式：

```json
{
  "selection_stage": "19b",
  "selection_input_sha256": "...",
  "rules_version": "1.0",
  "selected_cells": [
    {"drug_id": "D0", "predictor_id": "P0", "reason": "historical_anchor"},
    {"drug_id": "D0", "predictor_id": "P2", "reason": "top_atom"}
  ]
}
```

### 16.2 Shuffled controls

建議選 4 個 context-sensitive candidates × 3 folds = 12 controls。

control ID：

```text
context_shuffle_modelid_seed1901
```

true/shuffled pair 必須共用：

- train/val split；
- model seed；
- drug graph；
- optimizer；
- epoch policy；
- O2 或 O3 的 Z component。

唯一差異是 context mapping。

### 16.3 必要輸出

```text
round19c_o2_vs_o0_pairs.csv
round19c_o3_vs_o2_pairs.csv
round19c_o4_gap.csv
round19c_true_vs_shuffled_pairs.csv
round19c_control_integrity.json
round19c_selected_for_19d.json
```

### 16.4 Faithfulness integrity tests

- shuffled mapping 與 true mapping 不同；
- 每個 ModelID mapping 固定；
- Z vector 完全相同；
- label、drug、split 完全相同；
- projection artifact 相同；
- context norm distribution 有比較報告；
- true/shuffled 的 row order 不影響結果。

### 16.5 19C → 19D 候選鎖定

固定 6 candidates：

```text
F0 historical MLP anchor
F1 parsimonious atom context（優先 O2）
F2 full atom context（O3）
F3 best pooled comparator
F4 source-only O4 candidate
F5 optional efficient／capacity comparator
```

F5 可以為 null；但若 19D manifest 規定 6 candidates，需以明確的候選填入，不得用 duplicate model 湊數。

---

# Part IV — Round 19D：Repeated Grouped 5CV

## 17. Round 19D 設計

```text
6 candidates
× split seeds 52、62、72
× 5 folds
= 90 jobs
```

model seed 是否固定 101：建議固定，以隔離 split variability。若要評估 model-seed variability，應另立附加實驗，不混入 19D headline。

### 17.1 Manifest

```text
19d__F1__D0__P2__O2__split52__seed101__fold0
```

### 17.2 Aggregate 層級

必須同時輸出：

1. per-fold；
2. per-split-seed mean；
3. mean-of-means；
4. pooled out-of-fold metrics；
5. paired delta；
6. rank stability；
7. failure／missingness。

### 17.3 不可混用的統計量

不得將：

```text
所有 15 folds 的 row-level predictions直接混成一個 AUC
```

當作唯一 headline。至少要保留 split-seed 層級，以避免某一 seed row count 影響權重。

### 17.4 Candidate comparison

建議輸出：

```text
F1-F0
F1-F3
F2-F1
F4-F1
```

每個 delta 包含：

- mean；
- median；
- std；
- 15 paired folds positive count；
- seed-level positive count；
- worst-seed delta。

### 17.5 19D Gate

- 90/90 jobs 完成，或完整說明缺失；
- 不允許 silent retry 改變 seed；
- OOM retry 只能調整 micro batch／accumulation，不改 optimizer effective batch；
- F0 historical anchor 表現不可出現無解釋的大幅 drift；
- O2/O3 feature hash 與 19B/19C 一致。

---

# Part V — Round 19E：Domain-shift Validation

## 18. Round 19E 設計

```text
6 candidates × 5 folds × 3 shifts = 90 jobs
```

三種 shift：

```text
cancer_type_held_out
drug_held_out
scaffold_held_out
```

### 18.1 不允許的總分

不得：

```text
mean(cancer AUC, drug AUC, scaffold AUC)
```

原因是三者回答不同泛化問題。

### 18.2 每種 shift 分開 selection

輸出：

```text
round19e_cancer_shift_summary.csv
round19e_drug_shift_summary.csv
round19e_scaffold_shift_summary.csv
round19e_shift_guardrails.csv
round19e_calibration_summary.csv
round19e_failure_modes.csv
```

### 18.3 Guardrail 建議

在 `config/params_round19_domain_shift.json` 中明確配置，不要硬寫於 selection code：

```json
{
  "major_fail": {
    "absolute_auc_drop": 0.03,
    "relative_to_anchor_auc_drop": 0.02,
    "min_valid_groups": 3
  },
  "non_worse_margin": 0.005,
  "calibration": {
    "max_brier_increase": 0.02,
    "max_ece_increase": 0.03
  }
}
```

以上是建議預設值，不是已確認的歷史 threshold。正式使用前應以既有 Round 19E artifact 或 protocol 核對。

### 18.4 Cancer specialist selection

建議 lexicographic：

1. 通過 cancer shift MAJOR_FAIL guardrail；
2. cancer-type macro DrugMacro AUC；
3. cancer-type macro AUPRC；
4. worst-cancer performance；
5. calibration；
6. complexity。

### 18.5 Chemical specialist selection

不要把 cancer shift 放入 chemical specialist。建議：

1. drug 與 scaffold shift 均不得 MAJOR_FAIL；
2. 最大化 `min(relative_delta_drug, relative_delta_scaffold)`；
3. tie 時最大化兩個 chemical shift 的 mean delta；
4. 再以 calibration、variance、complexity 決勝。

這是 minimax 思路，可避免只在 drug-held-out 強、但 scaffold-held-out 崩潰。

### 18.6 Failure label

每個 candidate × shift 可為：

```text
PASS
NON_WORSE
MINOR_FAIL
MAJOR_FAIL
INSUFFICIENT_GROUPS
INCOMPLETE
```

任何 `MAJOR_FAIL` 必須進入 final role lock 的 `known_failures`。

---

# Part VI — Round 19F：Final Role Lock 與 Deployment Policy

## 19. Round 19F 的核心目標

Round 19F 不再訓練新 gating model；只以已鎖定 evidence 建立 deterministic policy。

### 19.1 角色

```text
historical_anchor
source_performance_champion
parsimonious_context_model
cancer_shift_specialist
chemical_shift_specialist
source_only_domain_candidate
efficient_model
general_recommended_model（可為 null）
```

### 19.2 Selection 資料白名單

建立白名單，不使用黑名單為主：

```text
19B source grouped CV
19C controls
19D repeated grouped CV
19E held-out shift metrics
runtime／parameter count
```

任何其他資料來源預設拒絕。

### 19.3 `round19_final_role_lock.json` 建議 schema

```json
{
  "schema_version": "1.0",
  "created_at_utc": "...",
  "git_sha": "...",
  "evidence": {
    "stage19b_sha256": "...",
    "stage19c_sha256": "...",
    "stage19d_sha256": "...",
    "stage19e_sha256": "..."
  },
  "roles": {
    "historical_anchor": {
      "candidate_id": "F0",
      "model_spec": {"omics_id": "O1", "drug_id": "D0", "predictor_id": "P0"},
      "checkpoint_set": ["..."],
      "selection_reason": "historical parity"
    },
    "source_performance_champion": null,
    "parsimonious_context_model": null,
    "cancer_shift_specialist": null,
    "chemical_shift_specialist": null,
    "source_only_domain_candidate": null,
    "efficient_model": null,
    "general_recommended_model": null
  },
  "known_failures": [],
  "selection_policy_version": "1.0",
  "selection_input_columns": [],
  "selection_input_sha256": "..."
}
```

### 19.4 General model 允許 null

`general_recommended_model` 應為 null 的條件：

- 無模型同時通過 source、cancer、drug、scaffold guardrails；
- 最佳模型角色高度互斥；
- general model 會隱藏明確的 chemical failure；
- evidence 不完整。

不得為了文件完整而強迫填入模型。

### 19.5 Deployment policy schema

```json
{
  "schema_version": "1.0",
  "policy_type": "deterministic_metadata_routing",
  "rules": [
    {
      "priority": 10,
      "when": {"drug_seen_in_training": false},
      "route_to_role": "chemical_shift_specialist"
    },
    {
      "priority": 20,
      "when": {"scaffold_seen_in_training": false},
      "route_to_role": "chemical_shift_specialist"
    },
    {
      "priority": 30,
      "when": {
        "drug_seen_in_training": true,
        "scaffold_seen_in_training": true,
        "cancer_type_seen_in_training": false
      },
      "route_to_role": "cancer_shift_specialist"
    },
    {
      "priority": 40,
      "when": {"source_like": true},
      "route_to_role": "parsimonious_context_model"
    }
  ],
  "fallback_role": "historical_anchor",
  "reject_conditions": [],
  "required_metadata": [
    "canonical_drug_id",
    "canonical_scaffold_id",
    "cancer_type"
  ]
}
```

### 19.6 Routing precedence

建議：

```text
unseen drug
→ chemical specialist

seen drug but unseen scaffold
→ chemical specialist

seen drug/scaffold but unseen cancer type
→ cancer specialist

source-like
→ parsimonious context 或 source champion

metadata incomplete
→ fallback／reject；不可猜測
```

若 drug 與 cancer 同時 unseen，chemical shift 優先，因 drug-held-out failure 已被觀察為較嚴重風險。

### 19.7 Router 實作

新增：

```python
class Round19DeploymentRouter:
    def __init__(self, role_lock: dict, policy: dict): ...

    def route(self, metadata: dict) -> RouteDecision: ...
```

`RouteDecision`：

```python
@dataclass(frozen=True)
class RouteDecision:
    role: str
    candidate_id: str
    reason: str
    matched_rule_priority: int | None
    warnings: tuple[str, ...]
    rejected: bool
```

### 19.8 Router tests

至少測：

- unseen drug；
- seen drug／unseen scaffold；
- unseen cancer only；
- source-like；
- all metadata missing；
- requested role is null；
- checkpoint missing；
- conflicting rules；
- priority deterministic；
- policy schema invalid。

---

## 20. 詳細檔案修改清單

### 20.1 新增檔案

| 檔案 | 修改內容 | 優先度 |
|---|---|---:|
| `tools/round19_schema.py` | canonical IDs、dataclasses、manifest schema、selection input guard | P0 |
| `tools/round19_registry.py` | O/D/P registry、13-cell compatibility | P0 |
| `tools/round19_feature_adapter.py` | O0–O4 resolution、metadata、projection lineage、shuffle | P0 |
| `tools/round19_drug_encoders.py` | D0–D4 factory、unified output、MACCS adapter | P0 |
| `drugmodels/gineconv.py` | D3 bond-aware encoder | P0 |
| `tools/round19_fusion_models.py` | P0–P2 factory、forward contract、attention aux | P0 |
| `tools/round19_dataset.py` | unified batch、graph/fingerprint mode、control injection | P0 |
| `tools/round19_splits.py` | ModelID／drug／scaffold／cancer split 與 leakage audit | P0 |
| `tools/round19_config_builder.py` | 19A–19F manifests | P0 |
| `step1_finetune_latent_pipeline_round19_cv.py` | canonical ID runner、train/eval/infer | P0 |
| `tools/round19_metrics.py` | group macro、calibration、guardrail labels | P0 |
| `tools/analyze_round19.py` | stage aggregation、paired deltas、missing jobs | P0 |
| `tools/round19_selection.py` | 19B→19C、19C→19D、candidate locks | P0 |
| `tools/round19_role_lock.py` | role selection、JSON lock、policy | P0 |
| `tools/round19_release_audit.py` | hash、checkpoint、manifest、environment audit | P1 |
| `tools/round19_attention_faithfulness.py` | post-lock masking／consistency | P2 |

### 20.2 建議小幅修改既有檔案

#### `tools/graph_utils.py`

建議新增：

- canonical SMILES helper；
- GINE edge attributes；
- Bemis–Murcko scaffold；
- cache metadata helper。

不得改變既有 Round 18 graph output 的預設 schema。新增功能需透過 flag 或新函式。

#### `drugmodels/ginconv.py`

只在必要時新增 backward-compatible 方法：

```python
forward_with_nodes(...)
```

但不可改變原 `forward()` 回傳。Round 19 adapter 可偵測兩者。

#### `tools/round18_cv_metrics.py`

若 metric 已穩定，Round 19 直接 import。若需 calibration，新增 `round19_metrics.py`，不要讓 Round 18 headline metric 行為改變。

#### `tools/round18_oom_runner.py`

可直接重用。若新增 job metadata callback，必須有 Round 18 regression test。

#### `README.md`

完成後新增：

```markdown
## Round 19

- [Round 19 IDE manual](docs/round19_followup_ide_manual.md)
- [Round 19 final report](docs/round19_stage19f_report.md)
- [Project development history](docs/project_development_history_and_findings.md)
```

---

## 21. Config 修改範例

### 21.1 `config/round19_factorial_settings.json`

```json
{
  "round": 19,
  "output_root": "result/optimization_runs/round19",
  "reuse_round18": {
    "eligible_response": "result/optimization_runs/round18_architecture/data/round18_eligible_response.csv",
    "internal_split": "result/optimization_runs/round18_architecture/splits/internal_test_split.csv",
    "grouped_cv_assignment": "result/optimization_runs/round18_architecture/splits/cv_assignments.csv"
  },
  "features": {
    "feature_root": "result/optimization_runs/round19/stage19a/features",
    "model_key": "r13_exp_008",
    "omics_ids": ["O0", "O1", "O2", "O3", "O4"],
    "require_same_modelid_coverage": true,
    "require_o2_o3_projection_hash_match": true
  },
  "drugs": {
    "smiles_path": "<existing path>",
    "drug_ids": ["D0", "D1", "D2", "D3", "D4"],
    "graph_cache_root": "result/optimization_runs/round19/stage19a/graph_cache",
    "fingerprint_cache_root": "result/optimization_runs/round19/stage19a/fingerprint_cache"
  },
  "training": {
    "model_seed": 101,
    "amp": true,
    "early_stop_metric": "drug_macro_auc",
    "tie_break_metric": "drug_macro_auprc"
  },
  "selection": {
    "forbidden_column_patterns": [
      "internal",
      "tcga",
      "external",
      "integrated5",
      "posthoc"
    ]
  }
}
```

### 21.2 `config/params_round19_screening.json`

```json
{
  "stage": "19b",
  "omics_ids": ["O1", "O2", "O3"],
  "compatible_cells_source": "tools.round19_registry.COMPATIBLE_CELLS",
  "fold_ids": [0, 1, 2],
  "split_seed": 42,
  "model_seed": 101,
  "selection": {
    "primary": "drug_macro_auc",
    "tie_break": "drug_macro_auprc",
    "n_cells_for_stage19c": 7
  }
}
```

### 21.3 `config/params_round19_confirmation.json`

```json
{
  "stage": "19d",
  "candidate_lock": "result/optimization_runs/round19/stage19c/aggregate/round19c_selected_for_19d.json",
  "split_seeds": [52, 62, 72],
  "fold_ids": [0, 1, 2, 3, 4],
  "model_seed": 101
}
```

---

## 22. CLI 執行流程

### 22.1 環境與 baseline

```bash
python -m pip install -r requirements-round18.txt
python -m pip install -r tests/requirements-dev.txt
pytest -q tests/test_round18_*.py
```

### 22.2 建立 Round 19A

```bash
python -m tools.round19_config_builder \
  --stage 19a \
  --settings config/round19_factorial_settings.json \
  --outdir result/optimization_runs/round19/stage19a
```

### 22.3 Smoke

```bash
python step1_finetune_latent_pipeline_round19_cv.py \
  --mode smoke \
  --settings config/round19_factorial_settings.json \
  --outdir result/optimization_runs/round19/stage19a/smoke

python step1_finetune_latent_pipeline_round19_cv.py \
  --mode data_smoke \
  --settings config/round19_factorial_settings.json \
  --outdir result/optimization_runs/round19/stage19a/smoke
```

### 22.4 產生 19B manifest

```bash
python -m tools.round19_config_builder \
  --stage 19b \
  --settings config/round19_factorial_settings.json \
  --outdir result/optimization_runs/round19/stage19b
```

### 22.5 執行單一 manifest row

```bash
python step1_finetune_latent_pipeline_round19_cv.py \
  --mode train_manifest_row \
  --manifest result/optimization_runs/round19/stage19b/manifests/round19b_manifest.csv \
  --row-index 0
```

### 22.6 聚合

```bash
python -m tools.analyze_round19 \
  --stage 19b \
  --root result/optimization_runs/round19/stage19b
```

### 22.7 後續 stages

```bash
python -m tools.round19_selection --from-stage 19b --to-stage 19c ...
python -m tools.round19_config_builder --stage 19c ...
python -m tools.analyze_round19 --stage 19c ...

python -m tools.round19_selection --from-stage 19c --to-stage 19d ...
python -m tools.round19_config_builder --stage 19d ...
python -m tools.analyze_round19 --stage 19d ...

python -m tools.round19_config_builder --stage 19e ...
python -m tools.analyze_round19 --stage 19e ...

python -m tools.round19_role_lock ...
python -m tools.round19_release_audit ...
```

---

## 23. GPU、OOM 與 Resume 規格

### 23.1 OOM retry

允許調整：

- micro batch size；
- gradient accumulation steps。

不得自動調整：

- learning rate；
- model depth；
- d_model；
- hidden width；
- epoch cap；
- seed。

effective batch 應盡量保持：

```text
micro_batch × accumulation_steps
```

### 23.2 Resume

job state：

```text
PENDING
RUNNING
DONE
FAILED_RETRYABLE
FAILED_FINAL
STALE
```

Resume 規則：

- `DONE.json` 且 hash 一致 → skip；
- checkpoint 存在但無 DONE → 依 resume policy；
- resolved config hash 不同 → 不得 resume；
- split hash 不同 → 不得 resume；
- git SHA 不同 → 需 `--allow-code-drift` 並記錄，不建議正式跑使用。

### 23.3 Job completeness

每個 stage analyzer 都需比較：

```text
expected job IDs
vs
observed DONE job IDs
```

任何 headline report 必須列出 completeness。

---

## 24. Metrics 修改建議

### 24.1 核心 metrics

- Global AUC；
- Global AUPRC；
- DrugMacro AUC；
- DrugMacro AUPRC；
- cancer-type macro；
- Brier score；
- ECE；
- valid group count；
- skipped group count。

### 24.2 Group metric 邊界

當某個 drug 或 group 只有單一 label 時：

- 不計算該 group AUC；
- 記錄 skipped reason；
- 不填 0.5；
- aggregate JSON 要記錄 denominator。

### 24.3 Metric schema

```json
{
  "global": {"auc": 0.0, "auprc": 0.0},
  "drug_macro": {
    "auc": 0.0,
    "auprc": 0.0,
    "valid_groups": 0,
    "skipped_groups": 0
  },
  "calibration": {"brier": 0.0, "ece": 0.0},
  "n_rows": 0,
  "n_positive": 0,
  "n_negative": 0
}
```

---

## 25. Test Plan

### 25.1 Unit tests

#### Schema／registry

- invalid ID rejected；
- 13 cells exactly；
- D4 incompatible predictors rejected；
- job ID deterministic；
- forbidden selection columns rejected。

#### Features

- O0/O1/O2/O3 dimensions；
- O2/O3 projection hash equality；
- component slices；
- ModelID coverage；
- shuffled context only replaces context。

#### Drug encoders

- D0 parity；
- D1 graph dim 32；
- D2 graph dim 64；
- D3 edge_attr consumed；
- D4 bit vector deterministic；
- batch with variable nodes。

#### Predictors

- P0 logits shape；
- P1 token mask；
- P2 atom mask；
- P2 attention sum；
- no NaN；
- state dict round trip。

#### Splits

- group non-overlap；
- internal exclusion；
- drug identity non-overlap；
- scaffold non-overlap；
- cancer type non-overlap；
- deterministic seed。

#### Selection

- pairing key 正確；
- missing pair 報錯或標記；
- tie-break deterministic；
- external columns fail closed；
- general model can be null。

### 25.2 Integration tests

- builder → manifest → one job → aggregate；
- checkpoint save/load → same logits；
- OOM retry → same effective batch；
- role lock → deployment router；
- release audit detects missing checkpoint。

### 25.3 Regression tests

```bash
pytest -q tests/test_round18_*.py
pytest -q tests/test_round19_*.py
```

Round 19 PR 不可只跑新 tests。

---

## 26. Stage Acceptance Checklist

### 26.1 19A

- [ ] O0–O4 metadata 完整
- [ ] O2/O3 projection hash 相同
- [ ] D0 parity 通過
- [ ] D3 edge attrs 通過
- [ ] D4 cache deterministic
- [ ] 13 compatibility cells
- [ ] 19B manifest 117 rows
- [ ] selection external-column guard 通過
- [ ] real-data CUDA smoke 通過

### 26.2 19B

- [ ] 117 expected jobs
- [ ] missing/failed jobs 已列出
- [ ] O2−O1 paired table
- [ ] O3−O2 paired table
- [ ] P2−P1 interaction table
- [ ] 7 cells selection lock
- [ ] 未讀 internal／TCGA

### 26.3 19C

- [ ] 54 expected jobs
- [ ] true/shuffled integrity audit
- [ ] O2−O0 pairs
- [ ] O3−O2 pairs
- [ ] O4 comparison
- [ ] 6 candidates lock

### 26.4 19D

- [ ] 90 expected jobs
- [ ] seeds 52/62/72
- [ ] mean-of-means
- [ ] paired deltas
- [ ] rank stability
- [ ] historical anchor drift audit

### 26.5 19E

- [ ] 90 expected jobs
- [ ] three shift reports separated
- [ ] MAJOR_FAIL labels
- [ ] calibration
- [ ] chemical minimax summary
- [ ] no combined universal score

### 26.6 19F

- [ ] exact role assignments
- [ ] general role permits null
- [ ] policy precedence tested
- [ ] checkpoint set complete
- [ ] config/data/split hashes
- [ ] model cards
- [ ] release manifest
- [ ] internal／TCGA 僅標記 post-hoc

---

## 27. 報告修改建議

每個 stage report 都使用固定格式：

```markdown
# Round 19X Report

## 1. Objective
## 2. Frozen inputs
## 3. Variables tested
## 4. Manifest completeness
## 5. Primary results
## 6. Paired effects
## 7. Negative results
## 8. Failure modes
## 9. Decision gate
## 10. Artifacts and hashes
## 11. Reproduction commands
## 12. Limitations
```

### 27.1 禁止的報告寫法

避免：

```text
Cross-attention universally outperforms pooled models.
```

建議：

```text
Atom cross-attention was stronger in source-domain and cancer-type transfer,
whereas pooled integration was more robust under unseen-drug and
unseen-scaffold shifts.
```

### 27.2 結果數字來源

報告中的每一個 headline number 需連到：

```text
aggregate row
→ per-fold metrics
→ predictions
→ checkpoint
→ resolved config
→ manifest row
```

---

## 28. Release Audit

新增 `tools/round19_release_audit.py`。

### 28.1 Audit 項目

- Git working tree 是否乾淨；
- commit SHA；
- manifest hash；
- config hash；
- data／split hash；
- feature metadata hash；
- checkpoint existence；
- checkpoint readable；
- state dict keys match；
- role checkpoint coverage；
- policy role references valid；
- reports exist；
- environment captured；
- forbidden selection columns absent。

### 28.2 Release bundle

```text
release/round19_v1/
├── README.md
├── docs/
├── reports/
├── configs/
├── manifests/
├── splits/
├── feature_metadata/
├── checkpoints/
├── model_cards/
├── policies/
├── environment/
└── hashes/SHA256SUMS
```

### 28.3 Audit exit code

```text
0 = PASS
2 = incomplete non-critical
3 = reproducibility failure
4 = selection leakage risk
5 = policy/checkpoint inconsistency
```

---

## 29. Model Card 建議欄位

每個 role model：

```markdown
# Model Card: <role>

- Candidate ID
- O/D/P specification
- Training population
- Frozen omics encoder
- Feature context assumptions
- Validation evidence
- Known strengths
- Known failures
- Calibration
- Compute／latency／memory
- Required inference metadata
- Fallback behavior
- Prohibited use
- Checkpoint SHA256
- Config SHA256
- Git SHA
```

Chemical specialist 必須特別列出：

- drug-held-out；
- scaffold-held-out；
- canonicalization assumptions。

Cancer specialist 必須列出：

- cancer-type coverage；
- unseen cancer handling；
- prototype availability。

---

## 30. 建議 Commit／PR 順序

### PR 1 — Schema and registries

```text
round19_schema.py
round19_registry.py
unit tests
```

### PR 2 — Feature compositions

```text
round19_feature_adapter.py
O0–O4 metadata／export
shuffle controls
```

### PR 3 — Drug encoders

```text
GINE
MACCS cache
unified output
D0 parity tests
```

### PR 4 — Predictor and dataset

```text
round19_fusion_models.py
round19_dataset.py
forward contracts
```

### PR 5 — Splits and builder

```text
round19_splits.py
round19_config_builder.py
manifest tests
```

### PR 6 — Runner

```text
round19 pipeline
OOM/resume
smoke
```

### PR 7 — Analysis and selection

```text
metrics
analyze_round19
selection locks
```

### PR 8 — Domain shift and role lock

```text
19E guardrails
role lock
router
```

### PR 9 — Release and docs

```text
release audit
model cards
README links
reports
```

每個 PR 保持可測、可回退；不要用單一巨大 PR 同時加入所有模型與 selection。

---

## 31. IDE 執行時的工作規範

### 31.1 修改前

IDE agent 必須先：

1. 搜尋既有相同功能；
2. 列出準備重用的 Round 18 API；
3. 確認不會改變 historical output；
4. 建立或更新 tests；
5. 說明新 artifact path。

### 31.2 修改後

每次回覆格式：

```text
Changed:
- file A: ...
- file B: ...

Contracts:
- input ...
- output ...

Tests:
- ...

Commands run:
- ...

Remaining risks:
- ...
```

### 31.3 Agent 不得自行決定

- 變更 label threshold；
- 改變 eligible population；
- 移除 failed jobs；
- 以 TCGA 挑 winner；
- 解凍 omics encoder；
- 加入 hybrid fingerprint shortcut；
- 發明 Round 19F exact role；
- 將缺少的結果補成看似合理的數字。

---

## 32. 常見錯誤與修正

### 錯誤 1：把 d_model 加大當成 drug information 增加

**問題：** 32 維 graph embedding 投影到 128 維不會創造新化學內容。

**修正：** D1/D2 明確拆 node capacity 與 graph bottleneck。

### 錯誤 2：O2 重新 fit projection

**問題：** O2/O3 不再是可比較 ablation。

**修正：** 共用 O3 projection artifact，再移除 summary component。

### 錯誤 3：row-level shuffle context

**問題：** 同一 cell line 在不同 drug rows 取得不同 context，破壞 biological consistency。

**修正：** ModelID-level permutation。

### 錯誤 4：用原始 drug name 做 held-out

**問題：** alias／大小寫／鹽形式可能 leakage。

**修正：** canonical drug identity。

### 錯誤 5：把 unseen drug 等同 unseen scaffold

**問題：** seen-scaffold new drug 與 new scaffold 是不同難度。

**修正：** 分開 split 與報告。

### 錯誤 6：所有 shift 平均

**問題：** 遮蔽 scenario-specific failure。

**修正：** role-based policy。

### 錯誤 7：general model 必填

**問題：** 強迫一個具有已知 failure 的模型成為全域推薦。

**修正：** 允許 null。

### 錯誤 8：attention 當 causal importance

**問題：** attention weight 不等於遮罩後的性能影響。

**修正：** top-attention masking、matched random masking、connected substructure masking。

---

# Part VII — Round 19 完成後的詳細修改建議

## 33. 優先 0：補齊 Round 19F Exact Artifacts

在開始新架構前，先完成：

1. 核對 `round19_final_role_lock.json`；
2. 核對 `round19_deployment_policy.json`；
3. 將每個 role 連到 checkpoint set；
4. 補 internal post-hoc，但不作 selection；
5. 補 TCGA exploratory，但明確標記非 untouched；
6. 產生 SHA256SUMS；
7. 記錄 final commit；
8. 產生 model cards；
9. 執行 release audit。

這是最優先修改，不應先做 GIN128 或新 Transformer。

---

## 34. 優先 1：Attention／Context Faithfulness

建議新增：

```text
tools/round19_attention_export.py
tools/round19_attention_faithfulness.py
tests/test_round19_attention_export.py
tests/test_round19_attention_faithfulness.py
```

### 34.1 15-checkpoint consistency

若最終 atom role 有 3 split seeds × 5 folds，正好可比較 15 checkpoints：

- attention rank correlation；
- top-k atom overlap；
- scaffold-level consistency；
- patient-conditioned variation；
- entropy／concentration。

### 34.2 Perturbation controls

對每個 sample-drug pair：

```text
original
zero context
shuffled context
top-attention atom mask
matched-random atom mask
connected-substructure mask
```

輸出 performance delta，不只畫圖。

### 34.3 Selection freeze

Faithfulness 只能解釋 locked model，不得用於重新選 winner，除非正式開新 Round 且預註冊。

---

## 35. 優先 2：Untouched External Cohort Harness

新增獨立工具：

```text
tools/external_cohort_contract.py
tools/round19_locked_external_inference.py
tests/test_external_cohort_contract.py
```

必要流程：

1. 先鎖 role lock 與 protocol；
2. 建 external dataset card；
3. 驗證 feature／drug coverage；
4. 不讀 labels 執行 inference；
5. 封存 predictions hash；
6. 最後一次性解鎖 labels 計算 metrics；
7. 不根據結果修改模型。

---

## 36. 優先 3：Chemical Shift 改善

應另開新 Round，不要改 Round 19 lock。候選優先序：

1. pooling ablation：mean、max+mean、attention pooling、Set2Set；
2. pretrained molecular encoder；
3. scaffold-aware contrastive pretraining；
4. uncertainty／reject option；
5. graph positional or shortest-path bias。

不建議第一步：

```text
GIN hidden 128
Transformer depth++
GIN + MACCS concat
```

### 36.1 Uncertainty fallback

部署 router 可新增：

```text
low confidence／far OOD
→ abstain 或 fallback pooled specialist
```

但 threshold 需以獨立 validation 鎖定。

---

## 37. 優先 4：Biological Shift 改善

候選：

- cancer-invariant omics latent；
- source-only prototype redesign；
- domain-adversarial training；
- latent alignment；
- unlabeled target adaptation。

任何 target adaptation 都需清楚分類：

```text
inductive
transductive unlabeled
target-label supervised
```

不可混寫。

---

## 38. 優先 5：Adaptive Routing／MoE

目前只使用 deterministic metadata routing。若未來做 learned gating：

- 不能以 Round 19E 同一資料同時訓練與評估；
- 需要獨立 routing train/validation/test；
- gating feature 不得包含 response label proxy；
- 需與 deterministic routing 比較；
- router failure 需 fallback；
- 所有 expert checkpoint frozen 後再訓練 gate。

---

## 39. 最終 Done Definition

Round 19 工程上「完成」不是只有報告有數字，而是：

```text
code exists
+ tests pass
+ manifests complete
+ metrics reproducible
+ selection leakage guard passes
+ role lock exact
+ checkpoints complete
+ policy executable
+ release hashes complete
+ limitations documented
```

### 最終稽核命令範例

```bash
pytest -q tests/test_round18_*.py tests/test_round19_*.py

python -m tools.round19_release_audit \
  --root result/optimization_runs/round19 \
  --role-lock reports/round19_final_role_lock.json \
  --policy reports/round19_deployment_policy.json \
  --strict
```

預期：

```text
ROUND19_RELEASE_AUDIT=PASS
selection_leakage=0
missing_role_checkpoints=0
hash_mismatch=0
incomplete_required_jobs=0
```

---

## 40. 交接者快速閱讀順序

1. `docs/project_development_history_and_findings.md`
2. 本手冊
3. `config/round19_factorial_settings.json`
4. `tools/round19_registry.py`
5. `tools/round19_config_builder.py`
6. `step1_finetune_latent_pipeline_round19_cv.py`
7. `docs/round19_stage19b_report.md`
8. `docs/round19_stage19c_report.md`
9. `docs/round19_stage19d_report.md`
10. `docs/round19_stage19e_report.md`
11. `reports/round19_final_role_lock.json`
12. `reports/round19_deployment_policy.json`
13. `reports/round19_release_manifest.json`

---

## 41. 一頁式實作摘要

```text
保留 Round 18 不動。

19A：
建立 O0–O4、D0–D4、P0–P2 registry與adapter；
複用 eligible／internal／ModelID CV；新增 drug/scaffold/cancer splits；
產生 117-job manifest並通過 smoke。

19B：
13 cells × O1/O2/O3 × 3 folds；
成對分析 O2−O1、O3−O2、P2−P1；
鎖 7 cells，不讀 internal/TCGA。

19C：
7 cells × O0/O4 × 3 folds + 12 shuffled controls；
驗證 context sample-specific faithfulness；
鎖 6 candidates。

19D：
6 candidates × seeds 52/62/72 × 5 folds；
計算 mean-of-means、paired delta、rank stability。

19E：
6 candidates × 5 folds × cancer/drug/scaffold shifts；
分開報告，標記 MAJOR_FAIL，不做 universal average。

19F：
以白名單 evidence 鎖角色；
general model 可 null；
輸出 deterministic deployment policy、model cards、hash與release audit。
```
