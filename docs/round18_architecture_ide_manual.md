# drug-DCF Round 18 Cursor／Codex IDE 操作手冊

## Grouped-CV Drug–Omics Transformer and Atom-Level Cross-Attention Study

---

# 0. Round 18 定位

Round 18 的目標是：

```text
固定已驗證的 omics representations，
重新建立可信的 GDSC grouped CV benchmark，
比較 pooled GIN + MLP、pooled Transformer、
以及 GIN atom nodes × omics CLS cross-attention。
```

本輪不重新調整：

```text
VAEwC / CODE-AE pretraining
conditional adversarial loss
prototype alignment
VICReg
prototype feature extraction方法
omics encoder權重
```

本輪只改變：

```text
下游 drug encoder訓練語意
drug–omics fusion architecture
finetune超參數
cross-validation與報表邏輯
atom-level解釋性
```

---

# 1. 已封版的研究決策

## 1.1 Omics representations

Round 18 使用三種固定 representation：

```text
O0：z-only

O1：own_plus_summary
    主要 omics baseline

O2：own_proto_context_projected_16
    direct prototype control
```

所有 representations：

```text
從既有18-class-clean artifacts讀取
omics encoder完全 frozen
不參與 gradient
不重新跑 pretrain
```

---

## 1.2 Cross-attention方向

Round 18 只做：

```text
Q = omics CLS token
K = GIN atom node embeddings
V = GIN atom node embeddings
```

即：

```text
omics→atoms cross-attention
```

不做 atom→omics attention，因為本輪 omics只有一個 token，反方向只有一個 key/value，attention會退化。

---

## 1.3 Omics token

將完整 omics vector轉成單一 CLS/query token：

```text
omics_vector
→ Linear(input_dim, d_model)
→ LayerNorm
→ omics_cls [B, 1, d_model]
```

更新後的 omics CLS 作為主要分類 representation。

不新增 learnable CLS。

---

## 1.4 GIN訓練

GIN從頭 end-to-end訓練：

```text
drug_model.train()
fusion_model.train()
classifier.train()
```

optimizer包含：

```text
GIN parameters
Transformer / cross-attention parameters
response head parameters
```

不保留正式候選中的：

```text
drug_model.eval()但權重仍更新
```

舊語意只能作 legacy reproduction，不參與正式排名。

---

## 1.5 Graph與position設定

Round 18：

```text
sinusoidal positional encoding = off
graph attention bias = off
shortest-path bias = off
bond-type attention bias = off
token-type embedding = on
```

原因：

```text
atom index不是自然序列位置
GIN已先編碼圖拓樸與鄰域資訊
本輪先隔離cross-attention本身的效果
```

---

## 1.6 Primary CV metric

正式架構排名：

```text
Primary：
CV DrugMacro AUC

Tie-breaker：
CV DrugMacro AUPRC

Safety metrics：
CV Global AUC
CV Global AUPRC
```

Early stopping：

```text
Robust DrugMacro AUC
```

有效 drug條件：

```text
n_samples >= 10
n_positive >= 2
n_negative >= 2
```

若單一 validation fold的有效 drugs不足：

```text
fallback = Validation Global AUC
```

---

## 1.7 固定 seeds

```text
internal test split seed = 42
CV split seed = 42
model initialization seed = 101
DataLoader seed = 101
```

Round 18先固定單一 model seed，不進行多 seed展開。

---

## 1.8 TCGA規則

五個 TCGA targets：

```text
gdsc_intersect13
tcga_only3
dapl
aacdr_tcga_only
aacdr_gdsc_intersect
```

不得參與：

```text
架構選擇
超參數選擇
early stopping
batch size選擇
epoch選擇
```

TCGA只在架構與參數完全鎖定後執行。

最終 prediction：

```text
5-fold probability mean
```

不取單一最佳 fold。

---

## 1.9 Final model形式

只保留：

```text
5-fold ensemble
```

不額外訓練 full-development單模型。

---

# 2. Round 18 stages

```text
18A：
資料切分、baseline語意與OOM基礎建設

18B：
pooled GIN + MLP
pooled GIN + Transformer
3-fold architecture screening

18C：
atom-level cross-attention
pure與pooled-residual版本
3-fold screening

18D：
top architecture正式5CV
以及top 2 architecture drug-held-out CV

18E：
鎖定後5-target TCGA inference
5-fold probability ensemble

18F：
atom attention與masking解釋性
```

---

# 3. 建議新增檔案

## 3.1 Configs

```text
config/round18_architecture_settings.json
config/params_round18_screening.json
config/params_round18_formal_5cv.json
```

## 3.2 Model modules

```text
tools/cross_attention_switch.py
tools/round18_fusion_models.py
tools/round18_response_head.py
```

## 3.3 CV與metrics

```text
tools/round18_cv_splits.py
tools/round18_cv_metrics.py
tools/round18_prediction_ensemble.py
```

## 3.4 Runner與分析

```text
tools/round18_config_builder.py
tools/round18_oom_runner.py
tools/analyze_round18.py
tools/round18_interpretability.py
```

## 3.5 Pipeline

```text
step1_finetune_latent_pipeline_round18_cv.py
```

不要直接大改：

```text
step1_finetune_latent_pipeline_5fold_split.py
```

舊檔留作歷史重現。

## 3.6 Shell scripts

```text
tools/run_round18_stage18a_setup_smoke.sh
tools/run_round18_stage18b_pooled_screen.sh
tools/run_round18_stage18c_cross_attention.sh
tools/run_round18_stage18d_formal_5cv.sh
tools/run_round18_stage18e_tcga.sh
tools/run_round18_stage18f_interpretability.sh
```

---

# 4. 修改 GIN輸出介面

修改：

```text
drugmodels/ginconv.py
```

要求：

```text
保留舊forward輸出
新增可選node-level輸出
不破壞現有Round 1–17 pipeline
```

建議介面：

```python
def forward(
    self,
    data,
    return_node_embeddings: bool = False,
    return_graph_embedding: bool = True,
):
    node_embeddings = self.encode_nodes(data.x, data.edge_index)

    graph_embedding = None
    if return_graph_embedding:
        graph_embedding = self.pool_graph(
            node_embeddings,
            data.batch,
        )

    if not return_node_embeddings:
        return graph_embedding

    return {
        "node_embeddings": node_embeddings,
        "batch_index": data.batch,
        "graph_embedding": graph_embedding,
    }
```

## 4.1 第一階段 GIN固定設定

```text
input_dim = 78
hidden_dim = 32
num_layers = 5
JK = last
BatchNorm = true
pool = max
dropout = 0.1
```

Atom node embeddings再投影：

```text
32 → d_model
```

第二階段只對top cross-attention architecture測：

```text
JK = last
JK = cat
```

---

# 5. Dense atom batch

PyG的nodes是flat tensor，需要轉為dense：

```python
from torch_geometric.utils import to_dense_batch

atom_dense, atom_valid_mask = to_dense_batch(
    node_embeddings,
    batch=batch_index,
)
```

形狀：

```text
atom_dense:
[B, max_atoms, node_dim]

atom_valid_mask:
[B, max_atoms]
```

投影：

```python
atom_tokens = atom_projection(atom_dense)
```

Attention mask：

```python
key_padding_mask = ~atom_valid_mask
```

必須確認：

```text
True = padding position
False = valid atom
```

不能讓padding atoms進入softmax。

---

# 6. 新增 CrossAttentionSwitch

新增：

```text
tools/cross_attention_switch.py
```

不要修改舊 `TransformerSwitch` 的forward介面。

建議class：

```python
class CrossAttentionSwitch(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        temperature: float = 1.0,
        attn_out_mlp: bool = True,
        attn_out_activation: str = "GELU",
        ffn_activation: str = "ReLU",
    ):
        super().__init__()
        ...

    def forward(
        self,
        query_tokens,
        key_value_tokens,
        key_padding_mask=None,
        return_attention=False,
    ):
        ...
```

輸入：

```text
query_tokens：
[B, 1, d_model]

key_value_tokens：
[B, max_atoms, d_model]

key_padding_mask：
[B, max_atoms]
```

輸出：

```text
updated_query：
[B, 1, d_model]

attention_weights：
[num_layers, B, n_heads, 1, max_atoms]
```

## 6.1 Attention計算

```text
Q來自omics CLS
K、V來自atom tokens
```

每層：

```text
CrossAttention
→ residual + LayerNorm
→ FeedForward
→ residual + LayerNorm
```

不對atom tokens加sinusoidal PE。

---

# 7. Fusion architecture families

## 7.1 A0–A2：Pooled GIN + MLP

三個 omics inputs：

```text
A0：z-only
A1：own_plus_summary
A2：context16
```

流程：

```text
GIN nodes
→ global max pool
→ graph embedding

concat(omics_vector, graph_embedding)
→ fixed response head
```

---

## 7.2 A3–A5：Pooled GIN + Transformer

三個 omics inputs：

```text
A3：z-only
A4：own_plus_summary
A5：context16
```

Tokens：

```text
Token 0 = projected omics token
Token 1 = projected pooled GIN token
```

加入：

```text
omics token-type embedding
drug token-type embedding
```

不加入：

```text
sinusoidal positional encoding
```

經過 `TransformerSwitch` self-attention後：

```text
取Token 0作updated omics CLS
→ response head
```

固定兩個dense tokens，沒有padding。

即使historical config設定：

```text
use_mask = true
```

也不要使用依token第一個數值自動判定padding的mask。

應明確傳入：

```text
all_false_attention_mask
```

或在Round 18 wrapper中停用auto-pad detection。

---

## 7.3 C0：Pure atom cross-attention

```text
omics vector
→ omics CLS

GIN
→ atom tokens

omics CLS queries atom tokens
→ updated CLS
→ response head
```

不加入pooled graph residual。

---

## 7.4 C1：Cross-attention + pooled residual

```text
updated omics CLS
+
global max pooled GIN representation
+
original projected omics vector
→ concat
→ response head
```

三者都保留。

這是Round 18主要候選。

---

# 8. Fixed response head

所有架構使用相同後端容量：

```text
fusion representation
→ Linear(input_dim, 128)
→ ReLU
→ Dropout(0.1)
→ Linear(128, 1)
```

輸出：

```text
logit
```

不要在18B/18C同時搜尋：

```text
[256]
[128,64]
更深MLP
不同activation
```

確保architecture差異來自fusion，不是classifier容量。

---

# 9. Loss與optimizer

## 9.1 Loss

沿用現有：

```text
FocalLoss
現有sample weights
現有gamma
現有class weighting
```

所有架構完全一致。

本輪不搜尋：

```text
BCE
focal gamma
label smoothing
class weight
```

---

## 9.2 Optimizer parameter groups

建議分開learning rates：

```python
optimizer = AdamW([
    {
        "params": gin.parameters(),
        "lr": gin_lr,
    },
    {
        "params": fusion_model.parameters(),
        "lr": fusion_lr,
    },
    {
        "params": response_head.parameters(),
        "lr": head_lr,
    },
], weight_decay=weight_decay)
```

Screening初始值：

```text
gin_lr = 1e-4
fusion_lr = 3e-4
head_lr = 3e-4
weight_decay = 1e-4
```

Historical pooled Transformer baseline：

```text
global lr = 1e-3
```

---

## 9.3 Gradient clipping

裁切全部trainable parameters：

```python
trainable_params = [
    p
    for module in [gin, fusion_model, response_head]
    for p in module.parameters()
    if p.requires_grad
]

torch.nn.utils.clip_grad_norm_(
    trainable_params,
    max_norm=1.0,
)
```

---

# 10. Round 18資料切分

## 10.1 Fixed 10% internal test

使用：

```python
StratifiedGroupKFold(
    n_splits=10,
    shuffle=True,
    random_state=42,
)
```

輸入：

```text
X = response rows
y = 每列Label
groups = 每列ModelID
```

固定：

```text
fold 0 = internal test
fold 1–9 = development set
```

比例可能不是精確10%，因為ModelID group大小不同。

必須輸出實際比例。

---

## 10.2 Development 3-fold screening

在90% development data上：

```python
StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)
```

同一套split manifest供所有18B/18C candidates共用。

---

## 10.3 Formal development 5CV

在相同90% development data上：

```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

同一ModelID不可跨train/validation。

---

## 10.4 Split outputs

建立：

```text
result/optimization_runs/round18_architecture/splits/
```

輸出：

```text
internal_test_split.csv
development_rows.csv
screening_3fold_assignments.csv
formal_5fold_assignments.csv
split_summary.csv
split_metadata.json
```

`split_metadata.json`：

```json
{
  "internal_test_split_method": "StratifiedGroupKFold_10fold_fold0",
  "internal_test_split_seed": 42,
  "screening_cv_folds": 3,
  "formal_cv_folds": 5,
  "cv_split_seed": 42,
  "group_column": "ModelID",
  "stratification_column": "Label"
}
```

---

# 11. Split QC報表

每個split／fold輸出：

```text
n_rows
n_model_ids
n_drugs
n_positive
n_negative
positive_rate
drug_count_distribution
cancer_type_distribution（若可取得）
ModelID_overlap_count
```

Assertions：

```python
assert train_model_ids.isdisjoint(val_model_ids)
assert dev_model_ids.isdisjoint(test_model_ids)
assert set(train_labels).issubset({0, 1})
assert set(val_labels).issubset({0, 1})
```

輸出：

```text
fold_balance_report.csv
fold_drug_distribution.csv
fold_label_distribution.csv
fold_group_overlap_qc.csv
```

任何 overlap：

```text
立即fail
```

---

# 12. Internal test鎖定規則

18A–18D期間：

```text
不計算internal test metrics
不讀internal test labels做selection
不寫internal test排名
```

正式architecture與超參數鎖定後：

```text
5個formal fold models
各自對internal test inference
probability mean
計算一次final internal test metrics
```

internal test不得參與：

```text
early stopping
candidate ranking
epoch selection
batch selection
```

---

# 13. Robust DrugMacro metric

新增：

```text
tools/round18_cv_metrics.py
```

建議函數：

```python
def calculate_robust_drug_macro_metrics(
    prediction_df,
    drug_col="DRUG_NAME",
    label_col="Label",
    probability_col="probability",
    min_samples=10,
    min_positive=2,
    min_negative=2,
):
    ...
```

每個drug：

```text
符合valid條件：
計算AUC/AUPRC

不符合：
status = insufficient_class_support
不放進macro mean
```

輸出：

```text
DrugMacro_AUC
DrugMacro_AUPRC
n_valid_auc_drugs
n_valid_auprc_drugs
n_total_drugs
valid_drug_fraction
```

Early stopping fallback：

```text
若n_valid_auc_drugs < 3：
使用Global AUC
```

---

# 14. Historical Transformer baseline

附件中的historical candidate加入18B：

```text
candidate_id：
pooled_transformer_historical
```

設定：

```text
learning_rate = 1e-3
n_heads = 4
num_layers = 1
dim_feedforward = 128
dropout = 0.1
attn_dropout = 0.1
temperature = 1.0
use_mask = true
use_positional_encoding = false
attn_out_mlp = true
```

注意：

```text
只沿用architecture hyperparameters
不沿用舊TCGA結果
不沿用舊資料split
必須在新grouped CV重訓
```

---

# 15. Screening hyperparameter grid

## 15.1 Pooled Transformer curated configs

```text
P0 historical:
d_model=128
heads=4
layers=1
d_ff=128
dropout=0.1
attn_dropout=0.1

P1 compact64:
d_model=64
heads=4
layers=1
d_ff=128
dropout=0.1
attn_dropout=0.1

P2 standard128:
d_model=128
heads=4
layers=1
d_ff=256
dropout=0.1
attn_dropout=0.1

P3 deeper128:
d_model=128
heads=4
layers=2
d_ff=256
dropout=0.2
attn_dropout=0.1
```

每個套用：

```text
z-only
own_plus_summary
context16
```

---

## 15.2 Cross-attention curated configs

```text
X0:
d_model=64
heads=4
layers=1
d_ff=128
dropout=0.1

X1:
d_model=128
heads=4
layers=1
d_ff=256
dropout=0.1

X2:
d_model=128
heads=4
layers=2
d_ff=256
dropout=0.1

X3:
d_model=128
heads=4
layers=2
d_ff=256
dropout=0.2
```

每個同時測：

```text
pure cross-attention
cross-attention + pooled residual
```

初始screening只用：

```text
own_plus_summary
```

取top 2 cross-attention candidates後，再比較：

```text
z-only
own_plus_summary
context16
```

---

# 16. Job數量

## 16.1 Stage 18B

MLP baselines：

```text
3 omics × 3 folds
= 9 jobs
```

Pooled Transformer：

```text
4 configs × 3 omics × 3 folds
= 36 jobs
```

合計：

```text
45 jobs
```

## 16.2 Stage 18C first screen

```text
4 configs
× 2 residual modes
× own_plus_summary
× 3 folds
= 24 jobs
```

Top 2進omics comparison：

```text
2 candidates
× 3 omics
× 3 folds
= 18 jobs
```

Stage 18C合計約：

```text
42 jobs
```

## 16.3 Stage 18D formal 5CV

建議正式候選：

```text
MLP own_plus_summary baseline
best pooled Transformer
best pure cross-attention
best residual cross-attention
```

```text
4 candidates × 5 folds
= 20 jobs
```

## 16.4 Drug-held-out CV

只對top 2 architectures：

```text
2 candidates × 5 folds
= 10 jobs
```

總量約：

```text
117 training jobs
```

不含OOM retries。

---

# 17. OOM-safe runner

新增：

```text
tools/round18_oom_runner.py
```

## 17.1 GPU策略

```text
自動讀取torch.cuda.device_count()
每張GPU同時最多1個training job
```

因GPU數量未知，runner不得假定固定GPU數。

預設：

```text
MAX_JOBS_PER_GPU=1
```

---

## 17.2 Micro-batch probing

依序嘗試：

```text
512
256
128
64
32
```

Target effective batch：

```text
1024
```

Gradient accumulation：

```python
accum_steps = math.ceil(
    target_effective_batch / successful_micro_batch
)
```

例如：

```text
micro batch 256
→ accumulation 4
→ effective batch 1024
```

---

## 17.3 OOM retry

最多：

```text
4 retries
5 total attempts
```

OOM流程：

```text
1. 捕捉CUDA out of memory
2. 寫入OOM事件
3. 結束該training subprocess
4. 清理CUDA cache
5. micro batch減半
6. 從fold起點重新執行
```

不要在同一training process中直接續跑。

命令層級exit code建議：

```text
42 = retryable CUDA OOM
其他非0 = hard failure
```

---

## 17.4 AMP

使用：

```text
torch.cuda.amp.autocast
GradScaler
```

Config：

```text
amp_enabled = true
```

若發生NaN而非OOM：

```text
不可自動當作OOM降batch
必須標為numerical_failure
```

---

## 17.5 OOM metadata

每個job輸出：

```text
runtime_resource_summary.json
oom_retry_history.json
```

欄位：

```json
{
  "gpu_name": "NVIDIA RTX A6000",
  "requested_micro_batch": 512,
  "successful_micro_batch": 128,
  "gradient_accumulation_steps": 8,
  "effective_batch_size": 1024,
  "peak_gpu_memory_mb": 0,
  "oom_retry_count": 2,
  "oom_batch_history": [512, 256],
  "amp_enabled": true
}
```

---

# 18. Training epochs

## 18.1 18B/18C screening

```text
max_epochs = 500
early_stopping_patience = 50
early_stopping_start_epoch = 30
```

## 18.2 18D formal 5CV

```text
max_epochs = 1500
early_stopping_patience = 100
early_stopping_start_epoch = 50
```

Best checkpoint依：

```text
Robust DrugMacro AUC
```

不是Global AUC。

---

# 19. Pipeline training loop

新pipeline不得複製舊的錯誤語意。

建議：

```python
def train_one_epoch(
    gin_model,
    fusion_model,
    response_head,
    dataloader,
    optimizer,
    scaler,
    loss_fn,
    accumulation_steps,
):
    gin_model.train()
    fusion_model.train()
    response_head.train()

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        with autocast(enabled=True):
            logits = forward_round18(...)
            loss = loss_fn(...)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (
            (step + 1) % accumulation_steps == 0
            or step + 1 == len(dataloader)
        ):
            scaler.unscale_(optimizer)

            clip_grad_norm_(
                all_trainable_parameters,
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
```

Validation：

```text
all models eval()
torch.no_grad()
```

---

# 20. Omics feature dimension

不要硬編碼32、64、75等。

每個feature artifact應讀：

```text
feature_metadata.json
response_input_dim
feature_mode
feature_names
```

Model builder：

```python
omics_input_dim = feature_matrix.shape[1]
```

並assert metadata一致：

```python
assert omics_input_dim == metadata["response_input_dim"]
```

---

# 21. Drug-held-out CV

只對18D top 2 architectures執行。

Groups：

```text
DRUG_NAME
```

Splitter：

```python
GroupKFold(n_splits=5)
```

因不能同時保證每drug在train/val都出現，drug macro metric需重新定義：

```text
validation drugs全部為unseen drugs
主要報Global AUC / AUPRC
並報per-drug有效性
```

Drug-held-out結果為secondary robustness，不取代ModelID-held-out primary ranking。

---

# 22. TCGA five-fold ensemble

每個formal fold model：

```text
fold_0
fold_1
fold_2
fold_3
fold_4
```

各自對五個TCGA targets inference。

每列prediction必須包含：

```text
Patient_id
drug_name
Label
fold_id
logit
probability
target_key
architecture_id
omics_mode
```

Ensemble：

```python
ensemble_df = (
    fold_predictions
    .groupby([
        "target_key",
        "Patient_id",
        "drug_name",
        "Label",
    ])
    .agg(
        probability=("probability", "mean"),
        probability_std=("probability", "std"),
        n_folds=("fold_id", "nunique"),
    )
    .reset_index()
)
```

必須assert：

```python
assert ensemble_df["n_folds"].eq(5).all()
```

---

# 23. Integrated5 report

沿用現有5-target定義。

輸出：

```text
Historical Average_TCGA_AUC
Historical Global_TCGA_AUC

Integrated5_TargetMacro_TCGA_AUC
Integrated5_TargetMacro_TCGA_AUPRC

Integrated5_DrugMacro_TCGA_AUC
Integrated5_DrugMacro_TCGA_AUPRC
```

不使用weighted target mean。

---

# 24. Attention解釋性

只對：

```text
final locked cross-attention architecture
5-fold ensemble
```

輸出attention。

不對所有screening jobs保存完整attention，以避免資料量爆炸。

---

## 24.1 自動案例選擇

18E完成後依固定規則選：

```text
1. 樣本數最多的drug
2. cross-attention相對MLP改善最大的drug
3. cross-attention相對MLP退步最大的drug
4. AACDR新增且具有有效正負labels的drug
5. 每個drug取高預測正例與高預測負例
```

這些規則只能用於事後解釋，不可回頭調模型。

---

## 24.2 Attention輸出

```text
atom_attention_scores.csv
attention_entropy.csv
top_attended_atoms.csv
fold_attention_consistency.csv
```

欄位：

```text
target_key
Patient_id
drug_name
fold_id
atom_index
atom_symbol
attention_score
attention_rank
prediction
Label
```

---

## 24.3 Atom masking validation

Attention不是充分因果解釋，因此加入masking：

```text
mask top-1 atom
mask top-3 atoms
mask top-10% atoms
mask random matched atoms
```

比較：

```text
original_probability
masked_probability
prediction_delta
```

高attention atoms若有意義，應比random masking造成更大的prediction change。

輸出：

```text
atom_masking_ablation.csv
atom_masking_summary.csv
```

---

## 24.4 分子圖

使用RDKit輸出：

```text
drug__patient__fold_attention.png
drug__patient__ensemble_attention.png
```

顏色可表示attention強度。

報告文字避免直接宣稱：

```text
attention = biological mechanism
```

應寫成：

```text
attention identifies model-prioritized molecular substructures,
validated using masking sensitivity.
```

---

# 25. Config範例

## `config/round18_architecture_settings.json`

```json
{
  "round": "round18",
  "purpose": "Grouped-CV drug-omics transformer and atom cross-attention benchmark",
  "drug_smiles_path": "data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv",
  "response_data_path": "<REQUIRED_CLI_OR_CONFIG_VALUE>",
  "model_seed": 101,
  "dataloader_seed": 101,
  "split_seed": 42,
  "internal_test": {
    "enabled": true,
    "method": "stratified_group_10fold_fold0",
    "group_column": "ModelID",
    "label_column": "Label",
    "locked_until_final": true
  },
  "screening_cv": {
    "n_splits": 3,
    "max_epochs": 500,
    "early_stop_patience": 50,
    "early_stop_start_epoch": 30
  },
  "formal_cv": {
    "n_splits": 5,
    "max_epochs": 1500,
    "early_stop_patience": 100,
    "early_stop_start_epoch": 50
  },
  "selection": {
    "primary_metric": "CV_DrugMacro_AUC",
    "tie_breaker_metric": "CV_DrugMacro_AUPRC",
    "fallback_metric": "CV_Global_AUC"
  },
  "robust_drug_metric": {
    "min_samples": 10,
    "min_positive": 2,
    "min_negative": 2,
    "min_valid_drugs_for_early_stop": 3
  },
  "omics_modes": [
    "none",
    "own_plus_summary",
    "own_proto_context_projected_16"
  ],
  "gin": {
    "input_dim": 78,
    "hidden_dim": 32,
    "num_layers": 5,
    "jk_mode": "last",
    "pool": "max",
    "dropout": 0.1,
    "train_mode": "end_to_end_from_scratch"
  },
  "response_head": {
    "hidden_dim": 128,
    "activation": "relu",
    "dropout": 0.1
  },
  "oom": {
    "amp": true,
    "target_effective_batch": 1024,
    "micro_batch_candidates": [512, 256, 128, 64, 32],
    "max_retries": 4,
    "jobs_per_gpu": 1
  },
  "tcga": {
    "run_only_after_lock": true,
    "ensemble_method": "probability_mean",
    "require_all_five_folds": true
  },
  "interpretability": {
    "enabled_for_final_model_only": true,
    "attention_output": true,
    "atom_masking": true
  }
}
```

---

# 26. Config builder

新增：

```text
tools/round18_config_builder.py
```

Commands：

```bash
python tools/round18_config_builder.py \
  --settings config/round18_architecture_settings.json \
  --stage 18a \
  --outdir result/optimization_runs/round18_architecture
```

Stages：

```text
18a
18b
18c
18d
18e
18f
```

Manifest欄位：

```text
job_id
stage
architecture_id
omics_mode
transformer_config_id
residual_mode
cv_type
fold_id
model_seed
split_seed
drug_smiles_path
response_data_path
feature_dir
result_dir
requested_micro_batch
target_effective_batch
```

---

# 27. Stage scripts

## 27.1 18A

```bash
bash tools/run_round18_stage18a_setup_smoke.sh
```

執行：

```text
compile
tests
建立split manifests
split QC
GIN API smoke
CrossAttention shape smoke
OOM batch probe
```

---

## 27.2 18B

```bash
bash tools/run_round18_stage18b_pooled_screen.sh
```

執行：

```text
MLP 3 omics
Pooled Transformer 4 configs × 3 omics
3-fold screening
```

---

## 27.3 18C

```bash
bash tools/run_round18_stage18c_cross_attention.sh
```

執行：

```text
Pure與residual cross-attention
own_plus_summary first screen
top 2再測三種omics
```

---

## 27.4 18D

```bash
bash tools/run_round18_stage18d_formal_5cv.sh
```

執行：

```text
formal 5CV
internal test仍鎖定
top 2 drug-held-out CV
```

---

## 27.5 18E

**Status:** DONE（2026-07-14）。結果見 [`docs/round18_stage18e_report.md`](round18_stage18e_report.md) 與 [`docs/round18_final_report.md`](round18_final_report.md)。  
`cross_attention_external_success = false`（TCGA non-worse 2/5）。

```bash
SMOKE_ONLY=0 MAX_JOBS_PER_GPU=8 ROUND18_NUM_WORKERS=0 \
  bash tools/run_round18_stage18e_locked_eval.sh

python tools/analyze_round18_external_eval.py \
  --outdir result/optimization_runs/round18_architecture \
  --n-bootstrap 2000 --n-jobs 16
```

執行：

```text
internal test final ensemble（25 jobs）
5-target TCGA fold inference（125 jobs）
probability ensemble
Integrated5 + paired bootstrap report
verdict JSON（不得回頭改選模）
```

---

## 27.6 18F

```bash
bash tools/run_round18_stage18f_interpretability.sh
```

執行：

```text
attention export
case selection
atom masking
RDKit visualization
```

---

# 28. Tests

新增：

```text
tests/test_round18_cv_splits.py
tests/test_round18_gin_node_output.py
tests/test_round18_cross_attention.py
tests/test_round18_fusion_models.py
tests/test_round18_robust_drug_macro.py
tests/test_round18_oom_retry.py
tests/test_round18_tcga_ensemble.py
tests/test_round18_interpretability.py
tests/test_round18_config_builder.py
tests/test_round18_pipeline_smoke.py
```

---

## 28.1 CV tests

```text
ModelID不跨train/val
internal test不與development重疊
fold assignments固定可重現
同一split供所有架構共用
```

---

## 28.2 Cross-attention tests

```text
updated CLS shape正確
attention shape正確
padding atoms attention接近0
所有valid atom attention總和接近1
不同atom順序在相應重排後輸出一致
```

---

## 28.3 OOM tests

使用synthetic exception：

```text
512 OOM
256 OOM
128 success
```

assert：

```text
successful batch = 128
retry count = 2
accumulation = 8
history = [512,256]
```

---

## 28.4 Ensemble tests

```text
每個patient-drug必須有5 folds
probability為5fold mean
缺fold時fail
不允許以最佳fold替代
```

---

# 29. Analyzer輸出

新增：

```text
tools/analyze_round18.py
```

輸出：

```text
round18_job_completion_summary.csv
round18_split_balance_summary.csv
round18_screening_architecture_ranking.csv
round18_omics_mode_summary.csv
round18_formal_5cv_summary.csv
round18_drug_heldout_summary.csv
round18_internal_test_summary.csv
round18_five_target_tcga_summary.csv
round18_integrated5_summary.csv
round18_resource_usage_summary.csv
round18_oom_summary.csv
round18_attention_artifact_index.csv
round18_final_report.md
```

---

# 30. 報表底層生成邏輯

## 30.1 Screening ranking

每個candidate：

```text
先計算每fold DrugMacro AUC
再取3-fold arithmetic mean
```

Tie-break：

```text
3-fold mean DrugMacro AUPRC
```

不得使用：

```text
internal test
TCGA
single best fold
```

---

## 30.2 Formal 5CV ranking

每個candidate輸出：

```text
mean ± std
median
min
max
valid folds
```

Metrics：

```text
DrugMacro AUC
DrugMacro AUPRC
Global AUC
Global AUPRC
```

Primary排名：

```text
mean DrugMacro AUC descending
```

---

## 30.3 OOF report

將formal 5-fold validation predictions拼接：

```text
每個development row只出現一次
```

輸出：

```text
development_oof_predictions.csv
```

再計算：

```text
OOF Global AUC
OOF Global AUPRC
OOF DrugMacro AUC
OOF DrugMacro AUPRC
```

Fold mean與OOF metric都要報，不能混為同一數值。

---

## 30.4 Internal test report

只在18E生成：

```text
5 fold probabilities mean
```

輸出：

```text
internal_test_fold_predictions.csv
internal_test_ensemble_predictions.csv
internal_test_metrics_summary.csv
```

---

## 30.5 TCGA report

每個target：

```text
fold-level metrics
ensemble metrics
per-drug metrics
valid drug count
```

Integrated5只用ensemble predictions計算。

---

# 31. Acceptance criteria

## 31.1 18A成功

```text
split QC全過
CrossAttention smoke全過
GIN train mode確認
OOM fallback可用
extended SMILES可讀
```

## 31.2 18B成功

```text
45 jobs完成
無TCGA selection
historical Transformer在新CV可重現訓練
```

## 31.3 18C成功

```text
attention mask正確
pure/residual均可訓練
至少一個cross-attention candidate不低於MLP超過0.01
```

## 31.4 18D method success

```text
cross-attention 5CV DrugMacro AUC
高於MLP baseline
且至少4/5 folds不退步
```

或：

```text
平均差距在0.003內
但fold std降低至少20%
```

## 31.5 Strong success

```text
formal 5CV DrugMacro AUC提升
internal test不退步
Integrated5 TargetMacro不退步
且至少3/5 TCGA targets改善
```

---

# 32. Cursor／Codex實作順序

## Task 1：CV拆分

```text
建立round18_cv_splits.py。
不要取每個ModelID第一筆Label。
使用row-level Label + ModelID groups。
建立固定10% test、3CV、5CV manifests與QC。
```

## Task 2：GIN node API

```text
修改GINConvNet，保留舊forward相容性，
新增return_node_embeddings模式與graph embedding。
```

## Task 3：CrossAttentionSwitch

```text
新增獨立CrossAttentionSwitch。
Q來自omics CLS，K/V來自atom nodes。
支援key padding mask與attention export。
不修改舊TransformerSwitch介面。
```

## Task 4：Fusion models

```text
實作pooled MLP、pooled Transformer、
pure cross-attention、residual cross-attention。
共用固定response head。
```

## Task 5：Training loop

```text
GIN/fusion/head全部train mode。
AMP、gradient accumulation、all-parameter clipping。
robust DrugMacro early stopping。
```

## Task 6：OOM runner

```text
process-level OOM retry。
512→256→128→64→32。
每GPU一job。
記錄resource metadata。
```

## Task 7：Analyzer

```text
嚴格隔離screening、formal CV、
internal test與TCGA報表。
禁止internal test/TCGA參與selection。
```

## Task 8：Interpretability

```text
只對final model保存attention。
實作atom masking與RDKit視覺化。
```

---

# 33. 啟動前命令

```bash
python -m py_compile \
  tools/cross_attention_switch.py \
  tools/round18_fusion_models.py \
  tools/round18_cv_splits.py \
  tools/round18_cv_metrics.py \
  tools/round18_oom_runner.py \
  tools/round18_config_builder.py \
  tools/analyze_round18.py \
  step1_finetune_latent_pipeline_round18_cv.py
```

```bash
pytest tests/test_round18_*.py -q
```

```bash
bash tools/run_round18_stage18a_setup_smoke.sh
```

只有18A全部通過後，才能啟動18B。

---

# 34. 最終執行順序

```bash
bash tools/run_round18_stage18a_setup_smoke.sh
```

```bash
bash tools/run_round18_stage18b_pooled_screen.sh
```

```bash
bash tools/run_round18_stage18c_cross_attention.sh
```

檢查18B/18C報告，builder產生鎖定候選後：

```bash
bash tools/run_round18_stage18d_formal_5cv.sh
```

正式5CV完成、架構鎖定後：

```bash
bash tools/run_round18_stage18e_tcga.sh
```

最後：

```bash
bash tools/run_round18_stage18f_interpretability.sh
```

---

# 35. Round 18最終研究問題

Round 18報告必須回答：

```text
1. 修正GIN train/eval後，MLP baseline是否改變？
2. Pooled Transformer是否在grouped CV仍優於MLP？
3. 保留atom nodes的cross-attention是否優於early pooling？
4. Pooled residual是否提高穩定性？
5. 哪一種omics representation最適合各架構？
6. Cross-attention是否提升unseen ModelID泛化？
7. Top architecture是否提升unseen drug泛化？
8. CV優勢是否轉移到5-target TCGA？
9. 高attention atoms是否通過masking validation？
10. OOM降batch後是否仍維持相同effective batch與公平比較？
```

---

# 36. 最終定案

Round 18正式名稱：

```text
Round 18:
Grouped-CV Drug–Omics Transformer
and Atom-Level Cross-Attention Study
```

主線：

```text
固定omics
修正GIN訓練語意
以ModelID grouped CV選架構
比較early pooling與atom-level interaction
最後鎖定後才跑TCGA
```

本輪不以單次TCGA peak為目標。

本輪的主要成功定義是：

```text
在不使用TCGA調參的前提下，
atom-level cross-attention於正式grouped 5CV
穩定優於或追平pooled GIN + MLP，
並提供經atom masking驗證的分子層級解釋。
```
