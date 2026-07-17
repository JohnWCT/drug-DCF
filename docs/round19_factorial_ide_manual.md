# Round 19 IDE操作手冊

> Round 20 unseen-drug closure IDE manual：`docs/round20_unseen_drug_closure_ide_manual.md`
> Round 20-0 preflight report：`docs/round20_stage20_0_preflight_report.md`

## Omics Composition × Drug Representation × Predictor Integration Factorial Study

**版本：** Post-Round 18E revision  
**狀態：** Stage 19A GO · **19B ALL_DONE（117/117）** · **19C ALL_DONE（54/54）** · **19D ALL_DONE（90/90）** · **19E ALL_DONE（90/90）** · **19F ALL_DONE（540/540）** · **19G ALL_DONE（1,801/1,801）** · **19H LOCAL ARCHIVE COMPLETE** — Formal roles immutable locked；見 [`docs/round19_final_report.md`](round19_final_report.md)
**後續手冊：** 公開重建／adapter-first 實作規格見 [`docs/round19_followup_ide_manual.md`](round19_followup_ide_manual.md)
**前置結果：** Round 18A–18E完成  
**執行環境：** Docker container `DAPL`  
**原則：** MACCS與GIN／GINE互斥，不建立fingerprint + graph hybrid

---

# 0. Round 19定位

## 0.1 Round 18已回答的問題

Round 18顯示：

1. 在ModelID-grouped formal 5CV中，X3 pure cross-attention ×完整context輸入表現最佳。
2. Pure與pooled residual幾乎相同，因此pooled shortcut不是主要增益來源。
3. P1 compact Transformer比P3 deeper Transformer更穩。
4. Cross-attention的增益高度依賴omics prototype-context features，而不是所有omics輸入都能受益。公開Round 18D報告記錄了X3 pure、X3 residual、P1、P3、MLP的正式排序，以及context16對cross-attention的重要交互作用。
5. 18E顯示上述CV/internal增益未穩定轉移到TCGA。

因此Round 19不能再宣稱：

```text
提高CV AUC
=
提高TCGA泛化能力
```

Round 19的目標改成：

```text
1. 找出response predictor真正需要的omics內容。
2. 找出drug encoder缺的是容量、bond資訊，還是pooling表示。
3. 量化drug representation與predictor integration的交互作用。
4. 測試模型對ModelID、drug、scaffold及cancer-type shift的穩健性。
5. 不使用已看過的TCGA response結果進行Round 19選模。
```

---

# 1. 不可違反的實驗原則

## 1.1 既有TCGA五targets不得作Round 19 selection metric

18E結果已經被查看，因此五個TCGA targets不再是Round 19的untouched external test。

Round 19可以在全部架構鎖定後再次報告它們，但必須標記為：

```text
post-hoc exploratory external benchmark
```

不得根據TCGA結果：

```text
修改架構
修改omics mode
修改drug encoder
選擇predictor
修改hidden dimension
修改dropout
修改learning rate
```

若需要真正的最終external confirmation，應另外保留一個未曾查看response結果的新外部資料集。

## 1.2 不混合MACCS與GIN／GINE

禁止：

```text
GIN + MACCS
GINE + MACCS
MACCS residual
MACCS global token added to graph model
MACCS concatenated after atom cross-attention
```

允許：

```text
Graph-only family
或
MACCS-only family
```

這確保：

```text
graph效果
vs
fingerprint效果
```

可以直接歸因。

## 1.3 不重新展開Round 18已排除的超參數

Round 19不再完整搜尋：

```text
P0 historical Transformer
P2 standard Transformer
P3 deeper Transformer
X0 / X1 / X2
pure vs residual完整網格
Transformer層數
Transformer heads
FFN width
response head容量
loss function
omics encoder解凍
```

保留：

```text
P0 = pooled MLP
P1 = compact pooled Transformer
P2 = X3 pure atom cross-attention
```

## 1.4 所有比較使用相同資料族群

所有omics modes與drug families必須使用完全相同的：

```text
eligible response rows
ModelID集合
DRUG_NAME集合
Label
_row_id
CV assignments
sample weights
```

只要任何drug representation缺少某個drug，就必須：

```text
全體實驗共同移除
或
修正該drug資料
```

不能讓不同架構使用不同population。

---

# 2. Round 19研究假設

## H1：GIN node width可能不足

目前公開GIN實作將message-passing width硬編碼為32；`output_dim`主要控制pooling後的graph projection，而不是GIN node hidden本身。

單純把32維投影至64或128維，只是轉換到較大的fusion workspace，不會增加原本drug encoder沒有保留的資訊。

## H2：Pooled架構可能受early-pooling bottleneck限制

Pooled predictor先把所有atom embeddings壓縮成單一graph vector；pooling後再做`32→64`投影，無法恢復被pooling移除的atom資訊。

## H3：Bond-aware message passing可能比單純增加維度有效

目前GIN主路徑只使用：

```text
atom features
edge_index
```

未將bond type、aromaticity、conjugation等edge attributes送進message passing。

因此需要比較：

```text
GIN64
vs
GINE64
```

而不是只測GIN32／64／128。

## H4：同一drug representation在不同predictor中可能有不同效果

P1在attention前只留下單一pooled graph token；X3則保留所有atom tokens。因此node width增加，可能對兩者產生不同效應。

## H5：Omics composition與drug integration存在交互作用

需要分解：

```text
Z
Z + summary
Z + context
Z + summary + context
```

而不能再把91維完整輸入簡稱為「context16」。

## H6：target-informed prototype features可能提高source CV，但不一定提高external transfer

因18E已出現CV/internal與TCGA排序不一致，Round 19需要加入source-only prototype control，檢查target-informed features是否造成domain-specific依賴。

---

# 3. Omics factor定義

## 3.1 核心四種composition

| ID | 顯示名稱                       | 實際輸入                                     | 維度 |
| -- | -------------------------- | ---------------------------------------- | -: |
| O0 | `z_only`                   | Z                                        | 64 |
| O1 | `z_plus_summary`           | Z + own_plus_summary                     | 75 |
| O2 | `z_plus_context16`         | Z + projected prototype context          | 80 |
| O3 | `z_plus_summary_context16` | Z + own_plus_summary + projected context | 91 |

現有alias：

```text
O0 = none
O1 = own_plus_summary
O3 = own_proto_context_projected_16
```

新增：

```text
O2 = own_proto_context_projected_16_no_summary
```

## 3.2 O2實作規則

O2必須使用與O3完全相同的projected context artifact。

不得：

```text
重新fit PCA
重新fitscaler
改變prototype universe
改變projection seed
```

O2只做：

```python
response_features = np.concatenate(
    [latent_z, projected_context16],
    axis=1,
)
```

O3則為：

```python
response_features = np.concatenate(
    [latent_z, own_plus_summary, projected_context16],
    axis=1,
)
```

Metadata：

```json
{
  "mode": "own_proto_context_projected_16_no_summary",
  "display_name": "z_plus_context16",
  "latent_dim": 64,
  "summary_dim": 0,
  "context_dim": 16,
  "response_input_dim": 80,
  "includes_own_plus_summary": false,
  "includes_projected_context": true
}
```

## 3.3 Source-only domain-generalization control

新增一個不進完整主矩陣的控制：

```text
O4 = z_plus_source_proto_features
```

O4只能使用source-side prototype資訊，不使用target prototype。

建議內容：

```text
Z
+ source cosine
+ source L2
+ source init flag
+ source min distance
+ source top1 margin
+ source mean distance
+ source distance std
+ source-only projected context16
```

Source-only context projection只能從：

```text
source prototype
Z − source prototype
```

建立，且projection只在source pretraining population上fit。

O4目的不是取代O0–O3，而是回答：

```text
O3的CV優勢是否依賴target-informed geometry？
```

---

# 4. Drug representation factor定義

## 4.1 必須先重構GIN介面

目前GIN程式把node hidden固定為32，必須拆成：

```text
node_hidden_dim
graph_output_dim
fusion_d_model
```

建立向後相容constructor：

```python
GINConvNet(
    input_dim=78,
    node_hidden_dim=32,
    graph_output_dim=32,
    num_layers=5,
    jk_mode="last",
    pool_type="max",
    dropout=0.1,
    use_batch_norm=True,
)
```

保留legacy alias：

```python
output_dim -> graph_output_dim
```

若legacy呼叫只傳`output_dim`，node hidden仍保持32，確保Round 1–18不被破壞。

Forward新增：

```python
return_dict=True
```

回傳：

```python
{
    "node_embeddings": node_x,
    "batch_index": batch,
    "graph_embedding": graph_x,
    "node_dim": node_x.shape[-1],
    "graph_dim": graph_x.shape[-1],
}
```

預設`return_dict=False`時仍回傳原本graph embedding。

---

## 4.2 D0：Round 18 baseline GIN

```text
encoder_type = gin
node_hidden_dim = 32
graph_output_dim = 32
num_layers = 5
JK = last
pool = max
bond edge features = false
```

---

## 4.3 D1：GIN64 / graph32

```text
encoder_type = gin
node_hidden_dim = 64
graph_output_dim = 32
```

用途：

```text
只增加atom/node representation容量，
保留32維pooled bottleneck。
```

適用：

```text
P0 MLP
P1 pooled Transformer
```

不需要和P2 pure cross-attention分開測graph32／64，因為P2不使用graph output。

---

## 4.4 D2：GIN64 / graph64

```text
encoder_type = gin
node_hidden_dim = 64
graph_output_dim = 64
```

用途：

```text
解除node width與graph output width限制。
```

適用：

```text
P0
P1
P2
```

---

## 4.5 D3：GINE64 / graph64

```text
encoder_type = gine
node_hidden_dim = 64
graph_output_dim = 64
```

Edge attributes至少包括：

```text
bond type：
single
double
triple
aromatic

is_conjugated
is_in_ring
stereo
```

D3與D2必須保持相同：

```text
node width
graph width
layers
JK
pooling
dropout
BatchNorm
optimizer LR
```

因此：

```text
D3 − D2
```

才能解釋為bond-aware chemical content effect。

---

## 4.6 D4：MACCS-only

```text
encoder_type = maccs
graph encoder = none
node tokens = none
```

建立：

```python
class MACCSDrugEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        ...
```

建議結構：

```text
MACCS bits
→ Linear(input_dim, 128)
→ ReLU
→ Dropout(0.1)
→ Linear(128, 64)
→ LayerNorm
```

D4只能配：

```text
P0 MLP
P1 pooled Transformer
```

不能配P2 atom cross-attention，因為MACCS沒有atom-level tokens。

---

# 5. Predictor integration factor定義

## 5.1 P0：Pooled MLP

支援：

```text
D0
D1
D2
D3
D4
```

流程：

```text
omics vector
→ omics adapter

drug vector
→ drug adapter

concat
→ fixed response head
```

固定adapter輸出：

```text
omics_adapter_dim = 64
drug_adapter_dim = 64
fusion representation = 128
```

所有drug families都必須投影成相同64維，避免response head容量不一致。

---

## 5.2 P1：Compact pooled Transformer

支援：

```text
D0
D1
D2
D3
D4
```

固定兩個tokens：

```text
token 0 = omics
token 1 = drug
```

固定Round 18 P1：

```text
d_model = 64
heads = 4
layers = 1
d_ff = 128
dropout = 0.1
```

Graph family：

```text
pooled graph embedding → Linear → drug token
```

MACCS family：

```text
MACCS encoder output → drug token
```

不得新增第三個token。

---

## 5.3 P2：X3 pure atom cross-attention

支援：

```text
D0
D2
D3
```

流程：

```text
Q = omics CLS
K,V = atom node embeddings
```

固定Round 18 X3：

```text
d_model = 128
heads = 4
layers = 2
d_ff = 256
dropout = 0.2
residual_mode = pure
```

禁止：

```text
pooled GIN residual
MACCS token
MACCS residual
graph embedding concatenation
```

---

# 6. 相容組合矩陣

| Drug representation | P0 MLP | P1 pooled Transformer | P2 atom cross-attention |
| ------------------- | :----: | :-------------------: | :---------------------: |
| D0 GIN32 / graph32  |    ✓   |           ✓           |            ✓            |
| D1 GIN64 / graph32  |    ✓   |           ✓           |            —            |
| D2 GIN64 / graph64  |    ✓   |           ✓           |            ✓            |
| D3 GINE64 / graph64 |    ✓   |           ✓           |            ✓            |
| D4 MACCS-only       |    ✓   |           ✓           |            —            |

總相容cells：

```text
3 + 2 + 3 + 3 + 2 = 13
```

Config builder必須以compatibility table生成manifest，不得用完整`drug × predictor`笛卡兒積後再人工刪除。

---

# 7. Round 19階段設計

# Stage 19A：基礎建設與QC

## 19A-1 Git baseline

先確認18E已push：

```bash
git status
git log --oneline -5
git push origin main
git rev-parse HEAD
git rev-parse origin/main
```

必須符合：

```text
HEAD == origin/main
```

輸出：

```text
result/optimization_runs/round19_factorial/metadata/
  round19_baseline_git.json
```

內容：

```json
{
  "round18e_commit": "...",
  "round19_start_commit": "...",
  "working_tree_clean": true,
  "round18e_external_success": false
}
```

## 19A-2 建立O2與O4

新增或修改：

```text
tools/prototype_response_features.py
tools/round19_feature_builder.py
```

輸出：

```text
features/z_only/
features/z_plus_summary/
features/z_plus_context16/
features/z_plus_summary_context16/
features/z_plus_source_proto_features/
```

QC：

```text
O0/O1/O2/O3/O4 ModelID集合一致
O0/O1/O2/O3 row order一致
O2/O3 context projection hash一致
O4不含target prototype欄位
response dims正確
無NaN/Inf
```

## 19A-3 重構GIN並新增GINE

新增：

```text
drugmodels/gineconv.py
tools/round19_graph_features.py
```

不得直接覆蓋Round 18 graph cache。

Cache路徑：

```text
result/optimization_runs/round19_factorial/cache/
  gin_atom78_v1/
  gine_atom78_bond_v1/
```

Metadata：

```json
{
  "encoder_type": "gine",
  "atom_feature_dim": 78,
  "bond_feature_dim": "...",
  "atom_schema_hash": "...",
  "bond_schema_hash": "...",
  "rdkit_version": "...",
  "cache_version": "round19_gine_v1"
}
```

## 19A-4 MACCS-only loader

新增：

```text
tools/round19_drug_features.py
```

功能：

```python
load_maccs_by_drug_name(...)
validate_maccs_coverage(...)
```

必須檢查：

```text
每個eligible DRUG_NAME恰有一個MACCS向量
無重複drug mapping
無缺失bits
無GIN欄位被傳入MACCS model
```

## 19A-5 真實資料smoke

每種drug family至少跑一個batch：

```text
D0 + P0
D1 + P1
D2 + P2
D3 + P2
D4 + P1
```

檢查：

```text
GIN/GINE gradients存在
GINE edge encoder gradients存在
BatchNorm running stats更新
MACCS encoder gradients存在
所有logits finite
所有feature dims符合manifest
```

---

# Stage 19B：Drug Representation × Predictor診斷

## 19B目的

先檢查：

```text
drug representation
×
predictor integration
```

但不能只固定O3，因18E已顯示O3相關CV增益未穩定外推。

因此使用兩個anchor omics modes：

```text
O1 = Z + summary
O3 = Z + summary + context16
```

O1代表較簡單的prototype summary輸入；O3代表Round 18 CV champion使用的完整輸入。

## 19B矩陣

```text
13 compatible drug-predictor cells
× 3 omics modes（O1 / O2 / O3）
× 3 folds
= 117 jobs
```

O1 = Z + summary；O2 = Z + context16（本輪關鍵新模式）；O3 = Z + summary + context16。
O0／O4 不進 19B 主矩陣（O0 留 19C；O4 為 source-only control）。

固定：

```text
split seed = 42
model seed = 101
screening folds = 3
max epochs = 500
patience = 50
early stop start = 30
```

即使部分D0配置曾在Round 18執行，也建議在Round 19 commit下重跑，因GIN介面、dataset與manifest schema已改變。

## 19B必要paired effects

### Node capacity

Pooled：

```text
D1 − D0
```

Atom：

```text
D2/P2 − D0/P2
```

### Graph bottleneck

```text
D2 − D1
```

只在P0與P1比較。

### Bond-aware content

```text
D3 − D2
```

在P0、P1、P2分別配對。

### Predictor integration

固定drug representation與omics：

```text
P1 − P0
P2 − P1
P2 − P0
```

### Fingerprint integration

```text
D4/P1 − D4/P0
```

### Context dependency

固定drug與predictor：

```text
O3 − O1
```

特別檢查：

```text
context effect是否只在P2出現
context effect是否也出現在MACCS
context effect是否隨bond-aware encoder改變
```

## 19B輸出

```text
reports/round19b_architecture_ranking.csv
reports/round19b_node_capacity_effect.csv
reports/round19b_graph_bottleneck_effect.csv
reports/round19b_bond_content_effect.csv
reports/round19b_predictor_integration_effect.csv
reports/round19b_context_dependency.csv
reports/round19b_resource_summary.csv
```

---

# Stage 19C：完整Omics Composition Interaction

## 19C候選選擇

不能取總榜Top-N。

依角色選擇：

```text
C0 = P0 + D0 baseline
C1 = P0 + best nonbaseline pooled drug representation

C2 = P1 + D0 baseline
C3 = P1 + best nonbaseline pooled drug representation

C4 = P2 + D0 baseline
C5 = P2 + best nonbaseline atom drug representation

C6 = best MACCS cell，P0或P1
```

最多7個predictor-drug cells。

若同一cell重複角色，只保留一次。

## 19C核心矩陣

每個cell測：

```text
O0
O1
O2
O3
```

上限：

```text
7 cells × 4 omics × 3 folds
= 84 jobs
```

O1／O2／O3已在19B完成，可直接重用；通常新增主要是O0：

```text
最多約 7 × 1 × 3 = 21 jobs（O0）
```

## 19C source-only control

選擇三個角色：

```text
best pooled MLP
best pooled Transformer
best atom cross-attention
```

各補：

```text
O4 source-only prototype features
```

共：

```text
3 × 3 folds = 9 jobs
```

## 19C shuffled-context負對照

只對最佳atom cross-attention測：

```text
O2 with shuffled context16
O3 with shuffled context16
```

Shuffle規則：

```text
只在training fold內打亂ModelID-context mapping
validation保持真實mapping
每fold使用固定shuffle seed
保持context邊際分布與維度
```

共：

```text
2 × 3 folds = 6 jobs
```

目的：

```text
排除「多16維」或「額外參數」本身造成提升。
```

---

# Stage 19D：Repeated Grouped 5CV Confirmation

## 19D候選

選4–6個角色：

```text
1. best MLP
2. best pooled Transformer
3. best atom cross-attention
4. best MACCS-only
5. best O2 candidate
6. best source-only O4 candidate
```

角色重複可以合併。

## 19D split seeds

使用全新split seeds：

```text
52
62
72
```

不將seed42作主要confirmation，因為seed42已參與19B／19C選擇。

固定：

```text
model seed = 101
5 folds per split seed
max epochs = 1500
patience = 100
early stop start = 50
```

工作量：

```text
4 candidates → 60 jobs
6 candidates → 90 jobs
```

## 19D primary reporting

每個candidate報：

```text
每個split seed的5CV mean
每個split seed的5CV std
跨split seeds的mean of means
跨split seeds的std of means
15-fold paired delta
positive fold count / 15
positive split count / 3
```

不得只把15個fold混合後算一個平均值。

## 19D lock規則

Primary selection：

```text
mean of three 5CV means DrugMacro AUC
```

Tie-breaker：

```text
DrugMacro AUPRC
```

穩健性條件：

```text
至少2/3 split seeds不劣於對應baseline
15個paired folds中至少9個delta ≥ 0
任一split seed不得出現明顯崩潰
```

「明顯崩潰」建議定義為：

```text
相對同predictor D0 baseline
DrugMacro AUC下降 > 0.01
```

---

# Stage 19E：Source-domain Shift Validation

18E TCGA失敗表示，只做ModelID-grouped CV不足以驗證domain transfer。

對Round 19D top 4 candidates增加三種secondary validation。

## 19E-1 Drug-held-out CV

```text
groups = DRUG_NAME
5 folds
```

回答：

```text
能否預測訓練時未出現的drug？
```

## 19E-2 Scaffold-held-out CV

```text
groups = Bemis–Murcko scaffold
5 folds
```

回答：

```text
能否外插至不同chemical scaffold？
```

建立：

```text
tools/round19_scaffold_groups.py
```

規則：

```text
canonicalize SMILES
extract Murcko scaffold
empty scaffold使用明確fallback ID
同一scaffold不可跨fold
```

## 19E-3 Cancer-type-held-out CV

```text
groups = cancer_type
5 folds
```

或在有效癌別數量允許時採leave-one-cancer-type-out。

回答：

```text
是否能跨生物學組織／癌別轉移？
```

這一項比重複ModelID split更接近TCGA所面臨的生物domain shift。

## 19E工作量

Top 4：

```text
drug-held-out：4 × 5 = 20
scaffold-held-out：4 × 5 = 20
cancer-type-held-out：4 × 5 = 20

合計60 jobs
```

## 19E selection使用方式

這些不是新的調參資料，不得再次展開搜尋。

使用guardrail：

```text
最終推薦模型必須：
在三種shift tests中至少兩種不劣於Round 19 baseline
```

不要建立任意加權總分將三種task混在一起。

---

# Stage 19F：External Reporting

## 既有TCGA五targets

可再次推論，但標記：

```text
exploratory post-hoc
```

不得用於：

```text
改lock
改模型
改feature mode
```

報告內容：

```text
每個target
Integrated5
5-fold probability mean
paired bootstrap
與Round 18E相同candidate的變化
```

但不得稱為：

```text
untouched external validation
```

## 新external cohort

若取得未查看response結果的新資料集：

```text
先寫round19_external_lock.json
再執行一次性推論
```

這才可作Round 19真正的final external validation。

---

# Stage 19G：Interpretability

只有graph-based atom cross-attention在19D與19E仍穩定時才執行。

若MACCS或pooled模型最後勝出，不應硬做atom attention作為主要解釋。

對final atom model輸出：

```text
atom_attention_scores.csv
fold_attention_consistency.csv
attention_entropy.csv
top_atom_masking.csv
random_atom_masking.csv
scaffold_level_attention.csv
```

比較：

```text
top-attention masking
vs
matched random masking
```

並分層報：

```text
已知drug
drug-held-out
scaffold-held-out
不同cancer types
```

---

# 8. 建議檔案結構

## Config

```text
config/round19_factorial_settings.json
config/params_round19_stage19b.json
config/params_round19_stage19c.json
config/params_round19_stage19d.json
config/params_round19_shift_validation.json
```

## Models

```text
drugmodels/gineconv.py

tools/round19_drug_encoders.py
tools/round19_fusion_models.py
tools/round19_response_model.py
```

## Data與features

```text
tools/round19_feature_builder.py
tools/round19_drug_features.py
tools/round19_graph_features.py
tools/round19_dataset.py
tools/round19_scaffold_groups.py
```

## Splits與manifests

```text
tools/round19_cv_splits.py
tools/round19_config_builder.py
tools/round19_manifest_validator.py
```

## Training與analysis

```text
step1_finetune_latent_pipeline_round19.py
tools/round19_oom_runner.py
tools/analyze_round19.py
tools/round19_selection_lock.py
```

可以複用Round 18的process-level OOM runner，但Round 19 job metadata必須新增：

```text
omics_id
drug_representation_id
predictor_id
node_hidden_dim
graph_output_dim
edge_feature_schema
split_strategy
split_seed
```

## Shell scripts

```text
tools/run_round19_stage19a_setup_smoke.sh
tools/run_round19_stage19b_drug_predictor_screen.sh
tools/run_round19_stage19c_omics_interaction.sh
tools/run_round19_stage19d_repeated_5cv.sh
tools/run_round19_stage19e_shift_validation.sh
tools/run_round19_stage19f_external_report.sh
tools/run_round19_stage19g_interpretability.sh
```

---

# 9. Config範例

```json
{
  "round": "round19",
  "baseline_round": "round18e",
  "selection_uses_tcga": false,
  "model_seed": 101,
  "screening_split_seed": 42,
  "confirmation_split_seeds": [52, 62, 72],

  "omics_modes": [
    "z_only",
    "z_plus_summary",
    "z_plus_context16",
    "z_plus_summary_context16"
  ],

  "domain_generalization_omics_controls": [
    "z_plus_source_proto_features"
  ],

  "drug_representations": {
    "D0": {
      "type": "gin",
      "node_hidden_dim": 32,
      "graph_output_dim": 32,
      "edge_features": false
    },
    "D1": {
      "type": "gin",
      "node_hidden_dim": 64,
      "graph_output_dim": 32,
      "edge_features": false
    },
    "D2": {
      "type": "gin",
      "node_hidden_dim": 64,
      "graph_output_dim": 64,
      "edge_features": false
    },
    "D3": {
      "type": "gine",
      "node_hidden_dim": 64,
      "graph_output_dim": 64,
      "edge_features": true
    },
    "D4": {
      "type": "maccs",
      "output_dim": 64,
      "graph_encoder": false
    }
  },

  "predictors": {
    "P0": "pooled_mlp",
    "P1": "compact_pooled_transformer",
    "P2": "pure_atom_cross_attention"
  },

  "forbid_hybrid_drug_features": true
}
```

Config validator必須拒絕：

```json
{
  "drug_representations": ["D3", "D4"]
}
```

出現在同一job中。

---

# 10. 必要tests

## Feature tests

```text
tests/test_round19_omics_composition.py
tests/test_round19_source_only_features.py
tests/test_round19_feature_coverage.py
```

檢查：

```text
O2不含summary
O3包含summary+context
O2/O3 context部分完全相同
O4不使用target prototype
```

## Drug encoder tests

```text
tests/test_round19_gin_dimensions.py
tests/test_round19_gine_edge_features.py
tests/test_round19_maccs_encoder.py
tests/test_round19_no_hybrid.py
```

檢查：

```text
D0 node32 graph32
D1 node64 graph32
D2 node64 graph64
D3 edge_attr會影響輸出
D4無graph參數
MACCS不能與GIN/GINE同時建立
```

## Fusion tests

```text
tests/test_round19_fusion_compatibility.py
tests/test_round19_predictor_parameter_groups.py
tests/test_round19_cross_attention_node_dims.py
```

檢查：

```text
P0/P1接受D0–D4
P2只接受D0/D2/D3
所有P0/P1 drug adapter輸出64維
optimizer parameter groups無重複
```

## Split與lock tests

```text
tests/test_round19_repeated_group_cv.py
tests/test_round19_drug_heldout.py
tests/test_round19_scaffold_heldout.py
tests/test_round19_cancer_type_heldout.py
tests/test_round19_selection_lock.py
tests/test_round19_tcga_not_used_for_selection.py
```

最後一項必須掃描selection輸入欄位，確認不存在：

```text
TCGA_AUC
Integrated5
external_auc
internal_test_auc
```

## Pipeline tests

```text
tests/test_round19_data_smoke.py
tests/test_round19_oom_retry.py
tests/test_round19_manifest_counts.py
tests/test_round19_analyzer_effects.py
```

---

# 11. Stage manifest預期數量

## 19B

```text
13 compatible cells
× 3 omics（O1/O2/O3）
× 3 folds
= 117
```

## 19C核心

最多：

```text
7 cells × 4 omics × 3 folds
= 84
```

其中 O1／O2／O3 已在 19B 完成，可直接重用；通常新增主要是 O0。

Controls：

```text
source-only O4 = 9
shuffled context = 6
```

## 19D

```text
4–6 candidates
× 3 split seeds
× 5 folds
= 60–90
```

## 19E

```text
top 4
× 3 shift strategies
× 5 folds
= 60
```

---

# 12. Docker執行順序

## Baseline與tests

```bash
docker exec -w /workspace/DAPL DAPL bash -lc '
pip install -r requirements-round18.txt &&
pytest -q tests/test_round18_*.py tests/test_round19_*.py
'
```

## Stage 19A

```bash
docker exec -w /workspace/DAPL DAPL bash \
  tools/run_round19_stage19a_setup_smoke.sh
```

Go條件：

```text
all tests passed
all feature coverages identical
all five drug-family smokes passed
GINE edge gradients passed
no hybrid assertion passed
```

## Stage 19B pilot

先跑代表性5 jobs：

```text
D0-P0-O1
D1-P1-O3
D2-P2-O3
D3-P2-O1
D4-P1-O3
```

```bash
docker exec -w /workspace/DAPL DAPL bash -lc '
SMOKE_ONLY=0 LIMIT_JOBS=5 \
bash tools/run_round19_stage19b_drug_predictor_screen.sh
'
```

## Stage 19B full

```bash
docker exec -w /workspace/DAPL DAPL bash -lc '
SMOKE_ONLY=0 \
bash tools/run_round19_stage19b_drug_predictor_screen.sh
'
```

## 19B analysis

```bash
docker exec -w /workspace/DAPL DAPL python3 \
  tools/analyze_round19.py \
  --stage 19b \
  --outdir result/optimization_runs/round19_factorial
```

不得在19B直接寫formal lock。

## Stage 19C

```bash
docker exec -w /workspace/DAPL DAPL bash \
  tools/run_round19_stage19c_omics_interaction.sh
```

## Stage 19D lock與正式confirmation

只有19B、19C及controls全部完成後：

```bash
docker exec -w /workspace/DAPL DAPL python3 \
  tools/analyze_round19.py \
  --stage selection \
  --write-lock \
  --outdir result/optimization_runs/round19_factorial
```

接著：

```bash
docker exec -w /workspace/DAPL DAPL bash \
  tools/run_round19_stage19d_repeated_5cv.sh
```

## Stage 19E shift validation

```bash
docker exec -w /workspace/DAPL DAPL bash \
  tools/run_round19_stage19e_shift_validation.sh
```

---

# 13. Analyzer必須回答的問題

## Omics

```text
summary單獨有沒有價值？
context單獨有沒有價值？
summary與context是否互補？
source-only prototype是否更穩？
```

## Drug

```text
GIN64是否優於GIN32？
graph64是否優於graph32？
GINE是否優於同容量GIN？
MACCS是否足以取代graph encoder？
```

## Predictor

```text
相同pooled drug vector下，P1是否優於MLP？
相同graph encoder下，atom cross-attention是否優於pooled Transformer？
```

## Interaction

```text
context增益是否只出現在cross-attention？
bond-aware增益是否只出現在atom-level integration？
MACCS較適合MLP還是Transformer？
較高node width是否只幫助pooled或只幫助atom-level模型？
```

## Generalization

```text
ModelID CV的提升是否保留於：
drug-held-out
scaffold-held-out
cancer-type-held-out
```

---

# 14. 結果判讀規則

## GIN64值得保留

建議條件：

```text
相對GIN32 mean AUC提升 ≥ 0.003
至少2/3 screening folds改善
training–validation gap未明顯增加
```

## GINE值得保留

```text
GINE64相對GIN64：
至少兩種predictor integration改善
或
在scaffold-held-out明顯較穩
```

若只在training AUC改善，不保留。

## MACCS值得保留

MACCS不需要擊敗所有graph模型。

若：

```text
MACCS效能接近最佳graph模型
且參數／時間明顯較低
```

可作efficient baseline或deployment候選。

## O2值得保留

若：

```text
Z + context
接近或勝過
Z + summary + context
```

則summary可能冗餘。

建議「接近」定義：

```text
mean AUC差距 ≤ 0.003
且AUPRC不明顯下降
```

## O4值得保留

若source-only O4在ModelID CV略低，但在：

```text
cancer-type-held-out
或
新的external cohort
```

更穩，則應視為domain-generalization候選，而不是只因source CV較低就淘汰。

## Cross-attention值得保留

不能只看ModelID CV。

至少要求：

```text
Repeated ModelID CV不低於pooled baseline
且
drug/scaffold/cancer-type shift中至少兩項不退步
```

否則Round 18E的問題可能再次出現。

---

# 15. Round 19的Go／No-Go gates

## Gate 19A

GO：

```text
feature與graph schema QC全部通過
```

## Gate 19B

GO：

```text
117/117完成（O1/O2/O3 全 cell）
每個cell有3個有效fold
無資料族群差異
不得在 O2 未完成前選定 best drug／predictor 或寫 lock
```

## Gate 19C

GO：

```text
核心matrix完整
O4完成
shuffled-context完成
```

## Gate 19D

GO：

```text
selection lock不含internal／TCGA metrics
所有候選角色有明確選擇理由
```

## Gate 19E

GO：

```text
19D repeated 5CV完成
不得根據19E結果重新開啟超參數搜尋
```

---

# 16. 建議最終文件

```text
docs/round19_factorial_ide_manual.md
docs/round19_stage19b_report.md
docs/round19_omics_interaction_report.md
docs/round19_repeated_cv_report.md
docs/round19_shift_validation_report.md
docs/round19_final_report.md
```

Final report必須分開標記：

```text
Development selection
Source-domain confirmation
Shift validation
Post-hoc TCGA exploration
Untouched external validation（若有）
```

不得把它們合併成單一「Average AUC」後選模型。

---

# 17. Round 19核心結論模板

Round 19最終應能形成以下類型的結論，而不是只報最佳config：

```text
1. Drug encoder瓶頸主要來自node width／pooling／bond content中的哪一項。

2. Graph與MACCS表示分別適合哪一種predictor integration。

3. Prototype context的價值是獨立訊號、summary補充，
   還是僅在特定cross-attention架構下有效。

4. ModelID CV增益能否保留於drug、scaffold與cancer-type shift。

5. 哪個模型是最高source-domain性能模型，
   哪個模型是最穩健的domain-generalization模型，
   哪個模型是最低成本的efficient model。
```

這三個角色不必由同一個模型擔任。
