# Round 19 Stage 19F Report — Role Proposal Review

**Date:** 2026-07-15  
**Status:** **ALL_DONE — ROLE_LOCKED + POSTHOC_COMPLETE**  
**Gate:** `EXPLORATORY_POSTHOC_COMPLETE`  
**Single champion:** `null`  
**Selection inputs:** Round 19D + 19E only  
**Internal / TCGA selection:** **PROHIBITED / NOT USED**

---

## 1. Scope

此 gate 只完成：

- Full-precision role selector 與預註冊 guardrails。
- Deterministic scenario-aware routing policy。
- 15-member ensemble completeness contract。
- 六個 unique role candidates 的 90-checkpoint inventory。
- Docker smoke 與 Telegram start/done 通知。

已於人工核准後建立 immutable `round19_final_role_lock.json`，並完成 internal／TCGA exploratory post-hoc inference。

---

## 2. Role proposal

`reports/round19_final_role_proposal.json`

| Role | Proposed candidate | Source definition | Evidence |
|------|--------------------|-------------------|----------|
| Historical anchor | E0 | F0 D0×P0×O1 | fixed |
| Source-performance champion | **F2** | D0×P2×O3 | 19D mean-of-means 0.620083 |
| Parsimonious context | **F1** | D0×P2×O2 | F2−F1 = 0.000703 ≤ 0.003 |
| Cancer-shift specialist | **E1** | F1 / O2 | E2−E1 = 0.001752 ≤ 0.003 → prefer O2 |
| Chemical-shift specialist | **E3** | F3 D0×P0×O2 | maximin Δ vs E0 = +0.001004 |
| Source-only domain candidate | **E4** | F4 D3×P2×O4 | gate passed |
| Efficient model | **E5** | F5 D4×P1×O2 | 2 shifts non-worse/pass + time/VRAM gates |
| General recommended | **E3** | F3 D0×P0×O2 | all three shifts pass/non-worse; no major fail |

`single_champion = null`：角色政策不宣稱單一架構普遍最佳。

---

## 3. Selection logic

### Source / parsimony

- F2 是 19D unrounded mean-of-three-5CV-means 最高者。
- F1 與 F2 source 差 0.000703。
- Cancer shift 中 E1 只落後 E2 0.001752。
- 兩者均在 0.003 practical-equivalence margin 內，因此 F1 保留 parsimonious role。

### Cancer specialist

E1/E2 的 cancer AUC 分別為 0.580611 / 0.582364。因差距 ≤0.003，依預註冊 tie rule 選 E1/O2，而非因 F2 是 source winner 就自動選 E2。

### Chemical maximin

| Candidate | worst drug/scaffold Δ vs E0 |
|-----------|-----------------------------|
| E3 | **+0.001004** |
| E0 | 0.000000 |
| E5 | −0.003615 |
| E4 | −0.006823 |
| E2 | −0.008289 |
| E1 | −0.015342（major fail） |

E3 由 worst-case 規則勝出，沒有建立任意跨 shift 加權分數。

### General model

Full-precision eligibility candidates 為 E0、E3。E3 的 chemical maximin 較高，且 cancer shift 為 PASS，因此 proposed general model 為 E3。  
General role 在 policy 中仍允許為 null；本次並非為填滿角色而硬選。

---

## 4. Scenario-aware routing

| Novelty class | Role |
|---------------|------|
| unseen drug | chemical_shift_specialist |
| unseen scaffold | chemical_shift_specialist |
| unseen cancer type | cancer_shift_specialist |
| source-like | source_performance_champion |
| metadata unknown | chemical_shift_specialist（low-confidence fallback） |

路由只使用預先可知 metadata，不使用預測信心，不是 learned mixture-of-experts。

---

## 5. 15-member readiness

`reports/round19_stage19f_checkpoint_inventory.csv`

| Item | Value |
|------|-------|
| Unique source candidates | 6 |
| Required grid | seeds 52/62/72 × folds 0–4 |
| Members per candidate | 15 |
| Total checkpoints | **90/90** |
| Missing checkpoints | 0 |
| Internal manifest created | no |
| TCGA manifest created | no |

Ensemble contract 使用 `(split_seed, fold_id)`，禁止只檢查重複的 fold ID，也禁止 best-fold selection。

---

## 6. Smoke

Docker command：

```bash
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL DAPL \
  bash tools/run_round19_stage19f_role_lock.sh --smoke-only
```

結果：

- Selector / guardrail / maximin / no-external-selection
- Novelty / routing
- 15-member completeness、missing/duplicate/identity drift guards
- 90-checkpoint inventory

**10/10 tests passed**；Telegram start/done 已執行。

---

## 7. Artefacts

```text
config/round19_stage19f_role_policy.json
config/round19_stage19f_inference_settings.json
reports/round19_final_role_proposal.json
reports/round19_stage19f_checkpoint_inventory.csv
reports/round19_stage19f_checkpoint_inventory_summary.json
tools/round19_stage19f_role_selector.py
tools/round19_novelty_classifier.py
tools/round19_deployment_policy.py
tools/round19_stage19f_ensemble.py
tools/round19_stage19f_manifest_builder.py
tools/run_round19_stage19f_role_lock.sh
```

---

## 8. Gate verdict

| Gate | Verdict |
|------|---------|
| Proposal completeness | **GO** |
| Selector did not use internal/TCGA | **GO** |
| Scenario-aware policy smoke | **GO** |
| 15-member checkpoint inventory | **GO** |
| Human proposal review | **APPROVED** |
| Final role lock | **LOCKED** |
| Internal/TCGA post-hoc | **COMPLETE（540/540，0 failed）** |

Proposal 已人工核准並鎖定；post-hoc 結果不得改變任何 locked role。

---

## 9. Exploratory post-hoc results

### Execution completeness

- Internal：6 candidates × 15 members = **90/90**
- TCGA：6 candidates × 5 targets × 15 members = **450/450**
- Total：**540/540，0 failed**
- GPU：RTX 6000 Ada，12 parallel slots，90% VRAM packing policy
- Ensemble：每個 candidate/target/eval row 嚴格使用 15-member probability mean
- Paired bootstrap：**120/120 computed**，每項 2,000 replicates

### Internal test

| Candidate | DrugMacro AUC | DrugMacro AUPRC |
|-----------|---------------|-----------------|
| F0 historical anchor | 0.595128 | 0.473176 |
| F1 primary O2 | 0.619614 | 0.501676 |
| F2 full omics O3 | 0.608804 | 0.475468 |
| F3 best pooled O2 | 0.600811 | 0.476302 |
| F4 source-only O4 | 0.610798 | 0.472159 |
| F5 MACCS efficient | 0.633049 | 0.510504 |

相對 F0 的 2,000-replicate paired-bootstrap DrugMacro AUC 95% CI 均跨 0；因此這些數值只作描述，不形成新的模型選擇結論。

### TCGA Integrated5

五個 target 先各自計算 DrugMacro metric，再做等權平均：

| Candidate | Integrated5 AUC | Integrated5 AUPRC |
|-----------|-----------------|-------------------|
| F0 historical anchor | 0.517395 | 0.613151 |
| F1 primary O2 | 0.486176 | 0.599116 |
| F2 full omics O3 | 0.483512 | 0.603685 |
| F3 best pooled O2 | 0.465508 | 0.586219 |
| F4 source-only O4 | 0.501717 | 0.598290 |
| F5 MACCS efficient | 0.507794 | 0.616659 |

這些 external 結果為 `exploratory_post_hoc`；不回寫 proposal、不重跑 selector，也不修改 scenario-aware routing。

### Result artefacts

```text
reports/round19_stage19f_posthoc/round19f_15member_ensemble_predictions.csv
reports/round19_stage19f_posthoc/round19f_internal_candidate_metrics.csv
reports/round19_stage19f_posthoc/round19f_tcga_per_target_metrics.csv
reports/round19_stage19f_posthoc/round19f_integrated5_equal_target_mean.csv
reports/round19_stage19f_posthoc/round19f_paired_bootstrap_deltas.csv
reports/round19_stage19f_posthoc/round19f_role_alias_view.csv
reports/round19_stage19f_posthoc/round19f_exploratory_posthoc_summary.json
```
