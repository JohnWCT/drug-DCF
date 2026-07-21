# Round 20 / Round 21 Scientific Audit (Q1–Q8)

This document answers the eight priority questions that must be resolved **before** any new architecture search.

Naming policy (mandatory):

| Role | Canonical name | Do not call it |
|------|----------------|----------------|
| Round 20 locked predictor | **BioCDA-Predictive** | bare “BioCDA” |
| Round 21 XA candidates | **BioCDA-XA-Candidate** | bare “BioCDA” / locked BioCDA |

Artifacts:

- `reports/round20_round21_architecture_diff.json`
- `reports/round21_rejection_root_cause.json`
- `reports/biocda_predictive_e3_architecture_spec.json`

---

## Q1 — Are Round 20 E3 and Round 21 XA the same architecture?

**Answer: No.**

Machine-readable diff: `reports/round20_round21_architecture_diff.json`  
`verdict.same_architecture = false`.

| Axis | BioCDA-Predictive (R20 E3/B_E3) | BioCDA-XA-Candidate (R21 M2) |
|------|--------------------------------|-----------------------------|
| Model class | `GINConvNet` + `AdapterMLPFusion` + `Round18ResponseHead` | `BioCDA` + `SampleAtomCrossAttention` + `BioCDAResponseHead` |
| architecture_version | none (`pooled_mlp` / `B_E3`) | `biocda-xa-v1` |
| Input features | Z64+C32 → **96** | Z64+C32 → **96** (same store) |
| Sample projector | omics/drug adapters → concat 128 | LN(Z)+LN(C)→Linear+LN+GELU → 96 |
| GIN | 5×32, jk=last, BN, dropout **0.1** | 5×32, jk=last, BN, dropout **0.2** |
| Pooling path | **global max** → graph emb | **no pool in predictor**; atom nodes → K/V |
| Cross-attention | **absent** | **1 layer**, 4 heads, dim 64 |
| Response head input | **128** | **160** = 96+64 |
| state_dict | `encoder.*` / `fusion.*` / `head.*` | `drug_encoder.gin.*` / `cross_attention.*` / `response_head.*` |
| Forward extras | logits/probs only | + atom attention, mask, ptr |

**Conclusion:** Do not share one “BioCDA” name. Predictive claims attach to **BioCDA-Predictive**. Interpretability claims attach only to a future **passed** XA model, currently **BioCDA-XA-Candidate (REJECTED)**.

---

## Q2 — Root cause of Round 21 REJECTED

**Primary root cause: `performance_failure`.**

| Category | Status | Implication |
|----------|--------|-------------|
| performance_failure | **CONFIRMED** | Fix XA vs pooled gap before search |
| external_domain_failure | **NOT_APPLICABLE** | Rejection is in-protocol development unseen-drug, not external GDSC/TCGA fail |
| attention_health_failure | **NOT_FAILED** (PASS) | Not uniform/collapse-driven rejection |
| context_utilization_failure | **NOT_FAILED** (PASS) | Context changes attention/prediction |
| reproduction_failure | **NOT_FAILED** | Audits/tests/9 runs OK |
| data_contract_failure | **NOT_FAILED** | Shapes/masks/splits OK |

Machine-readable: `reports/round21_rejection_root_cause.json`.

Evidence (DrugMacro AUC mean): M0 **0.746**, M1 **0.714**, M2 **0.709**; Δ(M2−M0) ≈ **−0.037** (guardrail ≥ −0.005).

**Do not open architecture search** until this performance gap is addressed (Outcome 3 path: residual / distillation / limited GIN last-block FT).

---

## Q3 — Does C32 independently help?

### What is already established (Round 20 Stage 20A)

Controlled comparison under **same** architecture (BioCDA-Predictive / E3), same D0 encoder family, same drug-held-out protocol:

| Setting | Mean DrugMacro AUC |
|---------|-------------------|
| Base64 + PCA16 (C16) | 0.7434 |
| Base64 + PCA32 (C32) | 0.7509 |
| Δ(C32−C16) | **+0.00745** (LOCKED selection) |

Source: `stage20a_dimension_decision.json` (`reason: stable_improvement`).

So for **BioCDA-Predictive**, C32 improves unseen-drug prediction vs C16 under fixed architecture.

### What is **not** yet established

1. **C32 vs Base64-only (no context)** under identical splits — Stage20A compared C16 vs C32, not Z64 alone.
2. **Whether C32 changes atom attention** — BioCDA-Predictive has **no atom attention**. Attention effects of C32 can only be measured on BioCDA-XA-Candidate.
3. Round 21 M2 vs M1 (ZC vs Z) on the *same* XA architecture: ΔAUC ≈ **−0.005** (M2 slightly worse/similar) while context shuffle **does** change attention (L1≈0.32). That pattern currently suggests: **context affects attention distributions, but does not improve XA prediction** under Phase-A freeze.

### Claim discipline

| Claim | Allowed now? |
|-------|----------------|
| C32 improves BioCDA-Predictive vs C16 | **Yes** (Stage20A) |
| “biological-context-guided atom attention” | **No** until XA predicts competitively **and** attention diagnostics (Q5–Q6) pass |
| C32 only changes prediction, not attention | N/A for Predictive (no attention); for XA, attention **does** change but prediction does not improve |

Required experiment (not done this round): Z64-only vs Z64+C32 on **fixed BioCDA-XA-Candidate**, shared splits/seeds/budget, report both ΔAUC and attention JS/L1.

---

## Q4 — Expand E3 beyond the experiment label

**E3 is a label, not a method name.** Use:

> **BioCDA-Predictive** = D0 GIN(5×32, jk=last, BN, dropout 0.1, global_max) + AdapterMLPFusion(adapter 64) + Round18ResponseHead(128→1), omics input Z64+C32=96, **no cross-attention**.

Full machine-readable expansion: `reports/biocda_predictive_e3_architecture_spec.json`.

Papers/README/manifests should cite that file (or an equivalent table), not “E3” alone.

---

## Q5 — Is attention actually used for prediction / interpretability-ready?

### Contract-level (done)

- Attended drug representation enters `BioCDAResponseHead` (no pooled bypass in M1/M2).
- Raw vs dropout attention separated; padding=0; valid sums≈1.

### Diagnostic-level (partially done on Round 21)

| Check | Round 21 status |
|-------|-----------------|
| Not near-uniform | PASS (mean norm entropy ≈ 0.43) |
| Not single-atom collapse | PASS (mean max attn ≈ 0.56) |
| Different samples → different attention | Partially (query sensitivity CSV; L1 small on synthetic smoke path) |
| Context shuffle changes attention | PASS (L1≈0.32) |
| Same-drug / same-cancer attention similarity | **NOT DONE** |
| Top-attention masking vs random masking | **NOT DONE** |

**Rule:** Even after a future LOCKED XA, claims of interpretability require the unfinished rows above.

---

## Q6 — Within-cancer attention consistency (biological hypothesis)

**Status: NOT VALIDATED.**

Hypothesis:

> For a fixed drug, patients of the same cancer type should have more similar atom-attention than different cancer types.

Requirements (Round 22+, only if XA is predictive-competitive):

1. Compare **within the same drug** only (atom indices not aligned across molecules).
2. Report within-cancer vs between-cancer similarity (JS / cosine / top-k Jaccard).
3. Exclude near-uniform attention (else similarity is spurious).

Round 21 explicitly deferred cancer-type aggregation and TCGA heatmaps.

---

## Q7 — Does REJECTED mean cross-dataset domain shift?

**Answer: No — not as currently observed.**

| Fact | Detail |
|------|--------|
| Round 21 selection data | Round19 `development_rows.csv` (GDSC2 cell-line responses) |
| Round 20 selection data | Same development scope (drug-held-out) |
| TCGA in selection | **false** for both |
| What failed | BioCDA-XA-Candidate **vs BioCDA-Predictive-style M0** on development unseen-drug |
| Protocol mismatch | R20 seeds 52/62/72 × 5-fold; R21 seeds 17/29/43 × 70/15/15 — **not identical splits** |

Therefore REJECTED ≠ “failed on external GDSC domain.” It is an **architecture/performance** failure under an unseen-drug development protocol.

If a future study evaluates a second label set / normalization / gene map, then decompose:

drug overlap, gene mapping, omics normalization, endpoint, binary threshold, cell-line mix, cancer coverage, molecule standardization — **before** claiming external generalization.

---

## Q8 — Scope of TCGA inference claims

Correct Round 20/21 policy: **TCGA not used for response-supervised model selection.**

If TCGA inference is run **after** locking BioCDA-Predictive (or a future LOCKED XA):

| May support | Must not claim without independent clinical labels |
|-------------|----------------|
| Model-predicted response probabilities | Clinical treatment efficacy |
| Patient-conditioned atom attention (XA only) | Causal drug mechanism |
| Cancer-specific attention/prediction patterns | Validated patient response |

---

## Recommended next actions (ordered)

1. Keep **BioCDA-Predictive** as the public predictive model name; stop calling Round 21 results “BioCDA” without `-XA-Candidate`.
2. Treat Round 21 REJECTED as **performance_failure** only; no architecture search yet.
3. Design a focused XA recovery round (residual to pool / last-block GIN FT / attention distillation) with **identical splits to Round 21**.
4. Add Z64 vs Z64+C32 XA ablation for Q3 attention+prediction joint reporting.
5. Only after predictive parity: Q5 masking faithfulness + Q6 within-cancer attention tests.
6. TCGA remains post-lock descriptive analysis under Q8 claim limits.
