# Round 13 — Prototype-distance Response Features

**狀態：** DONE

## 分數

| 參考 | Avg TCGA |
|------|----------|
| **R13 最佳 r13_exp_008_own_plus_summary** | **0.6112** |
| R12 exp_037 | 0.5972 |
| Stretch goal | 0.6200（未達） |

vs R12：**+0.0141**

## Feature mode

- **最佳：** `own_plus_summary`（peak on exp_008）
- `none` z-only 對 exp_035 仍強（0.6059）
- 6 個 source model 中 **4/6** proto 優於 z-only

## 結論

Prototype response features 有效，為 downstream finetune **全專案峰值**。效益具 model-specific 性。**→ 進入 Round 14 VICReg stabilizer。**
