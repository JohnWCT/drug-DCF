# Round 19 Final Report

## 最終狀態

Round 19F、19G 與本機 19H reproducibility archive 已完成。模型採 immutable
scenario-aware multi-role policy，沒有 single champion。

- Round 19F post-hoc：540/540 jobs，0 failed。
- Round 19G interpretability：1,801/1,801 jobs，0 failed，230 cases。
- Routing：100% match。
- Interpretability verdict：`PARTIALLY_SUPPORTED`。
- Final role lock 未修改；SHA-256：
  `e45df23826b31822e986517311969a5b7a540eed659c1f20e847e1c7b29e24ff`。

## Reproducibility

正式 19G lock 綁定本機 commit `282895f7d2fe7919cb31efc6a383eb8ef9496481`，
不要求 remote sync，也未執行 push、pull、fetch、merge 或 rebase。Docker 環境：
Python 3.8.13、Torch 1.13.1+cu117、PyG 2.3.1、RDKit 2023.3.1b1，
RTX 6000 Ada 49,140 MiB。

19H 產物包含 reproducibility audit、artifact manifest、portable symlink mapping、
model card 與 dataset card。封存策略為 plan-only；沒有複製 checkpoints、改寫 symlink、
清除檔案或修改 final role lock。

## 解讀限制

Round 19G 支持模型確實使用 drug 與 omics/context 訊息，且高排名 perturbation 通常比
matched random 造成更大輸出變化；但 attention 的跨 member 穩定度不足以支持唯一或因果
解釋。因此最終 claim 限於 post-lock model-behavior evidence。
