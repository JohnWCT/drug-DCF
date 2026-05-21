"""
AEwC pretraining + GAN alignment pipeline.

This script reuses the same training/evaluation pipeline as `pretrain_VAEwC.py`
but switches the backbone from VAE to AE.

t-SNE output (`tsne_gan_best.png`) uses the shared dual-panel plot
(A: source/target domain, B: cancer type) from `tools.pretrain_tsne`.

Usage example:
python pretrain_AEwC.py \
  --config config/params_grid_quick3_vaewc.json \
  --outfolder result/pretrain_aewc \
  --target_domain tcga
"""

import torch
import pretrain_VAEwC as core
from tools.model_opt import AE

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. No GPU detected.")


# Reuse the full VAEwC pipeline while switching model backbone/type.
core.MODEL_BACKBONE = AE
core.MODEL_TYPE_NAME = "AE"


if __name__ == "__main__":
    core.main()
