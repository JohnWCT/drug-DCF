import json
import pickle
import tempfile
from unittest import mock

import numpy as np
import pandas as pd

from tools.extract_round13_proto_features import build_combined_latent_dicts_own_proto
from tools.optimization_runner import _expand_finetune_combinations
from tools.round16_bruteforce_config_builder import build_round16_configs


def test_config_builder_stage16e_importable():
    import tools.round16_bruteforce_config_builder  # noqa: F401


def test_finetune_combos_for_stage16e():
    combos = _expand_finetune_combinations("config/params_finetune_round16_bruteforce.json")
    assert len(combos) >= 8


def test_own_proto_extraction_combined_dim_gt_latent(tmp_path):
    latent_dim = 8
    n = 4
    ccle = {f"s{i}": np.random.randn(latent_dim).astype(np.float32) for i in range(n)}
    src = np.random.randn(n, latent_dim).astype(np.float32)
    tgt = np.random.randn(n, latent_dim).astype(np.float32)

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    with open(ckpt / "ccle_latent_dict.pkl", "wb") as f:
        pickle.dump(ccle, f)
    with open(ckpt / "tcga_latent_dict.pkl", "wb") as f:
        pickle.dump({}, f)

    cache = tmp_path / "cache"
    cache.mkdir()
    import torch

    torch.save({"prototypes": torch.tensor(src), "initialized": torch.ones(n, dtype=torch.bool)}, cache / "source_anchor_prototypes.pt")
    torch.save({"prototypes": torch.tensor(tgt), "initialized": torch.ones(n, dtype=torch.bool)}, cache / "target_prototypes.pt")
    with open(cache / "cancer_type_mapping.json", "w") as f:
        json.dump({"id_to_name": {i: f"c{i}" for i in range(n)}, "name_to_id": {f"c{i}": i for i in range(n)}}, f)

    outdir = tmp_path / "feat"
    with mock.patch("tools.extract_round13_proto_features.find_latent_paths", return_value=(str(ckpt / "ccle_latent_dict.pkl"), str(ckpt / "tcga_latent_dict.pkl"))), \
         mock.patch("tools.extract_round13_proto_features._load_or_extract_prototypes") as mock_proto, \
         mock.patch("tools.extract_round13_proto_features._load_cancer_maps") as mock_maps:
        mock_proto.return_value = {
            "source_anchor_prototypes": src,
            "target_prototypes": tgt,
            "source_initialized": np.ones(n, dtype=bool),
            "target_initialized": np.ones(n, dtype=bool),
            "cancer_type_mapping": {"id_to_name": {i: f"c{i}" for i in range(n)}, "name_to_id": {f"c{i}": i for i in range(n)}},
        }
        mock_maps.return_value = (pd.Series({f"s{i}": f"c{i % n}" for i in range(n)}), pd.Series(dtype=str))
        meta = build_combined_latent_dicts_own_proto(
            checkpoint_dir=str(ckpt),
            feature_mode="own_proto_delta",
            outdir=str(outdir),
            strict=False,
            proto_cache_dir=str(cache),
        )
    assert meta["response_input_dim"] > meta["latent_dim"]
    assert meta["prototype_feature_mode"] == "own_proto_delta"
