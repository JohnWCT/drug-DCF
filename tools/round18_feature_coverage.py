"""Preflight: ensure Round 18 omics feature dirs share identical ModelID coverage."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_eligible_data import validate_feature_metadata


def _latent_keys(feature_dir: str):
    path = Path(feature_dir) / "ccle_latent_proto.pkl"
    if not path.is_file():
        raise FileNotFoundError("Missing latent pickle: {}".format(path))
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return set(map(str, raw.keys()))


def feature_dir_for_omics(settings: dict, omics_mode: str) -> str:
    root = settings.get("feature_root", "result/optimization_runs/round17r_18class/features")
    model_key = settings.get("feature_model_key", "r13_exp_008")
    mode = "none" if omics_mode in {"none", "z-only", "z_only"} else omics_mode
    return str(Path(root) / model_key / mode)


def assert_round18_feature_coverage(
    settings: dict,
    omics_modes=None,
    reference_omics: str = "own_plus_summary",
) -> Dict[str, Any]:
    """
    Fail hard if ModelID sets differ across omics feature dirs used by Round 18B.
    response_input_dim may differ (64/75/91); ModelID membership must match.
    """
    modes = list(omics_modes or settings.get("omics_modes") or [])
    if not modes:
        raise ValueError("No omics_modes to check")
    if reference_omics not in modes:
        reference_omics = modes[0]

    per_mode = {}
    keysets = {}
    for mode in modes:
        fdir = feature_dir_for_omics(settings, mode)
        meta = validate_feature_metadata(fdir)
        keys = _latent_keys(fdir)
        keysets[mode] = keys
        per_mode[mode] = {
            "feature_dir": fdir,
            "n_model_ids": len(keys),
            "response_input_dim": meta.get("response_input_dim"),
            "n_trainable_cancer_types": meta.get("n_trainable_cancer_types"),
            "uses_legacy_28class_cache": meta.get("uses_legacy_28class_cache"),
            "prototype_class_source": meta.get("prototype_class_source"),
        }

    ref_keys = keysets[reference_omics]
    mismatches = []
    for mode, keys in keysets.items():
        if keys != ref_keys:
            mismatches.append(
                {
                    "omics_mode": mode,
                    "n_model_ids": len(keys),
                    "n_ref": len(ref_keys),
                    "only_in_reference_sample": sorted(ref_keys - keys)[:10],
                    "only_in_mode_sample": sorted(keys - ref_keys)[:10],
                }
            )
    if mismatches:
        raise AssertionError(
            "Round 18 feature ModelID coverage mismatch: "
            + json.dumps(mismatches, indent=2)
        )

    for mode, info in per_mode.items():
        if int(info.get("n_trainable_cancer_types") or -1) != 18:
            raise AssertionError("{}: n_trainable_cancer_types != 18 ({})".format(mode, info))
        if info.get("uses_legacy_28class_cache") is True:
            raise AssertionError("{}: uses_legacy_28class_cache=true".format(mode))
        src = info.get("prototype_class_source")
        if src is not None and src != "checkpoint_metadata":
            raise AssertionError("{}: unexpected prototype_class_source={}".format(mode, src))

    expected_dims = {
        "none": 64,
        "own_plus_summary": 75,
        "own_proto_context_projected_16": 91,
    }
    for mode, info in per_mode.items():
        exp = expected_dims.get(mode)
        if exp is not None and int(info["response_input_dim"]) != exp:
            raise AssertionError(
                "{}: response_input_dim={} expected {}".format(
                    mode, info["response_input_dim"], exp
                )
            )

    return {
        "ok": True,
        "reference_omics": reference_omics,
        "n_model_ids": len(ref_keys),
        "omics_modes": modes,
        "per_mode": per_mode,
        "model_id_sets_identical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 18 omics feature coverage preflight")
    parser.add_argument("--settings", default="config/round18_architecture_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round18_architecture")
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    report = assert_round18_feature_coverage(settings)
    out = Path(args.outdir) / "data" / "round18_feature_coverage_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("wrote:", out)


if __name__ == "__main__":
    main()
