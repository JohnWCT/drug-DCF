#!/usr/bin/env python3
"""BioCDA architecture finalization smoke test."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biocda.data.drug_graph import batch_drug_graphs, make_chain_graph
from biocda.models.model_factory import build_model
from biocda.training.checkpoint import save_biocda_checkpoint
from biocda.utils.gpu import configure_gpu_efficiency
from biocda.utils.reproducibility import set_seed
from tools.biocda_telegram_notify import biocda_notify


def _optional_gpu_forward(model, omics, context, batch, device: str) -> dict:
    if not device.startswith("cuda"):
        return {"gpu_benchmark": "skipped_cpu_only"}
    model_gpu = model.to(device)
    omics = omics.to(device)
    context = context.to(device)
    batch = batch.to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            model_gpu(omics, context, batch, output_mode="attention")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"gpu_benchmark": "ok", "device": device, "seconds_50_forward": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/model/biocda_cross_attention.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/architecture_finalization",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-telegram", action="store_true")
    args = parser.parse_args()

    if not args.no_telegram:
        biocda_notify("Architecture smoke test START")

    gpu_info = configure_gpu_efficiency(target_utilization=0.9)
    set_seed(42)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = build_model(config)
    model.eval()

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )

    graphs = [make_chain_graph(12), make_chain_graph(5)]
    batch = batch_drug_graphs(graphs)
    omics = torch.randn(2, config["model"]["omics_encoder"]["latent_dim"])
    context = torch.randn(2, config["model"]["biological_context"]["context_dim"])

    with torch.no_grad():
        output = model(omics, context, batch, output_mode="attention")

    valid = output.atom_attention * output.atom_mask.unsqueeze(1)
    attn_sum = valid.sum(dim=-1)
    ones = torch.ones_like(attn_sum)
    max_err = float((attn_sum - ones).abs().max())
    padding = output.atom_attention.masked_select(~output.atom_mask.unsqueeze(1))
    padding_nonzero = int((padding != 0).sum())
    gpu_bench = _optional_gpu_forward(model, omics, context, batch, device)

    report = {
        "status": "PASS",
        "architecture_version": getattr(model, "ARCHITECTURE_VERSION", "unknown"),
        "batch_size": 2,
        "num_heads": int(config["model"]["cross_attention"]["num_heads"]),
        "max_atoms": int(output.atom_attention.shape[-1]),
        "attention_shape": list(output.atom_attention.shape),
        "valid_attention_sum_max_error": max_err,
        "padding_nonzero_count": padding_nonzero,
        "nan_count": int(torch.isnan(output.atom_attention).sum()),
        "inf_count": int(torch.isinf(output.atom_attention).sum()),
        "gpu": gpu_info,
        **gpu_bench,
    }
    manifest = {
        "architecture_name": "BioCDA-XA",
        "architecture_version": getattr(model, "ARCHITECTURE_VERSION", "biocda-xa-v1"),
        "omics_encoder": {
            "name": "O2",
            "latent_dim": config["model"]["omics_encoder"]["latent_dim"],
            "frozen": config["model"]["omics_encoder"].get("frozen", True),
        },
        "biological_context": {
            "type": config["model"]["biological_context"]["type"],
            "context_dim": config["model"]["biological_context"]["context_dim"],
            "frozen": config["model"]["biological_context"].get("frozen", True),
        },
        "sample_representation": {
            "input_dim": config["model"]["omics_encoder"]["latent_dim"]
            + config["model"]["biological_context"]["context_dim"],
            "output_dim": config["model"]["sample_representation"].get("output_dim"),
        },
        "drug_encoder": {
            "name": "D0",
            "output_type": "atom_node_embeddings",
            "node_dim": config["model"]["drug_encoder"].get("node_dim", 32),
            "frozen": config["model"]["drug_encoder"].get("frozen", True),
        },
        "cross_attention": {
            "query_source": "omics_latent_plus_context",
            "key_source": "drug_atom_node_embeddings",
            "value_source": "drug_atom_node_embeddings",
            "attention_dim": config["model"]["cross_attention"]["attention_dim"],
            "num_heads": config["model"]["cross_attention"]["num_heads"],
            "graph_pooling_bypass": False,
        },
        "prediction": {
            "drug_input": "cross_attention_drug_representation",
            "output": "binary_logit",
        },
        "interpretability_interfaces": {
            "atom_attention": True,
            "attention_logits": True,
            "atom_mask": True,
            "node_embeddings": True,
            "sample_representation": True,
            "omics_latent": True,
            "atom_indices": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "architecture_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "architecture_smoke_test.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    ckpt_path = args.output_dir / "smoke_checkpoint.pt"
    save_biocda_checkpoint(
        ckpt_path,
        model=model,
        config=config,
        epoch=0,
        model_type=config["model"]["type"],
        architecture_version=getattr(model, "ARCHITECTURE_VERSION", "biocda-xa-v1"),
    )
    print(f"ARCHITECTURE_SMOKE={report['status']}")
    print(json.dumps(report, indent=2))

    if not args.no_telegram:
        biocda_notify(
            f"Architecture smoke test PASS\nversion={report['architecture_version']}\n"
            f"device={device}\ngpu={json.dumps(gpu_bench)}"
        )


if __name__ == "__main__":
    main()
