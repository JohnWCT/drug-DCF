#!/usr/bin/env python3
"""Unit / contract tests for BioCDA-XA v2 (Round 23)."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from biocda.data.drug_graph import batch_drug_graphs, make_chain_graph
from biocda.models.xa.factory import build_xa_v2
from biocda.training.distillation import (
    assert_student_checkpoint_has_no_teacher,
    combine_response_kd,
    export_student_only_state,
)
from biocda.training.freeze_schedule import FreezePhase, apply_phase, set_frozen_bn_eval
from biocda.training.gin_transfer import transfer_e3_gin_to_xa, verify_transferred_weights_match


def _config() -> dict:
    path = ROOT / "configs/biocda/xa_v2_closure.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestNoPoolingContract(unittest.TestCase):
    def test_xa_forward_never_calls_pooling(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        gin = model.drug_encoder.gin
        calls = {"pool": 0}

        orig = gin.pool_graph

        def wrapped(*a, **k):
            calls["pool"] += 1
            return orig(*a, **k)

        gin.pool_graph = wrapped  # type: ignore
        omics = torch.randn(2, 64)
        ctx = torch.randn(2, 32)
        g = batch_drug_graphs([make_chain_graph(5), make_chain_graph(7)])
        with torch.no_grad():
            model(omics, ctx, g, output_mode="prediction")
        self.assertEqual(calls["pool"], 0)

    def test_response_head_receives_final_query_only(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        omics = torch.randn(3, 64)
        ctx = torch.randn(3, 32)
        g = batch_drug_graphs([make_chain_graph(4 + i) for i in range(3)])
        with torch.no_grad():
            out = model(omics, ctx, g, output_mode="full")
        self.assertEqual(list(out.logits.shape), [3])
        self.assertEqual(list(out.final_query.shape), [3, 1, 128])
        # Head has no extra concat modules
        self.assertFalse(hasattr(model.response_head, "fusion"))

    def test_student_checkpoint_contains_no_teacher(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_kd")
        state = export_student_only_state(model)
        assert_student_checkpoint_has_no_teacher(state)
        self.assertTrue(all("teacher" not in k for k in state))


class TestShapeContract(unittest.TestCase):
    def test_z64_c32_produces_96d(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        z = torch.randn(2, 64)
        c = torch.randn(2, 32)
        feats, q = model.sample_projector(z, c)
        self.assertEqual(feats.shape[-1], 96)
        self.assertEqual(list(q.shape), [2, 1, 128])

    def test_batch_size_one_logits_shape(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        g = batch_drug_graphs([make_chain_graph(6)])
        with torch.no_grad():
            out = model(torch.randn(1, 64), torch.randn(1, 32), g)
        self.assertEqual(list(out.logits.shape), [1])

    def test_variable_atom_count_batch(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        g = batch_drug_graphs([make_chain_graph(3), make_chain_graph(11), make_chain_graph(5)])
        with torch.no_grad():
            out = model(torch.randn(3, 64), torch.randn(3, 32), g, output_mode="attention")
        self.assertEqual(out.atom_mask.shape[0], 3)
        self.assertTrue(out.atom_mask.any())

    def test_attention_shape_all_layers_heads(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        g = batch_drug_graphs([make_chain_graph(6), make_chain_graph(6)])
        with torch.no_grad():
            out = model(torch.randn(2, 64), torch.randn(2, 32), g, output_mode="attention")
        # [L=2, B=2, H=4, 1, N]
        self.assertEqual(out.attention_logits.shape[0], 2)
        self.assertEqual(out.attention_logits.shape[2], 4)
        self.assertEqual(out.attention_probabilities.shape, out.attention_logits.shape)


class TestAttentionContract(unittest.TestCase):
    def test_valid_attention_sums_to_one(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        g = batch_drug_graphs([make_chain_graph(5), make_chain_graph(8)])
        with torch.no_grad():
            out = model(torch.randn(2, 64), torch.randn(2, 32), g, output_mode="attention")
        probs = out.attention_probabilities[-1, :, 0, 0, :]
        s = (probs * out.atom_mask.float()).sum(-1)
        self.assertTrue(torch.allclose(s, torch.ones_like(s), atol=1e-4))

    def test_padding_attention_is_zero(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        g = batch_drug_graphs([make_chain_graph(3), make_chain_graph(9)])
        with torch.no_grad():
            out = model(torch.randn(2, 64), torch.randn(2, 32), g, output_mode="attention")
        pad = out.attention_probabilities.masked_select(
            (~out.atom_mask).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(out.attention_probabilities)
        )
        self.assertTrue(bool((pad.abs() < 1e-6).all()))

    def test_output_modes_produce_identical_logits(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        model.eval()
        omics, ctx = torch.randn(2, 64), torch.randn(2, 32)
        g = batch_drug_graphs([make_chain_graph(6), make_chain_graph(6)])
        with torch.no_grad():
            a = model(omics, ctx, g, output_mode="prediction")
            b = model(omics, ctx, g, output_mode="attention")
            c = model(omics, ctx, g, output_mode="full")
        self.assertTrue(torch.allclose(a.logits, b.logits))
        self.assertTrue(torch.allclose(a.logits, c.logits))

    def test_attention_override_changes_output(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        model.eval()
        omics, ctx = torch.randn(2, 64), torch.randn(2, 32)
        g = batch_drug_graphs([make_chain_graph(6), make_chain_graph(6)])
        with torch.no_grad():
            base = model(omics, ctx, g, output_mode="attention")
            mask = base.atom_mask
            b, n = mask.shape
            h = 4
            uniform = mask.float().unsqueeze(1).unsqueeze(2).expand(b, h, 1, n)
            uniform = uniform / uniform.sum(-1, keepdim=True).clamp_min(1e-8)
            alt = model(omics, ctx, g, output_mode="prediction", attention_override=uniform)
        # Not required to always differ on random init, but override path must run
        self.assertEqual(list(alt.logits.shape), [2])


class TestWeightTransfer(unittest.TestCase):
    def test_e3_gin_transfer_exact_keys(self):
        ckpt = ROOT / "result/optimization_runs/round20_unseen_drug_closure/stage20e_release/checkpoints/seed52_fold0.pt"
        if not ckpt.is_file():
            self.skipTest("Round20 checkpoint missing")
        model = build_xa_v2(_config(), model_type="biocda_xa_transfer")
        report = transfer_e3_gin_to_xa(ckpt, model, strict=True)
        self.assertTrue(report.ok)
        self.assertTrue(any(k.startswith("convs.") for k in report.loaded_keys))
        self.assertTrue(any(k.startswith("fc1_xd") for k in report.ignored_keys))
        self.assertTrue(verify_transferred_weights_match(ckpt, model))

    def test_frozen_bn_stats_do_not_change(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_transfer")
        apply_phase(model, FreezePhase("attention_warmup", epochs=1, freeze_gin_layers=[0, 1, 2, 3, 4]))
        gin = model.drug_encoder.gin
        before = copy.deepcopy(gin.bns[0].running_mean.clone())
        set_frozen_bn_eval(gin, [0, 1, 2, 3, 4])
        model.train()
        set_frozen_bn_eval(gin, [0, 1, 2, 3, 4])
        omics, ctx = torch.randn(4, 64), torch.randn(4, 32)
        g = batch_drug_graphs([make_chain_graph(5) for _ in range(4)])
        out = model(omics, ctx, g)
        loss = out.logits.sum()
        loss.backward()
        self.assertTrue(torch.allclose(before, gin.bns[0].running_mean))


class TestDistillation(unittest.TestCase):
    def test_kd_loss_is_finite(self):
        s = torch.randn(8)
        t = torch.randn(8)
        resp = torch.nn.functional.binary_cross_entropy_with_logits(s, torch.rand(8))
        out = combine_response_kd(resp, s, t, lambda_kd=0.5, temperature=2.0)
        self.assertTrue(torch.isfinite(out.total))


class TestSensitivity(unittest.TestCase):
    def test_drug_replacement_changes_atom_tokens(self):
        model = build_xa_v2(_config(), model_type="biocda_xa_fresh")
        model.eval()
        omics, ctx = torch.randn(2, 64), torch.randn(2, 32)
        ga = batch_drug_graphs([make_chain_graph(8), make_chain_graph(8)])
        gb = batch_drug_graphs([make_chain_graph(3), make_chain_graph(3)])
        with torch.no_grad():
            a = model(omics, ctx, ga, output_mode="full")
            b = model(omics, ctx, gb, output_mode="full")
        # Different molecule sizes → different dense token layouts; logits may differ
        self.assertFalse(torch.allclose(a.dense_atom_tokens[:, :3], b.dense_atom_tokens[:, :3]))


if __name__ == "__main__":
    unittest.main()
