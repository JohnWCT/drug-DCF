from pathlib import Path
import yaml

from biocda.validation.round24_protocol import gate_table, load_eval3_config
from biocda.validation.round24_gate import evaluate_all_target_gate


ROOT = Path(__file__).resolve().parents[1]


def test_eval3_config_loads():
    cfg = load_eval3_config(ROOT / "configs/round24/eval3.yaml")
    assert cfg["protocol_name"] == "eval3"
    assert len(cfg["targets"]) == 5
    assert cfg["n_folds"] == 5


def test_gate_all_pass():
    cfg = load_eval3_config(ROOT / "configs/round24/eval3.yaml")
    gates = gate_table(cfg)
    aucs = {t: gates[t]["gate_auroc"] + 0.01 for t in cfg["target_priority"]}
    res = evaluate_all_target_gate(
        aucs, gates, target_priority=cfg["target_priority"], target_weights=cfg["target_weights"]
    )
    assert res["status"] == "PASS"


def test_gate_one_fail_is_no_lock():
    cfg = load_eval3_config(ROOT / "configs/round24/eval3.yaml")
    gates = gate_table(cfg)
    aucs = {t: gates[t]["gate_auroc"] + 0.01 for t in cfg["target_priority"]}
    aucs["gdsc_intersect13"] = gates["gdsc_intersect13"]["gate_auroc"] - 0.01
    res = evaluate_all_target_gate(
        aucs, gates, target_priority=cfg["target_priority"], target_weights=cfg["target_weights"]
    )
    assert res["status"] == "NO_LOCK"
    assert res["n_pass"] == 4
