#!/usr/bin/env python3
"""Build the Round 19F proposal-only role assignment from allowlisted summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SHIFTS = ("drug_heldout", "scaffold_heldout", "cancer_type_heldout")
CHEMICAL_SHIFTS = ("drug_heldout", "scaffold_heldout")
PASSING = {"PASS", "NON_WORSE"}

REPORT_FILES = {
    "stage19d_cross_summary": "round19d_cross_seed_summary.csv",
    "stage19d_paired": "round19d_paired_fold_deltas.csv",
    "stage19d_resource": "round19d_resource_summary.csv",
    "stage19e_per_shift": "round19e_per_shift_summary.csv",
    "stage19e_guardrails": "round19e_shift_guardrails.csv",
    "stage19e_paired": "round19e_paired_fold_deltas.csv",
    "stage19e_resource": "round19e_resource_summary.csv",
    "stage19d_experiment_lock": "round19_stage19d_experiment_lock.json",
    "stage19e_candidate_lock": "round19_stage19e_candidate_lock.json",
    "stage19e_experiment_lock": "round19_stage19e_experiment_lock.json",
}
POLICY_RELATIVE = Path("config/round19_stage19f_role_policy.json")
SETTINGS_RELATIVE = Path("config/round19_factorial_settings.json")
PREDICTOR_SOURCE_RELATIVE = Path("tools/round19_fusion_models.py")

# This is deliberately independent of display names in settings. Any implementation
# change to these pinned semantics changes the definition hash in the proposal.
PREDICTOR_DEFINITIONS = {
    "P0": {
        "family": "pooled_mlp",
        "fusion": "adapt_omics_and_drug_then_concat",
        "uses_atom_tokens": False,
    },
    "P1": {
        "family": "compact_pooled_transformer",
        "fusion": "pooled_token_transformer",
        "uses_atom_tokens": False,
    },
    "P2": {
        "family": "pure_atom_cross_attention",
        "fusion": "atom_tokens_cross_attend_to_omics",
        "uses_atom_tokens": True,
    },
}


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


PREDICTOR_DEFINITION_HASH = "772cb020a9b2ef399e442ae0ecf9356f427748f91cd7e8fcff549acb3ba0296f"
if _canonical_hash(PREDICTOR_DEFINITIONS) != PREDICTOR_DEFINITION_HASH:
    raise RuntimeError("Hardcoded predictor definitions changed without a policy hash update")


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _int(value: Any, *, label: str) -> int:
    number = _float(value, label=label)
    if not number.is_integer():
        raise ValueError(f"{label} is not an integer: {value!r}")
    return int(number)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Not a boolean: {value!r}")


def classify_shift(
    delta: float,
    *,
    practical_margin: float = 0.003,
    major_margin: float = 0.015,
) -> dict:
    """Classify an unrounded AUC delta without display-precision decisions."""
    delta = _float(delta, label="shift delta")
    if delta >= practical_margin:
        status = "PASS"
    elif delta > -practical_margin:
        status = "NON_WORSE"
    else:
        status = "FAIL"
    return {
        "delta_vs_E0": delta,
        "status": status,
        "major_fail": delta <= -major_margin,
        "practical_margin": practical_margin,
        "major_margin": major_margin,
    }


def _short_f(candidate_id: str) -> str:
    return str(candidate_id).split("_", 1)[0]


def _candidate_map(lock: Mapping[str, Any], *, stage: str) -> Dict[str, dict]:
    candidates = lock.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{stage} lock has no candidate list")
    result: Dict[str, dict] = {}
    for candidate in candidates:
        cid = str(candidate.get("candidate_id"))
        key = _short_f(cid) if stage == "19D" else cid
        if key in result:
            raise ValueError(f"Duplicate {stage} candidate {key}")
        result[key] = candidate
    return result


def _identity(candidate: Mapping[str, Any]) -> tuple[str, str, str]:
    return tuple(str(candidate.get(key)) for key in ("drug_id", "predictor_id", "omics_id"))


def _verify_candidate_locks(lock_d: dict, lock_ec: dict, lock_ee: dict) -> Dict[str, dict]:
    if lock_d.get("lock_type") != "stage19d_experiment_lock":
        raise ValueError("Unexpected 19D experiment lock type")
    if lock_ec.get("lock_type") != "stage19e_candidate_lock":
        raise ValueError("Unexpected 19E candidate lock type")
    if lock_ee.get("lock_type") != "stage19e_experiment_lock":
        raise ValueError("Unexpected 19E experiment lock type")
    dmap = _candidate_map(lock_d, stage="19D")
    ecmap = _candidate_map(lock_ec, stage="19E")
    eemap = _candidate_map(lock_ee, stage="19E")
    expected_e = {f"E{i}" for i in range(6)}
    expected_f = {f"F{i}" for i in range(6)}
    if set(dmap) != expected_f or set(ecmap) != expected_e or set(eemap) != expected_e:
        raise RuntimeError("Candidate locks must contain exactly F0-F5 and E0-E5")
    evidence: Dict[str, dict] = {}
    for index in range(6):
        eid, fid = f"E{index}", f"F{index}"
        source_ec = _short_f(str(ecmap[eid].get("source_candidate_id")))
        source_ee = _short_f(str(eemap[eid].get("source_candidate_id")))
        identities = {
            "stage19d": _identity(dmap[fid]),
            "stage19e_candidate": _identity(ecmap[eid]),
            "stage19e_experiment": _identity(eemap[eid]),
        }
        if source_ec != fid or source_ee != fid or len(set(identities.values())) != 1:
            raise RuntimeError(f"Candidate identity drift for {eid}/{fid}: {identities}")
        evidence[eid] = {
            "source_candidate_id": str(dmap[fid]["candidate_id"]),
            "drug_id": identities["stage19d"][0],
            "predictor_id": identities["stage19d"][1],
            "omics_id": identities["stage19d"][2],
            "locks_match": True,
        }
    return evidence


def _verify_completeness(
    cross: Sequence[dict],
    paired_d: Sequence[dict],
    resource_d: Sequence[dict],
    per_shift: Sequence[dict],
    paired_e: Sequence[dict],
    lock_d: Mapping[str, Any],
    lock_ec: Mapping[str, Any],
    lock_ee: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict:
    design = policy["required_design"]
    d = design["stage19d"]
    e = design["stage19e"]
    if len(cross) != int(d["candidate_count"]):
        raise RuntimeError(f"19D expected {d['candidate_count']} candidates, got {len(cross)}")
    if {_int(row["n_seeds"], label="19D n_seeds") for row in cross} != {int(d["seed_count"])}:
        raise RuntimeError("19D does not contain exactly three completed seeds per candidate")
    if (
        len(lock_d.get("split_seeds") or []) != int(d["seed_count"])
        or int(lock_d.get("n_folds", -1)) != int(d["fold_count"])
    ):
        raise RuntimeError("19D experiment lock design does not match the required seeds/folds")
    expected_d_jobs = int(d["candidate_count"]) * int(d["seed_count"]) * int(d["fold_count"])
    if len(paired_d) != int(d["seed_count"]) * int(d["fold_count"]):
        raise RuntimeError("19D paired F1/F2 evidence is incomplete")
    if len(resource_d) != 1:
        raise RuntimeError("19D resource summary must have one aggregate row")
    rd = resource_d[0]
    if _int(rd["n_jobs"], label="19D n_jobs") != expected_d_jobs or _int(
        rd["n_done"], label="19D n_done"
    ) != expected_d_jobs:
        raise RuntimeError("19D jobs are incomplete")

    expected_groups = int(e["candidate_count"]) * int(e["shift_count"])
    if (
        len(lock_ec.get("candidates") or []) != int(e["candidate_count"])
        or len(lock_ec.get("shift_strategies") or []) != int(e["shift_count"])
        or int(lock_ec.get("n_folds", -1)) != int(e["fold_count"])
        or len(lock_ee.get("candidates") or []) != int(e["candidate_count"])
        or len(lock_ee.get("shift_strategies") or []) != int(e["shift_count"])
        or int(lock_ee.get("n_folds", -1)) != int(e["fold_count"])
    ):
        raise RuntimeError("19E candidate/experiment lock design does not match the required grid")
    if len(per_shift) != expected_groups:
        raise RuntimeError(f"19E expected {expected_groups} candidate-shift rows, got {len(per_shift)}")
    expected_candidates = {f"E{i}" for i in range(int(e["candidate_count"]))}
    observed_pairs = {(row["candidate_id"], row["shift_strategy"]) for row in per_shift}
    expected_pairs = {(cid, shift) for cid in expected_candidates for shift in SHIFTS}
    if observed_pairs != expected_pairs:
        raise RuntimeError("19E candidate-shift coverage is incomplete or duplicated")
    for row in per_shift:
        if _int(row["n_folds"], label="19E n_folds") != int(e["fold_count"]):
            raise RuntimeError("19E does not contain five completed folds per candidate-shift")
        _float(row["mean_DrugMacro_AUC"], label="19E mean AUC")
        _float(row["mean_Global_AUC"], label="19E mean Global AUC")
    if len(paired_e) != 7 * int(e["shift_count"]) * int(e["fold_count"]):
        raise RuntimeError("19E paired-fold evidence is incomplete")
    expected_e_jobs = int(e["candidate_count"]) * int(e["shift_count"]) * int(e["fold_count"])
    return {
        "stage19d": {
            "candidates": int(d["candidate_count"]),
            "seeds": int(d["seed_count"]),
            "folds": int(d["fold_count"]),
            "completed_jobs": expected_d_jobs,
        },
        "stage19e": {
            "candidates": int(e["candidate_count"]),
            "shifts": int(e["shift_count"]),
            "folds": int(e["fold_count"]),
            "completed_jobs": expected_e_jobs,
            "failed_jobs": 0,
            "zero_failures_basis": "all expected candidate-shift-fold cells have complete finite summaries",
        },
    }


def _build_shift_evidence(
    per_shift: Sequence[dict],
    guardrails: Sequence[dict],
    *,
    practical_margin: float,
    major_margin: float,
) -> Dict[str, Dict[str, dict]]:
    rows = {(row["candidate_id"], row["shift_strategy"]): row for row in per_shift}
    reported = {(row["candidate_id"], row["shift_strategy"]): row for row in guardrails}
    if set(rows) != set(reported):
        raise RuntimeError("19E guardrail coverage does not match per-shift summary")
    result: Dict[str, Dict[str, dict]] = {}
    for cid, shift in sorted(rows):
        row = rows[(cid, shift)]
        e0 = rows[("E0", shift)]
        auc = _float(row["mean_DrugMacro_AUC"], label=f"{cid}/{shift} AUC")
        delta = auc - _float(e0["mean_DrugMacro_AUC"], label=f"E0/{shift} AUC")
        classification = classify_shift(
            delta, practical_margin=practical_margin, major_margin=major_margin
        )
        source = reported[(cid, shift)]
        reported_delta = _float(source["delta_vs_E0"], label=f"{cid}/{shift} reported delta")
        if reported_delta != delta:
            raise RuntimeError(f"Guardrail delta precision drift for {cid}/{shift}")
        if source["guardrail_vs_E0"] != classification["status"] or _bool(
            source["MAJOR_FAIL"]
        ) != classification["major_fail"]:
            raise RuntimeError(f"Guardrail classification drift for {cid}/{shift}")
        result.setdefault(cid, {})[shift] = {
            **classification,
            "mean_DrugMacro_AUC": auc,
            "mean_DrugMacro_AUPRC": _float(
                row["mean_DrugMacro_AUPRC"], label=f"{cid}/{shift} AUPRC"
            ),
            "std_DrugMacro_AUC": _float(
                row["std_DrugMacro_AUC"], label=f"{cid}/{shift} AUC std"
            ),
            "mean_Global_AUC": _float(row["mean_Global_AUC"], label=f"{cid}/{shift} Global AUC"),
        }
    return result


def _simplicity(cid: str, policy: Mapping[str, Any]) -> int:
    return int(policy["simplicity_order"][cid])


def _chemical_score(cid: str, shifts: Mapping[str, Mapping[str, float]], policy: dict) -> tuple:
    chemical = [shifts[cid][shift] for shift in CHEMICAL_SHIFTS]
    deltas = [row["delta_vs_E0"] for row in chemical]
    return (
        min(deltas),
        sum(deltas) / len(deltas),
        sum(row["mean_DrugMacro_AUPRC"] for row in chemical) / len(chemical),
        -sum(row["std_DrugMacro_AUC"] for row in chemical) / len(chemical),
        -_simplicity(cid, policy),
    )


def select_chemical_specialist(shifts: Mapping[str, Mapping[str, dict]], policy: dict) -> str:
    eligible = [
        cid
        for cid in shifts
        if not any(shifts[cid][shift]["major_fail"] for shift in CHEMICAL_SHIFTS)
    ]
    if not eligible:
        raise RuntimeError("No chemical specialist is eligible")
    return max(eligible, key=lambda cid: (_chemical_score(cid, shifts, policy), cid))


def _efficiency_evidence(lock_d: dict, lock_e: dict, resource_e: Sequence[dict]) -> dict:
    f5 = _candidate_map(lock_d, stage="19D")["F5"]
    e5 = _candidate_map(lock_e, stage="19E")["E5"]
    reasons = [f5.get("inclusion_reason") or {}, e5.get("inclusion_reason") or {}]
    time_ok = any(bool(reason.get("cond_b_time") or reason.get("time_ok")) for reason in reasons)
    vram_ok = any(bool(reason.get("cond_c_vram") or reason.get("vram_ok")) for reason in reasons)
    resource_ok = any(
        _int(row.get("n_done", 0), label="19E resource n_done") > 0 for row in resource_e
    )
    return {
        "time_ok": time_ok,
        "vram_ok": vram_ok,
        "resource_summary_present": resource_ok,
        "eligible": time_ok or vram_ok or resource_ok,
        "stage19d_inclusion_reason": f5.get("inclusion_reason"),
        "stage19e_inclusion_reason": e5.get("inclusion_reason"),
    }


def build_proposal(
    root: Path,
    *,
    project_root: Optional[Path] = None,
    policy_path: Optional[Path] = None,
) -> dict:
    root = Path(root)
    project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    reports = root / "reports"
    policy_path = Path(policy_path) if policy_path else project_root / POLICY_RELATIVE
    paths = {name: reports / filename for name, filename in REPORT_FILES.items()}
    policy = _read_json(policy_path)
    if not policy.get("proposal_only") or policy.get("single_champion") is not False:
        raise ValueError("19F policy must be proposal-only with no single champion")

    cross = _read_csv(paths["stage19d_cross_summary"])
    paired_d = _read_csv(paths["stage19d_paired"])
    resource_d = _read_csv(paths["stage19d_resource"])
    per_shift = _read_csv(paths["stage19e_per_shift"])
    guardrails = _read_csv(paths["stage19e_guardrails"])
    paired_e = _read_csv(paths["stage19e_paired"])
    resource_e = _read_csv(paths["stage19e_resource"])
    lock_d = _read_json(paths["stage19d_experiment_lock"])
    lock_ec = _read_json(paths["stage19e_candidate_lock"])
    lock_ee = _read_json(paths["stage19e_experiment_lock"])

    identities = _verify_candidate_locks(lock_d, lock_ec, lock_ee)
    completeness = _verify_completeness(
        cross,
        paired_d,
        resource_d,
        per_shift,
        paired_e,
        lock_d,
        lock_ec,
        lock_ee,
        policy,
    )
    margins = policy["margins"]
    practical_margin = float(margins["practical_tie_auc"])
    major_margin = float(margins["major_fail_auc"])
    shifts = _build_shift_evidence(
        per_shift,
        guardrails,
        practical_margin=practical_margin,
        major_margin=major_margin,
    )

    source_rows = {
        _short_f(row["candidate_id"]): {
            **row,
            "mean_auc": _float(
                row["mean_of_means_DrugMacro_AUC"], label=f"{row['candidate_id']} source AUC"
            ),
        }
        for row in cross
    }
    source_champion = max(source_rows, key=lambda fid: (source_rows[fid]["mean_auc"], fid))
    f1_disadvantage = max(0.0, source_rows["F2"]["mean_auc"] - source_rows["F1"]["mean_auc"])
    e1_cancer_disadvantage = max(
        0.0,
        shifts["E2"]["cancer_type_heldout"]["mean_DrugMacro_AUC"]
        - shifts["E1"]["cancer_type_heldout"]["mean_DrugMacro_AUC"],
    )
    parsimonious = (
        "F1"
        if f1_disadvantage <= practical_margin and e1_cancer_disadvantage <= practical_margin
        else None
    )

    cancer_auc = {
        cid: shifts[cid]["cancer_type_heldout"]["mean_DrugMacro_AUC"] for cid in ("E1", "E2")
    }
    if abs(cancer_auc["E1"] - cancer_auc["E2"]) <= practical_margin:
        cancer_specialist = "E1"
    else:
        cancer_specialist = max(
            ("E1", "E2"),
            key=lambda cid: (
                cancer_auc[cid],
                shifts[cid]["cancer_type_heldout"]["mean_DrugMacro_AUPRC"],
                -shifts[cid]["cancer_type_heldout"]["std_DrugMacro_AUC"],
                -_simplicity(cid, policy),
                cid,
            ),
        )
    chemical_specialist = select_chemical_specialist(shifts, policy)

    e4_rows = shifts["E4"]
    e4_chemical_passing = sum(e4_rows[shift]["status"] in PASSING for shift in CHEMICAL_SHIFTS)
    e4_fail_count = sum(row["status"] == "FAIL" for row in e4_rows.values())
    e4_no_collapse = all(
        math.isfinite(row["mean_DrugMacro_AUC"]) and row["mean_Global_AUC"] > 0.5
        for row in e4_rows.values()
    )
    e4_eligible = (
        not e4_rows["cancer_type_heldout"]["major_fail"]
        and e4_chemical_passing >= 1
        and e4_fail_count <= 1
        and e4_no_collapse
    )

    efficiency = _efficiency_evidence(lock_d, lock_ec, resource_e)
    e5_passing = sum(row["status"] in PASSING for row in shifts["E5"].values())
    e5_eligible = e5_passing >= 2 and efficiency["eligible"]

    general_eligible = [
        cid
        for cid, rows in shifts.items()
        if all(rows[shift]["status"] in PASSING for shift in SHIFTS)
        and not any(rows[shift]["major_fail"] for shift in SHIFTS)
    ]
    general = (
        max(
            general_eligible,
            key=lambda cid: (
                _chemical_score(cid, shifts, policy)[0],
                shifts[cid]["cancer_type_heldout"]["mean_DrugMacro_AUC"],
                -_simplicity(cid, policy),
                cid,
            ),
        )
        if general_eligible
        else None
    )
    if general is None and not policy["roles"]["general"]["allow_null"]:
        raise RuntimeError("No general candidate passed the policy")

    settings_path = project_root / SETTINGS_RELATIVE
    predictor_source_path = project_root / PREDICTOR_SOURCE_RELATIVE
    settings = _read_json(settings_path)
    if settings.get("predictors") != {
        key: value["family"] for key, value in PREDICTOR_DEFINITIONS.items()
    }:
        raise RuntimeError("Predictor names drifted from the hardcoded 19F definition")
    settings_hash = _sha256(settings_path)
    lock_settings_hashes = {
        str(lock_d.get("hashes", {}).get("settings_sha256")),
        str(lock_ee.get("hashes", {}).get("settings_sha256")),
    }
    if lock_settings_hashes != {settings_hash}:
        raise RuntimeError("Settings hash drifted from the 19D/19E experiment locks")

    selection_hashes = {name: _sha256(path) for name, path in paths.items()}
    selection_hashes["policy"] = _sha256(policy_path)
    proposal = {
        "lock_type": "round19_final_role_proposal",
        "artifact_type": "round19_stage19f_role_proposal",
        "proposal_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_champion": None,
        "selection_used_internal": False,
        "selection_used_tcga": False,
        "roles": {
            "historical_anchor": {
                "candidate_id": "E0",
                "source_candidate_id": "F0_historical_anchor",
            },
            "source_performance_champion": {
                "candidate_id": source_champion,
                "source_candidate_id": source_champion,
            },
            "parsimonious_context_model": {
                "candidate_id": parsimonious,
                "source_candidate_id": parsimonious,
                "status": "eligible" if parsimonious is not None else "not_eligible",
            },
            "cancer_shift_specialist": {
                "candidate_id": cancer_specialist,
                "source_candidate_id": identities[cancer_specialist]["source_candidate_id"],
            },
            "chemical_shift_specialist": {
                "candidate_id": chemical_specialist,
                "source_candidate_id": identities[chemical_specialist]["source_candidate_id"],
            },
            "source_only_domain_candidate": {
                "candidate_id": "E4" if e4_eligible else None,
                "source_candidate_id": (
                    identities["E4"]["source_candidate_id"] if e4_eligible else None
                ),
                "status": (
                    "source_only_domain_candidate"
                    if e4_eligible
                    else "archived_after_shift_validation"
                ),
            },
            "efficient_model": {
                "candidate_id": "E5" if e5_eligible else None,
                "source_candidate_id": (
                    identities["E5"]["source_candidate_id"] if e5_eligible else None
                ),
                "status": "eligible" if e5_eligible else "no_candidate_passed",
            },
            "general_recommended_model": {
                "candidate_id": general,
                "source_candidate_id": (
                    identities[general]["source_candidate_id"] if general is not None else None
                ),
                "reason": (
                    "Passed all three full-precision shift guardrails without MAJOR_FAIL."
                    if general is not None
                    else "No single model satisfied all pre-registered shift guardrails."
                ),
            },
        },
        "role_evidence": {
            "source_champion": {
                "mean_auc_by_candidate": {
                    fid: source_rows[fid]["mean_auc"] for fid in sorted(source_rows)
                },
                "rule": "maximum unrounded 19D cross-seed mean AUC",
            },
            "parsimonious": {
                "f1_absolute_disadvantage_vs_f2": f1_disadvantage,
                "e1_cancer_disadvantage_vs_e2": e1_cancer_disadvantage,
                "margin": practical_margin,
                "eligible": parsimonious is not None,
            },
            "cancer_specialist": {
                "auc_by_candidate": cancer_auc,
                "tie_margin": practical_margin,
                "o2_preferred_within_tie": abs(cancer_auc["E1"] - cancer_auc["E2"])
                <= practical_margin,
            },
            "chemical_specialist": {
                cid: {
                    "eligible": not any(
                        shifts[cid][shift]["major_fail"] for shift in CHEMICAL_SHIFTS
                    ),
                    "maximin_delta_vs_E0": min(
                        shifts[cid][shift]["delta_vs_E0"] for shift in CHEMICAL_SHIFTS
                    ),
                    "mean_delta_vs_E0": sum(
                        shifts[cid][shift]["delta_vs_E0"] for shift in CHEMICAL_SHIFTS
                    )
                    / 2.0,
                }
                for cid in sorted(shifts)
            },
            "domain_generalization": {
                "candidate": "E4",
                "cancer_no_major_fail": not e4_rows["cancer_type_heldout"]["major_fail"],
                "chemical_pass_or_non_worse_count": e4_chemical_passing,
                "fail_count": e4_fail_count,
                "no_prediction_collapse": e4_no_collapse,
                "eligible": e4_eligible,
            },
            "efficient": {
                "candidate": "E5",
                "pass_or_non_worse_shift_count": e5_passing,
                **efficiency,
                "gate_eligible": e5_eligible,
            },
            "general": {
                "eligible_candidates": sorted(general_eligible),
                "selected_by": [
                    "chemical_maximin_delta_vs_E0",
                    "cancer_mean_DrugMacro_AUC",
                    "simplicity",
                    "candidate_id",
                ],
            },
        },
        "raw_precision_shift_evidence": shifts,
        "candidate_identity_evidence": identities,
        "completion_evidence": completeness,
        "definition_snapshot": {
            "settings": {
                "predictors": settings["predictors"],
                "omics_ids": settings["omics_ids"],
                "drug_reps": settings["drug_reps"],
                "sha256": settings_hash,
                "matches_stage19d_and_stage19e_locks": True,
            },
            "predictor_hardcoded_definitions": PREDICTOR_DEFINITIONS,
            "predictor_hardcoded_definition_sha256": PREDICTOR_DEFINITION_HASH,
            "predictor_source": {
                "relative_path": str(PREDICTOR_SOURCE_RELATIVE),
                "sha256": _sha256(predictor_source_path),
            },
        },
        "input_hashes": {
            "selection_inputs": selection_hashes,
            "definition_validation_inputs": {
                str(SETTINGS_RELATIVE): settings_hash,
                str(PREDICTOR_SOURCE_RELATIVE): _sha256(predictor_source_path),
            },
        },
        "final_role_lock_created": False,
    }
    return proposal


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19F proposal-only role selector")
    parser.add_argument("--root", default="result/optimization_runs/round19_factorial")
    parser.add_argument(
        "--output",
        default=(
            "result/optimization_runs/round19_factorial/"
            "reports/round19_final_role_proposal.json"
        ),
    )
    parser.add_argument("--policy", default=str(POLICY_RELATIVE))
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Accepted for an auditable CLI; completeness is always required.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    proposal = build_proposal(
        Path(args.root),
        project_root=project_root,
        policy_path=project_root / args.policy,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(output), "roles": proposal["roles"]}, indent=2))


if __name__ == "__main__":
    main()
