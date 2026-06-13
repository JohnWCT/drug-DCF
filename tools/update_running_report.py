"""Write/refresh running_report.md for an optimization run."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _count_status(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty or "status" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["status"].value_counts().items()}


def _read_csv(path: str) -> pd.DataFrame | None:
    path = _resolve(path)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _stage_pretrain_section(run_dir: str, manifest: pd.DataFrame | None) -> list[str]:
    lines = ["## Stage 1: VAEwC Pretrain Sweep (72 jobs)", ""]
    if manifest is None:
        lines.append("- Status: **not started**")
        return lines
    counts = _count_status(manifest)
    total = len(manifest)
    success = counts.get("success", 0)
    failed = counts.get("failed", 0)
    pending = counts.get("pending", 0)
    running = counts.get("running", 0)
    pct = 100.0 * success / total if total else 0.0
    lines.extend(
        [
            f"- Progress: **{success}/{total}** success ({pct:.1f}%), pending={pending}, running={running}, failed={failed}",
            f"- Sweep axes: `lambda_proto`×`proto_temperature`×`proto_start_epoch`×`proto_full_epoch` = 72 combos",
        ]
    )
    if success > 0 and "lambda_proto" in manifest.columns:
        ok = manifest[manifest["status"] == "success"]
        lines.append(f"- Completed `lambda_proto=0` controls: {(ok['lambda_proto'] == 0).sum()} / 18")
        lines.append(f"- Completed InfoNCE runs: {(ok['lambda_proto'] > 0).sum()} / 54")
    if failed > 0:
        bad = manifest[manifest["status"] == "failed"].head(5)
        lines.append("- Recent failures:")
        for _, row in bad.iterrows():
            lines.append(f"  - `{row['job_id']}`: {row.get('error_message', '')}")
    if success == total and total > 0:
        lines.extend(
            [
                "",
                "### Stage 1 conclusion",
                "- All 72 pretrain jobs finished successfully.",
                "- Ready for visualization, filtering, and Top-10 selection.",
            ]
        )
    elif success + failed == total and pending == 0 and running == 0:
        lines.extend(
            [
                "",
                "### Stage 1 conclusion",
                f"- Sweep finished with **{failed} failed** job(s). Selection can proceed on successful runs.",
            ]
        )
    return lines


def _fmt_opt(val, prec: int = 4) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "NA"
    try:
        return f"{float(val):.{prec}f}"
    except (TypeError, ValueError):
        return str(val)


def _stage_selection_section(run_dir: str) -> list[str]:
    lines = ["## Stage 2: Selection (filter + Top-10 with controls)", ""]
    top10_path = _resolve(os.path.join(run_dir, "selection", "pretrain_top10.csv"))
    filtered_path = _resolve(os.path.join(run_dir, "selection", "pretrain_filtered_candidates.csv"))
    if not os.path.exists(top10_path):
        lines.append("- Status: **pending** (waiting for pretrain)")
        return lines
    top10 = pd.read_csv(top10_path)
    n_ctrl = int((top10.get("lambda_proto", pd.Series(dtype=float)).fillna(0) == 0).sum()) if "lambda_proto" in top10.columns else 0
    lines.append(f"- Top-10 selected: **{len(top10)}** candidates (controls in Top-10: {n_ctrl})")
    if os.path.exists(filtered_path):
        flt = pd.read_csv(filtered_path)
        lines.append(f"- After quality filter: {len(flt)} candidates")
    lines.append("- Selected IDs:")
    for _, row in top10.iterrows():
        tag = " [control]" if row.get("is_control") else ""
        score = row.get("sweetspot_score", row.get("score_total"))
        lines.append(
            f"  - `{row['ID']}` score={_fmt_opt(score)} lambda_proto={_fmt_opt(row.get('lambda_proto'))}{tag}"
        )
    lines.extend(
        [
            "",
            "### Stage 2 conclusion",
            "- Top-10 with mandatory controls exported to `selection/model_select.csv` for finetune.",
        ]
    )
    return lines


def _stage_finetune_section(run_dir: str, manifest: pd.DataFrame | None) -> list[str]:
    lines = ["## Stage 3: Downstream Finetune (Top-10 × 4 = 40 jobs)", ""]
    if manifest is None:
        lines.append("- Status: **pending**")
        return lines
    counts = _count_status(manifest)
    total = len(manifest)
    success = counts.get("success", 0)
    failed = counts.get("failed", 0)
    pending = counts.get("pending", 0)
    lines.append(
        f"- Progress: **{success}/{total}** success, pending={pending}, failed={failed}"
    )
    if success == total and total > 0:
        lines.extend(
            [
                "",
                "### Stage 3 conclusion",
                "- All 40 finetune mini-grid jobs completed.",
            ]
        )
    return lines


def _stage_aggregate_section(run_dir: str) -> list[str]:
    lines = ["## Stage 4: Aggregation & Final Ranking", ""]
    agg_path = _resolve(os.path.join(run_dir, "aggregate", "aggregate_scores.csv"))
    if not os.path.exists(agg_path):
        lines.append("- Status: **pending**")
        return lines
    agg = pd.read_csv(agg_path)
    if "Model_ID" in agg.columns:
        agg = agg.set_index("Model_ID")
    sort_col = (
        "Average_TCGA_AUC_mean"
        if "Average_TCGA_AUC_mean" in agg.columns
        else "Global_TCGA_AUC_mean"
        if "Global_TCGA_AUC_mean" in agg.columns
        else agg.columns[0]
    )
    ranked = agg.sort_values(sort_col, ascending=False)
    lines.append(f"- Primary metric: `{sort_col}`")
    lines.append("- Top downstream candidates:")
    for model_id, row in ranked.head(10).iterrows():
        lines.append(f"  - `{model_id}`: {sort_col}={row[sort_col]:.6f}")
    manifest = _read_csv(os.path.join(run_dir, "manifests", "pretrain_sweep_manifest.csv"))
    lam_map = {}
    if manifest is not None:
        for _, r in manifest.iterrows():
            if str(r.get("status")) == "success" and r.get("result_dir"):
                lam_map[os.path.basename(str(r["result_dir"]))] = float(r["lambda_proto"])
    best_id = str(ranked.index[0])
    best_lam = lam_map.get(best_id, float("nan"))
    ctrl = [i for i, v in lam_map.items() if v == 0 and i in ranked.index]
    inf = [i for i, v in lam_map.items() if v > 0 and i in ranked.index]
    if ctrl and inf:
        cm = float(ranked.loc[ctrl, sort_col].mean())
        im = float(ranked.loc[inf, sort_col].mean())
        lines.extend(
            [
                "",
                "### Stage 4 conclusion",
                f"- Best model: **`{best_id}`** ({sort_col}={ranked.iloc[0][sort_col]:.6f}, lambda_proto={best_lam})",
                f"- Control mean AUC: {cm:.6f} | InfoNCE mean AUC: {im:.6f}",
            ]
        )
        if best_lam and best_lam > 0:
            lines.append("- **InfoNCE candidate ranks #1 downstream.**")
        elif im > cm:
            lines.append("- InfoNCE group mean beats control; best single model may still be control.")
        else:
            lines.append("- Control leads downstream; review InfoNCE hyperparameters in next ablation.")
    return lines


def write_running_report(run_dir: str, note: str = "") -> str:
    run_dir = _resolve(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, "running_report.md")

    pre_manifest = _read_csv(os.path.join(run_dir, "manifests", "pretrain_sweep_manifest.csv"))
    ft_manifest = _read_csv(os.path.join(run_dir, "manifests", "finetune_dispatch_manifest.csv"))

    pre_success = _count_status(pre_manifest).get("success", 0)
    pre_total = len(pre_manifest) if pre_manifest is not None else 72
    ft_total = len(ft_manifest) if ft_manifest is not None else 0
    ft_success = _count_status(ft_manifest).get("success", 0)
    has_agg = os.path.exists(_resolve(os.path.join(run_dir, "aggregate", "aggregate_scores.csv")))

    if has_agg:
        overall = "Stage 4 complete"
    elif ft_success == ft_total and ft_total > 0:
        overall = "Stage 3 complete — ready to aggregate"
    elif pre_success == pre_total and pre_total > 0:
        overall = "Stage 1 complete — ready for selection"
    elif pre_success > 0 or _count_status(pre_manifest).get("running", 0) > 0:
        overall = f"Stage 1 in progress ({pre_success}/{pre_total})"
    else:
        overall = "Starting Stage 1"

    lines = [
        "# VAEwC Prototype InfoNCE — Running Report",
        "",
        f"**Last updated:** {_utc()}",
        f"**Run ID:** `{os.path.basename(run_dir)}`",
        f"**Overall:** {overall}",
        "",
        "**Parallel policy:** each subprocess = one hyperparameter combo; results merged via manifest + aggregate.",
        "**GPU tuning:** finetune `batch_size=4096`, `mini_batch=1024`, `max_parallel=26` (~59% GPU mem).",
        "",
    ]
    if note:
        lines.extend([f"> {note}", ""])

    lines.extend(_stage_pretrain_section(run_dir, pre_manifest))
    lines.append("")
    lines.extend(_stage_selection_section(run_dir))
    lines.append("")
    lines.extend(_stage_finetune_section(run_dir, ft_manifest))
    lines.append("")
    lines.extend(_stage_aggregate_section(run_dir))
    lines.append("")
    lines.extend(
        [
            "## Commands",
            "```bash",
            f"docker exec DAPL python3 /workspace/DAPL/tools/update_running_report.py --run-dir {os.path.relpath(run_dir, PROJECT_ROOT)}",
            f"docker exec DAPL python3 /workspace/DAPL/tools/optimization_runner.py pretrain --manifest {os.path.relpath(run_dir, PROJECT_ROOT)}/manifests/pretrain_sweep_manifest.csv --run-dir {os.path.relpath(run_dir, PROJECT_ROOT)}",
            "```",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser("update_running_report")
    parser.add_argument("--run-dir", default="result/optimization_runs/vaewc_proto_infonce_round1")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    path = write_running_report(args.run_dir, note=args.note)
    print(path)


if __name__ == "__main__":
    main()
