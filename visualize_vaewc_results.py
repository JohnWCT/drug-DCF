"""
Aggregate VAEwC GAN experiment outputs into CSV/HTML.

Usage example:
python visualize_vaewc_results.py \
  --result_dir result/pretrain_vaewc \
  --output_dir result/pretrain_vaewc/00_report \
  --per_page 80 \
  --select_top_k 2
"""

import os
import json
import math
import base64
import argparse
import pandas as pd
from glob import glob


def _read_json(path, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _read_last_csv_row(path):
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def _attach_prefixed_metrics(row, metrics, prefix):
    if not isinstance(metrics, dict):
        return
    for k, v in metrics.items():
        row[f"{prefix}{k}"] = v


def load_experiment_data(exp_dir):
    exp_id = os.path.basename(exp_dir)
    params = _read_json(os.path.join(exp_dir, "params.json"), {})
    metrics = _read_json(os.path.join(exp_dir, "gan_metrics.json"), {})
    run_summary = _read_json(os.path.join(exp_dir, "run_summary.json"), {})
    summary_metrics = run_summary.get("metrics", {}) if isinstance(run_summary, dict) else {}
    merged_metrics = {}
    if isinstance(summary_metrics, dict):
        merged_metrics.update(summary_metrics)
    if isinstance(metrics, dict):
        merged_metrics.update(metrics)
    tsne_path = os.path.join(exp_dir, "tsne_gan_best.png")
    if not os.path.exists(tsne_path):
        tsne_path = None
    row = {
        "ID": exp_id,
        "fid": merged_metrics.get("fid"),
        "mmd": merged_metrics.get("mmd"),
        "wasserstein": merged_metrics.get("wasserstein"),
        "best_gan_epoch": merged_metrics.get("best_gan_epoch"),
        "best_gan_loss": merged_metrics.get("best_gan_loss"),
        "tcga_raw_sample_count_for_latent": merged_metrics.get("tcga_raw_sample_count_for_latent"),
        "tcga_patient_count_for_latent": merged_metrics.get("tcga_patient_count_for_latent"),
        "kmeans_k": merged_metrics.get("kmeans_k"),
        "kmeans_ari": merged_metrics.get("kmeans_ari"),
        "kmeans_nmi": merged_metrics.get("kmeans_nmi"),
        "kmeans_silhouette": merged_metrics.get("kmeans_silhouette"),
        "kmeans_calinski_harabasz": merged_metrics.get("kmeans_calinski_harabasz"),
        "kmeans_davies_bouldin": merged_metrics.get("kmeans_davies_bouldin"),
        "pretrain_num_epochs": params.get("params", {}).get("pretrain_num_epochs"),
        "train_num_epochs": params.get("params", {}).get("train_num_epochs"),
        "pretrain_learning_rate": params.get("params", {}).get("pretrain_learning_rate"),
        "gan_learning_rate": params.get("params", {}).get("gan_learning_rate"),
        "dropout_rate": params.get("params", {}).get("dropout_rate"),
        "encoder_dims": str(params.get("params", {}).get("encoder_dims")),
        "lambda_cls": params.get("params", {}).get("lambda_cls"),
        "use_class_weight": params.get("params", {}).get("use_class_weight"),
        "tsne_image_path": tsne_path,
    }
    # Backward compatibility for old reports with split source/target KMeans fields.
    if row["kmeans_k"] is None:
        row["kmeans_k"] = merged_metrics.get("source_kmeans_k")
    if row["kmeans_ari"] is None:
        row["kmeans_ari"] = merged_metrics.get("source_kmeans_ari")
    if row["kmeans_nmi"] is None:
        row["kmeans_nmi"] = merged_metrics.get("source_kmeans_nmi")
    if row["kmeans_silhouette"] is None:
        row["kmeans_silhouette"] = merged_metrics.get("source_kmeans_silhouette")
    if row["kmeans_calinski_harabasz"] is None:
        row["kmeans_calinski_harabasz"] = merged_metrics.get("source_kmeans_calinski_harabasz")
    if row["kmeans_davies_bouldin"] is None:
        row["kmeans_davies_bouldin"] = merged_metrics.get("source_kmeans_davies_bouldin")

    pretrain_train_last = _read_last_csv_row(os.path.join(exp_dir, "pretrain_loss.csv"))
    pretrain_eval_last = _read_last_csv_row(os.path.join(exp_dir, "pretrain_eval_loss.csv"))
    gan_d_last = _read_last_csv_row(os.path.join(exp_dir, "d_loss.csv"))
    gan_g_last = _read_last_csv_row(os.path.join(exp_dir, "g_loss.csv"))
    _attach_prefixed_metrics(row, pretrain_train_last, "final_pretrain_train_")
    _attach_prefixed_metrics(row, pretrain_eval_last, "final_pretrain_eval_")
    _attach_prefixed_metrics(row, gan_d_last, "final_gan_d_")
    _attach_prefixed_metrics(row, gan_g_last, "final_gan_g_")

    if "metrics" in run_summary:
        row["run_summary_exists"] = True
    return row


def _to_html_table(df, output_path, title, subset_start=1):
    html = "<html><head><style>"
    html += "body { font-family: Arial, sans-serif; margin: 20px; }"
    html += "table { border-collapse: collapse; width: 100%; }"
    html += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }"
    html += "th { background-color: #f2f2f2; position: sticky; top: 0; }"
    html += "img.tsne { max-width: 350px; }"
    html += "tr:nth-child(even) { background-color: #f9f9f9; }"
    html += "</style></head><body>"
    html += f"<h1>{title}</h1>"
    html += "<table><thead><tr>"
    display_cols = [
        "ID", "fid", "mmd", "wasserstein", "best_gan_epoch", "best_gan_loss",
        "kmeans_k", "kmeans_ari", "kmeans_nmi",
        "kmeans_silhouette", "kmeans_calinski_harabasz", "kmeans_davies_bouldin",
        "final_pretrain_train_ortholoss", "final_pretrain_train_pVAE_loss",
        "final_pretrain_train_VAE_loss", "final_pretrain_train_cls_loss",
        "final_pretrain_eval_ortholoss", "final_pretrain_eval_pVAE_loss",
        "final_pretrain_eval_VAE_loss", "final_pretrain_eval_cls_loss",
        "final_pretrain_eval_total_loss",
        "final_gan_d_discrim_loss", "final_gan_d_g_p",
        "final_gan_g_ortho_loss", "final_gan_g_pvae_loss",
        "final_gan_g_gen_loss", "final_gan_g_vae_loss", "final_gan_g_cls_loss",
        "final_gan_g_temp_loss", "final_gan_g_early_stop_score",
        "pretrain_num_epochs", "train_num_epochs", "pretrain_learning_rate",
        "gan_learning_rate", "dropout_rate", "encoder_dims", "lambda_cls", "use_class_weight"
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    for col in display_cols:
        html += f"<th>{col}</th>"
    html += "<th>tSNE (GAN best)</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr>"
        for col in display_cols:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.6f}"
            html += f"<td>{val}</td>"
        html += "<td>"
        img_path = row.get("tsne_image_path")
        if isinstance(img_path, str) and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            html += f'<img class="tsne" src="data:image/png;base64,{encoded}" />'
        else:
            html += "N/A"
        html += "</td>"
        html += "</tr>"
    html += "</tbody></table></body></html>"
    with open(output_path, "w") as f:
        f.write(html)


def generate_html_report(df, output_dir, per_page=80):
    total = len(df)
    if total == 0:
        return
    pages = max(1, math.ceil(total / per_page))
    if pages == 1:
        _to_html_table(df, os.path.join(output_dir, "vaewc_report.html"), "VAEwC GAN Results")
        return
    index_html = "<html><body><h1>VAEwC GAN Results Index</h1><ul>"
    for i in range(pages):
        start = i * per_page
        end = min((i + 1) * per_page, total)
        sub = df.iloc[start:end]
        file_name = f"vaewc_report_part_{i+1}.html"
        _to_html_table(sub, os.path.join(output_dir, file_name), f"VAEwC GAN Results ({start+1}-{end})", subset_start=start + 1)
        index_html += f'<li><a href="{file_name}">{file_name}</a></li>'
    index_html += "</ul></body></html>"
    with open(os.path.join(output_dir, "vaewc_report_index.html"), "w") as f:
        f.write(index_html)


def _rank_normalized(series: pd.Series, ascending: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    out = pd.Series(0.0, index=s.index, dtype=float)
    if valid.sum() == 0:
        return out
    ranks = s[valid].rank(method="average", ascending=ascending)
    n = float(valid.sum())
    if n <= 1:
        out.loc[valid] = 1.0
        return out
    out.loc[valid] = 1.0 - (ranks - 1.0) / (n - 1.0)
    return out


def add_combined_scores(df: pd.DataFrame, deconf_weight: float, kmeans_weight: float) -> pd.DataFrame:
    out = df.copy()
    deconf_metrics = [("fid", True), ("mmd", True), ("wasserstein", True)]
    kmeans_metrics = [
        ("kmeans_ari", False),
        ("kmeans_nmi", False),
        ("kmeans_silhouette", False),
        ("kmeans_calinski_harabasz", False),
        ("kmeans_davies_bouldin", True),
    ]

    for col, asc in deconf_metrics + kmeans_metrics:
        out[f"score_{col}"] = _rank_normalized(out[col], ascending=asc) if col in out.columns else 0.0

    deconf_cols = [f"score_{c}" for c, _ in deconf_metrics]
    kmeans_cols = [f"score_{c}" for c, _ in kmeans_metrics]
    out["score_deconfounding"] = out[deconf_cols].mean(axis=1)
    out["score_kmeans"] = out[kmeans_cols].mean(axis=1)

    w_sum = float(deconf_weight + kmeans_weight)
    if w_sum <= 0:
        deconf_w = 0.5
        kmeans_w = 0.5
    else:
        deconf_w = float(deconf_weight) / w_sum
        kmeans_w = float(kmeans_weight) / w_sum
    out["score_total"] = deconf_w * out["score_deconfounding"] + kmeans_w * out["score_kmeans"]
    return out


def build_finetune_model_select(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    selected = df.head(max(1, int(top_k))).copy()
    selected["NO"] = ""
    selected["model_type"] = selected.get("model_type", "VAE")
    selected["pretrain_epochs"] = selected.get("pretrain_num_epochs")
    selected["train_epochs"] = selected.get("train_num_epochs")
    selected["pretrain_lr"] = selected.get("pretrain_learning_rate")
    selected["train_lr"] = selected.get("gan_learning_rate")
    selected["dropout"] = selected.get("dropout_rate")
    selected["result_folder"] = selected.get("ID")
    selected["selection_rank"] = range(1, len(selected) + 1)
    cols = [
        "ID", "NO", "model_type",
        "pretrain_epochs", "train_epochs", "pretrain_lr", "train_lr", "dropout",
        "encoder_dims", "lambda_cls", "use_class_weight",
        "fid", "mmd", "wasserstein",
        "kmeans_k", "kmeans_ari", "kmeans_nmi", "kmeans_silhouette",
        "kmeans_calinski_harabasz", "kmeans_davies_bouldin",
        "score_deconfounding", "score_kmeans", "score_total",
        "selection_rank", "result_folder",
    ]
    cols = [c for c in cols if c in selected.columns]
    return selected[cols]


def main():
    parser = argparse.ArgumentParser("visualize_vaewc_results")
    parser.add_argument("--result_dir", required=True, type=str, help="Directory with exp_XXX folders")
    parser.add_argument("--output_dir", default=None, type=str, help="Output dir for csv/html")
    parser.add_argument("--per_page", default=80, type=int, help="Rows per html page")
    parser.add_argument("--select_top_k", default=20, type=int, help="Top-K runs exported to model_select.csv for finetune")
    parser.add_argument("--deconf_weight", default=0.7, type=float, help="Weight of deconfounding score in total score")
    parser.add_argument("--kmeans_weight", default=0.3, type=float, help="Weight of kmeans score in total score")
    args = parser.parse_args()
    out = args.output_dir or args.result_dir
    os.makedirs(out, exist_ok=True)
    exp_dirs = sorted([d for d in glob(os.path.join(args.result_dir, "exp_*")) if os.path.isdir(d)])
    rows = [load_experiment_data(d) for d in exp_dirs]
    if not rows:
        print("No exp_* folders found.")
        return
    df = pd.DataFrame(rows)
    df = add_combined_scores(df, deconf_weight=args.deconf_weight, kmeans_weight=args.kmeans_weight)
    df = df.sort_values("score_total", ascending=False, na_position="last")
    csv_path = os.path.join(out, "aggregated_vaewc_results.csv")
    df.to_csv(csv_path, index=False)
    model_select_df = build_finetune_model_select(df, top_k=args.select_top_k)
    model_select_path = os.path.join(out, "model_select.csv")
    model_select_df.to_csv(model_select_path, index=False)
    generate_html_report(df, out, per_page=max(1, args.per_page))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved model_select for finetune: {model_select_path}")
    if len(df) <= args.per_page:
        print(f"Saved HTML: {os.path.join(out, 'vaewc_report.html')}")
    else:
        print(f"Saved HTML index: {os.path.join(out, 'vaewc_report_index.html')}")


if __name__ == "__main__":
    main()
