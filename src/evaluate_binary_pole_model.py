"""Evaluate binary pole model output against reviewed ground truth.

This pipeline is designed for the Clear / Critical workflow:

- Ground truth uses a reviewed binary label column (typically ``manual_risk``)
- Model output uses a prediction label column (typically ``status``)
- Only poles present in the ground-truth CSV are used for evaluation
- Rows are matched by image name first, then rounded coordinates

Outputs:
- matched_ground_truth_vs_model.csv
- ground_truth_missing_in_model.csv
- model_rows_not_used.csv
- confusion_matrix.csv
- metrics.csv
- confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


GROUND_TRUTH_IMAGE_COLUMNS = ("image_name", "filename", "image")
MODEL_IMAGE_COLUMNS = ("filename", "image_name", "image", "name")
GROUND_TRUTH_LAT_COLUMNS = ("pole_lat", "lat", "latitude")
GROUND_TRUTH_LON_COLUMNS = ("pole_lng", "lon", "lng", "longitude")
MODEL_LAT_COLUMNS = ("lat", "latitude", "pole_lat")
MODEL_LON_COLUMNS = ("lon", "lng", "longitude", "pole_lng")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Clear/Critical model CSV against reviewed ground truth."
    )
    parser.add_argument("--ground-truth-csv", required=True, help="Reviewed GT CSV path.")
    parser.add_argument("--model-csv", required=True, help="Model output CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory for saved reports.")
    parser.add_argument(
        "--ground-truth-label-col",
        default="manual_risk",
        help="Binary GT label column, usually manual_risk.",
    )
    parser.add_argument(
        "--model-label-col",
        default="status",
        help="Prediction label column in the model CSV.",
    )
    parser.add_argument(
        "--coord-decimals",
        type=int,
        default=6,
        help="Number of decimal places used for coordinate matching.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title for the confusion-matrix plot. Defaults to the model CSV stem.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_binary_label(value: object) -> str:
    text = normalize_text(value)
    if text is None:
        return "Unknown"

    key = re.sub(r"\s+", " ", text).strip().lower()
    if key in {"clear", "safe"}:
        return "Clear"
    if key in {"critical", "risk"}:
        return "Critical"
    return "Unknown"


def normalize_image_name(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    name = Path(text).name.strip().lower()
    for ext in IMAGE_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_coord_key(lat: object, lon: object, decimals: int) -> str | None:
    lat_f = safe_float(lat)
    lon_f = safe_float(lon)
    if lat_f is None or lon_f is None:
        return None
    return f"{lat_f:.{decimals}f}_{lon_f:.{decimals}f}"


def first_present(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    frame_columns = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        found = frame_columns.get(candidate.lower())
        if found:
            return found
    return None


def build_maps_link(lat: object, lon: object) -> str | None:
    lat_f = safe_float(lat)
    lon_f = safe_float(lon)
    if lat_f is None or lon_f is None:
        return None
    return f"https://www.google.com/maps?q={lat_f},{lon_f}"


def prepare_ground_truth(
    frame: pd.DataFrame,
    label_column: str,
    coord_decimals: int,
) -> pd.DataFrame:
    out = frame.copy()
    out["ground_truth_row_id"] = out.index

    image_col = first_present(out, GROUND_TRUTH_IMAGE_COLUMNS)
    lat_col = first_present(out, GROUND_TRUTH_LAT_COLUMNS)
    lon_col = first_present(out, GROUND_TRUTH_LON_COLUMNS)

    if label_column not in out.columns:
        raise KeyError(
            f"Ground-truth label column '{label_column}' not found. "
            f"Available columns: {list(out.columns)}"
        )

    out["ground_truth_label"] = out[label_column].map(normalize_binary_label)
    out["ground_truth_image_key"] = out[image_col].map(normalize_image_name) if image_col else None
    out["ground_truth_coord_key"] = (
        out.apply(
            lambda row: build_coord_key(row[lat_col], row[lon_col], coord_decimals),
            axis=1,
        )
        if lat_col and lon_col
        else None
    )

    if "pole_id" in out.columns:
        pole_id_key = out["pole_id"].map(normalize_image_name)
        out["ground_truth_coord_key"] = out["ground_truth_coord_key"].fillna(pole_id_key)

    if lat_col and lon_col:
        out["ground_truth_maps_link"] = out.apply(
            lambda row: build_maps_link(row[lat_col], row[lon_col]),
            axis=1,
        )
    else:
        out["ground_truth_maps_link"] = None

    return out


def prepare_model(
    frame: pd.DataFrame,
    label_column: str,
    coord_decimals: int,
) -> pd.DataFrame:
    out = frame.copy()
    out["model_row_id"] = out.index

    image_col = first_present(out, MODEL_IMAGE_COLUMNS)
    lat_col = first_present(out, MODEL_LAT_COLUMNS)
    lon_col = first_present(out, MODEL_LON_COLUMNS)

    if label_column not in out.columns:
        raise KeyError(
            f"Model label column '{label_column}' not found. "
            f"Available columns: {list(out.columns)}"
        )

    out["model_label"] = out[label_column].map(normalize_binary_label)
    out["model_image_key"] = out[image_col].map(normalize_image_name) if image_col else None
    out["model_coord_key"] = (
        out.apply(
            lambda row: build_coord_key(row[lat_col], row[lon_col], coord_decimals),
            axis=1,
        )
        if lat_col and lon_col
        else None
    )
    return out


def dedupe_keys(frame: pd.DataFrame, key_column: str) -> pd.DataFrame:
    return frame[frame[key_column].notna()].drop_duplicates(subset=[key_column], keep="first").copy()


def match_ground_truth_to_model(
    ground_truth: pd.DataFrame,
    model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_by_image = dedupe_keys(model, "model_image_key")
    model_by_coord = dedupe_keys(model, "model_coord_key")

    used_model_ids: set[int] = set()
    matched_rows: list[dict[str, object]] = []

    for _, gt_row in ground_truth.iterrows():
        model_match = None
        match_method = "missing_in_model"

        image_key = gt_row.get("ground_truth_image_key")
        if image_key is not None:
            hits = model_by_image[
                (model_by_image["model_image_key"] == image_key)
                & (~model_by_image["model_row_id"].isin(used_model_ids))
            ]
            if not hits.empty:
                model_match = hits.iloc[0]
                match_method = "image_name"

        if model_match is None:
            coord_key = gt_row.get("ground_truth_coord_key")
            if coord_key is not None:
                hits = model_by_coord[
                    (model_by_coord["model_coord_key"] == coord_key)
                    & (~model_by_coord["model_row_id"].isin(used_model_ids))
                ]
                if not hits.empty:
                    model_match = hits.iloc[0]
                    match_method = "coordinates"

        row = gt_row.to_dict()
        row["match_method"] = match_method

        if model_match is None:
            row["model_row_id"] = pd.NA
            row["model_label"] = "Missing"
            row["labels_match"] = False
        else:
            used_model_ids.add(int(model_match["model_row_id"]))
            for column, value in model_match.items():
                if column in row:
                    row[f"model__{column}"] = value
                else:
                    row[column] = value
            row["labels_match"] = row["ground_truth_label"] == row["model_label"]

        matched_rows.append(row)

    comparison = pd.DataFrame(matched_rows)
    missing_in_model = comparison[comparison["match_method"] == "missing_in_model"].copy()
    model_unused = model[~model["model_row_id"].isin(used_model_ids)].copy()
    return comparison, missing_in_model, model_unused


def compute_metrics(compare_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    eval_df = compare_df[
        compare_df["ground_truth_label"].isin(["Clear", "Critical"])
        & compare_df["model_label"].isin(["Clear", "Critical"])
    ].copy()

    cm = pd.crosstab(
        eval_df["ground_truth_label"],
        eval_df["model_label"],
        rownames=["Ground Truth"],
        colnames=["Prediction"],
        dropna=False,
    ).reindex(index=["Clear", "Critical"], columns=["Clear", "Critical"], fill_value=0)

    tn = int(cm.loc["Clear", "Clear"])
    fp = int(cm.loc["Clear", "Critical"])
    fn = int(cm.loc["Critical", "Clear"])
    tp = int(cm.loc["Critical", "Critical"])
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if pd.notna(precision) and pd.notna(recall) and (precision + recall)
        else np.nan
    )

    metrics = {
        "accuracy": accuracy,
        "precision_critical": precision,
        "recall_critical": recall,
        "specificity_clear": specificity,
        "f1_critical": f1,
        "tp_critical": tp,
        "fn_critical": fn,
        "fp_critical": fp,
        "tn_clear": tn,
        "n_used": total,
    }

    metrics_df = pd.DataFrame(
        [
            {
                "metric": "Accuracy",
                "value": round(accuracy, 4) if pd.notna(accuracy) else np.nan,
            },
            {
                "metric": "Precision (Critical)",
                "value": round(precision, 4) if pd.notna(precision) else np.nan,
            },
            {
                "metric": "Recall (Critical)",
                "value": round(recall, 4) if pd.notna(recall) else np.nan,
            },
            {
                "metric": "Specificity (Clear)",
                "value": round(specificity, 4) if pd.notna(specificity) else np.nan,
            },
            {
                "metric": "F1 (Critical)",
                "value": round(f1, 4) if pd.notna(f1) else np.nan,
            },
            {
                "metric": "GT rows used",
                "value": int(total),
            },
        ]
    )

    return cm, metrics, metrics_df


def format_pct(value: float | int) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def plot_confusion_matrix(
    cm: pd.DataFrame,
    metrics: dict[str, float | int],
    output_path: Path,
    title: str,
) -> None:
    display_cm = cm.loc[["Critical", "Clear"], ["Critical", "Clear"]].copy()
    display_cm.index = ["Manual: Critical", "Manual: Clear"]
    display_cm.columns = ["Predicted: Critical", "Predicted: Clear"]

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    sns.heatmap(
        display_cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=1.5,
        linecolor="white",
        cbar=False,
        square=True,
        annot_kws={"fontsize": 18, "fontweight": "bold"},
        ax=ax,
    )

    title_text = (
        f"{title}\n"
        f"Acc: {format_pct(metrics['accuracy'])} | "
        f"Prec: {format_pct(metrics['precision_critical'])} | "
        f"Recall: {format_pct(metrics['recall_critical'])} | "
        f"F1: {format_pct(metrics['f1_critical'])}"
    )
    ax.set_title(
        title_text,
        fontsize=16,
        weight="bold",
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    comparison: pd.DataFrame,
    missing_in_model: pd.DataFrame,
    model_unused: pd.DataFrame,
    cm: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "matched_ground_truth_vs_model.csv", index=False)
    missing_in_model.to_csv(output_dir / "ground_truth_missing_in_model.csv", index=False)
    model_unused.to_csv(output_dir / "model_rows_not_used.csv", index=False)
    cm.to_csv(output_dir / "confusion_matrix.csv")
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)


def main() -> None:
    args = parse_args()

    ground_truth_path = Path(args.ground_truth_csv)
    model_path = Path(args.model_csv)
    output_dir = Path(args.output_dir)
    plot_title = args.title or model_path.stem.replace("_", " ")

    ground_truth_df = pd.read_csv(ground_truth_path)
    model_df = pd.read_csv(model_path)

    ground_truth = prepare_ground_truth(
        ground_truth_df, args.ground_truth_label_col, args.coord_decimals
    )
    model = prepare_model(model_df, args.model_label_col, args.coord_decimals)

    comparison, missing_in_model, model_unused = match_ground_truth_to_model(ground_truth, model)
    cm, metrics, metrics_df = compute_metrics(comparison)
    save_outputs(comparison, missing_in_model, model_unused, cm, metrics_df, output_dir)
    plot_confusion_matrix(cm, metrics, output_dir / "confusion_matrix.png", plot_title)

    summary = {
        "ground_truth_rows": int(len(ground_truth)),
        "model_rows": int(len(model)),
        "matched_rows": int((comparison["match_method"] != "missing_in_model").sum()),
        "ground_truth_missing_in_model": int(len(missing_in_model)),
        "model_rows_not_used": int(len(model_unused)),
        "metrics": {
            key: (round(value, 6) if isinstance(value, float) and pd.notna(value) else value)
            for key, value in metrics.items()
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Ground-truth rows: {len(ground_truth)}")
    print(f"Model rows: {len(model)}")
    print(f"Matched GT rows: {(comparison['match_method'] != 'missing_in_model').sum()}")
    print(f"GT rows missing in model: {len(missing_in_model)}")
    print(f"Model rows not used: {len(model_unused)}")
    print(f"Metrics saved to: {output_dir / 'metrics.csv'}")
    print(f"Confusion matrix saved to: {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
