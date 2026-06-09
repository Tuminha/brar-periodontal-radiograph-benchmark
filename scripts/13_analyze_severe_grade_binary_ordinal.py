#!/usr/bin/env python3
"""Analyze binary severe-grade and ordinal BRAR signals across saved models."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
IMAGE_DIR = ROOT / "data" / "processed" / "image_baseline"
SENSITIVITY_DIR = ROOT / "data" / "processed" / "embedding_sensitivity"
NEGATIVE_CONTROL_PREDICTIONS = (
    ROOT / "data" / "processed" / "negative_controls" / "negative_control_predictions.csv"
)

OUTPUT_BINARY = REPORTS / "severe_grade_binary_metrics.csv"
OUTPUT_BINARY_SUMMARY = REPORTS / "severe_grade_binary_summary.csv"
OUTPUT_ORDINAL = REPORTS / "severe_grade_ordinal_metrics.csv"
OUTPUT_ORDINAL_SUMMARY = REPORTS / "severe_grade_ordinal_summary.csv"
OUTPUT_PAIRED = REPORTS / "severe_grade_paired_deltas.csv"
OUTPUT_MD = REPORTS / "severe_grade_binary_ordinal_summary.md"
OUTPUT_HTML = REPORTS / "severe_grade_binary_ordinal_report.html"
OUTPUT_JSON = REPORTS / "severe_grade_binary_ordinal_summary.json"

CLASSES = ["1", "2", "3"]
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES)}

PREDICTION_SPECS = [
    {
        "model_id": "image_efficientnet_b0",
        "path": IMAGE_DIR / "frozen_efficientnet_b0_384x192_predictions.csv",
        "temperature_scale": True,
        "kind": "image_only",
        "deployment_role": "primary image-only baseline",
    },
    {
        "model_id": "image_resnet50",
        "path": IMAGE_DIR / "frozen_resnet50_384x192_predictions.csv",
        "temperature_scale": True,
        "kind": "image_only",
        "deployment_role": "predeclared stronger encoder check",
    },
    {
        "model_id": "image_tile_efficientnet_b0_meanmax",
        "path": IMAGE_DIR / "tile_efficientnet_b0_384_meanmax_predictions.csv",
        "temperature_scale": True,
        "kind": "image_only",
        "deployment_role": "tile-based image-only baseline",
    },
    {
        "model_id": "image_plus_age_sex",
        "path": SENSITIVITY_DIR / "efficientnet_b0_384x192_sensitivity_predictions.csv",
        "temperature_scale": True,
        "kind": "metadata_sensitivity",
        "feature_set": "image_embedding_plus_age_sex",
        "deployment_role": "allowed metadata sensitivity",
    },
    {
        "model_id": "image_plus_downstream_age_sex_upper_bound",
        "path": SENSITIVITY_DIR / "efficientnet_b0_384x192_sensitivity_predictions.csv",
        "temperature_scale": True,
        "kind": "upper_bound",
        "feature_set": "image_embedding_plus_downstream_age_sex_upper_bound",
        "deployment_role": "non-deployment upper-bound sensitivity",
    },
]

NEGATIVE_MODEL_SPECS = [
    {
        "model_id": "age_sex",
        "model": "multinomial_logistic",
        "feature_set": "age_sex",
        "kind": "metadata_guardrail",
        "deployment_role": "metadata guardrail",
    },
    {
        "model_id": "downstream_plus_age_sex",
        "model": "multinomial_logistic",
        "feature_set": "downstream_plus_age_sex",
        "kind": "upper_bound",
        "deployment_role": "non-deployment upper-bound sensitivity",
    },
    {
        "model_id": "majority_class",
        "model": "majority_class",
        "feature_set": "class_prior",
        "kind": "baseline",
        "deployment_role": "minimum class-prior baseline",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        probs = softmax(logits / temperature)
        return float(log_loss(y_true, probs, labels=[0, 1, 2]))

    result = minimize_scalar(objective, bounds=(math.log(0.2), math.log(5.0)), method="bounded")
    if not result.success:
        return 1.0
    return float(math.exp(result.x))


def binned_ece(correct: np.ndarray, confidence: np.ndarray, bins: int = 10) -> float:
    ece = 0.0
    for bin_idx in range(bins):
        lower = bin_idx / bins
        upper = (bin_idx + 1) / bins
        if bin_idx == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if np.any(mask):
            ece += float(np.mean(mask) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))))
    return ece


def ece_binary(y_true: np.ndarray, positive_prob: np.ndarray, threshold: float, bins: int = 10) -> float:
    pred = (positive_prob >= threshold).astype(int)
    confidence = np.where(pred == 1, positive_prob, 1.0 - positive_prob)
    correct = (pred == y_true).astype(float)
    return binned_ece(correct, confidence, bins=bins)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    observed = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(float)
    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(1.0, observed.sum())
    weights = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            weights[i, j] = ((i - j) ** 2) / 4.0
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    return 1.0 - numerator / denominator if denominator else 0.0


def probability_frame_from_rows(
    rows: list[dict[str, str]],
    model_id: str,
    kind: str,
    deployment_role: str,
    temperature_scale: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    out_frames = []
    grouped = frame.groupby(["repeat", "seed", "fold", "eval_split"], sort=True)
    temperatures: dict[tuple[str, str, str], float] = {}
    if temperature_scale:
        for (repeat, seed, fold, eval_split), group in grouped:
            if eval_split != "val":
                continue
            logits = group[[f"logit_{label}" for label in CLASSES]].astype(float).to_numpy()
            y_true = group["y_true"].map(CLASS_TO_INDEX).astype(int).to_numpy()
            temperatures[(str(repeat), str(seed), str(fold))] = fit_temperature(logits, y_true)

    for (repeat, seed, fold, eval_split), group in frame.groupby(["repeat", "seed", "fold", "eval_split"], sort=True):
        if temperature_scale:
            temperature = temperatures[(str(repeat), str(seed), str(fold))]
            logits = group[[f"logit_{label}" for label in CLASSES]].astype(float).to_numpy()
            probs = softmax(logits / temperature)
        else:
            temperature = 1.0
            probs = normalize_probabilities(group[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy())
        out = group[["repeat", "seed", "fold", "eval_split", "file_name", "y_true"]].copy()
        out["model_id"] = model_id
        out["kind"] = kind
        out["deployment_role"] = deployment_role
        out["temperature"] = temperature
        for idx, label in enumerate(CLASSES):
            out[f"prob_{label}"] = probs[:, idx]
        out_frames.append(out)
    return pd.concat(out_frames, ignore_index=True)


def load_model_frames() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in PREDICTION_SPECS:
        path = Path(spec["path"])
        if not path.exists():
            continue
        rows = read_csv(path)
        feature_set = spec.get("feature_set")
        if feature_set is not None:
            rows = [row for row in rows if row.get("feature_set") == feature_set]
        frames.append(
            probability_frame_from_rows(
                rows,
                model_id=str(spec["model_id"]),
                kind=str(spec["kind"]),
                deployment_role=str(spec["deployment_role"]),
                temperature_scale=bool(spec["temperature_scale"]),
            )
        )

    negative_rows = read_csv(NEGATIVE_CONTROL_PREDICTIONS)
    for spec in NEGATIVE_MODEL_SPECS:
        rows = [
            row
            for row in negative_rows
            if row["model"] == spec["model"] and row["feature_set"] == spec["feature_set"]
        ]
        frames.append(
            probability_frame_from_rows(
                rows,
                model_id=str(spec["model_id"]),
                kind=str(spec["kind"]),
                deployment_role=str(spec["deployment_role"]),
                temperature_scale=False,
            )
        )
    return pd.concat(frames, ignore_index=True)


def binary_target(labels: pd.Series, task: str) -> np.ndarray:
    y = labels.astype(int).to_numpy()
    if task == "severe_level_3_vs_1_2":
        return (y == 3).astype(int)
    if task == "any_brar_level_2_3_vs_1":
        return (y >= 2).astype(int)
    raise ValueError(f"unknown task: {task}")


def positive_probability(frame: pd.DataFrame, task: str) -> np.ndarray:
    if task == "severe_level_3_vs_1_2":
        return frame["prob_3"].astype(float).to_numpy()
    if task == "any_brar_level_2_3_vs_1":
        return (frame["prob_2"].astype(float) + frame["prob_3"].astype(float)).to_numpy()
    raise ValueError(f"unknown task: {task}")


def best_validation_threshold(y_true: np.ndarray, positive_prob: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], positive_prob]))
    if len(candidates) > 301:
        candidates = np.linspace(0.0, 1.0, 301)
    best_threshold = 0.5
    best_ranking: tuple[float, float, float] | None = None
    for threshold in candidates:
        pred = (positive_prob >= threshold).astype(int)
        if len(np.unique(y_true)) < 2:
            continue
        sensitivity = recall_score(y_true, pred, zero_division=0)
        specificity = recall_score(1 - y_true, 1 - pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        ranking = (sensitivity + specificity - 1.0, f1, -abs(float(threshold) - 0.5))
        if best_ranking is None or ranking > best_ranking:
            best_ranking = ranking
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(prob_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tasks = ["severe_level_3_vs_1_2", "any_brar_level_2_3_vs_1"]
    for (model_id, repeat, seed, fold), group in prob_frame.groupby(["model_id", "repeat", "seed", "fold"], sort=True):
        val = group[group["eval_split"] == "val"]
        test = group[group["eval_split"] == "test"]
        if val.empty or test.empty:
            continue
        kind = str(group["kind"].iloc[0])
        deployment_role = str(group["deployment_role"].iloc[0])
        for task in tasks:
            y_val = binary_target(val["y_true"], task)
            p_val = positive_probability(val, task)
            threshold = best_validation_threshold(y_val, p_val)

            y_test = binary_target(test["y_true"], task)
            p_test = positive_probability(test, task)
            pred = (p_test >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
            try:
                auroc = float(roc_auc_score(y_test, p_test))
            except ValueError:
                auroc = math.nan
            try:
                ap = float(average_precision_score(y_test, p_test))
            except ValueError:
                ap = math.nan
            rows.append(
                {
                    "task": task,
                    "model_id": model_id,
                    "kind": kind,
                    "deployment_role": deployment_role,
                    "repeat": int(repeat),
                    "seed": int(seed),
                    "fold": int(fold),
                    "threshold": threshold,
                    "n": int(len(y_test)),
                    "positive_n": int(y_test.sum()),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "f1": float(f1_score(y_test, pred, zero_division=0)),
                    "precision": float(precision_score(y_test, pred, zero_division=0)),
                    "sensitivity": float(recall_score(y_test, pred, zero_division=0)),
                    "specificity": float(tn / (tn + fp)) if (tn + fp) else math.nan,
                    "log_loss": float(log_loss(y_test, np.column_stack([1.0 - p_test, p_test]), labels=[0, 1])),
                    "ece_10": float(ece_binary(y_test, p_test, threshold)),
                    "auroc": auroc,
                    "average_precision": ap,
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                }
            )
    return rows


def ordinal_metrics(prob_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (model_id, repeat, seed, fold), group in prob_frame[prob_frame["eval_split"] == "test"].groupby(
        ["model_id", "repeat", "seed", "fold"],
        sort=True,
    ):
        probs = group[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
        y_true = group["y_true"].map(CLASS_TO_INDEX).astype(int).to_numpy()
        y_pred = np.argmax(probs, axis=1)
        abs_error = np.abs(y_true - y_pred)
        severe_y = (y_true == 2).astype(int)
        severe_p = probs[:, 2]
        try:
            severe_auroc = float(roc_auc_score(severe_y, severe_p))
        except ValueError:
            severe_auroc = math.nan
        rows.append(
            {
                "model_id": model_id,
                "kind": str(group["kind"].iloc[0]),
                "deployment_role": str(group["deployment_role"].iloc[0]),
                "repeat": int(repeat),
                "seed": int(seed),
                "fold": int(fold),
                "n": int(len(group)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "ordinal_mae": float(np.mean(abs_error)),
                "adjacent_accuracy": float(np.mean(abs_error <= 1)),
                "two_grade_error_rate": float(np.mean(abs_error == 2)),
                "quadratic_weighted_kappa": float(quadratic_weighted_kappa(y_true, y_pred)),
                "severe_auroc_from_prob_3": severe_auroc,
            }
        )
    return rows


def aggregate(rows: list[dict[str, object]], group_cols: list[str], metric_cols: list[str]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    out: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        item = {col: value for col, value in zip(group_cols, key, strict=True)}
        item["folds"] = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            item[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            item[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out.append(item)
    return out


def paired_deltas(binary_rows: list[dict[str, object]], ordinal_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    comparisons = [
        ("image_efficientnet_b0", "age_sex"),
        ("image_tile_efficientnet_b0_meanmax", "image_efficientnet_b0"),
        ("image_tile_efficientnet_b0_meanmax", "age_sex"),
        ("image_plus_age_sex", "age_sex"),
    ]
    binary_frame = pd.DataFrame(binary_rows)
    ordinal_frame = pd.DataFrame(ordinal_rows)
    for task in sorted(binary_frame["task"].unique()):
        task_frame = binary_frame[binary_frame["task"] == task]
        for left, right in comparisons:
            left_frame = task_frame[task_frame["model_id"] == left].set_index(["repeat", "fold"])
            right_frame = task_frame[task_frame["model_id"] == right].set_index(["repeat", "fold"])
            common = left_frame.index.intersection(right_frame.index)
            for metric in ["balanced_accuracy", "sensitivity", "specificity", "f1", "auroc", "log_loss", "ece_10"]:
                delta = left_frame.loc[common, metric].astype(float) - right_frame.loc[common, metric].astype(float)
                out.append(
                    {
                        "analysis": "binary",
                        "task": task,
                        "left_model": left,
                        "right_model": right,
                        "metric": metric,
                        "paired_folds": int(len(delta)),
                        "mean_delta": float(delta.mean()) if len(delta) else math.nan,
                        "sd_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                        "left_better_folds": int((delta < 0).sum()) if metric in {"log_loss", "ece_10"} else int((delta > 0).sum()),
                    }
                )
    for left, right in comparisons:
        left_frame = ordinal_frame[ordinal_frame["model_id"] == left].set_index(["repeat", "fold"])
        right_frame = ordinal_frame[ordinal_frame["model_id"] == right].set_index(["repeat", "fold"])
        common = left_frame.index.intersection(right_frame.index)
        for metric in ["macro_f1", "ordinal_mae", "two_grade_error_rate", "quadratic_weighted_kappa"]:
            delta = left_frame.loc[common, metric].astype(float) - right_frame.loc[common, metric].astype(float)
            out.append(
                {
                    "analysis": "ordinal",
                    "task": "three_class_ordinal",
                    "left_model": left,
                    "right_model": right,
                    "metric": metric,
                    "paired_folds": int(len(delta)),
                    "mean_delta": float(delta.mean()) if len(delta) else math.nan,
                    "sd_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                    "left_better_folds": int((delta < 0).sum()) if metric in {"ordinal_mae", "two_grade_error_rate"} else int((delta > 0).sum()),
                }
            )
    return out


def format_float(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(format_float(value) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(
    binary_summary: list[dict[str, object]],
    ordinal_summary: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
) -> str:
    severe_rows = [
        row
        for row in binary_summary
        if row["task"] == "severe_level_3_vs_1_2"
    ]
    severe_rows = sorted(severe_rows, key=lambda row: float(row["balanced_accuracy_mean"]), reverse=True)
    ordinal_rows = sorted(ordinal_summary, key=lambda row: float(row["quadratic_weighted_kappa_mean"]), reverse=True)
    severe_deltas = [
        row
        for row in paired_rows
        if row["analysis"] == "binary"
        and row["task"] == "severe_level_3_vs_1_2"
        and row["metric"] in {"balanced_accuracy", "sensitivity", "specificity", "auroc"}
        and row["paired_folds"] > 0
    ]

    best = severe_rows[0] if severe_rows else {}
    image_severe_rows = [row for row in severe_rows if row["kind"] == "image_only"]
    best_image = image_severe_rows[0] if image_severe_rows else {}
    best_ordinal = ordinal_rows[0] if ordinal_rows else {}
    recommendation = "Review severe-grade and ordinal model rankings before manuscript drafting"
    if best_image and best_ordinal.get("model_id") == "image_tile_efficientnet_b0_meanmax":
        recommendation = "Use tile EfficientNet-B0 as the new leading image-only benchmark"
    if best and best.get("kind") in {"metadata_sensitivity", "upper_bound"} and best_image:
        recommendation += "; keep metadata/downstream models as sensitivity analyses"

    return f"""# Severe-Grade Binary And Ordinal Analysis

Date: 2026-06-08

## Recommendation

**{recommendation}.**

The severe-grade endpoint is defined as BRAR Level 3 versus Levels 1-2. Thresholds are selected on validation folds only, then evaluated on held-out test folds. Metadata and downstream models remain guardrails or sensitivity analyses, not primary image models.

Best severe-grade model by balanced accuracy: `{best.get("model_id", "NA")}`. Best image-only severe-grade model: `{best_image.get("model_id", "NA")}`. Best ordinal three-class model by quadratic weighted kappa: `{best_ordinal.get("model_id", "NA")}`.

## Severe-Grade Test Summary

{markdown_table(severe_rows, ["model_id", "kind", "folds", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean", "f1_mean", "auroc_mean", "ece_10_mean"])}

## Ordinal Three-Class Summary

{markdown_table(ordinal_rows, ["model_id", "kind", "folds", "macro_f1_mean", "ordinal_mae_mean", "two_grade_error_rate_mean", "quadratic_weighted_kappa_mean", "severe_auroc_from_prob_3_mean"])}

## Key Paired Deltas

Positive deltas favor the left model, except for log loss, ECE, ordinal MAE, and two-grade error rate.

{markdown_table(severe_deltas, ["left_model", "right_model", "metric", "paired_folds", "mean_delta", "sd_delta", "left_better_folds"])}
"""


def render_html(
    binary_summary: list[dict[str, object]],
    ordinal_summary: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
) -> str:
    def df_html(rows: list[dict[str, object]], columns: list[str]) -> str:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return "<p>No rows.</p>"
        return frame[columns].to_html(index=False, escape=True, float_format=lambda value: f"{value:.4f}")

    severe_rows = sorted(
        [row for row in binary_summary if row["task"] == "severe_level_3_vs_1_2"],
        key=lambda row: float(row["balanced_accuracy_mean"]),
        reverse=True,
    )
    any_rows = sorted(
        [row for row in binary_summary if row["task"] == "any_brar_level_2_3_vs_1"],
        key=lambda row: float(row["balanced_accuracy_mean"]),
        reverse=True,
    )
    ordinal_rows = sorted(ordinal_summary, key=lambda row: float(row["quadratic_weighted_kappa_mean"]), reverse=True)
    paired_display = [
        row
        for row in paired_rows
        if row["paired_folds"] > 0
        and row["metric"] in {"balanced_accuracy", "sensitivity", "specificity", "auroc", "macro_f1", "quadratic_weighted_kappa"}
    ]

    binary_cols = ["model_id", "kind", "folds", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean", "f1_mean", "auroc_mean", "average_precision_mean", "ece_10_mean"]
    ordinal_cols = ["model_id", "kind", "folds", "macro_f1_mean", "balanced_accuracy_mean", "ordinal_mae_mean", "adjacent_accuracy_mean", "two_grade_error_rate_mean", "quadratic_weighted_kappa_mean", "severe_auroc_from_prob_3_mean"]
    paired_cols = ["analysis", "task", "left_model", "right_model", "metric", "paired_folds", "mean_delta", "sd_delta", "left_better_folds"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BRAR Severe-Grade Binary/Ordinal Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
    main {{ max-width: 1240px; margin: 0 auto; }}
    h1, h2 {{ line-height: 1.2; }}
    .callout {{ border-left: 4px solid #25614c; background: #f1f8f5; padding: 14px 16px; margin: 16px 0 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d7dce2; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef1f4; }}
    .note {{ color: #5f6368; }}
  </style>
</head>
<body>
<main>
  <h1>BRAR Severe-Grade Binary/Ordinal Report</h1>
  <div class="callout">Endpoint: Level 3 versus Levels 1-2, with thresholds selected on validation folds only. Image-only models remain the primary deployment-relevant analyses.</div>

  <h2>Severe Grade: Level 3 vs Levels 1-2</h2>
  {df_html(severe_rows, binary_cols)}

  <h2>Any BRAR Abnormality: Levels 2-3 vs Level 1</h2>
  <p class="note">Secondary contrast included to show where age/sex metadata dominates.</p>
  {df_html(any_rows, binary_cols)}

  <h2>Ordinal Three-Class Metrics</h2>
  {df_html(ordinal_rows, ordinal_cols)}

  <h2>Paired Deltas</h2>
  <p class="note">Positive deltas favor the left model, except for metrics where lower is better.</p>
  {df_html(paired_display, paired_cols)}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    prob_frame = load_model_frames()
    binary_rows = binary_metrics(prob_frame)
    ordinal_rows = ordinal_metrics(prob_frame)
    binary_summary = aggregate(
        binary_rows,
        ["task", "model_id", "kind", "deployment_role"],
        [
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "sensitivity",
            "specificity",
            "log_loss",
            "ece_10",
            "auroc",
            "average_precision",
        ],
    )
    ordinal_summary = aggregate(
        ordinal_rows,
        ["model_id", "kind", "deployment_role"],
        [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "ordinal_mae",
            "adjacent_accuracy",
            "two_grade_error_rate",
            "quadratic_weighted_kappa",
            "severe_auroc_from_prob_3",
        ],
    )
    paired_rows = paired_deltas(binary_rows, ordinal_rows)

    write_csv(
        OUTPUT_BINARY,
        binary_rows,
        [
            "task",
            "model_id",
            "kind",
            "deployment_role",
            "repeat",
            "seed",
            "fold",
            "threshold",
            "n",
            "positive_n",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "sensitivity",
            "specificity",
            "log_loss",
            "ece_10",
            "auroc",
            "average_precision",
            "tp",
            "fp",
            "tn",
            "fn",
        ],
    )
    write_csv(
        OUTPUT_BINARY_SUMMARY,
        binary_summary,
        [
            "task",
            "model_id",
            "kind",
            "deployment_role",
            "folds",
            "threshold_mean",
            "threshold_sd",
            "accuracy_mean",
            "accuracy_sd",
            "balanced_accuracy_mean",
            "balanced_accuracy_sd",
            "f1_mean",
            "f1_sd",
            "precision_mean",
            "precision_sd",
            "sensitivity_mean",
            "sensitivity_sd",
            "specificity_mean",
            "specificity_sd",
            "log_loss_mean",
            "log_loss_sd",
            "ece_10_mean",
            "ece_10_sd",
            "auroc_mean",
            "auroc_sd",
            "average_precision_mean",
            "average_precision_sd",
        ],
    )
    write_csv(
        OUTPUT_ORDINAL,
        ordinal_rows,
        [
            "model_id",
            "kind",
            "deployment_role",
            "repeat",
            "seed",
            "fold",
            "n",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "ordinal_mae",
            "adjacent_accuracy",
            "two_grade_error_rate",
            "quadratic_weighted_kappa",
            "severe_auroc_from_prob_3",
        ],
    )
    write_csv(
        OUTPUT_ORDINAL_SUMMARY,
        ordinal_summary,
        [
            "model_id",
            "kind",
            "deployment_role",
            "folds",
            "accuracy_mean",
            "accuracy_sd",
            "balanced_accuracy_mean",
            "balanced_accuracy_sd",
            "macro_f1_mean",
            "macro_f1_sd",
            "ordinal_mae_mean",
            "ordinal_mae_sd",
            "adjacent_accuracy_mean",
            "adjacent_accuracy_sd",
            "two_grade_error_rate_mean",
            "two_grade_error_rate_sd",
            "quadratic_weighted_kappa_mean",
            "quadratic_weighted_kappa_sd",
            "severe_auroc_from_prob_3_mean",
            "severe_auroc_from_prob_3_sd",
        ],
    )
    write_csv(
        OUTPUT_PAIRED,
        paired_rows,
        [
            "analysis",
            "task",
            "left_model",
            "right_model",
            "metric",
            "paired_folds",
            "mean_delta",
            "sd_delta",
            "left_better_folds",
        ],
    )
    OUTPUT_MD.write_text(render_markdown(binary_summary, ordinal_summary, paired_rows), encoding="utf-8")
    OUTPUT_HTML.write_text(render_html(binary_summary, ordinal_summary, paired_rows), encoding="utf-8")
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "binary_summary": binary_summary,
                "ordinal_summary": ordinal_summary,
                "paired_deltas": paired_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"binary metrics: {OUTPUT_BINARY}")
    print(f"binary summary: {OUTPUT_BINARY_SUMMARY}")
    print(f"ordinal metrics: {OUTPUT_ORDINAL}")
    print(f"ordinal summary: {OUTPUT_ORDINAL_SUMMARY}")
    print(f"paired deltas: {OUTPUT_PAIRED}")
    print(f"summary: {OUTPUT_MD}")
    print(f"html report: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
