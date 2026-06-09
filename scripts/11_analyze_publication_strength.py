#!/usr/bin/env python3
"""Analyze whether current BRAR results support a strong publication angle.

This script uses saved validation/test predictions. It does not train an image
model and does not inspect label-folder names as predictors.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
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
MANIFEST = ROOT / "data" / "processed" / "brar_manifest.csv"
IMAGE_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "image_baseline"
    / "frozen_efficientnet_b0_384x192_predictions.csv"
)
NEGATIVE_CONTROL_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "negative_controls"
    / "negative_control_predictions.csv"
)
REPORTS = ROOT / "reports"

MODEL_COMPARISON = REPORTS / "publication_model_comparison.csv"
PAIRED_DELTAS = REPORTS / "publication_paired_deltas.csv"
BINARY_METRICS = REPORTS / "publication_binary_task_metrics.csv"
SUBGROUP_METRICS = REPORTS / "publication_subgroup_metrics.csv"
CONFIDENT_ERRORS = REPORTS / "publication_confident_errors.csv"
SUMMARY_JSON = REPORTS / "publication_strength_summary.json"
SUMMARY_MD = REPORTS / "publication_strength_summary.md"
REPORT_HTML = REPORTS / "publication_strength_report.html"

CLASSES = ["1", "2", "3"]
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES)}
SIMPLE_NEGATIVE_MODELS = [
    "majority_class",
    "stratified_random",
]
NEGATIVE_FEATURE_SETS = [
    "age_sex",
    "image_geometry_file",
    "admin_index",
    "downstream_plus_age_sex",
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


def ece_multiclass(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    pred = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)
    correct = (pred == y_true).astype(float)
    return binned_ece(correct, confidence, bins=bins)


def ece_binary(y_true: np.ndarray, positive_prob: np.ndarray, bins: int = 10) -> float:
    pred = (positive_prob >= 0.5).astype(int)
    confidence = np.where(pred == 1, positive_prob, 1.0 - positive_prob)
    correct = (pred == y_true).astype(float)
    return binned_ece(correct, confidence, bins=bins)


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


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_classes = len(CLASSES)
    observed = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(1.0, observed.sum())
    weights = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    return 1.0 - numerator / denominator if denominator else 0.0


def balanced_accuracy_present_classes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(int(label) for label in np.unique(y_true))
    recalls = []
    for label in labels:
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else math.nan


def aggregate_mean_sd(rows: list[dict[str, object]], group_cols: list[str], metric_cols: list[str]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    out: list[dict[str, object]] = []
    if frame.empty:
        return out
    grouped = frame.groupby(group_cols, dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        item: dict[str, object] = {col: value for col, value in zip(group_cols, key, strict=True)}
        item["folds"] = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            item[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            item[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out.append(item)
    return sorted(out, key=lambda row: str(tuple(row[col] for col in group_cols)))


def multiclass_metric_row(
    model_id: str,
    repeat: str,
    seed: str,
    fold: str,
    eval_split: str,
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, object]:
    y_pred = np.argmax(probs, axis=1)
    try:
        auroc = float(roc_auc_score(pd.get_dummies(y_true).reindex(columns=[0, 1, 2], fill_value=0), probs, average="macro", multi_class="ovr"))
    except ValueError:
        auroc = math.nan
    return {
        "model_id": model_id,
        "repeat": int(repeat),
        "seed": int(seed),
        "fold": int(fold),
        "eval_split": eval_split,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "ece_10": float(ece_multiclass(y_true, probs)),
        "ordinal_mae": float(np.mean(np.abs(y_true - y_pred))),
        "quadratic_weighted_kappa": float(quadratic_weighted_kappa(y_true, y_pred)),
        "ovr_auroc": auroc,
    }


def image_prediction_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["repeat"], row["seed"], row["fold"], row["eval_split"])].append(row)

    temperatures: dict[tuple[str, str, str], float] = {}
    for repeat, seed, fold, eval_split in grouped:
        if eval_split != "val":
            continue
        group_rows = grouped[(repeat, seed, fold, eval_split)]
        logits = np.asarray([[float(row[f"logit_{label}"]) for label in CLASSES] for row in group_rows])
        y_true = np.asarray([CLASS_TO_INDEX[row["y_true"]] for row in group_rows], dtype=int)
        temperatures[(repeat, seed, fold)] = fit_temperature(logits, y_true)

    out: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), int(item[0][2]), item[0][3])):
        repeat, seed, fold, eval_split = key
        logits = np.asarray([[float(row[f"logit_{label}"]) for label in CLASSES] for row in group_rows])
        temperature = temperatures[(repeat, seed, fold)]
        probs = softmax(logits / temperature)
        y_true = np.asarray([CLASS_TO_INDEX[row["y_true"]] for row in group_rows], dtype=int)
        metric = multiclass_metric_row(
            "image_efficientnet_b0_temperature_scaled",
            repeat,
            seed,
            fold,
            eval_split,
            y_true,
            probs,
        )
        metric["temperature"] = temperature
        out.append(metric)
    return out


def negative_control_metric_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["model"] in SIMPLE_NEGATIVE_MODELS:
            model_id = row["model"]
        elif row["feature_set"] in NEGATIVE_FEATURE_SETS:
            model_id = row["feature_set"]
        else:
            continue
        grouped[(model_id, row["repeat"], row["seed"], row["fold"], row["eval_split"])].append(row)

    out: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]), int(item[0][3]), item[0][4])):
        model_id, repeat, seed, fold, eval_split = key
        probs = normalize_probabilities(np.asarray([[float(row[f"prob_{label}"]) for label in CLASSES] for row in group_rows]))
        y_true = np.asarray([CLASS_TO_INDEX[row["y_true"]] for row in group_rows], dtype=int)
        metric = multiclass_metric_row(model_id, repeat, seed, fold, eval_split, y_true, probs)
        metric["temperature"] = 1.0
        out.append(metric)
    return out


def to_model_comparison(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate = aggregate_mean_sd(
        [row for row in metrics if row["eval_split"] == "test"],
        ["model_id"],
        [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "log_loss",
            "ece_10",
            "ordinal_mae",
            "quadratic_weighted_kappa",
            "ovr_auroc",
        ],
    )
    return sorted(aggregate, key=lambda row: float(row["macro_f1_mean"]), reverse=True)


def paired_deltas(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame([row for row in metrics if row["eval_split"] == "test"])
    keys = ["repeat", "fold"]
    out: list[dict[str, object]] = []
    comparisons = [
        ("image_efficientnet_b0_temperature_scaled", "age_sex"),
        ("image_efficientnet_b0_temperature_scaled", "downstream_plus_age_sex"),
        ("image_efficientnet_b0_temperature_scaled", "majority_class"),
        ("age_sex", "majority_class"),
    ]
    for left, right in comparisons:
        left_frame = frame[frame["model_id"] == left].set_index(keys)
        right_frame = frame[frame["model_id"] == right].set_index(keys)
        common = left_frame.index.intersection(right_frame.index)
        for metric in ["macro_f1", "balanced_accuracy", "log_loss", "ece_10", "quadratic_weighted_kappa"]:
            delta = left_frame.loc[common, metric].astype(float) - right_frame.loc[common, metric].astype(float)
            out.append(
                {
                    "left_model": left,
                    "right_model": right,
                    "metric": metric,
                    "paired_folds": int(len(delta)),
                    "mean_delta": float(delta.mean()) if len(delta) else math.nan,
                    "sd_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                    "min_delta": float(delta.min()) if len(delta) else math.nan,
                    "max_delta": float(delta.max()) if len(delta) else math.nan,
                    "left_better_folds": int((delta > 0).sum()) if metric not in {"log_loss", "ece_10"} else int((delta < 0).sum()),
                }
            )
    return out


def class_probs_from_prediction_rows(
    rows: list[dict[str, str]],
    model_id: str,
    use_temperature: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    records: list[pd.DataFrame] = []
    for (repeat, seed, fold, eval_split), group in frame.groupby(["repeat", "seed", "fold", "eval_split"], sort=True):
        probs = group[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
        if use_temperature:
            if eval_split == "val":
                val_group = group
            else:
                val_group = frame[
                    (frame["repeat"] == repeat)
                    & (frame["seed"] == seed)
                    & (frame["fold"] == fold)
                    & (frame["eval_split"] == "val")
                ]
            logits_val = val_group[[f"logit_{label}" for label in CLASSES]].astype(float).to_numpy()
            y_val = val_group["y_true"].map(CLASS_TO_INDEX).astype(int).to_numpy()
            temperature = fit_temperature(logits_val, y_val)
            logits = group[[f"logit_{label}" for label in CLASSES]].astype(float).to_numpy()
            probs = softmax(logits / temperature)
        else:
            probs = normalize_probabilities(probs)
        out = group[["repeat", "seed", "fold", "eval_split", "file_name", "y_true"]].copy()
        out["model_id"] = model_id
        for idx, label in enumerate(CLASSES):
            out[f"prob_{label}"] = probs[:, idx]
        records.append(out)
    return pd.concat(records, ignore_index=True)


def combined_probability_frame(image_rows: list[dict[str, str]], negative_rows: list[dict[str, str]]) -> pd.DataFrame:
    frames = [
        class_probs_from_prediction_rows(
            image_rows,
            "image_efficientnet_b0_temperature_scaled",
            use_temperature=True,
        )
    ]
    negative_frame = pd.DataFrame(negative_rows)
    for model_name in SIMPLE_NEGATIVE_MODELS:
        model_rows = negative_frame[negative_frame["model"] == model_name]
        if model_rows.empty:
            continue
        frames.append(
            class_probs_from_prediction_rows(
                model_rows.to_dict("records"),
                model_name,
                use_temperature=False,
            )
        )
    for feature_set in NEGATIVE_FEATURE_SETS:
        feature_rows = negative_frame[negative_frame["feature_set"] == feature_set]
        if feature_rows.empty:
            continue
        frames.append(
            class_probs_from_prediction_rows(
                feature_rows.to_dict("records"),
                feature_set,
                use_temperature=False,
            )
        )
    return pd.concat(frames, ignore_index=True)


def binary_target(y_labels: pd.Series, task: str) -> np.ndarray:
    y = y_labels.astype(int).to_numpy()
    if task == "level_1_vs_higher":
        return (y >= 2).astype(int)
    if task == "level_3_vs_lower":
        return (y == 3).astype(int)
    raise ValueError(f"unknown binary task: {task}")


def binary_positive_probability(frame: pd.DataFrame, task: str) -> np.ndarray:
    if task == "level_1_vs_higher":
        return (frame["prob_2"].astype(float) + frame["prob_3"].astype(float)).to_numpy()
    if task == "level_3_vs_lower":
        return frame["prob_3"].astype(float).to_numpy()
    raise ValueError(f"unknown binary task: {task}")


def best_validation_threshold(y_true: np.ndarray, positive_prob: np.ndarray) -> float:
    candidates = np.unique(np.clip(positive_prob, 0.0, 1.0))
    if len(candidates) > 200:
        candidates = np.linspace(0.0, 1.0, 201)
    else:
        candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], candidates]))
    best_threshold = 0.5
    best_score: tuple[float, float, float] | None = None
    for threshold in candidates:
        pred = (positive_prob >= threshold).astype(int)
        if len(np.unique(y_true)) < 2:
            continue
        sensitivity = recall_score(y_true, pred, zero_division=0)
        specificity = recall_score(1 - y_true, 1 - pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        ranking = (sensitivity + specificity - 1.0, f1, -abs(float(threshold) - 0.5))
        if best_score is None or ranking > best_score:
            best_score = ranking
            best_threshold = float(threshold)
    return best_threshold


def binary_metric_rows(prob_frame: pd.DataFrame) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    tasks = ["level_1_vs_higher", "level_3_vs_lower"]
    for (model_id, repeat, seed, fold), group in prob_frame.groupby(["model_id", "repeat", "seed", "fold"], sort=True):
        val = group[group["eval_split"] == "val"]
        test = group[group["eval_split"] == "test"]
        if val.empty or test.empty:
            continue
        for task in tasks:
            y_val = binary_target(val["y_true"], task)
            p_val = binary_positive_probability(val, task)
            threshold = best_validation_threshold(y_val, p_val)

            y_test = binary_target(test["y_true"], task)
            p_test = binary_positive_probability(test, task)
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
            out.append(
                {
                    "model_id": model_id,
                    "repeat": int(repeat),
                    "seed": int(seed),
                    "fold": int(fold),
                    "task": task,
                    "threshold_source": "validation_youden",
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
                    "ece_10": float(ece_binary(y_test, p_test)),
                    "auroc": auroc,
                    "average_precision": ap,
                }
            )
    return out


def subgroup_metric_rows(prob_frame: pd.DataFrame, manifest: pd.DataFrame) -> list[dict[str, object]]:
    metadata = manifest[["file_name", "age", "gender"]].copy()
    metadata["age"] = pd.to_numeric(metadata["age"], errors="coerce")
    metadata["gender"] = metadata["gender"].astype(str)
    metadata["age_band"] = pd.cut(
        metadata["age"],
        bins=[-math.inf, 34, 50, math.inf],
        labels=["age_<35", "age_35_50", "age_>50"],
    ).astype(str)
    metadata["gender_group"] = "gender_" + metadata["gender"]
    frame = prob_frame[(prob_frame["eval_split"] == "test") & (prob_frame["model_id"].isin(["image_efficientnet_b0_temperature_scaled", "age_sex"]))].merge(
        metadata,
        on="file_name",
        how="left",
    )
    out: list[dict[str, object]] = []
    for subgroup_type, subgroup_col in [("age_band", "age_band"), ("gender", "gender_group")]:
        for (model_id, repeat, seed, fold, subgroup), group in frame.groupby(
            ["model_id", "repeat", "seed", "fold", subgroup_col],
            dropna=False,
            sort=True,
        ):
            if len(group) < 10:
                continue
            y_true = group["y_true"].map(CLASS_TO_INDEX).astype(int).to_numpy()
            probs = group[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
            y_pred = np.argmax(probs, axis=1)
            out.append(
                {
                    "model_id": model_id,
                    "repeat": int(repeat),
                    "seed": int(seed),
                    "fold": int(fold),
                    "subgroup_type": subgroup_type,
                    "subgroup": str(subgroup),
                    "n": int(len(group)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_present_classes(y_true, y_pred)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                }
            )
    return out


def confident_error_rows(prob_frame: pd.DataFrame, manifest: pd.DataFrame) -> list[dict[str, object]]:
    frame = prob_frame[
        (prob_frame["model_id"] == "image_efficientnet_b0_temperature_scaled")
        & (prob_frame["eval_split"] == "test")
    ].copy()
    probs = frame[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
    pred_idx = np.argmax(probs, axis=1)
    frame["pred_label"] = [CLASSES[idx] for idx in pred_idx]
    frame["confidence"] = probs.max(axis=1)
    frame["is_error"] = frame["pred_label"] != frame["y_true"].astype(str)
    metadata = manifest[["file_name", "age", "gender", "number_of_missing_teeth", "implant", "residual_root"]]
    frame = frame.merge(metadata, on="file_name", how="left")

    out: list[dict[str, object]] = []
    for file_name, group in frame.groupby("file_name", sort=True):
        error_group = group[group["is_error"]]
        if error_group.empty:
            continue
        pred_counter = Counter(error_group["pred_label"].astype(str))
        true_counter = Counter(group["y_true"].astype(str))
        first = group.iloc[0]
        out.append(
            {
                "file_name": file_name,
                "true_label_mode": true_counter.most_common(1)[0][0],
                "wrong_predictions": int(len(error_group)),
                "test_appearances": int(len(group)),
                "most_common_wrong_pred": pred_counter.most_common(1)[0][0],
                "mean_wrong_confidence": float(error_group["confidence"].mean()),
                "max_wrong_confidence": float(error_group["confidence"].max()),
                "age": first["age"],
                "gender": first["gender"],
                "number_of_missing_teeth": first["number_of_missing_teeth"],
                "implant": first["implant"],
                "residual_root": first["residual_root"],
            }
        )
    return sorted(out, key=lambda row: (-int(row["wrong_predictions"]), -float(row["mean_wrong_confidence"]), row["file_name"]))[:50]


def format_float(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def table_to_markdown(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        rendered = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, int):
                rendered.append(str(value))
            elif isinstance(value, (float, np.floating)):
                rendered.append(format_float(value))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def summary_text(
    comparison: list[dict[str, object]],
    deltas: list[dict[str, object]],
    binary_summary: list[dict[str, object]],
) -> tuple[str, dict[str, object]]:
    by_model = {row["model_id"]: row for row in comparison}
    image = by_model.get("image_efficientnet_b0_temperature_scaled", {})
    age = by_model.get("age_sex", {})
    downstream = by_model.get("downstream_plus_age_sex", {})
    image_macro = float(image.get("macro_f1_mean", math.nan))
    age_macro = float(age.get("macro_f1_mean", math.nan))
    downstream_macro = float(downstream.get("macro_f1_mean", math.nan))
    delta_image_age = image_macro - age_macro
    delta_image_downstream = image_macro - downstream_macro

    if math.isnan(delta_image_age):
        recommendation = "Revise analysis"
        rationale = "Could not compute the image-versus-age/sex comparison."
    elif delta_image_age >= 0.03:
        recommendation = "Proceed toward image-first benchmark"
        rationale = "The image-only baseline shows a meaningful margin over age/sex."
    elif delta_image_age > 0:
        recommendation = "Proceed, but strengthen before manuscript drafting"
        rationale = "The image-only baseline beats age/sex, but the margin is small."
    else:
        recommendation = "Do not draft as an image-performance paper yet"
        rationale = "The image-only baseline does not beat the age/sex baseline."

    key_findings = {
        "recommendation": recommendation,
        "rationale": rationale,
        "image_macro_f1": image_macro,
        "age_sex_macro_f1": age_macro,
        "downstream_plus_age_sex_macro_f1": downstream_macro,
        "image_minus_age_sex_macro_f1": delta_image_age,
        "image_minus_downstream_macro_f1": delta_image_downstream,
    }

    selected_delta_rows = [
        row
        for row in deltas
        if row["left_model"] == "image_efficientnet_b0_temperature_scaled"
        and row["right_model"] in {"age_sex", "downstream_plus_age_sex"}
        and row["metric"] in {"macro_f1", "balanced_accuracy", "log_loss", "ece_10"}
    ]
    selected_binary = [
        row
        for row in binary_summary
        if row["model_id"] in {"image_efficientnet_b0_temperature_scaled", "age_sex"}
    ]
    comparison_cols = [
        "model_id",
        "folds",
        "macro_f1_mean",
        "macro_f1_sd",
        "balanced_accuracy_mean",
        "ece_10_mean",
        "quadratic_weighted_kappa_mean",
        "ovr_auroc_mean",
    ]
    delta_cols = ["left_model", "right_model", "metric", "mean_delta", "sd_delta", "left_better_folds"]
    binary_cols = ["task", "model_id", "folds", "balanced_accuracy_mean", "f1_mean", "auroc_mean", "ece_10_mean"]

    md = f"""# Publication Strength Summary

Date: 2026-06-08

## Recommendation

**{recommendation}.** {rationale}

The current EfficientNet-B0 frozen image baseline is methodologically clean, but its three-class macro-F1 margin over age/sex is small: `{format_float(delta_image_age)}`. This makes the strongest near-term article angle a leakage-aware calibrated benchmark with explicit metadata guardrails, not a claim of clinical-grade image AI.

## Primary Model Comparison

{table_to_markdown(comparison, comparison_cols)}

## Paired Deltas

Positive deltas favor the left model, except for log loss and ECE where negative deltas favor the left model.

{table_to_markdown(selected_delta_rows, delta_cols)}

## Binary Task Signals

These are derived from the same multiclass probabilities with thresholds selected on the validation fold only.

{table_to_markdown(selected_binary, binary_cols)}

## Next Analysis To Prioritize

1. Train an `image_plus_age_sex` frozen-embedding sensitivity model to test whether images add incremental signal to demographics.
2. Run one stronger image-only encoder, preferably ResNet50 or ConvNeXt-Tiny, using the same split manifest and no test-set model shopping.
3. Add repeat-aware confidence intervals and subgroup tables to support a transparent benchmark manuscript.
4. Treat binary severe/non-severe performance as a secondary analysis only if it is clearly more stable than the three-class task.
"""
    return md, key_findings


def html_report(
    comparison: list[dict[str, object]],
    deltas: list[dict[str, object]],
    binary_summary: list[dict[str, object]],
    subgroup_summary: list[dict[str, object]],
    confident_errors: list[dict[str, object]],
    key_findings: dict[str, object],
) -> str:
    def df_html(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> str:
        if limit is not None:
            rows = rows[:limit]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return "<p>No rows.</p>"
        return frame[columns].to_html(index=False, escape=True, float_format=lambda value: f"{value:.4f}")

    comparison_cols = [
        "model_id",
        "folds",
        "macro_f1_mean",
        "macro_f1_sd",
        "balanced_accuracy_mean",
        "accuracy_mean",
        "log_loss_mean",
        "ece_10_mean",
        "quadratic_weighted_kappa_mean",
        "ovr_auroc_mean",
    ]
    delta_cols = ["left_model", "right_model", "metric", "mean_delta", "sd_delta", "min_delta", "max_delta", "left_better_folds"]
    binary_cols = ["task", "model_id", "folds", "balanced_accuracy_mean", "f1_mean", "sensitivity_mean", "specificity_mean", "auroc_mean", "average_precision_mean", "ece_10_mean"]
    subgroup_cols = ["model_id", "subgroup_type", "subgroup", "folds", "n_mean", "macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"]
    error_cols = ["file_name", "true_label_mode", "wrong_predictions", "test_appearances", "most_common_wrong_pred", "mean_wrong_confidence", "age", "gender", "number_of_missing_teeth"]

    recommendation = html.escape(str(key_findings["recommendation"]))
    rationale = html.escape(str(key_findings["rationale"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BRAR Publication Strength Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1, h2 {{ line-height: 1.2; }}
    .callout {{ border-left: 4px solid #2457a6; background: #f4f7fb; padding: 14px 16px; margin: 16px 0 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d7dce2; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef1f4; position: sticky; top: 0; }}
    code {{ background: #eef1f4; padding: 1px 4px; border-radius: 4px; }}
    .note {{ color: #5f6368; }}
  </style>
</head>
<body>
<main>
  <h1>BRAR Publication Strength Report</h1>
  <div class="callout">
    <strong>{recommendation}.</strong> {rationale}
  </div>

  <h2>Primary Model Comparison</h2>
  <p class="note">Test-fold means across the 15 repeated fold evaluations. Calibration uses validation-fitted temperature scaling for the image model.</p>
  {df_html(comparison, comparison_cols)}

  <h2>Paired Deltas</h2>
  <p class="note">Positive deltas favor the left model, except for log loss and ECE where negative deltas favor the left model.</p>
  {df_html(deltas, delta_cols)}

  <h2>Binary Task Signals</h2>
  <p class="note">Derived from multiclass probabilities. Thresholds are selected on validation folds only.</p>
  {df_html(binary_summary, binary_cols)}

  <h2>Subgroup Checks</h2>
  <p class="note">Exploratory age and gender subgroup summaries for image and age/sex models.</p>
  {df_html(subgroup_summary, subgroup_cols)}

  <h2>Confident Image-Model Errors</h2>
  <p class="note">Rows show test images most consistently misclassified across repeats. This is for manual review, not exclusion.</p>
  {df_html(confident_errors, error_cols, limit=25)}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--image-predictions", type=Path, default=IMAGE_PREDICTIONS)
    parser.add_argument("--negative-control-predictions", type=Path, default=NEGATIVE_CONTROL_PREDICTIONS)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    image_rows = read_csv(args.image_predictions)
    negative_rows = read_csv(args.negative_control_predictions)

    all_metrics = image_prediction_rows(image_rows) + negative_control_metric_rows(negative_rows)
    comparison = to_model_comparison(all_metrics)
    deltas = paired_deltas(all_metrics)

    prob_frame = combined_probability_frame(image_rows, negative_rows)
    binary_rows = binary_metric_rows(prob_frame)
    binary_summary = aggregate_mean_sd(
        binary_rows,
        ["task", "model_id"],
        [
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
    binary_summary = sorted(binary_summary, key=lambda row: (str(row["task"]), -float(row["balanced_accuracy_mean"])))

    subgroup_rows = subgroup_metric_rows(prob_frame, manifest)
    subgroup_summary = aggregate_mean_sd(
        subgroup_rows,
        ["model_id", "subgroup_type", "subgroup"],
        ["n", "accuracy", "balanced_accuracy", "macro_f1"],
    )
    subgroup_summary = sorted(subgroup_summary, key=lambda row: (str(row["subgroup_type"]), str(row["subgroup"]), str(row["model_id"])))

    confident_errors = confident_error_rows(prob_frame, manifest)
    summary_md, key_findings = summary_text(comparison, deltas, binary_summary)

    write_csv(
        MODEL_COMPARISON,
        comparison,
        [
            "model_id",
            "folds",
            "accuracy_mean",
            "accuracy_sd",
            "balanced_accuracy_mean",
            "balanced_accuracy_sd",
            "macro_f1_mean",
            "macro_f1_sd",
            "log_loss_mean",
            "log_loss_sd",
            "ece_10_mean",
            "ece_10_sd",
            "ordinal_mae_mean",
            "ordinal_mae_sd",
            "quadratic_weighted_kappa_mean",
            "quadratic_weighted_kappa_sd",
            "ovr_auroc_mean",
            "ovr_auroc_sd",
        ],
    )
    write_csv(
        PAIRED_DELTAS,
        deltas,
        [
            "left_model",
            "right_model",
            "metric",
            "paired_folds",
            "mean_delta",
            "sd_delta",
            "min_delta",
            "max_delta",
            "left_better_folds",
        ],
    )
    write_csv(
        BINARY_METRICS,
        binary_rows,
        [
            "model_id",
            "repeat",
            "seed",
            "fold",
            "task",
            "threshold_source",
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
        ],
    )
    write_csv(
        SUBGROUP_METRICS,
        subgroup_rows,
        [
            "model_id",
            "repeat",
            "seed",
            "fold",
            "subgroup_type",
            "subgroup",
            "n",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
    )
    write_csv(
        CONFIDENT_ERRORS,
        confident_errors,
        [
            "file_name",
            "true_label_mode",
            "wrong_predictions",
            "test_appearances",
            "most_common_wrong_pred",
            "mean_wrong_confidence",
            "max_wrong_confidence",
            "age",
            "gender",
            "number_of_missing_teeth",
            "implant",
            "residual_root",
        ],
    )
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "key_findings": key_findings,
                "model_comparison": comparison,
                "paired_deltas": deltas,
                "binary_summary": binary_summary,
                "subgroup_summary": subgroup_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    SUMMARY_MD.write_text(summary_md, encoding="utf-8")
    REPORT_HTML.write_text(
        html_report(comparison, deltas, binary_summary, subgroup_summary, confident_errors, key_findings),
        encoding="utf-8",
    )

    print(f"model comparison: {MODEL_COMPARISON}")
    print(f"paired deltas: {PAIRED_DELTAS}")
    print(f"binary metrics: {BINARY_METRICS}")
    print(f"subgroup metrics: {SUBGROUP_METRICS}")
    print(f"confident errors: {CONFIDENT_ERRORS}")
    print(f"summary: {SUMMARY_MD}")
    print(f"html report: {REPORT_HTML}")


if __name__ == "__main__":
    main()
