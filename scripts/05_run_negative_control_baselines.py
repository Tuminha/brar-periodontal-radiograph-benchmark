#!/usr/bin/env python3
"""Run non-image negative-control baselines for BRAR severity prediction.

This script intentionally does not train an image model. It estimates how much
performance can be obtained from class priors, administrative/image-geometry
features, age/sex metadata, and downstream dental-status variables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "brar_manifest.csv"
SPLITS = ROOT / "data" / "processed" / "splits" / "brar_repeated_5fold_splits.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "negative_controls"
REPORTS = ROOT / "reports"

PREDICTIONS = OUTPUT_DIR / "negative_control_predictions.csv"
METRICS = REPORTS / "negative_control_metrics.csv"
MODEL_SELECTION = REPORTS / "negative_control_model_selection.csv"
CONFUSIONS = REPORTS / "negative_control_confusion_matrices.csv"
SUMMARY_JSON = REPORTS / "negative_control_summary.json"
SUMMARY_MD = REPORTS / "negative_control_summary.md"

CLASSES = ["1", "2", "3"]
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES)}
L2_GRID = [0.001, 0.01, 0.1, 1.0]

FORBIDDEN_COLUMNS = {
    "bone_resorption",
    "bone_resorption_age",
    "bl_mm",
    "rl_mm",
    "bl_rl_ratio",
    "max_bl_rl_ratio",
    "brar",
}

FEATURE_SETS = {
    "age_sex": ["age", "gender"],
    "image_geometry_file": ["pixel_width", "pixel_height", "aspect_ratio", "size_bytes"],
    "admin_index": ["row_index", "filename_index"],
    "age_sex_geometry": ["age", "gender", "pixel_width", "pixel_height", "aspect_ratio", "size_bytes"],
    "downstream_status": [
        "number_of_missing_teeth",
        "implant",
        "residual_root",
        "functional_tooth_logarithm",
    ],
    "downstream_plus_age_sex": [
        "age",
        "gender",
        "number_of_missing_teeth",
        "implant",
        "residual_root",
        "functional_tooth_logarithm",
    ],
}

PREDICTION_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "eval_split",
    "model",
    "feature_set",
    "selected_l2",
    "file_name",
    "y_true",
    "y_pred",
    "prob_1",
    "prob_2",
    "prob_3",
]

METRIC_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "eval_split",
    "model",
    "feature_set",
    "selected_l2",
    "n",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision_1",
    "precision_2",
    "precision_3",
    "recall_1",
    "recall_2",
    "recall_3",
    "f1_1",
    "f1_2",
    "f1_3",
    "brier_score",
    "log_loss",
    "ece_10",
]

SELECTION_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "feature_set",
    "selected_l2",
    "validation_macro_f1",
    "validation_balanced_accuracy",
    "validation_log_loss",
]

CONFUSION_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "eval_split",
    "model",
    "feature_set",
    "true_label",
    "pred_label",
    "count",
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


def filename_index(file_name: str) -> float:
    match = re.search(r"patient_image_(\d+)_", file_name)
    return float(match.group(1)) if match else math.nan


def enrich_manifest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched = []
    for row in rows:
        new_row = dict(row)
        new_row["filename_index"] = str(filename_index(row["file_name"]))
        enriched.append(new_row)
    return enriched


def assert_feature_policy() -> None:
    for feature_set, columns in FEATURE_SETS.items():
        forbidden = FORBIDDEN_COLUMNS.intersection(columns)
        if forbidden:
            raise RuntimeError(f"{feature_set} includes forbidden predictors: {sorted(forbidden)}")


def y_array(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([CLASS_TO_INDEX[row["severity_level"]] for row in rows], dtype=np.int64)


def x_array(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    data = []
    for row in rows:
        data.append([float(row[column]) for column in columns])
    return np.asarray(data, dtype=np.float64)


def standardize_train_apply(
    x_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (x_train - mean) / std, (x_eval - mean) / std, mean, std


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = Counter(int(value) for value in y)
    n = len(y)
    weights = {label: n / (len(CLASSES) * counts[label]) for label in range(len(CLASSES))}
    return np.array([weights[int(value)] for value in y], dtype=np.float64)


def weighted_loss(
    x: np.ndarray,
    y: np.ndarray,
    weights_matrix: np.ndarray,
    sample_weights: np.ndarray,
    l2: float,
) -> float:
    probs = softmax(x @ weights_matrix)
    y_onehot = one_hot(y, len(CLASSES))
    nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))
    data_loss = float(np.sum(sample_weights * nll) / np.sum(sample_weights))
    regularization = 0.5 * l2 * float(np.sum(weights_matrix[1:, :] ** 2))
    return data_loss + regularization


def fit_multinomial_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    l2: float,
    learning_rate: float = 0.08,
    max_iter: int = 900,
    tolerance: float = 1e-7,
) -> np.ndarray:
    x = np.column_stack([np.ones(len(x_train)), x_train])
    y_onehot = one_hot(y_train, len(CLASSES))
    sample_weights = class_weights(y_train)
    weight_sum = sample_weights.sum()
    weights_matrix = np.zeros((x.shape[1], len(CLASSES)), dtype=np.float64)
    last_loss = math.inf
    stale_checks = 0

    for iteration in range(max_iter):
        probs = softmax(x @ weights_matrix)
        residual = (probs - y_onehot) * sample_weights[:, None]
        gradient = (x.T @ residual) / weight_sum
        gradient[1:, :] += l2 * weights_matrix[1:, :]
        weights_matrix -= learning_rate * gradient

        if iteration % 50 == 0 or iteration == max_iter - 1:
            loss = weighted_loss(x, y_train, weights_matrix, sample_weights, l2)
            if last_loss - loss < tolerance:
                stale_checks += 1
            else:
                stale_checks = 0
            last_loss = loss
            if stale_checks >= 3:
                break

    return weights_matrix


def predict_proba(weights_matrix: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(x_eval)), x_eval])
    return softmax(x @ weights_matrix)


def majority_probabilities(y_train: np.ndarray, n_eval: int) -> np.ndarray:
    counts = Counter(int(value) for value in y_train)
    probs = np.array([counts.get(idx, 0) / len(y_train) for idx in range(len(CLASSES))], dtype=np.float64)
    return np.tile(probs, (n_eval, 1))


def stratified_random_probabilities_and_predictions(
    y_train: np.ndarray,
    n_eval: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    probs = majority_probabilities(y_train, n_eval)
    rng = random.Random(seed)
    cutoffs = [float(probs[0, 0]), float(probs[0, 0] + probs[0, 1]), 1.0]
    predictions = []
    for _ in range(n_eval):
        value = rng.random()
        if value <= cutoffs[0]:
            predictions.append(0)
        elif value <= cutoffs[1]:
            predictions.append(1)
        else:
            predictions.append(2)
    return probs, np.asarray(predictions, dtype=np.int64)


def labels_from_predictions(y_pred: np.ndarray) -> list[str]:
    return [CLASSES[int(value)] for value in y_pred]


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    total = len(y_true)
    accuracy = float(np.mean(y_true == y_pred)) if total else 0.0
    precision = []
    recall = []
    f1 = []
    for class_idx in range(len(CLASSES)):
        tp = int(np.sum((y_true == class_idx) & (y_pred == class_idx)))
        fp = int(np.sum((y_true != class_idx) & (y_pred == class_idx)))
        fn = int(np.sum((y_true == class_idx) & (y_pred != class_idx)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * p * r / (p + r) if p + r else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(score)

    y_onehot = one_hot(y_true, len(CLASSES))
    clipped = np.clip(probs, 1e-12, 1.0)
    confidence = probs.max(axis=1)
    correctness = (y_true == y_pred).astype(np.float64)
    ece = 0.0
    for bin_idx in range(10):
        lower = bin_idx / 10
        upper = (bin_idx + 1) / 10
        if bin_idx == 9:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if np.any(mask):
            ece += float(np.mean(mask) * abs(np.mean(correctness[mask]) - np.mean(confidence[mask])))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "precision_1": precision[0],
        "precision_2": precision[1],
        "precision_3": precision[2],
        "recall_1": recall[0],
        "recall_2": recall[1],
        "recall_3": recall[2],
        "f1_1": f1[0],
        "f1_2": f1[1],
        "f1_3": f1[2],
        "brier_score": float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(total), y_true]))),
        "ece_10": ece,
    }


def confusion_rows(
    repeat: str,
    seed: str,
    fold: str,
    eval_split: str,
    model: str,
    feature_set: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for true_idx, true_label in enumerate(CLASSES):
        for pred_idx, pred_label in enumerate(CLASSES):
            rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "fold": fold,
                    "eval_split": eval_split,
                    "model": model,
                    "feature_set": feature_set,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(np.sum((y_true == true_idx) & (y_pred == pred_idx))),
                }
            )
    return rows


def append_predictions(
    out: list[dict[str, object]],
    repeat: str,
    seed: str,
    fold: str,
    eval_split: str,
    model: str,
    feature_set: str,
    selected_l2: str,
    rows: list[dict[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> None:
    for idx, row in enumerate(rows):
        out.append(
            {
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                "eval_split": eval_split,
                "model": model,
                "feature_set": feature_set,
                "selected_l2": selected_l2,
                "file_name": row["file_name"],
                "y_true": CLASSES[int(y_true[idx])],
                "y_pred": CLASSES[int(y_pred[idx])],
                "prob_1": f"{probs[idx, 0]:.8f}",
                "prob_2": f"{probs[idx, 1]:.8f}",
                "prob_3": f"{probs[idx, 2]:.8f}",
            }
        )


def metric_row(
    repeat: str,
    seed: str,
    fold: str,
    eval_split: str,
    model: str,
    feature_set: str,
    selected_l2: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> dict[str, object]:
    values = metric_values(y_true, y_pred, probs)
    row: dict[str, object] = {
        "repeat": repeat,
        "seed": seed,
        "fold": fold,
        "eval_split": eval_split,
        "model": model,
        "feature_set": feature_set,
        "selected_l2": selected_l2,
        "n": len(y_true),
    }
    row.update({key: f"{value:.8f}" for key, value in values.items()})
    return row


def split_rows_for_fold(
    manifest_by_name: dict[str, dict[str, str]],
    split_rows: list[dict[str, str]],
    repeat: str,
    fold: str,
    split_name: str,
) -> list[dict[str, str]]:
    names = [
        row["file_name"]
        for row in split_rows
        if row["repeat"] == repeat and row["fold"] == fold and row["split"] == split_name
    ]
    return [manifest_by_name[name] for name in names]


def rows_by_repeat_fold(split_rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    seen = {(row["repeat"], row["seed"], row["fold"]) for row in split_rows}
    return sorted(seen, key=lambda key: (int(key[0]), int(key[2])))


def evaluate_fixed_prob_model(
    prediction_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    confusion_output_rows: list[dict[str, object]],
    repeat: str,
    seed: str,
    fold: str,
    eval_split: str,
    model: str,
    feature_set: str,
    eval_rows: list[dict[str, str]],
    y_train: np.ndarray,
    random_seed: int | None = None,
) -> None:
    y_eval = y_array(eval_rows)
    if model == "stratified_random":
        probs, y_pred = stratified_random_probabilities_and_predictions(y_train, len(eval_rows), random_seed or 0)
    else:
        probs = majority_probabilities(y_train, len(eval_rows))
        y_pred = np.argmax(probs, axis=1)

    append_predictions(
        prediction_rows,
        repeat,
        seed,
        fold,
        eval_split,
        model,
        feature_set,
        "",
        eval_rows,
        y_eval,
        y_pred,
        probs,
    )
    metric_rows.append(
        metric_row(repeat, seed, fold, eval_split, model, feature_set, "", y_eval, y_pred, probs)
    )
    confusion_output_rows.extend(
        confusion_rows(repeat, seed, fold, eval_split, model, feature_set, y_eval, y_pred)
    )


def train_select_evaluate_feature_set(
    prediction_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    selection_rows: list[dict[str, object]],
    confusion_output_rows: list[dict[str, object]],
    repeat: str,
    seed: str,
    fold: str,
    feature_set: str,
    columns: list[str],
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> None:
    y_train = y_array(train_rows)
    y_val = y_array(val_rows)
    x_train_raw = x_array(train_rows, columns)
    x_val_raw = x_array(val_rows, columns)
    x_train, x_val, _, _ = standardize_train_apply(x_train_raw, x_val_raw)

    best: tuple[float, float, float, float, np.ndarray] | None = None
    for l2 in L2_GRID:
        weights_matrix = fit_multinomial_logistic(x_train, y_train, l2=l2)
        val_probs = predict_proba(weights_matrix, x_val)
        val_pred = np.argmax(val_probs, axis=1)
        val_metrics = metric_values(y_val, val_pred, val_probs)
        ranking = (
            val_metrics["macro_f1"],
            val_metrics["balanced_accuracy"],
            -val_metrics["log_loss"],
            -l2,
        )
        if best is None or ranking > best[:4]:
            best = (*ranking, weights_matrix)

    if best is None:
        raise RuntimeError(f"no model selected for {feature_set}")

    selected_l2 = -best[3]
    selected_weights = best[4]
    val_probs = predict_proba(selected_weights, x_val)
    val_pred = np.argmax(val_probs, axis=1)
    val_metrics = metric_values(y_val, val_pred, val_probs)
    selection_rows.append(
        {
            "repeat": repeat,
            "seed": seed,
            "fold": fold,
            "feature_set": feature_set,
            "selected_l2": f"{selected_l2:.8f}",
            "validation_macro_f1": f"{val_metrics['macro_f1']:.8f}",
            "validation_balanced_accuracy": f"{val_metrics['balanced_accuracy']:.8f}",
            "validation_log_loss": f"{val_metrics['log_loss']:.8f}",
        }
    )

    for eval_split, eval_rows in [("val", val_rows), ("test", test_rows)]:
        y_eval = y_array(eval_rows)
        x_eval_raw = x_array(eval_rows, columns)
        x_train_scaled, x_eval, _, _ = standardize_train_apply(x_train_raw, x_eval_raw)
        # Refit with selected lambda on the original train data so each evaluation uses
        # the same train-only preprocessing and selected regularization.
        selected_weights = fit_multinomial_logistic(x_train_scaled, y_train, l2=selected_l2)
        probs = predict_proba(selected_weights, x_eval)
        y_pred = np.argmax(probs, axis=1)
        append_predictions(
            prediction_rows,
            repeat,
            seed,
            fold,
            eval_split,
            "multinomial_logistic",
            feature_set,
            f"{selected_l2:.8f}",
            eval_rows,
            y_eval,
            y_pred,
            probs,
        )
        metric_rows.append(
            metric_row(
                repeat,
                seed,
                fold,
                eval_split,
                "multinomial_logistic",
                feature_set,
                f"{selected_l2:.8f}",
                y_eval,
                y_pred,
                probs,
            )
        )
        confusion_output_rows.extend(
            confusion_rows(
                repeat,
                seed,
                fold,
                eval_split,
                "multinomial_logistic",
                feature_set,
                y_eval,
                y_pred,
            )
        )


def aggregate_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["eval_split"]), str(row["model"]), str(row["feature_set"]))].append(row)

    output = []
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "brier_score", "log_loss", "ece_10"]
    for key, group in sorted(grouped.items()):
        eval_split, model, feature_set = key
        row: dict[str, object] = {
            "eval_split": eval_split,
            "model": model,
            "feature_set": feature_set,
            "folds": len(group),
        }
        for metric in metric_names:
            values = [float(item[metric]) for item in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        output.append(row)
    return output


def markdown_summary(aggregate_rows: list[dict[str, object]]) -> str:
    test_rows = [row for row in aggregate_rows if row["eval_split"] == "test"]
    test_rows = sorted(test_rows, key=lambda row: float(row["macro_f1_mean"]), reverse=True)

    table_lines = [
        "| Model | Feature set | Macro-F1 mean | Balanced accuracy mean | Accuracy mean | Log loss mean | ECE mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in test_rows:
        table_lines.append(
            "| {model} | {features} | {macro:.4f} | {balanced:.4f} | {accuracy:.4f} | {loss:.4f} | {ece:.4f} |".format(
                model=row["model"],
                features=row["feature_set"],
                macro=float(row["macro_f1_mean"]),
                balanced=float(row["balanced_accuracy_mean"]),
                accuracy=float(row["accuracy_mean"]),
                loss=float(row["log_loss_mean"]),
                ece=float(row["ece_10_mean"]),
            )
        )

    return f"""# Negative-Control Baseline Summary

Date: 2026-06-07

## Purpose

These are non-image baselines. They estimate how much BRAR severity can be predicted before any image pixels are used. They are guardrails against overstating an image model if metadata, image geometry, file order, or downstream dental-status variables already carry substantial signal.

## Test-Set Aggregate Metrics

{chr(10).join(table_lines)}

## Interpretation Rules

- `majority_class` is the minimum baseline. It should have high raw accuracy because Level 2 is common, but low macro-F1 and balanced accuracy.
- `stratified_random` estimates random-label performance under the train class prevalence.
- `age_sex` is a metadata sensitivity baseline, not the primary model.
- `image_geometry_file` and `admin_index` are negative controls. Strong performance here would suggest acquisition/file-order confounding.
- `downstream_status` and `downstream_plus_age_sex` are upper-bound sensitivity models and are not deployment-ready.

## Generated Files

- `data/processed/negative_controls/negative_control_predictions.csv`
- `reports/negative_control_metrics.csv`
- `reports/negative_control_model_selection.csv`
- `reports/negative_control_confusion_matrices.csv`
- `reports/negative_control_summary.json`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    return parser.parse_args()


def main() -> None:
    assert_feature_policy()
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    manifest_rows = enrich_manifest(read_csv(args.manifest))
    manifest_by_name = {row["file_name"]: row for row in manifest_rows}
    split_rows = read_csv(args.splits)

    prediction_rows: list[dict[str, object]] = []
    metric_rows_output: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    confusion_output_rows: list[dict[str, object]] = []

    for repeat, seed, fold in rows_by_repeat_fold(split_rows):
        train_rows = split_rows_for_fold(manifest_by_name, split_rows, repeat, fold, "train")
        val_rows = split_rows_for_fold(manifest_by_name, split_rows, repeat, fold, "val")
        test_rows = split_rows_for_fold(manifest_by_name, split_rows, repeat, fold, "test")
        y_train = y_array(train_rows)

        for eval_split, eval_rows in [("val", val_rows), ("test", test_rows)]:
            evaluate_fixed_prob_model(
                prediction_rows,
                metric_rows_output,
                confusion_output_rows,
                repeat,
                seed,
                fold,
                eval_split,
                "majority_class",
                "class_prior",
                eval_rows,
                y_train,
            )
            evaluate_fixed_prob_model(
                prediction_rows,
                metric_rows_output,
                confusion_output_rows,
                repeat,
                seed,
                fold,
                eval_split,
                "stratified_random",
                "class_prior",
                eval_rows,
                y_train,
                random_seed=int(seed) + int(fold) * 1000 + (0 if eval_split == "val" else 500),
            )

        for feature_set, columns in FEATURE_SETS.items():
            train_select_evaluate_feature_set(
                prediction_rows,
                metric_rows_output,
                selection_rows,
                confusion_output_rows,
                repeat,
                seed,
                fold,
                feature_set,
                columns,
                train_rows,
                val_rows,
                test_rows,
            )

    aggregate_rows = aggregate_metrics(metric_rows_output)

    write_csv(PREDICTIONS, prediction_rows, PREDICTION_FIELDS)
    write_csv(METRICS, metric_rows_output, METRIC_FIELDS)
    write_csv(MODEL_SELECTION, selection_rows, SELECTION_FIELDS)
    write_csv(CONFUSIONS, confusion_output_rows, CONFUSION_FIELDS)
    SUMMARY_JSON.write_text(json.dumps(aggregate_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SUMMARY_MD.write_text(markdown_summary(aggregate_rows), encoding="utf-8")

    best_test = max(
        (row for row in aggregate_rows if row["eval_split"] == "test"),
        key=lambda row: float(row["macro_f1_mean"]),
    )
    print(f"predictions: {PREDICTIONS}")
    print(f"metrics: {METRICS}")
    print(f"model_selection: {MODEL_SELECTION}")
    print(f"summary: {SUMMARY_MD}")
    print(
        "best_test_macro_f1: "
        f"{best_test['model']} / {best_test['feature_set']} = {float(best_test['macro_f1_mean']):.4f}"
    )


if __name__ == "__main__":
    main()
