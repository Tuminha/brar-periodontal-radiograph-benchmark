#!/usr/bin/env python3
"""Evaluate frozen image baseline predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
IMAGE_BASELINE_DIR = ROOT / "data" / "processed" / "image_baseline"
REPORTS = ROOT / "reports"
DEFAULT_PREDICTIONS = IMAGE_BASELINE_DIR / "frozen_efficientnet_b0_384x192_predictions.csv"

CLASSES = ["1", "2", "3"]
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES)}

METRIC_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "eval_split",
    "model",
    "encoder",
    "image_width",
    "image_height",
    "probability_mode",
    "temperature",
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
    "ordinal_mae",
    "quadratic_weighted_kappa",
    "ovr_auroc",
]

AGGREGATE_FIELDS = [
    "eval_split",
    "model",
    "encoder",
    "probability_mode",
    "folds",
    "accuracy_mean",
    "accuracy_sd",
    "balanced_accuracy_mean",
    "balanced_accuracy_sd",
    "macro_f1_mean",
    "macro_f1_sd",
    "brier_score_mean",
    "brier_score_sd",
    "log_loss_mean",
    "log_loss_sd",
    "ece_10_mean",
    "ece_10_sd",
    "ovr_auroc_mean",
    "ovr_auroc_sd",
]

CONFUSION_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "eval_split",
    "model",
    "encoder",
    "probability_mode",
    "true_label",
    "pred_label",
    "count",
]

RELIABILITY_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "eval_split",
    "model",
    "encoder",
    "probability_mode",
    "bin",
    "n",
    "confidence_mean",
    "accuracy",
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


def output_paths(prefix: str) -> dict[str, Path]:
    return {
        "metrics": REPORTS / f"{prefix}_metrics.csv",
        "aggregate": REPORTS / f"{prefix}_metric_summary.csv",
        "confusions": REPORTS / f"{prefix}_confusion_matrices.csv",
        "reliability": REPORTS / f"{prefix}_reliability_bins.csv",
        "summary_json": REPORTS / f"{prefix}_summary.json",
        "summary_md": REPORTS / f"{prefix}_summary.md",
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray) -> np.ndarray:
    out = np.zeros((len(y), len(CLASSES)), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        probs = softmax(logits / temperature)
        return float(log_loss(y_true, probs, labels=[0, 1, 2]))

    result = minimize_scalar(objective, bounds=(math.log(0.2), math.log(5.0)), method="bounded")
    if not result.success:
        return 1.0
    return float(math.exp(result.x))


def ece_score(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    pred = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)
    correct = (pred == y_true).astype(float)
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


def metric_row(
    group_key: tuple[str, str, str, str],
    rows: list[dict[str, str]],
    probability_mode: str,
    temperature: float,
    probs: np.ndarray,
) -> dict[str, object]:
    repeat, seed, fold, eval_split = group_key
    y_true = np.asarray([CLASS_TO_INDEX[row["y_true"]] for row in rows], dtype=int)
    y_pred = np.argmax(probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )
    y_onehot = one_hot(y_true)
    try:
        auroc = float(roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        auroc = float("nan")

    base = rows[0]
    return {
        "repeat": repeat,
        "seed": seed,
        "fold": fold,
        "eval_split": eval_split,
        "model": base["model"],
        "encoder": base["encoder"],
        "image_width": base["image_width"],
        "image_height": base["image_height"],
        "probability_mode": probability_mode,
        "temperature": f"{temperature:.8f}",
        "n": len(rows),
        "accuracy": f"{accuracy_score(y_true, y_pred):.8f}",
        "balanced_accuracy": f"{balanced_accuracy_score(y_true, y_pred):.8f}",
        "macro_f1": f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.8f}",
        "precision_1": f"{precision[0]:.8f}",
        "precision_2": f"{precision[1]:.8f}",
        "precision_3": f"{precision[2]:.8f}",
        "recall_1": f"{recall[0]:.8f}",
        "recall_2": f"{recall[1]:.8f}",
        "recall_3": f"{recall[2]:.8f}",
        "f1_1": f"{f1[0]:.8f}",
        "f1_2": f"{f1[1]:.8f}",
        "f1_3": f"{f1[2]:.8f}",
        "brier_score": f"{np.mean(np.sum((probs - y_onehot) ** 2, axis=1)):.8f}",
        "log_loss": f"{log_loss(y_true, probs, labels=[0, 1, 2]):.8f}",
        "ece_10": f"{ece_score(y_true, probs):.8f}",
        "ordinal_mae": f"{np.mean(np.abs(y_true - y_pred)):.8f}",
        "quadratic_weighted_kappa": f"{quadratic_weighted_kappa(y_true, y_pred):.8f}",
        "ovr_auroc": f"{auroc:.8f}",
    }


def confusion_rows(
    group_key: tuple[str, str, str, str],
    rows: list[dict[str, str]],
    probability_mode: str,
    probs: np.ndarray,
) -> list[dict[str, object]]:
    repeat, seed, fold, eval_split = group_key
    base = rows[0]
    y_true = np.asarray([CLASS_TO_INDEX[row["y_true"]] for row in rows], dtype=int)
    y_pred = np.argmax(probs, axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    out = []
    for true_idx, true_label in enumerate(CLASSES):
        for pred_idx, pred_label in enumerate(CLASSES):
            out.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "fold": fold,
                    "eval_split": eval_split,
                    "model": base["model"],
                    "encoder": base["encoder"],
                    "probability_mode": probability_mode,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(matrix[true_idx, pred_idx]),
                }
            )
    return out


def reliability_rows(
    group_key: tuple[str, str, str, str],
    rows: list[dict[str, str]],
    probability_mode: str,
    probs: np.ndarray,
) -> list[dict[str, object]]:
    repeat, seed, fold, eval_split = group_key
    base = rows[0]
    y_true = np.asarray([CLASS_TO_INDEX[row["y_true"]] for row in rows], dtype=int)
    y_pred = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    out = []
    for bin_idx in range(10):
        lower = bin_idx / 10
        upper = (bin_idx + 1) / 10
        if bin_idx == 9:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        out.append(
            {
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                "eval_split": eval_split,
                "model": base["model"],
                "encoder": base["encoder"],
                "probability_mode": probability_mode,
                "bin": f"{lower:.1f}-{upper:.1f}",
                "n": int(mask.sum()),
                "confidence_mean": f"{float(confidence[mask].mean()) if np.any(mask) else 0.0:.8f}",
                "accuracy": f"{float(correct[mask].mean()) if np.any(mask) else 0.0:.8f}",
            }
        )
    return out


def aggregate_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        grouped[
            (
                str(row["eval_split"]),
                str(row["model"]),
                str(row["encoder"]),
                str(row["probability_mode"]),
            )
        ].append(row)

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "brier_score", "log_loss", "ece_10", "ovr_auroc"]
    out = []
    for key, rows in sorted(grouped.items()):
        eval_split, model, encoder, probability_mode = key
        item: dict[str, object] = {
            "eval_split": eval_split,
            "model": model,
            "encoder": encoder,
            "probability_mode": probability_mode,
            "folds": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if not math.isnan(float(row[metric]))]
            item[f"{metric}_mean"] = f"{float(np.mean(values)) if values else float('nan'):.8f}"
            item[f"{metric}_sd"] = f"{float(np.std(values, ddof=1)) if len(values) > 1 else 0.0:.8f}"
        out.append(item)
    return out


def markdown_summary(aggregate: list[dict[str, object]]) -> str:
    rows = [row for row in aggregate if row["eval_split"] == "test"]
    rows = sorted(rows, key=lambda row: float(row["macro_f1_mean"]), reverse=True)
    table = [
        "| Mode | Macro-F1 | Balanced accuracy | Accuracy | Log loss | ECE | AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            "| {mode} | {macro:.4f} | {balanced:.4f} | {accuracy:.4f} | {loss:.4f} | {ece:.4f} | {auroc:.4f} |".format(
                mode=row["probability_mode"],
                macro=float(row["macro_f1_mean"]),
                balanced=float(row["balanced_accuracy_mean"]),
                accuracy=float(row["accuracy_mean"]),
                loss=float(row["log_loss_mean"]),
                ece=float(row["ece_10_mean"]),
                auroc=float(row["ovr_auroc_mean"]),
            )
        )
    return f"""# Image Baseline Summary

Date: 2026-06-07

## Test Aggregate Metrics

{chr(10).join(table)}

## Interpretation

Compare image-only macro-F1 against the age/sex negative-control baseline (`0.4850`) and downstream-plus-age/sex upper-bound (`0.5063`). The image-only model is compelling only if it meaningfully exceeds age/sex without relying on metadata.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-prefix", default="image_baseline")
    args = parser.parse_args()
    paths = output_paths(args.output_prefix)

    prediction_rows = read_csv(args.predictions)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(row["repeat"], row["seed"], row["fold"], row["eval_split"])].append(row)

    temperatures: dict[tuple[str, str, str], float] = {}
    for repeat, seed, fold, eval_split in grouped:
        if eval_split != "val":
            continue
        rows = grouped[(repeat, seed, fold, eval_split)]
        logits = np.asarray([[float(row[f"logit_{label}"]) for label in CLASSES] for row in rows])
        y_true = np.asarray([CLASS_TO_INDEX[row["y_true"]] for row in rows], dtype=int)
        temperatures[(repeat, seed, fold)] = fit_temperature(logits, y_true)

    metric_rows: list[dict[str, object]] = []
    confusion_output: list[dict[str, object]] = []
    reliability_output: list[dict[str, object]] = []
    for group_key, rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), int(item[0][2]), item[0][3])):
        repeat, seed, fold, _ = group_key
        logits = np.asarray([[float(row[f"logit_{label}"]) for label in CLASSES] for row in rows])
        raw_probs = normalize_probabilities(
            np.asarray([[float(row[f"prob_{label}"]) for label in CLASSES] for row in rows])
        )
        temperature = temperatures[(repeat, seed, fold)]
        calibrated_probs = softmax(logits / temperature)
        for mode, probs, temp in [
            ("raw", raw_probs, 1.0),
            ("temperature_scaled", calibrated_probs, temperature),
        ]:
            metric_rows.append(metric_row(group_key, rows, mode, temp, probs))
            confusion_output.extend(confusion_rows(group_key, rows, mode, probs))
            reliability_output.extend(reliability_rows(group_key, rows, mode, probs))

    aggregate = aggregate_rows(metric_rows)
    write_csv(paths["metrics"], metric_rows, METRIC_FIELDS)
    write_csv(paths["aggregate"], aggregate, AGGREGATE_FIELDS)
    write_csv(paths["confusions"], confusion_output, CONFUSION_FIELDS)
    write_csv(paths["reliability"], reliability_output, RELIABILITY_FIELDS)
    paths["summary_json"].write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["summary_md"].write_text(markdown_summary(aggregate), encoding="utf-8")
    print(f"metrics: {paths['metrics']}")
    print(f"aggregate: {paths['aggregate']}")
    print(f"summary: {paths['summary_md']}")


if __name__ == "__main__":
    main()
