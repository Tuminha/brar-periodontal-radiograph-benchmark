#!/usr/bin/env python3
"""Train BRAR frozen-embedding sensitivity models.

The primary image-only baseline remains the deployment-relevant model. This
script tests whether cached image embeddings add signal when combined with
allowed metadata, and it labels downstream dental-status models as upper-bound
sensitivity analyses.
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "brar_manifest.csv"
SPLITS = ROOT / "data" / "processed" / "splits" / "brar_repeated_5fold_splits.csv"
EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "image_baseline"
    / "embeddings_efficientnet_b0_384x192.npz"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "embedding_sensitivity"
REPORTS = ROOT / "reports"

PREDICTIONS = OUTPUT_DIR / "efficientnet_b0_384x192_sensitivity_predictions.csv"
SELECTION = OUTPUT_DIR / "efficientnet_b0_384x192_sensitivity_model_selection.csv"
RUN_INFO = OUTPUT_DIR / "efficientnet_b0_384x192_sensitivity_run_info.json"
METRICS = REPORTS / "embedding_sensitivity_metrics.csv"
SUMMARY_CSV = REPORTS / "embedding_sensitivity_metric_summary.csv"
SUMMARY_MD = REPORTS / "embedding_sensitivity_summary.md"
REPORT_HTML = REPORTS / "embedding_sensitivity_report.html"

CLASSES = ["1", "2", "3"]
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES)}
DEFAULT_C_GRID = [0.01, 0.1, 1.0, 10.0]

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
    "image_embedding_plus_age_sex": {
        "metadata_columns": ["age", "gender"],
        "sensitivity_label": "allowed metadata sensitivity",
    },
    "image_embedding_plus_downstream_age_sex_upper_bound": {
        "metadata_columns": [
            "age",
            "gender",
            "number_of_missing_teeth",
            "implant",
            "residual_root",
            "functional_tooth_logarithm",
        ],
        "sensitivity_label": "non-deployment downstream upper-bound sensitivity",
    },
}

PREDICTION_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "eval_split",
    "model",
    "feature_set",
    "sensitivity_label",
    "encoder",
    "image_width",
    "image_height",
    "selected_c",
    "file_name",
    "y_true",
    "y_pred",
    "prob_1",
    "prob_2",
    "prob_3",
    "logit_1",
    "logit_2",
    "logit_3",
]

SELECTION_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "feature_set",
    "sensitivity_label",
    "selected_c",
    "validation_macro_f1",
    "validation_balanced_accuracy",
    "validation_log_loss",
]

METRIC_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "eval_split",
    "model",
    "feature_set",
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

SUMMARY_FIELDS = [
    "eval_split",
    "feature_set",
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
    "ordinal_mae_mean",
    "ordinal_mae_sd",
    "quadratic_weighted_kappa_mean",
    "quadratic_weighted_kappa_sd",
    "ovr_auroc_mean",
    "ovr_auroc_sd",
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


def assert_feature_policy() -> None:
    for feature_set, config in FEATURE_SETS.items():
        forbidden = FORBIDDEN_COLUMNS.intersection(config["metadata_columns"])
        if forbidden:
            raise RuntimeError(f"{feature_set} includes forbidden predictors: {sorted(forbidden)}")


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
    group_key: tuple[str, str, str, str, str],
    rows: list[dict[str, str]],
    probability_mode: str,
    temperature: float,
    probs: np.ndarray,
) -> dict[str, object]:
    feature_set, repeat, seed, fold, eval_split = group_key
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

    return {
        "repeat": int(repeat),
        "seed": int(seed),
        "fold": int(fold),
        "eval_split": eval_split,
        "model": "frozen_embedding_logistic_sensitivity",
        "feature_set": feature_set,
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


def aggregate_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["eval_split"]), str(row["feature_set"]), str(row["probability_mode"]))].append(row)

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "brier_score",
        "log_loss",
        "ece_10",
        "ordinal_mae",
        "quadratic_weighted_kappa",
        "ovr_auroc",
    ]
    out = []
    for key, rows in sorted(grouped.items()):
        eval_split, feature_set, probability_mode = key
        item: dict[str, object] = {
            "eval_split": eval_split,
            "feature_set": feature_set,
            "probability_mode": probability_mode,
            "folds": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if not math.isnan(float(row[metric]))]
            item[f"{metric}_mean"] = f"{float(np.mean(values)) if values else float('nan'):.8f}"
            item[f"{metric}_sd"] = f"{float(np.std(values, ddof=1)) if len(values) > 1 else 0.0:.8f}"
        out.append(item)
    return out


def load_embeddings(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    data = np.load(path, allow_pickle=False)
    features = data["features"].astype(np.float32)
    file_names = [str(item) for item in data["file_names"]]
    labels = [str(item) for item in data["labels"]]
    return features, file_names, labels


def split_combos(split_rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    return sorted(
        {(row["repeat"], row["seed"], row["fold"]) for row in split_rows},
        key=lambda item: (int(item[0]), int(item[2])),
    )


def rows_for(split_rows: list[dict[str, str]], repeat: str, fold: str, split: str) -> list[str]:
    return [
        row["file_name"]
        for row in split_rows
        if row["repeat"] == repeat and row["fold"] == fold and row["split"] == split
    ]


def metadata_matrix(manifest: pd.DataFrame, names: list[str], columns: list[str]) -> np.ndarray:
    indexed = manifest.set_index("file_name")
    return indexed.loc[names, columns].astype(float).to_numpy()


def make_feature_matrix(
    embedding_features: np.ndarray,
    name_to_index: dict[str, int],
    manifest: pd.DataFrame,
    names: list[str],
    metadata_columns: list[str],
) -> np.ndarray:
    image_part = embedding_features[[name_to_index[name] for name in names]]
    metadata_part = metadata_matrix(manifest, names, metadata_columns)
    return np.column_stack([image_part, metadata_part]).astype(np.float32)


def fit_select_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float],
    seed: int,
) -> tuple[Pipeline, float, dict[str, float]]:
    best_pipeline: Pipeline | None = None
    best_c = 0.0
    best_ranking: tuple[float, float, float, float] | None = None
    best_metrics: dict[str, float] = {}

    for c_value in c_grid:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=2500,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        pipeline.fit(x_train, y_train)
        probs = pipeline.predict_proba(x_val)
        pred = np.argmax(probs, axis=1)
        metrics = {
            "macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
            "log_loss": float(log_loss(y_val, probs, labels=[0, 1, 2])),
        }
        ranking = (metrics["macro_f1"], metrics["balanced_accuracy"], -metrics["log_loss"], -c_value)
        if best_ranking is None or ranking > best_ranking:
            best_pipeline = pipeline
            best_c = c_value
            best_ranking = ranking
            best_metrics = metrics

    if best_pipeline is None:
        raise RuntimeError("classifier selection failed")
    return best_pipeline, best_c, best_metrics


def append_predictions(
    out: list[dict[str, object]],
    repeat: str,
    seed: str,
    fold: str,
    eval_split: str,
    feature_set: str,
    sensitivity_label: str,
    selected_c: float,
    file_names: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    logits: np.ndarray,
) -> None:
    pred = np.argmax(probs, axis=1)
    for idx, file_name in enumerate(file_names):
        out.append(
            {
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                "eval_split": eval_split,
                "model": "frozen_embedding_logistic_sensitivity",
                "feature_set": feature_set,
                "sensitivity_label": sensitivity_label,
                "encoder": "efficientnet_b0",
                "image_width": 384,
                "image_height": 192,
                "selected_c": f"{selected_c:.8f}",
                "file_name": file_name,
                "y_true": CLASSES[int(y_true[idx])],
                "y_pred": CLASSES[int(pred[idx])],
                "prob_1": f"{probs[idx, 0]:.8f}",
                "prob_2": f"{probs[idx, 1]:.8f}",
                "prob_3": f"{probs[idx, 2]:.8f}",
                "logit_1": f"{logits[idx, 0]:.8f}",
                "logit_2": f"{logits[idx, 1]:.8f}",
                "logit_3": f"{logits[idx, 2]:.8f}",
            }
        )


def evaluate_predictions(prediction_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[
            (
                str(row["feature_set"]),
                str(row["repeat"]),
                str(row["seed"]),
                str(row["fold"]),
                str(row["eval_split"]),
            )
        ].append({key: str(value) for key, value in row.items()})

    temperatures: dict[tuple[str, str, str, str], float] = {}
    for feature_set, repeat, seed, fold, eval_split in grouped:
        if eval_split != "val":
            continue
        rows = grouped[(feature_set, repeat, seed, fold, eval_split)]
        logits = np.asarray([[float(row[f"logit_{label}"]) for label in CLASSES] for row in rows])
        y_true = np.asarray([CLASS_TO_INDEX[row["y_true"]] for row in rows], dtype=int)
        temperatures[(feature_set, repeat, seed, fold)] = fit_temperature(logits, y_true)

    metric_rows: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]), int(item[0][3]), item[0][4])):
        feature_set, repeat, seed, fold, _ = key
        logits = np.asarray([[float(row[f"logit_{label}"]) for label in CLASSES] for row in rows])
        raw_probs = normalize_probabilities(np.asarray([[float(row[f"prob_{label}"]) for label in CLASSES] for row in rows]))
        temperature = temperatures[(feature_set, repeat, seed, fold)]
        calibrated_probs = softmax(logits / temperature)
        for mode, probs, temp in [
            ("raw", raw_probs, 1.0),
            ("temperature_scaled", calibrated_probs, temperature),
        ]:
            metric_rows.append(metric_row(key, rows, mode, temp, probs))
    return metric_rows, aggregate_rows(metric_rows)


def markdown_summary(summary_rows: list[dict[str, object]]) -> str:
    test_rows = [
        row
        for row in summary_rows
        if row["eval_split"] == "test" and row["probability_mode"] == "temperature_scaled"
    ]
    test_rows = sorted(test_rows, key=lambda row: float(row["macro_f1_mean"]), reverse=True)
    table = [
        "| Feature set | Macro-F1 | Balanced accuracy | Accuracy | Log loss | ECE | AUROC | QWK |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in test_rows:
        table.append(
            "| {feature_set} | {macro:.4f} | {balanced:.4f} | {accuracy:.4f} | {loss:.4f} | {ece:.4f} | {auroc:.4f} | {qwk:.4f} |".format(
                feature_set=row["feature_set"],
                macro=float(row["macro_f1_mean"]),
                balanced=float(row["balanced_accuracy_mean"]),
                accuracy=float(row["accuracy_mean"]),
                loss=float(row["log_loss_mean"]),
                ece=float(row["ece_10_mean"]),
                auroc=float(row["ovr_auroc_mean"]),
                qwk=float(row["quadratic_weighted_kappa_mean"]),
            )
        )

    image_plus_age = next(
        (
            row
            for row in test_rows
            if row["feature_set"] == "image_embedding_plus_age_sex"
        ),
        None,
    )
    interpretation = "The image-plus-age/sex sensitivity model should be compared against both image-only and age/sex baselines before manuscript drafting."
    if image_plus_age is not None:
        macro_f1 = float(image_plus_age["macro_f1_mean"])
        if macro_f1 >= 0.52:
            interpretation = "The image-plus-age/sex sensitivity model shows a stronger combined signal and supports an incremental-value analysis."
        elif macro_f1 >= 0.50:
            interpretation = "The image-plus-age/sex sensitivity model is improved but still modest; it supports cautious incremental-value wording."
        else:
            interpretation = "The image-plus-age/sex sensitivity model does not yet provide a strong incremental-value result."

    return f"""# Embedding Sensitivity Summary

Date: 2026-06-08

## Test Aggregate Metrics

{chr(10).join(table)}

## Interpretation

{interpretation}

The downstream-status feature set is an upper-bound sensitivity analysis only and should not be described as deployment-ready.
"""


def html_report(summary_rows: list[dict[str, object]]) -> str:
    frame = pd.DataFrame(summary_rows)
    test = frame[frame["eval_split"] == "test"].copy()
    test = test.sort_values(["probability_mode", "macro_f1_mean"], ascending=[True, False])
    columns = [
        "feature_set",
        "probability_mode",
        "folds",
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "accuracy_mean",
        "log_loss_mean",
        "ece_10_mean",
        "ovr_auroc_mean",
        "quadratic_weighted_kappa_mean",
    ]
    table_html = test[columns].to_html(index=False, escape=True, float_format=lambda value: f"{float(value):.4f}")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BRAR Embedding Sensitivity Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
    main {{ max-width: 1100px; margin: 0 auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d7dce2; padding: 7px 8px; text-align: left; }}
    th {{ background: #eef1f4; }}
    .note {{ border-left: 4px solid #795548; background: #f8f5f2; padding: 12px 14px; }}
  </style>
</head>
<body>
<main>
  <h1>BRAR Embedding Sensitivity Report</h1>
  <p class="note">These are sensitivity models using cached EfficientNet-B0 embeddings. The downstream-status model is an upper-bound analysis, not a deployment-ready predictor.</p>
  {table_html}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS)
    parser.add_argument("--c-grid", default="0.01,0.1,1.0,10.0")
    args = parser.parse_args()

    assert_feature_policy()
    c_grid = [float(item.strip()) for item in args.c_grid.split(",") if item.strip()]
    manifest = pd.read_csv(args.manifest)
    split_rows = read_csv(args.splits)
    features, feature_names, labels = load_embeddings(args.embeddings)
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    y_by_name = {name: CLASS_TO_INDEX[label] for name, label in zip(feature_names, labels, strict=True)}

    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for feature_set, config in FEATURE_SETS.items():
        metadata_columns = list(config["metadata_columns"])
        sensitivity_label = str(config["sensitivity_label"])
        for repeat, seed, fold in split_combos(split_rows):
            train_names = rows_for(split_rows, repeat, fold, "train")
            val_names = rows_for(split_rows, repeat, fold, "val")
            test_names = rows_for(split_rows, repeat, fold, "test")

            x_train = make_feature_matrix(features, name_to_index, manifest, train_names, metadata_columns)
            y_train = np.asarray([y_by_name[name] for name in train_names], dtype=np.int64)
            x_val = make_feature_matrix(features, name_to_index, manifest, val_names, metadata_columns)
            y_val = np.asarray([y_by_name[name] for name in val_names], dtype=np.int64)
            x_test = make_feature_matrix(features, name_to_index, manifest, test_names, metadata_columns)
            y_test = np.asarray([y_by_name[name] for name in test_names], dtype=np.int64)

            classifier, selected_c, val_metrics = fit_select_classifier(
                x_train,
                y_train,
                x_val,
                y_val,
                c_grid,
                seed=int(seed) + int(fold),
            )
            selection_rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "fold": fold,
                    "feature_set": feature_set,
                    "sensitivity_label": sensitivity_label,
                    "selected_c": f"{selected_c:.8f}",
                    "validation_macro_f1": f"{val_metrics['macro_f1']:.8f}",
                    "validation_balanced_accuracy": f"{val_metrics['balanced_accuracy']:.8f}",
                    "validation_log_loss": f"{val_metrics['log_loss']:.8f}",
                }
            )
            for eval_split, eval_names, x_eval, y_eval in [
                ("val", val_names, x_val, y_val),
                ("test", test_names, x_test, y_test),
            ]:
                probs = classifier.predict_proba(x_eval)
                logits = classifier.decision_function(x_eval)
                append_predictions(
                    prediction_rows,
                    repeat,
                    seed,
                    fold,
                    eval_split,
                    feature_set,
                    sensitivity_label,
                    selected_c,
                    eval_names,
                    y_eval,
                    probs,
                    logits,
                )
            print(f"trained {feature_set} repeat={repeat} fold={fold} selected_c={selected_c}")

    metric_rows, summary_rows = evaluate_predictions(prediction_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(PREDICTIONS, prediction_rows, PREDICTION_FIELDS)
    write_csv(SELECTION, selection_rows, SELECTION_FIELDS)
    write_csv(METRICS, metric_rows, METRIC_FIELDS)
    write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    SUMMARY_MD.write_text(markdown_summary(summary_rows), encoding="utf-8")
    REPORT_HTML.write_text(html_report(summary_rows), encoding="utf-8")
    RUN_INFO.write_text(
        json.dumps(
            {
                "embeddings": str(args.embeddings.relative_to(ROOT)),
                "feature_sets": FEATURE_SETS,
                "c_grid": c_grid,
                "folds_run": len(split_combos(split_rows)),
                "prediction_rows": len(prediction_rows),
                "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"predictions: {PREDICTIONS}")
    print(f"selection: {SELECTION}")
    print(f"metrics: {METRICS}")
    print(f"summary: {SUMMARY_MD}")
    print(f"html report: {REPORT_HTML}")


if __name__ == "__main__":
    main()
