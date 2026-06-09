#!/usr/bin/env python3
"""Create publication-ready uncertainty tables and tile-model error audits.

This script consumes saved predictions only. It does not train models or tune
any decision threshold on test data.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MANIFEST = ROOT / "data" / "processed" / "brar_manifest.csv"
IMAGE_DIR = ROOT / "data" / "processed" / "image_baseline"
SENSITIVITY_DIR = ROOT / "data" / "processed" / "embedding_sensitivity"
NEGATIVE_CONTROL_PREDICTIONS = (
    ROOT / "data" / "processed" / "negative_controls" / "negative_control_predictions.csv"
)

OUTPUT_MODEL_TABLE = REPORTS / "publication_ready_model_table.csv"
OUTPUT_PAIRED_TABLE = REPORTS / "publication_paired_interval_table.csv"
OUTPUT_REPEAT_METRICS = REPORTS / "publication_repeat_level_metrics.csv"
OUTPUT_SUBGROUP_METRICS = REPORTS / "publication_subgroup_tile_metrics.csv"
OUTPUT_ERRORS = REPORTS / "tile_model_confident_errors.csv"
OUTPUT_MD = REPORTS / "publication_uncertainty_summary.md"
OUTPUT_HTML = REPORTS / "publication_uncertainty_report.html"
OUTPUT_JSON = REPORTS / "publication_uncertainty_summary.json"
CONTACT_DIR = REPORTS / "error_audit_contact_sheets"

CLASSES = ["1", "2", "3"]
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES)}
DISPLAY_ORDER = [
    "image_tile_efficientnet_b0_meanmax",
    "image_efficientnet_b0",
    "image_resnet50",
    "age_sex",
    "image_plus_age_sex",
    "image_plus_downstream_age_sex_upper_bound",
    "downstream_plus_age_sex",
    "majority_class",
]
PRIMARY_MODELS = [
    "image_tile_efficientnet_b0_meanmax",
    "image_efficientnet_b0",
    "age_sex",
]
PAIRED_COMPARISONS = [
    ("image_tile_efficientnet_b0_meanmax", "image_efficientnet_b0"),
    ("image_tile_efficientnet_b0_meanmax", "age_sex"),
    ("image_tile_efficientnet_b0_meanmax", "image_resnet50"),
    ("image_tile_efficientnet_b0_meanmax", "image_plus_age_sex"),
]
LOWER_IS_BETTER = {"ordinal_mae", "ece_10"}

PREDICTION_SPECS = [
    {
        "model_id": "image_tile_efficientnet_b0_meanmax",
        "label": "Tile EfficientNet-B0",
        "path": IMAGE_DIR / "tile_efficientnet_b0_384_meanmax_predictions.csv",
        "temperature_scale": True,
        "kind": "image_only",
        "deployment_role": "primary tile image-only benchmark",
    },
    {
        "model_id": "image_efficientnet_b0",
        "label": "Whole-image EfficientNet-B0",
        "path": IMAGE_DIR / "frozen_efficientnet_b0_384x192_predictions.csv",
        "temperature_scale": True,
        "kind": "image_only",
        "deployment_role": "simple whole-image image-only reference",
    },
    {
        "model_id": "image_resnet50",
        "label": "Whole-image ResNet50",
        "path": IMAGE_DIR / "frozen_resnet50_384x192_predictions.csv",
        "temperature_scale": True,
        "kind": "image_only",
        "deployment_role": "predeclared stronger encoder check",
    },
    {
        "model_id": "image_plus_age_sex",
        "label": "Image plus age/sex",
        "path": SENSITIVITY_DIR / "efficientnet_b0_384x192_sensitivity_predictions.csv",
        "temperature_scale": True,
        "kind": "metadata_sensitivity",
        "deployment_role": "metadata sensitivity, not primary",
        "feature_set": "image_embedding_plus_age_sex",
    },
    {
        "model_id": "image_plus_downstream_age_sex_upper_bound",
        "label": "Image plus downstream upper bound",
        "path": SENSITIVITY_DIR / "efficientnet_b0_384x192_sensitivity_predictions.csv",
        "temperature_scale": True,
        "kind": "upper_bound",
        "deployment_role": "non-deployment upper-bound sensitivity",
        "feature_set": "image_embedding_plus_downstream_age_sex_upper_bound",
    },
]

NEGATIVE_MODEL_SPECS = [
    {
        "model_id": "age_sex",
        "label": "Age/sex guardrail",
        "model": "multinomial_logistic",
        "feature_set": "age_sex",
        "kind": "metadata_guardrail",
        "deployment_role": "metadata guardrail",
    },
    {
        "model_id": "downstream_plus_age_sex",
        "label": "Downstream plus age/sex upper bound",
        "model": "multinomial_logistic",
        "feature_set": "downstream_plus_age_sex",
        "kind": "upper_bound",
        "deployment_role": "non-deployment upper-bound sensitivity",
    },
    {
        "model_id": "majority_class",
        "label": "Majority class",
        "model": "majority_class",
        "feature_set": "class_prior",
        "kind": "baseline",
        "deployment_role": "minimum class-prior baseline",
    },
]

MODEL_INFO = {
    str(spec["model_id"]): {
        "label": str(spec["label"]),
        "kind": str(spec["kind"]),
        "deployment_role": str(spec["deployment_role"]),
    }
    for spec in PREDICTION_SPECS + NEGATIVE_MODEL_SPECS
}


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
            ece += float(
                np.mean(mask)
                * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
            )
    return ece


def multiclass_ece(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    pred = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)
    correct = (pred == y_true).astype(float)
    return binned_ece(correct, confidence, bins=bins)


def binary_ece(y_true: np.ndarray, positive_prob: np.ndarray, pred: np.ndarray, bins: int = 10) -> float:
    confidence = np.where(pred == 1, positive_prob, 1.0 - positive_prob)
    correct = (pred == y_true).astype(float)
    return binned_ece(correct, confidence, bins=bins)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    observed = np.zeros((3, 3), dtype=float)
    np.add.at(observed, (y_true.astype(int), y_pred.astype(int)), 1)
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


def safe_binary_auroc(y_true: np.ndarray, positive_prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(positive_prob, method="average")
    rank_sum_pos = float(ranks[positives].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def safe_multiclass_auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    try:
        target = pd.get_dummies(y_true).reindex(columns=[0, 1, 2], fill_value=0)
        return float(roc_auc_score(target, probs, average="macro", multi_class="ovr"))
    except ValueError:
        return math.nan


def multiclass_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = normalize_probabilities(probs)
    y_true = y_true.astype(int)
    pred = np.argmax(probs, axis=1)
    abs_error = np.abs(y_true - pred)
    accuracy = float(np.mean(y_true == pred))
    recalls: list[float] = []
    f1_values: list[float] = []
    for label in range(3):
        true_mask = y_true == label
        pred_mask = pred == label
        tp = int(np.sum(true_mask & pred_mask))
        support = int(np.sum(true_mask))
        pred_count = int(np.sum(pred_mask))
        recall = tp / support if support else 0.0
        precision = tp / pred_count if pred_count else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        recalls.append(float(recall))
        f1_values.append(float(f1))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "ordinal_mae": float(np.mean(abs_error)),
        "two_grade_error_rate": float(np.mean(abs_error == 2)),
        "quadratic_weighted_kappa": float(quadratic_weighted_kappa(y_true, pred)),
        "ece_10": float(multiclass_ece(y_true, probs)),
        "ovr_auroc": safe_multiclass_auroc(y_true, probs),
    }


def severe_metrics(y_true: np.ndarray, positive_prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    pred = pred.astype(int)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    return {
        "severe_balanced_accuracy": float((sensitivity + specificity) / 2.0) if not math.isnan(specificity) else math.nan,
        "severe_sensitivity": float(sensitivity),
        "severe_specificity": float(specificity),
        "severe_f1": float(f1),
        "severe_auroc": safe_binary_auroc(y_true, positive_prob),
        "severe_ece_10": float(binary_ece(y_true, positive_prob, pred)),
        "severe_tp": int(tp),
        "severe_fp": int(fp),
        "severe_tn": int(tn),
        "severe_fn": int(fn),
    }


def format_float(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


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

    out_frames: list[pd.DataFrame] = []
    temperatures: dict[tuple[str, str, str], float] = {}
    if temperature_scale:
        for (repeat, seed, fold, eval_split), group in frame.groupby(
            ["repeat", "seed", "fold", "eval_split"],
            sort=True,
        ):
            if eval_split != "val":
                continue
            logits = group[[f"logit_{label}" for label in CLASSES]].astype(float).to_numpy()
            y_true = group["y_true"].map(CLASS_TO_INDEX).astype(int).to_numpy()
            temperatures[(str(repeat), str(seed), str(fold))] = fit_temperature(logits, y_true)

    for (repeat, seed, fold, eval_split), group in frame.groupby(
        ["repeat", "seed", "fold", "eval_split"],
        sort=True,
    ):
        if temperature_scale:
            temperature = temperatures[(str(repeat), str(seed), str(fold))]
            logits = group[[f"logit_{label}" for label in CLASSES]].astype(float).to_numpy()
            probs = softmax(logits / temperature)
        else:
            temperature = 1.0
            probs = normalize_probabilities(
                group[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
            )
        out = group[["repeat", "seed", "fold", "eval_split", "file_name", "y_true"]].copy()
        out["model_id"] = model_id
        out["kind"] = kind
        out["deployment_role"] = deployment_role
        out["temperature"] = temperature
        for idx, label in enumerate(CLASSES):
            out[f"prob_{label}"] = probs[:, idx]
        out_frames.append(out)

    result = pd.concat(out_frames, ignore_index=True)
    result["repeat"] = result["repeat"].astype(int)
    result["seed"] = result["seed"].astype(int)
    result["fold"] = result["fold"].astype(int)
    result["y_true"] = result["y_true"].astype(str)
    result["y_true_idx"] = result["y_true"].map(CLASS_TO_INDEX).astype(int)
    probs = result[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
    pred_idx = np.argmax(probs, axis=1)
    result["pred_label"] = [CLASSES[idx] for idx in pred_idx]
    result["confidence"] = probs.max(axis=1)
    return result


def load_probability_frame() -> pd.DataFrame:
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

    if not frames:
        raise RuntimeError("No prediction frames found.")
    return pd.concat(frames, ignore_index=True)


def best_validation_threshold(y_true: np.ndarray, positive_prob: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], np.clip(positive_prob, 0.0, 1.0)]))
    if len(candidates) > 301:
        candidates = np.linspace(0.0, 1.0, 301)
    best_threshold = 0.5
    best_ranking: tuple[float, float, float] | None = None
    for threshold in candidates:
        pred = (positive_prob >= threshold).astype(int)
        if len(np.unique(y_true)) < 2:
            continue
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        tn = int(np.sum((y_true == 0) & (pred == 0)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        ranking = (sensitivity + specificity - 1.0, f1, -abs(float(threshold) - 0.5))
        if best_ranking is None or ranking > best_ranking:
            best_ranking = ranking
            best_threshold = float(threshold)
    return best_threshold


def build_severe_test_frame(prob_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (model_id, repeat, seed, fold), group in prob_frame.groupby(
        ["model_id", "repeat", "seed", "fold"],
        sort=True,
    ):
        val = group[group["eval_split"] == "val"]
        test = group[group["eval_split"] == "test"]
        if val.empty or test.empty:
            continue
        y_val = (val["y_true"].astype(int).to_numpy() == 3).astype(int)
        p_val = val["prob_3"].astype(float).to_numpy()
        threshold = best_validation_threshold(y_val, p_val)

        out = test.copy()
        p_test = out["prob_3"].astype(float).to_numpy()
        y_test = (out["y_true"].astype(int).to_numpy() == 3).astype(int)
        out["severe_threshold"] = threshold
        out["severe_y_true"] = y_test
        out["severe_prob"] = p_test
        out["severe_pred"] = (p_test >= threshold).astype(int)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def fold_level_metrics(prob_frame: pd.DataFrame, severe_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test = prob_frame[prob_frame["eval_split"] == "test"]
    for (model_id, repeat, seed, fold), group in test.groupby(["model_id", "repeat", "seed", "fold"], sort=True):
        probs = group[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
        y_true = group["y_true_idx"].astype(int).to_numpy()
        metrics = multiclass_metrics(y_true, probs)
        row: dict[str, object] = {
            "model_id": model_id,
            "repeat": int(repeat),
            "seed": int(seed),
            "fold": int(fold),
            "n": int(len(group)),
        }
        row.update(metrics)
        severe_group = severe_frame[
            (severe_frame["model_id"] == model_id)
            & (severe_frame["repeat"] == repeat)
            & (severe_frame["fold"] == fold)
        ]
        if not severe_group.empty:
            row.update(
                severe_metrics(
                    severe_group["severe_y_true"].astype(int).to_numpy(),
                    severe_group["severe_prob"].astype(float).to_numpy(),
                    severe_group["severe_pred"].astype(int).to_numpy(),
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def repeat_level_metrics(prob_frame: pd.DataFrame, severe_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test = prob_frame[prob_frame["eval_split"] == "test"]
    for (model_id, repeat, seed), group in test.groupby(["model_id", "repeat", "seed"], sort=True):
        probs = group[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
        y_true = group["y_true_idx"].astype(int).to_numpy()
        row: dict[str, object] = {
            "model_id": model_id,
            "model_label": MODEL_INFO[model_id]["label"],
            "kind": MODEL_INFO[model_id]["kind"],
            "repeat": int(repeat),
            "seed": int(seed),
            "n": int(len(group)),
        }
        row.update(multiclass_metrics(y_true, probs))
        severe_group = severe_frame[(severe_frame["model_id"] == model_id) & (severe_frame["repeat"] == repeat)]
        if not severe_group.empty:
            row.update(
                severe_metrics(
                    severe_group["severe_y_true"].astype(int).to_numpy(),
                    severe_group["severe_prob"].astype(float).to_numpy(),
                    severe_group["severe_pred"].astype(int).to_numpy(),
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def image_level_probabilities(prob_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test = prob_frame[prob_frame["eval_split"] == "test"]
    for (model_id, file_name), group in test.groupby(["model_id", "file_name"], sort=True):
        probs = group[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
        mean_probs = normalize_probabilities(np.mean(probs, axis=0, keepdims=True))[0]
        y_true = str(group["y_true"].iloc[0])
        pred_idx = int(np.argmax(mean_probs))
        rows.append(
            {
                "model_id": model_id,
                "model_label": MODEL_INFO[model_id]["label"],
                "kind": MODEL_INFO[model_id]["kind"],
                "deployment_role": MODEL_INFO[model_id]["deployment_role"],
                "file_name": file_name,
                "y_true": y_true,
                "y_true_idx": CLASS_TO_INDEX[y_true],
                "test_appearances": int(len(group)),
                "prob_1": float(mean_probs[0]),
                "prob_2": float(mean_probs[1]),
                "prob_3": float(mean_probs[2]),
                "pred_label": CLASSES[pred_idx],
                "confidence": float(mean_probs[pred_idx]),
            }
        )
    return pd.DataFrame(rows)


def image_level_severe(severe_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_id, file_name), group in severe_frame.groupby(["model_id", "file_name"], sort=True):
        severe_pred_votes = group["severe_pred"].astype(int).to_numpy()
        severe_pred = int(np.mean(severe_pred_votes) >= 0.5)
        rows.append(
            {
                "model_id": model_id,
                "model_label": MODEL_INFO[model_id]["label"],
                "kind": MODEL_INFO[model_id]["kind"],
                "file_name": file_name,
                "severe_y_true": int(group["severe_y_true"].iloc[0]),
                "severe_prob": float(group["severe_prob"].astype(float).mean()),
                "severe_pred": severe_pred,
                "severe_test_appearances": int(len(group)),
                "severe_threshold_mean": float(group["severe_threshold"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for label in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        sampled.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(sampled)


def percentile_interval(values: list[float], confidence_level: float) -> tuple[float, float]:
    clean = np.asarray([value for value in values if not math.isnan(float(value))], dtype=float)
    if clean.size == 0:
        return math.nan, math.nan
    alpha = 1.0 - confidence_level
    return (
        float(np.percentile(clean, 100.0 * alpha / 2.0)),
        float(np.percentile(clean, 100.0 * (1.0 - alpha / 2.0))),
    )


def bootstrap_metric(
    frame: pd.DataFrame,
    metric_func,
    iterations: int,
    confidence_level: float,
    seed: int,
    stratify_col: str,
) -> tuple[float, float, float]:
    estimate = float(metric_func(frame))
    labels = frame[stratify_col].to_numpy()
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        idx = stratified_bootstrap_indices(labels, rng)
        values.append(float(metric_func(frame.iloc[idx])))
    low, high = percentile_interval(values, confidence_level)
    return estimate, low, high


def bootstrap_multiclass_metric_set(
    frame: pd.DataFrame,
    metrics: list[str],
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    y_true = frame["y_true_idx"].astype(int).to_numpy()
    probs = frame[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
    estimates = multiclass_metrics(y_true, probs)
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in metrics}
    for _ in range(iterations):
        idx = stratified_bootstrap_indices(y_true, rng)
        sampled = multiclass_metrics(y_true[idx], probs[idx])
        for metric in metrics:
            values[metric].append(float(sampled[metric]))
    return {
        metric: (float(estimates[metric]), *percentile_interval(values[metric], confidence_level))
        for metric in metrics
    }


def bootstrap_severe_metric_set(
    frame: pd.DataFrame,
    metrics: list[str],
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    y_true = frame["severe_y_true"].astype(int).to_numpy()
    positive_prob = frame["severe_prob"].astype(float).to_numpy()
    pred = frame["severe_pred"].astype(int).to_numpy()
    estimates = severe_metrics(y_true, positive_prob, pred)
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in metrics}
    for _ in range(iterations):
        idx = stratified_bootstrap_indices(y_true, rng)
        sampled = severe_metrics(y_true[idx], positive_prob[idx], pred[idx])
        for metric in metrics:
            values[metric].append(float(sampled[metric]))
    return {
        metric: (float(estimates[metric]), *percentile_interval(values[metric], confidence_level))
        for metric in metrics
    }


def metric_value(frame: pd.DataFrame, metric: str) -> float:
    if metric in {"macro_f1", "balanced_accuracy", "quadratic_weighted_kappa", "ordinal_mae", "ece_10"}:
        probs = frame[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
        y_true = frame["y_true_idx"].astype(int).to_numpy()
        return float(multiclass_metrics(y_true, probs)[metric])
    if metric == "severe_balanced_accuracy":
        return float(
            severe_metrics(
                frame["severe_y_true"].astype(int).to_numpy(),
                frame["severe_prob"].astype(float).to_numpy(),
                frame["severe_pred"].astype(int).to_numpy(),
            )[metric]
        )
    if metric == "severe_auroc":
        return safe_binary_auroc(
            frame["severe_y_true"].astype(int).to_numpy(),
            frame["severe_prob"].astype(float).to_numpy(),
        )
    raise ValueError(f"unknown metric: {metric}")


def build_model_table(
    fold_metrics: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    image_frame: pd.DataFrame,
    severe_image_frame: pd.DataFrame,
    bootstrap_iterations: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    multiclass_metric_names = [
        "macro_f1",
        "balanced_accuracy",
        "quadratic_weighted_kappa",
        "ordinal_mae",
        "ece_10",
    ]
    severe_metric_names = [
        "severe_balanced_accuracy",
        "severe_auroc",
    ]
    metric_names = multiclass_metric_names + severe_metric_names
    for order, model_id in enumerate(DISPLAY_ORDER):
        model_image = image_frame[image_frame["model_id"] == model_id]
        if model_image.empty:
            continue
        model_severe = severe_image_frame[severe_image_frame["model_id"] == model_id]
        fold_model = fold_metrics[fold_metrics["model_id"] == model_id]
        repeat_model = repeat_metrics[repeat_metrics["model_id"] == model_id]
        row: dict[str, object] = {
            "model_id": model_id,
            "model_label": MODEL_INFO[model_id]["label"],
            "kind": MODEL_INFO[model_id]["kind"],
            "deployment_role": MODEL_INFO[model_id]["deployment_role"],
            "display_order": order,
            "cv_folds": int(len(fold_model)),
            "image_count": int(model_image["file_name"].nunique()),
            "test_appearances_mean": float(model_image["test_appearances"].mean()),
        }
        intervals: dict[str, tuple[float, float, float]] = {}
        intervals.update(
            bootstrap_multiclass_metric_set(
                model_image,
                multiclass_metric_names,
                bootstrap_iterations,
                confidence_level,
                seed + order * 1000,
            )
        )
        if not model_severe.empty:
            intervals.update(
                bootstrap_severe_metric_set(
                    model_severe,
                    severe_metric_names,
                    bootstrap_iterations,
                    confidence_level,
                    seed + order * 1000 + 503,
                )
            )
        for metric in metric_names:
            estimate, low, high = intervals.get(metric, (math.nan, math.nan, math.nan))
            row[f"oof_{metric}"] = estimate
            row[f"oof_{metric}_low"] = low
            row[f"oof_{metric}_high"] = high
            if metric in fold_model:
                row[f"cv_{metric}_mean"] = float(fold_model[metric].astype(float).mean())
                row[f"cv_{metric}_sd"] = float(fold_model[metric].astype(float).std(ddof=1))
            if metric in repeat_model:
                row[f"repeat_{metric}_min"] = float(repeat_model[metric].astype(float).min())
                row[f"repeat_{metric}_max"] = float(repeat_model[metric].astype(float).max())
        rows.append(row)
    return rows


def paired_source_frames(
    metric: str,
    image_frame: pd.DataFrame,
    severe_image_frame: pd.DataFrame,
    left_model: str,
    right_model: str,
) -> pd.DataFrame:
    source = severe_image_frame if metric.startswith("severe_") else image_frame
    left = source[source["model_id"] == left_model].set_index("file_name")
    right = source[source["model_id"] == right_model].set_index("file_name")
    common = left.index.intersection(right.index)
    rows: list[dict[str, object]] = []
    for file_name in common:
        left_row = left.loc[file_name]
        right_row = right.loc[file_name]
        if metric.startswith("severe_"):
            rows.append(
                {
                    "file_name": file_name,
                    "severe_y_true": int(left_row["severe_y_true"]),
                    "left_severe_prob": float(left_row["severe_prob"]),
                    "left_severe_pred": int(left_row["severe_pred"]),
                    "right_severe_prob": float(right_row["severe_prob"]),
                    "right_severe_pred": int(right_row["severe_pred"]),
                }
            )
        else:
            item: dict[str, object] = {
                "file_name": file_name,
                "y_true_idx": int(left_row["y_true_idx"]),
            }
            for label in CLASSES:
                item[f"left_prob_{label}"] = float(left_row[f"prob_{label}"])
                item[f"right_prob_{label}"] = float(right_row[f"prob_{label}"])
            rows.append(item)
    return pd.DataFrame(rows)


def paired_metric_value(frame: pd.DataFrame, metric: str, side: str) -> float:
    if metric.startswith("severe_"):
        return float(
            severe_metrics(
                frame["severe_y_true"].astype(int).to_numpy(),
                frame[f"{side}_severe_prob"].astype(float).to_numpy(),
                frame[f"{side}_severe_pred"].astype(int).to_numpy(),
            )[metric]
        )
    probs = frame[[f"{side}_prob_{label}" for label in CLASSES]].astype(float).to_numpy()
    y_true = frame["y_true_idx"].astype(int).to_numpy()
    return float(multiclass_metrics(y_true, probs)[metric])


def paired_bootstrap_metric_set(
    frame: pd.DataFrame,
    metrics: list[str],
    iterations: int,
    confidence_level: float,
    seed: int,
    severe: bool,
) -> dict[str, tuple[float, float, float, float, float]]:
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in metrics}
    if severe:
        labels = frame["severe_y_true"].astype(int).to_numpy()
        left_prob = frame["left_severe_prob"].astype(float).to_numpy()
        left_pred = frame["left_severe_pred"].astype(int).to_numpy()
        right_prob = frame["right_severe_prob"].astype(float).to_numpy()
        right_pred = frame["right_severe_pred"].astype(int).to_numpy()
        left_estimates = severe_metrics(
            labels,
            left_prob,
            left_pred,
        )
        right_estimates = severe_metrics(
            labels,
            right_prob,
            right_pred,
        )
        for _ in range(iterations):
            idx = stratified_bootstrap_indices(labels, rng)
            left_sample = severe_metrics(
                labels[idx],
                left_prob[idx],
                left_pred[idx],
            )
            right_sample = severe_metrics(
                labels[idx],
                right_prob[idx],
                right_pred[idx],
            )
            for metric in metrics:
                values[metric].append(float(left_sample[metric]) - float(right_sample[metric]))
    else:
        labels = frame["y_true_idx"].astype(int).to_numpy()
        left_probs = frame[[f"left_prob_{label}" for label in CLASSES]].astype(float).to_numpy()
        right_probs = frame[[f"right_prob_{label}" for label in CLASSES]].astype(float).to_numpy()
        left_estimates = multiclass_metrics(labels, left_probs)
        right_estimates = multiclass_metrics(labels, right_probs)
        for _ in range(iterations):
            idx = stratified_bootstrap_indices(labels, rng)
            left_sample = multiclass_metrics(labels[idx], left_probs[idx])
            right_sample = multiclass_metrics(labels[idx], right_probs[idx])
            for metric in metrics:
                values[metric].append(float(left_sample[metric]) - float(right_sample[metric]))

    out: dict[str, tuple[float, float, float, float, float]] = {}
    for metric in metrics:
        left_value = float(left_estimates[metric])
        right_value = float(right_estimates[metric])
        delta = left_value - right_value
        low, high = percentile_interval(values[metric], confidence_level)
        out[metric] = (left_value, right_value, delta, low, high)
    return out


def build_paired_table(
    image_frame: pd.DataFrame,
    severe_image_frame: pd.DataFrame,
    bootstrap_iterations: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    multiclass_metrics_to_compare = [
        "macro_f1",
        "balanced_accuracy",
        "quadratic_weighted_kappa",
        "ordinal_mae",
        "ece_10",
    ]
    severe_metrics_to_compare = [
        "severe_balanced_accuracy",
        "severe_auroc",
    ]
    for comparison_idx, (left_model, right_model) in enumerate(PAIRED_COMPARISONS):
        grouped_sources = [
            (
                paired_source_frames("macro_f1", image_frame, severe_image_frame, left_model, right_model),
                multiclass_metrics_to_compare,
                False,
                seed + comparison_idx * 5000,
            ),
            (
                paired_source_frames(
                    "severe_balanced_accuracy",
                    image_frame,
                    severe_image_frame,
                    left_model,
                    right_model,
                ),
                severe_metrics_to_compare,
                True,
                seed + comparison_idx * 5000 + 271,
            ),
        ]
        for merged, metrics, severe, metric_seed in grouped_sources:
            if merged.empty:
                continue
            intervals = paired_bootstrap_metric_set(
                merged,
                metrics,
                bootstrap_iterations,
                confidence_level,
                metric_seed,
                severe,
            )
            for metric in metrics:
                left_value, right_value, estimate, low, high = intervals[metric]
                lower_better = metric in LOWER_IS_BETTER
                if lower_better:
                    if high < 0:
                        interpretation = "left_interval_better"
                    elif low > 0:
                        interpretation = "right_interval_better"
                    else:
                        interpretation = "overlaps_no_difference"
                else:
                    if low > 0:
                        interpretation = "left_interval_better"
                    elif high < 0:
                        interpretation = "right_interval_better"
                    else:
                        interpretation = "overlaps_no_difference"
                rows.append(
                    {
                        "left_model": left_model,
                        "left_label": MODEL_INFO[left_model]["label"],
                        "right_model": right_model,
                        "right_label": MODEL_INFO[right_model]["label"],
                        "metric": metric,
                        "paired_images": int(len(merged)),
                        "left_value": left_value,
                        "right_value": right_value,
                        "delta_left_minus_right": estimate,
                        "delta_low": low,
                        "delta_high": high,
                        "lower_is_better": lower_better,
                        "interpretation": interpretation,
                    }
                )
    return rows


def add_subgroup_columns(manifest: pd.DataFrame) -> pd.DataFrame:
    metadata = manifest.copy()
    metadata["age_numeric"] = pd.to_numeric(metadata["age"], errors="coerce")
    metadata["size_bytes_numeric"] = pd.to_numeric(metadata["size_bytes"], errors="coerce")
    metadata["aspect_ratio_numeric"] = pd.to_numeric(metadata["aspect_ratio"], errors="coerce")
    metadata["age_band"] = pd.cut(
        metadata["age_numeric"],
        bins=[-math.inf, 34, 50, math.inf],
        labels=["age_<35", "age_35_50", "age_>50"],
    ).astype(str)
    metadata["gender_group"] = "gender_" + metadata["gender"].astype(str)
    metadata["aspect_ratio_tertile"] = pd.qcut(
        metadata["aspect_ratio_numeric"],
        q=3,
        labels=["aspect_low", "aspect_mid", "aspect_high"],
        duplicates="drop",
    ).astype(str)
    metadata["file_size_tertile"] = pd.qcut(
        metadata["size_bytes_numeric"],
        q=3,
        labels=["file_size_small", "file_size_mid", "file_size_large"],
        duplicates="drop",
    ).astype(str)
    return metadata


def build_subgroup_metrics(
    image_frame: pd.DataFrame,
    severe_image_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    min_subgroup_n: int,
) -> list[dict[str, object]]:
    metadata = add_subgroup_columns(manifest)
    keep_cols = [
        "file_name",
        "age_band",
        "gender_group",
        "aspect_ratio_tertile",
        "file_size_tertile",
    ]
    frame = image_frame[image_frame["model_id"].isin(PRIMARY_MODELS)].merge(
        metadata[keep_cols],
        on="file_name",
        how="left",
    )
    severe = severe_image_frame[severe_image_frame["model_id"].isin(PRIMARY_MODELS)]
    rows: list[dict[str, object]] = []
    subgroup_defs = [
        ("age_band", "age_band"),
        ("gender", "gender_group"),
        ("aspect_ratio_tertile", "aspect_ratio_tertile"),
        ("file_size_tertile", "file_size_tertile"),
    ]
    for subgroup_type, column in subgroup_defs:
        for (model_id, subgroup), group in frame.groupby(["model_id", column], dropna=False, sort=True):
            if subgroup == "nan" or len(group) < min_subgroup_n:
                continue
            probs = group[[f"prob_{label}" for label in CLASSES]].astype(float).to_numpy()
            y_true = group["y_true_idx"].astype(int).to_numpy()
            metrics = multiclass_metrics(y_true, probs)
            severe_group = severe[
                (severe["model_id"] == model_id)
                & (severe["file_name"].isin(set(group["file_name"].astype(str))))
            ]
            severe_values = (
                severe_metrics(
                    severe_group["severe_y_true"].astype(int).to_numpy(),
                    severe_group["severe_prob"].astype(float).to_numpy(),
                    severe_group["severe_pred"].astype(int).to_numpy(),
                )
                if not severe_group.empty and severe_group["severe_y_true"].nunique() > 1
                else {}
            )
            row: dict[str, object] = {
                "model_id": model_id,
                "model_label": MODEL_INFO[model_id]["label"],
                "subgroup_type": subgroup_type,
                "subgroup": str(subgroup),
                "n": int(len(group)),
                "positive_severe_n": int(severe_group["severe_y_true"].sum()) if not severe_group.empty else 0,
            }
            for metric in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "quadratic_weighted_kappa",
                "ordinal_mae",
                "ece_10",
            ]:
                row[metric] = metrics[metric]
            for metric in [
                "severe_balanced_accuracy",
                "severe_sensitivity",
                "severe_specificity",
                "severe_auroc",
            ]:
                row[metric] = severe_values.get(metric, math.nan)
            rows.append(row)
    return rows


def build_error_audit(
    prob_frame: pd.DataFrame,
    severe_frame: pd.DataFrame,
    manifest: pd.DataFrame,
) -> list[dict[str, object]]:
    tile = prob_frame[
        (prob_frame["model_id"] == "image_tile_efficientnet_b0_meanmax")
        & (prob_frame["eval_split"] == "test")
    ].copy()
    severe = severe_frame[severe_frame["model_id"] == "image_tile_efficientnet_b0_meanmax"].copy()
    metadata = manifest[
        [
            "file_name",
            "absolute_image_path",
            "age",
            "gender",
            "number_of_missing_teeth",
            "implant",
            "residual_root",
            "pixel_width",
            "pixel_height",
            "size_bytes",
        ]
    ]
    tile["is_error"] = tile["pred_label"].astype(str) != tile["y_true"].astype(str)
    tile["two_grade_error"] = (
        tile["pred_label"].map(CLASS_TO_INDEX).astype(int) - tile["y_true_idx"].astype(int)
    ).abs() == 2
    rows: list[dict[str, object]] = []
    for file_name, group in tile.groupby("file_name", sort=True):
        severe_group = severe[severe["file_name"] == file_name]
        error_group = group[group["is_error"]]
        if error_group.empty and severe_group.empty:
            continue
        two_grade_group = group[group["two_grade_error"]]
        severe_false_negatives = severe_group[
            (severe_group["severe_y_true"].astype(int) == 1)
            & (severe_group["severe_pred"].astype(int) == 0)
        ]
        severe_false_positives = severe_group[
            (severe_group["severe_y_true"].astype(int) == 0)
            & (severe_group["severe_pred"].astype(int) == 1)
        ]
        if error_group.empty and severe_false_negatives.empty and severe_false_positives.empty:
            continue
        first = group.iloc[0]
        meta = metadata[metadata["file_name"] == file_name].iloc[0]
        pred_counter = Counter(error_group["pred_label"].astype(str))
        true_counter = Counter(group["y_true"].astype(str))
        wrong_conf = error_group["confidence"].astype(float)
        row = {
            "file_name": file_name,
            "absolute_image_path": meta["absolute_image_path"],
            "true_label_mode": true_counter.most_common(1)[0][0],
            "test_appearances": int(len(group)),
            "wrong_predictions": int(len(error_group)),
            "most_common_wrong_pred": pred_counter.most_common(1)[0][0] if pred_counter else "",
            "mean_wrong_confidence": float(wrong_conf.mean()) if len(wrong_conf) else math.nan,
            "max_wrong_confidence": float(wrong_conf.max()) if len(wrong_conf) else math.nan,
            "two_grade_error_count": int(len(two_grade_group)),
            "severe_false_negative_count": int(len(severe_false_negatives)),
            "severe_false_positive_count": int(len(severe_false_positives)),
            "mean_severe_prob": float(severe_group["severe_prob"].astype(float).mean()) if not severe_group.empty else math.nan,
            "mean_severe_threshold": float(severe_group["severe_threshold"].astype(float).mean()) if not severe_group.empty else math.nan,
            "age": meta["age"],
            "gender": meta["gender"],
            "number_of_missing_teeth": meta["number_of_missing_teeth"],
            "implant": meta["implant"],
            "residual_root": meta["residual_root"],
            "pixel_width": meta["pixel_width"],
            "pixel_height": meta["pixel_height"],
            "size_bytes": meta["size_bytes"],
            "rank_score": (
                int(len(error_group)) * 1000
                + int(len(severe_false_negatives)) * 100
                + int(len(two_grade_group)) * 50
                + float(wrong_conf.mean() if len(wrong_conf) else 0.0)
            ),
            "example_pred_label": str(first["pred_label"]),
            "example_confidence": float(first["confidence"]),
        }
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -float(row["rank_score"]),
            -float(row["mean_wrong_confidence"]) if not math.isnan(float(row["mean_wrong_confidence"])) else 0.0,
            str(row["file_name"]),
        ),
    )


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width_chars: int,
    fill: tuple[int, int, int],
) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars):
        draw.text((x, y), line, fill=fill)
        y += 13
    return y


def render_contact_sheet(
    rows: list[dict[str, object]],
    output_path: Path,
    title: str,
    limit: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:limit]
    if not selected:
        image = Image.new("RGB", (900, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), title, fill=(20, 20, 20))
        draw.text((20, 48), "No cases matched this contact sheet.", fill=(80, 80, 80))
        image.save(output_path)
        return

    columns = 4
    panel_w = 330
    panel_h = 235
    title_h = 42
    rows_n = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * panel_w, title_h + rows_n * panel_h), (246, 247, 248))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, title_h), fill=(32, 48, 62))
    draw.text((14, 13), title, fill=(255, 255, 255))

    for idx, row in enumerate(selected):
        col = idx % columns
        row_idx = idx // columns
        x0 = col * panel_w
        y0 = title_h + row_idx * panel_h
        draw.rectangle((x0 + 8, y0 + 8, x0 + panel_w - 8, y0 + panel_h - 8), fill=(255, 255, 255))
        try:
            image = Image.open(str(row["absolute_image_path"]))
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((panel_w - 28, 132))
            image_x = x0 + (panel_w - image.width) // 2
            sheet.paste(image, (image_x, y0 + 16))
        except Exception:
            draw.rectangle((x0 + 14, y0 + 16, x0 + panel_w - 14, y0 + 148), fill=(224, 228, 232))
            draw.text((x0 + 20, y0 + 70), "Image failed to load", fill=(80, 80, 80))

        text_y = y0 + 154
        name = str(row["file_name"])
        if len(name) > 39:
            name = name[:36] + "..."
        text_y = draw_wrapped_text(draw, name, (x0 + 16, text_y), 39, (30, 30, 30))
        pred = row.get("most_common_wrong_pred") or row.get("example_pred_label") or ""
        conf = row.get("mean_wrong_confidence")
        conf_text = format_float(conf) if conf not in ("", None) else format_float(row.get("example_confidence"))
        details = (
            f"true {row['true_label_mode']} -> pred {pred}; conf {conf_text}; "
            f"age {row['age']}; sex {row['gender']}; missing {row['number_of_missing_teeth']}"
        )
        draw_wrapped_text(draw, details, (x0 + 16, text_y + 2), 41, (74, 74, 74))
    sheet.save(output_path)


def build_contact_sheets(error_rows: list[dict[str, object]], limit: int) -> dict[str, str]:
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    top_errors = error_rows
    severe_false_negatives = [
        row for row in error_rows if int(row["severe_false_negative_count"]) > 0
    ]
    two_grade_errors = [
        row for row in error_rows if int(row["two_grade_error_count"]) > 0
    ]
    outputs = {
        "top_confident_errors": CONTACT_DIR / "top_confident_tile_errors.png",
        "severe_false_negatives": CONTACT_DIR / "severe_false_negatives.png",
        "two_grade_errors": CONTACT_DIR / "two_grade_errors.png",
    }
    render_contact_sheet(top_errors, outputs["top_confident_errors"], "Tile model: top confident errors", limit)
    render_contact_sheet(
        severe_false_negatives,
        outputs["severe_false_negatives"],
        "Tile model: severe false negatives",
        limit,
    )
    render_contact_sheet(two_grade_errors, outputs["two_grade_errors"], "Tile model: two-grade errors", limit)
    return {key: str(path.relative_to(ROOT)) for key, path in outputs.items()}


def model_table_markdown(rows: list[dict[str, object]]) -> str:
    columns = [
        "model_label",
        "kind",
        "image_count",
        "cv_macro_f1_mean",
        "oof_macro_f1",
        "oof_macro_f1_low",
        "oof_macro_f1_high",
        "cv_balanced_accuracy_mean",
        "oof_severe_balanced_accuracy",
        "oof_severe_auroc",
    ]
    return markdown_table(rows, columns)


def paired_table_markdown(rows: list[dict[str, object]]) -> str:
    selected = [
        row
        for row in rows
        if row["metric"]
        in {"macro_f1", "balanced_accuracy", "quadratic_weighted_kappa", "severe_balanced_accuracy", "severe_auroc"}
    ]
    columns = [
        "left_label",
        "right_label",
        "metric",
        "delta_left_minus_right",
        "delta_low",
        "delta_high",
        "interpretation",
    ]
    return markdown_table(selected, columns)


def markdown_table(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (float, np.floating)):
                cells.append(format_float(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(
    model_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    subgroup_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
    contact_sheets: dict[str, str],
    bootstrap_iterations: int,
    confidence_level: float,
) -> str:
    tile = next(row for row in model_rows if row["model_id"] == "image_tile_efficientnet_b0_meanmax")
    whole = next(row for row in model_rows if row["model_id"] == "image_efficientnet_b0")
    age = next(row for row in model_rows if row["model_id"] == "age_sex")
    subgroup_display = sorted(
        subgroup_rows,
        key=lambda row: (str(row["subgroup_type"]), str(row["subgroup"]), str(row["model_id"])),
    )[:30]
    return f"""# Publication Uncertainty And Error Audit

Date: 2026-06-09

## Recommendation

**Proceed toward a benchmark manuscript outline, with the tile EfficientNet-B0 model as the primary image-only benchmark.**

The new analysis keeps the current claim boundary: this is a reproducible, leakage-aware, calibrated BRAR benchmark with metadata guardrails and a severe-grade secondary endpoint, not a clinical-grade deployment model.

Uncertainty uses image-level out-of-fold bootstrap intervals after averaging each image's three test appearances. Severe-grade decisions use validation-selected thresholds only, then majority vote across the three held-out test appearances.

## Primary Table

{model_table_markdown(model_rows)}

Key reference points:

- Tile EfficientNet-B0 CV macro-F1 `{format_float(tile["cv_macro_f1_mean"])}` and balanced accuracy `{format_float(tile["cv_balanced_accuracy_mean"])}`.
- Whole-image EfficientNet-B0 CV macro-F1 `{format_float(whole["cv_macro_f1_mean"])}` and balanced accuracy `{format_float(whole["cv_balanced_accuracy_mean"])}`.
- Age/sex guardrail CV macro-F1 `{format_float(age["cv_macro_f1_mean"])}` and balanced accuracy `{format_float(age["cv_balanced_accuracy_mean"])}`.
- Tile severe-grade CV balanced accuracy `{format_float(tile["cv_severe_balanced_accuracy_mean"])}` and severe AUROC `{format_float(tile["cv_severe_auroc_mean"])}`.

## Paired Image-Level Intervals

Positive deltas favor the tile model except for ordinal MAE and ECE, where lower is better. These are interval estimates, not p-values.

{paired_table_markdown(paired_rows)}

## Subgroup Checks

Subgroup rows use image-level out-of-fold probabilities and skip cells with fewer than the configured minimum sample size.

{markdown_table(subgroup_display, ["model_label", "subgroup_type", "subgroup", "n", "macro_f1", "balanced_accuracy", "severe_balanced_accuracy", "severe_auroc"])}

## Tile Error Audit

Tile-model error rows: `{len(error_rows)}`.

Contact sheets:

- Top confident errors: `{contact_sheets["top_confident_errors"]}`
- Severe false negatives: `{contact_sheets["severe_false_negatives"]}`
- Two-grade errors: `{contact_sheets["two_grade_errors"]}`

Top error cases:

{markdown_table(error_rows, ["file_name", "true_label_mode", "wrong_predictions", "most_common_wrong_pred", "mean_wrong_confidence", "severe_false_negative_count", "two_grade_error_count", "age", "gender", "number_of_missing_teeth"], limit=15)}

## Method Notes

- Bootstrap iterations: `{bootstrap_iterations}`.
- Confidence level: `{confidence_level}`.
- Temperature scaling is fitted on validation folds only.
- Severe-grade thresholds are fitted on validation folds only.
- Metadata and downstream-status models remain guardrails or sensitivity analyses, not primary deployable models.
"""


def dataframe_html(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return "<p>No rows.</p>"
    return frame[columns].to_html(index=False, escape=True, float_format=lambda value: f"{value:.4f}")


def render_html(
    model_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    subgroup_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
    contact_sheets: dict[str, str],
) -> str:
    model_cols = [
        "model_label",
        "kind",
        "image_count",
        "cv_macro_f1_mean",
        "cv_macro_f1_sd",
        "oof_macro_f1",
        "oof_macro_f1_low",
        "oof_macro_f1_high",
        "cv_balanced_accuracy_mean",
        "oof_severe_balanced_accuracy",
        "oof_severe_auroc",
    ]
    paired_cols = [
        "left_label",
        "right_label",
        "metric",
        "paired_images",
        "left_value",
        "right_value",
        "delta_left_minus_right",
        "delta_low",
        "delta_high",
        "interpretation",
    ]
    subgroup_cols = [
        "model_label",
        "subgroup_type",
        "subgroup",
        "n",
        "macro_f1",
        "balanced_accuracy",
        "quadratic_weighted_kappa",
        "severe_balanced_accuracy",
        "severe_auroc",
    ]
    error_cols = [
        "file_name",
        "true_label_mode",
        "wrong_predictions",
        "most_common_wrong_pred",
        "mean_wrong_confidence",
        "severe_false_negative_count",
        "two_grade_error_count",
        "age",
        "gender",
        "number_of_missing_teeth",
    ]
    sheet_items = []
    for label, path in contact_sheets.items():
        href = Path(path)
        if href.parts and href.parts[0] == "reports":
            href = Path(*href.parts[1:])
        sheet_items.append(
            f'<li><a href="{html.escape(str(href))}">{html.escape(label.replace("_", " ").title())}</a></li>'
        )
    sheet_links = "".join(sheet_items)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BRAR Publication Uncertainty And Error Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
    main {{ max-width: 1240px; margin: 0 auto; }}
    h1, h2 {{ line-height: 1.2; }}
    .callout {{ border-left: 4px solid #2457a6; background: #f4f7fb; padding: 14px 16px; margin: 16px 0 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d7dce2; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef1f4; position: sticky; top: 0; }}
    .note {{ color: #5f6368; }}
  </style>
</head>
<body>
<main>
  <h1>BRAR Publication Uncertainty And Error Audit</h1>
  <div class="callout">Tile EfficientNet-B0 remains the primary image-only benchmark. Intervals use image-level out-of-fold bootstrap resampling, not p-values.</div>

  <h2>Publication-Ready Model Table</h2>
  {dataframe_html(model_rows, model_cols)}

  <h2>Paired Image-Level Intervals</h2>
  <p class="note">Positive deltas favor the left model except for ordinal MAE and ECE.</p>
  {dataframe_html(paired_rows, paired_cols)}

  <h2>Subgroup Checks</h2>
  {dataframe_html(subgroup_rows, subgroup_cols)}

  <h2>Error Audit Contact Sheets</h2>
  <ul>{sheet_links}</ul>

  <h2>Tile Model Confident Errors</h2>
  {dataframe_html(error_rows, error_cols, limit=50)}
</main>
</body>
</html>
"""


def validate_outputs(model_rows: list[dict[str, object]], contact_sheets: dict[str, str]) -> None:
    required_models = {
        "image_tile_efficientnet_b0_meanmax",
        "image_efficientnet_b0",
        "age_sex",
    }
    present_models = {str(row["model_id"]) for row in model_rows}
    missing = required_models - present_models
    if missing:
        raise RuntimeError(f"missing required models in model table: {sorted(missing)}")

    for row in model_rows:
        if int(row["image_count"]) != 988:
            raise RuntimeError(f"{row['model_id']} image_count is {row['image_count']}, expected 988")
        for key, value in row.items():
            if key.endswith("_low"):
                base = key.removesuffix("_low")
                estimate = float(row[base])
                low = float(value)
                high = float(row[f"{base}_high"])
                if not (low <= estimate <= high):
                    raise RuntimeError(f"interval check failed for {row['model_id']} {base}")

    for path in contact_sheets.values():
        full_path = ROOT / path
        if not full_path.exists() or full_path.stat().st_size == 0:
            raise RuntimeError(f"missing contact sheet: {full_path}")
        with Image.open(full_path) as image:
            image.verify()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--contact-sheet-limit", type=int, default=24)
    parser.add_argument("--min-subgroup-n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    prob_frame = load_probability_frame()
    severe_frame = build_severe_test_frame(prob_frame)
    fold_metrics = fold_level_metrics(prob_frame, severe_frame)
    repeat_metrics = repeat_level_metrics(prob_frame, severe_frame)
    image_frame = image_level_probabilities(prob_frame)
    severe_image_frame = image_level_severe(severe_frame)
    manifest = pd.read_csv(MANIFEST)

    model_rows = build_model_table(
        fold_metrics,
        repeat_metrics,
        image_frame,
        severe_image_frame,
        args.bootstrap_iterations,
        args.confidence_level,
        args.seed,
    )
    paired_rows = build_paired_table(
        image_frame,
        severe_image_frame,
        args.bootstrap_iterations,
        args.confidence_level,
        args.seed,
    )
    subgroup_rows = build_subgroup_metrics(
        image_frame,
        severe_image_frame,
        manifest,
        args.min_subgroup_n,
    )
    error_rows = build_error_audit(prob_frame, severe_frame, manifest)
    contact_sheets = build_contact_sheets(error_rows, args.contact_sheet_limit)
    validate_outputs(model_rows, contact_sheets)

    model_fieldnames = sorted({key for row in model_rows for key in row.keys()})
    paired_fieldnames = sorted({key for row in paired_rows for key in row.keys()})
    subgroup_fieldnames = sorted({key for row in subgroup_rows for key in row.keys()})
    error_fieldnames = [
        "file_name",
        "absolute_image_path",
        "true_label_mode",
        "test_appearances",
        "wrong_predictions",
        "most_common_wrong_pred",
        "mean_wrong_confidence",
        "max_wrong_confidence",
        "two_grade_error_count",
        "severe_false_negative_count",
        "severe_false_positive_count",
        "mean_severe_prob",
        "mean_severe_threshold",
        "age",
        "gender",
        "number_of_missing_teeth",
        "implant",
        "residual_root",
        "pixel_width",
        "pixel_height",
        "size_bytes",
        "rank_score",
        "example_pred_label",
        "example_confidence",
    ]
    repeat_fieldnames = [
        "model_id",
        "model_label",
        "kind",
        "repeat",
        "seed",
        "n",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "ordinal_mae",
        "two_grade_error_rate",
        "quadratic_weighted_kappa",
        "ece_10",
        "ovr_auroc",
        "severe_balanced_accuracy",
        "severe_sensitivity",
        "severe_specificity",
        "severe_f1",
        "severe_auroc",
        "severe_ece_10",
        "severe_tp",
        "severe_fp",
        "severe_tn",
        "severe_fn",
    ]

    write_csv(OUTPUT_MODEL_TABLE, model_rows, model_fieldnames)
    write_csv(OUTPUT_PAIRED_TABLE, paired_rows, paired_fieldnames)
    write_csv(OUTPUT_REPEAT_METRICS, repeat_metrics.to_dict("records"), repeat_fieldnames)
    write_csv(OUTPUT_SUBGROUP_METRICS, subgroup_rows, subgroup_fieldnames)
    write_csv(OUTPUT_ERRORS, error_rows, error_fieldnames)
    OUTPUT_MD.write_text(
        render_markdown(
            model_rows,
            paired_rows,
            subgroup_rows,
            error_rows,
            contact_sheets,
            args.bootstrap_iterations,
            args.confidence_level,
        ),
        encoding="utf-8",
    )
    OUTPUT_HTML.write_text(
        render_html(model_rows, paired_rows, subgroup_rows, error_rows, contact_sheets),
        encoding="utf-8",
    )
    OUTPUT_JSON.write_text(
        json.dumps(
            json_safe(
                {
                "bootstrap_iterations": args.bootstrap_iterations,
                "confidence_level": args.confidence_level,
                "model_table": model_rows,
                "paired_intervals": paired_rows,
                "subgroup_rows": subgroup_rows,
                "tile_error_rows": error_rows,
                "contact_sheets": contact_sheets,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"model table: {OUTPUT_MODEL_TABLE}")
    print(f"paired intervals: {OUTPUT_PAIRED_TABLE}")
    print(f"repeat metrics: {OUTPUT_REPEAT_METRICS}")
    print(f"subgroup metrics: {OUTPUT_SUBGROUP_METRICS}")
    print(f"tile errors: {OUTPUT_ERRORS}")
    print(f"summary: {OUTPUT_MD}")
    print(f"html: {OUTPUT_HTML}")
    print(f"contact sheets: {CONTACT_DIR}")


if __name__ == "__main__":
    main()
