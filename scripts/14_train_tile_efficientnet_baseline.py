#!/usr/bin/env python3
"""Train a tile-based frozen EfficientNet-B0 image-only BRAR baseline."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "brar_manifest.csv"
SPLITS = ROOT / "data" / "processed" / "splits" / "brar_repeated_5fold_splits.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "image_baseline"

CLASSES = ["1", "2", "3"]
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES)}
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
DEFAULT_C_GRID = [0.01, 0.1, 1.0, 10.0]

PREDICTION_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "eval_split",
    "model",
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
    "encoder",
    "image_width",
    "image_height",
    "tile_count",
    "aggregation",
    "selected_c",
    "validation_macro_f1",
    "validation_balanced_accuracy",
    "validation_log_loss",
]


@dataclass(frozen=True)
class ImageRecord:
    file_name: str
    relative_image_path: str
    severity_level: str


class BrarTileDataset(Dataset[tuple[torch.Tensor, str, int]]):
    def __init__(self, records: list[ImageRecord], tile_size: int, tile_count: int) -> None:
        self.records = records
        self.tile_size = tile_size
        self.tile_count = tile_count

    def __len__(self) -> int:
        return len(self.records) * self.tile_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, int]:
        record_index = index // self.tile_count
        tile_index = index % self.tile_count
        record = self.records[record_index]
        image = Image.open(ROOT / record.relative_image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        tile = horizontal_tile(image, tile_size=self.tile_size, tile_count=self.tile_count, tile_index=tile_index)
        tensor = TF.to_tensor(tile)
        tensor = TF.normalize(tensor, IMAGE_NET_MEAN, IMAGE_NET_STD)
        return tensor, record.file_name, tile_index


def horizontal_tile(image: Image.Image, tile_size: int, tile_count: int, tile_index: int) -> Image.Image:
    width, height = image.size
    scale = tile_size / height
    resized_width = max(tile_size, int(round(width * scale)))
    resized = image.resize((resized_width, tile_size), Image.Resampling.BILINEAR)
    max_start = resized_width - tile_size
    if tile_count <= 1 or max_start <= 0:
        start = max_start // 2
    else:
        starts = np.linspace(0, max_start, tile_count)
        start = int(round(starts[tile_index]))
    return resized.crop((start, 0, start + tile_size, tile_size))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def selected_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_encoder() -> tuple[nn.Module, str]:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier = nn.Identity()
    return model, str(weights)


def manifest_records(path: Path) -> list[ImageRecord]:
    rows = read_csv(path)
    return [
        ImageRecord(
            file_name=row["file_name"],
            relative_image_path=row["relative_image_path"],
            severity_level=row["severity_level"],
        )
        for row in rows
        if row["linkage_status"] == "PASS" and row["readability_status"] == "PASS"
    ]


def embedding_paths(tile_size: int, tile_count: int) -> tuple[Path, Path]:
    stem = f"tile_embeddings_efficientnet_b0_{tile_size}_{tile_count}tiles"
    return OUTPUT_DIR / f"{stem}.npz", OUTPUT_DIR / f"{stem}.json"


def extract_or_load_tile_embeddings(
    records: list[ImageRecord],
    tile_size: int,
    tile_count: int,
    batch_size: int,
    device: torch.device,
    force: bool,
) -> tuple[np.ndarray, list[str], list[str], Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = embedding_paths(tile_size, tile_count)
    if npz_path.exists() and not force:
        data = np.load(npz_path, allow_pickle=False)
        features = data["features"].astype(np.float32)
        file_names = [str(item) for item in data["file_names"]]
        labels = [str(item) for item in data["labels"]]
        return features, file_names, labels, npz_path

    model, weights_name = build_encoder()
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = BrarTileDataset(records, tile_size=tile_size, tile_count=tile_count)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_idx, (images, _names, _tile_indices) in enumerate(loader, start=1):
            images = images.to(device)
            features_tensor = model(images)
            if features_tensor.ndim > 2:
                features_tensor = torch.flatten(features_tensor, 1)
            feature_batches.append(features_tensor.detach().cpu().numpy().astype(np.float32))
            print(f"embedded tile batch {batch_idx}/{len(loader)}")

    flat_features = np.concatenate(feature_batches, axis=0)
    features = flat_features.reshape(len(records), tile_count, flat_features.shape[1])
    file_names = [record.file_name for record in records]
    labels = [record.severity_level for record in records]
    np.savez_compressed(
        npz_path,
        features=features,
        file_names=np.asarray(file_names),
        labels=np.asarray(labels),
    )
    json_path.write_text(
        json.dumps(
            {
                "encoder": "efficientnet_b0",
                "weights": weights_name,
                "tile_size": tile_size,
                "tile_count": tile_count,
                "features_shape": list(features.shape),
                "device": str(device),
                "preprocessing": "RGB conversion, resize to tile height, fixed overlapping horizontal square tiles, ImageNet normalization",
                "aggregation_for_classifier": "concatenated mean and max tile embeddings",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return features, file_names, labels, npz_path


def aggregate_tile_features(features: np.ndarray) -> np.ndarray:
    mean_pool = features.mean(axis=1)
    max_pool = features.max(axis=1)
    return np.concatenate([mean_pool, max_pool], axis=1).astype(np.float32)


def parse_int_list(value: str, all_values: list[int]) -> list[int]:
    if value == "all":
        return all_values
    selected = []
    for token in value.split(","):
        token = token.strip()
        if token:
            selected.append(int(token))
    return selected


def split_index(split_rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
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


def append_prediction_rows(
    out: list[dict[str, object]],
    repeat: str,
    seed: str,
    fold: str,
    eval_split: str,
    tile_size: int,
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
                "model": "tile_frozen_encoder_logistic",
                "encoder": "efficientnet_b0_tiles_meanmax",
                "image_width": tile_size,
                "image_height": tile_size,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--c-grid", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--repeats", default="all")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = selected_device(args.device)
    records = manifest_records(args.manifest)
    tile_features, feature_names, labels, embeddings_path = extract_or_load_tile_embeddings(
        records,
        tile_size=args.tile_size,
        tile_count=args.tile_count,
        batch_size=args.batch_size,
        device=device,
        force=args.force_embeddings,
    )
    features = aggregate_tile_features(tile_features)
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    y_by_name = {name: CLASS_TO_INDEX[label] for name, label in zip(feature_names, labels, strict=True)}

    split_rows = read_csv(args.splits)
    combos = split_index(split_rows)
    all_repeats = sorted({int(repeat) for repeat, _, _ in combos})
    all_folds = sorted({int(fold) for _, _, fold in combos})
    selected_repeats = set(parse_int_list(args.repeats, all_repeats))
    selected_folds = set(parse_int_list(args.folds, all_folds))
    combos = [
        combo
        for combo in combos
        if int(combo[0]) in selected_repeats and int(combo[2]) in selected_folds
    ]
    if args.max_folds:
        combos = combos[: args.max_folds]

    c_grid = [float(item.strip()) for item in args.c_grid.split(",") if item.strip()]
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for repeat, seed, fold in combos:
        train_names = rows_for(split_rows, repeat, fold, "train")
        val_names = rows_for(split_rows, repeat, fold, "val")
        test_names = rows_for(split_rows, repeat, fold, "test")

        train_idx = [name_to_index[name] for name in train_names]
        val_idx = [name_to_index[name] for name in val_names]
        test_idx = [name_to_index[name] for name in test_names]

        x_train = features[train_idx]
        y_train = np.asarray([y_by_name[name] for name in train_names], dtype=np.int64)
        x_val = features[val_idx]
        y_val = np.asarray([y_by_name[name] for name in val_names], dtype=np.int64)
        x_test = features[test_idx]
        y_test = np.asarray([y_by_name[name] for name in test_names], dtype=np.int64)

        classifier, selected_c, val_metrics = fit_select_classifier(
            x_train,
            y_train,
            x_val,
            y_val,
            c_grid,
            seed=args.seed + int(repeat) * 100 + int(fold),
        )
        selection_rows.append(
            {
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                "encoder": "efficientnet_b0_tiles_meanmax",
                "image_width": args.tile_size,
                "image_height": args.tile_size,
                "tile_count": args.tile_count,
                "aggregation": "meanmax",
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
            append_prediction_rows(
                prediction_rows,
                repeat,
                seed,
                fold,
                eval_split,
                args.tile_size,
                selected_c,
                eval_names,
                y_eval,
                probs,
                logits,
            )
        print(f"trained tile repeat={repeat} fold={fold} selected_c={selected_c}")

    prediction_path = OUTPUT_DIR / "tile_efficientnet_b0_384_meanmax_predictions.csv"
    selection_path = OUTPUT_DIR / "tile_efficientnet_b0_384_meanmax_model_selection.csv"
    write_csv(prediction_path, prediction_rows, PREDICTION_FIELDS)
    write_csv(selection_path, selection_rows, SELECTION_FIELDS)
    run_info_path = OUTPUT_DIR / "tile_efficientnet_b0_384_meanmax_run_info.json"
    run_info_path.write_text(
        json.dumps(
            {
                "encoder": "efficientnet_b0",
                "tile_size": args.tile_size,
                "tile_count": args.tile_count,
                "aggregation": "meanmax",
                "device": str(device),
                "embeddings_path": str(embeddings_path.relative_to(ROOT)),
                "folds_run": len(combos),
                "prediction_rows": len(prediction_rows),
                "c_grid": c_grid,
                "design_rationale": "BRAR severity is anchored to the worst tooth/site; fixed overlapping horizontal tiles test whether localized image signal is diluted by whole-panoramic pooling.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"predictions: {prediction_path}")
    print(f"selection: {selection_path}")
    print(f"run_info: {run_info_path}")


if __name__ == "__main__":
    main()
