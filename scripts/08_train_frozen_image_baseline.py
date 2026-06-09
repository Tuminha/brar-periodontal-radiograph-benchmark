#!/usr/bin/env python3
"""Train frozen pretrained image-encoder baselines for BRAR severity."""

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
REPORTS = ROOT / "reports"

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


class BrarImageDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, records: list[ImageRecord], image_width: int, image_height: int) -> None:
        self.records = records
        self.image_width = image_width
        self.image_height = image_height

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        image = Image.open(ROOT / record.relative_image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = aspect_fit_pad(image, self.image_width, self.image_height)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, IMAGE_NET_MEAN, IMAGE_NET_STD)
        return tensor, record.file_name


def aspect_fit_pad(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
    canvas.paste(resized, offset)
    return canvas


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


def build_encoder(name: str) -> tuple[nn.Module, str]:
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
        return model, str(weights)
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        return model, str(weights)
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        return model, str(weights)
    raise ValueError(f"unsupported encoder: {name}")


def manifest_records(path: Path) -> list[ImageRecord]:
    rows = read_csv(path)
    records = [
        ImageRecord(
            file_name=row["file_name"],
            relative_image_path=row["relative_image_path"],
            severity_level=row["severity_level"],
        )
        for row in rows
        if row["linkage_status"] == "PASS" and row["readability_status"] == "PASS"
    ]
    return records


def embedding_paths(encoder: str, width: int, height: int) -> tuple[Path, Path]:
    stem = f"embeddings_{encoder}_{width}x{height}"
    return OUTPUT_DIR / f"{stem}.npz", OUTPUT_DIR / f"{stem}.json"


def extract_or_load_embeddings(
    records: list[ImageRecord],
    encoder_name: str,
    image_width: int,
    image_height: int,
    batch_size: int,
    device: torch.device,
    force: bool,
) -> tuple[np.ndarray, list[str], list[str], Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = embedding_paths(encoder_name, image_width, image_height)
    if npz_path.exists() and not force:
        data = np.load(npz_path, allow_pickle=False)
        features = data["features"].astype(np.float32)
        file_names = [str(item) for item in data["file_names"]]
        labels = [str(item) for item in data["labels"]]
        return features, file_names, labels, npz_path

    model, weights_name = build_encoder(encoder_name)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = BrarImageDataset(records, image_width=image_width, image_height=image_height)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    feature_batches: list[np.ndarray] = []
    file_names: list[str] = []
    with torch.inference_mode():
        for batch_idx, (images, batch_names) in enumerate(loader, start=1):
            images = images.to(device)
            features_tensor = model(images)
            if features_tensor.ndim > 2:
                features_tensor = torch.flatten(features_tensor, 1)
            feature_batches.append(features_tensor.detach().cpu().numpy().astype(np.float32))
            file_names.extend(str(name) for name in batch_names)
            print(f"embedded batch {batch_idx}/{len(loader)}")

    features = np.concatenate(feature_batches, axis=0)
    labels_by_name = {record.file_name: record.severity_level for record in records}
    labels = [labels_by_name[name] for name in file_names]
    np.savez_compressed(
        npz_path,
        features=features,
        file_names=np.asarray(file_names),
        labels=np.asarray(labels),
    )
    json_path.write_text(
        json.dumps(
            {
                "encoder": encoder_name,
                "weights": weights_name,
                "image_width": image_width,
                "image_height": image_height,
                "features_shape": list(features.shape),
                "device": str(device),
                "preprocessing": "RGB conversion, aspect-fit black padding, ImageNet normalization",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return features, file_names, labels, npz_path


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
    combos = sorted(
        {(row["repeat"], row["seed"], row["fold"]) for row in split_rows},
        key=lambda item: (int(item[0]), int(item[2])),
    )
    return combos


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
                        max_iter=2000,
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
    encoder: str,
    image_width: int,
    image_height: int,
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
                "model": "frozen_encoder_logistic",
                "encoder": encoder,
                "image_width": image_width,
                "image_height": image_height,
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
    parser.add_argument("--encoder", default="efficientnet_b0", choices=["efficientnet_b0", "resnet18", "resnet50"])
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--c-grid", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--repeats", default="all")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = selected_device(args.device)
    records = manifest_records(args.manifest)
    features, feature_names, labels, embeddings_path = extract_or_load_embeddings(
        records,
        encoder_name=args.encoder,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        device=device,
        force=args.force_embeddings,
    )

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
                "encoder": args.encoder,
                "image_width": args.image_width,
                "image_height": args.image_height,
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
                args.encoder,
                args.image_width,
                args.image_height,
                selected_c,
                eval_names,
                y_eval,
                probs,
                logits,
            )
        print(f"trained repeat={repeat} fold={fold} selected_c={selected_c}")

    prediction_path = OUTPUT_DIR / f"frozen_{args.encoder}_{args.image_width}x{args.image_height}_predictions.csv"
    selection_path = OUTPUT_DIR / f"frozen_{args.encoder}_{args.image_width}x{args.image_height}_model_selection.csv"
    write_csv(prediction_path, prediction_rows, PREDICTION_FIELDS)
    write_csv(selection_path, selection_rows, SELECTION_FIELDS)
    run_info_path = OUTPUT_DIR / f"frozen_{args.encoder}_{args.image_width}x{args.image_height}_run_info.json"
    run_info_path.write_text(
        json.dumps(
            {
                "encoder": args.encoder,
                "image_width": args.image_width,
                "image_height": args.image_height,
                "device": str(device),
                "embeddings_path": str(embeddings_path.relative_to(ROOT)),
                "folds_run": len(combos),
                "prediction_rows": len(prediction_rows),
                "c_grid": c_grid,
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
