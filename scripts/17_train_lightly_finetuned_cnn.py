#!/usr/bin/env python3
"""Train one lightly fine-tuned CNN comparator for the BRAR benchmark.

This is a pre-specified same-split comparator, not an architecture search.
Validation folds are used only for early stopping and model selection; held-out
test folds are evaluated once from the selected epoch.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from torch import nn
from torch.nn import functional as F
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

PREDICTION_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "eval_split",
    "model",
    "encoder",
    "image_width",
    "image_height",
    "selected_epoch",
    "selected_validation_macro_f1",
    "selected_validation_balanced_accuracy",
    "selected_validation_log_loss",
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
    "unfreeze_feature_blocks",
    "selected_epoch",
    "stopped_epoch",
    "epochs_run",
    "validation_macro_f1",
    "validation_balanced_accuracy",
    "validation_log_loss",
    "train_loss_at_selected_epoch",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "trainable_parameters",
    "total_parameters",
]


@dataclass(frozen=True)
class ImageRecord:
    file_name: str
    relative_image_path: str
    severity_level: str


class BrarFineTuneDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        records: list[ImageRecord],
        image_width: int,
        image_height: int,
        training: bool,
        augmentation_jitter: float,
    ) -> None:
        self.records = records
        self.image_width = image_width
        self.image_height = image_height
        self.training = training
        self.augmentation_jitter = augmentation_jitter

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        image = Image.open(ROOT / record.relative_image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = aspect_fit_pad(image, self.image_width, self.image_height)
        if self.training:
            image = augment_image(image, self.augmentation_jitter)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, IMAGE_NET_MEAN, IMAGE_NET_STD)
        label = torch.tensor(CLASS_TO_INDEX[record.severity_level], dtype=torch.long)
        return tensor, label, record.file_name


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


def augment_image(image: Image.Image, jitter: float) -> Image.Image:
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if jitter > 0:
        brightness = 1.0 + random.uniform(-jitter, jitter)
        contrast = 1.0 + random.uniform(-jitter, jitter)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def build_model(unfreeze_feature_blocks: int, dropout: float) -> tuple[nn.Module, str]:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, len(CLASSES)))

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    if unfreeze_feature_blocks:
        for module in list(model.features.children())[-unfreeze_feature_blocks:]:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return model, str(weights)


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def class_weight_tensor(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    weights = len(labels) / (len(CLASSES) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(
    records: list[ImageRecord],
    image_width: int,
    image_height: int,
    training: bool,
    augmentation_jitter: float,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = BrarFineTuneDataset(
        records,
        image_width=image_width,
        image_height=image_height,
        training=training,
        augmentation_jitter=augmentation_jitter,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for images, labels, _names in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_n = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_n
        total_n += batch_n
    return total_loss / max(total_n, 1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    names_out: list[str] = []
    with torch.inference_mode():
        for images, labels, names in loader:
            logits = model(images.to(device))
            logits_out.append(logits.detach().cpu().numpy().astype(np.float64))
            labels_out.append(labels.detach().cpu().numpy().astype(np.int64))
            names_out.extend(str(name) for name in names)
    return np.concatenate(logits_out, axis=0), np.concatenate(labels_out, axis=0), names_out


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def validation_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = softmax_np(logits)
    pred = np.argmax(probs, axis=1)
    return {
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "log_loss": float(log_loss(labels, probs, labels=[0, 1, 2])),
    }


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def append_prediction_rows(
    out: list[dict[str, object]],
    repeat: str,
    seed: str,
    fold: str,
    eval_split: str,
    image_width: int,
    image_height: int,
    selected_epoch: int,
    selected_metrics: dict[str, float],
    file_names: list[str],
    y_true: np.ndarray,
    logits: np.ndarray,
) -> None:
    probs = softmax_np(logits)
    pred = np.argmax(probs, axis=1)
    for idx, file_name in enumerate(file_names):
        out.append(
            {
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                "eval_split": eval_split,
                "model": "lightly_finetuned_cnn",
                "encoder": "efficientnet_b0",
                "image_width": image_width,
                "image_height": image_height,
                "selected_epoch": selected_epoch,
                "selected_validation_macro_f1": f"{selected_metrics['macro_f1']:.8f}",
                "selected_validation_balanced_accuracy": f"{selected_metrics['balanced_accuracy']:.8f}",
                "selected_validation_log_loss": f"{selected_metrics['log_loss']:.8f}",
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


def train_fold(
    repeat: str,
    seed: str,
    fold: str,
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    test_records: list[ImageRecord],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    fold_seed = args.seed + int(repeat) * 100 + int(fold)
    set_seeds(fold_seed)
    model, weights_name = build_model(args.unfreeze_feature_blocks, args.dropout)
    model.to(device)
    trainable, total = parameter_counts(model)
    y_train = np.asarray([CLASS_TO_INDEX[record.severity_level] for record in train_records], dtype=np.int64)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(y_train, device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = make_loader(
        train_records,
        args.image_width,
        args.image_height,
        training=True,
        augmentation_jitter=args.augmentation_jitter,
        batch_size=args.batch_size,
        seed=fold_seed,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_records,
        args.image_width,
        args.image_height,
        training=False,
        augmentation_jitter=0.0,
        batch_size=args.batch_size,
        seed=fold_seed,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_records,
        args.image_width,
        args.image_height,
        training=False,
        augmentation_jitter=0.0,
        batch_size=args.batch_size,
        seed=fold_seed,
        num_workers=args.num_workers,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_ranking: tuple[float, float, float, int] | None = None
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    best_train_loss = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_logits, val_labels, _val_names = predict(model, val_loader, device)
        metrics = validation_metrics(val_logits, val_labels)
        ranking = (metrics["macro_f1"], metrics["balanced_accuracy"], -metrics["log_loss"], -epoch)
        improved = best_ranking is None or ranking > best_ranking
        if improved:
            best_state = state_dict_cpu(model)
            best_ranking = ranking
            best_metrics = metrics
            best_epoch = epoch
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            "fold"
            f" repeat={repeat} fold={fold} epoch={epoch}"
            f" train_loss={train_loss:.4f}"
            f" val_macro_f1={metrics['macro_f1']:.4f}"
            f" val_bal_acc={metrics['balanced_accuracy']:.4f}"
            f" val_log_loss={metrics['log_loss']:.4f}"
        )
        if (
            args.patience > 0
            and epoch >= args.min_epochs
            and epochs_without_improvement >= args.patience
        ):
            break

    if best_state is None:
        raise RuntimeError(f"no selected state for repeat={repeat} fold={fold}")
    model.load_state_dict(best_state)
    val_logits, val_labels, val_names = predict(model, val_loader, device)
    test_logits, test_labels, test_names = predict(model, test_loader, device)

    prediction_rows: list[dict[str, object]] = []
    append_prediction_rows(
        prediction_rows,
        repeat,
        seed,
        fold,
        "val",
        args.image_width,
        args.image_height,
        best_epoch,
        best_metrics,
        val_names,
        val_labels,
        val_logits,
    )
    append_prediction_rows(
        prediction_rows,
        repeat,
        seed,
        fold,
        "test",
        args.image_width,
        args.image_height,
        best_epoch,
        best_metrics,
        test_names,
        test_labels,
        test_logits,
    )
    selection_row = {
        "repeat": repeat,
        "seed": seed,
        "fold": fold,
        "encoder": "efficientnet_b0",
        "image_width": args.image_width,
        "image_height": args.image_height,
        "unfreeze_feature_blocks": args.unfreeze_feature_blocks,
        "selected_epoch": best_epoch,
        "stopped_epoch": epoch,
        "epochs_run": epoch,
        "validation_macro_f1": f"{best_metrics['macro_f1']:.8f}",
        "validation_balanced_accuracy": f"{best_metrics['balanced_accuracy']:.8f}",
        "validation_log_loss": f"{best_metrics['log_loss']:.8f}",
        "train_loss_at_selected_epoch": f"{best_train_loss:.8f}",
        "learning_rate": f"{args.learning_rate:.8f}",
        "weight_decay": f"{args.weight_decay:.8f}",
        "batch_size": args.batch_size,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }
    if args.save_checkpoints:
        checkpoint_dir = OUTPUT_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{args.output_stem}_repeat{repeat}_fold{fold}.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "weights": weights_name,
                "selection": selection_row,
                "classes": CLASSES,
            },
            checkpoint_path,
        )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction_rows, selection_row


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--output-stem", default="fine_tuned_efficientnet_b0_384x192")
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unfreeze-feature-blocks", type=int, default=2)
    parser.add_argument("--augmentation-jitter", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--repeats", default="all")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = selected_device(args.device)
    records = manifest_records(args.manifest)
    records_by_name = {record.file_name: record for record in records}
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
    if not combos:
        raise RuntimeError("No split folds selected.")

    prediction_path = OUTPUT_DIR / f"{args.output_stem}_predictions.csv"
    selection_path = OUTPUT_DIR / f"{args.output_stem}_model_selection.csv"
    run_info_path = OUTPUT_DIR / f"{args.output_stem}_run_info.json"
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    started_at = time.time()
    print(
        "starting lightly fine-tuned EfficientNet-B0"
        f" folds={len(combos)} device={device}"
        f" image={args.image_width}x{args.image_height}"
        f" epochs={args.epochs} patience={args.patience}"
    )
    for repeat, seed, fold in combos:
        train_names = rows_for(split_rows, repeat, fold, "train")
        val_names = rows_for(split_rows, repeat, fold, "val")
        test_names = rows_for(split_rows, repeat, fold, "test")
        train_records = [records_by_name[name] for name in train_names]
        val_records = [records_by_name[name] for name in val_names]
        test_records = [records_by_name[name] for name in test_names]
        fold_predictions, selection = train_fold(
            repeat,
            seed,
            fold,
            train_records,
            val_records,
            test_records,
            args,
            device,
        )
        prediction_rows.extend(fold_predictions)
        selection_rows.append(selection)
        write_csv(prediction_path, prediction_rows, PREDICTION_FIELDS)
        write_csv(selection_path, selection_rows, SELECTION_FIELDS)
        print(
            "selected"
            f" repeat={repeat} fold={fold}"
            f" epoch={selection['selected_epoch']}"
            f" val_macro_f1={selection['validation_macro_f1']}"
        )

    run_info = {
        "model": "lightly_finetuned_cnn",
        "encoder": "efficientnet_b0",
        "weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "image_width": args.image_width,
        "image_height": args.image_height,
        "device": str(device),
        "folds_run": len(combos),
        "prediction_rows": len(prediction_rows),
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "unfreeze_feature_blocks": args.unfreeze_feature_blocks,
        "augmentation_jitter": args.augmentation_jitter,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "design_rationale": (
            "Single lightly fine-tuned CNN comparator using the same repeated "
            "train/validation/test splits as the frozen-feature benchmark. "
            "Classifier and final EfficientNet feature blocks are trainable; "
            "validation macro-F1 selects the epoch before held-out test evaluation."
        ),
    }
    run_info_path.write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"predictions: {prediction_path}")
    print(f"selection: {selection_path}")
    print(f"run_info: {run_info_path}")


if __name__ == "__main__":
    main()
