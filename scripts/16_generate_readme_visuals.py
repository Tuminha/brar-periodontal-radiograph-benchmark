#!/usr/bin/env python3
"""Generate public-safe README visuals from published benchmark outputs."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "docs" / "assets"

NAVY = "#26364D"
BLUE = "#2F80A0"
GREEN = "#2F855A"
AMBER = "#C7862C"
GRAY = "#64748B"
LIGHT = "#F5F7FA"
LINE = "#334155"


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    width = 2.45
    height = 0.95
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=color,
        facecolor=LIGHT,
    )
    ax.add_patch(patch)
    lines = text.split("\n", 1)
    ax.text(x + width / 2, y + 0.60, lines[0], ha="center", va="center", fontsize=10.5, weight="bold", color="#111827")
    if len(lines) > 1:
        ax.text(
            x + width / 2,
            y + 0.34,
            textwrap.fill(lines[1], width=30),
            ha="center",
            va="center",
            fontsize=8.3,
            color="#1F2937",
        )


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color=LINE,
            shrinkA=5,
            shrinkB=5,
        )
    )


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.text(
        4.6,
        3.92,
        "BRAR benchmark workflow",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color="#111827",
    )

    positions = [(0.35, 2.55), (3.35, 2.55), (6.35, 2.55), (0.35, 0.85), (3.35, 0.85), (6.35, 0.85)]
    labels = [
        "Public BRAR release\n988 linked panoramic radiographs",
        "Linkage + leakage audit\nchecksum pass, outcome fields excluded",
        "Fixed repeated splits\n3 repeats x 5 folds, validation kept separate",
        "Image comparators\nfrozen encoders plus one light fine-tune",
        "Validation-only calibration\ntemperature scaling and severe thresholds",
        "Held-out analysis\nimage-level bootstrap intervals and paired deltas",
    ]
    colors = [BLUE, BLUE, NAVY, AMBER, AMBER, GREEN]
    for pos, label, color in zip(positions, labels, colors, strict=True):
        box(ax, pos, label, color)

    arrow(ax, (2.80, 3.02), (3.35, 3.02))
    arrow(ax, (5.80, 3.02), (6.35, 3.02))
    arrow(ax, (7.58, 2.55), (1.58, 1.80))
    arrow(ax, (2.80, 1.32), (3.35, 1.32))
    arrow(ax, (5.80, 1.32), (6.35, 1.32))

    ax.text(
        4.6,
        0.18,
        "Raw radiographs and contact sheets are not included in this repository.",
        ha="center",
        va="center",
        fontsize=9,
        color=GRAY,
    )
    save(fig, "brar_benchmark_workflow")


def model_intervals() -> None:
    table = pd.read_csv(REPORTS / "publication_ready_model_table.csv")
    model_ids = [
        "image_tile_efficientnet_b0_meanmax",
        "image_finetuned_efficientnet_b0",
        "image_efficientnet_b0",
        "image_resnet50",
        "age_sex",
        "image_plus_age_sex",
        "majority_class",
    ]
    frame = table[table["model_id"].isin(model_ids)].copy()
    frame["order"] = frame["model_id"].map({model_id: idx for idx, model_id in enumerate(model_ids)})
    frame = frame.sort_values("order", ascending=False)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = range(len(frame))
    x = frame["oof_macro_f1"].to_numpy()
    xerr = [
        x - frame["oof_macro_f1_low"].to_numpy(),
        frame["oof_macro_f1_high"].to_numpy() - x,
    ]
    colors = [
        GREEN
        if mid == "image_tile_efficientnet_b0_meanmax"
        else AMBER
        if mid == "image_finetuned_efficientnet_b0"
        else BLUE
        if kind == "image_only"
        else GRAY
        for mid, kind in zip(frame["model_id"], frame["kind"], strict=True)
    ]
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#475569", elinewidth=1.7, capsize=4, zorder=1)
    ax.scatter(x, y, s=58, color=colors, zorder=2)
    ax.set_yticks(list(y), frame["model_label"])
    ax.set_xlabel("Image-level out-of-fold macro-F1 with 95% bootstrap interval")
    ax.set_title("Three-class BRAR severity benchmark", fontsize=13, weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_xlim(0.20, 0.61)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "model_macro_f1_intervals")


def paired_deltas() -> None:
    table = pd.read_csv(REPORTS / "publication_paired_interval_table.csv")
    metrics = ["macro_f1", "balanced_accuracy", "quadratic_weighted_kappa", "severe_auroc"]
    labels = {
        "macro_f1": "Macro-F1",
        "balanced_accuracy": "Balanced accuracy",
        "quadratic_weighted_kappa": "QWK",
        "severe_auroc": "Severe AUROC",
    }
    comparators = ["Whole-image EfficientNet-B0", "Age/sex guardrail", "Whole-image ResNet50", "Image plus age/sex"]
    frame = table[
        table["metric"].isin(metrics)
        & table["left_model"].eq("image_tile_efficientnet_b0_meanmax")
        & table["right_label"].isin(comparators)
    ].copy()
    frame["metric_order"] = frame["metric"].map({metric: idx for idx, metric in enumerate(metrics)})
    frame["comparison_order"] = frame["right_label"].map({label: idx for idx, label in enumerate(comparators)})
    frame = frame.sort_values(["comparison_order", "metric_order"], ascending=False)
    frame["label"] = frame["right_label"] + " - " + frame["metric"].map(labels)

    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    y = range(len(frame))
    x = frame["delta_left_minus_right"].to_numpy()
    xerr = [
        x - frame["delta_low"].to_numpy(),
        frame["delta_high"].to_numpy() - x,
    ]
    significant = frame["delta_low"].to_numpy() > 0
    colors = [GREEN if value else GRAY for value in significant]
    ax.axvline(0, color="#111827", linewidth=1.0)
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#475569", elinewidth=1.4, capsize=3, zorder=1)
    ax.scatter(x, y, s=48, color=colors, zorder=2)
    ax.set_yticks(list(y), frame["label"])
    ax.set_xlabel("Paired image-level delta: Tile EfficientNet-B0 minus comparator")
    ax.set_title("Where the tile model gains are clearest", fontsize=13, weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_xlim(-0.06, 0.23)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "paired_tile_deltas")


def main() -> None:
    workflow()
    model_intervals()
    paired_deltas()
    for path in sorted(OUT.glob("*")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
