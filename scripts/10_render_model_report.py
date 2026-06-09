#!/usr/bin/env python3
"""Render an HTML report for the frozen image baseline."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREDICTIONS = ROOT / "data" / "processed" / "image_baseline" / "frozen_efficientnet_b0_384x192_predictions.csv"
IMAGE_SUMMARY = REPORTS / "image_baseline_summary.json"
IMAGE_METRICS = REPORTS / "image_baseline_metrics.csv"
IMAGE_CONFUSIONS = REPORTS / "image_baseline_confusion_matrices.csv"
IMAGE_RELIABILITY = REPORTS / "image_baseline_reliability_bins.csv"
NEGATIVE_SUMMARY = REPORTS / "negative_control_summary.json"
REPORT = REPORTS / "image_baseline_report.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def image_test_rows(image_summary: list[dict[str, object]]) -> list[list[object]]:
    rows = [row for row in image_summary if row["eval_split"] == "test"]
    rows = sorted(rows, key=lambda row: row["probability_mode"])
    return [
        [
            row["probability_mode"],
            f"{float(row['macro_f1_mean']):.4f}",
            f"{float(row['balanced_accuracy_mean']):.4f}",
            f"{float(row['accuracy_mean']):.4f}",
            f"{float(row['log_loss_mean']):.4f}",
            f"{float(row['ece_10_mean']):.4f}",
            f"{float(row['ovr_auroc_mean']):.4f}",
            row["folds"],
        ]
        for row in rows
    ]


def comparison_rows(image_summary: list[dict[str, object]], negative_summary: list[dict[str, object]]) -> list[list[object]]:
    rows: list[tuple[str, str, float, float, float]] = []
    for row in image_summary:
        if row["eval_split"] == "test":
            rows.append(
                (
                    "image",
                    str(row["probability_mode"]),
                    float(row["macro_f1_mean"]),
                    float(row["balanced_accuracy_mean"]),
                    float(row["accuracy_mean"]),
                )
            )
    for row in negative_summary:
        if row["eval_split"] == "test":
            rows.append(
                (
                    "negative_control",
                    f"{row['model']} / {row['feature_set']}",
                    float(row["macro_f1_mean"]),
                    float(row["balanced_accuracy_mean"]),
                    float(row["accuracy_mean"]),
                )
            )
    rows = sorted(rows, key=lambda item: item[2], reverse=True)
    return [[kind, name, f"{macro:.4f}", f"{balanced:.4f}", f"{accuracy:.4f}"] for kind, name, macro, balanced, accuracy in rows]


def aggregate_confusion_rows(confusions: list[dict[str, str]], mode: str) -> list[list[object]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in confusions:
        if row["eval_split"] == "test" and row["probability_mode"] == mode:
            counter[(row["true_label"], row["pred_label"])] += int(row["count"])
    return [
        [
            true_label,
            counter[(true_label, "1")],
            counter[(true_label, "2")],
            counter[(true_label, "3")],
        ]
        for true_label in ["1", "2", "3"]
    ]


def reliability_rows(reliability: list[dict[str, str]], mode: str) -> list[list[object]]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "confidence": 0.0, "accuracy": 0.0})
    for row in reliability:
        if row["eval_split"] != "test" or row["probability_mode"] != mode:
            continue
        n = int(row["n"])
        item = grouped[row["bin"]]
        item["n"] += n
        item["confidence"] += n * float(row["confidence_mean"])
        item["accuracy"] += n * float(row["accuracy"])
    out = []
    for bin_name in sorted(grouped):
        item = grouped[bin_name]
        n = item["n"]
        out.append(
            [
                bin_name,
                int(n),
                f"{item['confidence'] / n:.4f}" if n else "0.0000",
                f"{item['accuracy'] / n:.4f}" if n else "0.0000",
            ]
        )
    return out


def example_rows(predictions: list[dict[str, str]], correct: bool, limit: int = 12) -> str:
    test_rows = [
        row
        for row in predictions
        if row["eval_split"] == "test"
        and row["repeat"] == "1"
        and row["fold"] == "0"
        and ((row["y_true"] == row["y_pred"]) == correct)
    ]
    test_rows = sorted(test_rows, key=lambda row: max(float(row[f"prob_{label}"]) for label in ["1", "2", "3"]), reverse=True)
    figures = []
    for row in test_rows[:limit]:
        level = row["y_true"]
        image_path = f"../data/extracted/level_{level}/{row['file_name']}"
        confidence = max(float(row[f"prob_{label}"]) for label in ["1", "2", "3"])
        figures.append(
            f"""
            <figure>
              <img src="{esc(image_path)}" alt="BRAR image example">
              <figcaption>{esc(row['file_name'])}<br>true {esc(row['y_true'])} | pred {esc(row['y_pred'])} | conf {confidence:.3f}</figcaption>
            </figure>
            """
        )
    return "".join(figures)


def render(
    image_summary: list[dict[str, object]],
    image_metrics: list[dict[str, str]],
    confusions: list[dict[str, str]],
    reliability: list[dict[str, str]],
    negative_summary: list[dict[str, object]],
    predictions: list[dict[str, str]],
) -> str:
    test_raw = next(row for row in image_summary if row["eval_split"] == "test" and row["probability_mode"] == "raw")
    age_sex = next(
        row
        for row in negative_summary
        if row["eval_split"] == "test" and row["model"] == "multinomial_logistic" and row["feature_set"] == "age_sex"
    )
    downstream = next(
        row
        for row in negative_summary
        if row["eval_split"] == "test"
        and row["model"] == "multinomial_logistic"
        and row["feature_set"] == "downstream_plus_age_sex"
    )
    delta_age = float(test_raw["macro_f1_mean"]) - float(age_sex["macro_f1_mean"])
    delta_downstream = float(test_raw["macro_f1_mean"]) - float(downstream["macro_f1_mean"])

    cards = [
        ("Image raw macro-F1", f"{float(test_raw['macro_f1_mean']):.4f}"),
        ("Delta vs age/sex", f"{delta_age:+.4f}"),
        ("Delta vs upper bound", f"{delta_downstream:+.4f}"),
        ("Folds evaluated", test_raw["folds"]),
        ("Encoder", test_raw["encoder"]),
        ("Primary status", "beats age/sex" if delta_age > 0 else "does not beat age/sex"),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BRAR Frozen Image Baseline Report</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #657484;
      --line: #d8dee6;
      --panel: #f7f9fb;
      --warn: #fff8ea;
    }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: white; line-height: 1.45; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; line-height: 1.1; letter-spacing: 0; }}
    h2 {{ margin-top: 34px; padding-bottom: 8px; border-bottom: 1px solid var(--line); font-size: 22px; }}
    p {{ color: var(--muted); margin: 0 0 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 20px 0; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: var(--panel); }}
    .card strong {{ display: block; font-size: 24px; line-height: 1.1; }}
    .card span {{ display: block; margin-top: 6px; color: var(--muted); font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: var(--panel); }}
    .warning {{ border: 1px solid #e4b35f; background: var(--warn); border-radius: 8px; padding: 14px 18px; }}
    .thumb-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--panel); }}
    img {{ width: 100%; aspect-ratio: 2 / 1; object-fit: contain; display: block; background: #101820; }}
    figcaption {{ padding: 8px 10px 10px; font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }}
    code {{ background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>BRAR Frozen Image Baseline Report</h1>
  <p>Image-only frozen encoder baseline compared against non-image guardrails.</p>
  <div class="cards">{''.join(f'<div class="card"><strong>{esc(value)}</strong><span>{esc(label)}</span></div>' for label, value in cards)}</div>

  <h2>Decision Gate</h2>
  <p class="warning">The image-only model must beat the age/sex baseline to support a compelling image-signal claim. Raw image macro-F1 delta versus age/sex: <strong>{delta_age:+.4f}</strong>.</p>

  <h2>Image Model Test Metrics</h2>
  {table(["Probability mode", "Macro-F1", "Balanced accuracy", "Accuracy", "Log loss", "ECE", "AUROC", "Folds"], image_test_rows(image_summary))}

  <h2>Comparison Against Guardrails</h2>
  {table(["Kind", "Model / feature set", "Macro-F1", "Balanced accuracy", "Accuracy"], comparison_rows(image_summary, negative_summary))}

  <h2>Aggregate Test Confusion Matrix: Raw</h2>
  {table(["True label", "Pred 1", "Pred 2", "Pred 3"], aggregate_confusion_rows(confusions, "raw"))}

  <h2>Aggregate Test Confusion Matrix: Temperature Scaled</h2>
  {table(["True label", "Pred 1", "Pred 2", "Pred 3"], aggregate_confusion_rows(confusions, "temperature_scaled"))}

  <h2>Reliability Bins: Raw Test Probabilities</h2>
  {table(["Confidence bin", "N", "Mean confidence", "Accuracy"], reliability_rows(reliability, "raw"))}

  <h2>High-Confidence Correct Examples: Repeat 1 Fold 0</h2>
  <div class="thumb-grid">{example_rows(predictions, correct=True)}</div>

  <h2>High-Confidence Incorrect Examples: Repeat 1 Fold 0</h2>
  <div class="thumb-grid">{example_rows(predictions, correct=False)}</div>
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--image-summary", type=Path, default=IMAGE_SUMMARY)
    parser.add_argument("--image-metrics", type=Path, default=IMAGE_METRICS)
    parser.add_argument("--image-confusions", type=Path, default=IMAGE_CONFUSIONS)
    parser.add_argument("--image-reliability", type=Path, default=IMAGE_RELIABILITY)
    parser.add_argument("--negative-summary", type=Path, default=NEGATIVE_SUMMARY)
    parser.add_argument("--report", type=Path, default=REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_summary = read_json(args.image_summary)
    negative_summary = read_json(args.negative_summary)
    html_text = render(
        image_summary=image_summary,
        image_metrics=read_csv(args.image_metrics),
        confusions=read_csv(args.image_confusions),
        reliability=read_csv(args.image_reliability),
        negative_summary=negative_summary,
        predictions=read_csv(args.predictions),
    )
    args.report.write_text(html_text, encoding="utf-8")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
