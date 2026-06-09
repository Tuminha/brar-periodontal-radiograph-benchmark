#!/usr/bin/env python3
"""Render a static HTML report for pre-model guardrail results."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PRE_MODEL_SUMMARY = REPORTS / "pre_model_audit_summary.json"
NEGATIVE_SUMMARY = REPORTS / "negative_control_summary.json"
NEAR_DUPLICATES = REPORTS / "near_duplicate_candidates.csv"
REPORT = REPORTS / "guardrail_report.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def negative_control_rows(summary: list[dict[str, object]]) -> list[list[object]]:
    test_rows = [row for row in summary if row["eval_split"] == "test"]
    test_rows = sorted(test_rows, key=lambda row: float(row["macro_f1_mean"]), reverse=True)
    return [
        [
            row["model"],
            row["feature_set"],
            f"{float(row['macro_f1_mean']):.4f}",
            f"{float(row['balanced_accuracy_mean']):.4f}",
            f"{float(row['accuracy_mean']):.4f}",
            f"{float(row['log_loss_mean']):.4f}",
            f"{float(row['ece_10_mean']):.4f}",
        ]
        for row in test_rows
    ]


def split_balance_rows(pre_model: dict[str, object]) -> list[list[object]]:
    split_balance = pre_model["split_balance"]  # type: ignore[index]
    rows = []
    for split in ["train", "val", "test"]:
        item = split_balance[split]  # type: ignore[index]
        rows.append(
            [
                split,
                f"{item['n_min']}-{item['n_max']}",
                f"{item['age_mean_min']:.3f}-{item['age_mean_max']:.3f}",
                f"{item['gender_1_prop_min']:.4f}-{item['gender_1_prop_max']:.4f}",
                f"{item['aspect_ratio_mean_min']:.4f}-{item['aspect_ratio_mean_max']:.4f}",
            ]
        )
    return rows


def render(pre_model: dict[str, object], negative_summary: list[dict[str, object]], near_rows: list[dict[str, str]]) -> str:
    near = pre_model["near_duplicates"]  # type: ignore[index]
    env = pre_model["environment"]  # type: ignore[index]
    priority_counts = Counter(row["review_priority"] for row in near_rows)

    cards = [
        ("Images fingerprinted", pre_model["images_fingerprinted"]),
        ("High/medium near duplicates", int(near["high_priority_pairs"]) + int(near["medium_priority_pairs"])),
        ("Fold-crossing risk rows", near["crossing_fold_group_rows"]),
        ("Low-priority hash candidates", priority_counts.get("low", 0)),
        ("Best non-image macro-F1", f"{max(float(row['macro_f1_mean']) for row in negative_summary if row['eval_split'] == 'test'):.4f}"),
        ("Torch available", "yes" if not str(env["torch"]).startswith("not available") else "no"),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BRAR Pre-Model Guardrail Report</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #657484;
      --line: #d8dee6;
      --panel: #f7f9fb;
      --warn: #fff8ea;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 32px; line-height: 1.1; letter-spacing: 0; }}
    h2 {{ margin-top: 34px; padding-bottom: 8px; border-bottom: 1px solid var(--line); font-size: 22px; }}
    p {{ color: var(--muted); margin: 0 0 16px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin: 20px 0;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--panel);
    }}
    .card strong {{ display: block; font-size: 24px; line-height: 1.1; }}
    .card span {{ display: block; margin-top: 6px; color: var(--muted); font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: var(--panel); }}
    .warning {{
      border: 1px solid #e4b35f;
      background: var(--warn);
      border-radius: 8px;
      padding: 14px 18px;
    }}
    code {{ background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>BRAR Pre-Model Guardrail Report</h1>
  <p>Negative controls and pre-model leakage checks before image training.</p>

  <div class="cards">
    {''.join(f'<div class="card"><strong>{esc(value)}</strong><span>{esc(label)}</span></div>' for label, value in cards)}
  </div>

  <h2>Near-Duplicate And Split Risk</h2>
  {table(["Check", "Value"], [
      ["Exact SHA-256 duplicate pairs", near["exact_sha256_pairs"]],
      ["High-priority near-duplicate pairs", near["high_priority_pairs"]],
      ["Medium-priority near-duplicate pairs", near["medium_priority_pairs"]],
      ["Low-priority hash-only candidates", priority_counts.get("low", 0)],
      ["Fold-crossing high/medium risk rows", near["crossing_fold_group_rows"]],
  ])}
  <p class="warning">The low-priority candidates are hash-only similarities common in panoramic radiographs. The stricter low-resolution correlation check found no high/medium fold-crossing risk.</p>

  <h2>Split Balance</h2>
  {table(["Split", "N range", "Mean age range", "Gender=1 proportion range", "Mean aspect-ratio range"], split_balance_rows(pre_model))}

  <h2>Negative-Control Test Metrics</h2>
  {table(["Model", "Feature set", "Macro-F1", "Balanced accuracy", "Accuracy", "Log loss", "ECE"], negative_control_rows(negative_summary))}

  <h2>Interpretation</h2>
  <p>The image model must beat <code>age_sex</code>, not only the majority-class baseline. Image geometry and administrative filename order are weak, which reduces concern that the future image model will simply learn scanner geometry or row ordering. Downstream dental-status models are upper-bound sensitivity checks only.</p>

  <h2>Runtime</h2>
  {table(["Package", "Version/status"], [[key, value] for key, value in env.items()])}
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-model-summary", type=Path, default=PRE_MODEL_SUMMARY)
    parser.add_argument("--negative-summary", type=Path, default=NEGATIVE_SUMMARY)
    parser.add_argument("--near-duplicates", type=Path, default=NEAR_DUPLICATES)
    parser.add_argument("--report", type=Path, default=REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pre_model = read_json(args.pre_model_summary)
    negative_summary = read_json(args.negative_summary)
    near_rows = read_csv(args.near_duplicates)
    args.report.write_text(render(pre_model, negative_summary, near_rows), encoding="utf-8")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
