#!/usr/bin/env python3
"""Render a static HTML report for the BRAR manifest and split manifests."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MANIFEST = PROCESSED / "brar_manifest.csv"
MANIFEST_SUMMARY = PROCESSED / "brar_manifest_summary.json"
SPLIT_SUMMARY = PROCESSED / "splits" / "brar_split_summary.csv"
SPLIT_SUMMARY_JSON = PROCESSED / "splits" / "brar_split_summary.json"
REPORT = REPORTS / "dataset_split_report.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def html_table(headers: list[str], rows: list[list[object]], css_class: str = "") -> str:
    class_attr = f' class="{css_class}"' if css_class else ""
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>")
    return f"<table{class_attr}><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def level_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(row["severity_level"] for row in rows)


def bar_rows(counter: Counter[str]) -> str:
    total = sum(counter.values())
    max_count = max(counter.values()) if counter else 1
    rows = []
    for level in ["1", "2", "3"]:
        count = counter.get(level, 0)
        pct = (count / total * 100) if total else 0
        width = (count / max_count * 100) if max_count else 0
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">Level {esc(level)}</div>
              <div class="bar-track"><div class="bar-fill level-{esc(level)}" style="width: {width:.1f}%"></div></div>
              <div class="bar-value">{count} ({pct:.1f}%)</div>
            </div>
            """
        )
    return "\n".join(rows)


def dimension_summary(rows: list[dict[str, str]]) -> list[list[object]]:
    widths = [int(row["pixel_width"]) for row in rows if row["pixel_width"]]
    heights = [int(row["pixel_height"]) for row in rows if row["pixel_height"]]
    ratios = [float(row["aspect_ratio"]) for row in rows if row["aspect_ratio"]]
    if not widths or not heights or not ratios:
        return [["readable images", 0, "", "", ""]]
    return [
        ["pixel width", len(widths), min(widths), round(statistics.mean(widths), 1), max(widths)],
        ["pixel height", len(heights), min(heights), round(statistics.mean(heights), 1), max(heights)],
        ["aspect ratio", len(ratios), round(min(ratios), 3), round(statistics.mean(ratios), 3), round(max(ratios), 3)],
    ]


def split_stats(summary_rows: list[dict[str, str]]) -> list[list[object]]:
    values_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in summary_rows:
        values_by_key[(row["split"], row["severity_level"])].append(int(row["count"]))

    output = []
    for split in ["train", "val", "test"]:
        for level in ["1", "2", "3"]:
            values = values_by_key[(split, level)]
            output.append(
                [
                    split,
                    level,
                    min(values),
                    round(statistics.mean(values), 1),
                    max(values),
                ]
            )
    return output


def representative_fold_rows(summary_rows: list[dict[str, str]]) -> list[list[object]]:
    rows = [
        row
        for row in summary_rows
        if row["repeat"] == "1" and row["fold"] == "0"
    ]
    ordered = sorted(rows, key=lambda row: (["train", "val", "test"].index(row["split"]), row["severity_level"]))
    return [[row["split"], row["severity_level"], row["count"]] for row in ordered]


def sample_thumbnails(rows: list[dict[str, str]], per_level: int = 4) -> str:
    rows_by_level: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["readability_status"] == "PASS":
            rows_by_level[row["severity_level"]].append(row)

    sections = []
    for level in ["1", "2", "3"]:
        cards = []
        for row in rows_by_level[level][:per_level]:
            image_src = "../" + row["relative_image_path"]
            cards.append(
                f"""
                <figure>
                  <img src="{esc(image_src)}" alt="BRAR Level {esc(level)} sample">
                  <figcaption>{esc(row["file_name"])}<br>Level {esc(level)} | {esc(row["pixel_width"])}x{esc(row["pixel_height"])}</figcaption>
                </figure>
                """
            )
        sections.append(
            f"""
            <section class="sample-section">
              <h3>Level {esc(level)} Samples</h3>
              <div class="thumb-grid">{''.join(cards)}</div>
            </section>
            """
        )
    return "\n".join(sections)


def warning_items(manifest_rows: list[dict[str, str]], manifest_summary: dict[str, object]) -> list[str]:
    warnings = []
    counts = level_counts(manifest_rows)
    min_count = min(counts.values()) if counts else 0
    max_count = max(counts.values()) if counts else 0
    imbalance_ratio = max_count / min_count if min_count else 0

    if manifest_summary.get("missing_images"):
        warnings.append("One or more metadata rows have no matching image.")
    if manifest_summary.get("unlinked_images"):
        warnings.append("One or more image files have no metadata row.")
    if manifest_summary.get("duplicate_exact_image_hashes"):
        warnings.append("Exact duplicate image hashes were found and must stay grouped across folds.")
    if imbalance_ratio >= 3:
        warnings.append(
            f"Class imbalance is material: largest class is {imbalance_ratio:.2f} times the smallest class."
        )
    warnings.append("No separate patient_id is released; filename is the grouping key.")
    warnings.append("No tooth-level annotation table is released in the inspected ZIP.")
    warnings.append("Outcome-derived variables must stay out of predictors for Level.")
    return warnings


def render_report(
    manifest_rows: list[dict[str, str]],
    manifest_summary: dict[str, object],
    split_summary_rows: list[dict[str, str]],
    split_summary: dict[str, object],
) -> str:
    counts = level_counts(manifest_rows)
    warnings = warning_items(manifest_rows, manifest_summary)
    duplicate_hash_count = len(manifest_summary.get("duplicate_exact_image_hashes", {}))

    warnings_html = "".join(f"<li>{esc(item)}</li>" for item in warnings)
    status_cards = [
        ["Manifest rows", len(manifest_rows)],
        ["Image files", manifest_summary.get("image_files", "")],
        ["Readable images", manifest_summary.get("readability_status_counts", {}).get("PASS", "")],  # type: ignore[union-attr]
        ["Exact duplicate hash groups", duplicate_hash_count],
        ["Split seeds", ", ".join(str(seed) for seed in split_summary.get("seeds", []))],
        ["Folds per seed", split_summary.get("n_folds", "")],
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BRAR Dataset And Split Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #657484;
      --line: #d8dee6;
      --panel: #f7f9fb;
      --level1: #2a9d8f;
      --level2: #5176b8;
      --level3: #d66a35;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 32px;
      line-height: 1.1;
    }}
    h2 {{
      margin-top: 34px;
      font-size: 22px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }}
    h3 {{
      margin-top: 20px;
      font-size: 16px;
    }}
    p {{
      color: var(--muted);
      margin: 0 0 16px;
    }}
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
    .card strong {{
      display: block;
      font-size: 24px;
      line-height: 1.1;
    }}
    .card span {{
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0 20px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: var(--panel);
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 80px minmax(160px, 1fr) 110px;
      gap: 10px;
      align-items: center;
      margin: 8px 0;
    }}
    .bar-label, .bar-value {{
      color: var(--muted);
      font-size: 13px;
    }}
    .bar-track {{
      height: 18px;
      border-radius: 4px;
      background: #eef2f6;
      overflow: hidden;
      border: 1px solid var(--line);
    }}
    .bar-fill {{
      height: 100%;
    }}
    .level-1 {{ background: var(--level1); }}
    .level-2 {{ background: var(--level2); }}
    .level-3 {{ background: var(--level3); }}
    .warnings {{
      border: 1px solid #e4b35f;
      background: #fff8ea;
      border-radius: 8px;
      padding: 14px 18px;
    }}
    .warnings li {{
      margin: 5px 0;
    }}
    .thumb-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: var(--panel);
    }}
    img {{
      width: 100%;
      aspect-ratio: 2 / 1;
      object-fit: contain;
      display: block;
      background: #101820;
    }}
    figcaption {{
      padding: 8px 10px 10px;
      font-size: 12px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    code {{
      background: #eef2f6;
      padding: 2px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
<main>
  <h1>BRAR Dataset And Split Report</h1>
  <p>Static checkpoint report generated before model training. It verifies that the public BRAR release can support a leakage-aware image-level benchmark.</p>

  <div class="cards">
    {''.join(f'<div class="card"><strong>{esc(value)}</strong><span>{esc(label)}</span></div>' for label, value in status_cards)}
  </div>

  <h2>Class Distribution</h2>
  {bar_rows(counts)}
  {html_table(["Level", "Count"], [[level, counts.get(level, 0)] for level in ["1", "2", "3"]])}

  <h2>Image Geometry</h2>
  {html_table(["Measure", "N", "Min", "Mean", "Max"], dimension_summary(manifest_rows))}

  <h2>Split Balance</h2>
  <p>Each seed creates 5 fold groups. For a given fold, that fold is test, the next fold is validation, and the remaining three folds are training.</p>
  {html_table(["Split", "Level", "Min count", "Mean count", "Max count"], split_stats(split_summary_rows))}

  <h3>Representative Fold: Repeat 1, Fold 0</h3>
  {html_table(["Split", "Level", "Count"], representative_fold_rows(split_summary_rows))}

  <h2>Review Warnings</h2>
  <ul class="warnings">{warnings_html}</ul>

  <h2>Sample Images</h2>
  {sample_thumbnails(manifest_rows)}

  <h2>Next Modeling Checkpoint</h2>
  <p>The first model should be a frozen pretrained image encoder plus a class-balanced linear classifier. The primary model should use only image pixels and <code>severity_level</code>. Metadata and downstream dental-status variables should remain sensitivity analyses only.</p>
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--manifest-summary", type=Path, default=MANIFEST_SUMMARY)
    parser.add_argument("--split-summary", type=Path, default=SPLIT_SUMMARY)
    parser.add_argument("--split-summary-json", type=Path, default=SPLIT_SUMMARY_JSON)
    parser.add_argument("--report", type=Path, default=REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv(args.manifest)
    manifest_summary = read_json(args.manifest_summary)
    split_summary_rows = read_csv(args.split_summary)
    split_summary = read_json(args.split_summary_json)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_report(manifest_rows, manifest_summary, split_summary_rows, split_summary),
        encoding="utf-8",
    )
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
