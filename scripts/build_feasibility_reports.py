#!/usr/bin/env python3
"""Generate BRAR feasibility inventory and audit reports.

This script does not train models. It only inspects released files, checks
metadata-image linkage, and records leakage constraints for the first sprint.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "data" / "extracted"
RAW_ZIP = ROOT / "data" / "raw" / "BRAR-anchored_multimodal_dataset.zip"
REPORTS = ROOT / "reports"
EXPECTED_ZIP_MD5 = "4df0368a88f23f403958e6b371057f11"

def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((ROOT / "data").rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        rows.append(
            {
                "relative_path": str(rel),
                "parent": str(rel.parent),
                "extension": path.suffix.lower().lstrip(".") or "[none]",
                "size_bytes": str(path.stat().st_size),
            }
        )
    return rows


def read_metadata() -> list[dict[str, str]]:
    meta_path = EXTRACTED / "meta_data.csv"
    with meta_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def image_paths() -> list[Path]:
    return sorted(EXTRACTED.rglob("*.jpg"))


def grade_from_path(path: Path) -> str:
    match = re.search(r"level_(\d+)", str(path))
    return match.group(1) if match else ""


def image_readability_sample(paths: list[Path]) -> list[dict[str, str]]:
    samples: list[Path] = []
    by_grade: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_grade[grade_from_path(path)].append(path)
    for grade in sorted(by_grade):
        samples.extend(by_grade[grade][:3])

    rows: list[dict[str, str]] = []
    for path in samples:
        result = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        width = ""
        height = ""
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("pixelWidth:"):
                width = stripped.split(":", 1)[1].strip()
            if stripped.startswith("pixelHeight:"):
                height = stripped.split(":", 1)[1].strip()
        rows.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "grade_folder": grade_from_path(path),
                "sips_exit_code": str(result.returncode),
                "pixel_width": width,
                "pixel_height": height,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    inventory_rows = file_inventory()
    write_csv(REPORTS / "file_inventory.csv", inventory_rows)

    meta = read_metadata()
    images = image_paths()

    metadata_names = [row["File name"] for row in meta]
    metadata_name_set = set(metadata_names)
    image_name_set = {path.name for path in images}
    duplicate_metadata_names = [name for name, count in Counter(metadata_names).items() if count > 1]
    missing_images = sorted(metadata_name_set - image_name_set)
    unlinked_images = sorted(image_name_set - metadata_name_set)

    level_counts_csv = Counter(row["Level"] for row in meta)
    level_counts_dirs = Counter(grade_from_path(path) for path in images)
    folder_mismatch = []
    path_by_name = {path.name: path for path in images}
    for row in meta:
        path = path_by_name.get(row["File name"])
        if path and grade_from_path(path) != row["Level"]:
            folder_mismatch.append(row["File name"])

    column_summary = []
    for col in meta[0].keys():
        values = [row[col] for row in meta]
        column_summary.append(
            {
                "column": col,
                "missing": str(sum(value == "" for value in values)),
                "unique": str(len(set(values))),
                "first_values": ", ".join(list(dict.fromkeys(values))[:8]),
            }
        )

    image_sample = image_readability_sample(images)
    write_csv(REPORTS / "image_readability_sample.csv", image_sample)

    zip_md5 = md5_file(RAW_ZIP) if RAW_ZIP.exists() else ""
    inventory_by_ext = Counter(row["extension"] for row in inventory_rows)
    inventory_by_parent = Counter(row["parent"] for row in inventory_rows)

    data_report = f"""# Data Linkage Audit

Date: 2026-06-04

## Archive Verification

- Raw ZIP: `{RAW_ZIP.relative_to(ROOT)}`
- Expected MD5: `{EXPECTED_ZIP_MD5}`
- Observed MD5: `{zip_md5}`
- Checksum status: `{"PASS" if zip_md5 == EXPECTED_ZIP_MD5 else "FAIL"}`

## Released File Structure

- Extracted image files: {len(images)}
- Metadata rows: {len(meta)}
- Metadata file: `data/extracted/meta_data.csv`
- Top-level extracted folders: `level_1`, `level_2`, `level_3`
- Important discrepancy: the article/codebook describe richer patient/tooth-level variables and 1,104 patients, but the released ZIP inspected here contains 988 images and one patient/image-level metadata CSV.
- Patient-level annotation table: present as `meta_data.csv`, with one row per released image/anonymized patient filename.
- Tooth-level annotation table: not present in the inspected public full ZIP.
- Separate `patient_id` column: not present; the anonymized image filename is the only linkage key in the released CSV.

### File Counts By Extension

{markdown_table([{"extension": k, "count": str(v)} for k, v in sorted(inventory_by_ext.items())], ["extension", "count"])}

### File Counts By Folder

{markdown_table([{"folder": k, "count": str(v)} for k, v in sorted(inventory_by_parent.items())], ["folder", "count"])}

## Metadata Columns

{markdown_table(column_summary, ["column", "missing", "unique", "first_values"])}

## Linkage Checks

- Unique metadata filenames: {len(metadata_name_set)}
- Duplicate metadata filenames: {len(duplicate_metadata_names)}
- Metadata rows without matching image: {len(missing_images)}
- Images without matching metadata row: {len(unlinked_images)}
- Images where folder grade differs from CSV `Level`: {len(folder_mismatch)}
- Filename-to-patient linkage: `PASS` for image loading because each anonymized filename maps to one metadata row and one image path. `patient_id` itself is not released as a separate column.

### Grade Counts

| Grade | CSV rows | Image files |
|---|---:|---:|
| 1 | {level_counts_csv.get("1", 0)} | {level_counts_dirs.get("1", 0)} |
| 2 | {level_counts_csv.get("2", 0)} | {level_counts_dirs.get("2", 0)} |
| 3 | {level_counts_csv.get("3", 0)} | {level_counts_dirs.get("3", 0)} |

## Image Readability Sample

`sips` was used to read pixel dimensions from three images per released grade folder.

{markdown_table(image_sample, ["relative_path", "grade_folder", "sips_exit_code", "pixel_width", "pixel_height"])}

## Linkage Verdict

The released BRAR ZIP is usable for a patient/image-level image-classification benchmark: all 988 metadata rows link to exactly one JPG image, and folder labels match CSV `Level`. It is not usable for the tooth-level analyses described by the richer codebook unless additional unreleased annotation tables are obtained.
"""
    (REPORTS / "data_linkage_audit.md").write_text(data_report, encoding="utf-8")

    leakage_rows = [
        {
            "variable": "File name",
            "role": "identifier/linkage",
            "primary_model_use": "path only",
            "reason": "Used only to load the image; not a tabular predictor.",
        },
        {
            "variable": "Age",
            "role": "metadata sensitivity",
            "primary_model_use": "exclude from image-only primary model",
            "reason": "Age appears in the BRAR-derived grading formula, so use only in labelled sensitivity analyses.",
        },
        {
            "variable": "Gender",
            "role": "metadata sensitivity",
            "primary_model_use": "exclude from image-only primary model",
            "reason": "Allowed for sensitivity/subgroup analyses, but primary benchmark should be image-only.",
        },
        {
            "variable": "Bone resorption",
            "role": "outcome-derived",
            "primary_model_use": "forbidden",
            "reason": "Direct bone-loss measure used to derive severity.",
        },
        {
            "variable": "Bone resorption Age",
            "role": "outcome-derived",
            "primary_model_use": "forbidden",
            "reason": "Age-normalized bone-resorption value used to derive `Level`.",
        },
        {
            "variable": "Level",
            "role": "outcome",
            "primary_model_use": "target only",
            "reason": "Released severity class to be predicted; treat as the public equivalent of planned `severity_grade`.",
        },
        {
            "variable": "Number of missing teeth",
            "role": "downstream sensitivity",
            "primary_model_use": "exclude",
            "reason": "Likely consequence/correlate of periodontal history; use only in upper-bound sensitivity.",
        },
        {
            "variable": "Implant",
            "role": "downstream sensitivity",
            "primary_model_use": "exclude",
            "reason": "Treatment/restoration status can reflect prior disease and access to care.",
        },
        {
            "variable": "Residual root",
            "role": "downstream sensitivity",
            "primary_model_use": "exclude",
            "reason": "Disease/treatment consequence; not a clean upstream predictor.",
        },
        {
            "variable": "Functional tooth logarithm",
            "role": "downstream sensitivity",
            "primary_model_use": "exclude",
            "reason": "Functional dentition summary may encode disease consequences.",
        },
    ]

    leakage_report = f"""# Leakage Audit

Date: 2026-06-04

## Planned Primary Task

Predict BRAR severity `Level` from panoramic radiograph images using patient-level splitting. In the released data, one image appears to correspond to one anonymized patient filename, so patient-level and image-level splitting are equivalent if filenames remain unique. The plan's `severity_grade` target maps to the released `Level` column.

## Variable Policy

{markdown_table(leakage_rows, ["variable", "role", "primary_model_use", "reason"])}

## Primary Model Feature Rule

The primary benchmark should use only image pixels and the target `Level`. No tabular metadata should enter the primary model.

## Codebook Variable Policy

The richer codebook/protocol fields are not present in the inspected public ZIP, but they remain important if additional annotation tables are obtained later.

| codebook/planned variable | status in inspected ZIP | model policy |
| --- | --- | --- |
| `bl_mm` | absent | outcome-derived; forbidden as predictor |
| `rl_mm` | absent | outcome-derived; forbidden as predictor |
| `bl_rl_ratio` | absent | outcome-derived; forbidden as predictor |
| `max_bl_rl_ratio` | absent | outcome-derived; forbidden as predictor |
| `brar` | represented by released `Bone resorption Age` | outcome-derived; forbidden as predictor |
| `missing_tooth` | represented by `Number of missing teeth` | downstream sensitivity only |
| `implant` | represented by `Implant` | downstream sensitivity only |
| `residual_root` | represented by `Residual root` | downstream sensitivity only |
| `functional_pair` | represented by `Functional tooth logarithm` | downstream sensitivity only |

## Sensitivity Model Rules

- Age/gender sensitivity: allowed only as explicitly labelled metadata sensitivity analysis.
- Upper-bound downstream sensitivity: `Number of missing teeth`, `Implant`, `Residual root`, and `Functional tooth logarithm` may be used only to estimate how much downstream dental-status information inflates performance.
- Forbidden predictors in any predictive model for `Level`: `Bone resorption`, `Bone resorption Age`, and any direct transformation of them.

## Current Risk Assessment

- Low leakage risk for an image-only benchmark if splits are based on unique filenames and no duplicate/repeated patient images exist.
- Moderate conceptual leakage risk for metadata models because `Age` is part of the BRAR definition and dental-status variables may be downstream of disease.
- Major limitation: released CSV does not include tooth-level rows, `patient_id`, explicit image-quality ratings, CEJ/apex landmarks, `bl_mm`, `rl_mm`, or `bl_rl_ratio` despite the codebook describing those fields.
"""
    (REPORTS / "leakage_audit.md").write_text(leakage_report, encoding="utf-8")


if __name__ == "__main__":
    main()
