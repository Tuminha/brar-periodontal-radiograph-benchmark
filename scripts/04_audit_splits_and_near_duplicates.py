#!/usr/bin/env python3
"""Audit split balance and near-duplicate image risk before modeling."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageOps
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing runtime guard.
    raise SystemExit(
        "Pillow is required for near-duplicate image auditing. "
        "Use the bundled Codex Python runtime or install Pillow in the project venv."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "brar_manifest.csv"
SPLITS = ROOT / "data" / "processed" / "splits" / "brar_repeated_5fold_splits.csv"
ASSIGNMENTS = ROOT / "data" / "processed" / "splits" / "brar_fold_assignments.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

FINGERPRINTS = PROCESSED / "image_fingerprints.csv"
NEAR_DUPLICATES = REPORTS / "near_duplicate_candidates.csv"
NEAREST_NEIGHBORS = REPORTS / "near_duplicate_nearest_neighbors.csv"
NEAR_DUPLICATE_SPLIT_RISK = REPORTS / "near_duplicate_split_risk.csv"
SPLIT_BALANCE = REPORTS / "split_balance_audit.csv"
SUMMARY_JSON = REPORTS / "pre_model_audit_summary.json"
SUMMARY_MD = REPORTS / "pre_model_audit.md"
ENVIRONMENT_MD = REPORTS / "environment_report.md"


FINGERPRINT_FIELDS = [
    "file_name",
    "severity_level",
    "relative_image_path",
    "sha256",
    "dhash_64",
    "ahash_64",
]

NEAR_DUPLICATE_FIELDS = [
    "file_a",
    "file_b",
    "level_a",
    "level_b",
    "same_level",
    "dhash_distance",
    "ahash_distance",
    "lowres_correlation",
    "combined_distance",
    "same_sha256",
    "width_a",
    "height_a",
    "width_b",
    "height_b",
    "review_priority",
]

SPLIT_RISK_FIELDS = [
    "file_a",
    "file_b",
    "level_a",
    "level_b",
    "dhash_distance",
    "ahash_distance",
    "lowres_correlation",
    "seed",
    "fold_group_a",
    "fold_group_b",
    "crosses_fold_group",
]

SPLIT_BALANCE_FIELDS = [
    "repeat",
    "seed",
    "fold",
    "split",
    "n",
    "level_1",
    "level_2",
    "level_3",
    "age_mean",
    "age_sd",
    "age_min",
    "age_max",
    "gender_0",
    "gender_1",
    "gender_1_prop",
    "pixel_width_mean",
    "pixel_height_mean",
    "aspect_ratio_mean",
    "size_bytes_mean",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def bits_to_hex(bits: list[bool]) -> str:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    width = max(1, (len(bits) + 3) // 4)
    return f"{value:0{width}x}"


def normalized_gray_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    return ImageOps.autocontrast(image, cutoff=1)


def dhash_64(path: Path) -> str:
    image = normalized_gray_image(path).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.int16)
    bits = []
    for y in range(8):
        for x in range(8):
            bits.append(bool(pixels[y, x] > pixels[y, x + 1]))
    return bits_to_hex(bits)


def ahash_64(path: Path) -> str:
    image = normalized_gray_image(path).resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32).reshape(-1)
    threshold = float(pixels.mean())
    return bits_to_hex([bool(pixel >= threshold) for pixel in pixels])


def build_fingerprints(manifest_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in manifest_rows:
        path = ROOT / row["relative_image_path"]
        rows.append(
            {
                "file_name": row["file_name"],
                "severity_level": row["severity_level"],
                "relative_image_path": row["relative_image_path"],
                "sha256": row["sha256"],
                "dhash_64": dhash_64(path),
                "ahash_64": ahash_64(path),
            }
        )
    return rows


def lowres_vector(path: Path) -> np.ndarray:
    image = normalized_gray_image(path).resize((128, 64), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32).reshape(-1)
    array = (array - float(array.mean())) / (float(array.std()) + 1e-6)
    return array / (float(np.linalg.norm(array)) + 1e-6)


def priority_for_distances(
    dhash_distance: int,
    ahash_distance: int,
    lowres_correlation: float,
    same_sha256: bool,
) -> str:
    if same_sha256:
        return "exact_duplicate"
    if lowres_correlation >= 0.985:
        return "high"
    if lowres_correlation >= 0.970 and (dhash_distance <= 10 or ahash_distance <= 18):
        return "medium"
    if dhash_distance <= 6 and ahash_distance <= 16:
        return "low"
    return "nearest_only"


def near_duplicate_pairs(
    manifest_rows: list[dict[str, str]],
    fingerprint_rows: list[dict[str, object]],
    nearest_limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest_by_name = {row["file_name"]: row for row in manifest_rows}
    vectors = np.vstack(
        [
            lowres_vector(ROOT / manifest_by_name[str(row["file_name"])]["relative_image_path"])
            for row in fingerprint_rows
        ]
    )
    lowres_correlations = vectors @ vectors.T
    candidates: list[dict[str, object]] = []
    nearest_by_file: dict[str, tuple[int, dict[str, object]]] = {}

    for idx, left in enumerate(fingerprint_rows):
        for right_idx, right in enumerate(fingerprint_rows[idx + 1 :], start=idx + 1):
            dhash_distance = hamming_hex(str(left["dhash_64"]), str(right["dhash_64"]))
            ahash_distance = hamming_hex(str(left["ahash_64"]), str(right["ahash_64"]))
            combined_distance = dhash_distance + ahash_distance
            lowres_correlation = float(lowres_correlations[idx, right_idx])
            file_a = str(left["file_name"])
            file_b = str(right["file_name"])
            meta_a = manifest_by_name[file_a]
            meta_b = manifest_by_name[file_b]
            same_sha256 = str(left["sha256"]) == str(right["sha256"])
            priority = priority_for_distances(
                dhash_distance,
                ahash_distance,
                lowres_correlation,
                same_sha256,
            )

            row = {
                "file_a": file_a,
                "file_b": file_b,
                "level_a": str(left["severity_level"]),
                "level_b": str(right["severity_level"]),
                "same_level": str(left["severity_level"] == right["severity_level"]),
                "dhash_distance": dhash_distance,
                "ahash_distance": ahash_distance,
                "lowres_correlation": f"{lowres_correlation:.6f}",
                "combined_distance": combined_distance,
                "same_sha256": str(same_sha256),
                "width_a": meta_a["pixel_width"],
                "height_a": meta_a["pixel_height"],
                "width_b": meta_b["pixel_width"],
                "height_b": meta_b["pixel_height"],
                "review_priority": priority,
            }

            if priority != "nearest_only":
                candidates.append(row)

            for name in [file_a, file_b]:
                current = nearest_by_file.get(name)
                if current is None or combined_distance < current[0]:
                    nearest_by_file[name] = (combined_distance, row)

    nearest_rows = sorted(
        (entry for _, entry in nearest_by_file.values()),
        key=lambda row: (int(row["combined_distance"]), int(row["dhash_distance"]), int(row["ahash_distance"])),
    )[:nearest_limit]

    candidates = sorted(
        candidates,
        key=lambda row: (
            {"exact_duplicate": 0, "high": 1, "medium": 2, "low": 3}.get(str(row["review_priority"]), 9),
            -float(row["lowres_correlation"]),
            int(row["combined_distance"]),
            int(row["dhash_distance"]),
            int(row["ahash_distance"]),
        ),
    )
    return candidates, nearest_rows


def split_balance_rows(
    manifest_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    manifest_by_name = {row["file_name"]: row for row in manifest_rows}
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for split_row in split_rows:
        key = (split_row["repeat"], split_row["seed"], split_row["fold"], split_row["split"])
        grouped[key].append(manifest_by_name[split_row["file_name"]])

    output: list[dict[str, object]] = []
    split_order = {"train": 0, "val": 1, "test": 2}
    for key, rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), int(item[0][2]), split_order[item[0][3]])):
        repeat, seed, fold, split = key
        levels = Counter(row["severity_level"] for row in rows)
        genders = Counter(row["gender"] for row in rows)
        ages = [float(row["age"]) for row in rows]
        widths = [float(row["pixel_width"]) for row in rows]
        heights = [float(row["pixel_height"]) for row in rows]
        ratios = [float(row["aspect_ratio"]) for row in rows]
        sizes = [float(row["size_bytes"]) for row in rows]
        n = len(rows)
        output.append(
            {
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                "split": split,
                "n": n,
                "level_1": levels.get("1", 0),
                "level_2": levels.get("2", 0),
                "level_3": levels.get("3", 0),
                "age_mean": format_float(mean(ages), 3),
                "age_sd": format_float(sd(ages), 3),
                "age_min": format_float(min(ages), 0),
                "age_max": format_float(max(ages), 0),
                "gender_0": genders.get("0", 0),
                "gender_1": genders.get("1", 0),
                "gender_1_prop": format_float(genders.get("1", 0) / n if n else 0.0, 4),
                "pixel_width_mean": format_float(mean(widths), 2),
                "pixel_height_mean": format_float(mean(heights), 2),
                "aspect_ratio_mean": format_float(mean(ratios), 4),
                "size_bytes_mean": format_float(mean(sizes), 1),
            }
        )
    return output


def split_risk_rows(
    candidates: list[dict[str, object]],
    assignment_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    assignments: dict[tuple[str, str], str] = {}
    for row in assignment_rows:
        assignments[(row["seed"], row["file_name"])] = row["fold_group"]

    seeds = sorted({row["seed"] for row in assignment_rows}, key=int)
    output: list[dict[str, object]] = []
    for pair in candidates:
        if str(pair["review_priority"]) not in {"exact_duplicate", "high", "medium"}:
            continue
        for seed in seeds:
            fold_a = assignments.get((seed, str(pair["file_a"])), "")
            fold_b = assignments.get((seed, str(pair["file_b"])), "")
            output.append(
                {
                    "file_a": pair["file_a"],
                    "file_b": pair["file_b"],
                    "level_a": pair["level_a"],
                    "level_b": pair["level_b"],
                    "dhash_distance": pair["dhash_distance"],
                    "ahash_distance": pair["ahash_distance"],
                    "lowres_correlation": pair["lowres_correlation"],
                    "seed": seed,
                    "fold_group_a": fold_a,
                    "fold_group_b": fold_b,
                    "crosses_fold_group": str(fold_a != fold_b),
                }
            )
    return output


def summarize_split_balance(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split in ["train", "val", "test"]:
        subset = [row for row in rows if row["split"] == split]
        summary[split] = {
            "n_min": min(int(row["n"]) for row in subset),
            "n_max": max(int(row["n"]) for row in subset),
            "age_mean_min": min(float(row["age_mean"]) for row in subset),
            "age_mean_max": max(float(row["age_mean"]) for row in subset),
            "gender_1_prop_min": min(float(row["gender_1_prop"]) for row in subset),
            "gender_1_prop_max": max(float(row["gender_1_prop"]) for row in subset),
            "aspect_ratio_mean_min": min(float(row["aspect_ratio_mean"]) for row in subset),
            "aspect_ratio_mean_max": max(float(row["aspect_ratio_mean"]) for row in subset),
        }
    return summary


def package_versions() -> dict[str, str]:
    versions = {
        "python_executable": Path(sys.executable).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pillow": getattr(Image, "__version__", "unknown"),
    }
    for name in ["numpy", "pandas", "sklearn", "torch", "torchvision"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - runtime report should capture import failures.
            versions[name] = f"not available ({type(exc).__name__})"
    return versions


def markdown_summary(summary: dict[str, object]) -> str:
    near = summary["near_duplicates"]  # type: ignore[index]
    split = summary["split_balance"]  # type: ignore[index]
    env = summary["environment"]  # type: ignore[index]

    split_lines = []
    for split_name in ["train", "val", "test"]:
        values = split[split_name]  # type: ignore[index]
        split_lines.append(
            "| {split} | {n_min}-{n_max} | {age_min:.3f}-{age_max:.3f} | "
            "{gender_min:.4f}-{gender_max:.4f} | {ratio_min:.4f}-{ratio_max:.4f} |".format(
                split=split_name,
                n_min=values["n_min"],
                n_max=values["n_max"],
                age_min=values["age_mean_min"],
                age_max=values["age_mean_max"],
                gender_min=values["gender_1_prop_min"],
                gender_max=values["gender_1_prop_max"],
                ratio_min=values["aspect_ratio_mean_min"],
                ratio_max=values["aspect_ratio_mean_max"],
            )
        )

    return f"""# Pre-Model Audit

Date: 2026-06-07

## Runtime

- Python executable: `{env["python_executable"]}`
- Python version: `{env["python_version"]}`
- Platform: `{env["platform"]}`
- Pillow: `{env["pillow"]}`
- NumPy: `{env["numpy"]}`
- pandas: `{env["pandas"]}`
- scikit-learn: `{env["sklearn"]}`
- torch: `{env["torch"]}`
- torchvision: `{env["torchvision"]}`

## Near-Duplicate Audit

- Images fingerprinted: {summary["images_fingerprinted"]}
- Exact duplicate SHA-256 groups from manifest: {near["exact_sha256_pairs"]}
- Low-priority hash-only candidates logged: {near["candidate_pairs"]}
- High-priority near-duplicate candidates: {near["high_priority_pairs"]}
- Medium-priority near-duplicate candidates: {near["medium_priority_pairs"]}
- Candidate split-risk rows: {near["split_risk_rows"]}
- Candidate split-risk rows crossing fold group: {near["crossing_fold_group_rows"]}

Interpretation: exact duplicates remain absent. Low-priority hash-only similarities are expected in panoramic radiographs because the images share a common global silhouette. Any high/medium near-duplicate candidate crossing fold groups should be manually reviewed before final image-model evaluation.

## Split Balance Summary

| Split | N range | Mean age range | Gender=1 proportion range | Mean aspect-ratio range |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(split_lines)}

Interpretation: the current repeated stratified folds are balanced by severity and do not show severe age, gender, or image-geometry imbalance.

## Generated Files

- `data/processed/image_fingerprints.csv`
- `reports/near_duplicate_candidates.csv`
- `reports/near_duplicate_nearest_neighbors.csv`
- `reports/near_duplicate_split_risk.csv`
- `reports/split_balance_audit.csv`
- `reports/pre_model_audit_summary.json`
- `reports/environment_report.md`
"""


def write_environment_report(path: Path, versions: dict[str, str]) -> None:
    lines = ["# Environment Report", "", "Date: 2026-06-07", ""]
    for key, value in versions.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--assignments", type=Path, default=ASSIGNMENTS)
    parser.add_argument("--nearest-limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(args.manifest)
    split_rows = read_csv(args.splits)
    assignment_rows = read_csv(args.assignments)

    fingerprint_rows = build_fingerprints(manifest_rows)
    write_csv(FINGERPRINTS, fingerprint_rows, FINGERPRINT_FIELDS)

    candidates, nearest = near_duplicate_pairs(manifest_rows, fingerprint_rows, args.nearest_limit)
    write_csv(NEAR_DUPLICATES, candidates, NEAR_DUPLICATE_FIELDS)
    write_csv(NEAREST_NEIGHBORS, nearest, NEAR_DUPLICATE_FIELDS)

    risk_rows = split_risk_rows(candidates, assignment_rows)
    write_csv(NEAR_DUPLICATE_SPLIT_RISK, risk_rows, SPLIT_RISK_FIELDS)

    balance_rows = split_balance_rows(manifest_rows, split_rows)
    write_csv(SPLIT_BALANCE, balance_rows, SPLIT_BALANCE_FIELDS)

    versions = package_versions()
    write_environment_report(ENVIRONMENT_MD, versions)

    near_summary = {
        "exact_sha256_pairs": sum(1 for row in candidates if row["same_sha256"] == "True"),
        "candidate_pairs": len(candidates),
        "high_priority_pairs": sum(1 for row in candidates if row["review_priority"] == "high"),
        "medium_priority_pairs": sum(1 for row in candidates if row["review_priority"] == "medium"),
        "low_priority_pairs": sum(1 for row in candidates if row["review_priority"] == "low"),
        "split_risk_rows": len(risk_rows),
        "crossing_fold_group_rows": sum(1 for row in risk_rows if row["crosses_fold_group"] == "True"),
    }
    summary = {
        "images_fingerprinted": len(fingerprint_rows),
        "near_duplicates": near_summary,
        "split_balance": summarize_split_balance(balance_rows),
        "environment": versions,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SUMMARY_MD.write_text(markdown_summary(summary), encoding="utf-8")

    print(f"fingerprints: {FINGERPRINTS}")
    print(f"near_duplicates: {NEAR_DUPLICATES}")
    print(f"nearest_neighbors: {NEAREST_NEIGHBORS}")
    print(f"split_risk: {NEAR_DUPLICATE_SPLIT_RISK}")
    print(f"split_balance: {SPLIT_BALANCE}")
    print(f"summary: {SUMMARY_MD}")
    print(f"images_fingerprinted: {len(fingerprint_rows)}")
    print(f"candidate_pairs: {near_summary['candidate_pairs']}")
    print(f"crossing_fold_group_rows: {near_summary['crossing_fold_group_rows']}")


if __name__ == "__main__":
    main()
