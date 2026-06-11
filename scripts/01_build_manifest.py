#!/usr/bin/env python3
"""Build the BRAR modeling manifest.

This script prepares a machine-learning-ready manifest without training a
model. It links metadata rows to image paths, records image dimensions and
hashes, and writes a compact audit summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "data" / "extracted"
METADATA = EXTRACTED / "meta_data.csv"
PROCESSED = ROOT / "data" / "processed"
MANIFEST = PROCESSED / "brar_manifest.csv"
SUMMARY = PROCESSED / "brar_manifest_summary.json"


FIELDNAMES = [
    "row_index",
    "file_name",
    "relative_image_path",
    "severity_level",
    "grade_folder",
    "age",
    "gender",
    "bone_resorption",
    "bone_resorption_age",
    "number_of_missing_teeth",
    "implant",
    "residual_root",
    "functional_tooth_logarithm",
    "size_bytes",
    "sha256",
    "pixel_width",
    "pixel_height",
    "aspect_ratio",
    "linkage_status",
    "readability_status",
    "audit_notes",
]


SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Return JPEG width and height using marker parsing only."""
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError("not a JPEG file")

        while True:
            byte = handle.read(1)
            while byte and byte != b"\xff":
                byte = handle.read(1)
            if not byte:
                raise ValueError("JPEG SOF marker not found")

            marker_byte = handle.read(1)
            while marker_byte == b"\xff":
                marker_byte = handle.read(1)
            if not marker_byte:
                raise ValueError("truncated JPEG marker")

            marker = marker_byte[0]
            if marker in {0xD8, 0xD9, 0x01}:
                continue

            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                raise ValueError("truncated JPEG segment length")
            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                raise ValueError("invalid JPEG segment length")

            if marker in SOF_MARKERS:
                data = handle.read(segment_length - 2)
                if len(data) < 5:
                    raise ValueError("truncated JPEG SOF segment")
                height = int.from_bytes(data[1:3], "big")
                width = int.from_bytes(data[3:5], "big")
                return width, height

            handle.seek(segment_length - 2, 1)


def image_paths() -> list[Path]:
    return sorted(EXTRACTED.rglob("*.jpg"))


def grade_from_path(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith("level_"):
        return parent.split("_", 1)[1]
    return ""


def build_manifest() -> tuple[list[dict[str, str]], dict[str, object]]:
    metadata_rows = read_csv(METADATA)
    paths_by_name = {path.name: path for path in image_paths()}
    metadata_names = [row["File name"] for row in metadata_rows]
    metadata_name_set = set(metadata_names)
    image_name_set = set(paths_by_name)

    manifest_rows: list[dict[str, str]] = []
    hash_to_names: dict[str, list[str]] = defaultdict(list)
    dimension_counter: Counter[str] = Counter()
    level_counter: Counter[str] = Counter()
    linkage_counter: Counter[str] = Counter()
    readability_counter: Counter[str] = Counter()

    for idx, source_row in enumerate(metadata_rows, start=1):
        file_name = source_row["File name"]
        path = paths_by_name.get(file_name)
        level = source_row["Level"]
        grade_folder = grade_from_path(path) if path else ""
        notes: list[str] = []

        linkage_status = "PASS"
        if path is None:
            linkage_status = "MISSING_IMAGE"
            notes.append("metadata row has no matching image file")
        elif grade_folder != level:
            linkage_status = "GRADE_MISMATCH"
            notes.append("image folder grade differs from CSV Level")

        size_bytes = ""
        digest = ""
        width = ""
        height = ""
        aspect_ratio = ""
        readability_status = "NOT_CHECKED"

        if path is not None:
            size_bytes = str(path.stat().st_size)
            digest = sha256_file(path)
            hash_to_names[digest].append(file_name)
            try:
                image_width, image_height = read_jpeg_dimensions(path)
                width = str(image_width)
                height = str(image_height)
                aspect_ratio = f"{image_width / image_height:.6f}" if image_height else ""
                readability_status = "PASS"
                dimension_counter[f"{image_width}x{image_height}"] += 1
            except Exception as exc:  # noqa: BLE001 - report all readability failures.
                readability_status = "FAIL"
                notes.append(f"image dimension read failed: {exc}")

        level_counter[level] += 1
        linkage_counter[linkage_status] += 1
        readability_counter[readability_status] += 1

        manifest_rows.append(
            {
                "row_index": str(idx),
                "file_name": file_name,
                "relative_image_path": str(path.relative_to(ROOT)) if path else "",
                "severity_level": level,
                "grade_folder": grade_folder,
                "age": source_row["Age"],
                "gender": source_row["Gender"],
                "bone_resorption": source_row["Bone resorption"],
                "bone_resorption_age": source_row["Bone resorption Age"],
                "number_of_missing_teeth": source_row["Number of missing teeth"],
                "implant": source_row["Implant"],
                "residual_root": source_row["Residual root"],
                "functional_tooth_logarithm": source_row["Functional tooth logarithm"],
                "size_bytes": size_bytes,
                "sha256": digest,
                "pixel_width": width,
                "pixel_height": height,
                "aspect_ratio": aspect_ratio,
                "linkage_status": linkage_status,
                "readability_status": readability_status,
                "audit_notes": "; ".join(notes),
            }
        )

    duplicate_metadata_names = sorted(
        name for name, count in Counter(metadata_names).items() if count > 1
    )
    duplicate_hashes = {
        digest: sorted(names)
        for digest, names in hash_to_names.items()
        if digest and len(names) > 1
    }

    summary = {
        "metadata_rows": len(metadata_rows),
        "manifest_rows": len(manifest_rows),
        "image_files": len(paths_by_name),
        "missing_images": sorted(metadata_name_set - image_name_set),
        "unlinked_images": sorted(image_name_set - metadata_name_set),
        "duplicate_metadata_names": duplicate_metadata_names,
        "duplicate_exact_image_hashes": duplicate_hashes,
        "level_counts": dict(sorted(level_counter.items())),
        "linkage_status_counts": dict(sorted(linkage_counter.items())),
        "readability_status_counts": dict(sorted(readability_counter.items())),
        "top_dimensions": dict(dimension_counter.most_common(20)),
    }
    return manifest_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_manifest()
    write_csv(args.manifest, rows, FIELDNAMES)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"manifest: {args.manifest}")
    print(f"summary: {args.summary}")
    print(f"rows: {summary['manifest_rows']}")
    print(f"levels: {summary['level_counts']}")
    print(f"linkage: {summary['linkage_status_counts']}")
    print(f"readability: {summary['readability_status_counts']}")
    print(f"exact_duplicate_hashes: {len(summary['duplicate_exact_image_hashes'])}")


if __name__ == "__main__":
    main()
