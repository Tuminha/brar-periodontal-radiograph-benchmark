#!/usr/bin/env python3
"""Create repeated stratified BRAR train/validation/test split manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MANIFEST = PROCESSED / "brar_manifest.csv"
SPLIT_DIR = PROCESSED / "splits"
DEFAULT_SEEDS = "20260606,20260607,20260608"


SPLIT_FIELDNAMES = [
    "repeat",
    "seed",
    "fold",
    "file_name",
    "severity_level",
    "split",
    "fold_group",
    "group_id",
    "relative_image_path",
]

ASSIGNMENT_FIELDNAMES = [
    "seed",
    "file_name",
    "severity_level",
    "fold_group",
    "group_id",
    "relative_image_path",
]

SUMMARY_FIELDNAMES = [
    "repeat",
    "seed",
    "fold",
    "split",
    "severity_level",
    "count",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(value: str) -> list[int]:
    seeds = []
    for token in value.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def valid_manifest_rows(rows: list[dict[str, str]], allow_invalid: bool) -> list[dict[str, str]]:
    invalid = [
        row
        for row in rows
        if row["linkage_status"] != "PASS" or row["readability_status"] != "PASS"
    ]
    if invalid and not allow_invalid:
        examples = ", ".join(row["file_name"] for row in invalid[:5])
        raise RuntimeError(
            f"manifest contains {len(invalid)} invalid rows; examples: {examples}. "
            "Use --allow-invalid only for debugging."
        )
    return [row for row in rows if allow_invalid or row not in invalid]


def build_groups(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Use exact image hash as group id so exact duplicates cannot cross folds."""
    rows_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = row["sha256"] or row["file_name"]
        rows_by_hash[group_id].append(row)

    groups: list[dict[str, object]] = []
    for group_id, group_rows in sorted(rows_by_hash.items()):
        levels = {row["severity_level"] for row in group_rows}
        if len(levels) != 1:
            names = ", ".join(row["file_name"] for row in group_rows)
            raise RuntimeError(
                f"exact duplicate group has conflicting labels: {group_id} -> {names}"
            )
        groups.append(
            {
                "group_id": group_id,
                "severity_level": next(iter(levels)),
                "rows": group_rows,
            }
        )
    return groups


def assign_group_folds(
    groups: list[dict[str, object]],
    seed: int,
    n_folds: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    groups_by_level: dict[str, list[dict[str, object]]] = defaultdict(list)
    for group in groups:
        groups_by_level[str(group["severity_level"])].append(group)

    assignments: dict[str, int] = {}
    for level in sorted(groups_by_level, key=int):
        level_groups = groups_by_level[level][:]
        rng.shuffle(level_groups)
        for idx, group in enumerate(level_groups):
            assignments[str(group["group_id"])] = idx % n_folds
    return assignments


def split_for_fold(fold_group: int, test_fold: int, n_folds: int) -> str:
    validation_fold = (test_fold + 1) % n_folds
    if fold_group == test_fold:
        return "test"
    if fold_group == validation_fold:
        return "val"
    return "train"


def make_splits(
    rows: list[dict[str, str]],
    seeds: list[int],
    n_folds: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    groups = build_groups(rows)
    all_split_rows: list[dict[str, str]] = []
    all_assignment_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for repeat_idx, seed in enumerate(seeds, start=1):
        assignments = assign_group_folds(groups, seed, n_folds)

        for group in groups:
            group_id = str(group["group_id"])
            fold_group = assignments[group_id]
            for row in group["rows"]:  # type: ignore[index]
                all_assignment_rows.append(
                    {
                        "seed": str(seed),
                        "file_name": row["file_name"],
                        "severity_level": row["severity_level"],
                        "fold_group": str(fold_group),
                        "group_id": group_id,
                        "relative_image_path": row["relative_image_path"],
                    }
                )

        for fold in range(n_folds):
            counter: Counter[tuple[str, str]] = Counter()
            for group in groups:
                group_id = str(group["group_id"])
                fold_group = assignments[group_id]
                split = split_for_fold(fold_group, fold, n_folds)
                for row in group["rows"]:  # type: ignore[index]
                    counter[(split, row["severity_level"])] += 1
                    all_split_rows.append(
                        {
                            "repeat": str(repeat_idx),
                            "seed": str(seed),
                            "fold": str(fold),
                            "file_name": row["file_name"],
                            "severity_level": row["severity_level"],
                            "split": split,
                            "fold_group": str(fold_group),
                            "group_id": group_id,
                            "relative_image_path": row["relative_image_path"],
                        }
                    )

            for split in ["train", "val", "test"]:
                for level in ["1", "2", "3"]:
                    summary_rows.append(
                        {
                            "repeat": str(repeat_idx),
                            "seed": str(seed),
                            "fold": str(fold),
                            "split": split,
                            "severity_level": level,
                            "count": str(counter[(split, level)]),
                        }
                    )

    duplicate_groups = [group for group in groups if len(group["rows"]) > 1]  # type: ignore[arg-type]
    summary = {
        "manifest_rows_used": len(rows),
        "groups": len(groups),
        "exact_duplicate_groups": len(duplicate_groups),
        "n_folds": n_folds,
        "seeds": seeds,
        "split_rows": len(all_split_rows),
        "assignment_rows": len(all_assignment_rows),
        "level_counts": dict(sorted(Counter(row["severity_level"] for row in rows).items())),
    }
    return all_split_rows, all_assignment_rows, summary_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--allow-invalid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_folds < 3:
        raise ValueError("--n-folds must be at least 3 for train/val/test assignment")

    rows = valid_manifest_rows(read_csv(args.manifest), allow_invalid=args.allow_invalid)
    seeds = parse_seeds(args.seeds)
    split_rows, assignment_rows, summary_rows, summary = make_splits(rows, seeds, args.n_folds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "brar_repeated_5fold_splits.csv"
    assignment_path = args.output_dir / "brar_fold_assignments.csv"
    summary_path = args.output_dir / "brar_split_summary.csv"
    json_path = args.output_dir / "brar_split_summary.json"

    write_csv(split_path, split_rows, SPLIT_FIELDNAMES)
    write_csv(assignment_path, assignment_rows, ASSIGNMENT_FIELDNAMES)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDNAMES)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"splits: {split_path}")
    print(f"assignments: {assignment_path}")
    print(f"summary_csv: {summary_path}")
    print(f"summary_json: {json_path}")
    print(f"manifest_rows_used: {summary['manifest_rows_used']}")
    print(f"seeds: {summary['seeds']}")
    print(f"n_folds: {summary['n_folds']}")
    print(f"split_rows: {summary['split_rows']}")


if __name__ == "__main__":
    main()
