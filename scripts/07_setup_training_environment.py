#!/usr/bin/env python3
"""Record the BRAR image-training Python environment."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "training_environment_report.md"
REPORT_JSON = REPORTS / "training_environment_report.json"
LOCK_FILE = ROOT / "requirements-training.lock.txt"

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "sklearn",
    "PIL",
    "matplotlib",
    "numpy",
    "pandas",
]


def import_status(package: str) -> dict[str, str]:
    try:
        module = __import__(package)
        version = getattr(module, "__version__", "unknown")
        if package == "PIL":
            from PIL import Image

            version = getattr(Image, "__version__", version)
        return {"status": "available", "version": str(version)}
    except Exception as exc:  # noqa: BLE001 - environment report should capture all import failures.
        return {"status": "missing", "version": "", "error": f"{type(exc).__name__}: {exc}"}


def torch_device_info() -> dict[str, object]:
    try:
        import torch

        info: dict[str, object] = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "recommended_device": "cpu",
        }
        if info["cuda_available"]:
            info["recommended_device"] = "cuda"
        elif info["mps_available"]:
            info["recommended_device"] = "mps"
        return info
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "recommended_device": "unavailable"}


def pip_freeze() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def write_markdown(report: dict[str, object]) -> None:
    packages = report["packages"]  # type: ignore[index]
    device = report["device"]  # type: ignore[index]
    lines = [
        "# Training Environment Report",
        "",
        f"Date: {report['date']}",
        "",
        "## Python",
        "",
        f"- Executable: `{report['python_executable']}`",
        f"- Version: `{report['python_version']}`",
        f"- Platform: `{report['platform']}`",
        "",
        "## Package Status",
        "",
        "| Package | Status | Version / Error |",
        "| --- | --- | --- |",
    ]
    for package, status in packages.items():  # type: ignore[union-attr]
        detail = status.get("version") or status.get("error", "")  # type: ignore[union-attr]
        lines.append(f"| `{package}` | {status['status']} | `{detail}` |")  # type: ignore[index]
    lines.extend(
        [
            "",
            "## Device",
            "",
            f"- Recommended device: `{device.get('recommended_device', '')}`",  # type: ignore[union-attr]
            f"- CUDA available: `{device.get('cuda_available', False)}`",  # type: ignore[union-attr]
            f"- MPS available: `{device.get('mps_available', False)}`",  # type: ignore[union-attr]
            "",
            "## Lock File",
            "",
            f"- `{LOCK_FILE.relative_to(ROOT)}`",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {package: import_status(package) for package in REQUIRED_PACKAGES},
        "device": torch_device_info(),
    }
    freeze_lines = pip_freeze()
    LOCK_FILE.write_text("\n".join(freeze_lines) + "\n", encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report)

    missing = [
        package
        for package, status in report["packages"].items()  # type: ignore[union-attr]
        if status["status"] != "available"  # type: ignore[index]
    ]
    print(f"report: {REPORT_MD}")
    print(f"lock: {LOCK_FILE}")
    print(f"recommended_device: {report['device'].get('recommended_device')}")  # type: ignore[union-attr]
    if missing:
        print(f"missing: {', '.join(missing)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
