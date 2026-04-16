"""Environment inspection helpers for local and Colab execution."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import platform
import shutil
import subprocess
import sys


DEFAULT_PACKAGES = (
    "PIL",
    "pydantic",
    "pytesseract",
    "yaml",
)

DEFAULT_STORAGE_PATHS = (
    "/content",
    "/content/drive",
    "/content/drive/MyDrive",
)


def detect_runtime_type() -> str:
    """Return a lightweight runtime label without importing heavy packages."""
    if find_spec("google.colab") is not None:
        return "colab"
    if "ipykernel" in sys.modules:
        return "notebook"
    return "local"


def is_gpu_available() -> bool:
    """Best-effort GPU detection that stays dependency-free."""
    if shutil.which("nvidia-smi") is None:
        return False

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def collect_environment_report(
    packages: tuple[str, ...] = DEFAULT_PACKAGES,
    storage_paths: tuple[str, ...] = DEFAULT_STORAGE_PATHS,
) -> dict[str, object]:
    """Collect a concise environment report for setup validation."""
    package_status = {name: find_spec(name) is not None for name in packages}
    storage_status = {path: Path(path).exists() for path in storage_paths}

    return {
        "python_version": platform.python_version(),
        "runtime_type": detect_runtime_type(),
        "gpu_available": is_gpu_available(),
        "packages": package_status,
        "storage_paths": storage_status,
    }


def format_environment_report(report: dict[str, object]) -> str:
    """Render the report in a short, notebook-friendly format."""
    package_status = report["packages"]
    storage_status = report["storage_paths"]

    package_summary = ", ".join(
        f"{name}={'ok' if installed else 'missing'}"
        for name, installed in package_status.items()
    )
    storage_summary = ", ".join(
        f"{path}={'ok' if exists else 'missing'}"
        for path, exists in storage_status.items()
    )

    return "\n".join(
        [
            "Environment Report",
            f"Python: {report['python_version']}",
            f"Runtime: {report['runtime_type']}",
            f"GPU available: {'yes' if report['gpu_available'] else 'no'}",
            f"Packages: {package_summary}",
            f"Storage paths: {storage_summary}",
        ]
    )
