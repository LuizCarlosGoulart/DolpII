from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_declares_minimal_runtime_and_optional_dev_groups() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    optional = pyproject["project"]["optional-dependencies"]

    assert "pytesseract>=0.3.10,<0.4" in dependencies
    assert "pillow>=10,<12" in dependencies
    assert "PyYAML>=6,<7" in dependencies
    assert "pydantic>=2.7,<3" in dependencies
    assert "pytest>=8,<9" in optional["dev"]
    assert "jupyter>=1,<2" in optional["notebooks"]


def test_requirements_files_delegate_to_pyproject_install_targets() -> None:
    runtime = (ROOT / "requirements.txt").read_text(encoding="utf-8").strip()
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").strip()

    assert runtime == "-e ."
    assert dev == "-e .[dev,notebooks]"
