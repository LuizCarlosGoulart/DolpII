from src.observability.environment_report import (
    collect_environment_report,
    format_environment_report,
)


def test_collect_environment_report_includes_expected_keys() -> None:
    report = collect_environment_report(
        packages=("sys", "definitely_missing_package"),
        storage_paths=(".", "./definitely-missing-path"),
    )

    assert report["python_version"]
    assert report["runtime_type"] in {"local", "notebook", "colab"}
    assert isinstance(report["gpu_available"], bool)
    assert report["packages"] == {
        "sys": True,
        "definitely_missing_package": False,
    }
    assert report["storage_paths"] == {
        ".": True,
        "./definitely-missing-path": False,
    }


def test_format_environment_report_renders_concise_summary() -> None:
    report = {
        "python_version": "3.11.0",
        "runtime_type": "colab",
        "gpu_available": True,
        "packages": {"PIL": True, "yaml": False},
        "storage_paths": {"/content": True, "/content/drive": False},
    }

    formatted = format_environment_report(report)

    assert "Environment Report" in formatted
    assert "Python: 3.11.0" in formatted
    assert "Runtime: colab" in formatted
    assert "GPU available: yes" in formatted
    assert "PIL=ok" in formatted
    assert "yaml=missing" in formatted
    assert "/content=ok" in formatted
    assert "/content/drive=missing" in formatted
