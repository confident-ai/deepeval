"""Guards that compiled ``templates.json`` bundles stay in sync with sources.

Prompt templates live as ``.txt`` files under feature source trees and are
compiled by ``scripts/compile_metric_templates.py`` into BOTH the Python and
TypeScript packages. These tests fail when a ``.txt`` changes without a
recompile.
"""

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILE_SCRIPT = REPO_ROOT / "scripts" / "compile_metric_templates.py"


def _load_compiler():
    spec = importlib.util.spec_from_file_location(
        "_compile_metric_templates", COMPILE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_feature_up_to_date(feature: str) -> None:
    compiler = _load_compiler()
    expected = compiler.render_bundle_json(compiler.build_bundle(feature))
    py_path = (
        REPO_ROOT / "deepeval" / "templates" / feature / "templates.json"
    )
    ts_path = (
        REPO_ROOT
        / "typescript"
        / "src"
        / "templates"
        / feature
        / "templates.json"
    )
    for path in (py_path, ts_path):
        assert path.is_file(), f"Missing compiled bundle: {path}"
        assert path.read_text(encoding="utf-8") == expected, (
            f"{path} is out of date with the template .txt files. "
            f"Re-run `python scripts/compile_metric_templates.py {feature}`."
        )


def test_metrics_templates_json_is_up_to_date():
    _assert_feature_up_to_date("metrics")


def test_simulator_templates_json_is_up_to_date():
    _assert_feature_up_to_date("simulator")


def test_metrics_templates_json_is_valid_and_nonempty():
    path = (
        REPO_ROOT / "deepeval" / "templates" / "metrics" / "templates.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data
    for class_name, methods in data.items():
        assert isinstance(methods, dict), class_name
        for method, body in methods.items():
            assert isinstance(body, str), f"{class_name}.{method}"


def test_simulator_interrupt_templates_present():
    path = (
        REPO_ROOT / "deepeval" / "templates" / "simulator" / "templates.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    methods = data["SimulatorInterruptTemplate"]
    assert "decide_interrupt" in methods
    assert "interruption_bias_rare" in methods
    assert "interruption_bias_normal" in methods
    assert "interruption_bias_frequent" in methods
    assert "interruption_frustration" in methods
