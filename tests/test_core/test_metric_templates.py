"""Guards that the compiled ``templates.json`` stays in sync with its sources.

The metric prompt templates live as ``.txt`` files under
``deepeval/metrics/**/templates/<Class>/<method>.txt`` (plus shared fragments),
and are compiled into a single ``deepeval/metric_templates/templates.json`` by
``scripts/compile_metric_templates.py``. That JSON is a committed build artifact,
so it can silently drift if someone edits a ``.txt`` without recompiling. This
test fails loudly when that happens.
"""

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILE_SCRIPT = REPO_ROOT / "scripts" / "compile_metric_templates.py"
TEMPLATES_JSON = REPO_ROOT / "deepeval" / "metric_templates" / "templates.json"


def _load_compiler():
    spec = importlib.util.spec_from_file_location(
        "_compile_metric_templates", COMPILE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_templates_json_is_up_to_date():
    compiler = _load_compiler()
    expected = compiler.render_bundle_json(compiler.build_bundle())
    actual = TEMPLATES_JSON.read_text(encoding="utf-8")
    assert actual == expected, (
        "deepeval/metric_templates/templates.json is out of date with the "
        "template .txt files. Re-run `python scripts/compile_metric_templates.py`."
    )


def test_templates_json_is_valid_and_nonempty():
    data = json.loads(TEMPLATES_JSON.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data
    # Every class entry maps method -> string template.
    for class_name, methods in data.items():
        assert isinstance(methods, dict), class_name
        for method, body in methods.items():
            assert isinstance(body, str), f"{class_name}.{method}"
