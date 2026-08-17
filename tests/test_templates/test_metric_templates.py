"""Guards that the compiled ``templates.json`` bundles stay in sync with sources.

The metric prompt templates live as ``.txt`` files under
``deepeval/metrics/**/templates/<Class>/<method>.txt`` (plus shared fragments),
and are compiled into ``templates/metrics/templates.json`` by
``scripts/compile_metric_templates.py``, which emits the bundle into BOTH the
Python package (``deepeval/templates/metrics/``) and the TypeScript package
(``typescript/src/templates/metrics/``). Those JSON files are committed build
artifacts, so they can silently drift if someone edits a ``.txt`` without
recompiling. These tests fail loudly when that happens.
"""

import importlib.util
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILE_SCRIPT = REPO_ROOT / "scripts" / "compile_metric_templates.py"
PY_TEMPLATES_JSON = (
    REPO_ROOT / "deepeval" / "templates" / "metrics" / "templates.json"
)
TS_TEMPLATES_JSON = (
    REPO_ROOT
    / "typescript"
    / "src"
    / "templates"
    / "metrics"
    / "templates.json"
)


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
    for path in (PY_TEMPLATES_JSON, TS_TEMPLATES_JSON):
        assert path.read_text(encoding="utf-8") == expected, (
            f"{path} is out of date with the template .txt files. "
            "Re-run `python scripts/compile_metric_templates.py`."
        )


def test_templates_carry_no_threat_language():
    """No prompt may pressure the judge by threatening a consequence.

    The prompts are static, so a guardrail-protected provider can reject one
    before the metric runs, which fails the metric for reasons that have
    nothing to do with the output under evaluation. Every constraint stated
    this way is already stated without coercion elsewhere in the bundle --
    compare the faithfulness text-only and multimodal guideline fragments.

    Matches the threatened consequence, not violent words on their own:
    metrics like toxicity, bias and PII discuss harm legitimately, and
    SummarizationMetric.generate_answers has an example about one character
    killing another.
    """
    threats = re.compile(r"will die|you will suffer", re.I)
    data = json.loads(PY_TEMPLATES_JSON.read_text(encoding="utf-8"))
    offenders = [
        f"{class_name}.{method}: {match.group(0)!r}"
        for class_name, methods in data.items()
        for method, body in methods.items()
        if (match := threats.search(body))
    ]
    assert not offenders, "Coercive prompt language found in: " + ", ".join(
        offenders
    )


def test_templates_json_is_valid_and_nonempty():
    data = json.loads(PY_TEMPLATES_JSON.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data
    # Every class entry maps method -> string template.
    for class_name, methods in data.items():
        assert isinstance(methods, dict), class_name
        for method, body in methods.items():
            assert isinstance(body, str), f"{class_name}.{method}"
