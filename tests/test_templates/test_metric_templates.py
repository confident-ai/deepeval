"""Guards that compiled ``templates.json`` bundles stay in sync with sources.

Prompt templates live as ``.txt`` files under feature source trees and are
compiled by ``scripts/compile_metric_templates.py`` into BOTH the Python
package (``deepeval/templates/<feature>/``) and the TypeScript package
(``typescript/src/templates/<feature>/``). Those JSON files are committed build
artifacts, so they can silently drift if someone edits a ``.txt`` without
recompiling. These tests fail loudly when that happens.

The per-feature checks are parametrized off the compiler's own ``FEATURES``
rather than a list repeated here, so a newly compiled feature is guarded the
moment it is registered instead of whenever someone remembers to add it.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILE_SCRIPT = REPO_ROOT / "scripts" / "compile_metric_templates.py"


def _load_compiler():
    spec = importlib.util.spec_from_file_location(
        "_compile_metric_templates", COMPILE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FEATURES = tuple(sorted(_load_compiler().FEATURES))


def _py_bundle(feature: str) -> Path:
    return REPO_ROOT / "deepeval" / "templates" / feature / "templates.json"


def _ts_bundle(feature: str) -> Path:
    return (
        REPO_ROOT
        / "typescript"
        / "src"
        / "templates"
        / feature
        / "templates.json"
    )


def _bundle_data(feature: str) -> dict:
    return json.loads(_py_bundle(feature).read_text(encoding="utf-8"))


@pytest.mark.parametrize("feature", FEATURES)
def test_templates_json_is_up_to_date(feature: str):
    compiler = _load_compiler()
    expected = compiler.render_bundle_json(compiler.build_bundle(feature))
    for path in (_py_bundle(feature), _ts_bundle(feature)):
        assert path.is_file(), f"Missing compiled bundle: {path}"
        assert path.read_text(encoding="utf-8") == expected, (
            f"{path} is out of date with the template .txt files. "
            f"Re-run `python scripts/compile_metric_templates.py {feature}`."
        )


@pytest.mark.parametrize("feature", FEATURES)
def test_templates_json_is_valid_and_nonempty(feature: str):
    data = _bundle_data(feature)
    assert isinstance(data, dict) and data
    # Every class entry maps method -> string template.
    for class_name, methods in data.items():
        assert isinstance(methods, dict), class_name
        for method, body in methods.items():
            assert isinstance(body, str), f"{class_name}.{method}"


@pytest.mark.parametrize("feature", FEATURES)
def test_templates_carry_no_threat_language(feature: str):
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
    offenders = [
        f"{class_name}.{method}: {match.group(0)!r}"
        for class_name, methods in _bundle_data(feature).items()
        for method, body in methods.items()
        if (match := threats.search(body))
    ]
    assert not offenders, "Coercive prompt language found in: " + ", ".join(
        offenders
    )


def test_simulator_interrupt_templates_present():
    methods = _bundle_data("simulator")["SimulatorInterruptTemplate"]
    assert "decide_interrupt" in methods
    assert "interruption_bias_rare" in methods
    assert "interruption_bias_normal" in methods
    assert "interruption_bias_frequent" in methods
    assert "interruption_frustration" in methods
