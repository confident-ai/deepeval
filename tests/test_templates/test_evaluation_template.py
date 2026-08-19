"""Covers the `evaluation_template` override on metrics.

Prompt text lives in the shared Jinja bundle, and each metric ships a generated
`<Metric>Template` class whose methods render it. Passing a subclass as
`evaluation_template` replaces individual prompts. These tests pin the override
precedence, the kwarg tolerance the documented examples rely on, and the naming
rule that keeps a newly added metric from shipping without a template class.
"""

import importlib
import inspect
import json
from collections import defaultdict
from pathlib import Path

import pytest

from deepeval.metrics.base_metric import PromptMixin
from deepeval.templates import (
    DAG_NODE_TEMPLATE_CLASSES,
    clear_metric_template_cache,
    make_template_class,
    resolve_template,
    template_class_name,
)
from deepeval.templates.resolver import MetricTemplateNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = json.loads(
    (
        REPO_ROOT / "deepeval" / "templates" / "metrics" / "templates.json"
    ).read_text(encoding="utf-8")
)
BUNDLE_KEYS = sorted(k for k in BUNDLE if not k.startswith("_"))
METRIC_KEYS = [k for k in BUNDLE_KEYS if k not in DAG_NODE_TEMPLATE_CLASSES]


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_metric_template_cache()
    yield
    clear_metric_template_cache()


def stub(class_name, evaluation_template=None):
    """A bare `PromptMixin` whose class name is a bundle key.

    Lets us exercise `_get_prompt` without constructing a metric, which would
    require an LLM provider.
    """
    namespace = {}
    if evaluation_template is not None:
        namespace["evaluation_template"] = evaluation_template
    return type(class_name, (PromptMixin,), namespace)()


class TestDefaultRendering:
    def test_generated_method_matches_resolve_template(self):
        template = make_template_class("AnswerRelevancyMetric")
        assert template.generate_statements(
            actual_output="hello"
        ) == resolve_template(
            "metrics",
            "AnswerRelevancyMetric",
            "generate_statements",
            actual_output="hello",
        )

    def test_get_prompt_default_matches_resolve_template(self):
        metric = stub("AnswerRelevancyMetric")
        assert metric._get_prompt(
            "generate_statements", actual_output="hello"
        ) == resolve_template(
            "metrics",
            "AnswerRelevancyMetric",
            "generate_statements",
            actual_output="hello",
        )

    def test_multimodal_flag_reaches_the_template(self):
        metric = stub("AnswerRelevancyMetric")
        plain = metric._get_prompt("generate_statements", actual_output="x")
        multimodal = metric._get_prompt(
            "generate_statements", actual_output="x", multimodal=True
        )
        assert plain != multimodal

    def test_unknown_class_key_raises(self):
        with pytest.raises(MetricTemplateNotFoundError):
            make_template_class("NoSuchMetric")


class TestOverride:
    def test_override_replaces_the_prompt(self):
        base = make_template_class("AnswerRelevancyMetric")

        class Custom(base):
            @staticmethod
            def generate_statements(
                actual_output: str, multimodal: bool = False
            ):
                return f"CUSTOM {actual_output}"

        metric = stub("AnswerRelevancyMetric", Custom)
        assert (
            metric._get_prompt("generate_statements", actual_output="hi")
            == "CUSTOM hi"
        )

    def test_unoverridden_methods_still_use_the_bundle(self):
        base = make_template_class("AnswerRelevancyMetric")

        class Custom(base):
            @staticmethod
            def generate_statements(
                actual_output: str, multimodal: bool = False
            ):
                return "CUSTOM"

        metric = stub("AnswerRelevancyMetric", Custom)
        assert metric._get_prompt(
            "generate_verdicts", input="i", statements="s"
        ) == resolve_template(
            "metrics",
            "AnswerRelevancyMetric",
            "generate_verdicts",
            input="i",
            statements="s",
        )

    def test_documented_example_signature_is_accepted(self):
        """The docs override omits `multimodal`; passing it would be a TypeError.

        `_get_prompt` always supplies `multimodal` and `strict`, so the call has
        to be narrowed to the parameters the override actually declares.
        """
        base = make_template_class("AnswerRelevancyMetric")

        class Custom(base):
            @staticmethod
            def generate_statements(actual_output: str):
                return f"ONLY {actual_output}"

        metric = stub("AnswerRelevancyMetric", Custom)
        assert (
            metric._get_prompt(
                "generate_statements", actual_output="x", multimodal=True
            )
            == "ONLY x"
        )

    def test_override_declaring_var_keyword_receives_everything(self):
        base = make_template_class("AnswerRelevancyMetric")
        seen = {}

        class Custom(base):
            @staticmethod
            def generate_statements(**kwargs):
                seen.update(kwargs)
                return "ok"

        stub("AnswerRelevancyMetric", Custom)._get_prompt(
            "generate_statements", actual_output="x"
        )
        assert seen["actual_output"] == "x"
        assert seen["multimodal"] is False
        assert seen["strict"] is True

    def test_explicit_template_class_bypasses_the_override(self):
        """Borrowing another class's template must not hit this metric's override.

        `SummarizationMetric` renders `generate_truths` from
        `FaithfulnessMetric`'s bundled template.
        """
        base = make_template_class("SummarizationMetric")

        class Custom(base):
            @staticmethod
            def generate_truths(**kwargs):
                return "HIJACKED"

        metric = stub("SummarizationMetric", Custom)
        rendered = metric._get_prompt(
            "generate_truths",
            template_class="FaithfulnessMetric",
            strict=False,
            retrieval_context="ctx",
        )
        assert rendered != "HIJACKED"
        assert rendered == resolve_template(
            "metrics",
            "FaithfulnessMetric",
            "generate_truths",
            strict=False,
            retrieval_context="ctx",
        )


class TestNamingRule:
    def test_rule_is_injective_across_the_bundle(self):
        collisions = defaultdict(list)
        for key in BUNDLE_KEYS:
            collisions[template_class_name(key)].append(key)
        assert {n: k for n, k in collisions.items() if len(k) > 1} == {}

    def test_every_bundle_key_is_a_metric_or_a_known_dag_node(self):
        """Keeps the DAG node exclusion list from silently absorbing new keys."""
        assert DAG_NODE_TEMPLATE_CLASSES <= set(BUNDLE_KEYS)
        assert set(BUNDLE_KEYS) == set(METRIC_KEYS) | set(
            DAG_NODE_TEMPLATE_CLASSES
        )

    @pytest.mark.parametrize("key", METRIC_KEYS)
    def test_metric_exports_its_template_class(self, key):
        import deepeval.metrics

        metric_cls = getattr(deepeval.metrics, key)
        package = importlib.import_module(
            metric_cls.__module__.rsplit(".", 1)[0]
        )
        name = template_class_name(key)

        template = getattr(package, name)
        assert template.__name__ == name
        assert template._template_class == key
        assert name in package.__all__

    @pytest.mark.parametrize("key", METRIC_KEYS)
    def test_metric_defaults_evaluation_template_to_its_own_shim(self, key):
        import deepeval.metrics

        parameters = inspect.signature(
            getattr(deepeval.metrics, key).__init__
        ).parameters
        default = parameters["evaluation_template"].default
        assert default.__name__ == template_class_name(key)
        assert default._template_class == key

    @pytest.mark.parametrize("key", METRIC_KEYS)
    def test_every_bundled_method_is_exposed(self, key):
        template = make_template_class(key)
        for method in BUNDLE[key]:
            assert callable(getattr(template, method))


class TestMetricIntegration:
    @pytest.fixture(autouse=True)
    def _fake_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")

    def test_subclassed_metric_still_resolves_its_prompts(self):
        """`self.__class__.__name__` is not a bundle key for a subclass.

        The shim closes over the literal key, so the lookup no longer depends on
        the runtime class name.
        """
        from deepeval.metrics import AnswerRelevancyMetric

        class MyAnswerRelevancy(AnswerRelevancyMetric):
            pass

        assert MyAnswerRelevancy()._get_prompt(
            "generate_statements", actual_output="x"
        ) == resolve_template(
            "metrics",
            "AnswerRelevancyMetric",
            "generate_statements",
            actual_output="x",
        )

    def test_copy_metrics_preserves_a_custom_template(self):
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate
        from deepeval.metrics.utils import copy_metrics

        class Custom(AnswerRelevancyTemplate):
            @staticmethod
            def generate_statements(actual_output: str):
                return "CUSTOM"

        metric = AnswerRelevancyMetric(evaluation_template=Custom)
        assert copy_metrics([metric])[0].evaluation_template is Custom

    def test_metric_uses_the_override_end_to_end(self):
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate

        class Custom(AnswerRelevancyTemplate):
            @staticmethod
            def generate_statements(actual_output: str):
                return f"CUSTOM {actual_output}"

        metric = AnswerRelevancyMetric(evaluation_template=Custom)
        assert (
            metric._get_prompt("generate_statements", actual_output="y")
            == "CUSTOM y"
        )
