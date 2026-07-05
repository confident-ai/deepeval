from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set, Tuple

import jinja2

Feature = Literal["metrics"]

# Keep in sync with the method keys in `templates/metrics/templates.json`.
# `class_name` stays `str` on purpose: callers pass `self.__class__.__name__`,
# which a type checker can't narrow to a Literal anyway.
MetricTemplateMethod = Literal[
    "detect_role_violations",
    "evaluate_adherence",
    "evaluate_image_coherence",
    "evaluate_image_helpfulness",
    "evaluate_image_reference",
    "evaluate_plan_quality",
    "extract_data",
    "extract_goal_and_outcome",
    "extract_out_of_character_response_verdicts",
    "extract_pii",
    "extract_plan_from_trace",
    "extract_task_and_outcome_from_trace",
    "extract_task_from_trace",
    "extract_user_intentions",
    "generate_advices",
    "generate_alignment_verdicts",
    "generate_answers",
    "generate_arena_winner",
    "generate_binary_verdict",
    "generate_claims",
    "generate_evaluation_results",
    "generate_evaluation_steps",
    "generate_final_reason",
    "generate_misuses",
    "generate_non_binary_verdict",
    "generate_opinions",
    "generate_perceptual_quality_evaluation_results",
    "generate_questions",
    "generate_reason",
    "generate_semantic_consistency_evaluation_results",
    "generate_statements",
    "generate_strict_evaluation_results",
    "generate_task_output",
    "generate_truths",
    "generate_verdict",
    "generate_verdicts",
    "get_accuracy_score",
    "get_args_correctness_score",
    "get_argument_correctness_score",
    "get_execution_efficiency",
    "get_final_reason",
    "get_mcp_argument_correctness_prompt",
    "get_plan_evaluation_score",
    "get_primitive_correctness_prompt",
    "get_qa_pair_verdict",
    "get_qa_pairs",
    "get_task_completion_score",
    "get_tool_argument_final_reason",
    "get_tool_correctness_score",
    "get_tool_selection_final_reason",
    "get_tool_selection_score",
    "rewrite_reason",
]


class MetricTemplateNotFoundError(KeyError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MetricTemplateInterpolationError(ValueError):
    def __init__(self, message: str, unresolved: Set[str]) -> None:
        super().__init__(message)
        self.unresolved = unresolved


class TemplateRegistry:
    """Loads per-feature `templates/<feature>/templates.json` bundles and caches
    the Jinja environments plus compiled templates. Templates are static, so each
    `(feature, class, method, strict)` is compiled at most once."""

    def __init__(self) -> None:
        self._base_templates: Dict[str, Dict[str, Any]] = {}
        self._jinja_strict: Optional[jinja2.Environment] = None
        self._jinja_lenient: Optional[jinja2.Environment] = None
        self._compiled: Dict[Tuple[str, str, str, bool], jinja2.Template] = {}

    def clear(self) -> None:
        self._base_templates.clear()
        self._jinja_strict = None
        self._jinja_lenient = None
        self._compiled.clear()

    def get_base_templates(self, feature: str) -> Dict[str, Any]:
        if feature not in self._base_templates:
            try:
                ref = resources.files("deepeval.templates").joinpath(
                    feature, "templates.json"
                )
                raw = ref.read_text(encoding="utf-8")
            except (ModuleNotFoundError, OSError, TypeError, FileNotFoundError):
                raw = (
                    Path(__file__).resolve().parent / feature / "templates.json"
                ).read_text(encoding="utf-8")
            self._base_templates[feature] = json.loads(raw)
        return self._base_templates[feature]

    def get_jinja_env(self, strict: bool) -> jinja2.Environment:
        if strict:
            if self._jinja_strict is None:
                self._jinja_strict = jinja2.Environment(
                    undefined=jinja2.StrictUndefined
                )
            return self._jinja_strict
        if self._jinja_lenient is None:
            self._jinja_lenient = jinja2.Environment()
        return self._jinja_lenient

    def get_compiled(
        self, feature: str, class_name: str, method: str, strict: bool
    ) -> jinja2.Template:
        key = (feature, class_name, method, strict)
        template = self._compiled.get(key)
        if template is None:
            raw = get_raw_template(feature, class_name, method)
            template = self.get_jinja_env(strict).from_string(raw)
            self._compiled[key] = template
        return template


_registry = TemplateRegistry()


def clear_metric_template_cache() -> None:
    _registry.clear()


def get_raw_template(
    feature: Feature, class_name: str, method: MetricTemplateMethod
) -> str:
    """Return the raw (un-rendered) base template string for a class/method."""
    base = _registry.get_base_templates(feature)
    entry = base.get(class_name, {})
    body = entry.get(method)

    if not body or not isinstance(body, str):
        raise MetricTemplateNotFoundError(
            f"No template found for {feature!r}.{class_name!r}.{method!r}. "
            f"Available classes: {', '.join(base.keys())}"
        )
    return body


def iter_base_template_methods(
    feature: Feature, class_name: str
) -> list[tuple[str, str]]:
    """Iterate over all `(method, template)` pairs for a class in the bundle."""
    base = _registry.get_base_templates(feature)
    entry = base.get(class_name)

    if not entry or not isinstance(entry, dict):
        raise MetricTemplateNotFoundError(
            f"No templates for class {class_name!r} in feature {feature!r}."
        )

    pairs = [
        (k, v)
        for k, v in sorted(entry.items())
        if not k.startswith("_") and isinstance(v, str)
    ]
    if not pairs:
        raise MetricTemplateNotFoundError(
            f"No string template methods for class {class_name!r}."
        )
    return pairs


def resolve_template(
    feature: Feature,
    class_name: str,
    method: MetricTemplateMethod,
    *,
    multimodal: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> str:
    """Render a template to a final prompt via Jinja2.

    `feature` selects the `templates/<feature>/templates.json` bundle (e.g.
    "metrics"). `_fragments` (shared reusable snippets) and `multimodal` are
    always available to the template; everything else is passed via `kwargs`.
    """
    fragments = _registry.get_base_templates(feature).get("_fragments", {})
    try:
        template = _registry.get_compiled(feature, class_name, method, strict)
        return template.render(
            multimodal=multimodal,
            _fragments=fragments,
            **kwargs,
        )
    except jinja2.UndefinedError as e:
        raise MetricTemplateInterpolationError(
            f"Missing variable during template render: {e.message}",
            unresolved=set(),
        ) from e
    except jinja2.TemplateSyntaxError as e:
        raise MetricTemplateInterpolationError(
            f"Jinja syntax error in template: {e.message}",
            unresolved=set(),
        ) from e
