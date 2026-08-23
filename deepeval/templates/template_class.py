"""Generates the public `<Metric>Template` classes from the compiled bundle.

Metric prompts live as Jinja `.txt` files compiled into
`templates/metrics/templates.json`, which is mirrored to the TypeScript SDK. The
classes built here are thin views over that bundle: each bundled method becomes a
staticmethod that renders the shared template, so users can subclass a template
class and override individual prompts without the prompt text being duplicated
in Python.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Type, cast

from deepeval.templates.resolver import (
    Feature,
    TemplateMethod,
    iter_base_template_methods,
    resolve_template,
)

# Bundle keys owned by DAG nodes rather than metrics. Nodes always pass an
# explicit `template_class` to `_get_prompt`, which bypasses the override hook,
# so a template class for them would be unreachable.
DAG_NODE_TEMPLATE_CLASSES = frozenset(
    {
        "BinaryJudgement",
        "ConversationalBinaryJudgement",
        "ConversationalNonBinaryJudgement",
        "ConversationalTaskNode",
        "ConversationalVerdictNode",
        "NonBinaryJudgement",
        "TaskNode",
        "VerdictNode",
    }
)


def template_class_name(class_name: str) -> str:
    """`AnswerRelevancyMetric` -> `AnswerRelevancyTemplate`, `GEval` -> `GEvalTemplate`."""
    return (class_name.removesuffix("Metric") or class_name) + "Template"


def filter_template_kwargs(
    fn: Callable[..., Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Narrow `kwargs` to the parameters `fn` actually declares.

    Overrides name only the variables they interpolate, so handing them the full
    render context (which always includes `multimodal` and `strict`) would raise
    `TypeError`.
    """
    try:
        parameters = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return dict(kwargs)

    accepted = {
        p.name
        for p in parameters
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in kwargs.items() if k in accepted}


def _make_render_method(
    class_name: str, method: TemplateMethod, *, feature: Feature = "metrics"
) -> staticmethod:
    def render(**kwargs: Any) -> str:
        return resolve_template(feature, class_name, method, **kwargs)

    render.__name__ = method
    render.__qualname__ = f"{template_class_name(class_name)}.{method}"
    render.__doc__ = (
        f"Render the default `{method}` prompt for `{class_name}`.\n\n"
        "Override this in a subclass to supply your own prompt. The override "
        "only needs to declare the template variables it uses."
    )
    return staticmethod(render)


def make_template_class(
    class_name: str, *, feature: Feature = "metrics"
) -> Type[Any]:
    """Build the `<Metric>Template` class for a bundled class key.

    Raises `MetricTemplateNotFoundError` if the key has no bundled templates,
    which keeps a typo from silently producing an empty class.
    """
    name = template_class_name(class_name)
    namespace: Dict[str, Any] = {
        "_template_class": class_name,
        "_template_feature": feature,
        "__doc__": (
            f"Default prompt templates for `{class_name}`.\n\n"
            "Subclass this and override any method below to customize a prompt, "
            f"then pass the subclass as `evaluation_template` to {class_name}."
        ),
    }

    for method, _ in iter_base_template_methods(feature, class_name):
        namespace[method] = _make_render_method(
            class_name, cast(TemplateMethod, method), feature=feature
        )

    return type(name, (), namespace)
