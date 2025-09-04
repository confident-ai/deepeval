from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from deepeval.tracing import observe

try:
    from agents import function_tool as _agents_function_tool  # type: ignore
    from agents.function_schema import function_schema as _agents_function_schema  # type: ignore
except Exception:
    _agents_function_tool = None  # type: ignore
    _agents_function_schema = None  # type: ignore


def _compute_description(
    the_func: Callable[..., Any],
    *,
    name_override: Optional[str],
    description_override: Optional[str],
    docstring_style: Optional[str],
    use_docstring_info: Optional[bool],
    strict_mode: Optional[bool],
) -> Optional[str]:
    if _agents_function_schema is None:
        return None
    schema = _agents_function_schema(
        func=the_func,
        name_override=name_override,
        description_override=description_override,
        docstring_style=docstring_style,
        use_docstring_info=(
            use_docstring_info if use_docstring_info is not None else True
        ),
        strict_json_schema=strict_mode if strict_mode is not None else True,
    )
    return schema.description


def _wrap_with_observe(
    func: Callable[..., Any],
    metrics: Optional[str] = None,
    metric_collection: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[..., Any]:
    if getattr(func, "_is_deepeval_observed", False):
        return func

    observed = observe(
        metrics=metrics,
        metric_collection=metric_collection,
        description=description,
        type="tool",
    )(func)

    try:
        observed.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    except Exception:
        pass
    return observed


def function_tool(
    func: Optional[Callable[..., Any]] = None, /, *args: Any, **kwargs: Any
) -> Any:
    metrics = kwargs.pop("metrics", None)
    metric_collection = kwargs.pop("metric_collection", None)

    if _agents_function_tool is None:
        raise RuntimeError(
            "agents.function_tool is not available. Please install agents via your package manager"
        )

    # Peek decorator options to mirror description logic
    name_override = kwargs.get("name_override")
    description_override = kwargs.get("description_override")
    docstring_style = kwargs.get("docstring_style")
    use_docstring_info = kwargs.get("use_docstring_info")
    strict_mode = kwargs.get("strict_mode")

    if callable(func):
        description = _compute_description(
            func,
            name_override=name_override,
            description_override=description_override,
            docstring_style=docstring_style,
            use_docstring_info=use_docstring_info,
            strict_mode=strict_mode,
        )
        wrapped = _wrap_with_observe(
            func,
            metrics=metrics,
            metric_collection=metric_collection,
            description=description,
        )
        return _agents_function_tool(wrapped, *args, **kwargs)

    def decorator(real_func: Callable[..., Any]) -> Any:
        description = _compute_description(
            real_func,
            name_override=name_override,
            description_override=description_override,
            docstring_style=docstring_style,
            use_docstring_info=use_docstring_info,
            strict_mode=strict_mode,
        )
        wrapped = _wrap_with_observe(
            real_func,
            metrics=metrics,
            metric_collection=metric_collection,
            description=description,
        )
        return _agents_function_tool(wrapped, *args, **kwargs)

    return decorator
