import inspect
from typing import List, Optional


def observe_methods(
    cls,
    span_type: Optional[str] = None,
    allowed_methods: Optional[List[str]] = None,
):
    from deepeval.tracing.tracing import observe

    is_traceable = lambda v: inspect.isfunction(
        v
    ) or inspect.iscoroutinefunction(v)

    methods = {
        k: v
        for k, v in cls.__dict__.items()
        if not k.startswith("__") and is_traceable(v)
    }

    if allowed_methods is not None:
        methods = {k: v for k, v in methods.items() if k in allowed_methods}

    for name, method in methods.items():
        if getattr(method, "_is_deepeval_observed", False):
            continue
        setattr(
            cls,
            name,
            observe(
                type=span_type,
                _drop_if_root=True,
                _internal=True,
            )(method),
        )
