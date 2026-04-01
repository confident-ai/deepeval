import inspect
from typing import Optional


def observe_methods(cls, span_type: Optional[str] = None):
    from deepeval.tracing.tracing import observe

    is_traceable = lambda v: inspect.isfunction(
        v
    ) or inspect.iscoroutinefunction(v)
    methods = {
        k: v
        for k, v in cls.__dict__.items()
        if not k.startswith("__") and is_traceable(v)
    }

    for name, method in methods.items():
        setattr(
            cls,
            name,
            observe(
                type=span_type,
                _drop_if_root=True,
                _internal=True,
            )(method),
        )
