import sys
from typing import TYPE_CHECKING, Any, Callable
from wrapt import wrap_function_wrapper
from langchain_core.callbacks import BaseCallbackManager
from deepeval.integrations.langchain.callback import LangChainCallbackHandler


class _BaseCallbackManagerInit:
    __slots__ = ("_tracer",)

    def __init__(self, tracer: "LangChainCallbackHandler"):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., None],
        instance: "BaseCallbackManager",
        args: Any,
        kwargs: Any,
    ) -> None:
        wrapped(*args, **kwargs)
        for handler in instance.inheritable_handlers:

            if isinstance(handler, type(self._tracer)):
                break
        else:
            instance.add_handler(self._tracer, True)


def trace_langchain():
    wrap_function_wrapper(
        module="langchain_core.callbacks",
        name="BaseCallbackManager.__init__",
        wrapper=_BaseCallbackManagerInit(LangChainCallbackHandler()),
    )
