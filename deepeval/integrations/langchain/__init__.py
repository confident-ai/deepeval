# from typing import Any, Callable
# from langchain_core.callbacks import BaseCallbackManager
# from deepeval.integrations.langchain.callback import LangChainCallbackHandler


# class _BaseCallbackManagerInit:
#     __slots__ = ("_tracer",)

#     def __init__(self, tracer: "LangChainCallbackHandler"):
#         self._tracer = tracer

#     def __call__(
#         self,
#         wrapped: Callable[..., None],
#         instance: "BaseCallbackManager",
#         args: Any,
#         kwargs: Any,
#     ) -> None:
#         wrapped(*args, **kwargs)
#         for handler in instance.inheritable_handlers:

#             if isinstance(handler, type(self._tracer)):
#                 break
#         else:
#             instance.add_handler(self._tracer, True)
