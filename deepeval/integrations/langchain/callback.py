from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)
from time import perf_counter
from uuid import UUID
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from threading import RLock

from deepeval.tracing import (
    trace_manager,
    get_trace_stack,
    BaseTrace,
    LlmTrace,
    GenericTrace,
    EmbeddingTrace,
    RerankingTrace,
    RetrieverTrace,
    TraceStatus,
    LlmMetadata,
    EmbeddingMetadata,
    RerankingMetadata,
    RetrieverMetadata,
    TraceType,
    TraceProvider,
    LlamaIndexTraceType,
)
from deepeval.utils import dataclass_to_dict
from wrapt import ObjectProxy
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

class _DictWithLock(ObjectProxy, Generic[K, V]):  # type: ignore
    """
    A wrapped dictionary with lock
    """

    def __init__(self, wrapped: Optional[Dict[str, V]] = None) -> None:
        super().__init__(wrapped or {})
        self._self_lock = RLock()

    def get(self, key: K) -> Optional[V]:
        with self._self_lock:
            return cast(Optional[V], self.__wrapped__.get(key))

    def pop(self, key: K, *args: Any) -> Optional[V]:
        with self._self_lock:
            return cast(Optional[V], self.__wrapped__.pop(key, *args))

    def __getitem__(self, key: K) -> V:
        with self._self_lock:
            return cast(V, super().__getitem__(key))

    def __setitem__(self, key: K, value: V) -> None:
        with self._self_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key: K) -> None:
        with self._self_lock:
            super().__delitem__(key)

class LangChainCallbackHandler(BaseTracer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # print("hhihihih")
        # self.run_map = _DictWithLock[str, Run](self.run_map)
        # self._tracer = None
        # self._spans_by_run: None
        # self._lock = RLock()  # handlers may be run in a thread by langchain
        # print("badsfsadfa")

    # def _start_trace(self, run: Run) -> None:
    #     return

    def _end_trace(self, run: Run) -> None:
        print("#########################")
        print(run)
        print("#########################")
        pass

    def _persist_run(self, run: Run) -> None:
        print("#########################")
        pass

    def on_llm_error(self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any) -> Run:
        print("#########################")
        pass

    def on_chain_error(self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any) -> Run:
        print("#########################")
        pass

    def on_retriever_error(
        self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any
    ) -> Run:
        print("#########################")
        pass

    def on_tool_error(self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any) -> Run:
        print("#########################")
        pass

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> Run:
        print("#########################")
        pass