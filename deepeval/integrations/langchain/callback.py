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
    LangChainTraceType,
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
        self.event_map = {}

    def _start_trace(self, run: Run) -> None:
        self.run_map[str(run.id)] = run
        print("#########################")
        print("#########################")
        print(run.id)
        print(run.name)
        print(run.run_type)
        # trace_instance = self.create_trace_instance(
        #     run.id, run.name
        # )
        # self.event_map[run.id] = trace_instance
        # trace_manager.append_to_trace_stack(trace_instance)
    
    def convert_event_type_to_deepeval_trace_type(self, event_type: str):
        # TODO: add more types
        if event_type == "llm":
            return LlamaIndexTraceType.LLM
        elif event_type == "retriever":
            return LlamaIndexTraceType.RETRIEVER
        elif event_type == CBEventType.EMBEDDING:
            return LlamaIndexTraceType.EMBEDDING
        elif event_type == CBEventType.CHUNKING:
            return LlamaIndexTraceType.CHUNKING
        elif event_type == CBEventType.NODE_PARSING:
            return LlamaIndexTraceType.NODE_PARSING
        elif event_type == CBEventType.SYNTHESIZE:
            return LlamaIndexTraceType.SYNTHESIZE
        elif event_type == CBEventType.QUERY:
            return LlamaIndexTraceType.QUERY
        elif event_type == CBEventType.RERANKING:
            return LlamaIndexTraceType.RERANKING
        elif event_type == CBEventType.AGENT_STEP:
            return LlamaIndexTraceType.AGENT_STEP

        return event_type.capitalize()

    def create_trace_instance(
        self,
        event_type: str,
        processed_payload: Optional[Dict[str, Any]] = None,
    ) -> Union[EmbeddingTrace, LlmMetadata, GenericTrace]:

        current_time = perf_counter()
        type = self.convert_event_type_to_deepeval_trace_type(event_type)
        name = event_type
        trace_instance_input = None

        if "exception" in processed_payload:
            trace_instance = GenericTrace(
                traceProvider=TraceProvider.LLAMA_INDEX,
                type=type,
                executionTime=current_time,
                name=name,
                input=trace_instance_input,
                output={"exception": processed_payload["exception"]},
                status=TraceStatus.ERROR,
                traces=[],
            )

        elif event_type == CBEventType.LLM:
            trace_instance = LlmTrace(
                traceProvider=TraceProvider.LLAMA_INDEX,
                type=type,
                executionTime=current_time,
                name=name,
                input=processed_payload["llm_input_messages"],
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
                llmMetadata=LlmMetadata(
                    model=processed_payload["llm_model_name"],
                    output_messages=None,
                    token_count=None,
                    prompt_template=processed_payload.get(
                        "llm_prompt_template"
                    ),
                    prompt_template_variables=processed_payload.get(
                        "llm_prompt_template_variables"
                    ),
                ),
            )

        elif event_type == CBEventType.EMBEDDING:
            trace_instance = EmbeddingTrace(
                traceProvider=TraceProvider.LLAMA_INDEX,
                type=type,
                executionTime=current_time,
                name=name,
                input=trace_instance_input,
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
                embeddingMetadata=EmbeddingMetadata(
                    model=processed_payload["embedding_model_name"],
                ),
            )

        elif event_type == CBEventType.RETRIEVE:
            trace_instance = RetrieverTrace(
                traceProvider=TraceProvider.LLAMA_INDEX,
                type=type,
                executionTime=current_time,
                name=name,
                input=processed_payload["input_value"],
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
                retrieverMetadata=RetrieverMetadata(),
            )

        elif event_type == CBEventType.RERANKING:
            trace_instance = RerankingTrace(
                traceProvider=TraceProvider.LLAMA_INDEX,
                type=type,
                executionTime=current_time,
                name=name,
                input=processed_payload["input_value"],
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
                rerankingMetadata=RerankingMetadata(
                    model=processed_payload["reranker_model_name"],
                    top_k=processed_payload["reranker_top_k"],
                ),
            )

        elif event_type == CBEventType.QUERY:
            trace_instance = GenericTrace(
                traceProvider=TraceProvider.LLAMA_INDEX,
                type=type,
                executionTime=current_time,
                name=name,
                input=processed_payload["input_value"],
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
            )

        elif event_type == CBEventType.SYNTHESIZE:
            trace_instance = GenericTrace(
                traceProvider=TraceProvider.LLAMA_INDEX,
                type=type,
                executionTime=current_time,
                name=name,
                input=processed_payload["input_value"],
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
            )

        else:
            trace_instance = GenericTrace(
                traceProvider=TraceProvider.LLAMA_INDEX,
                type=type,
                executionTime=current_time,
                name=name,
                input=trace_instance_input,
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
            )
        return trace_instance

    # def _end_trace(self, run: Run) -> None:
    #     print("#########################")
    #     print("#########################")
    #     print("#########################")
    #     pass

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