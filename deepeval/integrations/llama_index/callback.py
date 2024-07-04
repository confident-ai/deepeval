from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Iterable,
    Iterator,
    Tuple,
    Mapping,
    cast,
)
from time import perf_counter
import json
import traceback

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import BaseNode

# Kritin
from llama_index.core import Response
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import BASE_TRACE_EVENT
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.tools import ToolMetadata

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
    LlmAttributes,
    EmbeddingAttributes,
    RerankingAttributes,
    RetrieverAttributes,
    TraceType,
    TraceProvider,
    LlamaIndexTraceType,
    RetrievalNode,
    QueryTrace,
    QueryAttributes,
    SynthesizeAttributes,
    SynthesizeTrace,
    GenericAttributes,
    AgentAttributes,
    AgentTrace,
    TraceData,
)
from deepeval.utils import dataclass_to_dict

events_to_ignore = [
    # CBEventType.CHUNKING,
    # CBEventType.NODE_PARSING,
    # CBEventType.EMBEDDING,
    # CBEventType.LLM,
    # CBEventType.QUERY,
    # CBEventType.RETRIEVE,
    # CBEventType.SYNTHESIZE,
    CBEventType.TREE,
    CBEventType.SUB_QUESTION,
    # CBEventType.TEMPLATING,
    CBEventType.FUNCTION_CALL,
    # CBEventType.RERANKING,
    # CBEventType.EXCEPTION,
    # CBEventType.AGENT_STEP,
]


class LlamaIndexCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.event_map = {}
        self._templating_parent_id = {}
        self._templating_payloads = {}
        super().__init__(
            event_starts_to_ignore=events_to_ignore,
            event_ends_to_ignore=events_to_ignore,
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self.event_map = {}
        # trace_manager.clear_trace_stack()
        return

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        return

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        processed_payload = self.process_payload(
            event_type, event_id, parent_id, payload, True
        )
        trace_instance = self.create_trace_instance(
            event_type, processed_payload
        )
        self.event_map[event_id] = trace_instance
        trace_manager.append_to_trace_stack(trace_instance)

        return

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> None:

        trace_instance = self.event_map[event_id]
        processed_payload = self.process_payload(
            event_type, event_id, parent_id, payload, True
        )
        trace_instance = self.update_trace_instance(
            trace_instance, event_type, processed_payload
        )
        current_trace_stack = trace_manager.get_trace_stack_copy()
        # trace_instance = current_trace_stack[-1]

        if len(current_trace_stack) > 1:
            parent_trace = current_trace_stack[-2]
            parent_trace.traces.append(trace_instance)
            trace_manager.set_trace_stack(current_trace_stack)

        if len(current_trace_stack) == 1:
            dict_representation = dataclass_to_dict(current_trace_stack[0])
            trace_manager.set_dict_trace_stack(dict_representation)
            trace_manager.clear_trace_stack()
        else:
            trace_manager.pop_trace_stack()

        return

    def create_trace_instance(
        self,
        event_type: CBEventType,
        processed_payload: Optional[Dict[str, Any]] = None,
    ) -> TraceData:

        trace_kwargs = {
            "traceProvider": TraceProvider.LLAMA_INDEX,
            "type": self.convert_event_type_to_deepeval_trace_type(event_type),
            "executionTime": perf_counter(),
            "name": event_type,
            "status": TraceStatus.SUCCESS,
            "traces": [],
        }

        if "exception" in processed_payload:
            trace_kwargs["status"] = TraceStatus.ERROR
            trace_instance = GenericTrace(
                **trace_kwargs,
            )

        ### Different Attributes ###############################

        elif event_type == CBEventType.AGENT_STEP:
            trace_instance = AgentTrace(
                **trace_kwargs,
                embeddingAttributes=AgentAttributes(
                    input="", output="", name="", description=""
                ),
            )

        elif event_type == CBEventType.EMBEDDING:
            trace_instance = EmbeddingTrace(
                **trace_kwargs,
                embeddingAttributes=EmbeddingAttributes(
                    embedding_text="",
                    # Optional variables
                    model=processed_payload["embedding_model_name"],
                    embedding_length=None,
                ),
            )

        elif event_type == CBEventType.LLM:
            messages = processed_payload.get("llm_input_messages")
            trace_instance = LlmTrace(
                **trace_kwargs,
                llmAttributes=LlmAttributes(
                    input_str=next(
                        m["message_content"]
                        for m in messages
                        if m["message_role"] == "user"
                    ),
                    output_str="",
                    # Optional variables
                    model=processed_payload["llm_model_name"],
                    total_token_count=None,
                    prompt_token_count=None,
                    completion_token_count=None,
                    prompt_template=processed_payload.get(
                        "llm_prompt_template"
                    ),
                    prompt_template_variables=processed_payload.get(
                        "llm_prompt_template_variables"
                    ),
                ),
            )

        elif event_type == CBEventType.RERANKING:
            trace_instance = RerankingTrace(
                **trace_kwargs,
                rerankingAttributes=RerankingAttributes(
                    input_nodes=[],
                    output_nodes=[],
                    # Optional variables
                    model=processed_payload["reranker_model_name"],
                    top_n=processed_payload["reranker_top_k"],
                    batch_size=None,
                    query_str=None,
                ),
            )

        elif event_type == CBEventType.RETRIEVE:
            trace_instance = RetrieverTrace(
                **trace_kwargs,
                retrieverAttributes=RetrieverAttributes(
                    query_str=processed_payload["input_value"],
                    nodes=[],
                    # Optional variables
                    top_k=None,
                    average_chunk_size=None,
                    top_score=None,
                    similarity_scorer=None,
                ),
            )

        elif event_type == CBEventType.QUERY:
            trace_instance = QueryTrace(
                **trace_kwargs,
                queryAttributes=QueryAttributes(
                    input=processed_payload["input_value"], output=""
                ),
            )

        elif event_type == CBEventType.SYNTHESIZE:
            trace_instance = SynthesizeTrace(
                **trace_kwargs,
                synthesizeAttributes=SynthesizeAttributes(
                    user_query=processed_payload["input_value"],
                    response="",
                    # Optional variables
                    retrieved_context=None,
                ),
            )

        else:
            trace_instance = GenericTrace(
                **trace_kwargs, genericAttributes=None
            )

        return trace_instance

    def update_trace_instance(
        self,
        trace_instance: BaseTrace,
        event_type: CBEventType,
        processed_payload: Optional[Dict[str, Any]] = None,
    ) -> TraceData:

        trace_instance.executionTime = (
            perf_counter() - trace_instance.executionTime
        )

        if "exception" in processed_payload:
            trace_instance.status = TraceStatus.ERROR

        elif event_type == CBEventType.LLM and isinstance(
            trace_instance, LlmTrace
        ):
            attributes = trace_instance.llmAttributes
            attributes.output_str = processed_payload["output_value"]
            attributes.total_token_count = processed_payload[
                "llm_token_count_total"
            ]
            attributes.prompt_token_count = processed_payload[
                "llm_token_prompt_count"
            ]
            attributes.completion_token_count = processed_payload[
                "llm_token_count_completion"
            ]

        elif event_type == CBEventType.EMBEDDING and isinstance(
            trace_instance, EmbeddingTrace
        ):
            attributes = trace_instance.embeddingAttributes
            embedding = processed_payload["embeddings"][0]
            attributes.embedding_text = embedding["embedding_text"]
            attributes.embedding_length = len(embedding["embedding_vector"])

        elif event_type == CBEventType.RETRIEVE and isinstance(
            trace_instance, RetrieverTrace
        ):
            attributes = trace_instance.retrieverAttributes
            total_chunk_length = 0
            top_score = 0
            nodes: List[RetrievalNode] = processed_payload[
                "retrieval_documents"
            ]
            for node in nodes:
                total_chunk_length += len(node.content)
                top_score = node.score if node.score > top_score else top_score
            attributes.nodes = nodes
            attributes.top_k = len(nodes)
            attributes.average_chunk_size = total_chunk_length // len(nodes)
            attributes.top_score = top_score

        elif event_type == CBEventType.QUERY and isinstance(
            trace_instance, QueryTrace
        ):
            attributes = trace_instance.queryAttributes
            attributes.output = processed_payload["output_value"]

        elif event_type == CBEventType.SYNTHESIZE and isinstance(
            trace_instance, SynthesizeTrace
        ):
            attributes = trace_instance.synthesizeAttributes
            attributes.response = processed_payload["output_value"]

        return trace_instance

    def convert_event_type_to_deepeval_trace_type(
        self, event_type: CBEventType
    ):
        # TODO: add more types
        if event_type == CBEventType.LLM:
            return LlamaIndexTraceType.LLM
        elif event_type == CBEventType.RETRIEVE:
            return LlamaIndexTraceType.RETRIEVER
        elif event_type == CBEventType.EMBEDDING:
            return LlamaIndexTraceType.EMBEDDING
        elif event_type == CBEventType.SYNTHESIZE:
            return LlamaIndexTraceType.SYNTHESIZE
        elif event_type == CBEventType.QUERY:
            return LlamaIndexTraceType.QUERY
        elif event_type == CBEventType.RERANKING:
            return LlamaIndexTraceType.RERANKING
        elif event_type == CBEventType.AGENT_STEP:
            return LlamaIndexTraceType.AGENT

        return event_type.value.capitalize()

    def process_payload(
        self,
        event_type: CBEventType,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        is_event_end: bool = True,
    ):
        attributes = {}
        if payload == None:
            return attributes

        exception = payload.get(EventPayload.EXCEPTION)
        if exception:
            attributes["exception"] = exception

        #########################
        # ignore these events
        if event_type in (CBEventType.NODE_PARSING, CBEventType.CHUNKING):
            return attributes

        #########################
        # process embedding vectors
        chunks = payload.get(EventPayload.CHUNKS)
        embeddings = payload.get(EventPayload.EMBEDDINGS)
        if chunks and embeddings:
            attributes["embeddings"] = [
                {"embedding_text": text, "embedding_vector": vector}
                for text, vector in zip(chunks, embeddings)
            ]

        #########################
        # process retrieval documents
        nodes = payload.get(EventPayload.NODES)
        if event_type is not CBEventType.RERANKING and nodes:
            attributes["retrieval_documents"] = self.process_nodes(nodes)

        #########################
        # process input (query string, prompts, messages)
        query_str = payload.get(EventPayload.QUERY_STR)
        prompt = payload.get(EventPayload.PROMPT)
        messages = payload.get(EventPayload.MESSAGES)

        if event_type is not CBEventType.RERANKING and query_str:
            attributes["input_value"] = query_str
        if prompt:
            attributes["llm_prompts"] = [prompt]
        if messages:
            if event_type is CBEventType.LLM:
                llm_msgs = [self.process_message(m) for m in messages]
                attributes["llm_input_messages"] = llm_msgs
            elif event_type is CBEventType.AGENT_STEP and len(messages):
                msg = messages[0]
                str_msg = (
                    msg.content if isinstance(msg, ChatMessage) else str(msg)
                )
                attributes["input_value"] = str_msg

        #########################
        # process response (still need to process token)
        response = payload.get(EventPayload.RESPONSE) or payload.get(
            EventPayload.COMPLETION
        )

        if response:
            attributes.update(self._get_response_output(response))
            if hasattr(response, "raw"):
                raw = response.raw
                assert isinstance(
                    raw, Mapping
                ), f"raw must be Mapping, found {type(raw)}"
                attributes.update(self._get_output_messages(raw))
                if "usage" in raw:
                    attributes.update(self._get_token_counts(raw["usage"]))

            if (
                additional_kwargs := getattr(
                    response, "additional_kwargs", None
                )
            ) is not None:
                attributes.update(self._get_token_counts(additional_kwargs))

        ###########################
        # process reranking
        reranker_top_k = payload.get(EventPayload.TOP_K)
        reranker_model_name = payload.get(EventPayload.MODEL_NAME)
        query_str = payload.get(EventPayload.QUERY_STR)
        nodes = payload.get(EventPayload.NODES)

        if event_type is CBEventType.RERANKING:
            if reranker_top_k:
                attributes["reranker_top_k"] = reranker_top_k
            if reranker_model_name:
                attributes["reranker_model_name"] = reranker_model_name
            if query_str:
                attributes["reranker_query"] = query_str
            if nodes:
                processed_nodes = self.process_nodes(nodes)
                if is_event_end:
                    attributes["reranker_output_documents"] = processed_nodes
                else:
                    attributes["reranker_input_documents"] = processed_nodes

        ###########################
        # process tool
        tool = payload.get(EventPayload.TOOL)
        tool_parameters = None

        if tool:
            tool_metadata = cast(ToolMetadata, tool)
            tool_parameters = tool_metadata.to_openai_tool()["function"][
                "parameters"
            ]
            attributes["tool_name"] = tool_metadata.name
            attributes["tool_description"] = tool_metadata.description

        if tool_parameters:
            attributes["tool_parameters"] = json.dumps(tool_parameters)

        ###########################
        # process tool
        serialized = payload.get(EventPayload.SERIALIZED)
        if serialized:
            if event_type is CBEventType.EMBEDDING:
                if model_name := serialized.get("model_name"):
                    attributes["embedding_model_name"] = model_name
            if event_type is CBEventType.LLM:
                if model_name := serialized.get("model"):
                    attributes["llm_model_name"] = model_name
                    invocation_parameters = self._extract_invocation_parameters(
                        serialized
                    )
                    invocation_parameters["model"] = model_name
                    attributes["llm_hyperparameters"] = json.dumps(
                        invocation_parameters
                    )

        ###########################
        # process templates
        payloads = [payload.copy()]
        if is_event_end == False:
            if event_type is CBEventType.TEMPLATING:
                self._templating_parent_id[event_id] = parent_id
                if payloads:
                    if parent_id in self._templating_payloads:
                        self._templating_payloads[parent_id].extend(payloads)
                    else:
                        self._templating_payloads[parent_id] = payloads

            if event_type is CBEventType.LLM:
                for templating_payload in self._templating_payloads.pop(
                    parent_id, ()
                ):
                    attributes.update(
                        self._template_attributes(templating_payload)
                    )
        else:
            if event_type is CBEventType.TEMPLATING:
                if (
                    parent_id := self._templating_parent_id.pop(event_id, None)
                ) and payload:
                    if parent_id in self._templating_payloads:
                        self._templating_payloads[parent_id].append(payload)
                    else:
                        self._templating_payloads[parent_id] = [payload]

        return attributes

    ##################################################
    ########### additional helper functions ##########
    ##################################################

    def _get_response_output(self, response: Any) -> Iterator[Tuple[str, Any]]:
        if isinstance(response, ChatResponse):
            message = response.message
            if content := message.content:
                yield "output_value", content
            else:
                yield "output_value", json.dumps(
                    message.additional_kwargs, cls=_CustomJSONEncoder
                )
        elif isinstance(response, Response):
            if response.response:
                yield "output_value", response.response
        elif isinstance(response, StreamingResponse):
            if response_txt := getattr(response, "response_txt", None):
                yield "output_value", response_txt
        else:  # if the response has unknown type, make a best-effort attempt to get the output
            yield "output_value", str(response)

    def _get_token_counts(
        self, usage: Union[object, Mapping[str, Any]]
    ) -> Iterator[Tuple[str, Any]]:
        if isinstance(usage, Mapping):
            yield from self._get_token_counts_from_mapping(usage)
        elif isinstance(usage, object):
            yield from self._get_token_counts_from_object(usage)

    def _template_attributes(
        self, payload: Dict[str, Any]
    ) -> Iterator[Tuple[str, Any]]:
        if template := payload.get(EventPayload.TEMPLATE):
            yield "llm_prompt_template", template
        if template_vars := payload.get(EventPayload.TEMPLATE_VARS):
            yield "llm_prompt_template_variables", json.dumps(template_vars)

    def _get_token_counts_from_object(
        self, usage: object
    ) -> Iterator[Tuple[str, Any]]:
        if (prompt_tokens := getattr(usage, "prompt_tokens", None)) is not None:
            yield "llm_token_prompt_count", prompt_tokens
        if (
            completion_tokens := getattr(usage, "completion_tokens", None)
        ) is not None:
            yield "llm_token_count_completion", completion_tokens
        if (total_tokens := getattr(usage, "total_tokens", None)) is not None:
            yield "llm_token_count_total", total_tokens

    def _get_token_counts_from_mapping(
        self,
        usage_mapping: Mapping[str, Any],
    ) -> Iterator[Tuple[str, Any]]:
        if (prompt_tokens := usage_mapping.get("prompt_tokens")) is not None:
            yield "llm_token_prompt_count", prompt_tokens
        if (
            completion_tokens := usage_mapping.get("completion_tokens")
        ) is not None:
            yield "llm_token_count_completion", completion_tokens
        if (total_tokens := usage_mapping.get("total_tokens")) is not None:
            yield "llm_token_count_total", total_tokens

    def process_nodes(self, nodes) -> Dict[str, Optional[str]]:
        processed_nodes = [
            RetrievalNode(
                content=node_with_score.node.text,
                id=node_with_score.node.node_id,
                score=node_with_score.score,
            )
            for node_with_score in nodes
        ]
        return processed_nodes

    def process_message(self, message: Any) -> Dict[str, Optional[str]]:
        if isinstance(message, ChatMessage):
            message_attributes = {
                "message_role": message.role.value,
                "message_content": message.content,
            }
            # Parse the kwargs to extract the function name and parameters for function calling
            # NB: these additional kwargs exist both for 'agent' and 'function' roles
            if "name" in message.additional_kwargs:
                message_attributes["message_name"] = message.additional_kwargs[
                    "name"
                ]
            if tool_calls := message.additional_kwargs.get("tool_calls"):
                assert isinstance(
                    tool_calls, Iterable
                ), f"tool_calls must be Iterable, found {type(tool_calls)}"
                message_tool_calls = []
                for tool_call in tool_calls:
                    if message_tool_call := dict(
                        self._get_tool_call(tool_call)
                    ):
                        message_tool_calls.append(message_tool_call)
                if message_tool_calls:
                    message_attributes["message_tool_calls"] = (
                        message_tool_calls
                    )
            return message_attributes

        return {
            "message_role": "user",  # assume user if not ChatMessage
            "message_content": str(message),
        }

    def _get_tool_call(self, tool_call: object) -> Iterator[Tuple[str, Any]]:
        if function := getattr(tool_call, "function", None):
            if name := getattr(function, "name", None):
                assert isinstance(
                    name, str
                ), f"name must be str, found {type(name)}"
                yield "tool_call_function_name", name
            if arguments := getattr(function, "arguments", None):
                assert isinstance(
                    arguments, str
                ), f"arguments must be str, found {type(arguments)}"
                yield "tool_call_function_arguments_json", arguments

    def _extract_invocation_parameters(
        self, serialized: Mapping[str, Any]
    ) -> Dict[str, Any]:
        # FIXME: this is only based on openai. Other models have different parameters.
        if not hasattr(serialized, "get"):
            return {}
        invocation_parameters: Dict[str, Any] = {}
        additional_kwargs = serialized.get("additional_kwargs")
        if additional_kwargs and isinstance(additional_kwargs, Mapping):
            invocation_parameters.update(additional_kwargs)
        for key in ("temperature", "max_tokens"):
            if (value := serialized.get(key)) is not None:
                invocation_parameters[key] = value
        return invocation_parameters

    def _get_output_messages(
        self, raw: Mapping[str, Any]
    ) -> Iterator[Tuple[str, Any]]:
        assert hasattr(raw, "get"), f"raw must be Mapping, found {type(raw)}"
        if not (choices := raw.get("choices")):
            return
        assert isinstance(
            choices, Iterable
        ), f"choices must be Iterable, found {type(choices)}"
        if messages := [
            dict(self._get_message(message))
            for choice in choices
            if (message := getattr(choice, "message", None)) is not None
        ]:
            yield "llm_output_messages", messages

    def _get_message(self, message: object) -> Iterator[Tuple[str, Any]]:
        if role := getattr(message, "role", None):
            assert isinstance(
                role, str
            ), f"content must be str, found {type(role)}"
            yield "message_role", role
        if content := getattr(message, "content", None):
            assert isinstance(
                content, str
            ), f"content must be str, found {type(content)}"
            yield "message_content", content
        if tool_calls := getattr(message, "tool_calls", None):
            assert isinstance(
                tool_calls, Iterable
            ), f"tool_calls must be Iterable, found {type(tool_calls)}"
            message_tool_calls = []
            for tool_call in tool_calls:
                if message_tool_call := dict(self._get_tool_call(tool_call)):
                    message_tool_calls.append(message_tool_call)
            if message_tool_calls:
                yield "message_tool_calls", message_tool_calls


class _CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: object) -> Any:
        try:
            return super().default(obj)
        except TypeError:
            if callable(as_dict := getattr(obj, "dict", None)):
                return as_dict()
            raise
