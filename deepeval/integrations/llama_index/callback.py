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
    TraceStatus,
    LlmMetadata,
    EmbeddingMetadata,
    TraceType,
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
        trace_manager.clear_trace_stack()
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
        current_trace_stack = trace_manager.get_trace_stack()

        if len(current_trace_stack) > 1:
            parent_trace = current_trace_stack[-2]
            parent_trace.traces.append(trace_instance)

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
    ) -> Union[EmbeddingTrace, LlmMetadata, GenericTrace]:

        current_time = perf_counter()
        type = self.convert_event_type_to_deepeval_trace_type(event_type)
        name = event_type.capitalize()
        trace_instance_input = None

        if "exception" in processed_payload:
            trace_instance = GenericTrace(
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
                type=type,
                executionTime=current_time,
                name=name,
                input=processed_payload["llm_input_messages"],
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
                llmMetadata=LlmMetadata(
                    model=processed_payload["llm_model_name"],
                    hyperparameters=processed_payload["llm_hyperparameters"],
                    outputMessages=None,
                    tokenCount=None,
                    llmPromptTemplate=processed_payload.get(
                        "llm_prompt_template"
                    ),
                    llmPromptTemplateVariables=processed_payload.get(
                        "llm_prompt_template_variables"
                    ),
                ),
            )

        elif event_type == CBEventType.EMBEDDING:
            trace_instance = EmbeddingTrace(
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
            trace_instance = GenericTrace(
                type=type,
                executionTime=current_time,
                name=name,
                input=processed_payload["input_value"],
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
            )

        elif event_type == CBEventType.QUERY:
            trace_instance = GenericTrace(
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
                type=type,
                executionTime=current_time,
                name=name,
                input=trace_instance_input,
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
            )
        return trace_instance

    def update_trace_instance(
        self,
        trace_instance: BaseTrace,
        event_type: CBEventType,
        processed_payload: Optional[Dict[str, Any]] = None,
    ) -> Union[EmbeddingTrace, LlmMetadata, GenericTrace]:

        trace_instance.executionTime = (
            perf_counter() - trace_instance.executionTime
        )

        if "exception" in processed_payload:
            trace_instance.output = {
                "exception": processed_payload["exception"]
            }

        elif event_type == CBEventType.LLM:
            trace_instance.output = processed_payload["output_value"]
            trace_instance.llmMetadata.outputMessages = processed_payload[
                "llm_output_messages"
            ]
            trace_instance.llmMetadata.tokenCount = {
                "prompt": processed_payload["llm_token_prompt_count"],
                "completion": processed_payload["llm_token_count_completion"],
                "total": processed_payload["llm_token_count_total"],
            }

        elif event_type == CBEventType.EMBEDDING:
            embeddings = processed_payload["embeddings"]
            trace_instance.output = [
                {
                    "embedding_text": item["embedding_text"],
                    "embedding_vector_length": len(item["embedding_vector"]),
                }
                for item in embeddings
            ]
            trace_instance.input = [t["embedding_text"] for t in embeddings]

        elif event_type == CBEventType.RETRIEVE:
            documents = processed_payload["retrieval_documents"]
            trace_instance.output = documents

        elif event_type == CBEventType.QUERY:
            trace_instance.output = processed_payload["output_value"]

        elif event_type == CBEventType.SYNTHESIZE:
            trace_instance.output = processed_payload["output_value"]

        return trace_instance

    def convert_event_type_to_deepeval_trace_type(
        self, event_type: CBEventType
    ):
        # TODO: add more types
        if event_type == CBEventType.LLM:
            return TraceType.LLM
        elif event_type == CBEventType.RETRIEVE:
            return TraceType.RETRIEVER
        elif event_type == CBEventType.EMBEDDING:
            return TraceType.EMBEDDING
        elif event_type == CBEventType.CHUNKING:
            return TraceType.CHUNKING
        elif event_type == CBEventType.NODE_PARSING:
            return TraceType.NODE_PARSING
        elif event_type == CBEventType.SYNTHESIZE:
            return TraceType.SYNTHESIZE
        elif event_type == CBEventType.QUERY:
            return TraceType.QUERY
        elif event_type == CBEventType.RERANKING:
            return TraceType.RERANKING
        elif event_type == CBEventType.AGENT_STEP:
            return TraceType.AGENT_STEP

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
            {
                "document_id": node_with_score.node.node_id,
                "document_score": node_with_score.score,
                "document_content": node_with_score.node.text,
                **(
                    {"document_metadata": json.dumps(metadata)}
                    if (metadata := node_with_score.node.metadata)
                    else {}
                ),
            }
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
