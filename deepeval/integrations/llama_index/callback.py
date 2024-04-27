from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Union, 
    Iterable, 
    Iterator,
    Tuple,
    cast
)
from time import perf_counter
import json

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
    #CBEventType.CHUNKING,
    #CBEventType.NODE_PARSING,
    CBEventType.EMBEDDING,
    CBEventType.TREE,
    CBEventType.SUB_QUESTION,
    CBEventType.FUNCTION_CALL,
    CBEventType.EXCEPTION,
    #CBEventType.AGENT_STEP,
]


class LlamaIndexCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.event_map = {}
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
        if payload is not None:
            processed_payload = self.process_payload(event_type, payload, False)
        trace_instance = self.create_trace_instance(event_type, processed_payload)
        print(payload)
        self.event_map[event_id] = trace_instance
        trace_manager.append_to_trace_stack(trace_instance)

        return

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        trace_instance = self.event_map[event_id]
        if payload is not None:
            processed_payload = self.process_payload(event_type, payload, False)
        trace_instance = self.update_trace_instance(trace_instance, event_type, processed_payload)

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
        trace_instance_input = {"args": None, "kwargs": None}

        print(event_type)
        print(processed_payload)
        
        if event_type == CBEventType.LLM:
            trace_instance = LlmTrace(
                type=type,
                executionTime=current_time,
                name=name,
                input=processed_payload['input_value'],
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
                llmMetadata=LlmMetadata(
                    model="None",
                    messages=processed_payload['messages']
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
                embeddingMetadata=EmbeddingMetadata(model="None"),
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
        return event_type.value.capitalize()


    def process_payload(
        self, 
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        is_event_end: bool = True,
    ):
 
        attributes={}

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
            {"embedding_text": text, 
             "embedding_vector": vector}
            for text, vector in zip(chunks, embeddings)
        ]
        
        #########################
        # process retrieval documents
        nodes = payload.get(EventPayload.NODES, None)
        if event_type is not CBEventType.RERANKING and nodes:
            attributes['retrieval_documents'] = self.process_nodes(nodes)

        #########################
        # process input (query string, prompts, messages)
        query_str = payload.get(EventPayload.QUERY_STR, None)
        prompt = payload.get(EventPayload.PROMPT, None)
        messages = payload.get(EventPayload.MESSAGES, None)

        if event_type is not CBEventType.RERANKING and query_str:
            attributes['input_value'] = query_str
        if prompt:
            attributes['llm_prompts'] = [prompt]
        if messages:
            if event_type is CBEventType.LLM:
                llm_msgs = [self.process_message(m) for m in messages]
                attributes["llm_messages"] = llm_msgs
            elif event_type is CBEventType.AGENT_STEP and len(messages):
                msg = messages[0]
                str_msg = msg.content if isinstance(msg, ChatMessage) else str(msg)
                attributes["input_value"] = str_msg

        #########################
        # process response (still need to process token)
        response = payload.get(EventPayload.RESPONSE)
        if isinstance(response, Response):
            attributes['response'] = response.response
        elif isinstance(response, ChatResponse):
            message = response.message
            content = message.content
            if content:
                attributes['response'] = content
            else:
                # need to process JSON
                attributes['response'] = message.additional_kwargs
        elif isinstance(response, StreamingResponse):
            if getattr(response, "response_txt", None):
                attributes['response'] = response.response_txt
        else:
            attributes['response'] = str(response)

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
            tool_parameters = tool_metadata.to_openai_tool()["function"]["parameters"]
            attributes['tool_name'] = tool_metadata.name
            attributes['tool_description'] = tool_metadata.description

        if tool_parameters:
            attributes["tool_parameters"] = json.dumps(tool_parameters)
        
        return attributes
    
    ##################################################
    ########### additional helper functions ##########
    ##################################################

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
            for node_with_score in nodes]
        return processed_nodes


    def process_message(self, message: Any) -> Dict[str, Optional[str]]:
        if isinstance(message, ChatMessage):
            message_attributes = {
                'message_role': message.role.value,
                'message_content': message.content,
            }
            # Parse the kwargs to extract the function name and parameters for function calling
            # NB: these additional kwargs exist both for 'agent' and 'function' roles
            if "name" in message.additional_kwargs:
                message_attributes['message_name'] = message.additional_kwargs["name"]
            if tool_calls := message.additional_kwargs.get("tool_calls"):
                assert isinstance(
                    tool_calls, Iterable
                ), f"tool_calls must be Iterable, found {type(tool_calls)}"
                message_tool_calls = []
                for tool_call in tool_calls:
                    if message_tool_call := dict(self._get_tool_call(tool_call)):
                        message_tool_calls.append(message_tool_call)
                if message_tool_calls:
                    message_attributes["message_tool_calls"] = message_tool_calls
            return message_attributes

        return {
            'message_role': "user",  # assume user if not ChatMessage
            'message_content': str(message),
        }
    
    def _get_tool_call(self, tool_call: object) -> Iterator[Tuple[str, Any]]:
        if function := getattr(tool_call, "function", None):
            if name := getattr(function, "name", None):
                assert isinstance(name, str), f"name must be str, found {type(name)}"
                yield "TOOL_CALL_FUNCTION_NAME", name
            if arguments := getattr(function, "arguments", None):
                assert isinstance(arguments, str), f"arguments must be str, found {type(arguments)}"
                yield "TOOL_CALL_FUNCTION_ARGUMENTS_JSON", arguments
        

