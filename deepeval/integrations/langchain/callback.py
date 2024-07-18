from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from langchain_core.messages import BaseMessage
from time import perf_counter
from itertools import chain
from enum import Enum
import json
import math
from copy import deepcopy
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from deepeval.utils import dataclass_to_dict, class_to_dict
from deepeval.tracing import (
    trace_manager,
    ChainTrace,
    ChainAttributes,
    ToolTrace,
    ToolAttributes,
    LlmTrace,
    LlmAttributes,
    GenericTrace,
    GenericAttributes,
    RetrieverTrace,
    RetrieverAttributes,
    RetrievalNode,
    TraceStatus,
    TraceProvider,
    LangChainTraceType,
    TraceData,
)
from deepeval.event import track


class LangChainCallbackHandler(BaseTracer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.event_map: Dict[str, TraceData] = {}
        self.track_params = {}

    def _start_trace(self, run: Run) -> None:
        # set outtermost provider
        if not trace_manager.get_outter_provider():
            trace_manager.set_outter_provider(TraceProvider.LANGCHAIN)
        self.run_map[str(run.id)] = run
        event_type = self.convert_event_type_to_deepeval_trace_type(
            run.run_type
        )
        trace_instance = self.create_trace_instance(
            event_type=event_type, name=run.name
        )
        self.event_map[str(run.id)] = trace_instance
        trace_manager.append_to_trace_stack(trace_instance)

    def _end_trace(self, run: Run) -> None:
        trace_instance = self.event_map[str(run.id)]
        event_type = trace_instance.type
        processed_payload = dict(
            self._flatten(
                chain(
                    self._as_input(self._convert_io(run.inputs)),
                    self._as_output(self._convert_io(run.outputs)),
                    self._prompts(run.inputs),
                    self._input_messages(run.inputs),
                    self._output_messages(run.outputs),
                    self._prompt_template(run),
                    self._invocation_parameters(run),
                    self._model_name(run.extra),
                    self._token_counts(run.outputs),
                    self._function_calls(run.outputs),
                    self._tools(run),
                    self._retrieval_documents(run),
                    self._metadata(run),
                )
            )
        )
        trace_instance = self.update_trace_instance(
            trace_instance, event_type, processed_payload, run
        )
        trace_instance.inputPayload = class_to_dict(run.inputs)
        trace_instance.outputPayload = class_to_dict(run.outputs)
        current_trace_stack = trace_manager.get_trace_stack_copy()

        if len(current_trace_stack) > 1:
            parent_trace = current_trace_stack[-2]
            parent_trace.traces.append(trace_instance)
            trace_manager.set_trace_stack(current_trace_stack)
        if len(current_trace_stack) == 1:
            dict_representation = dataclass_to_dict(current_trace_stack[0])
            trace_manager.set_dict_trace_stack(dict_representation)
            trace_manager.clear_trace_stack()

            if current_trace_stack[0].type == LangChainTraceType.CHAIN:
                self.track_params["input"] = current_trace_stack[
                    0
                ].chainAttributes.input
                self.track_params["response"] = current_trace_stack[
                    0
                ].chainAttributes.output

            if trace_manager.get_outter_provider() == TraceProvider.LANGCHAIN:
                track(
                    event_name=current_trace_stack[0].name,
                    model=self.track_params.get("model") or "NA",
                    input=self.track_params.get("input") or "NA",
                    response=self.track_params.get("response") or "NA",
                    retrieval_context=self.track_params.get(
                        "retrieval_context"
                    ),
                    completion_time=current_trace_stack[0].executionTime,
                    token_usage=self.track_params.get("token_usage"),
                    trace_stack=dict_representation,
                )
        else:
            trace_manager.pop_trace_stack()

    def create_trace_instance(
        self, event_type: LangChainTraceType | str, name: str
    ) -> TraceData:
        trace_kwargs = {
            "traceProvider": TraceProvider.LANGCHAIN,
            "type": event_type,
            "executionTime": perf_counter(),
            "name": name,
            "status": TraceStatus.SUCCESS,
            "traces": [],
            "inputPayload": None,
            "outputPayload": None,
        }
        if event_type == LangChainTraceType.CHAIN:
            trace_kwargs["chainAttributes"] = None
            trace_instance = ChainTrace(**trace_kwargs)
        elif event_type == LangChainTraceType.LLM:
            trace_kwargs["llmAttributes"] = None
            trace_instance = LlmTrace(**trace_kwargs)
        elif event_type == LangChainTraceType.RETRIEVER:
            trace_kwargs["retrieverAttributes"] = None
            trace_instance = RetrieverTrace(**trace_kwargs)
        elif event_type == LangChainTraceType.TOOL:
            trace_kwargs["toolAttributes"] = None
            trace_instance = ToolTrace(**trace_kwargs)
        else:
            trace_instance = GenericTrace(**trace_kwargs)
        return trace_instance

    def update_trace_instance(
        self,
        trace_instance: TraceData,
        event_type: LangChainTraceType,
        processed_payload: Optional[Dict[str, Any]],
        run: Run,
    ) -> TraceData:

        trace_instance.executionTime = (
            perf_counter() - trace_instance.executionTime
        )
        if event_type == LangChainTraceType.CHAIN:
            input_value = "NA"
            output_value = "NA"
            for key, value in run.inputs.items():
                if "input" in key and isinstance(value, str):
                    input_value = value
                    break
            for key, value in run.outputs.items():
                if "output" in key and isinstance(value, str):
                    output_value = value
                    break
            attributes = ChainAttributes(input=input_value, output=output_value)
            trace_instance.chainAttributes = attributes
        elif event_type == LangChainTraceType.LLM:
            prompt = (
                "\n".join(processed_payload.get("llm_prompts") or ["NA"])
                or "NA"
            )
            attributes = LlmAttributes(
                # Required Attributes
                input_str=prompt,
                output_str=processed_payload.get(
                    "llm_output_messages.0.message_content"
                )
                or "NA",
                # Optional Attributes
                model=processed_payload.get("llm_model"),
                total_token_count=processed_payload.get(
                    "llm_token_count_total"
                ),
                prompt_token_count=processed_payload.get(
                    "llm_token_count_prompt"
                ),
                completion_token_count=processed_payload.get(
                    "llm_token_count_completion"
                ),
                prompt_template=prompt,
                prompt_template_variables=None,
            )
            trace_instance.llmAttributes = attributes
            self.track_params["model"] = processed_payload.get("llm_model_name")
            self.track_params["token_usage"] = processed_payload.get(
                "llm_token_count_total"
            )
        elif event_type == LangChainTraceType.RETRIEVER:
            retrieval_documents: List[RetrievalNode] = []
            total_content_length = 0
            i = 0
            while True:
                content_key = f"retrieval_documents.{i}.document_content"
                metadata_key = f"retrieval_documents.{i}.document_metadata"
                if content_key not in processed_payload:
                    break
                document_content = processed_payload[content_key]
                document_metadata: Dict = json.loads(
                    processed_payload[metadata_key]
                )
                total_content_length += len(document_content)
                node = RetrievalNode(
                    content=document_content,
                    source_file=document_metadata.get("source"),
                )
                retrieval_documents.append(node)
                i += 1
            attributes = RetrieverAttributes(
                # Required Attributes
                query_str=processed_payload.get("input_value"),
                nodes=retrieval_documents,
                # Optional Attributes
                average_chunk_size=total_content_length
                // len("retrieval_documents"),
                top_k=len("retrieval_documents"),
            )
            trace_instance.retrieverAttributes = attributes
            self.track_params["retrieval_context"] = [
                doc.content for doc in retrieval_documents
            ]
        elif event_type == LangChainTraceType.TOOL:
            attributes = ToolAttributes(
                # Required Attributes
                name=processed_payload.get("tool_name") or "NA",
                description=processed_payload.get("tool_description") or "NA",
            )
            trace_instance.toolAttributes = attributes
        else:
            input_value = "NA"
            output_value = "NA"
            for key, value in run.inputs.items():
                if "input" in key and isinstance(value, str):
                    input_value = value
                    break
            for key, value in run.outputs.items():
                if "output" in key and isinstance(value, str):
                    output_value = value
                    break
            attributes = GenericAttributes(
                input=input_value, output=output_value
            )
            trace_instance.genericAttributes = attributes
        return trace_instance

    def convert_event_type_to_deepeval_trace_type(self, event_type: str):
        # TODO: add more types
        if event_type == "llm":
            return LangChainTraceType.LLM
        elif event_type == "retriever":
            return LangChainTraceType.RETRIEVER
        elif event_type == "embedding":
            return LangChainTraceType.EMBEDDING
        elif event_type == "tool":
            return LangChainTraceType.TOOL
        elif event_type == "chain":
            return LangChainTraceType.CHAIN

        return event_type.capitalize()

    def _persist_run(self, run: Run) -> None:
        pass

    def _flatten(
        self, key_values: Iterable[Tuple[str, Any]]
    ) -> Iterator[Tuple[str, Any]]:
        for key, value in key_values:
            if value is None:
                continue
            if isinstance(value, Mapping):
                for sub_key, sub_value in self._flatten(value.items()):
                    yield f"{key}.{sub_key}", sub_value
            elif isinstance(value, List) and any(
                isinstance(item, Mapping) for item in value
            ):
                for index, sub_mapping in enumerate(value):
                    for sub_key, sub_value in self._flatten(
                        sub_mapping.items()
                    ):
                        yield f"{key}.{index}.{sub_key}", sub_value
            else:
                if isinstance(value, Enum):
                    value = value.value
                yield key, value

    def safe_json_dumps(self, obj: Any, **kwargs: Any) -> str:
        """
        A convenience wrapper around `json.dumps` that ensures that any object can
        be safely encoded without a `TypeError` and that non-ASCII Unicode
        characters are not escaped.
        """
        return json.dumps(obj, default=str, ensure_ascii=False, **kwargs)

    def _as_input(self, values: Iterable[str]) -> Iterator[Tuple[str, str]]:
        return zip(("input_value", "input.mime_type"), values)

    def _as_output(self, values: Iterable[str]) -> Iterator[Tuple[str, str]]:
        return zip(("output_value", "output.mime_type"), values)

    def _convert_io(self, obj: Optional[Mapping[str, Any]]) -> Iterator[str]:
        if not obj:
            return
        assert isinstance(obj, dict), f"expected dict, found {type(obj)}"
        if len(obj) == 1 and isinstance(value := next(iter(obj.values())), str):
            yield value
        else:
            obj = dict(self._replace_nan(obj))
            yield self.safe_json_dumps(obj)
            yield "application/json"

    def _replace_nan(self, obj: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
        for k, v in obj.items():
            if isinstance(v, float) and not math.isfinite(v):
                yield k, None
            else:
                yield k, v

    # Unchecked
    def _prompts(
        self, inputs: Optional[Mapping[str, Any]]
    ) -> Iterator[Tuple[str, List[str]]]:
        """Yields prompts if present."""
        if not inputs:
            return
        assert hasattr(inputs, "get"), f"expected Mapping, found {type(inputs)}"
        if prompts := inputs.get("prompts"):
            yield "llm_prompts", prompts

    # Unchecked
    def _input_messages(
        self,
        inputs: Optional[Mapping[str, Any]],
    ) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
        """Yields chat messages if present."""
        if not inputs:
            return
        assert hasattr(inputs, "get"), f"expected Mapping, found {type(inputs)}"
        # There may be more than one set of messages. We'll use just the first set.
        if not (multiple_messages := inputs.get("messages")):
            return
        assert isinstance(
            multiple_messages, Iterable
        ), f"expected Iterable, found {type(multiple_messages)}"
        # This will only get the first set of messages.
        if not (first_messages := next(iter(multiple_messages), None)):
            return
        parsed_messages = []
        if isinstance(first_messages, list):
            for message_data in first_messages:
                if isinstance(message_data, BaseMessage):
                    parsed_messages.append(
                        dict(self._parse_message_data(message_data.to_json()))
                    )
                elif hasattr(message_data, "get"):
                    parsed_messages.append(
                        dict(self._parse_message_data(message_data))
                    )
                else:
                    raise ValueError(
                        f"failed to parse message of type {type(message_data)}"
                    )
        elif isinstance(first_messages, BaseMessage):
            parsed_messages.append(
                dict(self._parse_message_data(first_messages.to_json()))
            )
        elif hasattr(first_messages, "get"):
            parsed_messages.append(
                dict(self._parse_message_data(first_messages))
            )
        else:
            raise ValueError(
                f"failed to parse messages of type {type(first_messages)}"
            )
        if parsed_messages:
            yield "llm_input_messages", parsed_messages

    # Unchecked
    def _output_messages(
        self,
        outputs: Optional[Mapping[str, Any]],
    ) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
        """Yields chat messages if present."""
        if not outputs:
            return
        assert hasattr(
            outputs, "get"
        ), f"expected Mapping, found {type(outputs)}"
        # There may be more than one set of generations. We'll use just the first set.
        if not (multiple_generations := outputs.get("generations")):
            return
        assert isinstance(
            multiple_generations, Iterable
        ), f"expected Iterable, found {type(multiple_generations)}"
        # This will only get the first set of generations.
        if not (first_generations := next(iter(multiple_generations), None)):
            return
        assert isinstance(
            first_generations, Iterable
        ), f"expected Iterable, found {type(first_generations)}"
        parsed_messages = []
        for generation in first_generations:
            assert hasattr(
                generation, "get"
            ), f"expected Mapping, found {type(generation)}"
            if message_data := generation.get("message"):
                if isinstance(message_data, BaseMessage):
                    parsed_messages.append(
                        dict(self._parse_message_data(message_data.to_json()))
                    )
                elif hasattr(message_data, "get"):
                    parsed_messages.append(
                        dict(self._parse_message_data(message_data))
                    )
                else:
                    raise ValueError(
                        f"fail to parse message of type {type(message_data)}"
                    )
        if parsed_messages:
            yield "llm_output_messages", parsed_messages

    # Unchecked
    def _parse_message_data(
        self, message_data: Optional[Mapping[str, Any]]
    ) -> Iterator[Tuple[str, Any]]:
        """Parses message data to grab message role, content, etc."""
        if not message_data:
            return
        assert hasattr(
            message_data, "get"
        ), f"expected Mapping, found {type(message_data)}"
        id_ = message_data.get("id")
        assert isinstance(id_, List), f"expected list, found {type(id_)}"
        message_class_name = id_[-1]
        if message_class_name.startswith("HumanMessage"):
            role = "user"
        elif message_class_name.startswith("AIMessage"):
            role = "assistant"
        elif message_class_name.startswith("SystemMessage"):
            role = "system"
        elif message_class_name.startswith("FunctionMessage"):
            role = "function"
        elif message_class_name.startswith("ToolMessage"):
            role = "tool"
        elif message_class_name.startswith("ChatMessage"):
            role = message_data["kwargs"]["role"]
        else:
            raise ValueError(
                f"Cannot parse message of type: {message_class_name}"
            )
        yield "message_role", role
        if kwargs := message_data.get("kwargs"):
            assert hasattr(
                kwargs, "get"
            ), f"expected Mapping, found {type(kwargs)}"
            if content := kwargs.get("content"):
                assert isinstance(
                    content, str
                ), f"expected str, found {type(content)}"
                yield "message_content", content
            if additional_kwargs := kwargs.get("additional_kwargs"):
                assert hasattr(
                    additional_kwargs, "get"
                ), f"expected Mapping, found {type(additional_kwargs)}"
                if function_call := additional_kwargs.get("function_call"):
                    assert hasattr(
                        function_call, "get"
                    ), f"expected Mapping, found {type(function_call)}"
                    if name := function_call.get("name"):
                        assert isinstance(
                            name, str
                        ), f"expected str, found {type(name)}"
                        yield "message_function_call_name", name
                    if arguments := function_call.get("arguments"):
                        assert isinstance(
                            arguments, str
                        ), f"expected str, found {type(arguments)}"
                        yield "message_function_call_arguments_json", arguments
                if tool_calls := additional_kwargs.get("tool_calls"):
                    assert isinstance(
                        tool_calls, Iterable
                    ), f"expected Iterable, found {type(tool_calls)}"
                    message_tool_calls = []
                    for tool_call in tool_calls:
                        if message_tool_call := dict(
                            self._get_tool_call(tool_call)
                        ):
                            message_tool_calls.append(message_tool_call)
                    if message_tool_calls:
                        yield "message_tool_calls", message_tool_calls

    def _get_tool_call(
        self, tool_call: Optional[Mapping[str, Any]]
    ) -> Iterator[Tuple[str, Any]]:
        if not tool_call:
            return
        assert hasattr(
            tool_call, "get"
        ), f"expected Mapping, found {type(tool_call)}"
        if function := tool_call.get("function"):
            assert hasattr(
                function, "get"
            ), f"expected Mapping, found {type(function)}"
            if name := function.get("name"):
                assert isinstance(
                    name, str
                ), f"expected str, found {type(name)}"
                yield "tool_call_function_name", name
            if arguments := function.get("arguments"):
                assert isinstance(
                    arguments, str
                ), f"expected str, found {type(arguments)}"
                yield "tool_call_function_arguments_json", arguments

    def _prompt_template(self, run: Run) -> Iterator[Tuple[str, Any]]:
        """
        A best-effort attempt to locate the PromptTemplate object among the
        keyword arguments of a serialized object, e.g. an LLMChain object.
        """
        serialized: Optional[Mapping[str, Any]] = run.serialized
        if not serialized:
            return
        assert hasattr(
            serialized, "get"
        ), f"expected Mapping, found {type(serialized)}"
        if not (kwargs := serialized.get("kwargs")):
            return
        assert isinstance(kwargs, dict), f"expected dict, found {type(kwargs)}"
        for obj in kwargs.values():
            if not hasattr(obj, "get") or not (id_ := obj.get("id")):
                continue
            # The `id` field of the object is a list indicating the path to the
            # object's class in the LangChain package, e.g. `PromptTemplate` in
            # the `langchain.prompts.prompt` module is represented as
            # ["langchain", "prompts", "prompt", "PromptTemplate"]
            assert isinstance(
                id_, Sequence
            ), f"expected list, found {type(id_)}"
            if id_[-1].endswith("PromptTemplate"):
                if not (kwargs := obj.get("kwargs")):
                    continue
                assert hasattr(
                    kwargs, "get"
                ), f"expected Mapping, found {type(kwargs)}"
                if not (template := kwargs.get("template", "")):
                    continue
                yield "llm_prompt_template", template
                if input_variables := kwargs.get("input_variables"):
                    assert isinstance(
                        input_variables, list
                    ), f"expected list, found {type(input_variables)}"
                    template_variables = {}
                    for variable in input_variables:
                        if (value := run.inputs.get(variable)) is not None:
                            template_variables[variable] = value
                    if template_variables:
                        yield "llm_prompt_template_variables", self.safe_json_dumps(
                            template_variables
                        )
                break

    def _invocation_parameters(self, run: Run) -> Iterator[Tuple[str, str]]:
        """Yields invocation parameters if present."""
        if run.run_type.lower() != "llm":
            return
        if not (extra := run.extra):
            return
        assert hasattr(extra, "get"), f"expected Mapping, found {type(extra)}"
        if invocation_parameters := extra.get("invocation_params"):
            assert isinstance(
                invocation_parameters, Mapping
            ), f"expected Mapping, found {type(invocation_parameters)}"
            yield "llm_invocation_parameters", self.safe_json_dumps(
                invocation_parameters
            )

    def _model_name(
        self, extra: Optional[Mapping[str, Any]]
    ) -> Iterator[Tuple[str, str]]:
        """Yields model name if present."""
        if not extra:
            return
        assert hasattr(extra, "get"), f"expected Mapping, found {type(extra)}"
        if not (invocation_params := extra.get("invocation_params")):
            return
        for key in ["model_name", "model"]:
            if name := invocation_params.get(key):
                yield "llm_model_name", name
                return

    def _token_counts(
        self, outputs: Optional[Mapping[str, Any]]
    ) -> Iterator[Tuple[str, int]]:
        """Yields token count information if present."""
        if not outputs:
            return
        assert hasattr(
            outputs, "get"
        ), f"expected Mapping, found {type(outputs)}"
        if not (llm_output := outputs.get("llm_output")):
            return
        assert hasattr(
            llm_output, "get"
        ), f"expected Mapping, found {type(llm_output)}"
        if not (token_usage := llm_output.get("token_usage")):
            return
        assert hasattr(
            token_usage, "get"
        ), f"expected Mapping, found {type(token_usage)}"
        for attribute_name, key in [
            ("llm_token_count_prompt", "prompt_tokens"),
            ("llm_token_count_completion", "completion_tokens"),
            ("llm_token_count_total", "total_tokens"),
        ]:
            if (token_count := token_usage.get(key)) is not None:
                yield attribute_name, token_count

    def _function_calls(
        self, outputs: Optional[Mapping[str, Any]]
    ) -> Iterator[Tuple[str, str]]:
        """Yields function call information if present."""
        if not outputs:
            return
        assert hasattr(
            outputs, "get"
        ), f"expected Mapping, found {type(outputs)}"
        try:
            function_call_data = deepcopy(
                outputs["generations"][0][0]["message"]["kwargs"][
                    "additional_kwargs"
                ]["function_call"]
            )
            function_call_data["arguments"] = json.loads(
                function_call_data["arguments"]
            )
            yield "llm_function_call", self.safe_json_dumps(function_call_data)
        except Exception:
            pass

    def _tools(self, run: Run) -> Iterator[Tuple[str, str]]:
        """Yields tool attributes if present."""
        if run.run_type.lower() != "tool":
            return
        if not (serialized := run.serialized):
            return
        assert hasattr(
            serialized, "get"
        ), f"expected Mapping, found {type(serialized)}"
        if name := serialized.get("name"):
            yield "tool_name", name
        if description := serialized.get("description"):
            yield "tool_description", description

    def _retrieval_documents(
        self,
        run: Run,
    ) -> Iterator[Tuple[str, List[Mapping[str, Any]]]]:
        if run.run_type.lower() != "retriever":
            return
        if not (outputs := run.outputs):
            return
        assert hasattr(
            outputs, "get"
        ), f"expected Mapping, found {type(outputs)}"
        documents = outputs.get("documents")
        assert isinstance(
            documents, Iterable
        ), f"expected Iterable, found {type(documents)}"
        yield "retrieval_documents", [
            dict(self._as_document(document)) for document in documents
        ]

    def _metadata(self, run: Run) -> Iterator[Tuple[str, str]]:
        """
        Takes the LangChain chain metadata and adds it to the trace
        """
        if not run.extra or not (metadata := run.extra.get("metadata")):
            return
        assert isinstance(
            metadata, Mapping
        ), f"expected Mapping, found {type(metadata)}"
        if session_id := (
            metadata.get("langchain_session_id")
            or metadata.get("langchain_converation_id")
            or metadata.get("langchain_thread_id")
        ):
            yield "session_id", session_id
        yield "metadata", self.safe_json_dumps(metadata)

    def _as_document(self, document: Any) -> Iterator[Tuple[str, Any]]:
        if page_content := getattr(document, "page_content", None):
            assert isinstance(
                page_content, str
            ), f"expected str, found {type(page_content)}"
            yield "document_content", page_content
        if metadata := getattr(document, "metadata", None):
            assert isinstance(
                metadata, Mapping
            ), f"expected Mapping, found {type(metadata)}"
            yield "document_metadata", self.safe_json_dumps(metadata)
