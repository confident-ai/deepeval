"""Trace OpenAI streams while leaving their events and lifecycle intact."""

import json
from contextlib import suppress

from deepeval.model_integrations.types import OutputParameters
from deepeval.openai.extractors import safe_extract_output_parameters
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.context import current_span_context, current_trace_context


class _StreamTrace:
    def __init__(
        self,
        stream,
        observer,
        span,
        trace,
        is_completion,
        input_parameters,
        llm_context,
        update_attributes,
    ):
        self._stream = stream
        self._observer = observer
        self._span = span
        self._trace = trace
        self._is_completion = is_completion
        self._input = input_parameters
        self._llm_context = llm_context
        self._update_attributes = update_attributes
        self._done = False
        self._parts = []
        self._prompt_tokens = None
        self._completion_tokens = None
        self._response = None
        self._tool_calls = {}

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def _record(self, event):
        if self._is_completion:
            usage = getattr(event, "usage", None)
            if usage is not None:
                self._prompt_tokens = getattr(usage, "prompt_tokens", None)
                self._completion_tokens = getattr(
                    usage, "completion_tokens", None
                )
            for choice in getattr(event, "choices", ()):
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    self._parts.append(content)
                for call in getattr(delta, "tool_calls", ()) or ():
                    item = self._tool_calls.setdefault(
                        call.index, {"name": "", "arguments": ""}
                    )
                    function = getattr(call, "function", None)
                    if function is not None:
                        item["name"] += getattr(function, "name", None) or ""
                        item["arguments"] += (
                            getattr(function, "arguments", None) or ""
                        )
        elif getattr(event, "type", None) == "response.completed":
            self._response = event.response
        elif getattr(event, "type", None) == "response.output_text.delta":
            self._parts.append(event.delta)

    def _finish(self, error=None):
        if self._done:
            return
        self._done = True
        caller_span = current_span_context.get()
        caller_trace = current_trace_context.get()
        current_span_context.set(self._span)
        current_trace_context.set(self._trace)
        try:
            if self._response is not None:
                output = safe_extract_output_parameters(
                    False, self._response, self._input
                )
                if output.output is None:
                    output.output = "".join(self._parts) or None
            else:
                tools_called = []
                for item in self._tool_calls.values():
                    try:
                        arguments = json.loads(item["arguments"])
                    except (ValueError, TypeError):
                        continue  # An early close may leave incomplete arguments.
                    tools_called.append(
                        ToolCall(
                            name=item["name"],
                            input_parameters=arguments,
                            description=(
                                self._input.tool_descriptions or {}
                            ).get(item["name"]),
                        )
                    )
                output = OutputParameters(
                    output="".join(self._parts) or tools_called or None,
                    prompt_tokens=self._prompt_tokens,
                    completion_tokens=self._completion_tokens,
                    tools_called=tools_called or None,
                )
            self._update_attributes(
                self._input,
                output,
                self._llm_context.expected_tools,
                self._llm_context.expected_output,
                self._llm_context.context,
                self._llm_context.retrieval_context,
                self._llm_context,
            )
            self._observer.result = output.output
        finally:
            try:
                self._observer.__exit__(
                    type(error) if error else None,
                    error,
                    error.__traceback__ if error else None,
                )
            finally:
                current_span_context.set(caller_span)
                current_trace_context.set(caller_trace)

    def __del__(self):
        if not getattr(self, "_done", True):
            with suppress(Exception):
                self._finish()


class TracedStream(_StreamTrace):
    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
        except StopIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._finish(exc)
            raise
        try:
            self._record(event)
        except BaseException as exc:
            self._finish(exc)
            raise
        return event

    def close(self):
        try:
            return self._stream.close()
        except BaseException as exc:
            self._finish(exc)
            raise
        finally:
            self._finish()

    def __enter__(self):
        try:
            self._stream.__enter__()
        except BaseException as exc:
            self._finish(exc)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._stream.__exit__(exc_type, exc, tb)
        finally:
            self._finish(exc)


class AsyncTracedStream(_StreamTrace):
    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._finish(exc)
            raise
        try:
            self._record(event)
        except BaseException as exc:
            self._finish(exc)
            raise
        return event

    async def close(self):
        try:
            return await self._stream.close()
        except BaseException as exc:
            self._finish(exc)
            raise
        finally:
            self._finish()

    async def __aenter__(self):
        try:
            await self._stream.__aenter__()
        except BaseException as exc:
            self._finish(exc)
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            return await self._stream.__aexit__(exc_type, exc, tb)
        finally:
            self._finish(exc)
