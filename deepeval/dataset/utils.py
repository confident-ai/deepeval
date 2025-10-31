import asyncio
import inspect
import json
import re

from typing import List, Optional, Any
from opentelemetry.trace import Tracer

from deepeval.dataset.api import Golden
from deepeval.dataset.golden import ConversationalGolden
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn


def convert_test_cases_to_goldens(
    test_cases: List[LLMTestCase],
) -> List[Golden]:
    goldens = []
    for test_case in test_cases:
        golden = {
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "context": test_case.context,
            "retrieval_context": test_case.retrieval_context,
            "tools_called": test_case.tools_called,
            "expected_tools": test_case.expected_tools,
        }
        goldens.append(Golden(**golden))
    return goldens


def convert_goldens_to_test_cases(
    goldens: List[Golden],
    _alias: Optional[str] = None,
    _id: Optional[str] = None,
) -> List[LLMTestCase]:
    test_cases = []
    for index, golden in enumerate(goldens):
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=golden.actual_output,
            expected_output=golden.expected_output,
            context=golden.context,
            retrieval_context=golden.retrieval_context,
            tools_called=golden.tools_called,
            expected_tools=golden.expected_tools,
            name=golden.name,
            comments=golden.comments,
            additional_metadata=golden.additional_metadata,
            _dataset_alias=_alias,
            _dataset_id=_id,
            _dataset_rank=index,
        )
        test_cases.append(test_case)
    return test_cases


def convert_convo_test_cases_to_convo_goldens(
    test_cases: List[ConversationalTestCase],
) -> List[ConversationalGolden]:
    goldens = []
    for test_case in test_cases:
        if not test_case.scenario:
            raise ValueError(
                "Please provide a scenario in your 'ConversationalTestCase' to convert it to a 'ConversationalGolden'."
            )
        golden = {
            "scenario": test_case.scenario,
            "turns": test_case.turns,
            "expected_outcome": test_case.expected_outcome,
            "user_description": test_case.user_description,
            "context": test_case.context,
        }
        goldens.append(ConversationalGolden(**golden))
    return goldens


def convert_convo_goldens_to_convo_test_cases(
    goldens: List[ConversationalGolden],
    _alias: Optional[str] = None,
    _id: Optional[str] = None,
) -> List[ConversationalTestCase]:
    test_cases = []
    for index, golden in enumerate(goldens):
        test_case = ConversationalTestCase(
            turns=golden.turns or [],
            scenario=golden.scenario,
            user_description=golden.user_description,
            context=golden.context,
            name=golden.name,
            additional_metadata=golden.additional_metadata,
            comments=golden.comments,
            _dataset_alias=_alias,
            _dataset_id=_id,
            _dataset_rank=index,
        )
        test_cases.append(test_case)
    return test_cases


def trimAndLoadJson(input_string: str) -> Any:
    try:
        cleaned_string = re.sub(r",\s*([\]}])", r"\1", input_string.strip())
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {input_string}. Error: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def format_turns(turns: List[Turn]) -> str:
    res = []
    for turn in turns:
        # Safely convert nested Pydantic models (ToolCall/MCP calls) to dicts
        def _dump_list(models):
            if not models:
                return None
            dumped = []
            for m in models:
                if hasattr(m, "model_dump"):
                    dumped.append(
                        m.model_dump(by_alias=True, exclude_none=True)
                    )
                elif hasattr(m, "dict"):
                    dumped.append(m.dict(exclude_none=True))
                else:
                    dumped.append(m)
            return dumped if len(dumped) > 0 else None

        cur_turn = {
            "role": turn.role,
            "content": turn.content,
            "user_id": turn.user_id if turn.user_id is not None else None,
            "retrieval_context": (
                turn.retrieval_context if turn.retrieval_context else None
            ),
            "tools_called": _dump_list(turn.tools_called),
            "mcp_tools_called": _dump_list(turn.mcp_tools_called),
            "mcp_resources_called": _dump_list(turn.mcp_resources_called),
            "mcp_prompts_called": _dump_list(turn.mcp_prompts_called),
            "additional_metadata": (
                turn.additional_metadata if turn.additional_metadata else None
            ),
        }
        res.append(cur_turn)
    try:
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Error serializing turns: {e}")


def parse_turns(turns_str: Any) -> List[Turn]:
    # Accept either a JSON string or a Python list
    if isinstance(turns_str, str):
        try:
            parsed = json.loads(turns_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    elif isinstance(turns_str, list):
        parsed = turns_str
    else:
        raise TypeError("Expected a JSON string or a list of turns.")

    if not isinstance(parsed, list):
        raise TypeError("Expected a list of turns.")

    res = []
    for i, turn in enumerate(parsed):
        if not isinstance(turn, dict):
            raise TypeError(f"Turn at index {i} is not a dictionary.")

        # Ensuring 'role' and 'content' are strings
        if "role" not in turn or not isinstance(turn["role"], str):
            raise ValueError(f"Turn at index {i} is missing a valid 'role'.")
        if "content" not in turn or not isinstance(turn["content"], str):
            raise ValueError(f"Turn at index {i} is missing a valid 'content'.")

        try:
            # Pydantic v2
            res.append(Turn.model_validate(turn))
        except AttributeError:
            # Pydantic v1 fallback
            res.append(Turn.parse_obj(turn))

    return res


def check_tracer(tracer: Optional[Tracer] = None) -> Tracer:
    if tracer:
        return tracer
    # Prefer module-level test-run tracer if available
    try:
        from deepeval.dataset.test_run_tracer import (
            GLOBAL_TEST_RUN_TRACER,
        )

        if GLOBAL_TEST_RUN_TRACER is not None:
            return GLOBAL_TEST_RUN_TRACER
    except Exception:
        raise RuntimeError(
            "No global OpenTelemetry tracer provider is configured."  # TODO: link to docs
        )

    return GLOBAL_TEST_RUN_TRACER


def coerce_to_task(obj: Any) -> asyncio.Future[Any]:
    # already a Task so just return it
    if isinstance(obj, asyncio.Task):
        return obj

    # If it is a future, it is already scheduled, so just return it
    if asyncio.isfuture(obj):
        # type: ignore[return-value]  # it is an awaitable, gather accepts it
        return obj

    # bare coroutine must be explicitly scheduled using create_task to bind to loop & track
    if asyncio.iscoroutine(obj):
        return asyncio.create_task(obj)

    # generic awaitable (any object with __await__) will need to be wrapped so create_task accepts it
    if inspect.isawaitable(obj):

        async def _wrap(awaitable):
            return await awaitable

        return asyncio.create_task(_wrap(obj))

    # not awaitable, so time to sound the alarm!
    raise TypeError(
        f"Expected Task/Future/coroutine/awaitable, got {type(obj).__name__}"
    )
