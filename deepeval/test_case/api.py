from typing import Union, Optional
import os

from deepeval.test_run.api import (
    LLMApiTestCase,
    ConversationalApiTestCase,
    TurnApi,
    TraceApi,
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    MLLMTestCase,
    Turn,
)
from deepeval.constants import PYTEST_RUN_TEST_NAME


def create_api_turn(turn: Turn, index: int) -> TurnApi:
    return TurnApi(
        role=turn.role,
        content=turn.content,
        user_id=turn.user_id,
        retrievalContext=turn.retrieval_context,
        toolsCalled=turn.tools_called,
        additionalMetadata=turn.additional_metadata,
        order=index,
    )


def create_api_test_case(
    test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    trace: Optional[TraceApi] = None,
    index: Optional[int] = None,
) -> Union[LLMApiTestCase, ConversationalApiTestCase]:
    if isinstance(test_case, ConversationalTestCase):
        order = (
            test_case._dataset_rank
            if test_case._dataset_rank is not None
            else index
        )
        if test_case.name:
            name = test_case.name
        else:
            name = os.getenv(
                PYTEST_RUN_TEST_NAME, f"conversational_test_case_{order}"
            )

        api_test_case = ConversationalApiTestCase(
            name=name,
            success=True,
            metricsData=[],
            runDuration=0,
            evaluationCost=None,
            order=order,
            scenario=test_case.scenario,
            expectedOutcome=test_case.expected_outcome,
            userDescription=test_case.user_description,
            context=test_case.context,
            tags=test_case.tags,
            comments=test_case.comments,
            additionalMetadata=test_case.additional_metadata,
        )
        api_test_case.turns = [
            create_api_turn(
                turn=turn,
                index=index,
            )
            for index, turn in enumerate(test_case.turns)
        ]

        return api_test_case
    else:
        order = (
            test_case._dataset_rank
            if test_case._dataset_rank is not None
            else index
        )

        success = True
        if test_case.name is not None:
            name = test_case.name
        else:
            name = os.getenv(PYTEST_RUN_TEST_NAME, f"test_case_{order}")
        metrics_data = []

        if isinstance(test_case, LLMTestCase):
            api_test_case = LLMApiTestCase(
                name=name,
                input=test_case.input,
                actualOutput=test_case.actual_output,
                expectedOutput=test_case.expected_output,
                context=test_case.context,
                retrievalContext=test_case.retrieval_context,
                toolsCalled=test_case.tools_called,
                expectedTools=test_case.expected_tools,
                tokenCost=test_case.token_cost,
                completionTime=test_case.completion_time,
                tags=test_case.tags,
                success=success,
                metricsData=metrics_data,
                runDuration=None,
                evaluationCost=None,
                order=order,
                additionalMetadata=test_case.additional_metadata,
                comments=test_case.comments,
                trace=trace,
            )
        elif isinstance(test_case, MLLMTestCase):
            api_test_case = LLMApiTestCase(
                name=name,
                input="",
                multimodalInput=test_case.input,
                multimodalActualOutput=test_case.actual_output,
                multimodalExpectedOutput=test_case.expected_output,
                multimodalRetrievalContext=test_case.retrieval_context,
                multimodalContext=test_case.context,
                toolsCalled=test_case.tools_called,
                expectedTools=test_case.expected_tools,
                tokenCost=test_case.token_cost,
                completionTime=test_case.completion_time,
                success=success,
                metricsData=metrics_data,
                runDuration=None,
                evaluationCost=None,
                order=order,
                additionalMetadata=test_case.additional_metadata,
                comments=test_case.comments,
            )
        # llm_test_case_lookup_map[instance_id] = api_test_case
        return api_test_case
