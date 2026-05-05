from deepeval.metrics import GEval, ConversationalGEval
from deepeval.test_case import (
    SingleTurnParams,
    LLMTestCase,
    ConversationalTestCase,
    Turn,
    MultiTurnParams,
)
from deepeval import evaluate

metric = GEval(
    name="G-Eval",
    criteria="Determine whether the metadata has a source, if yes score favorably, if no score unfavorably.",
    evaluation_params=[SingleTurnParams.METADATA],
)

metric2 = ConversationalGEval(
    name="Conversational G-Eval",
    criteria="Determine whether the metadata has a source, if yes score favorably, if no score unfavorably.",
    evaluation_params=[MultiTurnParams.METADATA],
)

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="Paris",
    expected_output="Paris",
    metadata={"source": "wikipedia"},
    tags=["geography"],
)

test_case2 = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What is the capital of France?"),
        Turn(role="assistant", content="Paris"),
    ],
    scenario="User asks about the capital of France",
    expected_outcome="Assistant provides the capital of France",
    metadata={"source": "wikipedia"},
)

evaluate([test_case2], [metric2])
