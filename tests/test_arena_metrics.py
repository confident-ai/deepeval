from typing import Dict, List, Union
from deepeval.metrics import GEval
from deepeval.test_case import (
    LLMTestCaseParams,
    LLMTestCase,
    MLLMTestCase,
    ConversationalTestCase,
)


class ArenaTestCase:
    contestants: Dict[
        str, Union[LLMTestCase, MLLMTestCase, ConversationalTestCase]
    ]


class Arena:
    test_cases: List[ArenaTestCase]


g_eval = GEval(
    name="GEval",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    criteria="Determine if the actual output maintains a friendly tone. Give it a higher score if yes.",
    async_mode=True,
)


async def main():
    test_cases = [
        LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris",
        ),
        LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
        ),
        LLMTestCase(
            input="What is the capital of France?",
            actual_output="Absolutely! The capital of France is Paris 😊",
        ),
        LLMTestCase(
            input="What is the capital of France?",
            actual_output="Hey there! It’s Paris—the beautiful City of Light. Have a wonderful day!",
        ),
    ]
    g_eval.measure(test_case=test_cases)
    print(g_eval.best_test_case)
    print(g_eval.best_test_case_index)
    print(g_eval.reason)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
