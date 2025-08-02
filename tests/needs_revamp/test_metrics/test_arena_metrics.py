from typing import Dict, List, Union
from deepeval.metrics import ArenaGEval
from deepeval.test_case import (
    LLMTestCaseParams,
    LLMTestCase,
    MLLMTestCase,
    ConversationalTestCase,
    ArenaTestCase,
)


g_eval = ArenaGEval(
    name="Arena GEval",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.INPUT,
    ],
    criteria="Determine if the actual output maintains a friendly tone. Choose the winter of the more friendly contestant",
    async_mode=False,
)

# class ArenaTestCase:
#     contestants: Dict[str, LLMTestCase]


async def main():
    test_cases = ArenaTestCase(
        contestants={
            "contestant2": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris",
            ),
            "contestant2": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris is the capital of France.",
            ),
            "contestant3": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Absolutely! The capital of France is Paris 😊",
            ),
            "contestant4": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Hey there! It’s Paris—the beautiful City of Light. Have a wonderful day!",
            ),
        },
    )
    g_eval.measure(test_case=test_cases)
    print(g_eval.winner)
    print(g_eval.reason)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
