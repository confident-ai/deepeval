from typing import List, Optional

from openai import AsyncOpenAI, OpenAI

from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case.conversational_test_case import Turn


def sync_callback(
    input: str, turns: List[Turn], thread_id: Optional[str] = None
) -> Turn:
    client = OpenAI()
    messages = [{"role": turn.role, "content": turn.content} for turn in turns]
    messages.append({"role": "user", "content": input})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    print(thread_id)
    return Turn(role="assistant", content=response.choices[0].message.content)


async def async_callback_complete(
    input: str, turns: List[Turn], thread_id: Optional[str] = None
) -> Turn:
    client = AsyncOpenAI()
    messages = [{"role": turn.role, "content": turn.content} for turn in turns]
    messages.append({"role": "user", "content": input})
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    print(thread_id)
    return Turn(role="assistant", content=response.choices[0].message.content)


def static_callback(input: str) -> Turn:
    return Turn(role="assistant", content=f"Assistant response to {input}")


async def async_static_callback(input: str) -> Turn:
    return Turn(role="assistant", content=f"Assistant response to {input}")


class StaticSimulatorModel(DeepEvalBaseLLM):
    def __init__(
        self,
        expected_outcome_complete: bool = False,
        edge_indices: Optional[List[Optional[int]]] = None,
    ):
        self.schema_calls = []
        self.prompts = []
        self.user_input_count = 0
        self.expected_outcome_complete = expected_outcome_complete
        # Successive 1-based edge indices to return for EdgeChoice classification
        # (or None for "none of the above"). Exhausting the list keeps the last
        # value; an empty list defaults to None.
        self.edge_indices = list(edge_indices) if edge_indices else []
        self.edge_choice_count = 0
        super().__init__(model="static-simulator-model")

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None):
        self.prompts.append(prompt)
        if schema is None:
            return '{"simulated_input": "simulated user input"}'

        self.schema_calls.append(schema.__name__)
        if schema.__name__ == "SimulatedInput":
            self.user_input_count += 1
            return schema(
                simulated_input=f"simulated user input {self.user_input_count}"
            )
        if schema.__name__ == "ConversationCompletion":
            return schema(
                is_complete=self.expected_outcome_complete,
                reason="complete",
            )
        if schema.__name__ == "EdgeChoice":
            idx_list = self.edge_indices
            if not idx_list:
                index = None
            else:
                pos = min(self.edge_choice_count, len(idx_list) - 1)
                index = idx_list[pos]
            self.edge_choice_count += 1
            return schema(index=index, reason="static-mock")
        raise AssertionError(f"Unexpected schema: {schema.__name__}")

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return "static-simulator-model"
