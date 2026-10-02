"""Regression tests for DeepEvalAgent.__post_init__.

DeepEvalAgent is documented as a subclass of agents.Agent, so constructing it
must still run the base __post_init__ normalization: coercing model_settings
(including dicts) and validating the other Agent fields.

These tests are deterministic: no API key and no network.
"""

import pytest
from agents import Agent

from deepeval.openai_agents import Agent as DeepEvalAgent


@pytest.mark.parametrize("model", ["gpt-5", "gpt-4o"])
def test_model_settings_match_base_agent(model: str):
    base = Agent(name="agent", instructions="Be helpful.", model=model)
    deep_eval = DeepEvalAgent(
        name="agent", instructions="Be helpful.", model=model
    )

    assert deep_eval.model_settings == base.model_settings


def test_dict_model_settings_are_coerced():
    agent = DeepEvalAgent(
        name="agent",
        instructions="Be helpful.",
        model_settings={"temperature": 0.3},
    )

    assert agent.model_settings.temperature == 0.3


def test_invalid_tools_are_rejected():
    with pytest.raises(TypeError):
        DeepEvalAgent(
            name="agent", instructions="Be helpful.", tools="not-a-list"
        )
