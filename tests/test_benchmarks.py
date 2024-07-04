"""Tests for various benchmark functions.
"""

import pytest
from deepeval.benchmarks import GSM8K
from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
from deepeval.dataset import Golden


@pytest.fixture
def golden():
    return Golden(
        input="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        expected_output="72",
    )


@pytest.fixture
def train_sample():
    return {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    }


def test_gsm8k_template(golden, train_sample):
    max_n_shots = 3
    train_set = [train_sample for _ in range(max_n_shots)]

    no_cot_prompt = GSM8KTemplate.generate_output(
        golden.input, train_set, n_shots=max_n_shots, enable_cot=False
    )
    assert (
        "think step-by-step" not in no_cot_prompt
    ), f"Did not expect to see chain-of-thought in the prompt {no_cot_prompt}"

    cot_prompt = GSM8KTemplate.generate_output(
        golden.input, train_set, n_shots=max_n_shots, enable_cot=True
    )
    assert (
        "think step-by-step" in cot_prompt
    ), f"Expected to see chain-of-thought in the prompt {cot_prompt}"

    cot_prompt_zero_shot = GSM8KTemplate.generate_output(
        golden.input, train_set, n_shots=max_n_shots, enable_cot=True
    )
    assert (
        "think step-by-step" in cot_prompt_zero_shot
    ), f"Expected to see chain-of-thought in the prompt {cot_prompt_zero_shot}"
