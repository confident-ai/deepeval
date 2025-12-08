from __future__ import annotations
from typing import Dict
import pytest

from deepeval.errors import DeepEvalError
from deepeval.optimizer.scorer.scorer import (
    Scorer,
)
from deepeval.prompt.api import PromptType
from deepeval.prompt.prompt import Prompt


def _make_text_prompt(text: str) -> Prompt:
    p = Prompt(text_template=text)
    p.type = PromptType.TEXT
    return p


def test_select_module_prefers_default_module_id():
    adapter = Scorer()

    p_default = _make_text_prompt("default")
    p_other = _make_text_prompt("other")

    prompts_by_module: Dict[str, Prompt] = {
        adapter.DEFAULT_MODULE_ID: p_default,
        "other_module": p_other,
    }

    module_id = adapter._select_module_id_from_prompts(prompts_by_module)

    assert module_id == adapter.DEFAULT_MODULE_ID


def test_select_module_falls_back_to_first_key_when_default_missing():
    adapter = Scorer()

    p_a = _make_text_prompt("A")
    p_b = _make_text_prompt("B")

    prompts_by_module: Dict[str, Prompt] = {
        "module_a": p_a,
        "module_b": p_b,
    }

    module_id = adapter._select_module_id_from_prompts(prompts_by_module)

    # Python dicts preserve insertion order; we expect the first key.
    assert module_id == "module_a"


def test_select_module_raises_for_empty_prompts_dict():
    adapter = Scorer()

    with pytest.raises(DeepEvalError):
        adapter._select_module_id_from_prompts({})
