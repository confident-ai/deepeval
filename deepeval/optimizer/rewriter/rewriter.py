from __future__ import annotations
import random
import json
from typing import Optional, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.optimizer.scorer.schema import ScorerDiagnosisResult
from deepeval.optimizer.types import (
    ModuleId,
)
from deepeval.prompt.prompt import Prompt
from deepeval.optimizer.utils import _parse_prompt, _create_prompt
from deepeval.prompt.api import PromptType
from deepeval.metrics.utils import a_generate_with_schema_and_extract, generate_with_schema_and_extract, initialize_model
from .schema import RewriterSchema
from .template import RewriterTemplate


class Rewriter:
    """
    Uses a provided DeepEval model to rewrite the prompt for a module,
    guided by feedback_text (μ_f).

    For LIST prompts, the target message to rewrite is chosen according to
    `list_mutation_config` and `random_state`.
    """

    def __init__(
        self,
        optimizer_model: DeepEvalBaseLLM,
        max_chars: int = 4000,
        random_state: Optional[Union[int, random.Random]] = None,
    ):
        self.model, self.using_native_model = initialize_model(optimizer_model)
        self.max_chars = max_chars

        # Accept either an int seed or a Random instance.
        if isinstance(random_state, int):
            self.random_state: Optional[random.Random] = random.Random(
                random_state
            )
        else:
            self.random_state = random_state or random.Random()

    def rewrite(
        self,
        old_prompt: Prompt,
        feedback_diagnosis: ScorerDiagnosisResult,
    ) -> Prompt:
        if not feedback_diagnosis or not feedback_diagnosis.analysis:
            return old_prompt

        current_prompt_block = _parse_prompt(
            old_prompt
        )

        failures_block = feedback_diagnosis.failures
        successes_block = feedback_diagnosis.successes
        results_block = "\n\n---\n\n".join(feedback_diagnosis.results)

        mutation_prompt = RewriterTemplate.generate_mutation(
            original_prompt=current_prompt_block,
            failures=failures_block,
            successes=successes_block,
            results=results_block,
            analysis=feedback_diagnosis.analysis,
            is_list_format=old_prompt.type == PromptType.LIST,
        )

        revised_prompt_text = generate_with_schema_and_extract(
            metric=self, 
            prompt=mutation_prompt,
            schema_cls=RewriterSchema,
            extract_schema=lambda s: s.revised_prompt,
            extract_json=lambda data: data["revised_prompt"],
        )

        if isinstance(revised_prompt_text, list):
            revised_prompt_text = json.dumps(revised_prompt_text)

        return _create_prompt(
            old_prompt,
            revised_prompt_text
        )

    async def a_rewrite(
        self,
        old_prompt: Prompt,
        feedback_diagnosis: ScorerDiagnosisResult,
    ) -> Prompt:
        if not feedback_diagnosis or not feedback_diagnosis.analysis:
            return old_prompt

        current_prompt_block = _parse_prompt(
            old_prompt
        )

        failures_block = feedback_diagnosis.failures
        successes_block = feedback_diagnosis.successes
        results_block = "\n\n---\n\n".join(feedback_diagnosis.results)

        mutation_prompt = RewriterTemplate.generate_mutation(
            original_prompt=current_prompt_block,
            failures=failures_block,
            successes=successes_block,
            results=results_block,
            analysis=feedback_diagnosis.analysis,
            is_list_format=old_prompt.type == PromptType.LIST,
        )

        revised_prompt_text = await a_generate_with_schema_and_extract(
            metric=self,
            prompt=mutation_prompt,
            schema_cls=RewriterSchema,
            extract_schema=lambda s: s.revised_prompt,
            extract_json=lambda data: data["revised_prompt"],
        )

        if isinstance(revised_prompt_text, list):
            revised_prompt_text = json.dumps(revised_prompt_text)

        return _create_prompt(
            old_prompt,
            revised_prompt_text
        )

    def _accrue_cost(self, cost: float) -> None:
        pass
