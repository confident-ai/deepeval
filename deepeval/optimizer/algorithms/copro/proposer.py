from __future__ import annotations

import asyncio
import difflib
import json
import random
from typing import List, Optional, Tuple, Union

from deepeval.prompt.api import PromptType
from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    initialize_model,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.optimizer.utils import _create_prompt, _parse_prompt
from deepeval.prompt.prompt import Prompt

from .schema import COPROProposalSchema, GuidelineListSchema
from .template import COPROTemplate


class COPROProposer:
    """
    Generates N diverse prompt candidates using a 2-Pass Coordinate Ascent strategy.
    Pass 1: Brainstorm distinct variation guidelines (either 0-shot or history-aware).
    Pass 2: Concurrently generate specific prompt mutations based on those guidelines.
    """

    def __init__(
        self,
        optimizer_model: DeepEvalBaseLLM,
        random_state: Optional[Union[int, random.Random]] = None,
    ):
        self.model, self.using_native_model = initialize_model(optimizer_model)

        if isinstance(random_state, int):
            self.random_state = random.Random(random_state)
        else:
            self.random_state = random_state or random.Random()

    def _accrue_cost(self, cost: float) -> None:
        pass

    def _format_history(self, history: List[Tuple[Prompt, float, str]]) -> str:
        """Formats the history of evaluated prompts, their scores, and metric feedback."""
        if not history:
            return "No previous attempts."

        history_text = []
        for i, (p, score, feedback) in enumerate(history):
            text = _parse_prompt(p).strip()
            history_text.append(
                f"Attempt #{i+1}:\n"
                f"Prompt:\n{text}\n"
                f"Score: {score:.4f}\n"
                f"Evaluation Feedback:\n{feedback}\n"
            )
        return "\n".join(history_text)

    def _is_duplicate(self, new_prompt: Prompt, existing: List[Prompt]) -> bool:
        """Mathematically checks for duplication using SequenceMatcher to prevent prompt collapse."""
        new_text = _parse_prompt(new_prompt).strip().lower()

        for p in existing:
            existing_text = _parse_prompt(p).strip().lower()
            if new_text == existing_text:
                return True
            if len(new_text) > 0 and len(existing_text) > 0:
                similarity = difflib.SequenceMatcher(
                    None, new_text, existing_text
                ).ratio()
                if similarity > 0.90:
                    return True
        return False

    def propose_bootstrap(
        self, original_prompt: Prompt, breadth: int
    ) -> List[Prompt]:
        """Pass 1 (Bootstrap): Generate 0-shot variations of the base prompt."""
        is_list = original_prompt.type == PromptType.LIST
        prompt_text = _parse_prompt(original_prompt)

        template = COPROTemplate.generate_bootstrap_guidelines(
            prompt_text, breadth
        )
        try:
            guidelines = generate_with_schema_and_extract(
                metric=self,
                prompt=template,
                schema_cls=GuidelineListSchema,
                extract_schema=lambda s: s.guidelines,
                extract_json=lambda data: data["guidelines"],
            )
        except Exception:
            return []

        return self._generate_candidates_from_guidelines(
            original_prompt, prompt_text, guidelines[:breadth], is_list
        )

    def propose_from_history(
        self,
        original_prompt: Prompt,
        history: List[Tuple[Prompt, float, str]],
        breadth: int,
    ) -> List[Prompt]:
        """Pass 1 (History): Generate ascent variations based on past performance and feedback."""
        is_list = original_prompt.type == PromptType.LIST
        prompt_text = _parse_prompt(original_prompt)
        history_text = self._format_history(history)

        template = COPROTemplate.generate_history_guidelines(
            prompt_text, history_text, breadth
        )
        try:
            guidelines = generate_with_schema_and_extract(
                metric=self,
                prompt=template,
                schema_cls=GuidelineListSchema,
                extract_schema=lambda s: s.guidelines,
                extract_json=lambda data: data["guidelines"],
            )
        except Exception:
            return []

        return self._generate_candidates_from_guidelines(
            original_prompt, prompt_text, guidelines[:breadth], is_list
        )

    def _generate_candidates_from_guidelines(
        self,
        original_prompt: Prompt,
        prompt_text: str,
        guidelines: List[str],
        is_list: bool,
    ) -> List[Prompt]:
        """Pass 2 (Sync): Iteratively generates prompts from guidelines."""
        candidates = []
        for guideline in guidelines:
            try:
                template = COPROTemplate.generate_candidate(
                    prompt_text, guideline, is_list
                )
                revised_content = generate_with_schema_and_extract(
                    metric=self,
                    prompt=template,
                    schema_cls=COPROProposalSchema,
                    extract_schema=lambda s: s.revised_prompt,
                    extract_json=lambda data: data["revised_prompt"],
                )

                if isinstance(revised_content, list):
                    revised_content = json.dumps(revised_content)

                if revised_content and revised_content.strip():
                    new_prompt = _create_prompt(
                        original_prompt, revised_content
                    )
                    if not self._is_duplicate(new_prompt, candidates):
                        candidates.append(new_prompt)
            except Exception:
                continue

        return candidates

    async def a_propose_bootstrap(
        self, original_prompt: Prompt, breadth: int
    ) -> List[Prompt]:
        """Pass 1 (Bootstrap Async): Generate 0-shot variations of the base prompt."""
        is_list = original_prompt.type == PromptType.LIST
        prompt_text = _parse_prompt(original_prompt)

        template = COPROTemplate.generate_bootstrap_guidelines(
            prompt_text, breadth
        )
        try:
            guidelines = await a_generate_with_schema_and_extract(
                metric=self,
                prompt=template,
                schema_cls=GuidelineListSchema,
                extract_schema=lambda s: s.guidelines,
                extract_json=lambda data: data["guidelines"],
            )
        except Exception:
            return []

        return await self._a_generate_candidates_from_guidelines(
            original_prompt, prompt_text, guidelines[:breadth], is_list
        )

    async def a_propose_from_history(
        self,
        original_prompt: Prompt,
        history: List[Tuple[Prompt, float, str]],
        breadth: int,
    ) -> List[Prompt]:
        """Pass 1 (History Async): Generate ascent variations based on past performance and feedback."""
        is_list = (
            original_prompt.type.value == "list"
            if hasattr(original_prompt.type, "value")
            else original_prompt.type == "list"
        )
        prompt_text = _parse_prompt(original_prompt)
        history_text = self._format_history(history)

        template = COPROTemplate.generate_history_guidelines(
            prompt_text, history_text, breadth
        )
        try:
            guidelines = await a_generate_with_schema_and_extract(
                metric=self,
                prompt=template,
                schema_cls=GuidelineListSchema,
                extract_schema=lambda s: s.guidelines,
                extract_json=lambda data: data["guidelines"],
            )
        except Exception:
            return []

        return await self._a_generate_candidates_from_guidelines(
            original_prompt, prompt_text, guidelines[:breadth], is_list
        )

    async def _a_generate_candidates_from_guidelines(
        self,
        original_prompt: Prompt,
        prompt_text: str,
        guidelines: List[str],
        is_list: bool,
    ) -> List[Prompt]:
        """Pass 2 (Async): Concurrently generates prompts from guidelines for massive speedup."""

        async def _generate_one(guideline: str) -> Optional[Prompt]:
            try:
                template = COPROTemplate.generate_candidate(
                    prompt_text, guideline, is_list
                )
                revised_content = await a_generate_with_schema_and_extract(
                    metric=self,
                    prompt=template,
                    schema_cls=COPROProposalSchema,
                    extract_schema=lambda s: s.revised_prompt,
                    extract_json=lambda data: data["revised_prompt"],
                )

                if isinstance(revised_content, list):
                    revised_content = json.dumps(revised_content)
                elif not isinstance(revised_content, str):
                    revised_content = str(revised_content)

                if revised_content and revised_content.strip():
                    return _create_prompt(original_prompt, revised_content)
            except Exception:
                pass
            return None

        tasks = [_generate_one(g) for g in guidelines]
        results = await asyncio.gather(*tasks)

        candidates = []
        for p in results:
            if p is not None and not self._is_duplicate(p, candidates):
                candidates.append(p)

        return candidates
