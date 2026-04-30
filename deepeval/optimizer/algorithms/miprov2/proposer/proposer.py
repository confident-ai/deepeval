from __future__ import annotations
import asyncio
import random
import json
import difflib
from typing import List, Optional, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.prompt.prompt import Prompt
from deepeval.prompt.api import PromptType
from deepeval.metrics.utils import (
    initialize_model,
    generate_with_schema_and_extract,
    a_generate_with_schema_and_extract,
)
from deepeval.optimizer.utils import _parse_prompt, _create_prompt
from .schema import DatasetSummarySchema, InstructionProposalSchema
from .template import ProposerTemplate

from deepeval.dataset.golden import Golden, ConversationalGolden

INSTRUCTION_TIPS = [
    "Be creative and think outside the box.",
    "Be concise and direct.",
    "Use step-by-step reasoning.",
    "Focus on clarity and precision.",
    "Include specific examples where helpful.",
    "Emphasize the most important aspects.",
    "Consider edge cases and exceptions.",
    "Use structured formatting when appropriate.",
    "Be thorough but avoid unnecessary details.",
    "Prioritize accuracy over creativity.",
    "Make the instruction self-contained.",
    "Use natural, conversational language.",
    "Be explicit about expected output format.",
    "Include context about common mistakes to avoid.",
    "Focus on the user's intent and goals.",
]


class InstructionProposer:
    """
    Generates N diverse instruction candidates for a given prompt using 
    Program-and-Data-Aware grounding and Bayesian tip diversity.
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

    def _format_examples(
        self,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
        max_examples: int = 3,
    ) -> str:
        if not goldens:
            return "No examples available."

        examples = []
        sample = self.random_state.sample(
            goldens, min(max_examples, len(goldens))
        )

        for i, golden in enumerate(sample, 1):
            if isinstance(golden, Golden):
                inp = str(golden.input)
                out = str(golden.expected_output or "")
                examples.append(f"Example {i}:\n  Input: {inp}\n  Expected: {out}")
            else:
                msgs = golden.turns if golden.turns else []
                msg_str = " | ".join(str(m) for m in msgs)
                examples.append(f"Example {i}: {msg_str}")

        return "\n".join(examples) if examples else "No examples available."

    #############################
    # Synchronous Generation    #
    #############################

    def _generate_dataset_summary(self, examples_text: str) -> str:
        prompt = ProposerTemplate.generate_dataset_summary(examples_text)
        
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=DatasetSummarySchema,
            extract_schema=lambda s: s.summary,
            extract_json=lambda data: data["summary"],
        )

    def _generate_candidate_instruction(
        self,
        current_prompt: str,
        dataset_summary: str,
        examples_text: str,
        tip: str,
        candidate_index: int,
        is_list_format: bool = False,
    ) -> Union[str, List[dict]]:
        prompt = ProposerTemplate.generate_instruction_proposal(
            current_prompt=current_prompt,
            dataset_summary=dataset_summary,
            examples_text=examples_text,
            tip=tip,
            candidate_index=candidate_index,
            is_list_format=is_list_format,
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=InstructionProposalSchema,
            extract_schema=lambda s: s.revised_instruction,
            extract_json=lambda data: data["revised_instruction"],
        )

    def propose(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
        num_candidates: int,
    ) -> List[Prompt]:
        candidates: List[Prompt] = [prompt]

        # 1. Format inputs using the global utility
        is_list = prompt.type.value == "list" if hasattr(prompt.type, "value") else prompt.type == "list"
        prompt_text = _parse_prompt(prompt)
        examples_text = self._format_examples(goldens, max_examples=5)

        # 2. Generate Data-Aware Summary
        try:
            dataset_summary = self._generate_dataset_summary(examples_text)
        except Exception:
            dataset_summary = "A standard text processing task based on the provided inputs."

        # 3. Generate Candidates
        tips = self._select_tips(num_candidates - 1)

        for i, tip in enumerate(tips):
            try:
                new_text = self._generate_candidate_instruction(
                    current_prompt=prompt_text,
                    dataset_summary=dataset_summary,
                    examples_text=examples_text,
                    tip=tip,
                    candidate_index=i,
                    is_list_format=is_list,
                )

                if new_text:
                    if isinstance(new_text, list):
                        new_text = json.dumps(new_text)
                    
                    if new_text.strip():
                        new_prompt = _create_prompt(prompt, new_text)
                        if not self._is_duplicate(new_prompt, candidates):
                            candidates.append(new_prompt)
            except Exception:
                continue

        return candidates

    #############################
    # Asynchronous Generation   #
    #############################

    async def _a_generate_dataset_summary(self, examples_text: str) -> str:
        prompt = ProposerTemplate.generate_dataset_summary(examples_text)
        
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=DatasetSummarySchema,
            extract_schema=lambda s: s.summary,
            extract_json=lambda data: data["summary"],
        )

    async def _a_generate_candidate_instruction(
        self,
        current_prompt: str,
        dataset_summary: str,
        examples_text: str,
        tip: str,
        candidate_index: int,
        is_list_format: bool = False,
    ) -> Optional[Union[str, List[dict]]]:
        prompt = ProposerTemplate.generate_instruction_proposal(
            current_prompt=current_prompt,
            dataset_summary=dataset_summary,
            examples_text=examples_text,
            tip=tip,
            candidate_index=candidate_index,
            is_list_format=is_list_format,
        )

        try:
            return await a_generate_with_schema_and_extract(
                metric=self,
                prompt=prompt,
                schema_cls=InstructionProposalSchema,
                extract_schema=lambda s: s.revised_instruction,
                extract_json=lambda data: data["revised_instruction"],
            )
        except Exception:
            return None

    async def a_propose(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
        num_candidates: int,
    ) -> List[Prompt]:
        candidates: List[Prompt] = [prompt]

        is_list = prompt.type.value == "list" if hasattr(prompt.type, "value") else prompt.type == "list"
        prompt_text = _parse_prompt(prompt)
        examples_text = self._format_examples(goldens, max_examples=5)

        try:
            dataset_summary = await self._a_generate_dataset_summary(examples_text)
        except Exception:
            dataset_summary = "A standard text processing task based on the provided inputs."

        tips = self._select_tips(num_candidates - 1)

        # Run all N candidate generations concurrently
        tasks = [
            self._a_generate_candidate_instruction(
                current_prompt=prompt_text,
                dataset_summary=dataset_summary,
                examples_text=examples_text,
                tip=tip,
                candidate_index=i,
                is_list_format=is_list,
            )
            for i, tip in enumerate(tips)
        ]

        results = await asyncio.gather(*tasks)

        for new_text in results:
            if new_text:
                if isinstance(new_text, list):
                    new_text = json.dumps(new_text)
                
                if new_text.strip():
                    new_prompt = _create_prompt(prompt, new_text)
                    if not self._is_duplicate(new_prompt, candidates):
                        candidates.append(new_prompt)

        return candidates

    #############################
    # Internal Utility Methods  #
    #############################

    def _select_tips(self, count: int) -> List[str]:
        if count <= 0:
            return []
        if count >= len(INSTRUCTION_TIPS):
            tips = list(INSTRUCTION_TIPS)
            while len(tips) < count:
                tips.append(self.random_state.choice(INSTRUCTION_TIPS))
            return tips[:count]
        return self.random_state.sample(INSTRUCTION_TIPS, count)

    def _is_duplicate(self, new_prompt: Prompt, existing: List[Prompt]) -> bool:
        new_text = _parse_prompt(new_prompt).strip().lower()

        for p in existing:
            existing_text = _parse_prompt(p).strip().lower()
            
            # Exact match
            if new_text == existing_text:
                return True
                
            # Mathematical similarity match (>90% similar)
            if len(new_text) > 0 and len(existing_text) > 0:
                similarity = difflib.SequenceMatcher(None, new_text, existing_text).ratio()
                if similarity > 0.90:
                    return True
                    
        return False