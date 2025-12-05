# Instruction Proposer for MIPROv2
#
# This module generates N diverse instruction candidates upfront,
# following the original MIPROv2 paper approach. Each candidate is
# generated with different "tips" (e.g., "be creative", "be concise")
# to encourage diversity in the instruction space.

from __future__ import annotations
import asyncio
import random
from typing import List, Optional, Union, TYPE_CHECKING

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.prompt.prompt import Prompt
from deepeval.prompt.api import PromptType

if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


# Tips for encouraging diverse instruction generation (from DSPy MIPROv2)
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
    Generates N diverse instruction candidates for a given prompt.

    Following the MIPROv2 paper, this proposer:
    1. Analyzes the current prompt and task
    2. Optionally uses example inputs/outputs from goldens
    3. Applies different "tips" to encourage diversity
    4. Generates N candidate instructions
    """

    def __init__(
        self,
        optimizer_model: DeepEvalBaseLLM,
        random_state: Optional[Union[int, random.Random]] = None,
    ):
        self.optimizer_model = optimizer_model

        if isinstance(random_state, int):
            self.random_state = random.Random(random_state)
        else:
            self.random_state = random_state or random.Random()

    def _format_prompt(self, prompt: Prompt) -> str:
        """Format the prompt for the proposer context."""
        if prompt.type == PromptType.LIST:
            parts = []
            for msg in prompt.messages_template:
                role = msg.role or "unknown"
                content = msg.content or ""
                parts.append(f"[{role}]: {content}")
            return "\n".join(parts)
        else:
            return prompt.text_template or ""

    def _format_examples(
        self,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
        max_examples: int = 3,
    ) -> str:
        """Format example inputs/outputs from goldens."""
        if not goldens:
            return "No examples available."

        examples = []
        sample = self.random_state.sample(
            goldens, min(max_examples, len(goldens))
        )

        for i, golden in enumerate(sample, 1):
            # Handle both Golden and ConversationalGolden
            if hasattr(golden, "input"):
                inp = str(golden.input)
                out = str(golden.expected_output or "")
                examples.append(
                    f"Example {i}:\n  Input: {inp}\n  Expected: {out}"
                )
            elif hasattr(golden, "messages"):
                # ConversationalGolden
                msgs = golden.messages[:2] if golden.messages else []
                msg_str = " | ".join(str(m) for m in msgs)
                examples.append(f"Example {i}: {msg_str}")

        return "\n".join(examples) if examples else "No examples available."

    def _compose_proposer_prompt(
        self,
        current_prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
        tip: str,
        candidate_index: int,
    ) -> str:
        """Compose the prompt for generating an instruction candidate."""
        prompt_text = self._format_prompt(current_prompt)
        examples_text = self._format_examples(goldens)

        return f"""You are an expert prompt engineer. Your task is to propose an improved instruction/prompt for an LLM task.

[CURRENT PROMPT]
{prompt_text}

[EXAMPLE INPUTS/OUTPUTS FROM THE TASK]
{examples_text}

[GENERATION TIP]
{tip}

[INSTRUCTIONS]
Based on the current prompt, the example task inputs/outputs, and the generation tip above, propose an improved version of the prompt.

This is candidate #{candidate_index + 1}. Make it meaningfully different from trivial variations.
Focus on improving clarity, effectiveness, and alignment with the task requirements.

Return ONLY the new prompt text, with no explanations or meta-commentary."""

    def propose(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
        num_candidates: int,
    ) -> List[Prompt]:
        """
        Generate N instruction candidates synchronously.

        Args:
            prompt: The original prompt to improve
            goldens: Example inputs/outputs for context
            num_candidates: Number of candidates to generate

        Returns:
            List of Prompt candidates (including the original)
        """
        candidates: List[Prompt] = [prompt]  # Always include original

        # Select tips for diversity
        tips = self._select_tips(num_candidates - 1)

        for i, tip in enumerate(tips):
            proposer_prompt = self._compose_proposer_prompt(
                current_prompt=prompt,
                goldens=goldens,
                tip=tip,
                candidate_index=i,
            )

            try:
                output = self.optimizer_model.generate(proposer_prompt)
                new_text = self._normalize_output(output)

                if new_text and new_text.strip():
                    new_prompt = self._create_prompt_from_text(prompt, new_text)
                    if not self._is_duplicate(new_prompt, candidates):
                        candidates.append(new_prompt)
            except Exception:
                # Skip failed generations
                continue

        return candidates

    async def a_propose(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
        num_candidates: int,
    ) -> List[Prompt]:
        """
        Generate N instruction candidates asynchronously (concurrently).
        """
        candidates: List[Prompt] = [prompt]  # Always include original

        tips = self._select_tips(num_candidates - 1)

        # Build all proposer prompts upfront
        proposer_prompts = [
            self._compose_proposer_prompt(
                current_prompt=prompt,
                goldens=goldens,
                tip=tip,
                candidate_index=i,
            )
            for i, tip in enumerate(tips)
        ]

        # Generate all candidates concurrently
        async def generate_one(proposer_prompt: str) -> Optional[str]:
            try:
                output = await self.optimizer_model.a_generate(proposer_prompt)
                return self._normalize_output(output)
            except Exception:
                return None

        results = await asyncio.gather(
            *[generate_one(p) for p in proposer_prompts]
        )

        # Collect successful, non-duplicate candidates
        for new_text in results:
            if new_text and new_text.strip():
                new_prompt = self._create_prompt_from_text(prompt, new_text)
                if not self._is_duplicate(new_prompt, candidates):
                    candidates.append(new_prompt)

        return candidates

    def _select_tips(self, count: int) -> List[str]:
        """Select diverse tips for candidate generation."""
        if count <= 0:
            return []

        if count >= len(INSTRUCTION_TIPS):
            # Use all tips, possibly repeating
            tips = list(INSTRUCTION_TIPS)
            while len(tips) < count:
                tips.append(self.random_state.choice(INSTRUCTION_TIPS))
            return tips[:count]

        return self.random_state.sample(INSTRUCTION_TIPS, count)

    def _normalize_output(self, output) -> str:
        """Normalize LLM output to string."""
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, tuple):
            return str(output[0]).strip() if output else ""
        if isinstance(output, list):
            return str(output[0]).strip() if output else ""
        return str(output).strip()

    def _create_prompt_from_text(
        self, original: Prompt, new_text: str
    ) -> Prompt:
        """Create a new Prompt from generated text, preserving structure."""
        if original.type == PromptType.LIST:
            # For LIST prompts, update the system or first assistant message
            new_messages = []
            updated = False

            for msg in original.messages_template:
                if not updated and msg.role in ("system", "assistant"):
                    new_msg = type(msg)(role=msg.role, content=new_text)
                    new_messages.append(new_msg)
                    updated = True
                else:
                    new_messages.append(msg)

            if not updated and new_messages:
                # Update the first message if no system/assistant found
                first = new_messages[0]
                new_messages[0] = type(first)(role=first.role, content=new_text)

            return Prompt(messages_template=new_messages)
        else:
            return Prompt(text_template=new_text)

    def _is_duplicate(self, new_prompt: Prompt, existing: List[Prompt]) -> bool:
        """Check if a prompt is a duplicate of existing candidates."""
        new_text = self._get_prompt_text(new_prompt).strip().lower()

        for p in existing:
            existing_text = self._get_prompt_text(p).strip().lower()
            # Consider duplicates if >90% similar
            if new_text == existing_text:
                return True
            # Simple similarity check
            if len(new_text) > 0 and len(existing_text) > 0:
                shorter = min(len(new_text), len(existing_text))
                longer = max(len(new_text), len(existing_text))
                if shorter / longer > 0.9:
                    # Check prefix similarity
                    if new_text[:shorter] == existing_text[:shorter]:
                        return True
        return False

    def _get_prompt_text(self, prompt: Prompt) -> str:
        """Extract text from a prompt for comparison."""
        if prompt.type == PromptType.LIST:
            parts = []
            for msg in prompt.messages_template:
                parts.append(msg.content or "")
            return " ".join(parts)
        return prompt.text_template or ""
