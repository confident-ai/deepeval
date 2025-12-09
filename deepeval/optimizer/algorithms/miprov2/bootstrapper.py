# Demo Bootstrapper for MIPROv2
#
# This module implements few-shot demonstration bootstrapping following
# the original MIPROv2 paper. It runs the prompt on training examples
# and collects successful outputs as demonstrations.

from __future__ import annotations
import asyncio
import random
from dataclasses import dataclass, field
from typing import List, Optional, Union, TYPE_CHECKING, Callable, Tuple

from deepeval.prompt.prompt import Prompt

if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


@dataclass
class Demo:
    """
    A single demonstration example for few-shot prompting.

    Attributes:
        input_text: The input/question from the golden
        output_text: The successful output from the model
        golden_index: Index of the source golden (for tracking)
    """

    input_text: str
    output_text: str
    golden_index: int = -1


@dataclass
class DemoSet:
    """
    A set of demonstrations to be included in a prompt.

    Attributes:
        demos: List of Demo objects
        id: Unique identifier for this demo set
    """

    demos: List[Demo] = field(default_factory=list)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            import uuid

            self.id = str(uuid.uuid4())

    def to_text(self, max_demos: Optional[int] = None) -> str:
        """Render demos as text for inclusion in prompts."""
        demos_to_use = self.demos[:max_demos] if max_demos else self.demos
        if not demos_to_use:
            return ""

        lines = ["Here are some examples:", ""]
        for i, demo in enumerate(demos_to_use, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Input: {demo.input_text}")
            lines.append(f"Output: {demo.output_text}")
            lines.append("")

        lines.append("Now, please respond to the following:")
        return "\n".join(lines)


class DemoBootstrapper:
    """
    Bootstraps few-shot demonstrations by running the prompt on
    training examples and keeping successful outputs.

    Following MIPROv2, this:
    1. Samples examples from the training set
    2. Runs them through the model with the current prompt
    3. Evaluates outputs using a simple success check
    4. Keeps successful outputs as demonstration candidates
    5. Creates multiple demo sets for variety

    Parameters
    ----------
    max_bootstrapped_demos : int
        Maximum demos per set from bootstrapping. Default is 4.
    max_labeled_demos : int
        Maximum demos per set from labeled data (golden expected_output). Default is 4.
    num_demo_sets : int
        Number of different demo sets to create. Default is 5.
    random_state : random.Random, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        num_demo_sets: int = 5,
        random_state: Optional[Union[int, random.Random]] = None,
    ):
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.num_demo_sets = num_demo_sets

        if isinstance(random_state, int):
            self.random_state = random.Random(random_state)
        else:
            self.random_state = random_state or random.Random()

    def _extract_input(
        self,
        golden: Union["Golden", "ConversationalGolden"],
    ) -> str:
        """Extract input text from a golden."""
        if hasattr(golden, "input") and golden.input:
            return str(golden.input)
        if hasattr(golden, "messages") and golden.messages:
            # For conversational, use the last user message
            for msg in reversed(golden.messages):
                if hasattr(msg, "role") and msg.role == "user":
                    return (
                        str(msg.content)
                        if hasattr(msg, "content")
                        else str(msg)
                    )
            return str(golden.messages[-1])
        return ""

    def _extract_expected_output(
        self,
        golden: Union["Golden", "ConversationalGolden"],
    ) -> Optional[str]:
        """Extract expected output from a golden if available."""
        if hasattr(golden, "expected_output") and golden.expected_output:
            return str(golden.expected_output)
        return None

    def _is_successful(
        self,
        actual_output: str,
        expected_output: Optional[str],
    ) -> bool:
        """
        Simple success check for bootstrapping.

        For now, we consider an output successful if:
        - It's non-empty
        - If expected_output exists, actual has some overlap

        This is a simplified heuristic. In full MIPROv2, you'd use
        the actual metric to validate.
        """
        if not actual_output or not actual_output.strip():
            return False

        if expected_output:
            # Simple overlap check - could be more sophisticated
            actual_words = set(actual_output.lower().split())
            expected_words = set(expected_output.lower().split())
            if actual_words and expected_words:
                overlap = len(actual_words & expected_words) / len(
                    expected_words
                )
                return overlap > 0.3  # At least 30% word overlap

        # If no expected output, just check it's non-empty
        return len(actual_output.strip()) > 10

    def bootstrap(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
        generate_fn: Callable[
            [Prompt, Union["Golden", "ConversationalGolden"]], str
        ],
    ) -> List[DemoSet]:
        """
        Bootstrap demonstration sets synchronously.

        Args:
            prompt: The prompt to use for generation
            goldens: Training examples to bootstrap from
            generate_fn: Function that takes (prompt, golden) and returns output

        Returns:
            List of DemoSet objects, each containing a different set of demos
        """
        # Collect all successful demos
        all_demos: List[Demo] = []
        labeled_demos: List[Demo] = []

        # Shuffle goldens for variety
        shuffled_indices = list(range(len(goldens)))
        self.random_state.shuffle(shuffled_indices)

        # Try to bootstrap demos
        attempts = 0
        max_attempts = min(len(goldens), self.max_bootstrapped_demos * 3)

        for idx in shuffled_indices[:max_attempts]:
            golden = goldens[idx]
            input_text = self._extract_input(golden)
            expected = self._extract_expected_output(golden)

            if not input_text:
                continue

            # If we have expected output, use it as a labeled demo
            if (
                expected
                and len(labeled_demos)
                < self.max_labeled_demos * self.num_demo_sets
            ):
                labeled_demos.append(
                    Demo(
                        input_text=input_text,
                        output_text=expected,
                        golden_index=idx,
                    )
                )

            # Try to bootstrap
            if (
                len(all_demos)
                < self.max_bootstrapped_demos * self.num_demo_sets
            ):
                try:
                    output = generate_fn(prompt, golden)
                    if self._is_successful(output, expected):
                        all_demos.append(
                            Demo(
                                input_text=input_text,
                                output_text=output,
                                golden_index=idx,
                            )
                        )
                except Exception:
                    continue

            attempts += 1
            if (
                len(all_demos)
                >= self.max_bootstrapped_demos * self.num_demo_sets
                and len(labeled_demos)
                >= self.max_labeled_demos * self.num_demo_sets
            ):
                break

        # Create diverse demo sets
        return self._create_demo_sets(all_demos, labeled_demos)

    async def a_bootstrap(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
        a_generate_fn: Callable,
    ) -> List[DemoSet]:
        """
        Bootstrap demonstration sets asynchronously (concurrently).
        """
        labeled_demos: List[Demo] = []

        shuffled_indices = list(range(len(goldens)))
        self.random_state.shuffle(shuffled_indices)

        max_attempts = min(len(goldens), self.max_bootstrapped_demos * 3)
        selected_indices = shuffled_indices[:max_attempts]

        # First pass: collect labeled demos (no async needed) and prepare bootstrap tasks
        tasks_info: List[Tuple[int, str, Optional[str]]] = (
            []
        )  # (idx, input_text, expected)

        for idx in selected_indices:
            golden = goldens[idx]
            input_text = self._extract_input(golden)
            expected = self._extract_expected_output(golden)

            if not input_text:
                continue

            # Collect labeled demos
            if (
                expected
                and len(labeled_demos)
                < self.max_labeled_demos * self.num_demo_sets
            ):
                labeled_demos.append(
                    Demo(
                        input_text=input_text,
                        output_text=expected,
                        golden_index=idx,
                    )
                )

            # Queue for bootstrapping
            tasks_info.append((idx, input_text, expected))

        # Limit how many we need to bootstrap
        max_bootstrapped = self.max_bootstrapped_demos * self.num_demo_sets
        tasks_info = tasks_info[:max_bootstrapped]

        # Run all bootstrap generations concurrently
        async def generate_one(
            idx: int,
            input_text: str,
            expected: Optional[str],
        ) -> Optional[Demo]:
            golden = goldens[idx]
            try:
                output = await a_generate_fn(prompt, golden)
                if self._is_successful(output, expected):
                    return Demo(
                        input_text=input_text,
                        output_text=output,
                        golden_index=idx,
                    )
            except Exception:
                pass
            return None

        results = await asyncio.gather(
            *[generate_one(idx, inp, exp) for idx, inp, exp in tasks_info]
        )

        # Collect successful demos
        all_demos = [demo for demo in results if demo is not None]

        return self._create_demo_sets(all_demos, labeled_demos)

    def _create_demo_sets(
        self,
        bootstrapped_demos: List[Demo],
        labeled_demos: List[Demo],
    ) -> List[DemoSet]:
        """
        Create multiple demo sets from bootstrapped and labeled demos.

        Each set contains a mix of bootstrapped and labeled demos,
        selected randomly for diversity.
        """
        demo_sets: List[DemoSet] = []

        # Always include an empty demo set (0-shot option)
        demo_sets.append(DemoSet(demos=[], id="0-shot"))

        # Create varied demo sets
        for i in range(self.num_demo_sets):
            demos: List[Demo] = []

            # Sample from bootstrapped demos
            if bootstrapped_demos:
                n_boot = min(
                    self.max_bootstrapped_demos, len(bootstrapped_demos)
                )
                boot_sample = self.random_state.sample(
                    bootstrapped_demos, n_boot
                )
                demos.extend(boot_sample)

            # Sample from labeled demos
            if labeled_demos:
                n_labeled = min(self.max_labeled_demos, len(labeled_demos))
                labeled_sample = self.random_state.sample(
                    labeled_demos, n_labeled
                )
                # Avoid duplicates
                existing_indices = {d.golden_index for d in demos}
                for demo in labeled_sample:
                    if demo.golden_index not in existing_indices:
                        demos.append(demo)
                        existing_indices.add(demo.golden_index)

            if demos:
                self.random_state.shuffle(demos)
                demo_sets.append(DemoSet(demos=demos))

        return demo_sets


def render_prompt_with_demos(
    prompt: Prompt,
    demo_set: Optional[DemoSet],
    max_demos: int = 8,
) -> Prompt:
    """
    Create a new Prompt that includes demonstrations.

    This prepends the demo text to the prompt's content.

    Args:
        prompt: The base prompt
        demo_set: The demonstration set to include
        max_demos: Maximum number of demos to include

    Returns:
        A new Prompt with demos included
    """
    from deepeval.prompt.api import PromptType, PromptMessage

    if not demo_set or not demo_set.demos:
        return prompt

    demo_text = demo_set.to_text(max_demos=max_demos)

    if prompt.type == PromptType.LIST:
        # For LIST prompts, prepend demos to the system message or first message
        new_messages = []
        demo_added = False

        for msg in prompt.messages_template:
            if not demo_added and msg.role == "system":
                # Add demos to system message
                new_content = f"{msg.content}\n\n{demo_text}"
                new_messages.append(
                    PromptMessage(role=msg.role, content=new_content)
                )
                demo_added = True
            else:
                new_messages.append(msg)

        if not demo_added and new_messages:
            # No system message, add demos to first message
            first = new_messages[0]
            new_content = f"{demo_text}\n\n{first.content}"
            new_messages[0] = PromptMessage(
                role=first.role, content=new_content
            )

        return Prompt(messages_template=new_messages)
    else:
        # For TEXT prompts, prepend demos
        new_text = f"{demo_text}\n\n{prompt.text_template}"
        return Prompt(text_template=new_text)
