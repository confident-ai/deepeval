from __future__ import annotations
import asyncio
import random
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Union, TYPE_CHECKING, Tuple

from deepeval.errors import DeepEvalError
from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.prompt.prompt import Prompt
from deepeval.metrics.utils import copy_metrics
from deepeval.optimizer.scorer import Scorer
from deepeval.optimizer.scorer.utils import (
    _measure_no_indicator,
    _a_measure_no_indicator,
)


@dataclass
class Demonstration:
    """A single, mathematically verified few-shot example."""

    input_text: str
    output_text: str
    golden_index: int = -1


@dataclass
class DemonstrationSet:
    """A set of demonstrations to be dynamically injected into a prompt."""

    demonstrations: List[Demonstration] = field(default_factory=list)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_text(self, max_demonstrations: Optional[int] = None) -> str:
        """Render demonstrations as text for inclusion in prompts."""
        demos_to_use = (
            self.demonstrations[:max_demonstrations]
            if max_demonstrations
            else self.demonstrations
        )

        if not demos_to_use:
            return ""

        lines = ["Here are some examples:", ""]
        for i, demo in enumerate(demos_to_use, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Input: {demo.input_text}")
            lines.append(f"Output: {demo.output_text}\n\n")

        lines.append("Now, please respond to the following:")
        return "\n".join(lines)


class DemonstrationBootstrapper:
    """
    Bootstraps few-shot demonstrations by running the prompt on training
    examples and keeping strictly successful outputs based on metric success.
    """

    def __init__(
        self,
        scorer: Scorer,
        max_bootstrapped_demonstrations: int = 4,
        max_labeled_demonstrations: int = 4,
        num_demonstration_sets: int = 5,
        random_state: Optional[Union[int, random.Random]] = None,
    ):
        self.scorer = scorer
        self.max_bootstrapped_demonstrations = max_bootstrapped_demonstrations
        self.max_labeled_demonstrations = max_labeled_demonstrations
        self.num_demonstration_sets = num_demonstration_sets

        if isinstance(random_state, int):
            self.random_state = random.Random(random_state)
        else:
            self.random_state = random_state or random.Random()

    def _extract_input(
        self, golden: Union[Golden, ConversationalGolden]
    ) -> str:
        """Strictly extract the input text, throwing errors on invalid state."""
        if isinstance(golden, Golden):
            if not golden.input:
                raise DeepEvalError(
                    "Golden must have a valid 'input' for MIPROv2 bootstrapping."
                )
            return golden.input

        else:
            user_turns = [
                t.content for t in (golden.turns or []) if t.role == "user"
            ]
            if not user_turns:
                raise DeepEvalError(
                    "ConversationalGolden must have at least one 'user' turn for MIPROv2 bootstrapping."
                )
            return "\n".join(user_turns)

    def _extract_expected_output(
        self, golden: Union[Golden, ConversationalGolden]
    ) -> Optional[str]:
        """Strictly extract the expected output/outcome if it exists."""
        if isinstance(golden, Golden):
            if not golden.expected_output:
                raise DeepEvalError(
                    "Golden must have a valid 'expected_output' for MIPROv2 bootstrapping."
                )
            return str(golden.expected_output)
        else:
            if not golden.expected_outcome:
                raise DeepEvalError(
                    "ConversationalGolden must have a valid 'expected_outcome' for MIPROv2 bootstrapping."
                )
            return golden.expected_outcome

    def bootstrap(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> List[DemonstrationSet]:
        """Synchronously builds DemonstrationSets utilizing the Scorer to verify metric success."""
        all_demonstrations: List[Demonstration] = []
        labeled_demonstrations: List[Demonstration] = []

        shuffled_indices = list(range(len(goldens)))
        self.random_state.shuffle(shuffled_indices)

        max_attempts = min(
            len(goldens), self.max_bootstrapped_demonstrations * 3
        )
        prompt_dict = {"__module__": prompt}

        for idx in shuffled_indices[:max_attempts]:
            golden = goldens[idx]
            input_text = self._extract_input(golden)
            expected = self._extract_expected_output(golden)

            if (
                expected
                and len(labeled_demonstrations)
                < self.max_labeled_demonstrations * self.num_demonstration_sets
            ):
                labeled_demonstrations.append(
                    Demonstration(
                        input_text=input_text,
                        output_text=expected,
                        golden_index=idx,
                    )
                )

            if (
                len(all_demonstrations)
                < self.max_bootstrapped_demonstrations
                * self.num_demonstration_sets
            ):
                try:
                    # 1. Generate actual output
                    actual_output = self.scorer.generate(prompt_dict, golden)

                    # 2. Build the test case safely
                    test_case = self.scorer._golden_to_test_case(
                        golden, actual_output
                    )

                    # 3. Evaluate against all metrics
                    metrics = copy_metrics(self.scorer.metrics)
                    is_successful = True
                    for metric in metrics:
                        _measure_no_indicator(metric, test_case)
                        if not metric.is_successful():
                            is_successful = False
                            break

                    # 4. Save if all metrics passed
                    if is_successful:
                        all_demonstrations.append(
                            Demonstration(
                                input_text=input_text,
                                output_text=actual_output,
                                golden_index=idx,
                            )
                        )
                except Exception:
                    continue

            if (
                len(all_demonstrations)
                >= self.max_bootstrapped_demonstrations
                * self.num_demonstration_sets
                and len(labeled_demonstrations)
                >= self.max_labeled_demonstrations * self.num_demonstration_sets
            ):
                break

        demo_sets = self._create_demonstration_sets(
            all_demonstrations, labeled_demonstrations
        )

        if not demo_sets or all(not ds.demonstrations for ds in demo_sets):
            raise DeepEvalError(
                "Bootstrapper failed to generate any demonstrations. "
                "Please ensure your goldens contain an 'expected_output' for labeled demonstrations."
            )

        return demo_sets

    async def a_bootstrap(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> List[DemonstrationSet]:
        """Asynchronously builds DemonstrationSets utilizing the Scorer to verify metric success."""
        labeled_demonstrations: List[Demonstration] = []
        shuffled_indices = list(range(len(goldens)))
        self.random_state.shuffle(shuffled_indices)

        max_attempts = min(
            len(goldens), self.max_bootstrapped_demonstrations * 3
        )
        selected_indices = shuffled_indices[:max_attempts]

        tasks_info: List[Tuple[int, str, Optional[str]]] = []
        prompt_dict = {"__module__": prompt}

        for idx in selected_indices:
            golden = goldens[idx]
            input_text = self._extract_input(golden)
            expected = self._extract_expected_output(golden)

            if (
                expected
                and len(labeled_demonstrations)
                < self.max_labeled_demonstrations * self.num_demonstration_sets
            ):
                labeled_demonstrations.append(
                    Demonstration(
                        input_text=input_text,
                        output_text=expected,
                        golden_index=idx,
                    )
                )

            tasks_info.append((idx, input_text, expected))

        max_bootstrapped = (
            self.max_bootstrapped_demonstrations * self.num_demonstration_sets
        )
        tasks_info = tasks_info[:max_bootstrapped]

        async def evaluate_one(
            idx: int, input_text: str, expected: Optional[str]
        ) -> Optional[Demonstration]:
            golden = goldens[idx]
            try:
                # 1. Generate actual output
                actual_output = await self.scorer.a_generate(
                    prompt_dict, golden
                )

                # 2. Build the test case safely
                test_case = self.scorer._golden_to_test_case(
                    golden, actual_output
                )

                # 3. Evaluate against all metrics
                metrics = copy_metrics(self.scorer.metrics)
                is_successful = True
                for metric in metrics:
                    await _a_measure_no_indicator(metric, test_case)
                    if not metric.is_successful():
                        is_successful = False
                        break

                # 4. Save if all metrics passed
                if is_successful:
                    return Demonstration(
                        input_text=input_text,
                        output_text=actual_output,
                        golden_index=idx,
                    )
            except Exception:
                pass
            return None

        results = await asyncio.gather(
            *[evaluate_one(idx, inp, exp) for idx, inp, exp in tasks_info]
        )
        all_demonstrations = [demo for demo in results if demo is not None]

        demo_sets = self._create_demonstration_sets(
            all_demonstrations, labeled_demonstrations
        )

        if not demo_sets or all(not ds.demonstrations for ds in demo_sets):
            raise DeepEvalError(
                "Bootstrapper failed to generate any demonstrations. "
                "Please ensure your goldens contain an 'expected_output' for labeled demonstrations."
            )

        return demo_sets

    def _create_demonstration_sets(
        self,
        bootstrapped_demonstrations: List[Demonstration],
        labeled_demonstrations: List[Demonstration],
    ) -> List[DemonstrationSet]:

        demo_sets: List[DemonstrationSet] = [
            DemonstrationSet(demonstrations=[], id="0-shot")
        ]

        for _ in range(self.num_demonstration_sets):
            demos: List[Demonstration] = []

            if bootstrapped_demonstrations:
                n_boot = min(
                    self.max_bootstrapped_demonstrations,
                    len(bootstrapped_demonstrations),
                )
                demos.extend(
                    self.random_state.sample(
                        bootstrapped_demonstrations, n_boot
                    )
                )

            if labeled_demonstrations:
                n_labeled = min(
                    self.max_labeled_demonstrations, len(labeled_demonstrations)
                )
                labeled_sample = self.random_state.sample(
                    labeled_demonstrations, n_labeled
                )
                existing_indices = {d.golden_index for d in demos}
                for demo in labeled_sample:
                    if demo.golden_index not in existing_indices:
                        demos.append(demo)
                        existing_indices.add(demo.golden_index)

            if demos:
                self.random_state.shuffle(demos)
                demo_sets.append(DemonstrationSet(demonstrations=demos))

        return demo_sets


def render_prompt_with_demonstrations(
    prompt: Prompt,
    demonstration_set: Optional[DemonstrationSet],
    max_demonstrations: int = 8,
) -> Prompt:
    from deepeval.prompt.api import PromptType, PromptMessage

    if not demonstration_set or not demonstration_set.demonstrations:
        return prompt

    demo_text = demonstration_set.to_text(max_demonstrations=max_demonstrations)

    if prompt.type == PromptType.LIST:
        new_messages = []
        demo_added = False
        for msg in prompt.messages_template:
            if not demo_added and msg.role == "system":
                new_messages.append(
                    PromptMessage(
                        role=msg.role, content=f"{msg.content}\n\n{demo_text}"
                    )
                )
                demo_added = True
            else:
                new_messages.append(msg)

        if not demo_added and new_messages:
            first = new_messages[0]
            new_messages[0] = PromptMessage(
                role=first.role, content=f"{demo_text}\n\n{first.content}"
            )
        return Prompt(messages_template=new_messages)
    else:
        return Prompt(text_template=f"{demo_text}\n\n{prompt.text_template}")
