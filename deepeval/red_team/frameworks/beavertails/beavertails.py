import asyncio
import random
from typing import Literal, Optional, Union
from deepeval.models import DeepEvalBaseLLM
from rich.progress import Progress

from deepteam.utils import update_pbar
from deepteam.frameworks import RedTeamingFramework
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.frameworks.beavertails.types import BeaverTailsType
from deepteam.test_case import RTTestCase
from deepteam.metrics import HarmMetric

"""
BeaverTails Dataset-Driven Safety Framework
============================================

The BeaverTails framework integrates the BeaverTails dataset — a large, human-labeled 
collection of harmful and borderline prompts — to enable dataset-driven red teaming 
against real-world adversarial inputs. BeaverTails is designed to surface safety failures 
across a wide range of harm categories (abuse, illegal instructions, privacy violations, 
self-harm, sexual content, misinformation, and more) by testing how models respond to 
actual annotated malicious prompts.

BeaverTails focuses on sampling real harmful prompts from a curated dataset rather than 
synthetic generation. This provides higher-fidelity signals for model behavior in the wild 
and helps validate safety defenses, content filters, and ingestion pipelines.

The dataset includes 14 harm subcategories:
- animal_abuse
- child_abuse
- discrimination / hate_speech
- drug_abuse, weapons
- financial_crime
- misinformation
- privacy_violation
- self_harm
- sexually_explicit
- terrorism
- violence
- and more

How it works:
1. Loads the dataset using Hugging Face's datasets with the specified BeaverTails split
2. Filters out unsafe test cases to focus only on potentially harmful attacks
3. Randomly samples num_attacks prompts
4. Evaluates if the model appropriately refuses and handles the harmful attacks from dataset

Each test case includes:
- Input: Real unsafe user prompts from BeaverTails dataset
- Vulnerability: One or more harm categories assigned in the dataset
- Metric: Harm scoring using the selected evaluation model

Reference: https://huggingface.co/datasets/PKU-Alignment/BeaverTails
"""


class BeaverTails(RedTeamingFramework):
    _DATASET = "PKU-Alignment/BeaverTails"
    name = "BeaverTails"
    description = "A dataset-driven safety framework that uses the BeaverTails dataset — a large, human-labeled collection of harmful and borderline prompts. Tests models against real-world adversarial inputs across 14 harm categories including abuse, illegal instructions, privacy violations, self-harm, sexual content, and misinformation. Provides higher-fidelity signals by sampling actual annotated malicious prompts rather than synthetic generation, helping validate safety defenses and content filters."

    def __init__(
        self,
        num_attacks: int = 15,
        data_split: Literal["330k_train", "30k_train"] = "30k_train",
        async_mode: bool = True,
        verbose_mode: bool = False,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
    ):
        self.num_attacks = num_attacks
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.evaluation_model = evaluation_model
        self.data_split = data_split
        self._has_dataset = True
        self.vulnerabilities = []
        self.attacks = []

    def load_dataset(self):
        from datasets import load_dataset as hf_load_dataset

        dataset = hf_load_dataset(self._DATASET, split=self.data_split)
        attacks = [row for row in dataset if row["is_safe"] == False]

        if len(attacks) < self.num_attacks:
            raise ValueError(
                f"There are only {len(attacks)} harmful attacks that can be used in the {self.get_name()} dataset. Please set your 'num_attacks' to less than or equal to {len(attacks)}."
            )

        sampled = random.sample(attacks, k=self.num_attacks)
        test_cases = [
            RTTestCase(
                input=attack["prompt"],
                vulnerability=" ".join(
                    [
                        category
                        for category, boolean in attack["category"].items()
                        if boolean
                    ]
                ),
                vulnerability_type=BeaverTailsType.BEAVERTAILS,
            )
            for attack in sampled
        ]
        self.test_cases = test_cases
        self.vulnerabilities = [
            test_case.vulnerability for test_case in test_cases
        ]
        return self.test_cases

    def assess(
        self,
        model_callback: CallbackType,
        progress: Optional[Progress],
        task_id: Optional[int],
        ignore_errors: bool = True,
    ):
        for test_case in self.test_cases:
            try:
                try:
                    response = model_callback(test_case.input)
                except TypeError:
                    response = model_callback(test_case.input, test_case.turns)
                test_case.actual_output = response.content
                metric = self._get_metric(test_case.vulnerability)
                metric.measure(test_case)
                test_case.score = metric.score
                test_case.reason = metric.reason
            except Exception as e:
                if ignore_errors:
                    test_case.error = "Error generating output from target LLM"
                else:
                    raise
            update_pbar(progress=progress, pbar_id=task_id)

        return self.test_cases

    async def a_assess(
        self,
        model_callback: CallbackType,
        progress: Optional[Progress],
        task_id: Optional[int],
        ignore_errors: bool = True,
    ):

        async def evaluate_test_case(test_case: RTTestCase):
            try:
                try:
                    response = await model_callback(test_case.input)
                except TypeError:
                    response = await model_callback(
                        test_case.input, test_case.turns
                    )
                test_case.actual_output = response.content
                metric = self._get_metric(test_case.vulnerability)
                await metric.a_measure(test_case)
                test_case.score = metric.score
                test_case.reason = metric.reason
            except Exception as e:
                if ignore_errors:
                    test_case.error = "Error generating output from target LLM"
                else:
                    raise
            update_pbar(progress=progress, pbar_id=task_id)

        tasks = [evaluate_test_case(tc) for tc in self.test_cases]

        await asyncio.gather(*tasks)

        return self.test_cases

    def _get_metric(self, harm_category: str):
        return HarmMetric(
            harm_category=harm_category,
            model=self.evaluation_model,
            async_mode=self.async_mode,
            verbose_mode=self.verbose_mode,
        )

    def get_name(self):
        return self.name
