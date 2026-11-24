import asyncio
import random
from typing import Optional, Union
from deepeval.models import DeepEvalBaseLLM
from rich.progress import Progress

from deepteam.frameworks import RedTeamingFramework
from deepteam.utils import update_pbar
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.frameworks.aegis.types import AegisType
from deepteam.test_case import RTTestCase
from deepteam.metrics import HarmMetric

"""
NVIDIA Aegis AI Content Safety Dataset Framework
=================================================

The Aegis framework integrates the NVIDIA Aegis AI Content Safety Dataset — an open-source 
dataset aligned with NVIDIA's Content Safety Taxonomy across 13 critical harm categories. 
Aegis enables DeepTeam to perform dataset-driven red teaming using real human-labeled 
safety violations to validate model robustness against harmful or unsafe user inputs.

Aegis focuses on real-world unsafe content from public conversations. This allows 
evaluation of model refusal behaviors and fine-tuning of safety filters using authentic, 
labeled harm categories.

The dataset covers 13 harm domains:
- sexual_content
- violence
- hate_speech
- self_harm
- misinformation
- privacy_violation
- child_exploitation
- drugs
- terrorism
- and more

How it works:
1. Loads the Aegis dataset from Hugging Face
2. Filters unsafe samples labeled under NVIDIA's harm taxonomy
3. Randomly samples num_attacks test cases
4. Evaluates how well the model identifies, refuses, or mitigates these harmful prompts

Each test case includes:
- Input: Real harmful user messages from public conversations (Aegis dataset)
- Vulnerability: Labeled harm category indicating expected safety violation
- Metric: Harm scoring using the evaluation model

Reference: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0
"""


class Aegis(RedTeamingFramework):
    _DATASET = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
    name = "Aegis"
    description = "A dataset-driven safety framework that uses the NVIDIA Aegis AI Content Safety Dataset — an open-source dataset aligned with NVIDIA's Content Safety Taxonomy. Tests models using real human-labeled safety violations from public conversations across 13 critical harm categories including sexual content, violence, hate speech, self-harm, misinformation, privacy violations, child exploitation, drugs, and terrorism. Enables evaluation of model refusal behaviors and fine-tuning of safety filters using authentic, labeled harm categories."

    def __init__(
        self,
        num_attacks: int = 15,
        async_mode: bool = True,
        verbose_mode: bool = False,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
    ):
        self.num_attacks = num_attacks
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.evaluation_model = evaluation_model
        self._has_dataset = True
        self.vulnerabilities = []
        self.attacks = []

    def load_dataset(self):
        from datasets import load_dataset as hf_load_dataset

        dataset = hf_load_dataset(self._DATASET, split="train")
        attacks = [
            row
            for row in dataset
            if row.get("text_type") == "user_message"
            and row.get("labels_0") != "Safe"
        ]

        if len(attacks) < self.num_attacks:
            raise ValueError(
                f"There are only {len(attacks)} harmful attacks that can be used in the {self.get_name()} dataset. Please set your 'num_attacks' to less than or equal to {len(attacks)}."
            )

        sampled = random.sample(attacks, k=self.num_attacks)
        test_cases = [
            RTTestCase(
                input=attack["text"],
                vulnerability=attack["labels_0"],
                vulnerability_type=AegisType.AEGIS,
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
