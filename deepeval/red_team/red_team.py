import copy
import json
import time
import base64
import random
import asyncio
import requests
import aiohttp
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from typing import List, Optional, Union, Dict

from deepeval.synthesizer.templates.template_red_team import (
    RedTeamSynthesizerTemplate,
    RTAdversarialAttackContextlessTemplate,
)
from deepeval.synthesizer.types import RTAdversarialAttack, RTVulnerability
from deepeval.synthesizer.synthesizer_red_team import RTSynthesizer
from deepeval.synthesizer.utils import initialize_embedding_model
from deepeval.synthesizer.schema import *
from deepeval.synthesizer.types import *
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.metrics.red_teaming_metrics import *
from deepeval.metrics import BaseMetric
from deepeval.dataset.golden import Golden
from deepeval.test_case import LLMTestCase
from deepeval.utils import get_or_create_event_loop
from deepeval.telemetry import capture_red_teamer_run


class RedTeamer:
    def __init__(
        self,
        target_purpose: str,
        target_system_prompt: str,
        target_model: DeepEvalBaseLLM,
        synthesizer_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        async_mode: bool = True,
    ):
        self.target_purpose = target_purpose
        self.target_system_prompt = target_system_prompt
        self.target_model = target_model
        self.synthesizer_model, self.synthesizer_using_native_model = (
            initialize_model(synthesizer_model)
        )
        self.evaluation_model, self.evaluators_using_native_model = (
            initialize_model(evaluation_model)
        )
        self.async_mode = async_mode
        self.rt_synthesizer = RTSynthesizer(
            target_model, synthesizer_model, async_mode
        )
        self.synthetic_goldens: List[Golden] = []

    def initialize_metrics_map(self):
        metrics_map = {
            RTVulnerability.BIAS.value: lambda: BiasGrader(
                model=self.evaluation_model,
                strict_mode=True,
                async_mode=self.async_mode,
            ),
            RTVulnerability.CONTRACTS.value: lambda: ContractsGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            RTVulnerability.DATA_LEAKAGE.value: lambda: PIIGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            RTVulnerability.DEBUG_ACCESS.value: lambda: DebugAccessGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            RTVulnerability.EXCESSIVE_AGENCY.value: lambda: ExcessiveAgencyGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            RTVulnerability.HALLUCINATION.value: lambda: HallucinationGrader(
                model=self.evaluation_model, purpose=self.target_purpose
            ),
            RTVulnerability.HARMFUL_CHEMICAL_BIOLOGICAL_WEAPONS.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_CHEMICAL_BIOLOGICAL_WEAPONS.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_CHILD_EXPLOITATION.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_CHILD_EXPLOITATION.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_COPYRIGHT_VIOLATIONS.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_COPYRIGHT_VIOLATIONS.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_CYBERCRIME.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_CYBERCRIME.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_GRAPHIC_CONTENT.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_GRAPHIC_CONTENT.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_HARASSMENT_BULLYING.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_HARASSMENT_BULLYING.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_HATE.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_HATE.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_ILLEGAL_ACTIVITIES.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_ILLEGAL_ACTIVITIES.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_ILLEGAL_DRUGS.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_ILLEGAL_DRUGS.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_INDISCRIMINATE_WEAPONS.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_INDISCRIMINATE_WEAPONS.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_INSULTS.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_INSULTS.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_INTELLECTUAL_PROPERTY.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_INTELLECTUAL_PROPERTY.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_MISINFORMATION_DISINFORMATION.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_MISINFORMATION_DISINFORMATION.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_NON_VIOLENT_CRIME.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_NON_VIOLENT_CRIME.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_PRIVACY.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_PRIVACY.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_PROFANITY.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_PROFANITY.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_RADICALIZATION.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_RADICALIZATION.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_SELF_HARM.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_SELF_HARM.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_SEX_CRIME.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_SEX_CRIME.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_SEXUAL_CONTENT.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_SEXUAL_CONTENT.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_SPECIALIZED_ADVICE_FINANCIAL.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_SPECIALIZED_ADVICE_FINANCIAL.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_UNSAFE_PRACTICES.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_UNSAFE_PRACTICES.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.HARMFUL_VIOLENT_CRIME.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.HARMFUL_VIOLENT_CRIME.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.IMITATION.value: lambda: ImitationGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            RTVulnerability.OFFENSIVE.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.OFFENSIVE.value,
                async_mode=self.async_mode,
            ),
            RTVulnerability.PII_API_DB.value: lambda: PIIGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            RTVulnerability.PII_DIRECT.value: lambda: PIIGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            RTVulnerability.PII_SESSION.value: lambda: PIIGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            RTVulnerability.PII_SOCIAL.value: lambda: PIIGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            RTVulnerability.POLITICS.value: lambda: PoliticsGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            RTVulnerability.RBAC.value: lambda: RBACGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            RTVulnerability.SHELL_INJECTION.value: lambda: ShellInjectionGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            RTVulnerability.SQL_INJECTION.value: lambda: SQLInjectionGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            RTVulnerability.UNFORMATTED.value: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=RTVulnerability.OFFENSIVE.value,
                async_mode=self.async_mode,
            ),
        }
        self.metrics_map = metrics_map
        return metrics_map

    def generate_red_teaming_goldens(
        self,
        n_goldens_per_vulnerability: int,
        vulnerabilities: List[RTVulnerability],
        attacks: List[RTAdversarialAttack],
    ):
        with capture_red_teamer_run(
            "generate "
            + n_goldens_per_vulnerability * len(vulnerabilities)
            + " goldens"
        ):
            red_teaming_goldens = (
                self.rt_synthesizer.generate_red_teaming_goldens(
                    n_goldens_per_vulnerability=n_goldens_per_vulnerability,
                    vulnerabilities=vulnerabilities,
                    attacks=attacks,
                    purpose=self.target_purpose,
                    system_prompt=self.target_system_prompt,
                )
            )
            return red_teaming_goldens

    async def a_generate_red_teaming_goldens(
        self,
        n_goldens_per_vulnerability: int,
        vulnerabilities: List[RTVulnerability],
        attacks: List[RTAdversarialAttack],
    ):
        with capture_red_teamer_run(
            f""" generate {n_goldens_per_vulnerability * len(vulnerabilities) } goldens """
        ):
            red_teaming_goldens = (
                await self.rt_synthesizer.a_generate_red_teaming_goldens(
                    n_goldens_per_vulnerability=n_goldens_per_vulnerability,
                    vulnerabilities=vulnerabilities,
                    attacks=attacks,
                    purpose=self.target_purpose,
                    system_prompt=self.target_system_prompt,
                )
            )
            return red_teaming_goldens

    def scan(
        self,
        n_goldens_per_vulnerability: int,
        vulnerabilities: List[RTVulnerability],
        attacks: List[RTAdversarialAttack],
    ):
        if self.async_mode:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.a_scan(
                    n_goldens_per_vulnerability,
                    vulnerabilities,
                    attacks,
                )
            )
        else:
            with capture_red_teamer_run("scan"):
                # initialize metric map
                metrics_map = self.initialize_metrics_map()

                # generate red_teaming goldens
                red_teaming_goldens = self.generate_red_teaming_goldens(
                    n_goldens_per_vulnerability=n_goldens_per_vulnerability,
                    vulnerabilities=vulnerabilities,
                    attacks=attacks,
                )

                # create red_teaming goldens dict by vulnerability
                golden_dict = {}
                for golden in red_teaming_goldens:
                    vulnerability: RTVulnerability = golden.additional_metadata[
                        "vulnerability"
                    ]
                    if golden_dict.get(vulnerability.value, None) is None:
                        golden_dict[vulnerability.value] = [golden]
                    else:
                        golden_dict[vulnerability.value].append(golden)

                # evalute each golden by vulnerability
                red_teaming_results = []
                red_teaming_results_breakdown = []

                pbar = tqdm(
                    golden_dict.items(), desc="Evaluating vulnerability"
                )
                for vulnerability_value, goldens_list in pbar:
                    pbar.set_description(
                        f"Evaluating vulnerability - {vulnerability.value}"
                    )
                    scores = []
                    for golden in goldens_list:
                        metric: BaseMetric = metrics_map[vulnerability_value]()
                        # Generate actual output using the 'input'
                        actual_output = (
                            self.target_model.generate(golden.input),
                        )
                        test_case = LLMTestCase(
                            input=golden.input,
                            actual_output=actual_output,
                        )
                        metric.measure(test_case)
                        scores.append(metric.score)
                        red_teaming_results_breakdown.append(
                            {
                                "Vulnerability": vulnerability_value,
                                "Input": test_case.input,
                                "Target Output": test_case.actual_output,
                                "Score": metric.score,
                                "Reason": metric.reason,
                            }
                        )

                    # Calculate average score for the current vulnerability
                    average_score = sum(scores) / len(scores) if scores else 0
                    red_teaming_results.append(
                        {
                            "Vulnerability": vulnerability_value,
                            "Average Score": average_score,
                        }
                    )

                # Convert results to pandas DataFrames
                df_results = pd.DataFrame(red_teaming_results)
                df_breakdown = pd.DataFrame(red_teaming_results_breakdown)
                self.vulnerability_scores_breakdown = df_breakdown
                self.vulnerability_scores = df_results

                return df_results

    async def a_scan(
        self,
        n_goldens_per_vulnerability: int,
        vulnerabilities: List[RTVulnerability],
        attacks: List[RTAdversarialAttack],
    ):
        with capture_red_teamer_run("scan"):
            # initialize metric map
            metrics_map = self.initialize_metrics_map()

            # generate red_teaming goldens
            red_teaming_goldens = await self.a_generate_red_teaming_goldens(
                n_goldens_per_vulnerability=n_goldens_per_vulnerability,
                vulnerabilities=vulnerabilities,
                attacks=attacks,
            )

            # create red_teaming goldens dict by vulnerability
            golden_dict = {}
            for golden in red_teaming_goldens:
                vulnerability: RTVulnerability = golden.additional_metadata[
                    "vulnerability"
                ]
                if golden_dict.get(vulnerability.value, None) is None:
                    golden_dict[vulnerability.value] = [golden]
                else:
                    golden_dict[vulnerability.value].append(golden)

            red_teaming_results = []
            red_teaming_results_breakdown = []

            pbar = tqdm(
                golden_dict.items(),
                desc="Evaluating vulnerability asynchronously",
            )
            for vulnerability_value, goldens_list in pbar:
                avg_score, vulnerability_results = (
                    await self.a_evaluate_vulnerability(
                        vulnerability_value, goldens_list, metrics_map
                    )
                )

                red_teaming_results.append(
                    {
                        "Vulnerability": vulnerability_value,
                        "Average Score": avg_score,
                    }
                )

                for score, test_case, reason in vulnerability_results:
                    red_teaming_results_breakdown.append(
                        {
                            "Vulnerability": vulnerability_value,
                            "Input": test_case.input,
                            "Target Output": test_case.actual_output,
                            "Score": score,
                            "Reason": reason,
                        }
                    )

            # Convert the results to pandas DataFrames
            red_teaming_results_df = pd.DataFrame(red_teaming_results)
            red_teaming_results_breakdown_df = pd.DataFrame(
                red_teaming_results_breakdown
            )
            self.vulnerability_scores_breakdown = (
                red_teaming_results_breakdown_df
            )
            self.vulnerability_scores = red_teaming_results_df

            return red_teaming_results_df

    async def a_evaluate_golden(self, golden, vulnerability_value, metrics_map):
        metric: BaseMetric = metrics_map[vulnerability_value]()
        # Generate actual output using the 'input'
        actual_output = await self.target_model.a_generate(
            golden.input
        )  # Assuming generate is async
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=actual_output,
        )
        await metric.a_measure(test_case)  # Assuming a_measure is async
        return metric.score, test_case, metric.reason

    async def a_evaluate_vulnerability(
        self, vulnerability_value, goldens_list, metrics_map
    ):
        tasks = [
            self.a_evaluate_golden(golden, vulnerability_value, metrics_map)
            for golden in goldens_list
        ]
        results = await asyncio.gather(*tasks)

        # Calculate the average score
        avg_score = sum(score for score, _, _ in results) / len(results)
        return avg_score, results
