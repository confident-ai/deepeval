import asyncio
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Union

from deepeval.red_teaming.types import AttackEnhancement, Vulnerability
from deepeval.red_teaming.synthesizer import AttackSynthesizer
from deepeval.synthesizer.schema import *
from deepeval.synthesizer.types import *
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model
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
        synthesizer_model: Optional[
            Union[str, DeepEvalBaseLLM]
        ] = "gpt-3.5-turbo-0125",
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
        async_mode: bool = True,
    ):
        self.target_purpose = target_purpose
        self.target_system_prompt = target_system_prompt
        self.synthesizer_model, self.synthesizer_using_native_model = (
            initialize_model(synthesizer_model)
        )
        self.evaluation_model, self.evaluators_using_native_model = (
            initialize_model(evaluation_model)
        )
        self.async_mode = async_mode
        self.synthetic_goldens: List[Golden] = []

    ##################################################
    ### Scans ########################################
    ##################################################

    def scan(
        self,
        target_model: DeepEvalBaseLLM,
        attacks_per_vulnerability: int,
        vulnerabilities: List[Vulnerability] = [v for v in Vulnerability],
        attack_enhancements: Dict[AttackEnhancement, float] = {
            AttackEnhancement.BASE64: 0.125,
            AttackEnhancement.GRAY_BOX_ATTACK: 0.125,
            AttackEnhancement.JAILBREAK_LINEAR: 0.125,
            AttackEnhancement.JAILBREAK_TREE: 0.125,
            AttackEnhancement.LEETSPEAK: 0.125,
            AttackEnhancement.PROMPT_INJECTION: 0.125,
            AttackEnhancement.PROMPT_PROBING: 0.125,
            AttackEnhancement.ROT13: 0.125,
        },
        max_concurrent_tasks: int = 10,
    ):
        if self.async_mode:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.a_scan(
                    target_model,
                    attacks_per_vulnerability,
                    vulnerabilities,
                    attack_enhancements,
                    max_concurrent_tasks,
                )
            )
        else:
            with capture_red_teamer_run("scan"):
                # Initialize metric map
                metrics_map = self.get_red_teaming_metrics_map()

                # Generate attacks
                attacks = self.generate_attacks(
                    target_model=target_model,
                    attacks_per_vulnerability=attacks_per_vulnerability,
                    vulnerabilities=vulnerabilities,
                    attack_enhancements=attack_enhancements,
                )

                # Create a mapping of vulnerabilities to attacks
                vulnerability_to_attacks_map: Dict[
                    "Vulnerability", List["Golden"]
                ] = {}
                for attack in attacks:
                    vulnerability: Vulnerability = (
                        attack.additional_metadata.get("vulnerability")
                    )
                    if vulnerability:
                        if vulnerability not in vulnerability_to_attacks_map:
                            vulnerability_to_attacks_map[vulnerability] = [
                                attack
                            ]
                        else:
                            vulnerability_to_attacks_map[vulnerability].append(
                                attack
                            )

                # Evaluate each attack by vulnerability
                red_teaming_results = []
                red_teaming_results_breakdown = []

                pbar = tqdm(
                    vulnerability_to_attacks_map.items(),
                    desc="📝 Evaluating attack(s)",
                )
                for vulnerability, attacks in pbar:
                    scores = []
                    for attack in attacks:
                        metric: BaseMetric = metrics_map.get(vulnerability)()

                        test_case = LLMTestCase(
                            input=attack.input,
                            actual_output=target_model.generate(attack.input),
                        )
                        metric.measure(test_case)
                        scores.append(metric.score)
                        red_teaming_results_breakdown.append(
                            {
                                "Vulnerability": vulnerability.value,
                                "Attack Enhancement": attack.additional_metadata.get(
                                    "attack enhancement", "Unknown"
                                ),  # Ensure fallback value
                                "Input": attack.input,
                                "Target Output": attack.actual_output,
                                "Score": metric.score,
                                "Reason": metric.reason,
                            }
                        )

                    # Calculate average score for each vulnerability
                    vulnerability_score = (
                        sum(scores) / len(scores) if scores else 0
                    )
                    red_teaming_results.append(
                        {
                            "Vulnerability": vulnerability.value,
                            "Average Score": vulnerability_score,
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
        target_model: DeepEvalBaseLLM,
        attacks_per_vulnerability: int,
        vulnerabilities: List[Vulnerability] = [v for v in Vulnerability],
        attack_enhancements: Dict[AttackEnhancement, float] = {
            AttackEnhancement.BASE64: 0.125,
            AttackEnhancement.GRAY_BOX_ATTACK: 0.125,
            AttackEnhancement.JAILBREAK_LINEAR: 0.125,
            AttackEnhancement.JAILBREAK_TREE: 0.125,
            AttackEnhancement.LEETSPEAK: 0.125,
            AttackEnhancement.PROMPT_INJECTION: 0.125,
            AttackEnhancement.PROMPT_PROBING: 0.125,
            AttackEnhancement.ROT13: 0.125,
        },
        max_concurrent_tasks: int = 10,  # Throttling limit, control concurrency
    ):
        with capture_red_teamer_run("scan"):
            # Initialize metric map
            metrics_map = self.get_red_teaming_metrics_map()

            # Generate attacks
            attacks = await self.a_generate_attacks(
                target_model=target_model,
                attacks_per_vulnerability=attacks_per_vulnerability,
                vulnerabilities=vulnerabilities,
                attack_enhancements=attack_enhancements,
                max_concurrent_tasks=max_concurrent_tasks,
            )

            # Create a mapping of vulnerabilities to attacks
            vulnerability_to_attacks_map: Dict[
                "Vulnerability", List["Golden"]
            ] = {}
            for attack in attacks:
                vulnerability: Vulnerability = attack.additional_metadata.get(
                    "vulnerability"
                )
                if vulnerability:
                    if vulnerability not in vulnerability_to_attacks_map:
                        vulnerability_to_attacks_map[vulnerability] = [attack]
                    else:
                        vulnerability_to_attacks_map[vulnerability].append(
                            attack
                        )

            red_teaming_results = []
            red_teaming_results_breakdown = []

            # Create a semaphore for throttling concurrent tasks
            semaphore = asyncio.Semaphore(max_concurrent_tasks)

            # Total number of attacks across all vulnerabilities
            total_attacks = sum(
                len(attacks_list)
                for attacks_list in vulnerability_to_attacks_map.values()
            )

            # Create a progress bar for attack evaluations
            pbar = tqdm(total=total_attacks, desc="📝 Evaluating attack(s)")

            async def throttled_evaluate_vulnerability(
                vulnerability, attacks_list
            ):
                async with (
                    semaphore
                ):  # Ensures only `max_concurrent_tasks` run concurrently
                    avg_score, vulnerability_results = (
                        await self._a_evaluate_vulnerability(
                            target_model,
                            vulnerability,
                            attacks_list,
                            metrics_map,
                        )
                    )
                    pbar.update(
                        len(attacks_list)
                    )  # Update the progress bar by the number of attacks evaluated
                    return avg_score, vulnerability_results

            # Create a list of tasks for evaluating each vulnerability, with throttling
            tasks = [
                throttled_evaluate_vulnerability(vulnerability, attacks_list)
                for vulnerability, attacks_list in vulnerability_to_attacks_map.items()
            ]

            # Execute tasks concurrently with throttling using asyncio.gather
            results = await asyncio.gather(*tasks)

            # Close the progress bar after all tasks are done
            pbar.close()

            # Process results
            for (vulnerability, _), (avg_score, vulnerability_results) in zip(
                vulnerability_to_attacks_map.items(), results
            ):
                red_teaming_results.append(
                    {
                        "Vulnerability": vulnerability.value,
                        "Average Score": avg_score,
                    }
                )

                for score, test_case, reason in vulnerability_results:
                    red_teaming_results_breakdown.append(
                        {
                            "Vulnerability": vulnerability.value,
                            "Attack Enhancement": test_case.additional_metadata.get(
                                "attack enhancement", "Unknown"
                            ),
                            "Input": test_case.input,
                            "Target Output": test_case.actual_output,
                            "Score": score,
                            "Reason": reason,
                        }
                    )

            # Convert results to pandas DataFrames
            red_teaming_results_df = pd.DataFrame(red_teaming_results)
            red_teaming_results_breakdown_df = pd.DataFrame(
                red_teaming_results_breakdown
            )
            self.vulnerability_scores_breakdown = (
                red_teaming_results_breakdown_df
            )
            self.vulnerability_scores = red_teaming_results_df

            return red_teaming_results_df

    async def _a_evaluate_golden(
        self,
        attack: Golden,
        target_model: DeepEvalBaseLLM,
        vulnerability: Vulnerability,
        metrics_map,
    ):
        metric: BaseMetric = metrics_map[vulnerability]()
        # Generate actual output using the 'input'
        actual_output = await target_model.a_generate(attack.input)

        # No need to create another object for the enhancement if not required
        attack_enhancement = attack.additional_metadata.get(
            "attack enhancement", "Unknown"
        )

        test_case = LLMTestCase(
            input=attack.input,
            actual_output=actual_output,
            additional_metadata={"attack enhancement": attack_enhancement},
        )

        await metric.a_measure(test_case)
        return metric.score, test_case, metric.reason

    async def _a_evaluate_vulnerability(
        self,
        target_model: DeepEvalBaseLLM,
        vulnerability: Vulnerability,
        attacks_list: List[Golden],
        metrics_map,
    ):
        tasks = [
            self._a_evaluate_golden(
                attack, target_model, vulnerability, metrics_map
            )
            for attack in attacks_list
        ]
        results = await asyncio.gather(*tasks)

        # Calculate the average score
        avg_score = sum(score for score, _, _ in results) / len(results)
        return avg_score, results

    ##################################################
    ### Generating Attacks ###########################
    ##################################################

    def generate_attacks(
        self,
        target_model: DeepEvalBaseLLM,
        attacks_per_vulnerability: int,
        vulnerabilities: List[Vulnerability],
        attack_enhancements: Dict[AttackEnhancement, float] = {
            AttackEnhancement.BASE64: 0.125,
            AttackEnhancement.GRAY_BOX_ATTACK: 0.125,
            AttackEnhancement.JAILBREAK_LINEAR: 0.125,
            AttackEnhancement.JAILBREAK_TREE: 0.125,
            AttackEnhancement.LEETSPEAK: 0.125,
            AttackEnhancement.PROMPT_INJECTION: 0.125,
            AttackEnhancement.PROMPT_PROBING: 0.125,
            AttackEnhancement.ROT13: 0.125,
        },
    ) -> List[Golden]:
        with capture_red_teamer_run(
            f"generate {attacks_per_vulnerability * len(vulnerabilities)} goldens"
        ):
            attack_synthesizer = AttackSynthesizer(
                target_model=target_model,
                synthesizer_model=self.synthesizer_model,
                async_mode=self.async_mode,
                purpose=self.target_purpose,
                system_prompt=self.target_system_prompt,
            )
            attacks = attack_synthesizer.generate_attacks(
                attacks_per_vulnerability=attacks_per_vulnerability,
                vulnerabilities=vulnerabilities,
                attack_enhancements=attack_enhancements,
            )
            return attacks

    async def a_generate_attacks(
        self,
        target_model: DeepEvalBaseLLM,
        attacks_per_vulnerability: int,
        vulnerabilities: List[Vulnerability],
        attack_enhancements: Dict[AttackEnhancement, float] = {
            AttackEnhancement.BASE64: 0.125,
            AttackEnhancement.GRAY_BOX_ATTACK: 0.125,
            AttackEnhancement.JAILBREAK_LINEAR: 0.125,
            AttackEnhancement.JAILBREAK_TREE: 0.125,
            AttackEnhancement.LEETSPEAK: 0.125,
            AttackEnhancement.PROMPT_INJECTION: 0.125,
            AttackEnhancement.PROMPT_PROBING: 0.125,
            AttackEnhancement.ROT13: 0.125,
        },
        max_concurrent_tasks: int = 10,
    ) -> List[Golden]:
        with capture_red_teamer_run(
            f"generate {attacks_per_vulnerability * len(vulnerabilities)} goldens"
        ):
            # Initialize the attack synthesizer
            attack_synthesizer = AttackSynthesizer(
                target_model=target_model,
                synthesizer_model=self.synthesizer_model,
                async_mode=self.async_mode,
                purpose=self.target_purpose,
                system_prompt=self.target_system_prompt,
            )

            # Call the synthesizer to generate attacks with throttling
            attacks = await attack_synthesizer.a_generate_attacks(
                attacks_per_vulnerability=attacks_per_vulnerability,
                vulnerabilities=vulnerabilities,
                attack_enhancements=attack_enhancements,
                max_concurrent_tasks=max_concurrent_tasks,  # Pass the throttling limit
            )

        return attacks

    ##################################################
    ### Metrics Map ##################################
    ##################################################

    def get_red_teaming_metrics_map(self):
        metrics_map = {
            Vulnerability.BIAS: lambda: BiasGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            Vulnerability.CONTRACTS: lambda: ContractsGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            Vulnerability.DATA_LEAKAGE: lambda: PIIGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            Vulnerability.DEBUG_ACCESS: lambda: DebugAccessGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            Vulnerability.EXCESSIVE_AGENCY: lambda: ExcessiveAgencyGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            Vulnerability.HALLUCINATION: lambda: HallucinationGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_CHEMICAL_BIOLOGICAL_WEAPONS: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_CHEMICAL_BIOLOGICAL_WEAPONS.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_CHILD_EXPLOITATION: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_CHILD_EXPLOITATION.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_COPYRIGHT_VIOLATIONS: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_COPYRIGHT_VIOLATIONS.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_CYBERCRIME: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_CYBERCRIME.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_GRAPHIC_CONTENT: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_GRAPHIC_CONTENT.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_HARASSMENT_BULLYING: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_HARASSMENT_BULLYING.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_HATE: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_HATE.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_ILLEGAL_ACTIVITIES: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_ILLEGAL_ACTIVITIES.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_ILLEGAL_DRUGS: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_ILLEGAL_DRUGS.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_INDISCRIMINATE_WEAPONS: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_INDISCRIMINATE_WEAPONS.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_INSULTS: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_INSULTS.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_INTELLECTUAL_PROPERTY: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_INTELLECTUAL_PROPERTY.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_MISINFORMATION_DISINFORMATION: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_MISINFORMATION_DISINFORMATION.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_NON_VIOLENT_CRIME: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_NON_VIOLENT_CRIME.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_PRIVACY: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_PRIVACY.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_PROFANITY: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_PROFANITY.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_RADICALIZATION: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_RADICALIZATION.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_SELF_HARM: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_SELF_HARM.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_SEX_CRIME: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_SEX_CRIME.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_SEXUAL_CONTENT: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_SEXUAL_CONTENT.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_SPECIALIZED_ADVICE_FINANCIAL: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_SPECIALIZED_ADVICE_FINANCIAL.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_UNSAFE_PRACTICES: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_UNSAFE_PRACTICES.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.HARMFUL_VIOLENT_CRIME: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.HARMFUL_VIOLENT_CRIME.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.IMITATION: lambda: ImitationGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            Vulnerability.OFFENSIVE: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.OFFENSIVE.value,
                async_mode=self.async_mode,
            ),
            Vulnerability.PII_API_DB: lambda: PIIGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            Vulnerability.PII_DIRECT: lambda: PIIGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            Vulnerability.PII_SESSION: lambda: PIIGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            Vulnerability.PII_SOCIAL: lambda: PIIGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            Vulnerability.POLITICS: lambda: PoliticsGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            Vulnerability.RBAC: lambda: RBACGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            Vulnerability.SHELL_INJECTION: lambda: ShellInjectionGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            Vulnerability.SQL_INJECTION: lambda: SQLInjectionGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            Vulnerability.UNFORMATTED: lambda: HarmGrader(
                model=self.evaluation_model,
                harm_category=Vulnerability.OFFENSIVE.value,
                async_mode=self.async_mode,
            ),
        }
        self.metrics_map = metrics_map
        return metrics_map
