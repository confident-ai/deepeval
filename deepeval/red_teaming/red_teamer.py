import asyncio
import pandas as pd
from pydantic import Field
from tqdm import tqdm
from typing import Dict, List, Optional, Union

from deepeval.red_teaming.types import AttackEnhancement, Vulnerability
from deepeval.red_teaming.attack_synthesizer import AttackSynthesizer, Attack
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


class VulnerabilityResult(BaseModel):
    vulnerability: str
    attack_enhancement: Optional[str] = None
    input: Optional[str] = None
    actual_output: Optional[str] = None
    score: Optional[int] = None
    reason: Optional[str] = None
    error: Optional[str] = None


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
        self.attack_synthesizer = AttackSynthesizer(
            synthesizer_model=self.synthesizer_model,
            purpose=self.target_purpose,
            system_prompt=self.target_system_prompt,
        )

    ##################################################
    ### Scans ########################################
    ##################################################

    def scan(
        self,
        target_model: DeepEvalBaseLLM,
        attacks_per_vulnerability: int,
        vulnerabilities: List[Vulnerability] = [v for v in Vulnerability],
        attack_enhancements: Dict[AttackEnhancement, float] = {
            AttackEnhancement.BASE64: 1 / 11,
            AttackEnhancement.GRAY_BOX_ATTACK: 1 / 11,
            AttackEnhancement.JAILBREAK_CRESCENDO: 1 / 11,
            AttackEnhancement.JAILBREAK_LINEAR: 1 / 11,
            AttackEnhancement.JAILBREAK_TREE: 1 / 11,
            AttackEnhancement.LEETSPEAK: 1 / 11,
            AttackEnhancement.PROMPT_INJECTION: 1 / 11,
            AttackEnhancement.PROMPT_PROBING: 1 / 11,
            AttackEnhancement.ROT13: 1 / 11,
            AttackEnhancement.MATH_PROBLEM: 1 / 11,
            AttackEnhancement.MULTILINGUAL: 1 / 11,
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
                attacks: List[Attack] = (
                    self.attack_synthesizer.generate_attacks(
                        target_model=target_model,
                        attacks_per_vulnerability=attacks_per_vulnerability,
                        vulnerabilities=vulnerabilities,
                        attack_enhancements=attack_enhancements,
                    )
                )

                # Create a mapping of vulnerabilities to attacks
                vulnerability_to_attacks_map: Dict[
                    Vulnerability, List[Attack]
                ] = {}
                for attack in attacks:
                    if attack.vulnerability not in vulnerability_to_attacks_map:
                        vulnerability_to_attacks_map[attack.vulnerability] = [
                            attack
                        ]
                    else:
                        vulnerability_to_attacks_map[
                            attack.vulnerability
                        ].append(attack)

                # Evaluate each attack by vulnerability
                red_teaming_results = []
                red_teaming_results_breakdown = []

                pbar = tqdm(
                    vulnerability_to_attacks_map.items(),
                    desc=f"📝 Evaluating {len(vulnerabilities)} vulnerability(s)",
                )
                for vulnerability, attacks in pbar:
                    scores = []
                    for attack in attacks:
                        metric: BaseMetric = metrics_map.get(vulnerability)()
                        result = {
                            "Vulnerability": vulnerability.value,
                            "Attack Enhancement": attack.attack_enhancement,
                            "Input": attack.input,
                            "Target Output": None,
                            "Score": None,
                            "Reason": None,
                            "Error": None,
                        }

                        if attack.error:
                            result["Error"] = attack.error
                            red_teaming_results_breakdown.append(result)
                            continue

                        try:
                            target_output = target_model.generate(attack.input)
                            result["Target Output"] = target_output
                        except Exception:
                            result["Error"] = (
                                "Error generating output from target LLM"
                            )
                            red_teaming_results_breakdown.append(result)
                            continue

                        test_case = LLMTestCase(
                            input=attack.input,
                            actual_output=target_output,
                        )

                        try:
                            metric.measure(test_case)
                            result["Score"] = metric.score
                            result["Reason"] = metric.reason
                            scores.append(metric.score)
                        except Exception:
                            result["Error"] = (
                                f"Error evaluating target LLM output for the '{vulnerability.value}' vulnerability"
                            )
                            red_teaming_results_breakdown.append(result)
                            continue

                        red_teaming_results_breakdown.append(result)

                    # Calculate average score for each vulnerability
                    avg_vulnerability_score = (
                        sum(scores) / len(scores) if scores else None
                    )
                    red_teaming_results.append(
                        {
                            "Vulnerability": vulnerability.value,
                            "Average Score": avg_vulnerability_score,
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
            AttackEnhancement.BASE64: 1 / 11,
            AttackEnhancement.GRAY_BOX_ATTACK: 1 / 11,
            AttackEnhancement.JAILBREAK_CRESCENDO: 1 / 11,
            AttackEnhancement.JAILBREAK_LINEAR: 1 / 11,
            AttackEnhancement.JAILBREAK_TREE: 1 / 11,
            AttackEnhancement.LEETSPEAK: 1 / 11,
            AttackEnhancement.PROMPT_INJECTION: 1 / 11,
            AttackEnhancement.PROMPT_PROBING: 1 / 11,
            AttackEnhancement.ROT13: 1 / 11,
            AttackEnhancement.MATH_PROBLEM: 1 / 11,
            AttackEnhancement.MULTILINGUAL: 1 / 11,
        },
        max_concurrent_tasks: int = 10,  # Throttling limit, control concurrency
    ):
        with capture_red_teamer_run("scan"):
            # Initialize metric map
            metrics_map = self.get_red_teaming_metrics_map()

            # Generate attacks
            attacks = await self.attack_synthesizer.a_generate_attacks(
                target_model=target_model,
                attacks_per_vulnerability=attacks_per_vulnerability,
                vulnerabilities=vulnerabilities,
                attack_enhancements=attack_enhancements,
                max_concurrent_tasks=max_concurrent_tasks,
            )

            # Create a mapping of vulnerabilities to attacks
            vulnerability_to_attacks_map: Dict[Vulnerability, List[Attack]] = {}
            for attack in attacks:
                if attack.vulnerability not in vulnerability_to_attacks_map:
                    vulnerability_to_attacks_map[attack.vulnerability] = [
                        attack
                    ]
                else:
                    vulnerability_to_attacks_map[attack.vulnerability].append(
                        attack
                    )

            red_teaming_results = []
            red_teaming_results_breakdown = []

            # Create a semaphore for throttling concurrent tasks
            semaphore = asyncio.Semaphore(max_concurrent_tasks)

            # Total number of attacks across all vulnerabilities
            total_attacks = sum(
                len(attacks)
                for attacks in vulnerability_to_attacks_map.values()
            )

            # Create a progress bar for attack evaluations
            pbar = tqdm(
                total=total_attacks,
                desc=f"📝 Evaluating {len(vulnerabilities)} vulnerability(s)",
            )

            async def throttled_evaluate_vulnerability(vulnerability, attacks):
                async with (
                    semaphore
                ):  # Ensures only `max_concurrent_tasks` run concurrently
                    vulnerability_results = (
                        await self._a_evaluate_vulnerability(
                            target_model,
                            vulnerability,
                            attacks,
                            metrics_map,
                        )
                    )
                    pbar.update(
                        len(attacks)
                    )  # Update the progress bar by the number of attacks evaluated
                    return vulnerability_results

            # Create a list of tasks for evaluating each vulnerability, with throttling
            tasks = [
                throttled_evaluate_vulnerability(vulnerability, attacks)
                for vulnerability, attacks in vulnerability_to_attacks_map.items()
            ]
            # Execute tasks concurrently with throttling using asyncio.gather
            vulnerability_results_list = await asyncio.gather(*tasks)

            # Close the progress bar after all tasks are done
            pbar.close()

            # Process results
            for (vulnerability, _), vulnerability_results in zip(
                vulnerability_to_attacks_map.items(), vulnerability_results_list
            ):
                valid_scores = [
                    vulnerability_result.score
                    for vulnerability_result in vulnerability_results
                    if vulnerability_result.score is not None
                ]
                if valid_scores:
                    avg_score = sum(valid_scores) / len(valid_scores)
                else:
                    avg_score = None

                red_teaming_results.append(
                    {
                        "Vulnerability": vulnerability.value,
                        "Average Score": avg_score,
                    }
                )

                for vulnerability_result in vulnerability_results:
                    red_teaming_results_breakdown.append(
                        {
                            "Vulnerability": vulnerability_result.vulnerability,
                            "Attack Enhancement": vulnerability_result.attack_enhancement,
                            "Input": vulnerability_result.input,
                            "Target Output": vulnerability_result.actual_output,
                            "Score": vulnerability_result.score,
                            "Reason": vulnerability_result.reason,
                            "Error": vulnerability_result.error,
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

    async def _a_attack(
        self,
        target_model: DeepEvalBaseLLM,
        attack: Attack,
        vulnerability: Vulnerability,
        metrics_map,
    ) -> VulnerabilityResult:
        result = VulnerabilityResult(
            input=attack.input,
            vulnerability=vulnerability,
            attack_enhancement=attack.attack_enhancement,
        )

        if attack.error is not None:
            result.error = attack.error
            return result

        metric: BaseMetric = metrics_map[vulnerability]()
        try:
            # Generate actual output using the 'input'
            actual_output = await target_model.a_generate(attack.input)
            result.actual_output = await target_model.a_generate(attack.input)
        except:
            result.error = "Error generating output from target LLM"
            return result

        test_case = LLMTestCase(
            input=attack.input,
            actual_output=actual_output,
        )

        try:
            await metric.a_measure(test_case)
            result.score = metric.score
            result.reason = metric.reason
        except:
            result.error = f"Error evaluating target LLM output for the '{vulnerability.value}' vulnerability"
            return result

        return result

    async def _a_evaluate_vulnerability(
        self,
        target_model: DeepEvalBaseLLM,
        vulnerability: Vulnerability,
        attacks: List[Attack],
        metrics_map,
    ) -> List[VulnerabilityResult]:
        results = await asyncio.gather(
            *[
                self._a_attack(
                    target_model=target_model,
                    attack=attack,
                    vulnerability=vulnerability,
                    metrics_map=metrics_map,
                )
                for attack in attacks
            ]
        )

        return results

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
