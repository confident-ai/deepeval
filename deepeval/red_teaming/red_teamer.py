import asyncio
from tqdm import tqdm
from typing import Dict, List, Optional, Union

from deepeval.vulnerability import BaseVulnerability
from deepeval.red_teaming.types import (
    AttackEnhancement,
    VulnerabilityType,
    CallbackType,
    llm_risk_categories_map,
)

from deepeval.vulnerability.intellectual_property import (
    IntellectualPropertyType,
)
from deepeval.vulnerability.unauthorized_access import UnauthorizedAccessType
from deepeval.vulnerability.illegal_activity import IllegalActivityType
from deepeval.vulnerability.excessive_agency import ExcessiveAgencyType
from deepeval.vulnerability.personal_safety import PersonalSafetyType
from deepeval.vulnerability.graphic_content import GraphicContentType
from deepeval.vulnerability.misinformation import MisinformationType
from deepeval.vulnerability.prompt_leakage import PromptLeakageType
from deepeval.vulnerability.competition import CompetitionType
from deepeval.vulnerability.pii_leakage import PIILeakageType
from deepeval.vulnerability.robustness import RobustnessType
from deepeval.vulnerability.toxicity import ToxicityType
from deepeval.vulnerability.bias import BiasType
from deepeval.red_teaming.types import VulnerabilityType

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
import inspect


class VulnerabilityResult(BaseModel):
    vulnerability: str
    vulnerability_type: VulnerabilityType
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
        target_model_callback: CallbackType,
        attacks_per_vulnerability_type: int,
        vulnerabilities: List[BaseVulnerability],
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
        max_concurrent: int = 10,
        ignore_errors: bool = False,
    ):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install pandas to use this method. 'pip install pandas'"
            )
        if self.async_mode:
            assert inspect.iscoroutinefunction(
                target_model_callback
            ), "`target_model_callback` needs to be async. `async_mode` has been set to True."
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.a_scan(
                    target_model_callback,
                    attacks_per_vulnerability_type,
                    vulnerabilities,
                    attack_enhancements,
                    max_concurrent,
                    ignore_errors=ignore_errors,
                )
            )
        else:
            assert not inspect.iscoroutinefunction(
                target_model_callback
            ), "`target_model_callback` needs to be sync. `async_mode` has been set to False."
            with capture_red_teamer_run(
                attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                vulnerabilities=vulnerabilities,
                attack_enhancements=attack_enhancements,
            ):
                # Initialize metric map
                metrics_map = self.get_red_teaming_metrics_map()

                # Generate attacks
                attacks: List[Attack] = (
                    self.attack_synthesizer.generate_attacks(
                        target_model_callback=target_model_callback,
                        attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                        vulnerabilities=vulnerabilities,
                        attack_enhancements=attack_enhancements,
                        ignore_errors=ignore_errors,
                    )
                )

                # Create a mapping of vulnerabilities to attacks
                vulnerability_to_attacks_map: Dict[
                    VulnerabilityType, List[Attack]
                ] = {}
                for attack in attacks:
                    if (
                        attack.vulnerability_type
                        not in vulnerability_to_attacks_map
                    ):
                        vulnerability_to_attacks_map[
                            attack.vulnerability_type
                        ] = [attack]
                    else:
                        vulnerability_to_attacks_map[
                            attack.vulnerability_type
                        ].append(attack)

                # Evaluate each attack by vulnerability
                red_teaming_results = []
                red_teaming_results_breakdown = []

                num_vulnerability_types = sum(
                    len(v.get_types()) for v in vulnerabilities
                )
                pbar = tqdm(
                    vulnerability_to_attacks_map.items(),
                    desc=f"📝 Evaluating {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerabilities",
                )
                for vulnerability_type, attacks in pbar:
                    scores = []
                    for attack in attacks:
                        metric: BaseMetric = metrics_map.get(
                            vulnerability_type
                        )()
                        risk = llm_risk_categories_map.get(vulnerability_type)
                        result = {
                            "Vulnerability": attack.vulnerability,
                            "Vulnerability Type": vulnerability_type.value,
                            "Attack Enhancement": attack.attack_enhancement,
                            "Risk Category": (
                                risk.value if risk is not None else "Others"
                            ),
                            "Input": attack.input,
                            "Target Output": None,
                            "Score": None,
                            "Reason": None,
                            "Error": None,
                        }

                        # this will only go through if ignore_errors == True
                        if attack.error:
                            result["Error"] = attack.error
                            red_teaming_results_breakdown.append(result)
                            continue

                        try:
                            target_output = target_model_callback(attack.input)
                            result["Target Output"] = target_output
                        except Exception:
                            if ignore_errors:
                                result["Error"] = (
                                    "Error generating output from target LLM"
                                )
                                red_teaming_results_breakdown.append(result)
                                continue
                            else:
                                raise

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
                            if ignore_errors:
                                result["Error"] = (
                                    f"Error evaluating target LLM output for the '{vulnerability_type.value}' vulnerability"
                                )
                                red_teaming_results_breakdown.append(result)
                                continue
                            else:
                                raise

                        red_teaming_results_breakdown.append(result)

                    # Calculate average score for each vulnerability
                    avg_vulnerability_score = (
                        sum(scores) / len(scores) if scores else None
                    )
                    red_teaming_results.append(
                        {
                            "Vulnerability": attack.vulnerability,
                            "Vulnerability Type": vulnerability_type.value,
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
        target_model_callback: CallbackType,
        attacks_per_vulnerability_type: int,
        vulnerabilities: List[BaseVulnerability],
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
        max_concurrent: int = 10,  # Throttling limit, control concurrency
        ignore_errors: bool = False,
    ):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install pandas to use this method. 'pip install pandas'"
            )
        with capture_red_teamer_run(
            attacks_per_vulnerability_type=attacks_per_vulnerability_type,
            vulnerabilities=vulnerabilities,
            attack_enhancements=attack_enhancements,
        ):
            # Initialize metric map
            metrics_map = self.get_red_teaming_metrics_map()

            # Generate attacks
            attacks: List[Attack] = (
                await self.attack_synthesizer.a_generate_attacks(
                    target_model_callback=target_model_callback,
                    attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                    vulnerabilities=vulnerabilities,
                    attack_enhancements=attack_enhancements,
                    ignore_errors=ignore_errors,
                    max_concurrent=max_concurrent,
                )
            )

            # Create a mapping of vulnerabilities to attacks
            vulnerability_type_to_attacks_map: Dict[
                VulnerabilityType, List[Attack]
            ] = {}
            for attack in attacks:
                if (
                    attack.vulnerability_type
                    not in vulnerability_type_to_attacks_map
                ):
                    vulnerability_type_to_attacks_map[
                        attack.vulnerability_type
                    ] = [attack]
                else:
                    vulnerability_type_to_attacks_map[
                        attack.vulnerability_type
                    ].append(attack)

            red_teaming_results = []
            red_teaming_results_breakdown = []

            # Create a semaphore for throttling concurrent tasks
            semaphore = asyncio.Semaphore(max_concurrent)

            # Total number of attacks across all vulnerabilities
            total_attacks = sum(
                len(attacks)
                for attacks in vulnerability_type_to_attacks_map.values()
            )
            # Create a progress bar for attack evaluations
            num_vulnerability_types = sum(
                len(v.get_types()) for v in vulnerabilities
            )
            pbar = tqdm(
                total=total_attacks,
                desc=f"📝 Evaluating {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerabilities",
            )

            async def throttled_evaluate_vulnerability_type(
                vulnerability_type, attacks
            ):
                async with (
                    semaphore
                ):  # Ensures only `max_concurrent` run concurrently
                    vulnerability_results = (
                        await self._a_evaluate_vulnerability_type(
                            target_model_callback,
                            vulnerability_type,
                            attacks,
                            metrics_map,
                            ignore_errors=ignore_errors,
                        )
                    )
                    pbar.update(
                        len(attacks)
                    )  # Update the progress bar by the number of attacks evaluated
                    return vulnerability_results

            # Create a list of tasks for evaluating each vulnerability, with throttling
            tasks = [
                throttled_evaluate_vulnerability_type(
                    vulnerability_type, attacks
                )
                for vulnerability_type, attacks in vulnerability_type_to_attacks_map.items()
            ]
            # Execute tasks concurrently with throttling using asyncio.gather
            vulnerability_results_list = await asyncio.gather(*tasks)

            # Close the progress bar after all tasks are done
            pbar.close()

            # Process results
            for (vulnerability_type, attacks), vulnerability_results in zip(
                vulnerability_type_to_attacks_map.items(),
                vulnerability_results_list,
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
                        "Vulnerability": attacks[0].vulnerability,
                        "Vulnerability Type": vulnerability_type,
                        "Average Score": avg_score,
                    }
                )

                for vulnerability_result in vulnerability_results:
                    risk = llm_risk_categories_map.get(
                        vulnerability_result.vulnerability
                    )
                    red_teaming_results_breakdown.append(
                        {
                            "Vulnerability": vulnerability_result.vulnerability,
                            "Vulnerability Type": vulnerability_result.vulnerability_type,
                            "Attack Enhancement": vulnerability_result.attack_enhancement,
                            "Risk Category": (
                                risk.value if risk is not None else "Others"
                            ),
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
        target_model_callback: CallbackType,
        attack: Attack,
        vulnerability: str,
        vulnerability_type: VulnerabilityType,
        metrics_map,
        ignore_errors: bool,
    ) -> VulnerabilityResult:
        result = VulnerabilityResult(
            input=attack.input,
            vulnerability=vulnerability,
            vulnerability_type=vulnerability_type,
            attack_enhancement=attack.attack_enhancement,
        )

        if attack.error is not None:
            result.error = attack.error
            return result

        metric: BaseMetric = metrics_map[vulnerability_type]()
        try:
            # Generate actual output using the 'input'
            actual_output = await target_model_callback(attack.input)
            result.actual_output = actual_output
        except Exception:
            if ignore_errors:
                result.error = "Error generating output from target LLM"
                return result
            else:
                raise

        test_case = LLMTestCase(
            input=attack.input,
            actual_output=actual_output,
        )

        try:
            await metric.a_measure(test_case)
            result.score = metric.score
            result.reason = metric.reason
        except:
            if ignore_errors:
                result.error = f"Error evaluating target LLM output for the '{vulnerability_type.value}' vulnerability type"
                return result
            else:
                raise

        return result

    async def _a_evaluate_vulnerability_type(
        self,
        target_model_callback: CallbackType,
        vulnerability_type: VulnerabilityType,
        attacks: List[Attack],
        metrics_map,
        ignore_errors: bool,
    ) -> List[VulnerabilityResult]:
        results = await asyncio.gather(
            *[
                self._a_attack(
                    target_model_callback=target_model_callback,
                    attack=attack,
                    vulnerability=attack.vulnerability,
                    vulnerability_type=vulnerability_type,
                    metrics_map=metrics_map,
                    ignore_errors=ignore_errors,
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
            #### Bias ####
            **{
                bias_type: lambda: BiasGrader(
                    model=self.evaluation_model,
                    purpose=self.target_purpose,
                    async_mode=self.async_mode,
                )
                for bias_type in BiasType
            },
            #### Toxicity ####
            **{
                toxicity_type: lambda tt=toxicity_type: HarmGrader(
                    model=self.evaluation_model,
                    harm_category=tt.value,
                    async_mode=self.async_mode,
                )
                for toxicity_type in ToxicityType
            },
            #### Misinformation ####
            **{
                misinformation_type: lambda mt=misinformation_type: HarmGrader(
                    model=self.evaluation_model,
                    harm_category=mt.value,
                    async_mode=self.async_mode,
                )
                for misinformation_type in MisinformationType
            },
            #### Illegal ####
            **{
                illegal_activity_type: lambda iat=illegal_activity_type: HarmGrader(
                    model=self.evaluation_model,
                    harm_category=iat.value,
                    async_mode=self.async_mode,
                )
                for illegal_activity_type in IllegalActivityType
            },
            #### Prompt Leakage ####
            **{
                prompt_leakage_type: lambda: PromptExtractionGrader(
                    model=self.evaluation_model,
                    purpose=self.target_purpose,
                    async_mode=self.async_mode,
                )
                for prompt_leakage_type in PromptLeakageType
            },
            #### PII Leakage ####
            **{
                pii_type: lambda: PIIGrader(
                    model=self.evaluation_model,
                    purpose=self.target_purpose,
                    async_mode=self.async_mode,
                )
                for pii_type in PIILeakageType
            },
            #### Unauthorized Access ####
            UnauthorizedAccessType.DEBUG_ACCESS: lambda: DebugAccessGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            UnauthorizedAccessType.RBAC: lambda: RBACGrader(
                model=self.evaluation_model,
                purpose=self.target_purpose,
                async_mode=self.async_mode,
            ),
            UnauthorizedAccessType.SHELL_INJECTION: lambda: ShellInjectionGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            UnauthorizedAccessType.SQL_INJECTION: lambda: SQLInjectionGrader(
                model=self.evaluation_model, async_mode=self.async_mode
            ),
            UnauthorizedAccessType.BFLA: lambda: BFLAGrader(
                purpose=self.target_purpose,
                model=self.evaluation_model,
                async_mode=self.async_mode,
            ),
            UnauthorizedAccessType.BOLA: lambda: BOLAGrader(
                model=self.evaluation_model,
                async_mode=self.async_mode,
            ),
            UnauthorizedAccessType.SSRF: lambda: SSRFGrader(
                purpose=self.target_purpose,
                model=self.evaluation_model,
                async_mode=self.async_mode,
            ),
            #### Excessive Agency ####
            **{
                excessive_agency_type: lambda: ExcessiveAgencyGrader(
                    model=self.evaluation_model,
                    purpose=self.target_purpose,
                    async_mode=self.async_mode,
                )
                for excessive_agency_type in ExcessiveAgencyType
            },
            #### Robustness ####
            RobustnessType.HIJACKING: lambda: HijackingGrader(
                purpose=self.target_purpose,
                model=self.evaluation_model,
                async_mode=self.async_mode,
            ),
            RobustnessType.INPUT_OVERRELIANCE: lambda: OverrelianceGrader(
                purpose=self.target_purpose,
                model=self.evaluation_model,
                async_mode=self.async_mode,
            ),
            #### Intellectual Property ####
            **{
                ip_type: lambda: IntellectualPropertyGrader(
                    model=self.evaluation_model,
                    purpose=self.target_purpose,
                    async_mode=self.async_mode,
                )
                for ip_type in IntellectualPropertyType
            },
            #### Competition ####
            **{
                competiton_type: lambda: CompetitorsGrader(
                    model=self.evaluation_model,
                    purpose=self.target_purpose,
                    async_mode=self.async_mode,
                )
                for competiton_type in CompetitionType
            },
            #### Graphic Content ####
            **{
                content_type: lambda ct=content_type: HarmGrader(
                    model=self.evaluation_model,
                    harm_category=ct.value,
                    async_mode=self.async_mode,
                )
                for content_type in GraphicContentType
            },
            #### Personal Safety ####
            **{
                safety_type: lambda st=safety_type: HarmGrader(
                    model=self.evaluation_model,
                    harm_category=st.value,
                    async_mode=self.async_mode,
                )
                for safety_type in PersonalSafetyType
            },
        }
        self.metrics_map = metrics_map
        return metrics_map
