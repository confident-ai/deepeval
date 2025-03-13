import json
import random
import asyncio
import requests
import aiohttp
from tqdm import tqdm
from pydantic import BaseModel
from typing import List, Optional, Union, Dict

from deepeval.red_teaming.types import (
    AttackEnhancement,
    VulnerabilityType,
    CallbackType,
)
from deepeval.red_teaming.utils import generate_schema, a_generate_schema
from deepeval.red_teaming.template import RedTeamSynthesizerTemplate
from deepeval.red_teaming.schema import (
    Attack,
    ApiGenerateBaselineAttack,
    GenerateBaselineAttackResponseData,
)
from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM
from deepeval.synthesizer.schema import *
from deepeval.synthesizer.types import *
from deepeval.red_teaming.attack_enhancements import (
    Base64,
    GrayBox,
    JailbreakingLinear,
    JailbreakingTree,
    Leetspeak,
    PromptInjection,
    PromptProbing,
    Rot13,
    MathProblem,
    Multilingual,
    JailbreakingCrescendo,
)
from deepeval.confident.api import Api, HttpMethods, Endpoints
from deepeval.vulnerability import BaseVulnerability
from deepeval.utils import is_confident
import os

BASE_URL = "https://deepeval.confident-ai.com"


class AttackSynthesizer:
    def __init__(
        self,
        purpose: str,
        system_prompt: str,
        synthesizer_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    ):
        # Initialize models and async mode
        self.purpose = purpose
        self.system_prompt = system_prompt
        self.synthesizer_model, self.using_native_model = initialize_model(
            synthesizer_model
        )

        # Define list of attacks and unaligned vulnerabilities
        self.synthetic_attacks: List[Attack] = []

    ##################################################
    ### Generating Attacks ###########################
    ##################################################

    def generate_attacks(
        self,
        target_model_callback: CallbackType,
        attacks_per_vulnerability_type: int,
        vulnerabilities: List[BaseVulnerability],
        attack_enhancements: Dict[AttackEnhancement, float],
        ignore_errors: bool,
    ) -> List[Attack]:
        # Generate unenhanced attacks for each vulnerability
        base_attacks: List[Attack] = []
        num_vulnerability_types = sum(
            len(v.get_types()) for v in vulnerabilities
        )
        pbar = tqdm(
            vulnerabilities,
            desc=f"💥 Generating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (for {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerabilities)",
        )
        for vulnerability in pbar:
            base_attacks.extend(
                self.generate_base_attacks(
                    attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                    vulnerability=vulnerability,
                    ignore_errors=ignore_errors,
                )
            )

        # Enhance attacks by sampling from the provided distribution
        enhanced_attacks: List[Attack] = []
        pbar = tqdm(
            base_attacks,
            desc=f"✨ Enhancing {num_vulnerability_types * attacks_per_vulnerability_type} attacks (using {len(attack_enhancements.keys())} enhancements)",
        )
        attack_enhancement_choices = list(attack_enhancements.keys())
        enhancement_weights = list(
            attack_enhancements.values()
        )  # Weights based on distribution

        for base_attack in pbar:
            # Randomly sample an enhancement based on the distribution
            sampled_enhancement = random.choices(
                attack_enhancement_choices, weights=enhancement_weights, k=1
            )[0]

            enhanced_attack = self.enhance_attack(
                target_model_callback=target_model_callback,
                base_attack=base_attack,
                attack_enhancement=sampled_enhancement,
                ignore_errors=ignore_errors,
            )
            enhanced_attacks.append(enhanced_attack)

        self.synthetic_attacks.extend(enhanced_attacks)
        return enhanced_attacks

    async def a_generate_attacks(
        self,
        target_model_callback: CallbackType,
        attacks_per_vulnerability_type: int,
        vulnerabilities: List[BaseVulnerability],
        attack_enhancements: Dict[AttackEnhancement, float],
        ignore_errors: bool,
        max_concurrent: int = 10,
    ) -> List[Attack]:
        # Create a semaphore to control the number of concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent)

        # Generate unenhanced attacks for each vulnerability
        base_attacks: List[Attack] = []
        num_vulnerability_types = sum(
            len(v.get_types()) for v in vulnerabilities
        )
        pbar = tqdm(
            vulnerabilities,
            desc=f"💥 Generating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (for {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerabilities)",
        )

        async def throttled_generate_base_attack(vulnerability):
            async with semaphore:  # Throttling applied here
                result = await self.a_generate_base_attacks(
                    attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                    vulnerability=vulnerability,
                    ignore_errors=ignore_errors,
                )
                pbar.update(1)
                return result

        generate_tasks = [
            asyncio.create_task(throttled_generate_base_attack(vulnerability))
            for vulnerability in vulnerabilities
        ]

        attack_results = await asyncio.gather(*generate_tasks)
        for result in attack_results:
            base_attacks.extend(result)
        pbar.close()

        # Enhance attacks by sampling from the provided distribution
        enhanced_attacks: List[Attack] = []
        pbar = tqdm(
            total=len(base_attacks),
            desc=f"✨ Enhancing {num_vulnerability_types * attacks_per_vulnerability_type} attacks (using {len(attack_enhancements.keys())} enhancements)",
        )

        async def throttled_attack_enhancement(base_attack):
            async with semaphore:  # Throttling applied here
                # Randomly sample an enhancement based on the distribution
                attack_enhancement_choices = list(attack_enhancements.keys())
                enhancement_weights = list(attack_enhancements.values())
                sampled_enhancement = random.choices(
                    attack_enhancement_choices, weights=enhancement_weights, k=1
                )[0]
                result = await self.a_enhance_attack(
                    target_model_callback=target_model_callback,
                    base_attack=base_attack,
                    attack_enhancement=sampled_enhancement,
                    ignore_errors=ignore_errors,
                )
                pbar.update(1)
                return result

        enhanced_attacks.extend(
            await asyncio.gather(
                *[
                    asyncio.create_task(
                        throttled_attack_enhancement(base_attack)
                    )
                    for base_attack in base_attacks
                ]
            )
        )
        pbar.close()

        # Store the generated and enhanced attacks
        self.synthetic_attacks.extend(enhanced_attacks)

        return enhanced_attacks

    ##################################################
    ### Generating Base (Unenhanced) Attacks #########
    ##################################################

    def generate_base_attacks(
        self,
        attacks_per_vulnerability_type: int,
        vulnerability: BaseVulnerability,
        ignore_errors: bool,
    ) -> List[Attack]:
        base_attacks: List[Attack] = []

        for vulnerability_type in vulnerability.get_types():
            try:
                remote_attacks = self.generate_remote_attack(
                    self.purpose,
                    vulnerability_type,
                    attacks_per_vulnerability_type,
                )
                base_attacks.extend(
                    [
                        Attack(
                            vulnerability=vulnerability.get_name(),
                            vulnerability_type=vulnerability_type,
                            input=remote_attack,
                        )
                        for remote_attack in remote_attacks
                    ]
                )
            except:
                if ignore_errors:
                    for _ in range(attacks_per_vulnerability_type):
                        base_attacks.append(
                            Attack(
                                vulnerability=vulnerability.get_name(),
                                vulnerability_type=vulnerability_type,
                                error="Error generating aligned attacks.",
                            )
                        )
                else:
                    raise
        return base_attacks

    async def a_generate_base_attacks(
        self,
        attacks_per_vulnerability_type: int,
        vulnerability: BaseVulnerability,
        ignore_errors: bool,
    ) -> List[Attack]:
        base_attacks: List[Attack] = []
        for vulnerability_type in vulnerability.get_types():
            try:
                remote_attacks = self.generate_remote_attack(
                    self.purpose,
                    vulnerability_type,
                    attacks_per_vulnerability_type,
                )
                base_attacks.extend(
                    [
                        Attack(
                            vulnerability=vulnerability.get_name(),
                            vulnerability_type=vulnerability_type,
                            input=remote_attack,
                        )
                        for remote_attack in remote_attacks
                    ]
                )
            except:
                if ignore_errors:
                    for _ in range(attacks_per_vulnerability_type):
                        base_attacks.append(
                            Attack(
                                vulnerability=vulnerability.get_name(),
                                vulnerability_type=vulnerability_type,
                                error="Error generating aligned attacks.",
                            )
                        )
                else:
                    raise
        return base_attacks

    ##################################################
    ### Enhance attacks ##############################
    ##################################################

    def enhance_attack(
        self,
        target_model_callback: CallbackType,
        base_attack: Attack,
        attack_enhancement: AttackEnhancement,
        ignore_errors: bool,
        jailbreaking_iterations: int = 5,
    ):
        attack_input = base_attack.input
        if attack_input is None:
            return base_attack

        base_attack.attack_enhancement = attack_enhancement.value
        try:
            if attack_enhancement == AttackEnhancement.BASE64:
                enhanced_attack = Base64().enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.ROT13:
                enhanced_attack = Rot13().enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.LEETSPEAK:
                enhanced_attack = Leetspeak().enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.PROMPT_INJECTION:
                enhanced_attack = PromptInjection().enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.GRAY_BOX_ATTACK:
                enhanced_attack = GrayBox(
                    self.synthesizer_model, self.using_native_model
                ).enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.PROMPT_PROBING:
                enhanced_attack = PromptProbing(
                    self.synthesizer_model, self.using_native_model
                ).enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.JAILBREAK_LINEAR:
                enhanced_attack = JailbreakingLinear(
                    target_model_callback=target_model_callback,
                    synthesizer_model=self.synthesizer_model,
                    using_native_model=self.using_native_model,
                ).enhance(attack_input, jailbreaking_iterations)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.JAILBREAK_TREE:
                enhanced_attack = JailbreakingTree(
                    target_model_callback=target_model_callback,
                    synthesizer_model=self.synthesizer_model,
                    using_native_model=self.using_native_model,
                ).enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.MATH_PROBLEM:
                enhanced_attack = MathProblem(
                    self.synthesizer_model, self.using_native_model
                ).enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.MULTILINGUAL:
                enhanced_attack = Multilingual(
                    self.synthesizer_model, self.using_native_model
                ).enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.JAILBREAK_CRESCENDO:
                enhanced_attack = JailbreakingCrescendo(
                    target_model_callback=target_model_callback,
                    synthesizer_model=self.synthesizer_model,
                    using_native_model=self.using_native_model,
                ).enhance(attack_input)
                base_attack.input = enhanced_attack
        except:
            if ignore_errors:
                base_attack.error = "Error enhancing attack"
                return base_attack
            else:
                raise

        return base_attack

    async def a_enhance_attack(
        self,
        target_model_callback: CallbackType,
        base_attack: Attack,
        attack_enhancement: AttackEnhancement,
        ignore_errors: bool,
        jailbreaking_iterations: int = 5,
    ):
        attack_input = base_attack.input
        if attack_input is None:
            return base_attack

        base_attack.attack_enhancement = attack_enhancement.value
        try:
            if attack_enhancement == AttackEnhancement.BASE64:
                enhanced_attack = await Base64().a_enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.ROT13:
                enhanced_attack = await Rot13().a_enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.LEETSPEAK:
                enhanced_attack = await Leetspeak().a_enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.PROMPT_INJECTION:
                enhanced_attack = await PromptInjection().a_enhance(
                    attack_input
                )
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.GRAY_BOX_ATTACK:
                enhanced_attack = await GrayBox(
                    self.synthesizer_model, self.using_native_model
                ).a_enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.PROMPT_PROBING:
                enhanced_attack = await PromptProbing(
                    self.synthesizer_model, self.using_native_model
                ).a_enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.JAILBREAK_LINEAR:
                enhanced_attack = await JailbreakingLinear(
                    target_model_callback=target_model_callback,
                    synthesizer_model=self.synthesizer_model,
                    using_native_model=self.using_native_model,
                ).a_enhance(attack_input, jailbreaking_iterations)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.JAILBREAK_TREE:
                enhanced_attack = await JailbreakingTree(
                    target_model_callback=target_model_callback,
                    synthesizer_model=self.synthesizer_model,
                    using_native_model=self.using_native_model,
                ).a_enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.MATH_PROBLEM:
                enhanced_attack = await MathProblem(
                    self.synthesizer_model, self.using_native_model
                ).a_enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.MULTILINGUAL:
                enhanced_attack = await Multilingual(
                    self.synthesizer_model, self.using_native_model
                ).a_enhance(attack_input)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.JAILBREAK_CRESCENDO:
                enhanced_attack = await JailbreakingCrescendo(
                    target_model_callback=target_model_callback,
                    synthesizer_model=self.synthesizer_model,
                    using_native_model=self.using_native_model,
                ).a_enhance(attack_input)
                base_attack.input = enhanced_attack
        except:
            if ignore_errors:
                base_attack.error = "Error enhancing attack"
                return base_attack
            else:
                raise

        return base_attack

    ##################################################
    ### Utils ########################################
    ##################################################

    def _generate_schema(self, prompt: str, schema: BaseModel):
        return generate_schema(
            prompt, schema, self.using_native_model, self.synthesizer_model
        )

    async def _a_generate_schema(self, prompt: str, schema: BaseModel):
        return await a_generate_schema(
            prompt, schema, self.using_native_model, self.synthesizer_model
        )

    def generate_remote_attack(
        self,
        purpose: str,
        vulnerability_type: VulnerabilityType,
        num_attacks: int,
    ) -> List[Attack]:

        # Prepare parameters for API request
        guard_params = ApiGenerateBaselineAttack(
            purpose=purpose,
            vulnerability=vulnerability_type.value,
            num_attacks=num_attacks,
        )
        body = guard_params.model_dump(by_alias=True, exclude_none=True)

        api = Api(base_url=BASE_URL, api_key="NA")

        try:
            # API request
            response = api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.BASELINE_ATTACKS_ENDPOINT,
                body=body,
            )
        except Exception as e:
            print(e)

        return GenerateBaselineAttackResponseData(**response).baseline_attacks
