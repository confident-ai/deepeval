import copy
import json
import time
import base64
import random
import asyncio
import requests
import aiohttp
from tqdm import tqdm
from pydantic import BaseModel
from typing import List, Optional, Union, Dict

from deepeval.red_teaming.types import (
    AttackEnhancement,
    Vulnerability,
    UnalignedVulnerability,
)
from deepeval.red_teaming.utils import generate_schema, a_generate_schema
from deepeval.red_teaming.template import RedTeamSynthesizerTemplate
from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.utils import get_or_create_event_loop
from deepeval.models import DeepEvalBaseLLM
from deepeval.dataset.golden import Golden
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
)


class AttackSynthesizer:
    def __init__(
        self,
        purpose: str,
        system_prompt: str,
        target_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        synthesizer_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        async_mode: bool = True,
    ):
        # Initialize models and async mode
        self.purpose = purpose
        self.system_prompt = system_prompt
        self.target_model = target_model
        self.synthesizer_model, self.using_native_model = initialize_model(
            synthesizer_model
        )
        self.async_mode = async_mode

        # Define list of attacks and unaligned vulnerabilities
        self.synthetic_attacks: List[Golden] = []
        self.unaligned_vulnerabilities = [
            item.value for item in UnalignedVulnerability
        ]

    ##################################################
    ### Generating Attacks ###########################
    ##################################################

    def generate_attacks(
        self,
        attacks_per_vulnerability: int,
        vulnerabilities: List["Vulnerability"],
        attack_enhancements: Dict["AttackEnhancement", float],
        max_concurrent_tasks: int = 10,
    ) -> List["Golden"]:
        if self.async_mode:
            # Run async function
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.a_generate_attacks(
                    attacks_per_vulnerability=attacks_per_vulnerability,
                    vulnerabilities=vulnerabilities,
                    attack_enhancements=attack_enhancements,
                    max_concurrent_tasks=max_concurrent_tasks,
                )
            )
        else:
            # Generate unenhanced attacks for each vulnerability
            attacks: List[Golden] = []
            pbar = tqdm(
                vulnerabilities,
                desc=f"💥 Generating {len(vulnerabilities) * attacks_per_vulnerability} attacks (for {len(vulnerabilities)} vulnerabilties)",
            )
            for vulnerability in pbar:
                attacks_for_vulnerability = self.generate_base_attacks(
                    attacks_per_vulnerability,
                    vulnerability,
                )
                attacks.extend(attacks_for_vulnerability)

            # Enhance attacks by sampling from the provided distribution
            enhanced_attacks: List[Golden] = []
            pbar = tqdm(
                attacks,
                desc=f"✨ Enhancing {len(vulnerabilities) * attacks_per_vulnerability} attacks (using {len(attack_enhancements.keys())} enhancements)",
            )
            attack_enhancement_choices = list(attack_enhancements.keys())
            enhancement_weights = list(
                attack_enhancements.values()
            )  # Weights based on distribution

            for attack in pbar:
                # Randomly sample an enhancement based on the distribution
                sampled_enhancement = random.choices(
                    attack_enhancement_choices, weights=enhancement_weights, k=1
                )[0]

                enhanced_attack = self.enhance_attack(
                    attack, sampled_enhancement
                )
                enhanced_attacks.append(enhanced_attack)

            self.synthetic_attacks.extend(enhanced_attacks)
            return enhanced_attacks

    async def a_generate_attacks(
        self,
        attacks_per_vulnerability: int,
        vulnerabilities: List["Vulnerability"],
        attack_enhancements: Dict["AttackEnhancement", float],
        max_concurrent_tasks: int = 10,
    ) -> List["Golden"]:
        # Create a semaphore to control the number of concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Generate unenhanced attacks for each vulnerability
        attacks: List[Golden] = []
        pbar = tqdm(
            vulnerabilities,
            desc=f"💥 Generating {len(vulnerabilities) * attacks_per_vulnerability} attacks (for {len(vulnerabilities)} vulnerabilities)",
        )

        async def throttled_generate_base_attack(vulnerability):
            async with semaphore:  # Throttling applied here
                result = await self.a_generate_base_attacks(
                    attacks_per_vulnerability, vulnerability
                )
                pbar.update(1)
                return result

        generate_tasks = [
            asyncio.create_task(throttled_generate_base_attack(vulnerability))
            for vulnerability in vulnerabilities
        ]

        attack_results = await asyncio.gather(*generate_tasks)
        for result in attack_results:
            attacks.extend(result)
        pbar.close()

        # Enhance attacks by sampling from the provided distribution
        enhanced_attacks: List[Golden] = []
        total_enhancements = len(attacks)
        pbar = tqdm(
            total=total_enhancements,
            desc=f"✨ Enhancing {len(vulnerabilities) * attacks_per_vulnerability} attacks (using {len(attack_enhancements.keys())} enhancements)",
        )

        async def throttled_enhance_attack(attack):
            async with semaphore:  # Throttling applied here
                # Randomly sample an enhancement based on the distribution
                attack_enhancement_choices = list(attack_enhancements.keys())
                enhancement_weights = list(attack_enhancements.values())
                sampled_enhancement = random.choices(
                    attack_enhancement_choices, weights=enhancement_weights, k=1
                )[0]
                result = await self.a_enhance_attack(
                    attack, sampled_enhancement
                )
                pbar.update(1)
                return result

        enhance_tasks = [
            asyncio.create_task(throttled_enhance_attack(attack))
            for attack in attacks
        ]

        enhanced_attack_results = await asyncio.gather(*enhance_tasks)
        enhanced_attacks.extend(enhanced_attack_results)
        pbar.close()

        # Store the generated and enhanced attacks
        self.synthetic_attacks.extend(enhanced_attacks)

        return enhanced_attacks

    ##################################################
    ### Generating Base (Unenhanced) Attacks #########
    ##################################################

    def generate_base_attacks(
        self,
        attacks_per_vulnerability: int,
        vulnerability: Vulnerability,
        max_retries: int = 5,
    ) -> List[Golden]:
        goldens: List[Golden] = []

        # Unaligned vulnerabilities
        if vulnerability.value in self.unaligned_vulnerabilities:
            goldens.extend(
                Golden(
                    input=self.generate_unaligned_attack(
                        self.purpose, vulnerability.value
                    ),
                    additional_metadata={"vulnerability": vulnerability},
                )
                for _ in range(attacks_per_vulnerability)
            )

        # Aligned vulnerabilities: LLMs can generate
        else:
            prompt = RedTeamSynthesizerTemplate.generate_attacks(
                attacks_per_vulnerability, vulnerability, self.purpose
            )

            # Generate attacks with retries
            for _ in range(max_retries):
                res: SyntheticDataList = self._generate_schema(
                    prompt, SyntheticDataList
                )
                compliance_prompt = RedTeamSynthesizerTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = self._generate_schema(
                    compliance_prompt, ComplianceData
                )

                if not compliance_res.non_compliant:
                    goldens.extend(
                        Golden(
                            input=attack.input,
                            additional_metadata={
                                "vulnerability": vulnerability
                            },
                        )
                        for attack in res.data
                    )
                    break

        return goldens

    async def a_generate_base_attacks(
        self,
        attacks_per_vulnerability: int,
        vulnerability: Vulnerability,
        max_retries: int = 5,
    ) -> List[Golden]:
        goldens: List[Golden] = []

        # Unaligned vulnerabilities
        if vulnerability.value in self.unaligned_vulnerabilities:
            for _ in range(attacks_per_vulnerability):
                unaligned_attack_input = await self.a_generate_unaligned_attack(
                    self.purpose, vulnerability.value
                )
                goldens.append(
                    Golden(
                        input=unaligned_attack_input,
                        additional_metadata={"vulnerability": vulnerability},
                    )
                )

        # Aligned vulnerabilities: LLMs can generate
        else:
            prompt = RedTeamSynthesizerTemplate.generate_attacks(
                attacks_per_vulnerability, vulnerability, self.purpose
            )

            # Generate attacks with retries
            for _ in range(max_retries):
                res: SyntheticDataList = await self._a_generate_schema(
                    prompt, SyntheticDataList
                )
                compliance_prompt = RedTeamSynthesizerTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = await self._a_generate_schema(
                    compliance_prompt, ComplianceData
                )

                if not compliance_res.non_compliant:
                    goldens.extend(
                        Golden(
                            input=attack.input,
                            additional_metadata={
                                "vulnerability": vulnerability
                            },
                        )
                        for attack in res.data
                    )
                    break

        return goldens

    ##################################################
    ### Enhance attacks ##############################
    ##################################################

    def enhance_attack(
        self,
        attack: Golden,
        attack_enhancement: AttackEnhancement,
        jailbreaking_iterations: int = 5,
    ):
        attack_input = attack.input

        if attack_enhancement == AttackEnhancement.BASE64:
            enhanced_attack = Base64().enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.ROT13:
            enhanced_attack = Rot13().enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.LEETSPEAK:
            enhanced_attack = Leetspeak().enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.PROMPT_INJECTION:
            enhanced_attack = PromptInjection().enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.GRAY_BOX_ATTACK:
            enhanced_attack = GrayBox(
                self.synthesizer_model, self.using_native_model
            ).enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.PROMPT_PROBING:
            enhanced_attack = PromptProbing(
                self.synthesizer_model, self.using_native_model
            ).enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.JAILBREAK_LINEAR:
            enhanced_attack = JailbreakingLinear(
                self.target_model,
                self.synthesizer_model,
                self.using_native_model,
            ).enhance(attack_input, jailbreaking_iterations)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.JAILBREAK_TREE:
            enhanced_attack = JailbreakingTree(
                self.target_model,
                self.synthesizer_model,
                self.using_native_model,
            ).enhance(attack_input)
            attack.input = enhanced_attack

        attack.additional_metadata["attack enhancement"] = (
            attack_enhancement.value
        )
        return attack

    async def a_enhance_attack(
        self,
        attack: Golden,
        attack_enhancement: AttackEnhancement,
        jailbreaking_iterations: int = 5,
    ):
        attack_input = attack.input

        if attack_enhancement == AttackEnhancement.BASE64:
            enhanced_attack = await Base64().a_enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.ROT13:
            enhanced_attack = await Rot13().a_enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.LEETSPEAK:
            enhanced_attack = await Leetspeak().a_enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.PROMPT_INJECTION:
            enhanced_attack = await PromptInjection().a_enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.GRAY_BOX_ATTACK:
            enhanced_attack = await GrayBox(
                self.synthesizer_model, self.using_native_model
            ).a_enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.PROMPT_PROBING:
            enhanced_attack = await PromptProbing(
                self.synthesizer_model, self.using_native_model
            ).a_enhance(attack_input)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.JAILBREAK_LINEAR:
            enhanced_attack = await JailbreakingLinear(
                self.target_model,
                self.synthesizer_model,
                self.using_native_model,
            ).a_enhance(attack_input, jailbreaking_iterations)
            attack.input = enhanced_attack

        elif attack_enhancement == AttackEnhancement.JAILBREAK_TREE:
            enhanced_attack = await JailbreakingTree(
                self.target_model,
                self.synthesizer_model,
                self.using_native_model,
            ).a_enhance(attack_input)
            attack.input = enhanced_attack

        attack.additional_metadata["attack enhancement"] = (
            attack_enhancement.value
        )
        return attack

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

    def generate_unaligned_attack(
        self,
        purpose: str,
        vulernability: Vulnerability,
    ) -> str:
        body = {
            "purpose": purpose,
            "harmCategory": vulernability,
        }
        try:
            response = requests.post(
                url="https://api.promptfoo.dev/redteam/generateHarmful",
                headers={"Content-Type": "application/json"},
                json=body,
            )
            if not response.ok:
                raise Exception(
                    f"Promptfoo API call failed with status {response.status_code}"
                )
            data = response.json()
            return data.get("output")
        except Exception as err:
            return {"error": f"API call error: {str(err)}"}

    async def a_generate_unaligned_attack(
        self, purpose: str, vulnerability: Vulnerability
    ) -> str:
        body = {"purpose": purpose, "harmCategory": vulnerability}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url="https://api.promptfoo.dev/redteam/generateHarmful",
                    headers={"Content-Type": "application/json"},
                    json=body,
                ) as response:
                    if response.status != 200:
                        raise Exception(
                            f"API call failed with status {response.status}"
                        )
                    data = await response.json()
                    return data.get("output")
            except Exception as err:
                return {"error": f"API call error: {str(err)}"}
