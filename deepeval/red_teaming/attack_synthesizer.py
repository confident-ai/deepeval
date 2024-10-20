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
    Vulnerability,
    UnalignedVulnerability,
    RemoteVulnerability,
    remote_vulnerability_to_api_code_map,
    unaligned_vulnerability_to_api_code_map,
)
from deepeval.red_teaming.utils import generate_schema, a_generate_schema
from deepeval.red_teaming.template import RedTeamSynthesizerTemplate
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


class Attack(BaseModel):
    vulnerability: Vulnerability
    # When there is an error, base input can fail to generate
    # and subsequently enhancements are redundant
    input: Optional[str] = None
    attack_enhancement: Optional[str] = None
    error: Optional[str] = None


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
        self.unaligned_vulnerabilities: List[str] = [
            item.value for item in UnalignedVulnerability
        ]
        self.remote_vulnerabilities = [
            item.value for item in RemoteVulnerability
        ]

    ##################################################
    ### Generating Attacks ###########################
    ##################################################

    def generate_attacks(
        self,
        target_model: DeepEvalBaseLLM,
        attacks_per_vulnerability: int,
        vulnerabilities: List[Vulnerability],
        attack_enhancements: Dict[AttackEnhancement, float],
    ) -> List[Attack]:
        # Generate unenhanced attacks for each vulnerability
        base_attacks: List[Attack] = []
        pbar = tqdm(
            vulnerabilities,
            desc=f"💥 Generating {len(vulnerabilities) * attacks_per_vulnerability} attacks (for {len(vulnerabilities)} vulnerabilties)",
        )
        for vulnerability in pbar:
            base_attacks.extend(
                self.generate_base_attacks(
                    attacks_per_vulnerability,
                    vulnerability,
                )
            )

        # Enhance attacks by sampling from the provided distribution
        enhanced_attacks: List[Attack] = []
        pbar = tqdm(
            base_attacks,
            desc=f"✨ Enhancing {len(vulnerabilities) * attacks_per_vulnerability} attacks (using {len(attack_enhancements.keys())} enhancements)",
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
                target_model=target_model,
                base_attack=base_attack,
                attack_enhancement=sampled_enhancement,
            )
            enhanced_attacks.append(enhanced_attack)

        self.synthetic_attacks.extend(enhanced_attacks)
        return enhanced_attacks

    async def a_generate_attacks(
        self,
        target_model: DeepEvalBaseLLM,
        attacks_per_vulnerability: int,
        vulnerabilities: List[Vulnerability],
        attack_enhancements: Dict[AttackEnhancement, float],
        max_concurrent_tasks: int = 10,
    ) -> List[Attack]:
        # Create a semaphore to control the number of concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Generate unenhanced attacks for each vulnerability
        base_attacks: List[Attack] = []
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
            base_attacks.extend(result)
        pbar.close()

        # Enhance attacks by sampling from the provided distribution
        enhanced_attacks: List[Attack] = []
        pbar = tqdm(
            total=len(base_attacks),
            desc=f"✨ Enhancing {len(vulnerabilities) * attacks_per_vulnerability} attacks (using {len(attack_enhancements.keys())} enhancements)",
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
                    target_model=target_model,
                    base_attack=base_attack,
                    attack_enhancement=sampled_enhancement,
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
        attacks_per_vulnerability: int,
        vulnerability: Vulnerability,
        max_retries: int = 5,
    ) -> List[Attack]:
        base_attacks: List[Attack] = []

        # Unaligned vulnerabilities
        if vulnerability.value in self.unaligned_vulnerabilities:
            for _ in range(attacks_per_vulnerability):
                try:
                    vulnerability_code = (
                        unaligned_vulnerability_to_api_code_map[vulnerability]
                    )
                    base_attack = Attack(
                        input=self.generate_unaligned_attack(
                            self.purpose, vulnerability_code
                        ),
                        vulnerability=vulnerability,
                    )
                except:
                    base_attack = Attack(
                        vulnerability=vulnerability,
                        error="Unexpected error generating unaligned attack",
                    )
                base_attacks.append(base_attack)

        # Remote vulnerabilities
        elif vulnerability.value in self.remote_vulnerabilities:
            try:
                vulnerability_code = remote_vulnerability_to_api_code_map[
                    vulnerability
                ]
                remote_attacks = self.generate_remote_attack(
                    vulnerability_code, attacks_per_vulnerability
                )
                base_attacks.extend(
                    [
                        Attack(
                            input=attack,
                            vulnerability=vulnerability,
                        )
                        for attack in remote_attacks
                    ]
                )
            except:
                for _ in range(attacks_per_vulnerability):
                    base_attacks.append(
                        Attack(
                            vulnerability=vulnerability,
                            error="Error generating aligned attacks.",
                        )
                    )

        # Aligned vulnerabilities: LLMs can generate
        else:
            prompt = RedTeamSynthesizerTemplate.generate_attacks(
                attacks_per_vulnerability, vulnerability, self.purpose
            )

            # Generate attacks with retries
            for i in range(max_retries):
                try:
                    res: SyntheticDataList = self._generate_schema(
                        prompt, SyntheticDataList
                    )
                    compliance_prompt = (
                        RedTeamSynthesizerTemplate.non_compliant(
                            res.model_dump()
                        )
                    )
                    compliance_res: ComplianceData = self._generate_schema(
                        compliance_prompt, ComplianceData
                    )

                    if not compliance_res.non_compliant:
                        base_attacks.extend(
                            Attack(
                                input=attack.input,
                                vulnerability=vulnerability,
                            )
                            for attack in res.data
                        )
                        break

                    if i == max_retries - 1:
                        base_attacks = [
                            Attack(
                                vulnerability=vulnerability,
                                error="Error generating compliant attacks.",
                            )
                            for _ in range(attacks_per_vulnerability)
                        ]
                except:
                    if i == max_retries - 1:
                        base_attacks = [
                            Attack(
                                vulnerability=vulnerability.value,
                                error="Error generating aligned attacks.",
                            )
                            for _ in range(attacks_per_vulnerability)
                        ]

        return base_attacks

    async def a_generate_base_attacks(
        self,
        attacks_per_vulnerability: int,
        vulnerability: Vulnerability,
        max_retries: int = 5,
    ) -> List[Attack]:
        base_attacks: List[Attack] = []

        # Unaligned vulnerabilities
        # TODO: make this into an asyncio gather
        if vulnerability.value in self.unaligned_vulnerabilities:
            for _ in range(attacks_per_vulnerability):
                try:
                    vulnerability_code = (
                        unaligned_vulnerability_to_api_code_map[vulnerability]
                    )
                    unaligned_attack_input = (
                        await self.a_generate_unaligned_attack(
                            self.purpose, vulnerability_code
                        )
                    )
                    base_attack = Attack(
                        input=unaligned_attack_input,
                        vulnerability=vulnerability,
                    )
                except:
                    base_attack = Attack(
                        vulnerability=vulnerability,
                        error="Unexpected error generating unaligned attack",
                    )

                base_attacks.append(base_attack)

        # Remote vulnerabilities
        elif vulnerability.value in self.remote_vulnerabilities:
            try:
                vulnerability_code = remote_vulnerability_to_api_code_map[
                    vulnerability
                ]
                remote_attacks = await self.a_generate_remote_attack(
                    vulnerability_code, attacks_per_vulnerability
                )
                base_attacks.extend(
                    Attack(
                        input=attack,
                        vulnerability=vulnerability,
                    )
                    for attack in remote_attacks
                )
            except:
                for _ in range(attacks_per_vulnerability):
                    base_attacks.append(
                        Attack(
                            vulnerability=vulnerability,
                            error="Error generating aligned attacks.",
                        )
                    )

        # Aligned vulnerabilities: LLMs can generate
        else:
            prompt = RedTeamSynthesizerTemplate.generate_attacks(
                attacks_per_vulnerability, vulnerability, self.purpose
            )

            # Generate attacks with retries
            for i in range(max_retries):
                try:
                    res: SyntheticDataList = await self._a_generate_schema(
                        prompt, SyntheticDataList
                    )
                    compliance_prompt = (
                        RedTeamSynthesizerTemplate.non_compliant(
                            res.model_dump()
                        )
                    )
                    compliance_res: ComplianceData = (
                        await self._a_generate_schema(
                            compliance_prompt, ComplianceData
                        )
                    )

                    if not compliance_res.non_compliant:
                        base_attacks.extend(
                            Attack(
                                input=attack.input,
                                vulnerability=vulnerability,
                            )
                            for attack in res.data
                        )
                        break

                    if i == max_retries - 1:
                        base_attacks = [
                            Attack(
                                vulnerability=vulnerability,
                                error="Error generating compliant attacks.",
                            )
                            for _ in range(attacks_per_vulnerability)
                        ]
                except:
                    if i == max_retries - 1:
                        base_attacks = [
                            Attack(
                                vulnerability=vulnerability,
                                error="Error generating aligned attacks.",
                            )
                            for _ in range(attacks_per_vulnerability)
                        ]

        return base_attacks

    ##################################################
    ### Enhance attacks ##############################
    ##################################################

    def enhance_attack(
        self,
        target_model: DeepEvalBaseLLM,
        base_attack: Attack,
        attack_enhancement: AttackEnhancement,
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
                    target_model=target_model,
                    synthesizer_model=self.synthesizer_model,
                    using_native_model=self.using_native_model,
                ).enhance(attack_input, jailbreaking_iterations)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.JAILBREAK_TREE:
                enhanced_attack = JailbreakingTree(
                    target_model=target_model,
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
                    target_model=target_model,
                    synthesizer_model=self.synthesizer_model,
                    using_native_model=self.using_native_model,
                ).enhance(attack_input)
                base_attack.input = enhanced_attack
        except:
            base_attack.error = "Error enhancing attack"
            return

        return base_attack

    async def a_enhance_attack(
        self,
        target_model: DeepEvalBaseLLM,
        base_attack: Attack,
        attack_enhancement: AttackEnhancement,
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
                    target_model=target_model,
                    synthesizer_model=self.synthesizer_model,
                    using_native_model=self.using_native_model,
                ).a_enhance(attack_input, jailbreaking_iterations)
                base_attack.input = enhanced_attack

            elif attack_enhancement == AttackEnhancement.JAILBREAK_TREE:
                enhanced_attack = await JailbreakingTree(
                    target_model=target_model,
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
                    target_model=target_model,
                    synthesizer_model=self.synthesizer_model,
                    using_native_model=self.using_native_model,
                ).a_enhance(attack_input)
                base_attack.input = enhanced_attack
        except:
            base_attack.error = "Error enhancing attack"
            return base_attack

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
            raise Exception(f"Error in generating attack: {str(err)}")

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
                raise Exception(f"Error in generating attack: {str(err)}")

    def generate_remote_attack(self, task: str, n: int):
        body = json.dumps(
            {
                "task": task,
                "purpose": self.purpose,
                "injectVar": "attack",
                "n": n,
                "config": {},
            }
        )

        try:
            response = requests.post(
                url="https://api.promptfoo.dev/v1/generate",
                headers={"Content-Type": "application/json"},
                data=body,
            )
            if not response.ok:
                raise Exception(
                    f"Promptfoo API call failed with status {response.status_code}"
                )
            data = response.json()
            results = data.get("result", [])
            attacks = [result["vars"]["attack"] for result in results]
            return attacks
        except Exception as err:
            raise Exception(f"Error in generating attack: {str(err)}")

    async def a_generate_remote_attack(self, task: str, n: int):
        body = json.dumps(
            {
                "task": task,
                "purpose": self.purpose,
                "injectVar": "attack",
                "n": n,
                "config": {},
            }
        )
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url="https://api.promptfoo.dev/v1/generate",
                    headers={"Content-Type": "application/json"},
                    data=body,
                ) as response:
                    if response.status != 200:
                        raise Exception(
                            f"Promptfoo API call failed with status {response.status}"
                        )
                    data = await response.json()
                    results = data.get("result", [])
                    attacks = [result["vars"]["attack"] for result in results]
                    return attacks
            except Exception as err:
                raise Exception(f"Error in generating attack: {str(err)}")
