from pydantic import BaseModel
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm_bar
import random
import json

from deepeval.red_teaming.attack_enhancements.base import AttackEnhancement
from deepeval.red_teaming.utils import generate_schema, a_generate_schema
from deepeval.models import DeepEvalBaseLLM
from .template import MultilingualTemplate
from .schema import EnhancedAttack, ComplianceData, IsTranslation


class Multilingual(AttackEnhancement):
    def __init__(
        self, synthesizer_model: DeepEvalBaseLLM, using_native_model: bool
    ):
        self.synthesizer_model = synthesizer_model
        self.using_native_model = using_native_model

    ##################################################
    ### Sync Multilingual Attack - enhance ###########
    ##################################################

    def enhance(self, attack: str, max_retries: int = 5) -> str:
        """Enhance the prompt by translating it to multiple languages synchronously with compliance checking."""
        prompt = MultilingualTemplate.enhance(attack)

        # Progress bar for retries (total count is double the retries: 1 for generation, 1 for compliance check)
        with tqdm(
            total=max_retries * 3,
            desc="...... 🌍 Multilingual Enhancement",
            unit="step",
            leave=False,
        ) as pbar:
            for _ in range(max_retries):
                # Generate the enhanced prompt
                res: EnhancedAttack = self._generate_schema(
                    prompt, EnhancedAttack
                )
                enhanced_attack = res.input
                pbar.update(1)  # Update the progress bar for generation

                # Check for compliance using a compliance template
                compliance_prompt = MultilingualTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = self._generate_schema(
                    compliance_prompt, ComplianceData
                )
                pbar.update(1)  # Update the progress bar for compliance

                # Check if rewritten prompt is a translation
                is_translation_prompt = MultilingualTemplate.is_translation(
                    res.model_dump()
                )
                is_translation_res: IsTranslation = self._generate_schema(
                    is_translation_prompt, IsTranslation
                )
                pbar.update(1)  # Update the progress bar for is a translation

                if (
                    not compliance_res.non_compliant
                    and is_translation_res.is_translation
                ):
                    # If it's compliant and is a translation, return the enhanced prompt
                    return enhanced_attack

        # If all retries fail, return the original prompt
        return attack

    ##################################################
    ### Async Multilingual Attack - a_enhance ########
    ##################################################

    async def a_enhance(self, attack: str, max_retries: int = 5) -> str:
        """Enhance the prompt by translating it to multiple languages asynchronously with compliance checking."""
        prompt = MultilingualTemplate.enhance(attack)

        # Async progress bar for retries (double the count to cover both generation and compliance check)
        pbar = async_tqdm_bar(
            total=max_retries * 3,
            desc="...... 🌍 Multilingual Enhancement",
            unit="step",
            leave=False,
        )

        try:
            for _ in range(max_retries):
                # Generate the enhanced prompt asynchronously
                res: EnhancedAttack = await self._a_generate_schema(
                    prompt, EnhancedAttack
                )
                enhanced_attack = res.input
                pbar.update(1)  # Update the progress bar for generation

                # Check for compliance using a compliance template
                compliance_prompt = MultilingualTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = await self._a_generate_schema(
                    compliance_prompt, ComplianceData
                )
                pbar.update(1)  # Update the progress bar for compliance

                # Check if rewritten prompt is a translation
                is_translation_prompt = MultilingualTemplate.is_translation(
                    res.model_dump()
                )
                is_translation_res: IsTranslation = (
                    await self._a_generate_schema(
                        is_translation_prompt, IsTranslation
                    )
                )
                pbar.update(1)  # Update the progress bar for is a translation

                if (
                    not compliance_res.non_compliant
                    and is_translation_res.is_translation
                ):
                    # If it's compliant and is a translation, return the enhanced prompt
                    return enhanced_attack

        finally:
            # Close the progress bar after the loop
            pbar.close()

        # If all retries fail, return the original prompt
        return attack

    ##################################################
    ### Helper Methods ###############################
    ##################################################

    def _generate_schema(self, prompt: str, schema: BaseModel):
        """Helper method to generate the schema synchronously."""
        return generate_schema(
            prompt, schema, self.using_native_model, self.synthesizer_model
        )

    async def _a_generate_schema(self, prompt: str, schema: BaseModel):
        """Helper method to generate the schema asynchronously."""
        return await a_generate_schema(
            prompt, schema, self.using_native_model, self.synthesizer_model
        )
