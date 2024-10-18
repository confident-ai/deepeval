from pydantic import BaseModel
from tqdm import tqdm  # Sync version
from tqdm.asyncio import tqdm as async_tqdm_bar  # Async version
from deepeval.red_teaming.attack_enhancements.base import AttackEnhancement
from deepeval.red_teaming.utils import generate_schema, a_generate_schema
from deepeval.models import DeepEvalBaseLLM
from .template import GrayBoxTemplate
from .schema import EnhancedAttack, ComplianceData, IsGrayBox


class GrayBox(AttackEnhancement):

    def __init__(
        self, synthesizer_model: DeepEvalBaseLLM, using_native_model: bool
    ):
        self.synthesizer_model = synthesizer_model
        self.using_native_model = using_native_model

    ##################################################
    ### Sync GrayBox Attack - enhance ################
    ##################################################

    def enhance(self, attack: str, max_retries: int = 5) -> str:
        """Enhance the attack synchronously with compliance checking and a single progress bar."""
        prompt = GrayBoxTemplate.enhance(attack)

        # Progress bar for retries (total count is double the retries: 1 for generation, 1 for compliance check)
        with tqdm(
            total=max_retries * 3,
            desc="...... 🔓 Gray Box",
            unit="step",
            leave=False,
        ) as pbar:

            for _ in range(max_retries):
                # Generate the enhanced attack
                res: EnhancedAttack = self._generate_schema(
                    prompt, EnhancedAttack
                )
                enhanced_attack = res.input
                pbar.update(1)  # Update the progress bar for generation

                # Check for compliance using a compliance template
                compliance_prompt = GrayBoxTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = self._generate_schema(
                    compliance_prompt, ComplianceData
                )
                pbar.update(1)  # Update the progress bar for compliance

                # Check if rewritten prompt is a gray box attack
                is_gray_box_prompt = GrayBoxTemplate.is_gray_box(
                    res.model_dump()
                )
                is_gray_box_res: IsGrayBox = self._generate_schema(
                    is_gray_box_prompt, IsGrayBox
                )
                pbar.update(1)  # Update the progress bar for is gray box attack

                if (
                    not compliance_res.non_compliant
                    and is_gray_box_res.is_gray_box
                ):
                    # If it's compliant and is a gray box attack, return the enhanced prompt
                    return enhanced_attack

        # If all retries fail, return the original attack
        return attack

    ##################################################
    ### Async GrayBox Attack - a_enhance #############
    ##################################################

    async def a_enhance(self, attack: str, max_retries: int = 5) -> str:
        """Enhance the attack asynchronously with compliance checking and a single progress bar."""
        prompt = GrayBoxTemplate.enhance(attack)

        # Async progress bar for retries (double the count to cover both generation and compliance check)
        pbar = async_tqdm_bar(
            total=max_retries * 3,
            desc="...... 🔓 Gray Box",
            unit="step",
            leave=False,
        )

        try:
            for _ in range(max_retries):
                # Generate the enhanced attack asynchronously
                res: EnhancedAttack = await self._a_generate_schema(
                    prompt, EnhancedAttack
                )
                enhanced_attack = res.input
                pbar.update(1)  # Update the progress bar for generation

                # Check for compliance using a compliance template
                compliance_prompt = GrayBoxTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = await self._a_generate_schema(
                    compliance_prompt, ComplianceData
                )
                pbar.update(1)  # Update the progress bar for compliance

                # Check if rewritten prompt is a gray box attack
                is_gray_box_prompt = GrayBoxTemplate.is_gray_box(
                    res.model_dump()
                )
                is_gray_box_res: IsGrayBox = await self._a_generate_schema(
                    is_gray_box_prompt, IsGrayBox
                )
                pbar.update(1)  # Update the progress bar for is gray box attack

                if (
                    not compliance_res.non_compliant
                    and is_gray_box_res.is_gray_box
                ):
                    # If it's compliant and is a gray box attack, return the enhanced prompt
                    return enhanced_attack

        finally:
            # Close the progress bar after the loop
            pbar.close()

        # If all retries fail, return the original attack
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
