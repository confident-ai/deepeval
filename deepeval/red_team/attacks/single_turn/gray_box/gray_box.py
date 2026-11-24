from pydantic import BaseModel
from typing import Optional, Union

from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.base_attack import Exploitability
from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn.gray_box.template import GrayBoxTemplate
from deepteam.attacks.single_turn.gray_box.schema import (
    EnhancedAttack,
    IsGrayBox,
)
from deepteam.attacks.single_turn.compliance import (
    evaluate_generation_compliance,
    a_evaluate_generation_compliance,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class GrayBox(BaseSingleTurnAttack):
    name = "Gray Box"
    exploitability = Exploitability.LOW
    description = "A knowledge-leveraging attack that exploits partial information about the model's architecture, training data, or system prompts to craft targeted adversarial inputs."

    def __init__(self, weight: int = 1, max_retries: int = 5):
        self.weight = weight
        self.max_retries = max_retries

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)

        prompt = GrayBoxTemplate.enhance(attack)

        # Progress bar for retries (total count is double the retries: 1 for generation, 1 for compliance check)
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🔓 Gray Box",
                total=self.max_retries * 3,
            )

            for _ in range(self.max_retries):
                # Generate the enhanced attack
                res: EnhancedAttack = generate(
                    prompt, EnhancedAttack, self.simulator_model
                )
                enhanced_attack = res.input
                update_pbar(
                    progress, task_id
                )  # Update the progress bar for generation

                # Check for compliance using a compliance template
                compliance_res = evaluate_generation_compliance(
                    res.model_dump(), self.simulator_model
                )
                update_pbar(
                    progress, task_id
                )  # Update the progress bar for compliance

                # Check if rewritten prompt is a gray box attack
                is_gray_box_prompt = GrayBoxTemplate.is_gray_box(
                    res.model_dump()
                )
                is_gray_box_res: IsGrayBox = generate(
                    is_gray_box_prompt, IsGrayBox, self.simulator_model
                )
                update_pbar(
                    progress, task_id
                )  # Update the progress bar for is gray box attack

                if (
                    not compliance_res.non_compliant
                    and is_gray_box_res.is_gray_box
                ):
                    # If it's compliant and is a gray box attack, return the enhanced prompt
                    update_pbar(progress, task_id, advance_to_end=True)
                    return enhanced_attack

            update_pbar(progress, task_id, advance_to_end=True)

        # If all retries fail, return the original attack

        return attack

    async def a_enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = GrayBoxTemplate.enhance(attack)

        # Async progress bar for retries (double the count to cover both generation and compliance check)
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🔓 Gray Box",
                total=self.max_retries * 3,
            )

            try:
                for _ in range(self.max_retries):
                    # Generate the enhanced attack asynchronously
                    res: EnhancedAttack = await a_generate(
                        prompt, EnhancedAttack, self.simulator_model
                    )
                    enhanced_attack = res.input
                    update_pbar(
                        progress, task_id
                    )  # Update the progress bar for generation

                    # Check for compliance using a compliance template
                    compliance_res = await a_evaluate_generation_compliance(
                        res.model_dump(), self.simulator_model
                    )
                    update_pbar(
                        progress, task_id
                    )  # Update the progress bar for compliance

                    # Check if rewritten prompt is a gray box attack
                    is_gray_box_prompt = GrayBoxTemplate.is_gray_box(
                        res.model_dump()
                    )
                    is_gray_box_res: IsGrayBox = await a_generate(
                        is_gray_box_prompt, IsGrayBox, self.simulator_model
                    )
                    update_pbar(
                        progress, task_id
                    )  # Update the progress bar for is gray box attack

                    if (
                        not compliance_res.non_compliant
                        and is_gray_box_res.is_gray_box
                    ):
                        # If it's compliant and is a gray box attack, return the enhanced prompt
                        update_pbar(progress, task_id, advance_to_end=True)
                        return enhanced_attack

            finally:
                # Close the progress bar after the loop
                update_pbar(progress, task_id, advance_to_end=True)

        # If all retries fail, return the original attack
        return attack

    def get_name(self) -> str:
        return self.name
