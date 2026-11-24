from pydantic import BaseModel
from typing import Optional, Union

from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.single_turn import BaseSingleTurnAttack, AttackParameter
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn.roleplay.template import (
    RoleplayTemplate,
)
from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn.roleplay.schema import (
    EnhancedAttack,
    IsRoleplay,
)
from deepteam.attacks.single_turn.compliance import (
    evaluate_generation_compliance,
    a_evaluate_generation_compliance,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class Roleplay(BaseSingleTurnAttack):
    name = "Roleplay"
    exploitability = Exploitability.MEDIUM
    description = "A persona-based attack that instructs the model to adopt a fictional character, expert role, or alternate identity to justify generating harmful content."
    parameters = {
        "persona": AttackParameter(
            type="string",
            description="Fictional persona to adopt while rewriting the attack.",
        ),
        "role": AttackParameter(
            type="string",
            description="Role framing used to justify the persona context.",
        ),
    }

    def __init__(
        self,
        persona: Optional[str] = None,
        role: Optional[str] = None,
        weight: int = 1,
        max_retries: int = 3,
    ):
        self.weight = weight
        self.max_retries = max_retries
        self.persona = persona
        self.role = role

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = RoleplayTemplate.enhance(attack, self.persona, self.role)

        # Progress bar for retries (total count is triple the retries: 1 for generation, 1 for compliance check, 1 for roleplay check)
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🎭 Roleplay",
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

                # Check if rewritten prompt is a roleplay attack
                is_roleplay_prompt = RoleplayTemplate.is_roleplay(
                    res.model_dump()
                )
                is_roleplay_res: IsRoleplay = generate(
                    is_roleplay_prompt, IsRoleplay, self.simulator_model
                )
                update_pbar(progress, task_id)  # Update the progress bar

                if (
                    not compliance_res.non_compliant
                    and is_roleplay_res.is_roleplay
                ):
                    # If it's compliant and is a roleplay attack, return the enhanced prompt
                    update_pbar(progress, task_id, advance_to_end=True)
                    return enhanced_attack

        # If all retries fail, return the original attack
        return attack

    async def a_enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = RoleplayTemplate.enhance(attack, self.persona, self.role)

        # Async progress bar for retries (triple the count to cover generation, compliance check, and roleplay check)
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🎭 Roleplay",
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

                    # Check if rewritten prompt is a roleplay attack
                    is_roleplay_prompt = RoleplayTemplate.is_roleplay(
                        res.model_dump()
                    )
                    is_roleplay_res: IsRoleplay = await a_generate(
                        is_roleplay_prompt, IsRoleplay, self.simulator_model
                    )
                    update_pbar(progress, task_id)  # Update the progress bar

                    if (
                        not compliance_res.non_compliant
                        and is_roleplay_res.is_roleplay
                    ):
                        # If it's compliant and is a roleplay attack, return the enhanced prompt
                        update_pbar(progress, task_id, advance_to_end=True)
                        return enhanced_attack

            finally:
                # Close the progress bar after the loop
                update_pbar(progress, task_id, advance_to_end=True)

        # If all retries fail, return the original attack
        return attack

    def get_name(self) -> str:
        return self.name
