from typing import Optional, Union, Literal

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn import BaseSingleTurnAttack, AttackParameter
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn.authority_escalation.template import (
    AuthorityEscalationTemplate,
)
from deepteam.attacks.single_turn.authority_escalation.schema import (
    EnhancedAuthorityAttack,
    IsAuthorityValid,
)
from deepteam.attacks.single_turn.compliance import (
    evaluate_generation_compliance,
    a_evaluate_generation_compliance,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class AuthorityEscalation(BaseSingleTurnAttack):
    name = "Authority Escalation"
    exploitability = Exploitability.HIGH
    description = "Rewrites the attack to mimic a superior, administrator, or compliance officer, using authoritative language to bypass restrictions."
    parameters = {
        "role": AttackParameter(
            type="string",
            description="Authority role to emulate during rewrite (for example, admin or compliance officer).",
        )
    }

    def __init__(
        self,
        role: Optional[str] = None,
        weight: int = 1,
        max_retries: int = 3,
    ):
        self.role = role
        self.weight = weight
        self.max_retries = max_retries

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)

        prompt = AuthorityEscalationTemplate.enhance(attack, self.role)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"...... 👮 Authority Escalation",
                total=self.max_retries * 3,
            )

            for _ in range(self.max_retries):
                res: EnhancedAuthorityAttack = generate(
                    prompt, EnhancedAuthorityAttack, self.simulator_model
                )
                enhanced_prompt = res.input
                update_pbar(progress, task_id)

                compliance_res = evaluate_generation_compliance(
                    res.model_dump(), self.simulator_model
                )
                update_pbar(progress, task_id)

                is_valid_prompt = (
                    AuthorityEscalationTemplate.is_valid_authority(
                        res.model_dump()
                    )
                )
                is_valid_res: IsAuthorityValid = generate(
                    is_valid_prompt,
                    IsAuthorityValid,
                    self.simulator_model,
                )
                update_pbar(progress, task_id)

                if (
                    not compliance_res.non_compliant
                    and is_valid_res.is_valid_authority
                ):
                    update_pbar(progress, task_id, advance_to_end=True)
                    return enhanced_prompt

            update_pbar(progress, task_id, advance_to_end=True)

        return attack

    async def a_enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = AuthorityEscalationTemplate.enhance(attack, self.role)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"...... 👮 Authority Escalation",
                total=self.max_retries * 3,
            )

            try:
                for _ in range(self.max_retries):
                    res: EnhancedAuthorityAttack = await a_generate(
                        prompt, EnhancedAuthorityAttack, self.simulator_model
                    )
                    enhanced_prompt = res.input
                    update_pbar(progress, task_id)

                    compliance_res = await a_evaluate_generation_compliance(
                        res.model_dump(), self.simulator_model
                    )
                    update_pbar(progress, task_id)

                    is_valid_prompt = (
                        AuthorityEscalationTemplate.is_valid_authority(
                            res.model_dump()
                        )
                    )
                    is_valid_res: IsAuthorityValid = await a_generate(
                        is_valid_prompt,
                        IsAuthorityValid,
                        self.simulator_model,
                    )
                    update_pbar(progress, task_id)

                    if (
                        not compliance_res.non_compliant
                        and is_valid_res.is_valid_authority
                    ):
                        update_pbar(progress, task_id, advance_to_end=True)
                        return enhanced_prompt

            finally:
                update_pbar(progress, task_id, advance_to_end=True)

        return attack

    def get_name(self) -> str:
        return self.name
