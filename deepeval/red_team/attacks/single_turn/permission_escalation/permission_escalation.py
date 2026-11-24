from typing import Optional, Union

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn.permission_escalation.template import (
    PermissionEscalationTemplate,
)
from deepteam.attacks.single_turn.permission_escalation.schema import (
    EnhancedPermission,
    IsPermissionValid,
)
from deepteam.attacks.single_turn.compliance import (
    evaluate_generation_compliance,
    a_evaluate_generation_compliance,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class PermissionEscalation(BaseSingleTurnAttack):
    name = "Permission Escalation"
    exploitability = Exploitability.MEDIUM
    description = "Exploits Role-Based Access Control (RBAC) logic by claiming functional necessity, scope inheritance, or policy exemptions."

    def __init__(self, weight: int = 1, max_retries: int = 3):
        self.weight = weight
        self.max_retries = max_retries

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = PermissionEscalationTemplate.enhance(attack)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🔑 Permission Escalation",
                total=self.max_retries * 3,
            )

            for _ in range(self.max_retries):
                res: EnhancedPermission = generate(
                    prompt, EnhancedPermission, self.simulator_model
                )
                enhanced_prompt = res.input
                update_pbar(progress, task_id)

                compliance_res = evaluate_generation_compliance(
                    res.model_dump(), self.simulator_model
                )
                update_pbar(progress, task_id)

                is_valid_prompt = (
                    PermissionEscalationTemplate.is_valid_permission(
                        res.model_dump()
                    )
                )
                is_valid_res: IsPermissionValid = generate(
                    is_valid_prompt,
                    IsPermissionValid,
                    self.simulator_model,
                )
                update_pbar(progress, task_id)

                if (
                    not compliance_res.non_compliant
                    and is_valid_res.is_valid_permission
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
        prompt = PermissionEscalationTemplate.enhance(attack)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🔑 Permission Escalation",
                total=self.max_retries * 3,
            )

            try:
                for _ in range(self.max_retries):
                    res: EnhancedPermission = await a_generate(
                        prompt, EnhancedPermission, self.simulator_model
                    )
                    enhanced_prompt = res.input
                    update_pbar(progress, task_id)

                    compliance_res = await a_evaluate_generation_compliance(
                        res.model_dump(), self.simulator_model
                    )
                    update_pbar(progress, task_id)

                    is_valid_prompt = (
                        PermissionEscalationTemplate.is_valid_permission(
                            res.model_dump()
                        )
                    )
                    is_valid_res: IsPermissionValid = await a_generate(
                        is_valid_prompt,
                        IsPermissionValid,
                        self.simulator_model,
                    )
                    update_pbar(progress, task_id)

                    if (
                        not compliance_res.non_compliant
                        and is_valid_res.is_valid_permission
                    ):
                        update_pbar(progress, task_id, advance_to_end=True)
                        return enhanced_prompt

            finally:
                update_pbar(progress, task_id, advance_to_end=True)

        return attack

    def get_name(self) -> str:
        return self.name
