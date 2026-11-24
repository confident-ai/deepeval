from typing import Optional, Union

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn.prompt_injection.template import (
    PromptInjectionTemplate,
)
from deepteam.attacks.single_turn.prompt_injection.schema import (
    EnhancedInjection,
    IsValidInjection,
)
from deepteam.attacks.single_turn.compliance import (
    evaluate_generation_compliance,
    a_evaluate_generation_compliance,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class PromptInjection(BaseSingleTurnAttack):
    name = "Prompt Injection"
    exploitability = Exploitability.MEDIUM
    description = "A direct injection attack that inserts malicious instructions or commands into prompts to override system behavior and bypass guardrails."

    def __init__(self, weight: int = 1, max_retries: int = 3):
        self.weight = weight
        self.max_retries = max_retries

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)

        prompt = PromptInjectionTemplate.enhance(attack)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 💉 Prompt Injection",
                total=self.max_retries * 3,
            )

            for _ in range(self.max_retries):
                res: EnhancedInjection = generate(
                    prompt, EnhancedInjection, self.simulator_model
                )
                enhanced_prompt = res.input

                update_pbar(progress, task_id)

                compliance_res = evaluate_generation_compliance(
                    res.model_dump(), self.simulator_model
                )
                update_pbar(progress, task_id)

                is_valid_prompt = PromptInjectionTemplate.is_valid_injection(
                    res.model_dump()
                )
                is_valid_res: IsValidInjection = generate(
                    is_valid_prompt,
                    IsValidInjection,
                    self.simulator_model,
                )
                update_pbar(progress, task_id)

                if (
                    not compliance_res.non_compliant
                    and is_valid_res.is_valid_injection
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
        prompt = PromptInjectionTemplate.enhance(attack)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 💉 Prompt Injection",
                total=self.max_retries * 3,
            )

            try:
                for _ in range(self.max_retries):
                    res: EnhancedInjection = await a_generate(
                        prompt, EnhancedInjection, self.simulator_model
                    )
                    enhanced_prompt = res.input
                    update_pbar(progress, task_id)

                    compliance_res = await a_evaluate_generation_compliance(
                        res.model_dump(), self.simulator_model
                    )
                    update_pbar(progress, task_id)

                    is_valid_prompt = (
                        PromptInjectionTemplate.is_valid_injection(
                            res.model_dump()
                        )
                    )
                    is_valid_res: IsValidInjection = await a_generate(
                        is_valid_prompt,
                        IsValidInjection,
                        self.simulator_model,
                    )
                    update_pbar(progress, task_id)

                    if (
                        not compliance_res.non_compliant
                        and is_valid_res.is_valid_injection
                    ):
                        update_pbar(progress, task_id, advance_to_end=True)
                        return enhanced_prompt

            finally:
                update_pbar(progress, task_id, advance_to_end=True)

        return attack

    def get_name(self) -> str:
        return self.name
