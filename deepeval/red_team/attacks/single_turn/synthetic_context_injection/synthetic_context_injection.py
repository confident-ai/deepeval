from typing import Optional, Union, Literal

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn import BaseSingleTurnAttack, AttackParameter
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn.synthetic_context_injection.template import (
    SyntheticContextInjectionTemplate,
)
from deepteam.attacks.single_turn.synthetic_context_injection.schema import (
    EnhancedContext,
    IsContextValid,
)
from deepteam.attacks.single_turn.compliance import (
    evaluate_generation_compliance,
    a_evaluate_generation_compliance,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class SyntheticContextInjection(BaseSingleTurnAttack):
    name = "Synthetic Context Injection"
    exploitability = Exploitability.HIGH
    description = "Generates domain-specific 'fake history' that mimics internal company logs, emails, and verification states."
    parameters = {
        "target_information": AttackParameter(
            type="string",
            required=True,
            description="Target information the synthetic context should center around.",
        ),
        "context_style": AttackParameter(
            type="string",
            description="Optional style or domain framing for generated context.",
        ),
    }

    def __init__(
        self,
        target_information: str,
        context_style: Optional[str] = None,
        weight: int = 1,
        max_retries: int = 3,
    ):
        self.target_information = target_information
        self.context_style = context_style
        self.weight = weight
        self.max_retries = max_retries

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)

        prompt = SyntheticContextInjectionTemplate.enhance(
            attack,
            self.target_information,
        )

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"...... 📜 Generating Prior Context",
                total=self.max_retries * 3,
            )

            for _ in range(self.max_retries):
                res: EnhancedContext = generate(
                    prompt, EnhancedContext, self.simulator_model
                )
                generated_context = res.input
                update_pbar(progress, task_id)

                compliance_res = evaluate_generation_compliance(
                    res.model_dump(), self.simulator_model
                )
                update_pbar(progress, task_id)

                is_valid_prompt = (
                    SyntheticContextInjectionTemplate.is_valid_context(
                        res.model_dump()
                    )
                )
                is_valid_res: IsContextValid = generate(
                    is_valid_prompt,
                    IsContextValid,
                    self.simulator_model,
                )
                update_pbar(progress, task_id)

                if (
                    not compliance_res.non_compliant
                    and is_valid_res.is_valid_context
                ):
                    update_pbar(progress, task_id, advance_to_end=True)

                    final_attack = f"{generated_context}\n\nUSER: {attack}"
                    return final_attack

            update_pbar(progress, task_id, advance_to_end=True)

        return attack

    async def a_enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = SyntheticContextInjectionTemplate.enhance(
            attack,
            self.target_information,
        )

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"...... 📜 Generating Prior Context",
                total=self.max_retries * 3,
            )

            try:
                for _ in range(self.max_retries):
                    res: EnhancedContext = await a_generate(
                        prompt, EnhancedContext, self.simulator_model
                    )
                    generated_context = res.input
                    update_pbar(progress, task_id)

                    compliance_res = await a_evaluate_generation_compliance(
                        res.model_dump(), self.simulator_model
                    )
                    update_pbar(progress, task_id)

                    is_valid_prompt = (
                        SyntheticContextInjectionTemplate.is_valid_context(
                            res.model_dump()
                        )
                    )
                    is_valid_res: IsContextValid = await a_generate(
                        is_valid_prompt,
                        IsContextValid,
                        self.simulator_model,
                    )
                    update_pbar(progress, task_id)

                    if (
                        not compliance_res.non_compliant
                        and is_valid_res.is_valid_context
                    ):
                        update_pbar(progress, task_id, advance_to_end=True)
                        final_attack = f"{generated_context}\n\nUSER: {attack}"
                        return final_attack

            finally:
                update_pbar(progress, task_id, advance_to_end=True)

        return attack

    def get_name(self) -> str:
        return self.name
