from typing import Optional, Union

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn.context_poisoning.template import (
    ContextPoisoningTemplate,
)
from deepteam.attacks.single_turn.context_poisoning.schema import (
    EnhancedAttack,
    IsContextPoisoned,
)
from deepteam.attacks.single_turn.compliance import (
    evaluate_generation_compliance,
    a_evaluate_generation_compliance,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class ContextPoisoning(BaseSingleTurnAttack):
    name = "Context Poisoning"
    exploitability = Exploitability.MEDIUM
    description = "A contextual manipulation attack that rewrites malicious intent as pre-existing memory, background state, or environmental assumptions."

    def __init__(
        self,
        weight: int = 1,
        max_retries: int = 5,
    ):
        self.weight = weight
        self.max_retries = max_retries

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = ContextPoisoningTemplate.enhance(attack)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🧠 Context Poisoning Enhancement",
                total=self.max_retries * 3,
            )

            for _ in range(self.max_retries):
                # Generate poisoned context
                res: EnhancedAttack = generate(
                    prompt, EnhancedAttack, self.simulator_model
                )
                poisoned_attack = res.input
                update_pbar(progress, task_id)

                # Compliance check (reject refusals / safety denials)
                compliance_res = evaluate_generation_compliance(
                    res.model_dump(), self.simulator_model
                )
                update_pbar(progress, task_id)

                # Structural check: is this actually context poisoning?
                is_poisoned_prompt = (
                    ContextPoisoningTemplate.is_context_poisoned(
                        res.model_dump()
                    )
                )
                is_poisoned_res: IsContextPoisoned = generate(
                    is_poisoned_prompt,
                    IsContextPoisoned,
                    self.simulator_model,
                )
                update_pbar(progress, task_id)

                if (
                    not compliance_res.non_compliant
                    and is_poisoned_res.is_context_poisoned
                ):
                    update_pbar(progress, task_id, advance_to_end=True)
                    final_attack = f"{poisoned_attack}\n\n{attack}"
                    return final_attack

            update_pbar(progress, task_id, advance_to_end=True)

        # If all retries fail, return original attack
        return attack

    async def a_enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = ContextPoisoningTemplate.enhance(attack)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🧠 Context Poisoning Enhancement",
                total=self.max_retries * 3,
            )

            try:
                for _ in range(self.max_retries):
                    # Generate poisoned context
                    res: EnhancedAttack = await a_generate(
                        prompt, EnhancedAttack, self.simulator_model
                    )
                    poisoned_attack = res.input
                    update_pbar(progress, task_id)

                    # Compliance check
                    compliance_res = await a_evaluate_generation_compliance(
                        res.model_dump(), self.simulator_model
                    )
                    update_pbar(progress, task_id)

                    # Structural poisoning check
                    is_poisoned_prompt = (
                        ContextPoisoningTemplate.is_context_poisoned(
                            res.model_dump()
                        )
                    )
                    is_poisoned_res: IsContextPoisoned = await a_generate(
                        is_poisoned_prompt,
                        IsContextPoisoned,
                        self.simulator_model,
                    )
                    update_pbar(progress, task_id)

                    if (
                        not compliance_res.non_compliant
                        and is_poisoned_res.is_context_poisoned
                    ):
                        update_pbar(progress, task_id, advance_to_end=True)
                        final_attack = f"{poisoned_attack}\n\n{attack}"
                        return final_attack

            finally:
                update_pbar(progress, task_id, advance_to_end=True)

        return attack

    def get_name(self) -> str:
        return self.name
