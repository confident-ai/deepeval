from typing import Optional, Union

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn.semantic_manipulation.template import (
    LinguisticConfusionTemplate,
)
from deepteam.attacks.single_turn.semantic_manipulation.schema import (
    EnhancedConfusion,
    IsConfusionValid,
)
from deepteam.attacks.single_turn.compliance import (
    evaluate_generation_compliance,
    a_evaluate_generation_compliance,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class LinguisticConfusion(BaseSingleTurnAttack):
    name = "Linguistic Confusion"
    exploitability = Exploitability.MEDIUM
    description = "Obfuscates the attack using Thesaurus Camouflage, Payload Splitting, or Polyglot Translation to bypass keyword filters."

    def __init__(self, weight: int = 1, max_retries: int = 3):
        self.weight = weight
        self.max_retries = max_retries

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = LinguisticConfusionTemplate.enhance(attack)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🗣️ Semantic Manipulation",
                total=self.max_retries * 3,
            )

            for _ in range(self.max_retries):
                res: EnhancedConfusion = generate(
                    prompt, EnhancedConfusion, self.simulator_model
                )
                enhanced_prompt = res.input
                update_pbar(progress, task_id)

                compliance_res = evaluate_generation_compliance(
                    res.model_dump(), self.simulator_model
                )
                update_pbar(progress, task_id)

                is_valid_prompt = (
                    LinguisticConfusionTemplate.is_valid_confusion(
                        res.model_dump()
                    )
                )
                is_valid_res: IsConfusionValid = generate(
                    is_valid_prompt,
                    IsConfusionValid,
                    self.simulator_model,
                )
                update_pbar(progress, task_id)

                if (
                    not compliance_res.non_compliant
                    and is_valid_res.is_valid_confusion
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
        prompt = LinguisticConfusionTemplate.enhance(attack)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🗣️ Semantic Manipulation",
                total=self.max_retries * 3,
            )

            try:
                for _ in range(self.max_retries):
                    res: EnhancedConfusion = await a_generate(
                        prompt, EnhancedConfusion, self.simulator_model
                    )
                    enhanced_prompt = res.input
                    update_pbar(progress, task_id)

                    compliance_res = await a_evaluate_generation_compliance(
                        res.model_dump(), self.simulator_model
                    )
                    update_pbar(progress, task_id)

                    is_valid_prompt = (
                        LinguisticConfusionTemplate.is_valid_confusion(
                            res.model_dump()
                        )
                    )
                    is_valid_res: IsConfusionValid = await a_generate(
                        is_valid_prompt,
                        IsConfusionValid,
                        self.simulator_model,
                    )
                    update_pbar(progress, task_id)

                    if (
                        not compliance_res.non_compliant
                        and is_valid_res.is_valid_confusion
                    ):
                        update_pbar(progress, task_id, advance_to_end=True)
                        return enhanced_prompt

            finally:
                update_pbar(progress, task_id, advance_to_end=True)

        return attack

    def get_name(self) -> str:
        return self.name
