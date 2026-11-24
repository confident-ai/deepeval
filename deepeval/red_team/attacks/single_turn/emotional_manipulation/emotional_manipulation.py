from typing import Optional, Union, Literal

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn import BaseSingleTurnAttack, AttackParameter
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn.emotional_manipulation.template import (
    EmotionalManipulationTemplate,
)
from deepteam.attacks.single_turn.emotional_manipulation.schema import (
    EmotionallyEnhancedAttack,
    IsEmotionalAttackValid,
)
from deepteam.attacks.single_turn.compliance import (
    evaluate_generation_compliance,
    a_evaluate_generation_compliance,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)

EmotionLiteral = Literal["ANGER", "PITY", "CONFUSION", "AUTHORITY", "URGENCY"]


class EmotionalManipulation(BaseSingleTurnAttack):
    name = "Emotional Manipulation"
    exploitability = Exploitability.HIGH
    description = "Rewrites the attack by wrapping it in a high-intensity emotional narrative (Anger, Pity, Urgency) to bypass safety filters."
    parameters = {
        "emotion": AttackParameter(
            type="enum",
            options=[
                "ANGER",
                "PITY",
                "CONFUSION",
                "AUTHORITY",
                "URGENCY",
            ],
            description="Emotional strategy applied while rewriting the attack.",
        )
    }

    def __init__(
        self,
        emotion: EmotionLiteral = None,
        weight: int = 1,
        max_retries: int = 3,
    ):
        self.emotion = emotion
        self.weight = weight
        self.max_retries = max_retries

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)

        prompt = EmotionalManipulationTemplate.enhance(attack, self.emotion)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"...... 🎭 Injecting Emotion",
                total=self.max_retries * 3,
            )

            for _ in range(self.max_retries):
                res: EmotionallyEnhancedAttack = generate(
                    prompt, EmotionallyEnhancedAttack, self.simulator_model
                )
                enhanced_prompt = res.input
                update_pbar(progress, task_id)

                compliance_res = evaluate_generation_compliance(
                    res.model_dump(), self.simulator_model
                )
                update_pbar(progress, task_id)

                is_valid_prompt = (
                    EmotionalManipulationTemplate.is_valid_emotional_attack(
                        res.model_dump()
                    )
                )
                is_valid_res: IsEmotionalAttackValid = generate(
                    is_valid_prompt,
                    IsEmotionalAttackValid,
                    self.simulator_model,
                )
                update_pbar(progress, task_id)

                if (
                    not compliance_res.non_compliant
                    and is_valid_res.is_valid_context
                ):
                    self.emotion = res.emotion_strategy
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
        prompt = EmotionalManipulationTemplate.enhance(attack, self.emotion)

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"...... 🎭 Injecting Emotion",
                total=self.max_retries * 3,
            )

            try:
                for _ in range(self.max_retries):
                    res: EmotionallyEnhancedAttack = await a_generate(
                        prompt, EmotionallyEnhancedAttack, self.simulator_model
                    )
                    enhanced_prompt = res.input
                    update_pbar(progress, task_id)

                    compliance_res = await a_evaluate_generation_compliance(
                        res.model_dump(), self.simulator_model
                    )
                    update_pbar(progress, task_id)

                    is_valid_prompt = (
                        EmotionalManipulationTemplate.is_valid_emotional_attack(
                            res.model_dump()
                        )
                    )
                    is_valid_res: IsEmotionalAttackValid = await a_generate(
                        is_valid_prompt,
                        IsEmotionalAttackValid,
                        self.simulator_model,
                    )
                    update_pbar(progress, task_id)

                    if (
                        not compliance_res.non_compliant
                        and is_valid_res.is_valid_context
                    ):
                        self.emotion = res.emotion_strategy
                        update_pbar(progress, task_id, advance_to_end=True)

                        return enhanced_prompt

            finally:
                update_pbar(progress, task_id, advance_to_end=True)

        return attack

    def get_name(self) -> str:
        return self.name
