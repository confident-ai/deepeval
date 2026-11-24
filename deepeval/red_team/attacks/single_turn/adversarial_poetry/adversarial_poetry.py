from typing import Optional, Union

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn.adversarial_poetry.template import (
    AdversarialPoetryTemplate,
)
from deepteam.attacks.single_turn.adversarial_poetry.schema import (
    EnhancedAttack,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class AdversarialPoetry(BaseSingleTurnAttack):
    name = "Adversarial Poetry"
    exploitability = Exploitability.HIGH
    description = "A stylistic obfuscation attack that reformats harmful requests into poetic verse using metaphor, imagery, and rhythmic structure to bypass safety mechanisms while preserving harmful intent."

    def __init__(
        self,
        weight: int = 1,
    ):
        self.weight = weight

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = AdversarialPoetryTemplate.enhance(attack)

        # Progress bar for single transformation
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 📖 Adversarial Poetry Enhancement",
                total=1,
            )

            try:
                res: EnhancedAttack = generate(
                    prompt, EnhancedAttack, self.simulator_model
                )
                enhanced_attack = res.input
            except:
                return attack

            update_pbar(progress, task_id, advance_to_end=True)

        return enhanced_attack

    async def a_enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = AdversarialPoetryTemplate.enhance(attack)

        # Async progress bar for single transformation
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 📖 Adversarial Poetry Enhancement",
                total=1,
            )

            try:
                res: EnhancedAttack = await a_generate(
                    prompt, EnhancedAttack, self.simulator_model
                )
                enhanced_attack = res.input
            except:
                update_pbar(progress, task_id, advance_to_end=True)
                return attack
            update_pbar(progress, task_id, advance_to_end=True)

        return enhanced_attack

    def get_name(self) -> str:
        return self.name
