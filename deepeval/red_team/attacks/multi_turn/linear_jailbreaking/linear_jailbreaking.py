from typing import List, Optional, Union

from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.attack_simulator.utils import a_generate, generate
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.multi_turn.base_schema import NonRefusal
from deepteam.attacks.multi_turn.base_template import BaseMultiTurnTemplate
from deepteam.attacks.multi_turn.linear_jailbreaking.schema import (
    Feedback,
    Improvement,
)
from deepteam.attacks.multi_turn.linear_jailbreaking.template import (
    JailBreakingTemplate,
)
from deepteam.attacks.multi_turn.base_multi_turn_attack import (
    BaseMultiTurnAttack,
)
from deepteam.attacks.multi_turn.progression import (
    Progression,
    StopReason,
)
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.single_turn.base_single_turn_attack import (
    AttackParameter,
)
from deepteam.utils import add_pbar, create_progress, update_pbar


class LinearJailbreaking(BaseMultiTurnAttack):
    name = "Linear Jailbreaking"
    exploitability = Exploitability.LOW
    description = "An iterative multi-turn attack that uses LLM-as-judge feedback to refine prompts across turns, systematically improving attack effectiveness until jailbreak succeeds."
    parameters = {
        "num_turns": AttackParameter(
            type="integer",
            default=5,
            description="How many turns to spend before giving up.",
        )
    }

    def __init__(
        self,
        weight: int = 1,
        num_turns: int = 5,
        turn_level_attacks: Optional[List[BaseSingleTurnAttack]] = None,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = "gpt-4o-mini",
    ):
        self.weight = weight
        self.num_turns = num_turns
        self.simulator_model = simulator_model
        self.turn_level_attacks = turn_level_attacks

        if self.turn_level_attacks is not None:
            if not isinstance(self.turn_level_attacks, list) or not all(
                attack.multi_turn == False for attack in self.turn_level_attacks
            ):
                raise ValueError(
                    "The 'turn_level_attacks' passed must be a list of single-turn attacks"
                )

    def _attack(self, progression: Progression) -> None:
        attack = progression.goal
        progress = create_progress()

        with progress:
            pbar = add_pbar(
                progress,
                description="...... ⛓️  Linear Jailbreaking",
                total=self.num_turns,
            )

            for _ in range(self.num_turns):
                feedback: Feedback = generate(
                    JailBreakingTemplate.linear_judge(
                        progression.goal,
                        attack,
                        progression.last_response,
                        progression.vulnerability_data,
                    ),
                    Feedback,
                    progression.simulator_model,
                )

                improvement: Improvement = generate(
                    JailBreakingTemplate.improvement_prompt(
                        progression.turns,
                        feedback.suggestion,
                        progression.vulnerability_data,
                    ),
                    Improvement,
                    progression.simulator_model,
                )

                refusal: NonRefusal = generate(
                    BaseMultiTurnTemplate.non_refusal(
                        progression.goal, improvement.new_prompt
                    ),
                    NonRefusal,
                    progression.simulator_model,
                )
                if refusal.refusal:
                    update_pbar(progress, pbar, advance_to_end=True)
                    return progression.stop(StopReason.SIMULATOR_REFUSED)

                attack = improvement.new_prompt
                progression.commit(attack)
                update_pbar(progress, pbar)

                if progression.shift_detected():
                    update_pbar(progress, pbar, advance_to_end=True)
                    return

    async def _a_attack(self, progression: Progression) -> None:
        attack = progression.goal
        progress = create_progress()

        with progress:
            pbar = add_pbar(
                progress,
                description="...... ⛓️  Linear Jailbreaking",
                total=self.num_turns,
            )

            for _ in range(self.num_turns):
                feedback: Feedback = await a_generate(
                    JailBreakingTemplate.linear_judge(
                        progression.goal,
                        attack,
                        progression.last_response,
                        progression.vulnerability_data,
                    ),
                    Feedback,
                    progression.simulator_model,
                )

                improvement: Improvement = await a_generate(
                    JailBreakingTemplate.improvement_prompt(
                        progression.turns,
                        feedback.suggestion,
                        progression.vulnerability_data,
                    ),
                    Improvement,
                    progression.simulator_model,
                )

                refusal: NonRefusal = await a_generate(
                    BaseMultiTurnTemplate.non_refusal(
                        progression.goal, improvement.new_prompt
                    ),
                    NonRefusal,
                    progression.simulator_model,
                )
                if refusal.refusal:
                    update_pbar(progress, pbar, advance_to_end=True)
                    return progression.stop(StopReason.SIMULATOR_REFUSED)

                attack = improvement.new_prompt
                await progression.a_commit(attack)
                update_pbar(progress, pbar)

                if await progression.a_shift_detected():
                    update_pbar(progress, pbar, advance_to_end=True)
                    return
