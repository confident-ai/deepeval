from typing import Callable, Dict, List, Optional, Union

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from deepteam.attacks.attack_simulator.utils import a_generate, generate
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.multi_turn.sequential_break.schema import (
    AdaptiveDialogueTemplate,
    AdaptiveGameEnvironmentTemplate,
    AdaptiveQuestionBankTemplate,
    DialogueJudge,
    DialogueTypeLiteral,
    GameEnvironmentJudge,
    ImprovedAttack,
    QuestionBankJudge,
    RewrittenDialogue,
    SequentialJailbreakTypeLiteral,
)
from deepteam.attacks.multi_turn.sequential_break.template import (
    SequentialBreakTemplate,
)
from deepteam.attacks.multi_turn.base_multi_turn_attack import (
    BaseMultiTurnAttack,
)
from deepteam.attacks.multi_turn.progression import Progression
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.single_turn.base_single_turn_attack import (
    AttackParameter,
)
from deepteam.utils import add_pbar, create_progress, update_pbar


class _AttackTypeSpec(BaseModel):
    """Everything that varies between the three attack types."""

    rewrite_prompt: Callable
    static_template: Callable
    adaptive_prompt: Callable
    adaptive_schema: type
    judge_prompt: Callable
    judge_schema: type


SPECS: Dict[str, _AttackTypeSpec] = {
    "question_bank": _AttackTypeSpec(
        rewrite_prompt=SequentialBreakTemplate.rewrite_question_bank_prompt,
        static_template=SequentialBreakTemplate.get_question_bank_template,
        adaptive_prompt=SequentialBreakTemplate.generate_adaptive_question_bank_template,
        adaptive_schema=AdaptiveQuestionBankTemplate,
        judge_prompt=SequentialBreakTemplate.question_bank_judge,
        judge_schema=QuestionBankJudge,
    ),
    "game_environment": _AttackTypeSpec(
        rewrite_prompt=SequentialBreakTemplate.rewrite_game_environment_prompt,
        static_template=SequentialBreakTemplate.get_game_environment_template,
        adaptive_prompt=SequentialBreakTemplate.generate_adaptive_game_environment_template,
        adaptive_schema=AdaptiveGameEnvironmentTemplate,
        judge_prompt=SequentialBreakTemplate.game_environment_judge,
        judge_schema=GameEnvironmentJudge,
    ),
    "dialogue": _AttackTypeSpec(
        rewrite_prompt=SequentialBreakTemplate.rewrite_dialogue_prompt,
        static_template=SequentialBreakTemplate.get_dialogue_template,
        adaptive_prompt=SequentialBreakTemplate.generate_adaptive_dialogue_template,
        adaptive_schema=AdaptiveDialogueTemplate,
        judge_prompt=SequentialBreakTemplate.dialogue_judge,
        judge_schema=DialogueJudge,
    ),
}


class SequentialJailbreak(BaseMultiTurnAttack):
    name = "Sequential Jailbreak"
    exploitability = Exploitability.LOW
    description = "A scenario-based multi-turn attack that disguises harmful requests within dialogue, question banks, or game environments to incrementally build toward policy violations."
    parameters = {
        "type": AttackParameter(
            type="enum",
            default="dialogue",
            options=list(SPECS),
            description="Which scenario the harmful request is disguised inside.",
        ),
        "persona": AttackParameter(
            type="enum",
            options=["prisoner", "student", "researcher", "generic"],
            description="Who the dialogue is with. Dialogue type only.",
        ),
        "num_turns": AttackParameter(
            type="integer",
            default=5,
            description="How many turns to spend before giving up.",
        ),
    }

    def __init__(
        self,
        weight: int = 1,
        type: Optional[SequentialJailbreakTypeLiteral] = None,
        persona: Optional[DialogueTypeLiteral] = None,
        num_turns: int = 5,
        turn_level_attacks: Optional[List[BaseSingleTurnAttack]] = None,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = "gpt-4o-mini",
    ):
        self.weight = weight
        self.attack_type = type if type is not None else "dialogue"
        self.persona = persona
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

        if (
            self.attack_type in ["question_bank", "game_environment"]
            and persona is not None
        ):
            raise ValueError(
                f"{self.attack_type} attack type does not use persona parameter"
            )
        if self.attack_type == "dialogue" and persona is None:
            self.persona = "student"

    @property
    def _spec(self) -> _AttackTypeSpec:
        return SPECS[self.attack_type]

    def _attack(self, progression: Progression) -> None:
        spec = self._spec
        base_attack = progression.goal
        previous_template: Optional[str] = None
        previous_response: Optional[str] = None
        progress = create_progress()

        with progress:
            pbar = add_pbar(
                progress,
                description=f"🎯 Sequential Jailbreak ({self.attack_type})",
                total=self.num_turns,
            )

            for _ in range(self.num_turns):
                rewritten: RewrittenDialogue = generate(
                    spec.rewrite_prompt(
                        base_attack, progression.vulnerability_data
                    ),
                    RewrittenDialogue,
                    progression.simulator_model,
                )
                content = self._clean(rewritten.paraphrased_version)

                if previous_template is None:
                    attack = self._static_template(spec, content)
                else:
                    adaptive = generate(
                        self._adaptive_prompt(
                            spec,
                            progression,
                            content,
                            previous_template,
                            previous_response,
                        ),
                        spec.adaptive_schema,
                        progression.simulator_model,
                    )
                    attack = adaptive.template

                response = progression.commit(attack)
                update_pbar(progress, pbar)

                if progression.shift_detected():
                    update_pbar(progress, pbar, advance_to_end=True)
                    return

                judge = generate(
                    spec.judge_prompt(
                        progression.goal,
                        attack,
                        response.content,
                        progression.vulnerability_data,
                    ),
                    spec.judge_schema,
                    progression.simulator_model,
                )

                try:
                    improved: ImprovedAttack = generate(
                        SequentialBreakTemplate.improve_attack_prompt(
                            progression.goal,
                            attack,
                            response.content,
                            self._judge_feedback(judge),
                            self.attack_type,
                            progression.vulnerability_data,
                        ),
                        ImprovedAttack,
                        progression.simulator_model,
                    )
                    base_attack = improved.improved_attack
                except Exception:
                    # Keep escalating from the current base attack.
                    pass

                previous_template = attack
                previous_response = response.content

    async def _a_attack(self, progression: Progression) -> None:
        spec = self._spec
        base_attack = progression.goal
        previous_template: Optional[str] = None
        previous_response: Optional[str] = None
        progress = create_progress()

        with progress:
            pbar = add_pbar(
                progress,
                description=f"🎯 Sequential Jailbreak ({self.attack_type})",
                total=self.num_turns,
            )

            for _ in range(self.num_turns):
                rewritten: RewrittenDialogue = await a_generate(
                    spec.rewrite_prompt(
                        base_attack, progression.vulnerability_data
                    ),
                    RewrittenDialogue,
                    progression.simulator_model,
                )
                content = self._clean(rewritten.paraphrased_version)

                if previous_template is None:
                    attack = self._static_template(spec, content)
                else:
                    adaptive = await a_generate(
                        self._adaptive_prompt(
                            spec,
                            progression,
                            content,
                            previous_template,
                            previous_response,
                        ),
                        spec.adaptive_schema,
                        progression.simulator_model,
                    )
                    attack = adaptive.template

                response = await progression.a_commit(attack)
                update_pbar(progress, pbar)

                if await progression.a_shift_detected():
                    update_pbar(progress, pbar, advance_to_end=True)
                    return

                judge = await a_generate(
                    spec.judge_prompt(
                        progression.goal,
                        attack,
                        response.content,
                        progression.vulnerability_data,
                    ),
                    spec.judge_schema,
                    progression.simulator_model,
                )

                try:
                    improved: ImprovedAttack = await a_generate(
                        SequentialBreakTemplate.improve_attack_prompt(
                            progression.goal,
                            attack,
                            response.content,
                            self._judge_feedback(judge),
                            self.attack_type,
                            progression.vulnerability_data,
                        ),
                        ImprovedAttack,
                        progression.simulator_model,
                    )
                    base_attack = improved.improved_attack
                except Exception:
                    pass

                previous_template = attack
                previous_response = response.content

    @staticmethod
    def _clean(content: str) -> str:
        return (
            content.replace("Paraphrased version:", "")
            .strip()
            .replace("<end>", "")
            .strip()
        )

    def _static_template(self, spec: _AttackTypeSpec, content: str) -> str:
        if self.attack_type == "dialogue":
            return spec.static_template(content, self.persona)
        return spec.static_template(content)

    def _adaptive_prompt(
        self,
        spec: _AttackTypeSpec,
        progression: Progression,
        content: str,
        previous_template: str,
        previous_response: Optional[str],
    ) -> str:
        feedback = "Previous probe was not successful"
        if previous_response:
            feedback += f"\nPrevious response: {previous_response[:500]}"

        if self.attack_type == "dialogue":
            return spec.adaptive_prompt(
                content,
                self.persona or "student",
                previous_template,
                previous_response or "",
                feedback,
                progression.vulnerability_data,
            )
        return spec.adaptive_prompt(
            content,
            previous_template,
            previous_response or "",
            feedback,
            progression.vulnerability_data,
        )

    @staticmethod
    def _judge_feedback(judge) -> str:
        feedback = f"Harmful: {judge.harmful}"
        if hasattr(judge, "reasoning"):
            feedback += f"\nReasoning: {judge.reasoning}"
        return feedback
