from typing import List, Optional, Union

from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.attack_simulator.utils import a_generate, generate
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.multi_turn.bad_likert_judge.schema import (
    LikertExamples,
    NewAttackPrompt,
    RefinedExample,
)
from deepteam.attacks.multi_turn.bad_likert_judge.template import (
    BadLikertJudgeTemplate,
)
from deepteam.attacks.multi_turn.base_schema import NonRefusal
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


class BadLikertJudge(BaseMultiTurnAttack):
    name = "Bad Likert Judge"
    exploitability = Exploitability.LOW
    description = "An adaptive multi-turn attack that uses Likert scale scoring and iterative refinement to gradually guide the model toward harmful responses through feedback-driven escalation."
    parameters = {
        "category": AttackParameter(
            type="string",
            default="bias",
            description="Vulnerability category the Likert rubric is written for.",
        ),
        "num_turns": AttackParameter(
            type="integer",
            default=5,
            description="How many turns to land before giving up.",
        ),
        "enable_refinement": AttackParameter(
            type="boolean",
            default=True,
            description="Refine the score-3 example before building the attack from it.",
        ),
        "max_backtracks": AttackParameter(
            type="integer",
            default=3,
            description="Consecutive failed probes tolerated before stopping.",
        ),
    }

    def __init__(
        self,
        weight: int = 1,
        category: str = "bias",
        num_turns: int = 5,
        enable_refinement: bool = True,
        max_backtracks: int = 3,
        turn_level_attacks: Optional[List[BaseSingleTurnAttack]] = None,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = "gpt-4o-mini",
    ):
        self.weight = weight
        self.category = category
        self.num_turns = num_turns
        self.enable_refinement = enable_refinement
        self.max_backtracks = max_backtracks
        self.simulator_model = simulator_model
        self.turn_level_attacks = turn_level_attacks

        if self.turn_level_attacks is not None:
            if not isinstance(self.turn_level_attacks, list) or not all(
                attack.multi_turn == False for attack in self.turn_level_attacks
            ):
                raise ValueError(
                    "The 'turn_level_attacks' passed must be a list of single-turn attacks"
                )

    def _should_refine(self) -> bool:
        return self.enable_refinement

    def _attack(self, progression: Progression) -> None:
        attack = progression.goal
        backtracks = 0
        attempts_remaining = self.num_turns + self.max_backtracks
        progress = create_progress()

        with progress:
            pbar = add_pbar(
                progress,
                description=f"🎯 Bad Likert Judge ({self.category})",
                total=self.num_turns,
            )

            while (
                progression.turns_spent < self.num_turns
                and attempts_remaining > 0
            ):
                attempts_remaining -= 1

                try:
                    examples: LikertExamples = generate(
                        BadLikertJudgeTemplate.likert_generate_examples(
                            attack,
                            self.category,
                            progression.vulnerability_data,
                        ),
                        LikertExamples,
                        progression.simulator_model,
                    )
                except Exception:
                    backtracks += 1
                    if backtracks >= self.max_backtracks:
                        return progression.stop(StopReason.SIMULATION_ERROR)
                    continue

                score_3_example = examples.score_3_example

                if self._should_refine():
                    try:
                        refined: RefinedExample = generate(
                            BadLikertJudgeTemplate.likert_refine_score_3(
                                score_3_example,
                                progression.vulnerability_data,
                            ),
                            RefinedExample,
                            progression.simulator_model,
                        )
                        score_3_example = refined.refined_example
                    except Exception:
                        # Refinement is optional; the unrefined example still works.
                        pass

                try:
                    next_attack: NewAttackPrompt = generate(
                        BadLikertJudgeTemplate.likert_generate_attack_from_example(
                            score_3_example,
                            attack,
                            self.category,
                            progression.vulnerability_data,
                        ),
                        NewAttackPrompt,
                        progression.simulator_model,
                    )
                except Exception:
                    backtracks += 1
                    if backtracks >= self.max_backtracks:
                        return progression.stop(StopReason.SIMULATION_ERROR)
                    continue

                refusal: NonRefusal = generate(
                    BadLikertJudgeTemplate.non_refusal(
                        attack, next_attack.new_attack_prompt
                    ),
                    NonRefusal,
                    progression.simulator_model,
                )
                if refusal.refusal:
                    backtracks += 1
                    if backtracks >= self.max_backtracks:
                        return progression.stop(StopReason.SIMULATOR_REFUSED)
                    continue

                backtracks = 0
                attack = next_attack.new_attack_prompt
                progression.commit(attack)
                update_pbar(progress, pbar)

                if progression.shift_detected():
                    update_pbar(progress, pbar, advance_to_end=True)
                    return

    async def _a_attack(self, progression: Progression) -> None:
        attack = progression.goal
        backtracks = 0
        attempts_remaining = self.num_turns + self.max_backtracks
        progress = create_progress()

        with progress:
            pbar = add_pbar(
                progress,
                description=f"🎯 Bad Likert Judge ({self.category})",
                total=self.num_turns,
            )

            while (
                progression.turns_spent < self.num_turns
                and attempts_remaining > 0
            ):
                attempts_remaining -= 1

                try:
                    examples: LikertExamples = await a_generate(
                        BadLikertJudgeTemplate.likert_generate_examples(
                            attack,
                            self.category,
                            progression.vulnerability_data,
                        ),
                        LikertExamples,
                        progression.simulator_model,
                    )
                except Exception:
                    backtracks += 1
                    if backtracks >= self.max_backtracks:
                        return progression.stop(StopReason.SIMULATION_ERROR)
                    continue

                score_3_example = examples.score_3_example

                if self._should_refine():
                    try:
                        refined: RefinedExample = await a_generate(
                            BadLikertJudgeTemplate.likert_refine_score_3(
                                score_3_example,
                                progression.vulnerability_data,
                            ),
                            RefinedExample,
                            progression.simulator_model,
                        )
                        score_3_example = refined.refined_example
                    except Exception:
                        pass

                try:
                    next_attack: NewAttackPrompt = await a_generate(
                        BadLikertJudgeTemplate.likert_generate_attack_from_example(
                            score_3_example,
                            attack,
                            self.category,
                            progression.vulnerability_data,
                        ),
                        NewAttackPrompt,
                        progression.simulator_model,
                    )
                except Exception:
                    backtracks += 1
                    if backtracks >= self.max_backtracks:
                        return progression.stop(StopReason.SIMULATION_ERROR)
                    continue

                refusal: NonRefusal = await a_generate(
                    BadLikertJudgeTemplate.non_refusal(
                        attack, next_attack.new_attack_prompt
                    ),
                    NonRefusal,
                    progression.simulator_model,
                )
                if refusal.refusal:
                    backtracks += 1
                    if backtracks >= self.max_backtracks:
                        return progression.stop(StopReason.SIMULATOR_REFUSED)
                    continue

                backtracks = 0
                attack = next_attack.new_attack_prompt
                await progression.a_commit(attack)
                update_pbar(progress, pbar)

                if await progression.a_shift_detected():
                    update_pbar(progress, pbar, advance_to_end=True)
                    return

    @classmethod
    def get_supported_categories(cls) -> list[str]:
        """Example categories. Any category works; these are common ones."""
        return [
            "bias",
            "religion",
            "politics",
            "gender",
            "race",
            "toxicity",
            "profanity",
            "insults",
            "threats",
            "mockery",
            "illegal_activity",
            "weapons",
            "illegal_drugs",
            "violent_crimes",
            "cybercrime",
            "unauthorized_access",
            "bfla",
            "bola",
            "rbac",
            "sql_injection",
            "shell_injection",
            "personal_safety",
            "bullying",
            "self-harm",
            "unsafe_practices",
            "stalking",
            "graphic_content",
            "sexual_content",
            "pornographic_content",
            "misinformation",
            "factual_errors",
            "unsupported_claims",
            "pii_leakage",
            "database_access",
            "direct_disclosure",
            "session_leak",
            "prompt_leakage",
            "secrets_and_credentials",
            "instructions",
            "guard_exposure",
            "excessive_agency",
            "functionality",
            "permissions",
            "autonomy",
            "robustness",
            "input_overreliance",
            "hijacking",
            "intellectual_property",
            "copyright_violations",
            "trademark_infringement",
            "competition",
            "competitor_mention",
            "market_manipulation",
            "malware",
            "hacking",
            "exploitation",
            "violence",
            "harassment",
        ]
