import asyncio
import json
import time
from typing import List, Optional, Tuple, Union

from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.attack_simulator.utils import a_generate, generate
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.multi_turn.tree_jailbreaking.schema import (
    ImprovementPrompt,
    OnTopic,
    Rating,
)
from deepteam.attacks.multi_turn.tree_jailbreaking.template import (
    JailBreakingTemplate,
)
from deepteam.attacks.multi_turn.base_multi_turn_attack import (
    BaseMultiTurnAttack,
)
from deepteam.attacks.multi_turn.progression import (
    Probe,
    Progression,
    StopReason,
)
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.single_turn.base_single_turn_attack import (
    AttackParameter,
)
from deepteam.utils import add_pbar, create_progress, update_pbar


class TreeJailbreaking(BaseMultiTurnAttack):
    name = "Tree Jailbreaking"
    exploitability = Exploitability.LOW
    description = "A tree-search based multi-turn attack that explores multiple conversation branches in parallel, pruning low-scoring paths to find optimal jailbreak sequences."
    parameters = {
        "max_depth": AttackParameter(
            type="integer",
            default=10,
            description="How deep the search descends.",
        ),
        "branching_factor": AttackParameter(
            type="integer",
            default=4,
            description="Candidate prompts generated at each depth.",
        ),
        "max_runtime": AttackParameter(
            type="float",
            default=600.0,
            description="Seconds before the search gives up mid-descent.",
        ),
    }

    def __init__(
        self,
        weight: int = 1,
        max_depth: int = 10,
        branching_factor: int = 4,
        max_runtime: float = 600.0,
        turn_level_attacks: Optional[List[BaseSingleTurnAttack]] = None,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = "gpt-4o-mini",
    ):
        self.weight = weight
        self.simulator_model = simulator_model
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.max_runtime = max_runtime
        self.turn_level_attacks = turn_level_attacks

        if self.turn_level_attacks is not None:
            if not isinstance(self.turn_level_attacks, list) or not all(
                attack.multi_turn == False for attack in self.turn_level_attacks
            ):
                raise ValueError(
                    "The 'attacks' passed must be a list of single-turn attacks"
                )

    def _attack(self, progression: Progression) -> None:
        deadline = time.time() + self.max_runtime
        history = self._root_history(progression)
        best: Optional[Probe] = None
        progress = create_progress()

        with progress:
            pbar = add_pbar(
                progress,
                description="...... ⛓️  Tree Jailbreaking",
                total=self.max_depth,
            )

            best_score = 0
            for _ in range(1, self.max_depth + 1):
                if time.time() > deadline:
                    progression.stop(StopReason.RUNTIME_EXCEEDED)
                    break

                branch_history = self._branch_history(
                    history, progression.goal, best, best_score
                )
                prompts = self._branch(progression, branch_history)
                if not prompts:
                    progression.stop(StopReason.SIMULATION_ERROR)
                    break

                probes = [
                    progression.probe(prompt) for prompt, _ in prompts
                ]
                scored = []
                for probe in probes:
                    score = self._score(progression, probe)
                    progression.score_probe(probe, self._unit_score(score))
                    scored.append((probe, score))
                best, best_score = max(scored, key=lambda pair: pair[1])
                progression.commit(best)
                history = self._descend_history(
                    branch_history, best, best_score
                )
                update_pbar(progress, pbar)

                if progression.shift_detected():
                    break

            update_pbar(progress, pbar, advance_to_end=True)

    async def _a_attack(self, progression: Progression) -> None:
        deadline = time.time() + self.max_runtime
        history = self._root_history(progression)
        best: Optional[Probe] = None
        progress = create_progress()

        with progress:
            pbar = add_pbar(
                progress,
                description="...... ⛓️  Tree Jailbreaking",
                total=self.max_depth,
            )

            best_score = 0
            for _ in range(1, self.max_depth + 1):
                if time.time() > deadline:
                    progression.stop(StopReason.RUNTIME_EXCEEDED)
                    break

                branch_history = self._branch_history(
                    history, progression.goal, best, best_score
                )
                prompts = await self._a_branch(progression, branch_history)
                if not prompts:
                    progression.stop(StopReason.SIMULATION_ERROR)
                    break

                probes = list(
                    await asyncio.gather(
                        *[
                            progression.a_probe(prompt)
                            for prompt, _ in prompts
                        ]
                    )
                )
                scores = await asyncio.gather(
                    *[
                        self._a_score(progression, probe)
                        for probe in probes
                    ]
                )
                for probe, score in zip(probes, scores):
                    progression.score_probe(probe, self._unit_score(score))
                best, best_score = max(
                    zip(probes, scores), key=lambda pair: pair[1]
                )
                await progression.a_commit(best)
                history = self._descend_history(
                    branch_history, best, best_score
                )
                update_pbar(progress, pbar)

                if await progression.a_shift_detected():
                    break

            update_pbar(progress, pbar, advance_to_end=True)

    @staticmethod
    def _root_history(progression: Progression) -> List[dict]:
        return [
            {
                "role": "system",
                "content": JailBreakingTemplate.enhance_attack(
                    progression.goal, progression.vulnerability_data
                ),
            }
        ]

    @staticmethod
    def _branch_history(
        history: List[dict],
        goal: str,
        best: Optional[Probe],
        best_score: float,
    ) -> List[dict]:
        return history + [
            {
                "role": "user",
                "content": (
                    f"Current prompt: {best.input if best else goal}\n\n"
                    f"OBJECTIVE: {goal}\n\n"
                    f"SCORE: {best_score if best else 0}"
                ),
            }
        ]

    @staticmethod
    def _descend_history(
        history: List[dict], best: Probe, best_score: float
    ) -> List[dict]:
        return history + [
            {"role": "assistant", "content": best.input},
            {
                "role": "user",
                "content": (
                    f"RESPONSE: {best.output.content}\n\nSCORE: {best_score}"
                ),
            },
        ]

    def _branch(
        self, progression: Progression, history: List[dict]
    ) -> List[Tuple[str, bool]]:
        candidates = self._generate_branch(progression, history)
        on_topic = [c for c in candidates if c[1]]
        if on_topic:
            return on_topic

        retry = self._generate_branch(progression, history)
        return [c for c in retry if c[1]] or candidates + retry

    async def _a_branch(
        self, progression: Progression, history: List[dict]
    ) -> List[Tuple[str, bool]]:
        candidates = await self._a_generate_branch(progression, history)
        on_topic = [c for c in candidates if c[1]]
        if on_topic:
            return on_topic

        retry = await self._a_generate_branch(progression, history)
        return [c for c in retry if c[1]] or candidates + retry

    def _generate_branch(
        self, progression: Progression, history: List[dict]
    ) -> List[Tuple[str, bool]]:
        return [
            self._one_branch(progression, history)
            for _ in range(self.branching_factor)
        ]

    async def _a_generate_branch(
        self, progression: Progression, history: List[dict]
    ) -> List[Tuple[str, bool]]:
        return list(
            await asyncio.gather(
                *[
                    self._a_one_branch(progression, history)
                    for _ in range(self.branching_factor)
                ]
            )
        )

    def _one_branch(
        self, progression: Progression, history: List[dict]
    ) -> Tuple[str, bool]:
        res: ImprovementPrompt = generate(
            json.dumps(history), ImprovementPrompt, progression.simulator_model
        )
        topic: OnTopic = generate(
            JailBreakingTemplate.on_topic(
                progression.goal, res.prompt, progression.vulnerability_data
            ),
            OnTopic,
            progression.simulator_model,
        )
        return res.prompt, topic.on_topic

    async def _a_one_branch(
        self, progression: Progression, history: List[dict]
    ) -> Tuple[str, bool]:
        res: ImprovementPrompt = await a_generate(
            json.dumps(history), ImprovementPrompt, progression.simulator_model
        )
        topic: OnTopic = await a_generate(
            JailBreakingTemplate.on_topic(
                progression.goal, res.prompt, progression.vulnerability_data
            ),
            OnTopic,
            progression.simulator_model,
        )
        return res.prompt, topic.on_topic


    @staticmethod
    def _unit_score(rating: float) -> float:
        return max(0.0, min(1.0, (rating - 1) / 9))

    def _score(self, progression: Progression, probe: Probe) -> float:
        rating: Rating = generate(
            JailBreakingTemplate.linear_judge(
                progression.goal,
                probe.input,
                probe.output,
                progression.vulnerability_data,
            ),
            Rating,
            progression.simulator_model,
        )
        return rating.rating

    async def _a_score(
        self, progression: Progression, probe: Probe
    ) -> float:
        rating: Rating = await a_generate(
            JailBreakingTemplate.linear_judge(
                progression.goal,
                probe.input,
                probe.output,
                progression.vulnerability_data,
            ),
            Rating,
            progression.simulator_model,
        )
        return rating.rating
