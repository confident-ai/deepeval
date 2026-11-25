from typing import Any, Dict, List

from deepeval.models import DeepEvalBaseLLM

from deepeval.red_team.attacks.attack_simulator.utils import a_generate
from deepeval.red_team.orchestration.schema import CriteriaVerdict
from deepeval.red_team.orchestration.templates import SubAgentTemplate


class CriteriaJudge:

    def __init__(
        self,
        model: DeepEvalBaseLLM,
        task: str,
        scenario: str,
    ):
        self.model = model
        self.task = task
        self.scenario = scenario

    async def a_judge(
        self,
        criteria: str,
        current_turn: Dict[str, str],
        chat_history: List[Dict[str, str]],
        previous_scores: List[Dict[str, Any]],
    ) -> CriteriaVerdict:
        return await a_generate(
            SubAgentTemplate.judge(
                self.task,
                self.scenario,
                criteria,
                current_turn,
                chat_history,
                previous_scores,
            ),
            CriteriaVerdict,
            self.model,
        )
