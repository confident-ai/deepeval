from typing import Sequence, List
from dataclasses import dataclass
from deepeval.optimization.gepa.loop import GEPARunner, GEPAConfig
from deepeval.optimization.types import (
    Candidate,
    Evaluator,
    GoldenLike,
    ModuleId,
)
from tests.test_core.stubs import AddBetterRewriter


# A tiny evaluator that rewards prompts containing the word "BETTER"
@dataclass
class ToyEvaluator(Evaluator):
    pareto_set_size: int
    module_ids: List[ModuleId]

    def score_on_pareto(self, candidate: Candidate) -> List[float]:
        base_score = sum(
            "BETTER" in prompt for prompt in candidate.prompts.values()
        )
        return [float(base_score) for _ in range(self.pareto_set_size)]

    def minibatch_score(
        self, candidate: Candidate, minibatch: Sequence[GoldenLike]
    ) -> float:
        return float(
            sum("BETTER" in prompt for prompt in candidate.prompts.values())
        )

    def minibatch_feedback(
        self,
        candidate: Candidate,
        module_id: ModuleId,
        minibatch: Sequence[GoldenLike],
    ) -> str:
        return "Try adding BETTER."

    def select_module(self, candidate: Candidate) -> ModuleId:
        # Always mutate the first module for determinism
        return self.module_ids[0]


def test_gepa_improves_under_toy_evaluator():
    root_candidate = Candidate.new(prompts={"gen": "do x", "crit": "check y"})
    evaluator = ToyEvaluator(pareto_set_size=5, module_ids=["gen", "crit"])
    config = GEPAConfig(
        budget=3, minibatch_size=2, seed=42, rewriter=AddBetterRewriter()
    )
    runner = GEPARunner(evaluator, config=config)

    feedback_examples = list(range(20))
    pareto_examples = list(range(5))
    best_candidate, optimization_report = runner.optimize(
        root_candidate, feedback_examples, pareto_examples
    )

    assert "BETTER" in best_candidate.prompts["gen"]
    assert len(optimization_report["accepted_steps"]) >= 1
