from __future__ import annotations
from typing import Sequence, List
from ..types import Candidate, Evaluator, ModuleId


class DeepEvalEvaluator(Evaluator):
    def __init__(self, *, metric, d_pareto: Sequence[object], module_selector):
        self.metric = metric
        self._d_pareto = d_pareto
        self._module_selector = module_selector

    def score_on_pareto(self, cand: Candidate) -> List[float]:
        # TODO: call evaluation runner on each (x_i, m_i) in d_pareto,
        #       return [ μ( cand(x_i), m_i ) ... ]
        raise NotImplementedError

    def minibatch_score(
        self, cand: Candidate, batch: Sequence[object]
    ) -> float:
        # TODO: run cand on batch, return mean μ
        raise NotImplementedError

    def minibatch_feedback(
        self, cand: Candidate, module: ModuleId, batch: Sequence[object]
    ) -> str:
        # TODO: collect metric.reason and module span traces; concatenate into μ_f
        raise NotImplementedError

    def select_module(self, cand: Candidate) -> ModuleId:
        return self._module_selector(cand)
