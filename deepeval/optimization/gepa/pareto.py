from __future__ import annotations
import random
from ..types import CandidateId, ScoreTable
from ..policies.selection import select_candidate_pareto


def pick_candidate(pareto_scores: ScoreTable, *, seed: int) -> CandidateId:
    random_state = random.Random(seed)
    return select_candidate_pareto(pareto_scores, random_state=random_state)
