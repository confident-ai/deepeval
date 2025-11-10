from __future__ import annotations
from typing import Dict, List, Sequence
import random

from ..types import CandidateId, ScoreTable


def _is_dominated(
    candidate_scores: List[float], other_scores: List[float]
) -> bool:
    """
    Return True if `candidate_scores` is dominated by `other_scores`:
    (other >= candidate on all dimensions) AND (other > candidate on at least one).
    """
    other_ge_everywhere = all(
        other_score >= candidate_score
        for candidate_score, other_score in zip(candidate_scores, other_scores)
    )
    other_gt_somewhere = any(
        other_score > candidate_score
        for candidate_score, other_score in zip(candidate_scores, other_scores)
    )
    return other_ge_everywhere and other_gt_somewhere


def pareto_frontier(
    candidate_ids: Sequence[CandidateId], score_table: ScoreTable
) -> List[CandidateId]:
    """
    Compute the set of non-dominated candidates given their scores.
    Returns candidate ids on the Pareto frontier.
    """
    frontier: List[CandidateId] = []
    for candidate_id in candidate_ids:
        candidate_vector = score_table[candidate_id]
        dominated = False

        # If any existing frontier member dominates this candidate, skip it.
        for frontier_id in frontier:
            if _is_dominated(candidate_vector, score_table[frontier_id]):
                dominated = True
                break
        if dominated:
            continue

        # Remove any frontier member that is dominated by this candidate.
        frontier = [
            f_id
            for f_id in frontier
            if not _is_dominated(score_table[f_id], candidate_vector)
        ]
        frontier.append(candidate_id)

    return frontier


def frequency_weights(score_table: ScoreTable) -> Dict[CandidateId, int]:
    """
    Build best sets, remove dominated candidates, and count appearances.

    Returns:
        A map {candidate_id -> frequency} counting how often each globally
        non-dominated candidate appears among the instance Pareto sets.
    """
    if not score_table:
        return {}

    # Assume all score vectors have the same length.
    example_vector = next(iter(score_table.values()))
    num_instances = len(example_vector)
    all_candidates = list(score_table.keys())

    per_instance_frontiers: List[List[CandidateId]] = []
    for i in range(num_instances):
        best_score_i = max(
            score_table[candidate_id][i] for candidate_id in all_candidates
        )
        winners_i = [
            candidate_id
            for candidate_id in all_candidates
            if score_table[candidate_id][i] == best_score_i
        ]

        # Instance frontier among winners. We pass 1-D score vectors
        # so this reduces to "all candidates with the max score at instance i",
        instance_frontier = pareto_frontier(
            winners_i,
            {
                candidate_id: [score_table[candidate_id][i]]
                for candidate_id in winners_i
            },
        )
        per_instance_frontiers.append(instance_frontier)

    # Global candidate set appearing in any winners
    candidate_union = sorted(
        {
            candidate_id
            for winners in per_instance_frontiers
            for candidate_id in winners
        }
    )
    global_frontier = pareto_frontier(candidate_union, score_table)

    # Count frequency only for candidates on the global frontier
    frequency_by_candidate: Dict[CandidateId, int] = {
        candidate_id: 0 for candidate_id in global_frontier
    }
    for winners in per_instance_frontiers:
        for candidate_id in winners:
            if candidate_id in frequency_by_candidate:
                frequency_by_candidate[candidate_id] += 1

    return frequency_by_candidate


def sample_by_frequency(
    frequency_by_candidate: Dict[CandidateId, int],
    *,
    random_state: random.Random,
) -> CandidateId:
    """
    Sample a candidate id with probability proportional to its frequency.
    Falls back to uniform if the total weight is zero.
    """
    if not frequency_by_candidate:
        raise ValueError("No candidates to sample.")

    items = list(frequency_by_candidate.items())
    total_weight = sum(weight for _, weight in items)

    if total_weight == 0:
        # Uniform fallback
        return random_state.choice([candidate_id for candidate_id, _ in items])

    r = random_state.uniform(0, total_weight)
    cumulative = 0.0
    for candidate_id, weight in items:
        cumulative += weight
        if r <= cumulative:
            return candidate_id
    return items[-1][0]


def select_candidate_pareto(
    score_table: ScoreTable, *, random_state: random.Random
) -> CandidateId:
    """
    Frequency weighted sampling over the Pareto winners,
    restricted to globally non-dominated candidates. A candidate
    is globally non-dominated if no other candidate dominates it using
    the full vector.
    """
    freq = frequency_weights(score_table)
    return sample_by_frequency(freq, random_state=random_state)
