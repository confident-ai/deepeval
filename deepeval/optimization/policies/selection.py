from __future__ import annotations
from typing import Dict, List, Sequence
import random

from deepeval.errors import DeepEvalError
from deepeval.optimization.types import PromptConfigurationId, ScoreTable


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
    prompt_configuration_ids: Sequence[PromptConfigurationId],
    score_table: ScoreTable,
) -> List[PromptConfigurationId]:
    """
    Compute the set of non-dominated candidates given their scores.
    Returns PromptConfigurationIds on the Pareto frontier.
    """
    frontier: List[PromptConfigurationId] = []
    for prompt_configuration_id in prompt_configuration_ids:
        candidate_vector = score_table[prompt_configuration_id]
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
        frontier.append(prompt_configuration_id)

    return frontier


def frequency_weights(
    score_table: ScoreTable,
) -> Dict[PromptConfigurationId, int]:
    """
    Build best sets, remove dominated candidates, and count appearances.

    Returns:
        A map {prompt_configuration_id -> frequency} counting how often each
        globally non-dominated prompt configuration appears among the instance
        Pareto sets.
    """
    if not score_table:
        return {}

    # Assume all score vectors have the same length.
    example_vector = next(iter(score_table.values()))
    num_instances = len(example_vector)
    all_candidates = list(score_table.keys())

    per_instance_frontiers: List[List[PromptConfigurationId]] = []
    for i in range(num_instances):
        best_score_i = max(
            score_table[prompt_configuration_id][i]
            for prompt_configuration_id in all_candidates
        )
        winners_i = [
            prompt_configuration_id
            for prompt_configuration_id in all_candidates
            if score_table[prompt_configuration_id][i] == best_score_i
        ]

        # Instance frontier among winners. We pass 1-D score vectors
        # so this reduces to "all candidates with the max score at instance i",
        instance_frontier = pareto_frontier(
            winners_i,
            {
                prompt_configuration_id: [
                    score_table[prompt_configuration_id][i]
                ]
                for prompt_configuration_id in winners_i
            },
        )
        per_instance_frontiers.append(instance_frontier)

    # Global candidate set appearing in any winners
    candidate_union = sorted(
        {
            prompt_configuration_id
            for winners in per_instance_frontiers
            for prompt_configuration_id in winners
        }
    )
    global_frontier = pareto_frontier(candidate_union, score_table)

    # Count frequency only for candidates on the global frontier
    frequency_by_prompt_config: Dict[PromptConfigurationId, int] = {
        prompt_configuration_id: 0
        for prompt_configuration_id in global_frontier
    }
    for winners in per_instance_frontiers:
        for prompt_configuration_id in winners:
            if prompt_configuration_id in frequency_by_prompt_config:
                frequency_by_prompt_config[prompt_configuration_id] += 1

    return frequency_by_prompt_config


def sample_by_frequency(
    frequency_by_prompt_config: Dict[PromptConfigurationId, int],
    *,
    random_state: random.Random,
) -> PromptConfigurationId:
    """
    Sample a prompt configuration id with probability proportional to its frequency.
    Falls back to uniform if the total weight is zero.
    """
    if not frequency_by_prompt_config:
        raise DeepEvalError("No prompt configurations to sample.")

    items = list(frequency_by_prompt_config.items())
    total_weight = sum(weight for _, weight in items)

    if total_weight == 0:
        # Uniform fallback
        return random_state.choice(
            [prompt_configuration_id for prompt_configuration_id, _ in items]
        )

    r = random_state.uniform(0, total_weight)
    cumulative = 0.0
    for prompt_configuration_id, weight in items:
        cumulative += weight
        if r <= cumulative:
            return prompt_configuration_id
    return items[-1][0]


def select_prompt_configuration_pareto(
    score_table: ScoreTable, *, random_state: random.Random
) -> PromptConfigurationId:
    """
    Frequency weighted sampling over the Pareto winners,
    restricted to globally non-dominated prompt configurations. A configuration
    is globally non-dominated if no other configuration dominates it using
    the full vector.
    """
    freq = frequency_weights(score_table)
    return sample_by_frequency(freq, random_state=random_state)
