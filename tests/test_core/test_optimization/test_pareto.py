import random

from deepeval.optimization.policies.selection import (
    pareto_frontier,
    frequency_weights,
    select_prompt_configuration_pareto,
)


def test_pareto_frontier_basic():
    candidate_scores_by_instance = {
        "a": [1, 0],
        "b": [0, 1],
        "c": [0.5, 0.5],
        "d": [0.4, 0.4],
    }
    # a and b are non-dominated, c is also non-dominated, d is dominated by c
    frontier_set = set(
        pareto_frontier(
            list(candidate_scores_by_instance.keys()),
            candidate_scores_by_instance,
        )
    )
    assert {"a", "b", "c"} == frontier_set


def test_frequency_weights_counts_matches_alg2_with_global_frontier():
    candidate_scores_by_instance = {
        "a": [1, 0, 1],
        "b": [0, 1, 0],
        "c": [0.9, 0.9, 0.9],  # good everywhere but not an instance winner
    }
    frequency_by_candidate = frequency_weights(candidate_scores_by_instance)

    # According to Algorithm 2 + global frontier:
    # - instance winners:
    #   i=0 -> a
    #   i=1 -> b
    #   i=2 -> a
    # - Candidate union among winners: {a, b}
    # - Global frontier among {a, b} is {a, b}
    # => a appears twice, b once, c is excluded.
    assert frequency_by_candidate == {"a": 2, "b": 1}


def test_select_prompt_configuration_deterministic_membership():
    candidate_scores_by_instance = {
        "a": [1, 0, 1],
        "b": [0, 1, 0],
        "c": [0.9, 0.9, 0.9],
    }
    random_state = random.Random(123)

    selected = select_prompt_configuration_pareto(
        candidate_scores_by_instance,
        random_state=random_state,
    )

    # Must return a valid key from the score table
    assert selected in candidate_scores_by_instance


def test_frequency_weights_excludes_nonwinners_and_dominated():
    candidate_scores_by_instance = {
        "a": [1, 0, 1],
        "b": [0, 1, 0],
        "c": [0.9, 0.9, 0.9],
    }
    freq = frequency_weights(candidate_scores_by_instance)

    # Only a and b should be present with the current algorithm
    assert set(freq.keys()) == {"a", "b"}
    assert "c" not in freq
