from deepeval.optimization.policies.selection import (
    pareto_frontier,
    frequency_weights,
    select_candidate_pareto,
)
import random


def test_pareto_frontier_basic():
    candidate_scores_by_instance = {
        "a": [1, 0],
        "b": [0, 1],
        "c": [0.5, 0.5],
        "d": [0.4, 0.4],
    }
    # a and b are non-dominated; c is also non-dominated; d is dominated by c
    frontier_set = set(
        pareto_frontier(
            list(candidate_scores_by_instance.keys()),
            candidate_scores_by_instance,
        )
    )
    assert {"a", "b", "c"} == frontier_set


def test_frequency_weights_counts_matches_alg2():
    candidate_scores_by_instance = {
        "a": [1, 0, 1],
        "b": [0, 1, 0],
        "c": [0.9, 0.9, 0.9],  # good everywhere, likely non-dominated
    }
    frequency_by_candidate = frequency_weights(candidate_scores_by_instance)
    # According to Algorithm 2, frequency is computed over instance winners.
    # 'a' wins instances 0 and 2; 'b' wins instance 1; 'c' wins none.
    assert frequency_by_candidate == {"a": 2, "b": 1}


def test_select_candidate_deterministic():
    candidate_scores_by_instance = {
        "a": [1, 0, 1],
        "b": [0, 1, 0],
        "c": [0.9, 0.9, 0.9],
    }
    random_state = random.Random(123)
    selected_candidate = select_candidate_pareto(
        candidate_scores_by_instance, random_state=random_state
    )
    assert selected_candidate in candidate_scores_by_instance


def test_frequency_weights_excludes_nonwinners():
    candidate_scores_by_instance = {
        "a": [1, 0, 1],
        "b": [0, 1, 0],
        "c": [0.9, 0.9, 0.9],
    }
    assert "c" not in frequency_weights(candidate_scores_by_instance)
