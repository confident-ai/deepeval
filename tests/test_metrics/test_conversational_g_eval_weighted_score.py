"""Unit tests for ConversationalGEval.generate_weighted_summed_score.

Regression coverage for a ZeroDivisionError that occurred when every top
logprob token is filtered out (all below the 1% linear-probability floor or
non-decimal), leaving sum_linear_probability == 0. The non-conversational
GEval path already guards this (see calculate_weighted_summed_score); this
mirrors that guard for ConversationalGEval.
"""

import math
from types import SimpleNamespace

from deepeval.metrics import ConversationalGEval


def _chat_completion(match_token, top_logprobs):
    """Build a minimal ChatCompletion-like object with one logprob content
    entry whose token equals match_token and whose top_logprobs are given as
    (token, logprob) pairs."""
    top = [SimpleNamespace(token=t, logprob=lp) for t, lp in top_logprobs]
    content = [SimpleNamespace(token=match_token, top_logprobs=top)]
    logprobs = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=logprobs)])


def _call(raw_score, raw_response):
    # The method uses no instance state, so bypass __init__ (which would need a
    # model / API key) and invoke it directly.
    metric = object.__new__(ConversationalGEval)
    return metric.generate_weighted_summed_score(raw_score, raw_response)


def test_all_tokens_filtered_falls_back_to_raw_score():
    # Every top token is below 1% linear probability, so all are filtered and
    # sum_linear_probability stays 0. Must not raise ZeroDivisionError.
    resp = _chat_completion("5", [("5", -10.0), ("6", -12.0)])
    assert _call(5, resp) == 5


def test_non_decimal_tokens_filtered_falls_back_to_raw_score():
    # High-probability but non-decimal tokens are filtered too.
    resp = _chat_completion("5", [("five", math.log(0.9)), (" ", math.log(0.8))])
    assert _call(5, resp) == 5


def test_normal_weighted_score_is_computed():
    resp = _chat_completion(
        "5", [("5", math.log(0.9)), ("4", math.log(0.1))]
    )
    score = _call(5, resp)
    assert 4.0 < score <= 5.0
