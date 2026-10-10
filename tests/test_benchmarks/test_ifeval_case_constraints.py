"""Regression test: IFEval's `change_case:english_lowercase` /
`english_uppercase` checkers rejected responses with no cased characters at
all, even though such responses trivially satisfy "no capital letters are
allowed" (there are no letters to be capital).

`response.lower() == response` already means "response has no uppercase
letter" -- exactly what the lowercase instruction requires. The extra `and
response.islower()` added an unrelated requirement (at least one *lowercase*
letter present), since Python's `str.islower()`/`str.isupper()` return False
when there is no cased character at all. A response like "42" (a valid
answer to a prompt combined with a lowercase-formatting instruction) or an
empty response therefore failed a constraint it never violated.
"""

import pytest

from deepeval.benchmarks.ifeval.ifeval import IFEvalInstructionVerifier


@pytest.mark.parametrize(
    "response",
    ["42", "", "   ", "3.14", "#1"],
)
def test_lowercase_constraint_passes_when_there_are_no_letters(response):
    assert IFEvalInstructionVerifier.verify_case_constraints(
        response, "change_case:english_lowercase", {}
    )


@pytest.mark.parametrize(
    "response",
    ["42", "", "   ", "3.14", "#1"],
)
def test_uppercase_constraint_passes_when_there_are_no_letters(response):
    assert IFEvalInstructionVerifier.verify_case_constraints(
        response, "change_case:english_uppercase", {}
    )


def test_lowercase_constraint_still_rejects_a_capital_letter():
    assert not IFEvalInstructionVerifier.verify_case_constraints(
        "Hello world", "change_case:english_lowercase", {}
    )


def test_lowercase_constraint_still_accepts_all_lowercase_text():
    assert IFEvalInstructionVerifier.verify_case_constraints(
        "hello world", "change_case:english_lowercase", {}
    )


def test_uppercase_constraint_still_rejects_a_lowercase_letter():
    assert not IFEvalInstructionVerifier.verify_case_constraints(
        "HELLo", "change_case:english_uppercase", {}
    )


def test_uppercase_constraint_still_accepts_all_uppercase_text():
    assert IFEvalInstructionVerifier.verify_case_constraints(
        "HELLO WORLD", "change_case:english_uppercase", {}
    )
