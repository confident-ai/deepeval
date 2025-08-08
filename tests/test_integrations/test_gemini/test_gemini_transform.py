from deepeval.models.llms.gemini_model import GeminiModel


class _Part:
    def __init__(self, text: str):
        self.text = text


class _Content:
    def __init__(self, parts):
        self.parts = parts


class _ChosenCandidate:
    def __init__(self, token: str):
        self.token = token


class _TopCandidateAlt:
    def __init__(self, token: str, log_probability: float):
        self.token = token
        self.log_probability = log_probability


class _TopCandidate:
    def __init__(self, candidates):
        self.candidates = candidates


class _LogprobsResult:
    def __init__(self, chosen_candidates, top_candidates):
        self.chosen_candidates = chosen_candidates
        self.top_candidates = top_candidates


class _Candidate:
    def __init__(self, content=None, logprobs_result=None):
        self.content = content
        self.logprobs_result = logprobs_result


class _Raw:
    def __init__(self, text=None, candidates=None):
        self.text = text
        self.candidates = candidates or []


def test_transform_uses_text_field_when_present():
    raw = _Raw(text="Hello from text field", candidates=[])

    wrapped = GeminiModel.transform_gemini_to_openai_like(raw)

    assert hasattr(wrapped, "choices")
    assert wrapped.choices[0].message.content == "Hello from text field"
    # No logprobs provided
    assert hasattr(wrapped.choices[0], "logprobs")
    assert isinstance(wrapped.choices[0].logprobs.content, list)
    assert len(wrapped.choices[0].logprobs.content) == 0


def test_transform_builds_text_from_parts_when_no_text():
    parts = [_Part("Hello "), _Part("from "), _Part("parts")]
    content = _Content(parts)
    candidate = _Candidate(content=content)
    raw = _Raw(text=None, candidates=[candidate])

    wrapped = GeminiModel.transform_gemini_to_openai_like(raw)

    assert wrapped.choices[0].message.content == "Hello from parts"


def test_transform_maps_logprobs_structure():
    # chosen tokens A, B with alternatives per position
    chosen = [_ChosenCandidate("A"), _ChosenCandidate("B")]
    top0 = _TopCandidate(
        [
            _TopCandidateAlt("A", -0.1),
            _TopCandidateAlt("X", -2.5),
        ]
    )
    top1 = _TopCandidate(
        [
            _TopCandidateAlt("B", -0.2),
            _TopCandidateAlt("Y", -3.0),
        ]
    )
    logprobs_result = _LogprobsResult(
        chosen_candidates=chosen, top_candidates=[top0, top1]
    )

    candidate = _Candidate(
        content=_Content([_Part("AB")]), logprobs_result=logprobs_result
    )
    raw = _Raw(candidates=[candidate])

    wrapped = GeminiModel.transform_gemini_to_openai_like(raw)

    # Validate content
    assert wrapped.choices[0].message.content == "AB"

    # Validate logprobs shape and values
    token_entries = wrapped.choices[0].logprobs.content
    assert isinstance(token_entries, list)
    assert len(token_entries) == 2

    # First token position
    assert getattr(token_entries[0], "token") == "A"
    top_logprobs0 = getattr(token_entries[0], "top_logprobs")
    assert isinstance(top_logprobs0, list)
    # Ensure both alternatives are present and mapped
    tokens0 = {getattr(t, "token"): getattr(t, "logprob") for t in top_logprobs0}
    assert tokens0 == {"A": -0.1, "X": -2.5}

    # Second token position
    assert getattr(token_entries[1], "token") == "B"
    top_logprobs1 = getattr(token_entries[1], "top_logprobs")
    tokens1 = {getattr(t, "token"): getattr(t, "logprob") for t in top_logprobs1}
    assert tokens1 == {"B": -0.2, "Y": -3.0}


##############################################
# Test Everything
##############################################

if __name__ == "__main__":
    test_transform_uses_text_field_when_present()
    test_transform_builds_text_from_parts_when_no_text()
    test_transform_maps_logprobs_structure()
