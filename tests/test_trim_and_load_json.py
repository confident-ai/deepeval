"""
Tests for the proposed trim_and_load_json fix (deepeval #2280 / #2299).
Run with pytest, or directly: python test_trim_and_load_json.py
"""

from deepeval.metrics.utils.generation import (
    EMPTY_OUTPUT_ERROR,
    INVALID_JSON_ERROR,
    TRUNCATED_OUTPUT_ERROR,
    trimAndLoadJson as trim_and_load_json,
)


class FakeMetric:
    error = None


def expect_error(raw, expected_message):
    metric = FakeMetric()
    try:
        trim_and_load_json(raw, metric)
    except ValueError as e:
        assert str(e) == expected_message, str(e)
        assert metric.error == expected_message
        return
    raise AssertionError(f"expected ValueError for {raw!r}")


def test_valid_json_unchanged():
    assert trim_and_load_json('{"score": 5, "reason": "ok"}') == {
        "score": 5,
        "reason": "ok",
    }


def test_surrounding_text_and_fences_trimmed():
    raw = 'Sure! ```json\n{"verdict": "yes", "reason": "fine"}\n``` Hope that helps.'
    assert trim_and_load_json(raw) == {"verdict": "yes", "reason": "fine"}


def test_trailing_comma_tolerated():
    assert trim_and_load_json('{"a": [1, 2,], "b": 3,}') == {
        "a": [1, 2],
        "b": 3,
    }


def test_invalid_escape_repaired():
    # the exact error class from #2280: "JSONDecodeError: Invalid \\escape"
    raw = r'{"reason": "The fee is \$500 per \(item\)", "score": 4}'
    assert trim_and_load_json(raw) == {
        "reason": r"The fee is \$500 per \(item\)",
        "score": 4,
    }


def test_windows_path_escape_repaired():
    raw = r'{"reason": "see C:\data\scores.txt", "score": 3}'
    assert trim_and_load_json(raw)["reason"] == r"see C:\data\scores.txt"


def test_legal_escapes_follow_json_spec():
    # Known limitation, by design: "\f" is a LEGAL JSON escape (form feed), so
    # in "C:\data\file.txt" only "\d" is repaired. Guessing that "\f" was meant
    # literally would risk corrupting valid output, so the spec wins.
    raw = r'{"reason": "C:\data\file.txt"}'
    assert trim_and_load_json(raw)["reason"] == "C:\\data\file.txt"


def test_valid_escapes_next_to_invalid_ones_preserved():
    # a legitimate escaped backslash (\\), a newline escape (\n) and a stray \$
    raw = r'{"reason": "path \\ ok\nnext line costs \$5"}'
    assert (
        trim_and_load_json(raw)["reason"] == "path \\ ok\nnext line costs \\$5"
    )


def test_empty_output_gives_specific_error():
    # reasoning model spent its whole budget thinking: content == ""
    expect_error("", EMPTY_OUTPUT_ERROR)
    expect_error("   \n ", EMPTY_OUTPUT_ERROR)
    expect_error(None, EMPTY_OUTPUT_ERROR)


def test_truncated_output_gives_specific_error():
    expect_error(
        '{"verdict": "yes", "reason": "The answer is supported by',
        TRUNCATED_OUTPUT_ERROR,
    )


def test_genuinely_invalid_json_keeps_original_message():
    expect_error("I cannot evaluate this.", INVALID_JSON_ERROR)
    expect_error('{"verdict": yes}', INVALID_JSON_ERROR)


def test_works_without_metric():
    try:
        trim_and_load_json("")
    except ValueError as e:
        assert str(e) == EMPTY_OUTPUT_ERROR
    else:
        raise AssertionError


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
