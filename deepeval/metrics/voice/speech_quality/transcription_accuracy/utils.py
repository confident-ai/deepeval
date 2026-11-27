import re
from typing import List, Tuple

# Differences the LLM judge is told are harmless are normalised away first,
# so both modes agree on what counts as a misheard word.
FILLER_WORDS = {"um", "uh", "er", "ah", "hmm", "mm"}
# One or two words are too few to count errors over: a single slip, or a
# transcriber guessing at a mumble, reads as half the line wrong.
WORD_ERROR_MIN_WORDS = 3
NUMBER_UNITS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven "
        "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
        "nineteen".split()
    )
}
NUMBER_TENS = {
    word: (index + 2) * 10
    for index, word in enumerate(
        "twenty thirty forty fifty sixty seventy eighty ninety".split()
    )
}
NUMBER_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}


def _number_words_to_digits(words: List[str]) -> List[str]:
    converted: List[str] = []
    total = current = 0
    last = None

    def flush():
        nonlocal total, current, last
        if last is not None:
            converted.append(str(total + current))
        total = current = 0
        last = None

    for index, word in enumerate(words):
        following = words[index + 1] if index + 1 < len(words) else None
        if (
            word == "and"
            and last == "scale"
            and (following in NUMBER_UNITS or following in NUMBER_TENS)
        ):
            continue
        if word in NUMBER_UNITS:
            if last in ("unit", "teen"):
                flush()
            current += NUMBER_UNITS[word]
            last = "unit" if NUMBER_UNITS[word] < 10 else "teen"
        elif word in NUMBER_TENS:
            if last in ("unit", "teen", "ten"):
                flush()
            current += NUMBER_TENS[word]
            last = "ten"
        elif word in NUMBER_SCALES and last is not None:
            if word == "hundred":
                current *= 100
            else:
                total += current * NUMBER_SCALES[word]
                current = 0
            last = "scale"
        else:
            flush()
            converted.append(word)
    flush()
    # Digits read one at a time ("five five five") match the number written
    # whole ("555").
    merged: List[str] = []
    for word in converted:
        if word.isdigit() and merged and merged[-1].isdigit():
            merged[-1] += word
        else:
            merged.append(word)
    return merged


def _without_repeated_phrases(words: List[str]) -> List[str]:
    """Drop a phrase said twice in a row, which some speech-to-text engines
    emit for one utterance."""
    words = list(words)
    length = len(words) // 2
    while length >= 2:
        start = 0
        while start + 2 * length <= len(words):
            if (
                words[start : start + length]
                == words[start + length : start + 2 * length]
            ):
                del words[start + length : start + 2 * length]
            else:
                start += 1
        length -= 1
    return words


def _with_spellings_joined(words: List[str]) -> List[str]:
    """Letters and digits spelled out one by one ("h m a 7 2") are one word."""
    joined: List[str] = []
    run: List[str] = []

    def flush():
        if sum(len(word) == 1 and word.isalpha() for word in run) >= 3:
            joined.append("".join(run))
        else:
            joined.extend(run)
        run.clear()

    for word in words:
        if (len(word) == 1 and word.isalpha()) or word.isdigit():
            run.append(word)
        else:
            flush()
            joined.append(word)
    flush()
    return joined


def words_for_error_rate(text: str) -> List[str]:
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    # "U.S." is the word "US".
    text = re.sub(
        r"\b(?:[a-z]\.){2,}", lambda match: match.group().replace(".", ""), text
    )
    # An address written out reads as spoken: "a@b.com" is "a at b dot com".
    text = text.replace("@", " at ")
    text = re.sub(r"(?<=[a-z0-9])\.(?=[a-z])", " dot ", text)
    text = re.sub(r"[^\w']+", " ", text)
    words = [
        word.strip("'")
        for word in text.split()
        if word.strip("'") and word.strip("'") not in FILLER_WORDS
    ]
    return _without_repeated_phrases(
        _with_spellings_joined(_number_words_to_digits(words))
    )


def word_errors(spoken: str, transcribed: str) -> Tuple[int, int]:
    """Words swapped, missing or added in the transcript, and the number of
    words spoken."""
    reference = words_for_error_rate(spoken)
    hypothesis = words_for_error_rate(transcribed)
    # "home page" and "homepage" are the same words written two ways.
    if "".join(reference) == "".join(hypothesis):
        return 0, len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row, spoken_word in enumerate(reference, start=1):
        current = [row]
        for column, heard_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (spoken_word != heard_word),
                )
            )
        previous = current
    return previous[-1], len(reference)


def word_errors_within(transcribed: str, nearby: str) -> Tuple[int, int]:
    """The fewest word errors of the transcript against any stretch of the
    speech heard nearby, and the number of words in the transcript."""
    hypothesis = words_for_error_rate(transcribed)
    reference = words_for_error_rate(nearby)
    # The transcript may start anywhere in the speech and stop anywhere.
    previous = [0] * (len(reference) + 1)
    for row, written_word in enumerate(hypothesis, start=1):
        current = [row]
        for column, heard_word in enumerate(reference, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (written_word != heard_word),
                )
            )
        previous = current
    return min(previous), len(hypothesis)
