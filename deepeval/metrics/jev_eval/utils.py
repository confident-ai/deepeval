"""Shared core for ``JevEval`` and ``ConversationalJevEval``, and for the
whole-chain System One path built-in metrics take under ``system_one`` eval
mode (``deepeval.metrics.utils.system_one``).

The metric is one Jev ``decide()`` call followed by one equation:

    score = sum(w_i * v_i) / sum(w_i)   over applicable questions

where ``v_i`` is the value each primitive's answer maps onto in ``[0, 1]``.
Everything here is pure: no LLM, no network.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel

from deepeval.errors import DeepEvalError
from deepeval.metrics.utils.decision import _jsonable
from deepeval.metrics.utils.turns import convert_turn_to_dict
from deepeval.models import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (
    SystemOneAnswers,
    SystemOneQuestion,
)
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MultiTurnParams,
    SingleTurnParams,
)

from .questions import Choice, JevQuestion, Noul, QuestionOutcome, Score

# A Choice whose "not applicable" options collect at least this much mass is
# dropped from the metric for that test case.
NOT_APPLICABLE_THRESHOLD = 0.5

# Score when nothing applied: there was nothing applicable to fail. Matches the
# `empty_score` convention of the QAG helpers.
EMPTY_SCORE = 1.0


###############################################
# Model
###############################################


def initialize_jev_model(
    model: Optional[Union[str, DeepEvalBaseSystemOneModel]] = None,
) -> DeepEvalBaseSystemOneModel:
    """JevEval needs Jev in every mode; there is no LLM to fall back to."""
    if isinstance(model, DeepEvalBaseSystemOneModel):
        return model
    if model is not None and not isinstance(model, str):
        raise TypeError(
            f"Unsupported type for system_one_model: {type(model)}. Expected "
            "None, str, or DeepEvalBaseSystemOneModel."
        )
    from deepeval.models.system_one.typesafe_model import TypeSafeModel

    try:
        return TypeSafeModel(model=model)
    except DeepEvalError as e:
        raise DeepEvalError(
            f"JevEval needs Jev to score, but it is not usable: {e} "
            "Pass a configured `TypeSafeModel` as `system_one_model`, or "
            "install `typesafe-sdk` and set TYPESAFE_API_KEY."
        ) from e


###############################################
# Questions
###############################################


def validate_questions(questions: Optional[Sequence[Any]]) -> List[JevQuestion]:
    if questions is None or len(questions) == 0:
        raise ValueError(
            "JevEval needs at least one question (Noul, Score or Choice)."
        )
    for question in questions:
        if not isinstance(question, (Noul, Score, Choice)):
            raise TypeError(
                f"Questions must be Noul, Score or Choice; got "
                f"{type(question).__name__}."
            )
    return list(questions)


def question_key(index: int) -> str:
    return f"q_{index}"


def build_questions(
    questions: Sequence[JevQuestion],
) -> Dict[str, SystemOneQuestion]:
    return {
        question_key(i): question.to_system_one()
        for i, question in enumerate(questions)
    }


###############################################
# Answers -> values
###############################################


def value_from_answer(
    question: JevQuestion, answers: SystemOneAnswers, key: str
) -> QuestionOutcome:
    if isinstance(question, Noul):
        answer = answers.nouls.get(key)
        if answer is None:
            raise DeepEvalError(f"Jev returned no Noul answer for {key}.")
        p = float(answer.probability)
        return QuestionOutcome(
            question=question.text,
            type="noul",
            weight=question.weight,
            value=p,
            applicable=True,
            probabilities={"true": p, "false": 1.0 - p},
            confidence=answer.confidence,
        )

    if isinstance(question, Score):
        answer = answers.scores.get(key)
        if answer is None:
            raise DeepEvalError(f"Jev returned no Score answer for {key}.")
        top = len(question.levels) - 1
        probabilities = {
            question.levels[level]: float(p)
            for level, p in answer.probabilities.items()
            if 0 <= level <= top
        }
        return QuestionOutcome(
            question=question.text,
            type="score",
            weight=question.weight,
            value=min(max(float(answer.score) / top, 0.0), 1.0),
            applicable=True,
            probabilities=probabilities,
            confidence=answer.confidence,
        )

    if isinstance(question, Choice):
        answer = answers.choices.get(key)
        if answer is None:
            raise DeepEvalError(f"Jev returned no Choice answer for {key}.")
        probabilities = {
            name: float(answer.probabilities.get(name, 0.0))
            for name in question.options
        }
        na_mass = sum(
            probabilities[name] for name in question.not_applicable_options
        )
        applicable = na_mass < NOT_APPLICABLE_THRESHOLD
        value: Optional[float] = None
        if applicable:
            credits = question.applicable_options
            mass = sum(probabilities[name] for name in credits)
            if mass > 0:
                value = (
                    sum(
                        probabilities[name] * credit
                        for name, credit in credits.items()
                    )
                    / mass
                )
            else:
                applicable = False
        return QuestionOutcome(
            question=question.text,
            type="choice",
            weight=question.weight,
            value=value,
            applicable=applicable,
            probabilities=probabilities,
            confidence=answer.confidence,
        )

    raise TypeError(f"Unsupported question type: {type(question).__name__}")


def outcomes_from_answers(
    questions: Sequence[JevQuestion], answers: SystemOneAnswers
) -> List[QuestionOutcome]:
    return [
        value_from_answer(question, answers, question_key(i))
        for i, question in enumerate(questions)
    ]


def aggregate(outcomes: Sequence[QuestionOutcome]) -> float:
    applicable = [o for o in outcomes if o.applicable and o.value is not None]
    if not applicable:
        return EMPTY_SCORE
    total_weight = sum(o.weight for o in applicable)
    return sum(o.weight * o.value for o in applicable) / total_weight


###############################################
# Strict mode
###############################################
#
# `strict_mode` keeps deepeval's contract: the score is 1 for perfection and
# 0 otherwise. "Perfection" is every applicable question answered in its best
# possible way, decided from Jev's probabilities without an LLM.

STRICT_NOUL_THRESHOLD = 0.5


def _argmax(probabilities: Dict[str, float]) -> Optional[str]:
    if not probabilities:
        return None
    return max(probabilities.items(), key=lambda kv: kv[1])[0]


def passes_strictly(question: JevQuestion, outcome: QuestionOutcome) -> bool:
    if isinstance(question, Noul):
        return outcome.probabilities.get("true", 0.0) >= STRICT_NOUL_THRESHOLD
    if isinstance(question, Score):
        return _argmax(outcome.probabilities) == question.levels[-1]
    if isinstance(question, Choice):
        credits = question.applicable_options
        chosen = _argmax(
            {name: outcome.probabilities.get(name, 0.0) for name in credits}
        )
        return chosen is not None and credits[chosen] >= 1.0
    raise TypeError(f"Unsupported question type: {type(question).__name__}")


def mark_strict(
    questions: Sequence[JevQuestion], outcomes: Sequence[QuestionOutcome]
) -> List[QuestionOutcome]:
    """Record, per applicable question, whether it met the strict bar."""
    marked: List[QuestionOutcome] = []
    for question, outcome in zip(questions, outcomes):
        passed = (
            passes_strictly(question, outcome) if outcome.applicable else None
        )
        marked.append(outcome.model_copy(update={"passed": passed}))
    return marked


def aggregate_strict(outcomes: Sequence[QuestionOutcome]) -> float:
    applicable = [o for o in outcomes if o.applicable]
    if not applicable:
        return EMPTY_SCORE
    return 1.0 if all(o.passed for o in applicable) else 0.0


def min_confidence(outcomes: Sequence[QuestionOutcome]) -> Optional[float]:
    """The least decisive answer across the questions. Choice and Score carry
    the API's confidence; a Noul's is derived (``|2p - 1|``, see
    ``NoulAnswer.confidence``), so every outcome contributes."""
    values = [o.confidence for o in outcomes if o.confidence is not None]
    return min(values) if values else None


###############################################
# Test case -> state
###############################################


def _is_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, dict)) and len(value) == 0:
        return False
    return True


def _field(value: Any) -> Any:
    # Structured, not repr(): Jev can then read a `ToolCall`'s `output` or a
    # `RetrievedContextData`'s `context` as data.
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_field(v) for v in value]
    return value


def construct_single_turn_state(
    evaluation_params: Sequence[SingleTurnParams], test_case: LLMTestCase
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for param in evaluation_params:
        value = getattr(test_case, param.value, None)
        if _is_populated(value):
            fields[param.value] = _field(value)
    return _jsonable({"test_case": fields})


_TURN_LEVEL_PARAMS = (
    MultiTurnParams.ROLE,
    MultiTurnParams.CONTENT,
    MultiTurnParams.RETRIEVAL_CONTEXT,
    MultiTurnParams.TOOLS_CALLED,
    MultiTurnParams.MCP_TOOLS,
    MultiTurnParams.MCP_RESOURCES,
    MultiTurnParams.MCP_PROMPTS,
)


def construct_multi_turn_state(
    evaluation_params: Sequence[MultiTurnParams],
    test_case: ConversationalTestCase,
) -> Dict[str, Any]:
    turn_params = [p for p in evaluation_params if p in _TURN_LEVEL_PARAMS]
    turns = [
        {
            k: _field(v)
            for k, v in convert_turn_to_dict(turn, turn_params).items()
        }
        for turn in test_case.turns
    ]
    fields: Dict[str, Any] = {}
    for param in evaluation_params:
        if param in _TURN_LEVEL_PARAMS:
            continue
        value = getattr(test_case, param.value, None)
        if _is_populated(value):
            fields[param.value] = _field(value)
    state: Dict[str, Any] = {"turns": turns}
    if fields:
        state["test_case"] = fields
    return _jsonable(state)


###############################################
# Verbalised outcomes (for the deterministic reason)
###############################################
#
# Each outcome is turned into words so the reason reads as a judgement, with
# the numbers that produced it alongside.

_RUNNER_UP_MARGIN = 0.15


def _verbalise_noul(p: float) -> str:
    if p >= 0.85:
        return "clearly holds"
    if p >= 0.65:
        return "likely holds"
    if p >= 0.35:
        return "unclear"
    if p >= 0.15:
        return "likely fails"
    return "clearly fails"


def _verbalise_ranked(probabilities: Dict[str, float]) -> str:
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return "no answer"
    top, top_p = ranked[0]
    if len(ranked) > 1:
        runner, runner_p = ranked[1]
        if top_p - runner_p <= _RUNNER_UP_MARGIN:
            return f'"{top}", leaning "{runner}"'
    return f'"{top}"'


def verbalise_outcome(outcome: QuestionOutcome) -> str:
    if outcome.type == "noul":
        return _verbalise_noul(outcome.probabilities.get("true", 0.0))
    if not outcome.applicable:
        return "not applicable"
    return _verbalise_ranked(outcome.probabilities)


def format_outcomes_for_logs(outcomes: Sequence[QuestionOutcome]) -> str:
    lines: List[str] = []
    for i, outcome in enumerate(outcomes, start=1):
        value = "n/a" if outcome.value is None else f"{outcome.value:.3f}"
        line = (
            f"{i}. [{outcome.type}, weight={outcome.weight:g}] "
            f"{outcome.question}\n   value={value} "
            f"applicable={outcome.applicable} probabilities="
            f"{ {k: round(v, 3) for k, v in outcome.probabilities.items()} }"
        )
        if outcome.confidence is not None:
            line += f" confidence={outcome.confidence:.3f}"
        if outcome.passed is not None:
            line += f" strict_pass={outcome.passed}"
        lines.append(line)
    return "\n".join(lines)
