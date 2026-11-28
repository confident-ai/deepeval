import asyncio
import json
from collections import Counter
from typing import Dict, List, Optional, Type, Union

from deepeval.metrics import BaseArenaMetric
from deepeval.metrics.arena_g_eval.arena_g_eval import ArenaGEvalTemplate
from deepeval.metrics.arena_g_eval.schema import Steps, Winner
from deepeval.metrics.arena_g_eval.utils import (
    FormattedArenaTestCase,
    format_arena_test_case,
)
from deepeval.metrics.g_eval.utils import (
    construct_g_eval_params_string,
    number_evaluation_steps,
    validate_criteria_and_evaluation_steps,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    check_arena_test_case_params,
    construct_verbose_logs,
    generate_with_schema_and_extract,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ArenaTestCase, SingleTurnParams
from deepeval.utils import get_or_create_event_loop, prettify_list


def _orderings(names: List[str]) -> List[List[str]]:
    """Orderings that balance both position and pairwise order.

    Cyclic rotations put every contestant once in each position. For three or
    more contestants they do not balance pairs (with A, B, C, A comes before B
    in two rotations out of three), so the rotations of the reversed list are
    added: reversing an ordering flips the relative order of every pair. Two
    contestants give A-B and B-A; three give all six permutations.
    """
    rotations = [names[i:] + names[:i] for i in range(len(names))]
    if len(names) <= 2:
        return rotations
    reversed_names = names[::-1]
    return rotations + [
        reversed_names[i:] + reversed_names[:i] for i in range(len(names))
    ]


def _render_contestants(
    formatted: FormattedArenaTestCase,
    order: List[str],
    real_to_dummy: Dict[str, str],
) -> str:
    """Same JSON shape as ``FormattedArenaTestCase.__repr__``, but with the
    contestants in a fixed order instead of a random one."""
    data = {}
    if formatted.input is not None:
        data["input"] = formatted.input
    if formatted.expected_output is not None:
        data["expected_output"] = formatted.expected_output
    data["arena_test_cases"] = {
        real_to_dummy[name]: repr(formatted.contestants[name]) for name in order
    }
    return json.dumps(data, indent=2)


class PositionConsistencyMetric(BaseArenaMetric):
    """Checks whether an arena judge picks the same winner when only the order
    of the contestants changes.

    The judge is asked once per ordering, or ``repeats`` times per ordering.
    The orderings are the cyclic rotations of the contestants plus, for three or
    more, the rotations of the reversed list, so every contestant appears
    equally often in each position and every pair appears equally often in each
    relative order. Contestant names are masked the same way ``ArenaGEval``
    masks them, the masking is kept fixed across orderings, and contestants are
    rendered in the chosen order (not shuffled), so position is the only thing
    that changes.

    ``score`` is the share of judge calls that agree with the most common
    winner. ``winner`` is that contestant, or ``None`` when several are tied
    for most wins. ``first_position_win_rate`` is how often the contestant
    shown first won; with no position bias it is about ``1 / n`` for ``n``
    contestants. With ``repeats`` > 1, ``repeat_consistency`` is the average
    share of repeats of the same ordering that agree with each other: low
    values point to judge sampling noise rather than position bias.
    """

    def __init__(
        self,
        name: str,
        evaluation_params: List[SingleTurnParams],
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 1.0,
        async_mode: bool = True,
        verbose_mode: bool = False,
        evaluation_template: Type[ArenaGEvalTemplate] = ArenaGEvalTemplate,
        repeats: int = 1,
    ):
        validate_criteria_and_evaluation_steps(criteria, evaluation_steps)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        if repeats < 1:
            raise ValueError("repeats must be at least 1.")
        self.repeats = repeats
        self.name = name
        self.evaluation_params = evaluation_params
        self.criteria = criteria
        self.evaluation_steps = (
            evaluation_steps
            if evaluation_steps and len(evaluation_steps) > 0
            else None
        )
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.threshold = threshold
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template

        self.score: Optional[float] = None
        self.winner: Optional[str] = None
        self.first_position_win_rate: Optional[float] = None
        self.repeat_consistency: Optional[float] = None
        self.winners_by_order: List[Dict[str, object]] = []
        self.success: Optional[bool] = None

    def measure(
        self,
        test_case: ArenaTestCase,
        _show_indicator: bool = True,
        *args,
        **kwargs,
    ) -> Optional[str]:
        self._prepare(test_case)
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                multimodal = test_case.multimodal
                self.evaluation_steps = self._generate_evaluation_steps(
                    multimodal
                )
                formatted, dummy_to_real, orders = self._setup(test_case)
                winners = [
                    self._judge(formatted, order, dummy_to_real, multimodal)
                    for order in orders
                ]
                self._finalize(orders, winners)
            return self.winner

    async def a_measure(
        self,
        test_case: ArenaTestCase,
        _show_indicator: bool = True,
        *args,
        **kwargs,
    ) -> Optional[str]:
        self._prepare(test_case)
        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            multimodal = test_case.multimodal
            self.evaluation_steps = await self._a_generate_evaluation_steps(
                multimodal
            )
            formatted, dummy_to_real, orders = self._setup(test_case)
            winners = await asyncio.gather(
                *[
                    self._a_judge(formatted, order, dummy_to_real, multimodal)
                    for order in orders
                ]
            )
            self._finalize(orders, list(winners))
            return self.winner

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return bool(self.success)

    @property
    def __name__(self):
        return f"{self.name} (Position Consistency)"

    # ------------------------------------------------------------------ setup

    def _prepare(self, test_case: ArenaTestCase) -> None:
        check_arena_test_case_params(
            test_case,
            self.evaluation_params,
            self,
            self.model,
            test_case.multimodal,
        )
        if len(test_case.contestants) < 2:
            raise ValueError(
                f"'{self.__name__}' needs at least 2 contestants to compare orderings."
            )
        self.error = None
        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None

    def _setup(self, test_case: ArenaTestCase):
        formatted, dummy_to_real = format_arena_test_case(
            self.evaluation_params, test_case
        )
        names = [contestant.name for contestant in test_case.contestants]
        orders = [
            order for order in _orderings(names) for _ in range(self.repeats)
        ]
        return formatted, dummy_to_real, orders

    def _winner_prompt(
        self,
        formatted: FormattedArenaTestCase,
        order: List[str],
        dummy_to_real: Dict[str, str],
        multimodal: bool,
    ):
        real_to_dummy = {real: dummy for dummy, real in dummy_to_real.items()}
        return self._get_prompt(
            "generate_arena_winner",
            evaluation_steps=number_evaluation_steps(self.evaluation_steps),
            test_case_contents=_render_contestants(
                formatted, order, real_to_dummy
            ),
            parameters=construct_g_eval_params_string(self.evaluation_params),
            multimodal=multimodal,
        )

    @staticmethod
    def _unmask(winner: str, dummy_to_real: Dict[str, str]) -> str:
        name = winner.strip().strip("$").strip()
        if name not in dummy_to_real:
            raise ValueError(
                f"Judge returned '{winner}', which is not one of the contestants "
                f"({', '.join(dummy_to_real)})."
            )
        return dummy_to_real[name]

    # -------------------------------------------------------------- judging

    def _judge(self, formatted, order, dummy_to_real, multimodal) -> str:
        prompt = self._winner_prompt(
            formatted, order, dummy_to_real, multimodal
        )
        winner = generate_with_schema_and_extract(
            self,
            prompt,
            Winner,
            extract_schema=lambda s: s.winner,
            extract_json=lambda data: data["winner"],
        )
        return self._unmask(winner, dummy_to_real)

    async def _a_judge(
        self, formatted, order, dummy_to_real, multimodal
    ) -> str:
        prompt = self._winner_prompt(
            formatted, order, dummy_to_real, multimodal
        )
        winner = await a_generate_with_schema_and_extract(
            self,
            prompt,
            Winner,
            extract_schema=lambda s: s.winner,
            extract_json=lambda data: data["winner"],
        )
        return self._unmask(winner, dummy_to_real)

    def _generate_evaluation_steps(self, multimodal: bool) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps
        prompt = self._get_prompt(
            "generate_evaluation_steps",
            criteria=self.criteria,
            parameters=construct_g_eval_params_string(self.evaluation_params),
            multimodal=multimodal,
        )
        return generate_with_schema_and_extract(
            self,
            prompt,
            Steps,
            extract_schema=lambda s: s.steps,
            extract_json=lambda data: data["steps"],
        )

    async def _a_generate_evaluation_steps(self, multimodal: bool) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps
        prompt = self._get_prompt(
            "generate_evaluation_steps",
            criteria=self.criteria,
            parameters=construct_g_eval_params_string(self.evaluation_params),
            multimodal=multimodal,
        )
        return await a_generate_with_schema_and_extract(
            self,
            prompt,
            Steps,
            extract_schema=lambda s: s.steps,
            extract_json=lambda data: data["steps"],
        )

    # -------------------------------------------------------------- scoring

    def _finalize(self, orders: List[List[str]], winners: List[str]) -> None:
        counts = Counter(winners)
        top = max(counts.values())
        leaders = [name for name, count in counts.items() if count == top]
        # No majority (e.g. a 1-1 split between two contestants): no winner.
        self.winner = leaders[0] if len(leaders) == 1 else None
        self.score = top / len(winners)
        self.repeat_consistency = (
            self._repeat_consistency(orders, winners)
            if self.repeats > 1
            else None
        )
        self.first_position_win_rate = sum(
            1 for order, winner in zip(orders, winners) if winner == order[0]
        ) / len(winners)
        self.winners_by_order = [
            {"order": order, "winner": winner}
            for order, winner in zip(orders, winners)
        ]
        self.success = self.score >= self.threshold
        self.reason = self._build_reason(orders, winners)
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Criteria:\n{self.criteria}",
                f"Evaluation Steps:\n{prettify_list(self.evaluation_steps)}",
                "Winners by order:\n"
                + "\n".join(
                    f"{' > '.join(order)} -> {winner}"
                    for order, winner in zip(orders, winners)
                ),
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    def _repeat_consistency(
        self, orders: List[List[str]], winners: List[str]
    ) -> float:
        groups: Dict[tuple, List[str]] = {}
        for order, winner in zip(orders, winners):
            groups.setdefault(tuple(order), []).append(winner)
        shares = [
            max(Counter(group).values()) / len(group)
            for group in groups.values()
        ]
        return sum(shares) / len(shares)

    def _build_reason(self, orders: List[List[str]], winners: List[str]) -> str:
        total = len(winners)
        noise = (
            f" Repeats of the same ordering agreed {self.repeat_consistency:.0%}"
            " of the time."
            if self.repeat_consistency is not None
            else ""
        )
        if self.winner is None:
            counts = Counter(winners)
            tally = ", ".join(
                f"{name} {count}" for name, count in counts.items()
            )
            return (
                f"No contestant won a majority of the {total} judge calls ({tally}), "
                "so the outcome depends on the order the contestants were shown in. "
                f"The contestant shown first won {self.first_position_win_rate:.0%} "
                f"of the time.{noise}"
            )
        agree = sum(1 for w in winners if w == self.winner)
        if agree == total:
            return (
                f"The judge picked {self.winner} in all {total} judge calls, so the "
                "result does not depend on the order the contestants were shown in."
                f"{noise}"
            )
        flips = "; ".join(
            f"{' > '.join(order)} -> {winner}"
            for order, winner in zip(orders, winners)
            if winner != self.winner
        )
        return (
            f"The judge picked {self.winner} in {agree} of {total} judge calls. "
            f"Other calls gave a different winner ({flips}). The contestant "
            f"shown first won {self.first_position_win_rate:.0%} of the time.{noise}"
        )
