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


def _rotations(names: List[str]) -> List[List[str]]:
    """Cyclic rotations of ``names``: every contestant appears once in each
    position. Two contestants give A-B and B-A."""
    return [names[i:] + names[:i] for i in range(len(names))]


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

    The judge is asked once per ordering (cyclic rotations, so every contestant
    is shown once in each position). Contestant names are masked the same way
    ``ArenaGEval`` masks them, and the masking is kept fixed across orderings so
    position is the only thing that changes.

    ``score`` is the fraction of orderings that agree with the majority winner.
    ``first_position_win_rate`` is how often the contestant shown first won.
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
    ):
        validate_criteria_and_evaluation_steps(criteria, evaluation_steps)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
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
        self.winners_by_order: List[Dict[str, object]] = []
        self.success: Optional[bool] = None

    def measure(
        self,
        test_case: ArenaTestCase,
        _show_indicator: bool = True,
        *args,
        **kwargs,
    ) -> str:
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
    ) -> str:
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
        return formatted, dummy_to_real, _rotations(names)

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
        # On a tie, prefer the contestant that won earliest in the run order.
        self.winner = next(w for w in winners if counts[w] == top)
        self.score = top / len(winners)
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

    def _build_reason(self, orders: List[List[str]], winners: List[str]) -> str:
        agree = sum(1 for w in winners if w == self.winner)
        total = len(winners)
        if agree == total:
            return (
                f"The judge picked {self.winner} in all {total} orderings, so the "
                "result does not depend on the order the contestants were shown in."
            )
        flips = "; ".join(
            f"{' > '.join(order)} -> {winner}"
            for order, winner in zip(orders, winners)
            if winner != self.winner
        )
        return (
            f"The judge picked {self.winner} in {agree} of {total} orderings. "
            f"Other orderings gave a different winner ({flips}). The contestant "
            f"shown first won {self.first_position_win_rate:.0%} of the time."
        )
