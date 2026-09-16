import asyncio
import inspect
from typing import Any, Dict, List, Optional, Tuple

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import BaseMetric
from deepeval.metrics.community.judge_self_consistency.schema import (
    JudgeSelfConsistencyResult,
    Replicate,
)
from deepeval.metrics.community.judge_self_consistency.stats import (
    bootstrap_mean_interval,
    decision_flip_rate,
    stability,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import construct_verbose_logs, copy_metrics
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.utils import get_or_create_event_loop


class JudgeSelfConsistencyMetric(BaseMetric):
    """How much a judge agrees with itself.

    Wraps another metric and runs it ``replicates`` times against the same
    ``LLMTestCase``, then reports how far apart the repeats landed. Every
    other metric in ``deepeval`` measures the application under test; this one
    measures the judge you are measuring it with.

    Two numbers come out of a run:

    - **Stability** (the metric's ``score``) — ``1 - normalized spread`` of the
      replicate scores. ``1.0`` when every repeat produced the same score,
      falling towards ``0.0`` as they spread across the range.
    - **Decision flip rate** (``self.decision_flip_rate``) — the fraction of
      replicate *pairs* that landed on opposite sides of the wrapped judge's
      threshold. This is usually the more alarming of the two: scores that vary
      only slightly still flip a verdict when they vary across the threshold,
      and a verdict flip is what actually changes a test result.

    Because a judge that disagrees with itself is a fact about the eval harness
    rather than about the application under test, a low reading here should not
    by itself fail a test suite. With the default ``auto_flaky=True``, the
    metric therefore sets its own ``flaky`` attribute when the replicates
    disagree, which keeps the reading in the report — in ``deepeval``'s flaky
    pass/fail sub-counts — without letting it gate.

    The wrapped judge needs some source of variation for this to be
    informative; a judge pinned to ``temperature=0`` may well return the same
    score every time, which the metric will faithfully report as perfect
    stability.
    """

    def __init__(
        self,
        metric: BaseMetric,
        replicates: int = 5,
        threshold: Optional[float] = 0.9,
        auto_flaky: bool = True,
        score_range: Tuple[float, float] = (0.0, 1.0),
        max_concurrent: Optional[int] = None,
        bootstrap_resamples: int = 2000,
        confidence: float = 0.95,
        random_seed: Optional[int] = 42,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
        _declared_flaky: Optional[bool] = None,
    ):
        if not isinstance(metric, BaseMetric):
            raise TypeError(
                f"'metric' must be a BaseMetric, got "
                f"{type(metric).__name__}. Conversational and arena metrics "
                "are not supported: this metric evaluates LLMTestCases."
            )
        if replicates < 2:
            raise ValueError(
                "'replicates' must be at least 2: a single run of the judge "
                "says nothing about whether it agrees with itself."
            )
        if bootstrap_resamples < 1:
            raise ValueError(
                f"'bootstrap_resamples' must be at least 1, got "
                f"{bootstrap_resamples}."
            )
        if not 0.0 < confidence < 1.0:
            raise ValueError(
                f"'confidence' must be strictly between 0 and 1, got "
                f"{confidence}. Pass 0.95, not 95."
            )
        if score_range[0] >= score_range[1]:
            raise ValueError(
                f"'score_range' must be (low, high) with low < high, got "
                f"{score_range}."
            )
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(
                f"'max_concurrent' must be at least 1, got {max_concurrent}."
            )

        self.metric = metric
        self.replicates = replicates
        self.threshold = 1 if strict_mode else threshold
        self.auto_flaky = auto_flaky
        self.score_range = score_range
        self.max_concurrent = max_concurrent
        self.bootstrap_resamples = bootstrap_resamples
        self.confidence = confidence
        self.random_seed = random_seed
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

        # `auto_flaky` raises `flaky` during a run, so the value sitting on the
        # instance afterwards is not necessarily what the caller asked for.
        # `copy_metrics` rebuilds a metric from its current attributes, so the
        # caller's own choice travels separately — otherwise one flaky run
        # would latch into every copy made from that instance thereafter.
        self._declared_flaky = (
            flaky if _declared_flaky is None else _declared_flaky
        )
        self.flaky = self._declared_flaky

        # The wrapped judge decides which test case fields are required, and
        # does its own validation on every replicate. `BaseMetric` leaves
        # `_required_params` as a bare type annotation, so only a real list
        # from the judge itself is worth adopting.
        judge_params = getattr(metric, "_required_params", None)
        self._required_params: List[SingleTurnParams] = (
            judge_params if isinstance(judge_params, list) else []
        )
        self.evaluation_model = getattr(metric, "evaluation_model", None)
        self.using_native_model = getattr(metric, "using_native_model", None)

        self.result: Optional[JudgeSelfConsistencyResult] = None
        self.decision_flip_rate: Optional[float] = None
        self.replicate_scores: List[float] = []

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        self._reset()
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                judges = self._build_judges()
                replicates = [
                    self._run_replicate(judge, test_case, _in_component)
                    for judge in judges
                ]
                self._finalize(judges, replicates)

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        self._reset()
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            judges = self._build_judges()
            semaphore = (
                asyncio.Semaphore(self.max_concurrent)
                if self.max_concurrent
                else None
            )

            async def run(judge: BaseMetric) -> Replicate:
                if semaphore is None:
                    return await self._a_run_replicate(
                        judge, test_case, _in_component
                    )
                async with semaphore:
                    return await self._a_run_replicate(
                        judge, test_case, _in_component
                    )

            settled = await asyncio.gather(
                *(run(j) for j in judges), return_exceptions=True
            )
            # Everything is gathered before anything is re-raised, so a
            # failing replicate never leaves its siblings dangling.
            for outcome in settled:
                if isinstance(outcome, MissingTestCaseParamsError):
                    # The harness handles this one: it is how a test case gets
                    # skipped rather than failed.
                    raise outcome
                if isinstance(outcome, BaseException) and not isinstance(
                    outcome, Exception
                ):
                    # Cancellation and the like are not the judge's doing.
                    raise outcome
            replicates = [
                (
                    outcome
                    if isinstance(outcome, Replicate)
                    else Replicate(error=str(outcome))
                )
                for outcome in settled
            ]
            self._finalize(judges, replicates)
            return self.score

    def _reset(self) -> None:
        """Clear the previous run so a reused metric instance starts clean."""
        self.error = None
        self.score = None
        self.reason = None
        self.success = None
        self.skipped = False
        self.flaky = self._declared_flaky
        self.score_breakdown = None
        self.verbose_logs = None
        self.result = None
        self.decision_flip_rate = None
        self.replicate_scores = []
        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None

    def _build_judges(self) -> List[BaseMetric]:
        """One copy of the wrapped judge per replicate.

        Metrics carry their per-run results as instance state, so the
        replicates cannot share an instance. ``copy_metrics`` is the same
        helper ``evaluate()`` uses to isolate metrics across test cases: it
        rebuilds each copy from the constructor arguments it can recover, so
        the copies get their own result state while sharing the judge's
        configuration (and its model client) by reference.
        """
        try:
            judges = copy_metrics([self.metric] * self.replicates)
        except TypeError as error:
            raise TypeError(
                f"Could not make replicate copies of "
                f"{getattr(self.metric, '__name__', type(self.metric).__name__)}: "
                f"{error}. This metric copies the judge with deepeval's "
                "'copy_metrics', which rebuilds a metric by passing its "
                "attributes back into its constructor, so every required "
                "constructor argument must be stored on the instance under "
                "that same name."
            ) from error

        for judge in judges:
            # The wrapper owns the console output for the whole run.
            judge.verbose_mode = False
        return judges

    @staticmethod
    def _supported_kwargs(func, **kwargs) -> Dict[str, Any]:
        """Drop private kwargs a custom metric's signature does not accept."""
        try:
            parameters = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return {}
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return kwargs
        return {
            name: value for name, value in kwargs.items() if name in parameters
        }

    def _run_replicate(
        self, judge: BaseMetric, test_case: LLMTestCase, _in_component: bool
    ) -> Replicate:
        kwargs = self._supported_kwargs(
            judge.measure, _show_indicator=False, _in_component=_in_component
        )
        try:
            judge.measure(test_case, **kwargs)
        except MissingTestCaseParamsError:
            # The harness handles this one: it is how a test case gets skipped
            # rather than failed, so it must not be folded into a replicate.
            raise
        except Exception as error:
            return Replicate(error=str(error))
        return self._read_replicate(judge)

    async def _a_run_replicate(
        self, judge: BaseMetric, test_case: LLMTestCase, _in_component: bool
    ) -> Replicate:
        if type(judge).a_measure is BaseMetric.a_measure:
            # The judge never implemented async; go straight to the sync path
            # rather than paying for a partial async run before it raises.
            return await asyncio.to_thread(
                self._run_replicate, judge, test_case, _in_component
            )

        kwargs = self._supported_kwargs(
            judge.a_measure, _show_indicator=False, _in_component=_in_component
        )
        try:
            await judge.a_measure(test_case, **kwargs)
        except MissingTestCaseParamsError:
            raise
        except Exception as error:
            return Replicate(error=str(error))
        return self._read_replicate(judge)

    @staticmethod
    def _read_replicate(judge: BaseMetric) -> Replicate:
        if judge.error is not None:
            return Replicate(error=judge.error)
        return Replicate(
            score=judge.score,
            success=judge.is_successful(),
            reason=judge.reason,
        )

    def _fail(self, message: str, replicates: List[Replicate]) -> None:
        self.error = message
        self.result = JudgeSelfConsistencyResult(
            stability=0.0,
            replicates=replicates,
            errored_replicates=sum(
                1 for r in replicates if r.error is not None
            ),
        )
        self.score = None
        self.reason = message
        self.success = self.is_successful()

    def _finalize(
        self, judges: List[BaseMetric], replicates: List[Replicate]
    ) -> None:
        scored = [
            r for r in replicates if r.error is None and r.score is not None
        ]
        # Only replicates that actually ran carry cost; reading an errored
        # judge's unset cost would null the whole accumulator.
        for judge, replicate in zip(judges, replicates):
            if replicate.error is None:
                self._accrue_cost(judge.evaluation_cost)
                self._accrue_tokens(judge.input_tokens, judge.output_tokens)

        if len(scored) < 2:
            first_error = next(
                (r.error for r in replicates if r.error is not None),
                "the judge returned no score.",
            )
            self._fail(
                f"Only {len(scored)} of {self.replicates} replicates produced "
                f"a score, which is too few to measure self-consistency. The "
                f"first error was: {first_error}",
                replicates,
            )
            return

        scores = [r.score for r in scored]
        low, high = self.score_range
        out_of_range = [s for s in scores if s < low or s > high]
        if out_of_range:
            self._fail(
                f"The judge returned scores outside 'score_range' "
                f"{self.score_range}: {out_of_range}. Stability is measured "
                f"relative to that range, so set 'score_range' to the range "
                f"the judge actually scores on.",
                replicates,
            )
            return

        # A judge with no threshold runs in score-only mode and reaches no
        # verdict, so there is nothing to flip.
        successes = [r.success for r in scored if r.success is not None]
        flip_rate = (
            decision_flip_rate(successes)
            if len(successes) == len(scored)
            else None
        )

        self.replicate_scores = scores
        self.decision_flip_rate = flip_rate
        self.result = JudgeSelfConsistencyResult(
            stability=stability(scores, self.score_range),
            decision_flip_rate=flip_rate,
            mean_score=sum(scores) / len(scores),
            min_score=min(scores),
            max_score=max(scores),
            score_interval=bootstrap_mean_interval(
                scores,
                resamples=self.bootstrap_resamples,
                confidence=self.confidence,
                seed=self.random_seed,
            ),
            replicates=replicates,
            errored_replicates=len(replicates) - len(scored),
        )

        self.score = self._calculate_score()
        self.score_breakdown = self.result.model_dump()
        self.reason = self._generate_reason()
        if self.auto_flaky and self._replicates_disagree():
            # `flaky` has always been hand-declared by whoever built the
            # metric. Here it is populated by measurement instead.
            self.flaky = True
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                "Replicate scores:\n"
                + ", ".join(f"{score:.4f}" for score in scores),
                f"Decision flip rate: {self._format_flip_rate()}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    def _replicates_disagree(self) -> bool:
        """Whether this run saw disagreement worth flagging.

        A verdict flip always counts, however small the score movement that
        caused it. Otherwise the reading has to actually fail its own
        threshold — floating-point jitter between two near-identical scores is
        not disagreement.
        """
        if self.result is None:
            return False
        if self.result.decision_flip_rate:
            return True
        if self.threshold is None:
            return False
        return self.score < self.threshold

    def _calculate_score(self) -> float:
        score = self.result.stability
        if self.strict_mode and score < self.threshold:
            return 0
        return score

    def _format_flip_rate(self) -> str:
        if self.decision_flip_rate is None:
            return "n/a (the judge has no threshold, so it reaches no verdict)"
        return f"{self.decision_flip_rate:.2f}"

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        result = self.result
        judge_name = getattr(self.metric, "__name__", "the judge")
        parts = [
            f"{len(result.replicates) - result.errored_replicates} repeats of "
            f"{judge_name} scored {result.min_score:.2f}-{result.max_score:.2f} "
            f"(mean {result.mean_score:.2f})"
        ]
        if result.score_interval is not None:
            low, high = result.score_interval
            parts.append(
                f"{int(self.confidence * 100)}% CI [{low:.2f}, {high:.2f}]"
            )
        if result.decision_flip_rate is None:
            parts.append(
                "the judge has no threshold, so no pass/fail verdict was "
                "reached"
            )
        elif result.decision_flip_rate == 0:
            parts.append("every repeat reached the same pass/fail verdict")
        else:
            parts.append(
                f"{result.decision_flip_rate:.0%} of repeat pairs disagreed on "
                "pass/fail"
            )
        if result.errored_replicates:
            parts.append(f"{result.errored_replicates} repeats errored")
        return "; ".join(parts) + "."

    @property
    def __name__(self):
        judge_name = getattr(self.metric, "__name__", None)
        if not judge_name:
            return "Judge Self-Consistency"
        return f"Judge Self-Consistency ({judge_name})"
