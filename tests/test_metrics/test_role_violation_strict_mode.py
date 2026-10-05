from deepeval.metrics import RoleViolationMetric
from deepeval.metrics.role_violation.schema import RoleViolationVerdict
from deepeval.models import DeepEvalBaseLLM


class _StubModel(DeepEvalBaseLLM):
    """The strict-mode behavior under test is the threshold initialization and
    the deterministic ``_calculate_score``, neither of which needs an LLM."""

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-model"

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError("strict-mode test must not call the LLM")

    async def a_generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError("strict-mode test must not call the LLM")


def _make_metric(
    strict_mode: bool, threshold: float = 0.5
) -> RoleViolationMetric:
    return RoleViolationMetric(
        role="helpful assistant",
        model=_StubModel(),
        strict_mode=strict_mode,
        threshold=threshold,
        async_mode=False,
        include_reason=False,
    )


def _score(metric: RoleViolationMetric, verdicts) -> tuple:
    metric.verdicts = [
        RoleViolationVerdict(verdict=v, reason=r) for v, r in verdicts
    ]
    metric.score = metric._calculate_score()
    return metric.score, metric.is_successful()


class TestRoleViolationStrictMode:
    def test_strict_mode_sets_threshold_to_one(self):
        # Regresses: strict_mode initialized threshold to 0, so a failing
        # check (0.0 >= 0) still reported success.
        assert _make_metric(strict_mode=True).threshold == 1

    def test_default_threshold_preserved(self):
        assert _make_metric(strict_mode=False).threshold == 0.5

    def test_explicit_threshold_preserved_when_not_strict(self):
        assert _make_metric(strict_mode=False, threshold=0.8).threshold == 0.8

    def test_strict_mode_fails_on_role_violation(self):
        metric = _make_metric(strict_mode=True)
        score, success = _score(metric, [("yes", "broke character")])
        assert score == 0.0
        assert success is False

    def test_strict_mode_passes_without_violation(self):
        metric = _make_metric(strict_mode=True)
        score, success = _score(metric, [("no", "stayed in role")])
        assert score == 1.0
        assert success is True

    def test_non_strict_still_fails_on_role_violation(self):
        metric = _make_metric(strict_mode=False)
        score, success = _score(metric, [("yes", "broke character")])
        assert score == 0.0
        assert success is False

    def test_no_violations_always_scores_one(self):
        for strict in (True, False):
            metric = _make_metric(strict_mode=strict)
            score, success = _score(metric, [])
            assert score == 1.0
            assert success is True
