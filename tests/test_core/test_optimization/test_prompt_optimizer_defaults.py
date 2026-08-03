from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.optimizer.algorithms import COPRO
from deepeval.optimizer.configs import DisplayConfig
from deepeval.optimizer.prompt_optimizer import PromptOptimizer
from deepeval.optimizer.scorer import Scorer

from tests.test_core.stubs import _DummyMetric


class _StubLLM(DeepEvalBaseLLM):
    """Keeps `initialize_model` off the GPTModel path so no API key is needed."""

    def load_model(self):
        return self

    def generate(self, *args, **kwargs):
        return ""

    async def a_generate(self, *args, **kwargs):
        return ""

    def get_model_name(self):
        return "stub"


def _callback_one(prompt, golden):
    return "from-callback-one"


def _callback_two(prompt, golden):
    return "from-callback-two"


def _build(model_callback):
    return PromptOptimizer(
        model_callback=model_callback,
        metrics=[_DummyMetric()],
        optimizer_model=_StubLLM(),
        display_config=DisplayConfig(show_indicator=False),
    )


def test_default_algorithm_is_not_shared_between_optimizers():
    first = _build(_callback_one)
    second = _build(_callback_two)

    assert first.algorithm is not second.algorithm
    assert first.algorithm.scorer is not second.algorithm.scorer


def test_building_a_second_optimizer_does_not_rewire_the_first():
    first = _build(_callback_one)
    second = _build(_callback_two)

    # `execute()` runs through `self.scorer`, so a shared algorithm instance
    # would make `first` optimize against the second optimizer's callback.
    first_scorer = first.algorithm.scorer
    second_scorer = second.algorithm.scorer
    assert isinstance(first_scorer, Scorer)
    assert isinstance(second_scorer, Scorer)
    assert first_scorer.model_callback is _callback_one
    assert second_scorer.model_callback is _callback_two

    # Progress/status callbacks must stay bound to their own optimizer.
    # Bound methods compare equal only when both the function and the
    # instance they are bound to match.
    assert first.algorithm.status_callback == first._on_status
    assert first.algorithm.step_callback == first._on_step
    assert second.algorithm.status_callback == second._on_status


def test_explicitly_passed_algorithm_instance_is_used_as_is():
    algorithm = COPRO()
    optimizer = PromptOptimizer(
        model_callback=_callback_one,
        metrics=[_DummyMetric()],
        optimizer_model=_StubLLM(),
        algorithm=algorithm,
        display_config=DisplayConfig(show_indicator=False),
    )

    assert optimizer.algorithm is algorithm
