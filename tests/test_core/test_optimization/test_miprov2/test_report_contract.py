import pytest
from types import SimpleNamespace

from deepeval.dataset.golden import Golden
from deepeval.optimizer.algorithms.miprov2.bootstrapper import DemonstrationSet
from deepeval.optimizer.algorithms.miprov2.miprov2 import MIPROV2
from deepeval.prompt.prompt import Prompt


class _DummyTrial:
    def __init__(self):
        self.params = {}

    def suggest_categorical(self, name, choices):
        choice = choices[0]
        self.params[name] = choice
        return choice


class _DummyStudy:
    def __init__(self):
        self._trial = _DummyTrial()

    def ask(self):
        return self._trial

    def tell(self, trial, score):
        self.best_trial = trial

    @property
    def best_trial(self):
        return self._trial

    @best_trial.setter
    def best_trial(self, trial):
        self._trial = trial


class _DummyProposer:
    def propose(self, prompt, goldens, num_candidates):
        return [prompt]

    async def a_propose(self, prompt, goldens, num_candidates):
        return [prompt]


class _DummyBootstrapper:
    def bootstrap(self, prompt, goldens):
        return [DemonstrationSet(demonstrations=[])]

    async def a_bootstrap(self, prompt, goldens):
        return [DemonstrationSet(demonstrations=[])]


class _DummyScorer:
    def score_minibatch(self, prompt_configuration, minibatch):
        return 0.5

    async def a_score_minibatch(self, prompt_configuration, minibatch):
        return 0.5

    def score_pareto(self, prompt_configuration, goldens):
        return [0.5 for _ in goldens]

    async def a_score_pareto(self, prompt_configuration, goldens):
        return [0.5 for _ in goldens]


@pytest.fixture
def _miprov2_with_stubs(monkeypatch):
    from deepeval.optimizer.algorithms.miprov2 import miprov2 as miprov2_module

    monkeypatch.setattr(miprov2_module, "OPTUNA_AVAILABLE", True)
    monkeypatch.setattr(miprov2_module, "TPESampler", lambda seed: None)
    monkeypatch.setattr(
        miprov2_module,
        "optuna",
        SimpleNamespace(
            create_study=lambda **kwargs: _DummyStudy(),
            logging=SimpleNamespace(
                WARNING=0,
                set_verbosity=lambda *args, **kwargs: None,
            ),
        ),
    )
    algo = MIPROV2(num_trials=1, num_candidates=1, minibatch_full_eval_steps=1)
    algo.scorer = _DummyScorer()
    algo.optimizer_model = object()
    algo._init_components = lambda: (
        setattr(algo, "proposer", _DummyProposer()),
        setattr(algo, "bootstrapper", _DummyBootstrapper()),
    )
    return algo


def test_miprov2_execute_report_contract(_miprov2_with_stubs):
    prompt = Prompt(text_template="base {input}")
    goldens = [Golden(input="q1", expected_output="a1")]

    best_prompt, report = _miprov2_with_stubs.execute(
        prompt=prompt, goldens=goldens
    )

    assert best_prompt is not None
    assert isinstance(report.pareto_scores, dict)
    assert report.pareto_scores
    assert all(isinstance(v, list) for v in report.pareto_scores.values())
    assert isinstance(report.accepted_iterations, list)


@pytest.mark.asyncio
async def test_miprov2_a_execute_report_contract(_miprov2_with_stubs):
    prompt = Prompt(text_template="base {input}")
    goldens = [Golden(input="q1", expected_output="a1")]

    best_prompt, report = await _miprov2_with_stubs.a_execute(
        prompt=prompt, goldens=goldens
    )

    assert best_prompt is not None
    assert isinstance(report.pareto_scores, dict)
    assert report.pareto_scores
    assert all(isinstance(v, list) for v in report.pareto_scores.values())
    assert isinstance(report.accepted_iterations, list)
