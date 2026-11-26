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


class _CountingProposer:
    """Records how many times propose/a_propose are invoked."""

    def __init__(self):
        self.propose_calls = 0
        self.a_propose_calls = 0

    def propose(self, prompt, goldens, num_candidates):
        self.propose_calls += 1
        return [prompt]

    async def a_propose(self, prompt, goldens, num_candidates):
        self.a_propose_calls += 1
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
def _miprov2_with_counting_proposer(monkeypatch):
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
    proposer = _CountingProposer()
    bootstrapper = _DummyBootstrapper()
    algo._init_components = lambda: (
        setattr(algo, "proposer", proposer),
        setattr(algo, "bootstrapper", bootstrapper),
    )
    return algo, proposer


def test_execute_calls_propose_exactly_once(_miprov2_with_counting_proposer):
    """
    MIPROV2.execute() must call InstructionProposer.propose() exactly once.

    Regression test for a duplicated call that silently doubled the number of
    LLM calls (and cost/latency) made during the "Propose" phase of every
    synchronous MIPROv2 optimization run: propose() was invoked twice in a
    row with identical arguments, and the first result was discarded.
    """
    algo, proposer = _miprov2_with_counting_proposer
    prompt = Prompt(text_template="base {input}")
    goldens = [Golden(input="q1", expected_output="a1")]

    algo.execute(prompt=prompt, goldens=goldens)

    assert proposer.propose_calls == 1, (
        f"expected propose() to be called exactly once, got "
        f"{proposer.propose_calls}"
    )


@pytest.mark.asyncio
async def test_a_execute_calls_a_propose_exactly_once(
    _miprov2_with_counting_proposer,
):
    """Async twin: a_execute() has always called a_propose() exactly once."""
    algo, proposer = _miprov2_with_counting_proposer
    prompt = Prompt(text_template="base {input}")
    goldens = [Golden(input="q1", expected_output="a1")]

    await algo.a_execute(prompt=prompt, goldens=goldens)

    assert proposer.a_propose_calls == 1, (
        f"expected a_propose() to be called exactly once, got "
        f"{proposer.a_propose_calls}"
    )
