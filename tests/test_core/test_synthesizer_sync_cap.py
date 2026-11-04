"""Sync goldens-from-contexts must cap synthetic data at max_goldens_per_context.

The async paths truncate right after generation; the sync paths did not, so a
model returning more items than requested crashed the evolve loop with an
IndexError against the max-sized progress-bar id list.
"""

from unittest.mock import MagicMock, patch

from deepeval.synthesizer.schema import SyntheticData

_INIT_MODEL_PATHS = [
    "deepeval.synthesizer.synthesizer.initialize_model",
    "deepeval.synthesizer.config.initialize_model",
]


def _make_synthesizer():
    from deepeval.synthesizer.synthesizer import Synthesizer

    fake_model = MagicMock()
    fake_model.get_model_name.return_value = "fake-model"
    fake_model.generate.return_value = ("fake", 0.0)
    with patch(_INIT_MODEL_PATHS[0], return_value=(fake_model, True)), patch(
        _INIT_MODEL_PATHS[1], return_value=(fake_model, True)
    ):
        return Synthesizer(model=fake_model, async_mode=False)


def _over_returning_synth(n_over: int):
    synth = _make_synthesizer()
    over = [SyntheticData(input=f"input_{i}") for i in range(n_over)]
    synth._generate_inputs = MagicMock(return_value=over)
    # 透传截断后的入参(与真实 _rewrite_inputs 的处理范围一致)
    synth._rewrite_inputs = MagicMock(
        side_effect=lambda ctx, inputs: (inputs, [0.9] * len(inputs))
    )
    synth._evolve_input = MagicMock(return_value=("evolved", []))
    synth._generate = MagicMock(return_value=("expected output", 0.0))
    synth._generate_schema = MagicMock(return_value=SyntheticData(input="x"))
    return synth


def test_sync_caps_goldens_at_max_per_context():
    """Model returns 3 items when 2 were requested: sync must not crash
    and must yield exactly max_goldens_per_context goldens (async parity)."""
    synth = _over_returning_synth(3)

    synth.generate_goldens_from_contexts(
        contexts=[["CREATE TABLE users (id INT)"]],
        max_goldens_per_context=2,
        )

    assert len(synth.synthetic_goldens) == 2
