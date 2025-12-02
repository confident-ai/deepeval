from deepeval.optimization.configs import OptimizerDisplayConfig


def test_optimizer_display_config_defaults():
    """
    Basic sanity check on OptimizerDisplayConfig defaults.
    """
    display = OptimizerDisplayConfig()
    assert display.show_indicator is True
    # Current default: do NOT announce ties unless explicitly enabled
    assert display.announce_ties is False
