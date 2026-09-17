from deepeval.models._summac_model import _SummaCConv

def test_summac_conv_init():
    try:
        _SummaCConv(models=['mnli'], imager_load_cache=False)
    except Exception as e:
        assert not isinstance(e, NameError), 'NameError should be fixed'
