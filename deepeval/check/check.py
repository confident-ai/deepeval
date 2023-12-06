from typing import Union

from .benchmarks import BenchmarkType

def check(alias: Union[str, BenchmarkType]):
    if alias == BenchmarkType.HELM:
        handleHELMCheck()
    if alias == BenchmarkType.LM_HARNESS:
        handleLMHarnessCheck()
    else:
        # catch all for custom benchmark checks
        pass


def handleHELMCheck():
    pass

def handleLMHarnessCheck():
    pass