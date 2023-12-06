from typing import Union

from .benchmarks import BenchmarkType


def check(benchmark: Union[str, BenchmarkType]):
    if benchmark == BenchmarkType.HELM:
        handleHELMCheck()
    if benchmark == BenchmarkType.LM_HARNESS:
        handleLMHarnessCheck()
    else:
        # catch all for custom benchmark checks
        pass


def handleHELMCheck():
    pass


def handleLMHarnessCheck():
    pass
