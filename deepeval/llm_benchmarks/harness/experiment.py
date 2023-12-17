from deepeval.llm_benchmarks.harness.task import HarnessTasks
from deepeval.llm_benchmarks.harness.evaluation import (
    HarnessEvaluate,
    HarnessConfig,
)


class Experiment:
    """An experiment will make a seperate folder in the following order:

    ExperimentName:
        - trial 1:
            config files
            result files
        - trial 2:
            config files
            result files
    """

    raise NotImplementedError()
