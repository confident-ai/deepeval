from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.tasks import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode
from deepeval.models import AnthropicModel


if __name__ == "__main__":
    # Example usage of the TruthfulQA benchmark
    # This will run the benchmark with the specified tasks and mode
    # using the Anthropic model.

    # Note: Ensure that you have the necessary API keys and configurations set up for the AnthropicModel.

    benchmark = TruthfulQA(
        tasks=[TruthfulQATask.FICTION], mode=TruthfulQAMode.MC2
    )
    benchmark.evaluate(model=AnthropicModel())
