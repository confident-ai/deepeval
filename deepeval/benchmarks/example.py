from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark


class ExampleBenchmark(DeepEvalBaseBenchmark):
    def __init__(self):
        super().__init__()
        # Now call the method to load the benchmark dataset and set test cases.
        self.load_benchmark_dataset()

    def load_benchmark_dataset(self):
        # load from hugging face
        pass
