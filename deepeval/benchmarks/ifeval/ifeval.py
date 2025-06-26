from typing import List, Optional, Dict
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.schema import StringSchema
from deepeval.telemetry import capture_benchmark_run
from deepeval.scorer import Scorer


class IFEval(DeepEvalBaseBenchmark):
    """
    IFEval (Instruction Following Evaluation) benchmark implementation.

    IFEval is a benchmark for evaluating instruction-following capabilities of language models.
    It tests various aspects of instruction following including format compliance, constraint 
    adherence, output structure requirements, and specific instruction types.

    Based on the original IFEval paper: https://arxiv.org/abs/2311.07911
    and implementation: https://github.com/google-research/google-research/tree/master/instruction_following_eval
    """

    def __init__(
        self,
        n_problems: Optional[int] = None,
        verbose_mode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scorer = Scorer()
        self.n_problems = n_problems
        self.verbose_mode = verbose_mode
        self.predictions = None
        self.overall_score = None

    def evaluate(self, model: DeepEvalBaseLLM) -> Dict:
        import pandas as pd

        with capture_benchmark_run("IFEval", self.n_problems or "all"):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []

            # Load all test cases
            goldens = self.load_benchmark_dataset()
            if self.n_problems and self.n_problems < len(goldens):
                goldens = goldens[:self.n_problems]

            overall_total_predictions = len(goldens)

            # Process each test case
            for idx, golden in enumerate(
                tqdm(
                    goldens, desc=f"Processing {len(goldens)} IFEval problems")
            ):
                prediction, score = self.predict(model, golden).values()
                if score:
                    overall_correct_predictions += 1
                predictions_row.append(
                    (golden.input, prediction, golden.expected_output, score)
                )
                if self.verbose_mode:
                    self.print_verbose_logs(
                        idx,
                        golden.input,
                        golden.expected_output,
                        prediction,
                        score,
                    )

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall IFEval Accuracy: {overall_accuracy}")

            # Create a DataFrame from results
            self.predictions = pd.DataFrame(
                predictions_row,
                columns=[
                    "Input",
                    "Prediction",
                    "Expected Output",
                    "Correct",
                ],
            )
            self.overall_score = overall_accuracy

            return {
                "overall_accuracy": overall_accuracy,
                "predictions": self.predictions,
            }

    def predict(self, model: DeepEvalBaseLLM, golden: Golden) -> Dict:
        """
        Generate prediction for a single IFEval test case.

        Args:
            model: The language model to evaluate
            golden: The golden test case

        Returns:
            Dictionary containing prediction and score
        """
        try:
            # Try structured generation first
            res: StringSchema = model.generate(
                prompt=golden.input, schema=StringSchema
            )
            prediction = res.answer
        except (TypeError, AttributeError):
            # Fallback to free-form generation
            res = model.generate(golden.input)
            prediction = str(res)

        # Score based on exact match
        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )

        return {"prediction": prediction, "score": score}

    def load_benchmark_dataset(self) -> List[Golden]:
        """
        Load IFEval dataset.

        Returns:
            List of Golden test cases
        """
        from datasets import load_dataset

        if self.dataset:
            dataset = self.dataset
        else:
            dataset = load_dataset("google/IFEval", trust_remote_code=True)
            self.dataset = dataset

        goldens: List[Golden] = []

        test_data = dataset["test"]

        for data in test_data:
            instruction = data.get("instruction", "")
            expected_output = data.get("expected_output", "")

            golden = Golden(
                input=instruction,
                expected_output=expected_output
            )
            goldens.append(golden)

        return goldens

    def print_verbose_logs(
        self,
        idx: int,
        input: str,
        expected_output: str,
        prediction: str,
        score: int,
    ) -> str:
        """
        Print verbose logs for debugging and analysis.

        Args:
            idx: Problem index
            input: Input instruction
            expected_output: Expected output
            prediction: Model prediction
            score: Score (0 or 1)

        Returns:
            Formatted verbose log string
        """
        steps = [
            f"Input:\n{input}",
            f"Score: {score}\nPrediction: {prediction}\nExpected Output: {expected_output}",
        ]
        verbose_logs = ""
        for i in range(len(steps) - 1):
            verbose_logs += steps[i]

            # don't add new line for penultimate step
            if i < len(steps) - 2:
                verbose_logs += " \n \n"

        if self.verbose_mode:
            print("*" * 50)
            print(f"Problem {idx + 1}")
            print("*" * 50)
            print("")
            print(verbose_logs + f"\n \n{steps[-1]}")
            print("")
            print("=" * 70)

        return verbose_logs
