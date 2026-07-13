from typing import Dict, List, Optional

from tqdm import tqdm

from deepeval.benchmarks.base_benchmark import (
    DeepEvalBaseBenchmark,
    DeepEvalBaseBenchmarkResult,
)
from deepeval.benchmarks.pub_med_qa.template import PubMedQATemplate
from deepeval.benchmarks.schema import PubMedQASchema
from deepeval.dataset import Golden
from deepeval.models import DeepEvalBaseLLM
from deepeval.telemetry import capture_benchmark_run


class PubMedQA(DeepEvalBaseBenchmark):
    def __init__(
        self,
        n_problems: int = 1000,
        verbose_mode: bool = False,
        confinement_instructions: Optional[str] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer
        import pandas as pd

        assert (
            0 < n_problems <= 1000
        ), "PubMedQA only supports 1 to 1000 problems"
        super().__init__(**kwargs)
        self.scorer = Scorer()
        self.n_problems = n_problems
        self.predictions: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode = verbose_mode
        self.confinement_instructions = confinement_instructions or (
            "Output only 'yes', 'no', or 'maybe'."
        )

    def evaluate(
        self, model: DeepEvalBaseLLM, *args, **kwargs
    ) -> DeepEvalBaseBenchmarkResult:
        import pandas as pd

        with capture_benchmark_run("PubMedQA", self.n_problems):
            correct_predictions = 0
            predictions_row = []
            goldens = self.load_benchmark_dataset()[: self.n_problems]

            for idx, golden in enumerate(
                tqdm(goldens, desc=f"Processing {self.n_problems} problems")
            ):
                result = self.predict(model, golden)
                prediction = result["prediction"]
                score = result["score"]
                correct_predictions += score
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

            overall_accuracy = correct_predictions / self.n_problems
            print(f"Overall PubMedQA Accuracy: {overall_accuracy}")

            self.predictions = pd.DataFrame(
                predictions_row,
                columns=["Input", "Prediction", "Expected Output", "Correct"],
            )
            self.overall_score = overall_accuracy
            return DeepEvalBaseBenchmarkResult(
                overall_accuracy=overall_accuracy
            )

    def predict(self, model: DeepEvalBaseLLM, golden: Golden) -> Dict:
        prompt = PubMedQATemplate.generate_output(golden.input)
        structured_prompt = (
            f'{prompt}\nReturn a JSON object with an "answer" field.'
        )

        try:
            res = model.generate(
                prompt=structured_prompt, schema=PubMedQASchema
            )
            if isinstance(res, tuple):
                res = res[0]
            prediction = res.answer
        except TypeError:
            prompt += f"\n\n{self.confinement_instructions}"
            prediction = model.generate(prompt)

        if isinstance(prediction, tuple):
            prediction = prediction[0]

        prediction = str(prediction).strip().lower()
        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )
        return {"prediction": prediction, "score": score}

    def load_benchmark_dataset(self) -> List[Golden]:
        from datasets import load_dataset

        if self.dataset is None:
            self.dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

        return [
            Golden(
                input=PubMedQATemplate.format_question(data),
                expected_output=PubMedQATemplate.format_answer(data),
            )
            for data in self.dataset["train"]
        ]

    def print_verbose_logs(
        self,
        idx: int,
        input: str,
        expected_output: str,
        prediction: str,
        score: int,
    ) -> str:
        verbose_logs = (
            f"Input:\n{input}\n \n"
            f"Score: {score}\nPrediction: {prediction}\n"
            f"Expected Output: {expected_output}"
        )
        if self.verbose_mode:
            print("*" * 50)
            print(f"Problem {idx + 1}")
            print("*" * 50)
            print(f"\n{verbose_logs}\n")
            print("=" * 70)
        return verbose_logs
