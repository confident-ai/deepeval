from typing import List, Optional, Dict
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.bool_q.template import BoolQTemplate
from deepeval.scorer import Scorer
from deepeval.benchmarks.schema import AffirmationSchema
from deepeval.telemetry import capture_benchmark_run


class BoolQ(DeepEvalBaseBenchmark):
    def __init__(
        self,
        n_shots: int = 5,
        n_problems: int = 3270,
        **kwargs,
    ):
        assert n_shots <= 5, "BoolQ only supports n_shots <= 5"
        assert n_problems <= 3270, "BoolQ only supports n_problems <= 3270"

        super().__init__(**kwargs)
        self.scorer = Scorer()

        self.n_shots: int = n_shots
        self.n_problems: int = n_problems

        self.predictions: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None

    def evaluate(self, model: DeepEvalBaseLLM) -> Dict:
        with capture_benchmark_run("BoolQ", self.n_problems):
            overall_correct_predictions = 0
            overall_total_predictions = self.n_problems
            predictions_row = []

            # Solving each problem
            goldens = self.load_benchmark_dataset()[: self.n_problems]
            for golden in tqdm(
                goldens, desc=f"Processing {self.n_problems} problems"
            ):
                prediction, score = self.predict(model, golden).values()
                if score:
                    overall_correct_predictions += 1
                predictions_row.append((golden.input, prediction, score))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall BoolQ Accuracy: {overall_accuracy}")

            self.predictions = pd.DataFrame(
                predictions_row, columns=["Input", "Prediction", "Correct"]
            )
            self.overall_score = overall_accuracy

            return overall_accuracy

    def predict(self, model: DeepEvalBaseLLM, golden: Golden) -> Dict:
        # Define prompt template
        prompt: dict = BoolQTemplate.generate_output(
            input=golden.input,
            n_shots=self.n_shots,
        )

        # Enforced model generation
        try:
            res: AffirmationSchema = model.generate(
                prompt=prompt, schema=AffirmationSchema
            )
            prediction = str(res.answer)
        except TypeError:
            prompt += "Make sure to output only 'Yes' or 'No'."
            prediction = model.generate(prompt)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )

        return {"prediction": prediction, "score": score}

    def load_benchmark_dataset(self) -> List[Golden]:
        # Load dataset
        if self.dataset:
            dataset = self.dataset
        else:
            dataset = load_dataset("boolq", "default", trust_remote_code=True)
            self.dataset = dataset

        # Construct test set
        goldens: List[Golden] = []
        for data in dataset["validation"]:
            input = BoolQTemplate.format_question(data)
            expected_output = BoolQTemplate.format_answer(data)
            golden = Golden(input=input, expected_output=expected_output)
            goldens.append(golden)

        return goldens
