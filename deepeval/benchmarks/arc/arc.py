from typing import List, Optional, Dict
import pandas as pd
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.arc.mode import ARCMode
from deepeval.benchmarks.arc.template import ARCTemplate
from deepeval.benchmarks.schema import MultipleChoiceSchema
from deepeval.telemetry import capture_benchmark_run


class ARC(DeepEvalBaseBenchmark):
    def __init__(
        self,
        n_shots: int = 5,
        n_problems: Optional[int] = None,
        mode: ARCMode = ARCMode.EASY,
        verbose_mode: bool = False,
        confinement_instructions: Optional[str] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer

        assert n_shots <= 5, "ARC only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.mode: ARCMode = mode
        self.scorer = Scorer()
        self.n_shots: int = n_shots
        if mode == ARCMode.EASY:
            self.n_problems: int = 2376 if n_problems is None else n_problems
            assert (
                self.n_problems <= 2376
            ), "ARC-Easy only supports n_problems <= 2376"
        else:
            self.n_problems: int = 1172 if n_problems is None else n_problems
            assert (
                self.n_problems <= 1172
            ), "ARC-Challenge only supports n_problems <= 1172"
        self.predictions: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode = verbose_mode
        if not confinement_instructions:
            self.confinement_instructions = (
                "Output 'A', 'B', 'C', or 'D'. Full answer not needed."
            )
        else:
            self.confinement_instructions = confinement_instructions

    def evaluate(self, model: DeepEvalBaseLLM) -> Dict:
        with capture_benchmark_run("ARC", self.n_problems):
            overall_correct_predictions = 0
            overall_total_predictions = self.n_problems
            predictions_row = []

            # Solving each problem
            goldens: List[Golden] = self.load_benchmark_dataset(self.mode)[
                : self.n_problems
            ]
            for idx, golden in enumerate(
                tqdm(goldens, desc=f"Processing {self.n_problems} problems")
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
            print(f"Overall ARC Accuracy: {overall_accuracy}")

            self.predictions = pd.DataFrame(
                predictions_row,
                columns=["Input", "Prediction", "Expected Output", "Correct"],
            )
            self.overall_score = overall_accuracy

            return overall_accuracy

    def predict(self, model: DeepEvalBaseLLM, golden: Golden) -> Dict:
        # Define prompt template
        prompt: dict = ARCTemplate.generate_output(
            input=golden.input,
            n_shots=self.n_shots,
        )

        # Enforced model generation
        try:
            res: MultipleChoiceSchema = model.generate(
                prompt=prompt, schema=MultipleChoiceSchema
            )
            prediction = res.answer
        except TypeError:
            prompt += f"\n\n{self.confinement_instructions}"
            prediction = model.generate(prompt)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )
        return {"prediction": prediction, "score": score}

    def load_benchmark_dataset(self, mode: ARCMode) -> List[Golden]:
        from datasets import load_dataset

        # Load full dataset
        dataset_mapping = {
            ARCMode.CHALLENGE: "challenge_dataset",
            ARCMode.EASY: "easy_dataset",
        }
        dataset_attr = dataset_mapping.get(mode)
        if dataset_attr:
            if not hasattr(self, dataset_attr):
                dataset = load_dataset(
                    "ai2_arc", mode.value, trust_remote_code=True
                )
                setattr(self, dataset_attr, dataset)
            else:
                dataset = getattr(self, dataset_attr)

        # Construct test set
        goldens: List[Golden] = []
        for data in dataset["train"]:
            input = ARCTemplate.format_question(data, False)
            expected_output = ARCTemplate.format_answer(data)
            golden = Golden(input=input, expected_output=expected_output)
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
