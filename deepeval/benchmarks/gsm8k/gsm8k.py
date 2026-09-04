from typing import List, Optional, Dict, Union
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import (
    DeepEvalBaseBenchmark,
    DeepEvalBaseBenchmarkResult,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
from deepeval.benchmarks.schema import NumberSchema
from deepeval.benchmarks.utils import should_use_batch
from deepeval.telemetry import capture_benchmark_run


class GSM8K(DeepEvalBaseBenchmark):
    def __init__(
        self,
        n_shots: int = 3,
        enable_cot: bool = True,
        n_problems: int = 1319,
        verbose_mode: bool = False,
        confinement_instructions: Optional[str] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer
        import pandas as pd

        assert n_shots <= 15, "GSM8K only supports n_shots <= 15"
        super().__init__(**kwargs)
        self.scorer = Scorer()
        self.shots_dataset: List[Dict] = None
        self.n_shots: int = n_shots
        self.enable_cot: bool = enable_cot
        self.n_problems: int = n_problems
        self.predictions: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode = verbose_mode
        if not confinement_instructions:
            self.confinement_instructions = (
                "Make sure to output only the numerical answer."
            )
        else:
            self.confinement_instructions = confinement_instructions

    def evaluate(
        self,
        model: DeepEvalBaseLLM,
        *args,
        batch_size: Union[int, None] = None,
        **kwargs,
    ) -> DeepEvalBaseBenchmarkResult:
        import pandas as pd

        with capture_benchmark_run("GSM8K", len(self.tasks)):
            overall_correct_predictions = 0
            overall_total_predictions = self.n_problems
            predictions_row = []
            use_batch = should_use_batch(model, batch_size)

            # Solving each problem
            goldens = self.load_benchmark_dataset()[: self.n_problems]

            if use_batch:
                for i in tqdm(
                    range(0, len(goldens), batch_size),
                    desc=f"Batch Processing {self.n_problems} problems (batch_size={batch_size})",
                ):
                    goldens_batch = goldens[i : i + batch_size]
                    batch_predictions = self.batch_predict(model, goldens_batch)
                    for golden, prediction_dict in zip(
                        goldens_batch, batch_predictions
                    ):
                        prediction = prediction_dict["prediction"]
                        score = prediction_dict["score"]
                        if score:
                            overall_correct_predictions += 1
                        predictions_row.append(
                            (
                                golden.input,
                                prediction,
                                golden.expected_output,
                                score,
                            )
                        )
            else:
                for idx, golden in enumerate(
                    tqdm(goldens, desc=f"Processing {self.n_problems} problems")
                ):
                    result = self.predict(model, golden)
                    prediction = result["prediction"]
                    score = result["score"]

                    if score:
                        overall_correct_predictions += 1
                    predictions_row.append(
                        (
                            golden.input,
                            prediction,
                            golden.expected_output,
                            score,
                        )
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
            print(f"Overall GSM8K Accuracy: {overall_accuracy}")

            self.predictions = pd.DataFrame(
                predictions_row,
                columns=["Input", "Prediction", "Expected Output", "Correct"],
            )
            self.overall_score = overall_accuracy

            return DeepEvalBaseBenchmarkResult(
                overall_accuracy=overall_accuracy
            )

    def predict(self, model: DeepEvalBaseLLM, golden: Golden) -> Dict:
        # Define prompt template
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."
        prompt: dict = GSM8KTemplate.generate_output(
            train_set=self.shots_dataset,
            input=golden.input,
            n_shots=self.n_shots,
            enable_cot=self.enable_cot,
        )

        # Enforced model generation
        prediction = None
        try:
            res: NumberSchema = model.generate(
                prompt=prompt, schema=NumberSchema
            )
            prediction = self._extract_prediction_from_response(res)
        except (TypeError, AttributeError) as e:

            prompt += f"\n\n{self.confinement_instructions}"
            res = model.generate(prompt)
            prediction = self._extract_prediction_from_response(res)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = str(prediction)

        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )

        return {"prediction": prediction, "score": score}

    def batch_predict(
        self, model: DeepEvalBaseLLM, goldens: List[Golden]
    ) -> List[Dict]:
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."

        prompts = []
        for golden in goldens:
            prompt = GSM8KTemplate.generate_output(
                train_set=self.shots_dataset,
                input=golden.input,
                n_shots=self.n_shots,
                enable_cot=self.enable_cot,
            )
            prompts.append(prompt)

        try:
            responses = model.batch_generate(
                prompts=prompts, schemas=[NumberSchema for _ in prompts]
            )
            if not isinstance(responses, list):
                raise TypeError(
                    "batch_generate must return a list of responses"
                )
            predictions = [
                self._extract_prediction_from_response(res) for res in responses
            ]
        except (TypeError, AttributeError):
            prompts = [
                prompt + f"\n\n{self.confinement_instructions}"
                for prompt in prompts
            ]
            responses = model.batch_generate(prompts)
            predictions = [
                self._extract_prediction_from_response(res) for res in responses
            ]

        if len(predictions) != len(goldens):
            raise ValueError(
                "Custom `batch_generate` method did not return the same number of generations as the number of prompts."
            )

        res = []
        for prediction, golden in zip(predictions, goldens):
            score = self.scorer.exact_match_score(
                golden.expected_output, prediction
            )
            res.append({"prediction": prediction, "score": score})

        return res

    def _extract_prediction_from_response(self, res) -> str:
        """
        Extract prediction from model response, handling various response types.
        """
        # Case 1: Response has .answer attribute (NumberSchema case)
        if hasattr(res, "answer"):
            return str(res.answer)

        # Case 2: Response is a tuple
        elif isinstance(res, tuple):
            return self._extract_from_tuple(res)

        else:
            return str(res)

    def _extract_from_tuple(self, res: tuple) -> str:
        """Extract prediction from tuple response."""
        if len(res) == 0:
            return ""
        first_elem = res[0]
        if hasattr(first_elem, "answer"):
            return str(first_elem.answer)

    def load_benchmark_dataset(self) -> List[Golden]:
        from datasets import load_dataset

        # Load dataset
        if self.dataset:
            dataset = self.dataset
        else:
            dataset = load_dataset("gsm8k", "main")
            self.dataset = dataset

        # Construct example dataset for n_shot inference
        if not self.shots_dataset:
            train_set = dataset["train"]
            shots_set = []
            for data in train_set:
                shots_set.append(data)
            self.shots_dataset = shots_set

        # Construct test set
        goldens: List[Golden] = []
        for data in dataset["test"]:
            input = data["question"]
            output = GSM8KTemplate.format_answer(data)
            golden = Golden(input=input, expected_output=output)
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
