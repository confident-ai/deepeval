from typing import Dict, List, Optional, Union
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import (
    DeepEvalBaseBenchmark,
    DeepEvalBaseBenchmarkResult,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.halu_eval.task import HaluEvalTask
from deepeval.benchmarks.halu_eval.template import HaluEvalTemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.benchmarks.schema import AffirmationSchema
from deepeval.telemetry import capture_benchmark_run

HALU_EVAL_DATASET = "pminervini/HaluEval"


class HaluEval(DeepEvalBaseBenchmark):
    """HaluEval: a benchmark for recognizing hallucinated responses.

    For each sample the model answers 'Yes' (the response is hallucinated)
    or 'No' (it is not), and is scored by exact match.
    """

    def __init__(
        self,
        tasks: Optional[List[HaluEvalTask]] = None,
        n_problems_per_task: Optional[int] = None,
        verbose_mode: bool = False,
        confinement_instructions: Optional[str] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer
        import pandas as pd

        super().__init__(**kwargs)
        self.tasks: List[HaluEvalTask] = (
            list(HaluEvalTask) if tasks is None else tasks
        )
        self.n_problems_per_task: Optional[int] = n_problems_per_task
        self.scorer = Scorer()
        self.datasets: Dict[HaluEvalTask, object] = {}
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode: bool = verbose_mode
        if not confinement_instructions:
            self.confinement_instructions = (
                "Make sure to output only 'Yes' or 'No'."
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

        with capture_benchmark_run("HaluEval", len(self.tasks)):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            scores_row = []
            use_batch = should_use_batch(model, batch_size)

            for task in self.tasks:
                goldens = self.load_benchmark_dataset(task)
                if (
                    self.n_problems_per_task is not None
                    and self.n_problems_per_task < len(goldens)
                ):
                    goldens = goldens[: self.n_problems_per_task]
                task_correct_predictions = 0
                task_total_predictions = len(goldens)
                overall_total_predictions += len(goldens)

                if use_batch:
                    results = []
                    for i in tqdm(
                        range(0, len(goldens), batch_size),
                        desc=f"Batch Processing {task.value} (batch_size={batch_size})",
                    ):
                        goldens_batch = goldens[i : i + batch_size]
                        results.extend(
                            zip(
                                goldens_batch,
                                self.batch_predict(model, goldens_batch, task),
                            )
                        )
                else:
                    results = []
                    for golden in tqdm(
                        goldens, desc=f"Processing {task.value}"
                    ):
                        results.append(
                            (golden, self.predict(model, golden, task))
                        )

                for idx, (golden, result) in enumerate(results):
                    prediction, score = result["prediction"], result["score"]
                    if score:
                        task_correct_predictions += 1
                        overall_correct_predictions += 1
                    predictions_row.append(
                        (
                            task.value,
                            golden.input,
                            prediction,
                            golden.expected_output,
                            score,
                        )
                    )
                    if self.verbose_mode:
                        self.print_verbose_logs(
                            idx,
                            task.value,
                            golden.input,
                            golden.expected_output,
                            prediction,
                            score,
                        )

                task_accuracy = (
                    task_correct_predictions / task_total_predictions
                    if task_total_predictions
                    else 0.0
                )
                print(
                    f"HaluEval Task Accuracy (task={task.value}): {task_accuracy}"
                )
                scores_row.append((task.value, task_accuracy))

            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
                if overall_total_predictions
                else 0.0
            )
            print(f"Overall HaluEval Accuracy: {overall_accuracy}")

            self.predictions = pd.DataFrame(
                predictions_row,
                columns=[
                    "Task",
                    "Input",
                    "Prediction",
                    "Expected Output",
                    "Correct",
                ],
            )
            self.task_scores = pd.DataFrame(
                scores_row, columns=["Task", "Score"]
            )
            self.overall_score = overall_accuracy

            return DeepEvalBaseBenchmarkResult(
                overall_accuracy=overall_accuracy
            )

    def predict(
        self, model: DeepEvalBaseLLM, golden: Golden, task: HaluEvalTask
    ) -> Dict:
        prompt = HaluEvalTemplate.generate_output(
            input=golden.input, task=task
        )

        # Enforced model generation
        try:
            res: AffirmationSchema = model.generate(
                prompt=prompt, schema=AffirmationSchema
            )
            prediction = str(res.answer)
        except (TypeError, ValueError, AttributeError):
            prompt += f"\n\n{self.confinement_instructions}"
            prediction = model.generate(prompt)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = HaluEvalTemplate.normalize_prediction(prediction)

        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )
        return {"prediction": prediction, "score": score}

    def batch_predict(
        self,
        model: DeepEvalBaseLLM,
        goldens: List[Golden],
        task: HaluEvalTask,
    ) -> List[Dict]:
        prompts = [
            HaluEvalTemplate.generate_output(input=golden.input, task=task)
            for golden in goldens
        ]

        # Enforced model generation
        try:
            responses: List[AffirmationSchema] = model.batch_generate(
                prompts=prompts,
                schemas=[AffirmationSchema for _ in prompts],
            )
            predictions = [str(res.answer) for res in responses]
        except (TypeError, ValueError, AttributeError):
            prompts = [
                f"{prompt}\n\n{self.confinement_instructions}"
                for prompt in prompts
            ]
            predictions = model.batch_generate(prompts)
            predictions = [
                pred[0] if isinstance(pred, tuple) else pred
                for pred in predictions
            ]

        if len(predictions) != len(goldens):
            raise ValueError(
                "Custom `batch_generate` method did not return the same number of generations as the number of prompts."
            )

        res = []
        for golden, prediction in zip(goldens, predictions):
            prediction = HaluEvalTemplate.normalize_prediction(prediction)
            score = self.scorer.exact_match_score(
                golden.expected_output, prediction
            )
            res.append({"prediction": prediction, "score": score})
        return res

    def load_benchmark_dataset(self, task: HaluEvalTask) -> List[Golden]:
        from datasets import load_dataset

        if task not in self.datasets:
            self.datasets[task] = load_dataset(HALU_EVAL_DATASET, task.value)[
                "data"
            ]
        dataset = self.datasets[task]

        goldens: List[Golden] = []
        for data in dataset:
            for input, expected_output in HaluEvalTemplate.format_samples(
                data, task
            ):
                goldens.append(
                    Golden(input=input, expected_output=expected_output)
                )
        return goldens

    def print_verbose_logs(
        self,
        idx: int,
        task_value: str,
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
            print(f"Problem {idx + 1} (Task = {task_value})")
            print("*" * 50)
            print("")
            print(verbose_logs + f"\n \n{steps[-1]}")
            print("")
            print("=" * 70)

        return verbose_logs
