from typing import List, Optional, Dict
from tqdm import tqdm
import pandas as pd
import requests
import json
import io

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.logi_qa.task import LogiQATask
from deepeval.benchmarks.logi_qa.template import LogiQATemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.benchmarks.schema import MultipleChoiceSchema
from deepeval.telemetry import capture_benchmark_run


class LogiQA(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[LogiQATask] = None,
        n_shots: int = 5,
        n_problems_per_task: Optional[int] = None,
        verbose_mode: bool = False,
        confinement_instructions: Optional[str] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer

        assert n_shots <= 5, "LogiQA only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.tasks: List[LogiQATask] = (
            list(LogiQATask) if tasks is None else tasks
        )
        self.n_problems_per_task: Optional[int] = n_problems_per_task
        self.scorer = Scorer()
        self.n_shots: int = n_shots
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode: bool = verbose_mode
        if not confinement_instructions:
            self.confinement_instructions = (
                "Output 'A', 'B', 'C', or 'D'. Full answer not needed."
            )
        else:
            self.confinement_instructions = confinement_instructions

    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        with capture_benchmark_run("LogiQA", len(self.tasks)):
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

                # Calculate task accuracy
                if use_batch:
                    for i in tqdm(
                        range(0, len(goldens), batch_size),
                        desc=f"Batch Processing {task.value} (batch_size={batch_size})",
                    ):
                        goldens_batch = goldens[i : i + batch_size]
                        batch_predictions = self.batch_predict(
                            model, goldens_batch
                        )
                        for golden, prediction_dict in zip(
                            goldens_batch, batch_predictions
                        ):
                            prediction = prediction_dict["prediction"]
                            score = prediction_dict["score"]
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
                else:
                    for idx, golden in enumerate(
                        tqdm(goldens, desc=f"Processing {task.value}")
                    ):
                        prediction, score = self.predict(model, golden).values()
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
                )
                print(
                    f"LogiQA Task Accuracy (task={task.value}): {task_accuracy}"
                )
                scores_row.append((task.value, task_accuracy))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall LogiQA Accuracy: {overall_accuracy}")

            # Create a DataFrame from task_results_data
            # Columns: 'Task', 'Input', 'Prediction', 'Score'
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

            return overall_accuracy

    def predict(self, model: DeepEvalBaseLLM, golden: Golden) -> Dict:
        # Define prompt template
        prompt: dict = LogiQATemplate.generate_output(
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

        # Define Metric
        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )
        return {"prediction": prediction, "score": score}

    def batch_predict(
        self, model: DeepEvalBaseLLM, goldens: List[Golden]
    ) -> List[Dict]:
        # Define prompt template
        prompts = []
        for golden in goldens:
            prompt: dict = LogiQATemplate.generate_output(
                input=golden.input,
                n_shots=self.n_shots,
            )
            prompts.append(prompt)

        # Enforced model generation
        try:
            responses: List[MultipleChoiceSchema] = model.batch_generate(
                prompts=prompts, schemas=[MultipleChoiceSchema for _ in prompts]
            )
            predictions = [res.answer for res in responses]
        except TypeError:
            prompts = [
                prompt
                + "\n\nOutput 'A', 'B', 'C', or 'D'. Full answer not needed."
                for prompt in prompts
            ]
            predictions = model.batch_generate(prompts)

        if len(predictions) is not len(goldens):
            raise ValueError(
                "Custom `batch_generate` method did not return the same number of generations as the number of prompts."
            )

        res = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            golden = goldens[i]
            # Define Metric
            score = self.scorer.exact_match_score(
                golden.expected_output, prediction
            )
            res.append({"prediction": prediction, "score": score})

        return res

    def load_benchmark_dataset(self, task: LogiQATask) -> List[Golden]:
        # Load dataset
        dataset_url = "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA/test.txt"
        if self.dataset:
            dataset = self.dataset
        else:
            dataset = self.download_and_load_hf_dataset(dataset_url)
            self.dataset = dataset

        # Construct test set
        goldens: List[Golden] = []
        for data in dataset:
            types: Dict = data["type"]
            if types.get(task.value) is True:
                input = LogiQATemplate.format_question(data)
                expected_output = LogiQATemplate.format_output(data)
                golden = Golden(input=input, expected_output=expected_output)
                goldens.append(golden)

        return goldens

    def download_and_load_hf_dataset(self, url):
        from datasets import Dataset

        try:
            response = requests.get(url)
            response.raise_for_status()
            raw_data = response.text.splitlines()
            parsed_data = [json.loads(line) for line in raw_data]
            hf_dataset = Dataset.from_list(parsed_data)
            return hf_dataset
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
        except Exception as e:
            print(f"Error processing dataset: {e}")

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
