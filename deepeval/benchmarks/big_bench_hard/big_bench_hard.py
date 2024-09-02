from typing import List, Optional, Dict
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
from deepeval.benchmarks.big_bench_hard.template import BigBenchHardTemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.scorer import Scorer
from deepeval.benchmarks.schema import *


class BigBenchHard(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[BigBenchHardTask] = None,
        n_shots: int = 3,
        enable_cot: bool = True,
        **kwargs,
    ):
        assert n_shots <= 3, "BBH only supports n_shots <= 3"
        super().__init__(**kwargs)
        self.tasks: List[BigBenchHardTask] = (
            list(BigBenchHardTask) if tasks is None else tasks
        )
        self.scorer = Scorer()
        self.n_shots: int = n_shots
        self.enable_cot: bool = enable_cot
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None

    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        overall_correct_predictions = 0
        overall_total_predictions = 0
        predictions_row = []
        scores_row = []
        use_batch = should_use_batch(model, batch_size)

        for task in self.tasks:
            goldens = self.load_benchmark_dataset(task)
            task_correct_predictions = 0
            task_total_predictions = len(goldens)
            overall_total_predictions += len(goldens)

            if use_batch:
                for i in tqdm(
                    range(0, len(goldens), batch_size),
                    desc=f"Batch Processing {task.value} (batch_size={batch_size})",
                ):
                    goldens_batch = goldens[i : i + batch_size]
                    batch_predictions = self.batch_predict(
                        model, task, goldens_batch
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
                            (task.value, golden.input, prediction, score)
                        )
            else:
                # Calculate task accuracy
                for golden in tqdm(goldens, desc=f"Processing {task.value}"):
                    prediction, score = self.predict(
                        model, task, golden
                    ).values()
                    if score:
                        task_correct_predictions += 1
                        overall_correct_predictions += 1
                    predictions_row.append(
                        (task.value, golden.input, prediction, score)
                    )

            task_accuracy = task_correct_predictions / task_total_predictions
            print(
                f"Big Bench Hard Task Accuracy (task={task.value}): {task_accuracy}"
            )
            scores_row.append((task.value, task_accuracy))

        # Calculate overall accuracy
        overall_accuracy = (
            overall_correct_predictions / overall_total_predictions
        )
        print(f"Overall Big Bench Hard Accuracy: {overall_accuracy}")

        # Create a DataFrame from task_results_data
        # Columns: 'Task', 'Input', 'Prediction', 'Score'
        self.predictions = pd.DataFrame(
            predictions_row, columns=["Task", "Input", "Prediction", "Correct"]
        )
        self.task_scores = pd.DataFrame(scores_row, columns=["Task", "Score"])
        self.overall_score = overall_accuracy

        return overall_accuracy

    def predict(
        self, model: DeepEvalBaseLLM, task: BigBenchHardTask, golden: Golden
    ) -> Dict:
        # Define prompt template
        prompt: str = BigBenchHardTemplate.generate_output(
            input=golden.input,
            task=task,
            n_shots=self.n_shots,
            enable_cot=self.enable_cot,
        )
        pydantic_model = bbh_models_dict[task.value]
        try:
            print("abc")
            res = model.generate(prompt=prompt, schema=pydantic_model)
            prediction = str(res.answer)
        except Exception as e:
            print("omg")
            prompt += bbh_confinement_statements_dict[task.value]
            prediction = str(model.generate(prompt))

        if isinstance(prediction, tuple):
            prediction = prediction[0]

        # Define Metric
        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )
        return {"prediction": prediction, "score": score}

    def batch_predict(
        self,
        model: DeepEvalBaseLLM,
        task: BigBenchHardTask,
        goldens: List[Golden],
    ) -> List[Dict]:
        prompts = []
        for golden in goldens:
            prompt: dict = BigBenchHardTemplate.generate_output(
                input=golden.input,
                task=task,
                n_shots=self.n_shots,
                enable_cot=self.enable_cot,
            )
            prompts.append(prompt)

        # Enforced model generation
        try:
            responses: List[NumberModel] = model.batch_generate(
                prompts=prompts, schemas=[NumberModel for i in prompts]
            )
            predictions = [res.answer for res in responses]
        except TypeError:
            prompts = [
                prompt + "Make sure to output only the numerical answer."
                for prompt in prompts
            ]
            predictions = model.batch_generate(prompts)
            predictions = [str(pred) for pred in predictions]

        if len(predictions) is not len(goldens):
            raise ValueError(
                "Custom `batch_generate` method did not return the same number of generations as the number of prompts."
            )

        res = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            prediction = prediction.split()[-1]
            prediction = prediction[:-1] if self.enable_cot else prediction
            golden = goldens[i]

            # Define Metric
            score = self.scorer.exact_match_score(
                golden.expected_output, prediction
            )
            res.append({"prediction": prediction, "score": score})

        return res

    def load_benchmark_dataset(self, task: BigBenchHardTask) -> List[Golden]:
        if self.dataset:
            dataset = self.dataset
        else:
            # load from hugging face
            dataset = load_dataset("lukaemon/bbh", task.value)

        goldens: List[Golden] = []
        for data in dataset["test"]:
            golden = Golden(input=data["input"], expected_output=data["target"])
            goldens.append(golden)

        return goldens
