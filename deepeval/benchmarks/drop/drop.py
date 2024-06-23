from typing import List, Optional, Dict
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.drop.task import DROPTask
from deepeval.benchmarks.drop.template import DROPTemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.scorer import Scorer

DELIMITER = ","


class DROP(DeepEvalBaseBenchmark):
    def __init__(
        self, tasks: List[DROPTask] = None, n_shots: int = 5, **kwargs
    ):
        assert n_shots <= 5, "DROP only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.tasks: List[DROPTask] = list(DROPTask) if tasks is None else tasks
        self.scorer = Scorer()
        self.shots_dataset: List[Dict] = None
        self.n_shots: int = n_shots
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

            # Calculate task accuracy
            if use_batch:
                for i in tqdm(
                    range(0, len(goldens), batch_size),
                    desc=f"Batch Processing {task.value} (batch_size={batch_size})",
                ):
                    goldens_batch = goldens[i : i + batch_size]
                    batch_predictions = self.batch_predict(model, goldens_batch)
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
                for golden in tqdm(goldens, desc=f"Processing {task.value}"):
                    prediction, score = self.predict(model, golden).values()
                    if score:
                        task_correct_predictions += 1
                        overall_correct_predictions += 1
                    predictions_row.append(
                        (task.value, golden.input, prediction, score)
                    )

            task_accuracy = task_correct_predictions / task_total_predictions
            print(f"DROP Task Accuracy (task={task.value}): {task_accuracy}")
            scores_row.append((task.value, task_accuracy))

        # Calculate overall accuracy
        overall_accuracy = (
            overall_correct_predictions / overall_total_predictions
        )
        print(f"Overall DROP Accuracy: {overall_accuracy}")

        # Create a DataFrame from task_results_data
        # Columns: 'Task', 'Input', 'Prediction', 'Score'
        self.predictions = pd.DataFrame(
            predictions_row, columns=["Task", "Input", "Prediction", "Correct"]
        )
        self.task_scores = pd.DataFrame(scores_row, columns=["Task", "Score"])
        self.overall_score = overall_accuracy

        return overall_accuracy

    def predict(self, model: DeepEvalBaseLLM, golden: Golden) -> Dict:
        # Define prompt template
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."
        prompt: dict = DROPTemplate.generate_output(
            train_set=self.shots_dataset,
            input=golden.input,
            type=golden.context[0],
            n_shots=self.n_shots,
        )
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
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."

        prompts = []
        for golden in goldens:
            prompt: dict = DROPTemplate.generate_output(
                train_set=self.shots_dataset,
                input=golden.input,
                type=golden.context[0],
                n_shots=self.n_shots,
            )
            prompts.append(prompt)

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

    def load_benchmark_dataset(self, task: DROPTask) -> List[Golden]:
        # cache dataset
        if self.dataset:
            dataset = self.dataset
        else:
            dataset = load_dataset("ucinlp/drop", trust_remote_code=True)
            self.dataset = dataset

        # construct example dataset
        if not self.shots_dataset:
            train_set = dataset["train"]
            shots_set = []
            categories_seen = set()
            for data in train_set:
                category = data["section_id"]
                if category not in categories_seen:
                    categories_seen.add(category)
                    shots_set.append(data)
            self.shots_dataset = shots_set

        val_set = dataset["validation"].filter(
            lambda data: data["section_id"] == task.value
        )

        # construct test set
        goldens: List[Golden] = []
        for data in val_set:
            input = DROPTemplate.format_question(data, include_answer=False)
            output = DELIMITER.join(tuple(data["answers_spans"]["spans"][0]))
            output_type = data["answers_spans"]["types"][0]
            golden = Golden(
                input=input, expected_output=output, context=[output_type]
            )
            goldens.append(golden)

        return goldens
