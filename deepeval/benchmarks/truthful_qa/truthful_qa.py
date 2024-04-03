from typing import List, Dict, Optional
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.truthful_qa.task import TruthfulQATask
from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
from deepeval.benchmarks.truthful_qa.template import TruthfulQATemplate
from deepeval.scorer import Scorer


class TruthfulQA(DeepEvalBaseBenchmark):

    def __init__(
        self,
        tasks: List[TruthfulQATask] = None,
        mode: TruthfulQAMode = TruthfulQAMode.MC1,
    ):
        super().__init__()
        self.tasks: List[TruthfulQATask] = (
            list(TruthfulQATask) if tasks is None else tasks
        )
        self.mode: TruthfulQAMode = mode
        self.scorer = Scorer()
        self.mc_dataset: Dataset = None

        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None

    def evaluate(self, model: DeepEvalBaseLLM) -> Dict:
        overall_correct_predictions = 0
        overall_total_predictions = 0
        predictions_row = []
        scores_row = []

        for task in self.tasks:
            goldens = self.load_benchmark_dataset(task, self.mode)
            task_correct_predictions = 0
            task_total_predictions = len(goldens)
            overall_total_predictions += len(goldens)

            # Calculate task accuracy
            for golden in tqdm(goldens, desc=f"Processing {task.value}"):
                prediction, score = self.predict(
                    model, golden, self.mode
                ).values()
                if score:
                    task_correct_predictions += score
                    overall_correct_predictions += score
                predictions_row.append(
                    (task.value, golden.input, prediction, score)
                )
            task_accuracy = task_correct_predictions / task_total_predictions
            print(
                f"TruthfulQA Task Accuracy (task={task.value}): {task_accuracy}"
            )
            scores_row.append((task.value, task_accuracy))

        # Calculate overall accuracy
        overall_accuracy = (
            overall_correct_predictions / overall_total_predictions
        )
        print(f"Overall TruthfulQA Accuracy: {overall_accuracy}")

        # Create a DataFrame from task_results_data
        # Columns: 'Task', 'Input', 'Prediction', 'Score'
        self.predictions = pd.DataFrame(
            predictions_row, columns=["Task", "Input", "Prediction", "Correct"]
        )
        self.task_scores = pd.DataFrame(scores_row, columns=["Task", "Score"])
        self.overall_score = overall_accuracy

        return overall_accuracy

    def predict(
        self, model: DeepEvalBaseLLM, golden: Golden, mode: TruthfulQAMode
    ) -> Dict:
        # Define prompt template
        prompt: dict = TruthfulQATemplate.generate_output(
            input=golden.input, mode=mode
        )
        prediction = model.generate(prompt)

        # Define Metric
        if mode == TruthfulQAMode.MC1:
            score = self.scorer.exact_match_score(
                prediction[0], golden.expected_output
            )

        if mode == TruthfulQAMode.MC2:
            prediction = model.generate(prompt)
            # Define Metric
            score = self.scorer.truth_identification_score(
                golden.expected_output, prediction
            )

        return {"prediction": prediction, "score": score}

    def load_benchmark_dataset(
        self, task: TruthfulQATask, mode: TruthfulQAMode
    ) -> List[Golden]:
        # Load full dataset
        if self.mc_dataset is None:
            gen_dataset = load_dataset("truthful_qa", "generation")[
                "validation"
            ]
            mc_dataset = load_dataset("truthful_qa", "multiple_choice")[
                "validation"
            ]
            df_mc, df_gen = mc_dataset.to_pandas(), gen_dataset.to_pandas()
            merged_df = pd.merge(
                df_mc,
                df_gen[["question", "category"]],
                on="question",
                how="left",
            )
            mc_dataset = Dataset.from_pandas(merged_df)
            self.mc_dataset = mc_dataset
        else:
            mc_dataset = self.mc_dataset
        dataset = self.mc_dataset.filter(
            lambda data: data["category"] == task.value
        )

        # Create goldens list from datset
        goldens: List[Golden] = []
        for data in dataset:
            if mode == TruthfulQAMode.MC1:
                input, expected_output = TruthfulQATemplate.format_mc1_question(
                    data
                )
                golden = Golden(input=input, expectedOutput=expected_output)
                goldens.append(golden)
            elif mode == TruthfulQAMode.MC2:
                input, expected_output = TruthfulQATemplate.format_mc2_question(
                    data
                )
                golden = Golden(
                    input=input, expectedOutput=str(expected_output)
                )
                goldens.append(golden)

        return goldens


########################
#### Example Usage #####
########################
from deepeval.models.gpt_model import GPTModel

model = GPTModel()
benchmark = TruthfulQA(
    [TruthfulQATask.ADVERTISING, TruthfulQATask.FICTION],
    mode=TruthfulQAMode.MC2,
)
benchmark.evaluate(model)

benchmark = TruthfulQA(
    [TruthfulQATask.ADVERTISING, TruthfulQATask.FICTION],
    mode=TruthfulQAMode.MC1,
)
benchmark.evaluate(model)
