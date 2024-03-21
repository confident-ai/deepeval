from typing import List, Dict, Optional
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.truthful_q.task import TruthfulQATask
from deepeval.benchmarks.truthful_q.template import TruthfulQATemplate
from deepeval.scorer import Scorer


class TruthfulQA(DeepEvalBaseBenchmark):

    def __init__(self, tasks: List[TruthfulQATask] = None, n_shots: int = 10):
        assert n_shots <= 15, "HellaSwag only supports n_shots <= 15."
        super().__init__()
        self.tasks: List[TruthfulQATask] = (
            list(TruthfulQATask) if tasks is None else tasks
        )
        self.scorer = Scorer()
        self.gen_dataset: Dataset = None
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
            goldens = self.load_benchmark_dataset(task)
            task_correct_predictions = 0
            task_total_predictions = len(goldens)
            overall_total_predictions += len(goldens)

            # Calculate task accuracy
            for golden in tqdm(goldens, desc=f"Processing {task.value}"):
                prediction, score = self.predict(model, task, golden).values()
                if score:
                    task_correct_predictions += 1
                    overall_correct_predictions += 1
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
        self, model: DeepEvalBaseLLM, task: TruthfulQATask, golden: Golden, mc_mode:bool
    ) -> Dict:
        # Define prompt template
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."
        prompt: dict = TruthfulQATemplate.generate_output(
            input=golden.input,
            task=task,
            n_shots=self.n_shots,
        )
        prediction = model.generate(prompt)[0]

        # Define Metric
        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )
        return {"prediction": prediction, "score": score}

    def load_benchmark_dataset(self, task: TruthfulQATask, mc_mode:bool) -> List[Golden]:

        # Load (cached) datasets from HF based on mode ('mc' and 'gen')
        if self.gen_dataset is None:
            gen_dataset = load_dataset("truthful_qa", 'generation')['validation']
            self.gen_dataset = gen_dataset
        else:
            gen_dataset = self.gen_dataset 
        if mc_mode:
            if self.mc_dataset is None:
                mc_dataset = load_dataset("truthful_qa", 'multiple_choice')['validation']
                df_mc, df_gen = mc_dataset.to_pandas(), gen_dataset.to_pandas()
                merged_df = pd.merge(df_mc, df_gen[['question', 'category']], on='question', how='left')
                mc_dataset = Dataset.from_pandas(merged_df)
                self.mc_dataset = mc_dataset
            else:
                mc_dataset = self.mc_dataset

        # Construct test set (using validation here because HellaSwag
        # does not provide outputs for test set in HF dataset)
      
        if mc_mode:
            dataset = mc_dataset.filter(lambda data: data["category"] == task.value)
        else:
            dataset = gen_dataset.filter(lambda data: data["category"] == task.value)
    
        choices = ["A", "B", "C", "D"]
        goldens: List[Golden] = []
        for data in dataset:
            if mc_mode:
                expected_output_index = data["mc1_targets"]["labels"].index(1)
                expected_output = choices[expected_output_index]
                golden = Golden(input=data['question'], expectedOutput=expected_output)
            else:
                golden = Golden(input=data['question'])
            goldens.append(golden)
        

        return goldens

benchmark = TruthfulQA()
benchmark.load_benchmark_dataset(TruthfulQATask.LANGUAGE, mc_mode=True)