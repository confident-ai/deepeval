from typing import List
from datasets import load_dataset
from tqdm import tqdm  
from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.mmlu.template import MMLUTemplate
from deepeval.scorer import Scorer

# TODO: to be deleted
from deepeval.models import GPTModel

class MMLU(DeepEvalBaseBenchmark):
    def __init__(self, tasks: List[MMLUTask] = None, n_shots: int = 5):
        assert n_shots <= 5, "MMLU only supports n_shots <= 5"
        super().__init__()
        self.tasks: List[MMLUTask] = tasks
        self.scorer = Scorer()
        self.dataset: Dataset = None
        self.shots_dataset: List[dict] = None
        self.n_shots: int = n_shots
        

    def evaluate(self, model: DeepEvalBaseLLM) -> dict:
        overall_correct_predictions = 0
        overall_total_predictions = 0
        task_results: dict[str, dict[str, List | float]] = {
            task.value: {'predictions': [], 'scores': [], 'accuracy': 0.0} for task in self.tasks
        }
        results = {'predictions': [], 'scores': [], 'accuracy': 0.0}

        for task in self.tasks:
            goldens = self.load_benchmark_dataset(task)
            task_correct_predictions = 0
            task_total_predictions = len(goldens)
            task_predictions = []
            task_scores = []

            # Calculate task accuracy
            for golden in tqdm(goldens, desc=f'Processing {task.value}'):
                prediction, score = self.predict(model, task, golden).values()
                if score:
                    task_correct_predictions += 1
                task_predictions.append(prediction)
                task_scores.append(score)
            task_accuracy = task_correct_predictions / task_total_predictions

            # Update task_results with predictions, scores, accuracy
            task_results[task.value]['predictions'] = task_predictions
            task_results[task.value]['scores'] = task_scores
            task_results[task.value]['accuracy'] = task_accuracy
            print(f"MMLU Task Accuracy (task={task.value}): {task_accuracy}")

            # Update overall predictions, scores
            overall_correct_predictions += task_correct_predictions
            overall_total_predictions += task_total_predictions
            results['predictions'].extend(task_predictions)
            results['scores'].extend(task_scores)

        # Calculate overall accuracy
        overall_accuracy = overall_correct_predictions / overall_total_predictions
        results['accuracy'] = overall_accuracy  
        print(f"Overall MMLU Accuracy: {overall_accuracy}")
        return {"results": results, "task_results": task_results}


    def predict(self, model: DeepEvalBaseLLM, task: MMLUTask, golden: Golden) -> dict:
        # Define prompt template
        assert self.shots_dataset != None, "Example dataset is empty. Call load_benchmark."
        prompt: dict = MMLUTemplate.generate_output(
            train_set=self.shots_dataset, input=golden.input, task=task, n_shots=self.n_shots
        )
        prediction = model(prompt)
    
        # Define Metric
        score = self.scorer.exact_match_score(golden.expected_output, prediction)
        return {'prediction': prediction, 'score': score}

    def load_benchmark_dataset(self, task: MMLUTask) -> List[Golden]:
        # If dataset has been previously loaded, load from 
        # instance var (to save time)
        if self.dataset:
            dataset = self.dataset
        else:
            dataset = load_dataset("lukaemon/mmlu", task.value, trust_remote_code=True)
            self.dataset = dataset

        # If dataset has not been previously loaded, construct 
        # dataset of examples and save as instance var (to save time)
        if not self.shots_dataset:
            train_set = dataset['train']
            shots_set = []
            for data in train_set:
                shots_set.append(data)
            self.shots_dataset = shots_set
            
        # Construct test set
        goldens: List[Golden] = []
        for data in dataset["test"]:
            input = MMLUTemplate.format_question(data, include_answer=False)
            golden = Golden(input=input, expectedOutput=data["target"])
            goldens.append(golden)
        return goldens
        

#############################
######## Example Usage ######
#############################

benchmark = MMLU(tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY])
results = benchmark.evaluate(model=GPTModel())
print(results)
print(benchmark.task_results)
