from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.truthful_qa.task import TruthfulQATask
from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
from deepeval.benchmarks.truthful_qa.template import TruthfulQATemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.benchmarks.schema import NumberSchema, ListOfNumbersSchema
from deepeval.telemetry import capture_benchmark_run

truthful_qa_confinement_statements_dict = {
    TruthfulQAMode.MC1: "\n\nOutput '1', '2', '3', '4', '5' etc. (number in front of answer choice). Full answer not needed.",
    TruthfulQAMode.MC2: "\n\nOutput the indices of all correct answers as a python list (e.g. '[1, 3, 4]'). Full answers are not needed.",
}


class TruthfulQA(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[TruthfulQATask] = None,
        mode: TruthfulQAMode = TruthfulQAMode.MC1,
        n_problems_per_task: Optional[int] = None,
        verbose_mode: bool = False,
        confinement_instructions_dict: Optional[
            Dict[TruthfulQAMode, str]
        ] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer
        from datasets import Dataset

        super().__init__(**kwargs)
        self.tasks: List[TruthfulQATask] = (
            list(TruthfulQATask) if tasks is None else tasks
        )
        self.n_problems_per_task: Optional[int] = n_problems_per_task
        self.mode: TruthfulQAMode = mode
        self.scorer = Scorer()
        self.mc_dataset: Dataset = self.dataset
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode: bool = verbose_mode
        if not confinement_instructions_dict:
            self.confinement_instructions_dict = (
                truthful_qa_confinement_statements_dict
            )
        else:
            self.confinement_instructions_dict = confinement_instructions_dict

    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        with capture_benchmark_run("TruthfulQA", len(self.tasks)):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            scores_row = []
            use_batch = should_use_batch(model, batch_size)

            for task in self.tasks:
                goldens = self.load_benchmark_dataset(task, self.mode)
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
                            model, goldens_batch, self.mode
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
                    for idx, golden in enumerate(
                        tqdm(goldens, desc=f"Processing {task.value}")
                    ):
                        prediction, score = self.predict(
                            model, golden, self.mode
                        ).values()
                        if score:
                            task_correct_predictions += score
                            overall_correct_predictions += score
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

    def predict(
        self, model: DeepEvalBaseLLM, golden: Golden, mode: TruthfulQAMode
    ) -> Dict:
        # Define prompt template
        prompt: dict = TruthfulQATemplate.generate_output(
            input=golden.input, mode=mode
        )

        # Enforced model generation
        try:
            if mode == TruthfulQAMode.MC1:
                res: NumberSchema = model.generate(
                    prompt=prompt, schema=NumberSchema
                )
                prediction = str(res.answer)
            elif mode == TruthfulQAMode.MC2:
                res: ListOfNumbersSchema = model.generate(
                    prompt=prompt, schema=ListOfNumbersSchema
                )
                prediction = str(res.answer)

        except TypeError:
            prompt += self.confinement_instructions_dict[mode]
            prediction = model.generate(prompt)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = str(prediction)

        # Define Metric
        if mode == TruthfulQAMode.MC1:
            score = self.scorer.exact_match_score(
                golden.expected_output, prediction
            )
        elif mode == TruthfulQAMode.MC2:
            score = self.scorer.truth_identification_score(
                golden.expected_output, prediction
            )

        return {"prediction": prediction, "score": score}

    def batch_predict(
        self,
        model: DeepEvalBaseLLM,
        goldens: List[Golden],
        mode: TruthfulQAMode,
    ) -> List[Dict]:
        # Define prompt template
        prompts = []
        for golden in goldens:
            prompt: dict = TruthfulQATemplate.generate_output(
                input=golden.input, mode=mode
            )
            prompts.append(prompt)
        # Enforced model generation
        try:
            if mode == TruthfulQAMode.MC1:
                responses: List[NumberSchema] = model.batch_generate(
                    prompts=prompts, schemas=[NumberSchema for i in prompts]
                )
                predictions = [str(res.answer) for res in responses]
            elif mode == TruthfulQAMode.MC2:
                responses: List[ListOfNumbersSchema] = model.batch_generate(
                    prompts=prompts,
                    schemas=[ListOfNumbersSchema for i in prompts],
                )
                predictions = [str(res.answer) for res in responses]

        except TypeError:
            if mode == TruthfulQAMode.MC1:
                prompts = [
                    prompt
                    + "\n\nOutput '1', '2', '3', '4', '5' etc. (number in front of answer choice). Full answer not needed."
                    for prompt in prompts
                ]
            elif mode == TruthfulQAMode.MC2:
                prompts = [
                    prompt
                    + "\n\nOutput the indices of all correct answers as a python list (e.g. '[1, 3, 4]'). Full answers are not needed."
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
            golden = goldens[i]
            # Define Metric
            if mode == TruthfulQAMode.MC1:
                score = self.scorer.exact_match_score(
                    golden.expected_output, prediction
                )
            elif mode == TruthfulQAMode.MC2:
                score = self.scorer.truth_identification_score(
                    golden.expected_output, prediction
                )
            res.append({"prediction": prediction, "score": score})

        return res

    def load_benchmark_dataset(
        self, task: TruthfulQATask, mode: TruthfulQAMode
    ) -> List[Golden]:
        from datasets import load_dataset, Dataset

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
                golden = Golden(input=input, expected_output=expected_output)
                goldens.append(golden)
            elif mode == TruthfulQAMode.MC2:
                input, expected_output = TruthfulQATemplate.format_mc2_question(
                    data
                )
                golden = Golden(
                    input=input, expected_output=str(expected_output)
                )
                goldens.append(golden)

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
