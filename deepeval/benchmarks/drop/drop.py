from typing import List, Optional, Dict
from tqdm import tqdm
from typing import Union

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.drop.task import DROPTask
from deepeval.benchmarks.drop.template import DROPTemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.benchmarks.schema import (
    DROPDateSchema,
    DROPNumberSchema,
    DROPStringSchema,
)
from deepeval.telemetry import capture_benchmark_run

DELIMITER = ","


class DROP(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[DROPTask] = None,
        n_shots: int = 5,
        n_problems_per_task: Optional[int] = None,
        verbose_mode: bool = False,
        **kwargs,
    ):
        from deepeval.scorer import Scorer
        import pandas as pd

        assert n_shots <= 5, "DROP only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.tasks: List[DROPTask] = list(DROPTask) if tasks is None else tasks
        self.n_problems_per_task: Optional[int] = n_problems_per_task

        self.scorer = Scorer()
        self.shots_dataset: List[Dict] = None
        self.n_shots: int = n_shots
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode: bool = verbose_mode

    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        import pandas as pd

        with capture_benchmark_run("DROP", len(self.tasks)):
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
                    f"DROP Task Accuracy (task={task.value}): {task_accuracy}"
                )
                scores_row.append((task.value, task_accuracy))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall DROP Accuracy: {overall_accuracy}")

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
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."
        prompt: dict = DROPTemplate.generate_output(
            train_set=self.shots_dataset,
            input=golden.input,
            n_shots=self.n_shots,
        )

        # Enforced model generation
        type_info = golden.context[0]
        try:
            if type_info == "number":
                schema = DROPNumberSchema
            elif type_info == "date":
                schema = DROPDateSchema
            elif type_info == "span":
                schema = DROPStringSchema
            res: Union[DROPNumberSchema, DROPDateSchema, DROPStringSchema] = (
                model.generate(prompt=prompt, schema=schema)
            )
            prediction = str(res.answer)
        except TypeError:
            prompt += f"Output should be a {type_info}. No explanation needed."
            prediction = model.generate(prompt)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = str(prediction)

        # Define Metric
        expected_output = DROPTemplate.parse_str_to_list(
            golden.expected_output, DELIMITER
        )
        score = self.scorer.quasi_contains_score(expected_output, prediction)
        return {"prediction": prediction, "score": score}

    def batch_predict(
        self, model: DeepEvalBaseLLM, goldens: List[Golden]
    ) -> List[Dict]:
        # Define prompt template
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."

        prompts = []
        schemas = []
        for golden in goldens:
            prompt: dict = DROPTemplate.generate_output(
                train_set=self.shots_dataset,
                input=golden.input,
                type=golden.context[0],
                n_shots=self.n_shots,
            )
            prompts.append(prompt)
            output_type = golden.context[0]
            if output_type == "number":
                schema = DROPNumberSchema
            elif output_type == "date":
                schema = DROPDateSchema
            elif output_type == "span":
                schema = DROPStringSchema
            schemas.append(schema)

        try:
            responses: List[
                Union[DROPNumberSchema, DROPDateSchema, DROPStringSchema]
            ] = model.batch_generate(prompts=prompts, schemas=schemas)
            predictions = [str(res.answer) for res in responses]
        except TypeError:
            prompts = [
                prompt
                + "Output should be of type {type}. No explanation needed.".format(
                    type=type
                )
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
            score = self.scorer.quasi_exact_match_score(
                golden.expected_output, prediction
            )
            res.append({"prediction": prediction, "score": score})

        return res

    def load_benchmark_dataset(self, task: DROPTask) -> List[Golden]:
        from datasets import load_dataset

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
            output = DROPTemplate.parse_list_to_str(
                data["answers_spans"]["spans"], DELIMITER
            )
            output_type = data["answers_spans"]["types"][0]
            golden = Golden(
                input=input, expected_output=output, context=[output_type]
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
            f"Score: {score}\nPrediction: {prediction}\nAccepted Expected Output: {expected_output}",
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
