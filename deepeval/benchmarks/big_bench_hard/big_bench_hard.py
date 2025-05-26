from typing import List, Optional, Dict
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
from deepeval.benchmarks.big_bench_hard.template import BigBenchHardTemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.benchmarks.schema import *
from deepeval.telemetry import capture_benchmark_run

bbh_confinement_statements_dict = {
    BigBenchHardTask.BOOLEAN_EXPRESSIONS: "\n\nOutput 'True' or 'False'. Full answer not needed.",
    BigBenchHardTask.CAUSAL_JUDGEMENT: "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    BigBenchHardTask.DATE_UNDERSTANDING: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', or '(F)'. Full answer not needed.",
    BigBenchHardTask.DISAMBIGUATION_QA: "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    BigBenchHardTask.DYCK_LANGUAGES: "\n\nOutput only the sequence of parentheses characters separated by white space. Full answer not needed.",
    BigBenchHardTask.FORMAL_FALLACIES: "\n\nOutput 'invalid' or 'valid'. Full answer not needed.",
    BigBenchHardTask.GEOMETRIC_SHAPES: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', or '(K)'. Full answer not needed.",
    BigBenchHardTask.HYPERBATON: "\n\nOutput '(A)' or'(B)'. Full answer not needed.",
    BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS: "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    BigBenchHardTask.LOGICAL_DEDUCTION_FIVE_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.LOGICAL_DEDUCTION_SEVEN_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', or '(G)'. Full answer not needed.",
    BigBenchHardTask.MOVIE_RECOMMENDATION: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.MULTISTEP_ARITHMETIC_TWO: "\n\nOutput the numerical answer. Full answer not needed.",
    BigBenchHardTask.NAVIGATE: "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    BigBenchHardTask.OBJECT_COUNTING: "\n\nOutput the numerical answer. Full answer not needed.",
    BigBenchHardTask.PENGUINS_IN_A_TABLE: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.REASONING_ABOUT_COLORED_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', or '(R)'. Full answer not needed.",
    BigBenchHardTask.RUIN_NAMES: "\n\nOutput '(A)', '(B)', '(C)', or '(D)'. Full answer not needed.",
    BigBenchHardTask.SALIENT_TRANSLATION_ERROR_DETECTION: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', or '(F)'. Full answer not needed.",
    BigBenchHardTask.SNARKS: "\n\nOutput '(A)' or'(B)'. Full answer not needed.",
    BigBenchHardTask.SPORTS_UNDERSTANDING: "\n\nOutput 'yes' or 'no'. Full answer not needed.",
    BigBenchHardTask.TEMPORAL_SEQUENCES: "\n\nOutput '(A)', '(B)', '(C)', or '(D)'. Full answer not needed.",
    BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS: "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', or '(G)'. Full answer not needed.",
    BigBenchHardTask.WEB_OF_LIES: "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    BigBenchHardTask.WORD_SORTING: "\n\nOutput only the sequence of words separated by white space. Full answer not needed.",
}


class BigBenchHard(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[BigBenchHardTask] = None,
        n_shots: int = 3,
        enable_cot: bool = True,
        n_problems_per_task: Optional[int] = None,
        verbose_mode: bool = False,
        confinement_instructions_dict: Optional[
            Dict[BigBenchHardTask, str]
        ] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer
        import pandas as pd

        assert n_shots <= 3, "BBH only supports n_shots <= 3"
        super().__init__(**kwargs)
        self.tasks: List[BigBenchHardTask] = (
            list(BigBenchHardTask) if tasks is None else tasks
        )
        self.n_problems_per_task: Optional[int] = n_problems_per_task
        self.scorer = Scorer()
        self.n_shots: int = n_shots
        self.enable_cot: bool = enable_cot
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode: bool = verbose_mode
        if not confinement_instructions_dict:
            self.confinement_instructions_dict = bbh_confinement_statements_dict
        else:
            self.confinement_instructions_dict = confinement_instructions_dict

    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        import pandas as pd

        with capture_benchmark_run("Big Bench Hard", len(self.tasks)):
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
                                (
                                    task.value,
                                    golden.input,
                                    prediction,
                                    golden.expected_output,
                                    score,
                                )
                            )
                else:
                    # Calculate task accuracy
                    for idx, golden in enumerate(
                        tqdm(goldens, desc=f"Processing {task.value}")
                    ):
                        prediction, score = self.predict(
                            model, task, golden
                        ).values()
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
            res = model.generate(prompt=prompt, schema=pydantic_model)
            prediction = str(res.answer)
        except (AttributeError, TypeError):
            prompt += self.confinement_instructions_dict[task]
            prediction = model.generate(prompt)

        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = str(prediction)

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
            pydantic_model = bbh_models_dict[task.value]
            responses: List = model.batch_generate(
                prompts=prompts, schemas=[pydantic_model for i in prompts]
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
        from datasets import load_dataset

        dataset_mapping = {
            task: f"{task.value}_dataset" for task in BigBenchHardTask
        }
        dataset_attr = dataset_mapping.get(task)
        if dataset_attr:
            if not hasattr(self, dataset_attr):
                dataset = load_dataset(
                    "lukaemon/bbh", task.value, trust_remote_code=True
                )
                setattr(self, dataset_attr, dataset)
            else:
                dataset = getattr(self, dataset_attr)

        goldens: List[Golden] = []
        for data in dataset["test"]:
            golden = Golden(input=data["input"], expected_output=data["target"])
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
