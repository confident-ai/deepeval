import os
import json
import difflib
import pandas as pd
from typing import Optional, Union
from pathlib import PosixPath
from pydantic import BaseModel, Field

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from typing import List
from deepeval.experimental.harness.evaluation import (
    HarnessEvaluate,
    HarnessConfig,
    HarnessTasks,
)
from deepeval.experimental.base_experiment import BaseEvaluationExperiment
from deepeval.metrics import ExactMatchAccuracyMetric


class EvaluationResult(BaseModel):
    task_name: str = Field(..., description="The name of the task")
    saved_path: Union[str, PosixPath] = Field(
        ..., description="The evaluation csv path."
    )
    accuracy: float = Field(
        ...,
        description="The final value of the number of test cases passed out of total number of test cases.",
    )


class EvaluationResult(BaseEvaluationExperiment):
    def __init__(self, task_name: str, task_limit: int) -> None:
        self.task_name, self.task_limit = task_name, task_limit
    
    def visualize_result(self, sub_task: Optional[Union[str, List[str]]]=None) -> None:
        raise NotImplementedError

    def collect_results_and_push(self, push_to_hub: Optional[bool] = False):
        all_task_data = HarnessTasks.get_dataset_from_task(
            task_names=self.task_name, task_limit=self.task_limit
        )
        all_task_results = {}
        all_jsonl = os.listdir(self.experiment_folder)

        for task_name, task_data in all_task_data.items():
            test_cases = []
            # todo: this is not applicable, need to find the string which matches

            try:
                closest_match = difflib.get_close_matches(
                    task_name, all_jsonl, n=1
                )
                task_jsonl = closest_match[0]

                task_evaluation_dict = json.load(
                    open(self.experiment_folder / task_jsonl, "r")
                )
                all_responses, is_correct_overall = [], []
                # also collect the accuracy
                scores = []
                all_prompts, all_targets = [], []

                for eval_dict in task_evaluation_dict:
                    responses = eval_dict["filtered_resps"]
                    filtered_responses, is_response_correct = zip(*responses)
                    all_responses.extend(list(filtered_responses))
                    is_correct_overall.extend(list(is_response_correct))
                    scores.append(eval_dict["acc"])

                for _, prompt, target, response in zip(
                    task_data["doc_id"],
                    task_data["prompt"],
                    task_data["target"],
                    all_responses,
                ):
                    test_case = LLMTestCase(
                        input=prompt,
                        actual_output=str(response),
                        expected_output=str(target),
                    )
                    test_cases.append(test_case)
                    all_prompts.append(prompt)
                    all_targets.append(target)

                dataset = EvaluationDataset(test_cases=test_cases)
                # Provide an alias when pushing a dataset
                dataset.evaluate(
                    metrics=[ExactMatchAccuracyMetric(minimum_score=0.5)]
                )

                if push_to_hub:
                    # this is very unstable, for each task it opens a new window in confident-ai.
                    dataset.push(alias=task_name)

                # do not save in the memory
                pd.DataFrame(
                    {
                        "id": list(range(1, len(all_prompts) + 1)),
                        "prompt": all_prompts,
                        "target": all_targets,
                        "response": all_responses,
                        "is_correct": is_correct_overall,
                    }
                ).to_csv(self.evaluation_csvs_folder / f"{task_name}.csv"),

                all_task_results[task_name] = EvaluationResult(
                    task_name=task_name,
                    saved_path=self.evaluation_csvs_folder / f"{task_name}.csv",
                    accuracy=sum(scores) / len(scores),
                )
            except Exception as e:
                print(f"Task {task_name} not found or not run.\nError: {e}")
                continue
        return all_task_results


class HarnessExperiment(BaseEvaluationExperiment):
    def __init__(
        self, experiment_name: str, experiment_desc: str, config: HarnessConfig
    ) -> None:
        if (
            len(experiment_name.split("/")) != 2
            and experiment_name.split("/")[0] != "harness"
        ):
            raise ValueError(
                "Need proper experiment naming. Example: harness/<experiment-name>"
            )

        super().__init__(
            experiment_name=experiment_name, experiment_desc=experiment_desc
        )

        self.tasks = []
        self.config, self.experiment = None, None

        # create the experiment
        self.config = config
        self.config.log_samples = True
        self.config.output_path = self.experiment_folder
        self.experiment = HarnessEvaluate(harness_config=config)

        # write the desc files
        desc_file_path = self.experiment_folder / "desc.txt"
        with open(desc_file_path, "w") as desc_file:
            desc_file.write(experiment_desc)

    def update(self, updated_config: HarnessConfig, *args, **kwargs):
        self.config = updated_config
        self.config.log_samples = True
        self.config.output_path = self.experiment_folder
        self.experiment = HarnessEvaluate(harness_config=self.config)

    def run(self, tasks: str | List[str], *args, **kwargs):
        assert self.config is not None, ValueError(
            "Config is None, please create the experiment first."
        )
        assert self.experiment is not None, ValueError(
            "Experiment not initialized. run experiment.create method."
        )
        assert tasks is not None, ValueError(
            "Tasks can not be None, please provide either a task in string or a list of tasks."
        )

        self.experiment.harness_config.tasks = (
            [tasks] if isinstance(tasks, str) else tasks
        )
        self.tasks = [tasks] if isinstance(tasks, str) else tasks

        _ = self.experiment.evaluate(tasks=self.tasks)
        return EvaluationResult(
            task_name=self.tasks[0], task_limit=self.config.limit
        )
