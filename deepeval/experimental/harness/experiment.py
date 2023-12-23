import json
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from typing import List
from deepeval.experimental.harness.evaluation import (
    HarnessEvaluate,
    HarnessConfig,
    HarnessTasks
)
from deepeval.experimental.base_experiment import BaseEvaluationExperiment
from deepeval.metrics import ExactMatchAccuracyMetric


class HarnessExperiment(BaseEvaluationExperiment):
    def __init__(self, experiment_name: str, experiment_desc: str, config: HarnessConfig) -> None:
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
        desc_file_path = self.experiment_folder / 'desc.txt'
        with open(desc_file_path, 'w') as desc_file:
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
        
        self.experiment.harness_config.tasks = [tasks] if isinstance(tasks, str) else tasks
        self.tasks = [tasks] if isinstance(tasks, str) else tasks
        
        results = self.experiment.evaluate(tasks=self.tasks)
        return results

    def collect_results_and_push(self):
        # todo: for now just passing string, need to figure out how to pass list
        all_task_data = HarnessTasks.get_dataset_from_task(task_names=self.tasks[0], task_limit=self.config.limit)
        all_datasets = {}
        for task_name, task_data in all_task_data.items():
            test_cases = [] 
            for _, prompt, target in zip(task_data['doc_id'], task_data['prompt'], task_data['target']):
                # todo: change actual_output by parsing it from the output 
                test_case = LLMTestCase(
                    input = prompt,
                    actual_output=str(target),
                    expected_output=str(target)
                )
                test_cases.append(test_case)
            
            dataset = EvaluationDataset(test_cases=test_cases)
            all_datasets[task_name] = dataset
            dataset.evaluate(metrics=[ExactMatchAccuracyMetric(minimum_score=0.5)])
        # todo: also upload the overall thing metrics to harness
        