from typing import List
from pydantic import BaseModel
from deepeval.experimental.harness.evaluation import (
    HarnessEvaluate,
    HarnessConfig,
)
from deepeval.experimental.base_experiment import BaseEvaluationExperiment


class HarnessExperiment(BaseEvaluationExperiment):
    def __init__(self, experiment_name: str, experiment_desc: str) -> None:
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

        self.results = {}
        self.tasks = []
        self.config, self.experiment = None, None
        
        # write the desc files

    def create(self, config: HarnessConfig, *args, **kwargs):
        self.config = config
        self.config.log_samples = True
        self.config.output_path = self.experiment_folder
        
        # put a mock task
        self.config.tasks = None 
        self.experiment = HarnessEvaluate(harness_config=config)

    def update(self, updated_config: BaseModel, *args, **kwargs):
        self.create(config=updated_config)

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
