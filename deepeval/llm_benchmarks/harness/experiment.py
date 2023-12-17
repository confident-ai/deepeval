from typing import List, Union
from pydantic import BaseModel
from deepeval.llm_benchmarks.harness.task import HarnessTasks
from deepeval.llm_benchmarks.harness.evaluation import (
    HarnessEvaluate,
    HarnessConfig,
)
from deepeval.llm_benchmarks.base_experiment import BaseEvaluationExperiment


class HarnessExperiment(BaseEvaluationExperiment):
    def __init__(self, experiment_name: str, experiment_desc: str) -> None:
        if len(experiment_name.split('/')) != 2 and experiment_name.split('/')[0] != 'harness':
            raise ValueError('Need proper experiment naming. Example: harness/<experiment-name>')
        
        super().__init__(experiment_name=experiment_name, experiment_desc=experiment_desc)
        
        self.results = {}
        self.tasks = [] 
        self.config, self.experiment = None, None
    
    def create(self, config: HarnessConfig, *args, **kwargs):
        self.config = config
        self.config.log_samples = True
        self.config.write_out = self.experiment_folder
        self.experiment = HarnessEvaluate(harness_config=config)
    
    def update(self, updated_config: BaseModel, *args, **kwargs):
        self.create(config=updated_config)
        
    def run(self, tasks: str | List[str], *args, **kwargs):
        assert self.config is not None, ValueError("Config is None, please create the experiment first.")
        assert self.experiment is not None, ValueError("Experiment not initialized. run experiment.create method.")
        assert tasks is not None, ValueError("Tasks can not be None, please provide either a task in string or a list of tasks.")
        
        if isinstance(tasks, str):
            self.tasks = [tasks]
        
        results = self.experiment.evaluate(tasks=self.tasks)
        return results
    
        