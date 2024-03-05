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
from deepeval.experimental.base_experiment import BaseEvaluationExperiment
from deepeval.metrics import ExactMatchAccuracyMetric
from deepeval.experimental.harness.models import DeepEvalHarnessModel
from deepeval.experimental.harness.config import (
    GeneralConfig,
    APIEndpointConfig,
)

# TODO: Write the tests results in eval_csvs
# TODO: Write Tasks methods from easy_eval


class HarnessExperiment(BaseEvaluationExperiment):
    def __init__(
        self,
        experiment_name: str,
        model_name_or_path: str,
        model_backend: str,
        experiment_desc: Optional[str] = None,
        config: Optional[Union[GeneralConfig, APIEndpointConfig]] = None,
        **kwargs,
    ) -> None:
        if (
            len(experiment_name.split("/")) != 2
            and experiment_name.split("/")[0] != "harness"
        ):
            raise ValueError(
                "Need proper experiment naming. Example: harness/<experiment-name>"
            )

        if experiment_desc is None:
            experiment_desc = f"Experiment Name: {experiment_name}, Model: {model_name_or_path} running on backend: {model_backend}"

        super().__init__(
            experiment_name=experiment_name, experiment_desc=experiment_desc
        )

        if config is None:
            if model_backend == "openai":
                self.config = APIEndpointConfig()
            else:
                self.config = GeneralConfig()
        else:
            self.config = config

        self.model_name_or_path, self.model_backend = (
            model_name_or_path,
            model_backend,
        )

        # create the experiment
        self.config.log_samples = True
        self.config.output_path = self.experiment_folder
        self.evaluator = DeepEvalHarnessModel(
            model_name_or_path=self.model_name_or_path,
            model_backend=self.model_backend,
            **kwargs,
        )

        # write the desc files
        desc_file_path = self.experiment_folder / "desc.txt"
        with open(desc_file_path, "w") as desc_file:
            desc_file.write(experiment_desc)

    def _update_evaluator_config(self):
        from easy_eval.config import EvaluatorConfig

        evaluator_config = EvaluatorConfig()
        for key, value in self.config.__dict__.items():
            if hasattr(evaluator_config, key):
                setattr(evaluator_config, key, value)

        return evaluator_config

    def run(self, tasks: Union[str, List[str]]):

        assert self.config is not None, ValueError(
            "Config is None, please create the experiment first."
        )
        assert self.evaluator is not None, ValueError(
            "Experiment not initialized. run experiment.create method."
        )
        assert tasks is not None, ValueError(
            "Tasks can not be None, please provide either a task in string or a list of tasks."
        )

        config_to_pass = self._update_evaluator_config()
        print(config_to_pass)
        results = self.evaluator(tasks=tasks, config=config_to_pass)
        return results
