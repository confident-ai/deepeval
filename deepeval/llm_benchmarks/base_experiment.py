import os
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel
from typing import Union, List, Any, Optional


class BaseEvaluationExperiment(ABC):
    def __init__(
        self, experiment_name: str, experiment_desc: str, *args, **kwargs
    ) -> None:
        """
        An experiment name / alias should be witten like: <experiment-group> / <experiment-name>
        For custom experiments there can be custom experiment groups. However, for helm and harness
        the experiment group should be written in this format:

        For HELM based experiments: helm/<experiment-name>
        For Harness based experiments: harness/<experiment-name>
        """
        self.experiment_name = experiment_name
        self.experiment_desc = experiment_desc

        deep_eval_cache_path = Path.home() / ".cache" / "deepeval"

        if not deep_eval_cache_path.exists():
            deep_eval_cache_path.mkdir(parents=True, exist_ok=True)

        self._experiment_root_folder = deep_eval_cache_path
        self.experiment_folder = deep_eval_cache_path / experiment_name

        if not self.experiment_folder.exists():
            self.experiment_folder.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def create(self, config: BaseModel, *args, **kwargs):
        """Create an experiment by giving certain configs."""
        pass

    @abstractmethod
    def update(self, updated_config: BaseModel, *args, **kwargs):
        """Update an experiment with a new updated config."""
        pass

    @abstractmethod
    @property
    def delete(self):
        """Deletes the created experiment."""
        os.remove(self.experiment_folder)
        print(f"=> Experiment: {self.experiment_name} deleted.")

    @abstractmethod
    def run(self, tasks: Union[str, List[str]], *args, **kwargs):
        """Runs the experiments."""
        pass

    @abstractmethod
    @property
    def list_experiments(self):
        return os.listdir(self.experiment_folder)

    @abstractmethod
    @classmethod
    def run_experiments(
        cls,
        experiments: List[Any],
        num_workers: Optional[int] = 1,
        *args,
        **kwargs,
    ):
        pass

    @abstractmethod
    @classmethod
    def compare_experiments(cls, experiments_ids: List[str], *args, **kwargs):
        """Returns a pandas dataframe which will contain the comparision of different experiments.

        Note: These experiments will be grouped under the parent of the experiments. For example, if the parent
        is Harness, then comparision will be done for all the harness experiments.
        Similarly, if the experiments are done on HELM, then comparision will be done for HELM based experiments.
        """
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def delete_multiple_experiments(cls, *args, **kwargs):
        raise NotImplementedError
