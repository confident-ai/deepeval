import os
import glob
import logging
from typing import Union, List, Any

from lm_eval.api.registry import ALL_TASKS
from lm_eval import utils
from lm_eval.tasks import initialize_tasks

initialize_tasks()

eval_logger = utils.eval_logger
eval_logger.setLevel(getattr(logging, "INFO"))
eval_logger.info("Verbosity set to INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Todo:
# For everytask, we need to save a task descriptor. This can be done by going through each task and use gpt to summarize
# and store it.

# Todo:
# For every sub task or so, we should have an option to download that very task and use it under deepeval seperately
# or may be outside of deepeval.


class HarnessTasks:
    @classmethod
    @property
    def list_all_main_tasks(cls) -> set:
        return set([task.split("_")[0].split("-")[0] for task in ALL_TASKS])

    @classmethod
    def get_subtasks(cls, main_task: str):
        assert main_task in cls.list_all_main_tasks, ValueError(
            f"Task {main_task} not found"
        )
        return sorted(
            [task for task in ALL_TASKS if task.startswith(main_task)]
        )

    @classmethod
    def describe_task(cls, task_name: str) -> str:
        raise NotImplementedError()

    @classmethod
    def load_task(cls, tasks: Union[List[str], str]) -> List[Any]:
        if isinstance(tasks, str):
            task_names = []
            yaml_path = os.path.join(tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_names = utils.pattern_match(tasks, ALL_TASKS)
            for task in [task for task in tasks if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task
                for task in tasks
                if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks {missing} were not found. Try `lm-eval --tasks list` for list of available tasks."
                )
        return task_names

    @classmethod
    def group_tasks(task_names: Union[str, List[str]]) -> None:
        raise NotImplementedError()
    
    @classmethod
    def download_task(cls, task_name: str) -> None:
        raise NotImplementedError()
    
