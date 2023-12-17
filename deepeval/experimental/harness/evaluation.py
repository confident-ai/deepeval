import os
import re
import json
import logging
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List, Union

from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.tasks import initialize_tasks, include_path
from deepeval.llm_benchmarks.harness.task import HarnessTasks

# Todo:
# Bring a concept of Grouped evaluations

from typing import List, Optional, Union
from pydantic import BaseModel, Field


class HarnessConfig(BaseModel):
    model: str = Field(
        default="hf", description="Name of the model type e.g. `hf`"
    )
    model_args: str = Field(
        default="", description="String arguments for the model."
    )
    tasks: Optional[Union[str, List[str]]] = Field(
        default=None, description="The task name or list of tasks."
    )
    num_fewshot: Optional[int] = Field(
        default=None, description="Number of few-shot examples."
    )
    max_batch_size: Optional[int] = Field(
        default=None, description="Maximal batch size used."
    )
    batch_size: int = Field(
        default=1, description="The batch size for evaluation."
    )
    device: Optional[str] = Field(
        default=None, description="Device to compute on (e.g. cuda:0, cpu)."
    )
    use_cache: Optional[str] = Field(
        default=None, description="Path to load evaluations from cache."
    )
    limit: Optional[float] = Field(
        default=None, description="Limit for number of examples."
    )
    decontamination_ngrams_path: Optional[
        str
    ] = None  # To be removed by the harness as it's unused.
    output_path: Optional[str] = Field(
        default=None, description="Path to store the logs."
    )
    check_integrity: bool = Field(
        default=False, description="Check integrity for tasks."
    )
    write_out: bool = Field(
        default=False, description="Print prompt for the first few documents."
    )
    log_samples: bool = Field(
        default=True,
        description="Write all model outputs and documents for per-sample measurement and analysis.",
    )
    show_config: bool = Field(
        default=True,
        description="Show full config of tasks at the end of evaluation.",
    )
    include_path: Optional[str] = Field(
        None, description="Additional path for external tasks."
    )
    gen_kwargs: Optional[str] = Field(
        None,
        description="String arguments for model generation on certain tasks.",
    )
    verbosity: str = Field(
        "INFO", description="Log error when tasks are not registered."
    )


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


class HarnessEvaluate:
    def __init__(self, harness_config: HarnessConfig) -> None:
        self.harness_config = harness_config
        self.eval_logger = utils.eval_logger

        self.eval_logger = utils.eval_logger
        self.eval_logger.setLevel(
            getattr(logging, f"{self.harness_config.verbosity}")
        )
        self.eval_logger.info(
            f"Verbosity set to {self.harness_config.verbosity}"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        initialize_tasks(self.harness_config.verbosity)

        if self.harness_config.limit:
            self.eval_logger.warning(
                "limit SHOULD ONLY BE USED FOR TESTING."
                "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
            )
        if self.harness_config.include_path is not None:
            self.eval_logger.info(
                f"Including path: {self.harness_config.include_path}"
            )
            include_path(self.harness_config.include_path)

        if self.harness_config.output_path:
            self.path = Path(self.harness_config.output_path)
            # check if file or 'dir/results.json' exists
            if (
                self.path.is_file()
                or Path(self.harness_config.output_path)
                .joinpath("results.json")
                .is_file()
            ):
                self.eval_logger.warning(
                    f"File already exists at {self.path}. Results will be overwritten."
                )
                self.output_path_file = self.path.joinpath("results.json")
                assert not self.path.is_file(), "File already exists"
            # if self.path json then get parent dir
            elif self.path.suffix in (".json", ".jsonl"):
                self.output_path_file = self.path
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path = self.path.parent
            else:
                self.path.mkdir(parents=True, exist_ok=True)
                self.output_path_file = self.path.joinpath("results.json")
        elif (
            self.harness_config.log_samples
            and not self.harness_config.output_path
        ):
            assert self.harness_config.output_path, "Specify --output_path"

    def evaluate(self, tasks: Union[List[str], str]):
        if self.harness_config.tasks is None:
            self.task_names = ALL_TASKS
        else:
            self.task_names = HarnessTasks.load_task(tasks)

        self.harness_config.tasks = tasks
        self.eval_logger.info(f"Selected Tasks: {self.task_names}")

        results = evaluator.simple_evaluate(
            model=self.harness_config.model,
            model_args=self.harness_config.model_args,
            tasks=tasks,
            num_fewshot=self.harness_config.num_fewshot,
            batch_size=self.harness_config.batch_size,
            max_batch_size=self.harness_config.max_batch_size,
            device=self.harness_config.device,
            use_cache=self.harness_config.use_cache,
            limit=self.harness_config.limit,
            decontamination_ngrams_path=self.harness_config.decontamination_ngrams_path,
            check_integrity=self.harness_config.check_integrity,
            write_out=self.harness_config.write_out,
            log_samples=self.harness_config.log_samples,
            gen_kwargs=self.harness_config.gen_kwargs,
        )

        if results is not None:
            if self.harness_config.log_samples:
                samples = results.pop("samples")
            dumped = json.dumps(
                results, indent=2, default=_handle_non_serializable
            )
            if self.harness_config.show_config:
                print(dumped)

            batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

            if self.harness_config.output_path:
                self.output_path_file.open("w").write(dumped)

                if self.harness_config.log_samples:
                    for task_name, config in results["configs"].items():
                        output_name = "{}_{}".format(
                            re.sub("/|=", "__", self.harness_config.model_args),
                            task_name,
                        )
                        filename = self.path.joinpath(f"{output_name}.jsonl")
                        samples_dumped = json.dumps(
                            samples[task_name],
                            indent=2,
                            default=_handle_non_serializable,
                        )
                        filename.open("w").write(samples_dumped)

            print(
                f"{self.harness_config.model} ({self.harness_config.model_args}), gen_kwargs: ({self.harness_config.gen_kwargs}), limit: {self.harness_config.limit}, num_fewshot: {self.harness_config.num_fewshot}, "
                f"batch_size: {self.harness_config.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
            )
            print(evaluator.make_table(results))
            if "groups" in results:
                print(evaluator.make_table(results, "groups"))

            # Todo: need to somewhere sabe this table in the form of CSV along with configs
            return results


if __name__ == "__main__":
    config = HarnessConfig(
        model="hf",
        model_args="pretrained=gpt2",
        device="cpu",
        limit=5,
        tasks=["babi"],
        batch_size=1,
        log_samples=False,  # if true need to specify output paths
    )

    tasks = ["hellaswag", "babi"]
    eval_ = HarnessEvaluate(harness_config=config)
    results = eval_.evaluate(tasks=tasks)

    print(results)
