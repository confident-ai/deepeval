from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
from deepeval.benchmarks.big_bench_hard.cot_prompts import *
from deepeval.benchmarks.big_bench_hard.shot_prompts import *


class BigBenchHardTemplate:

    # COT prompts were taken directly from BBH Github Repo
    # Few-shot prompts were adpated from COT prompts by removing CoT Reasoning

    @staticmethod
    def generate_output(
        input: str, task: BigBenchHardTask, n_shots: int, enable_cot: bool
    ):

        # generate folder path based on enable_cot
        path = "./"
        if enable_cot:
            path += "cot_prompts/"
        else:
            path += "shot_prompts/"

        # get prompt from text file based on n_shots and folder path
        path += BigBenchHardTemplate.get_filename(task)
        prompt = "Task description: "
        prompt += "\n\n".join(
            BigBenchHardTemplate.read_file(path)[: n_shots + 1]
        )
        prompt += "\n\nQ: " + input + "\nA: "

        return prompt

    # a function to
    def read_file(file_path):
        # Open the file in read mode ('r')
        with open(file_path, "r") as file:
            # Read the entire content of the file
            file_content = file.read()

        # Now you can use file_content as you need
        sections = file_content.split("\n\n")
        return sections

    def get_filename(task):
        # generate prompts
        return task.value + ".txt"
