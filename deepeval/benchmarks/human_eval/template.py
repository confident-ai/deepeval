from deepeval.benchmarks.human_eval.task import HumanEvalTask
import re


class HumanEvalTemplate:

    # Most of this template was taken from https://arxiv.org/pdf/2107.03374.pdf

    @staticmethod
    def generate_output(input: str, task: HumanEvalTask):

        prompt = "Complete the following function.\n"
        prompt += input
        prompt += "Only output the function with the following entry_point: `{ep}` in string format.".format(
            ep=task.value
        )
        prompt += "Make sure your output begins with 'def'. No explanations needed. Do not format as markdown (such as *```python ... ```*)."

        return prompt
