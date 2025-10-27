from typing import List


class VerdictNodeTemplate:
    @staticmethod
    def generate_reason(verbose_steps: List[str], score: float, name: str):
        return f"""Given the metric name, the score of that metric, and the DAG traversal, generate a reason for why the score is that way. 
In this case, the "DAG Traversal" is the steps it took to the final leaf "VerdictNode". The DAG allows for deterministic decision trees, where depending on the outcome of the previous parent nodes results in the current path you're seeing.
Your reason should directly reference the DAG traversal path to make it concrete, factual and concise.

Metric Name:
{name}

Score:
{score}

DAG Traversal:
{verbose_steps}

**
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <metric_name_score> because <your_reason>."
}}
**

JSON:
"""


class TaskNodeTemplate:
    @staticmethod
    def generate_task_output(instructions: str, text: str):
        return f"""Given the following instructions, generate an output.

{instructions}

{text}

===END OF INSTRUCTIONS===

**
IMPORTANT: Please make sure to only return in JSON format, with the 'output' key as the output from the instructions.
Example JSON:
{{
    "output": "your output goes here"
}}
**

JSON:
"""


class BinaryJudgementTemplate:
    @staticmethod
    def generate_binary_verdict(criteria: str, text: str):
        return f"""{criteria}

{text}

**
IMPORTANT: Please make sure to only return a json with two keys: `verdict` (True or False), and the 'reason' key providing the reason. The verdict must be a boolean only, either True or False.
Example JSON:
{{
    "verdict": True,
    "reason": "..."
}}
**

JSON:
"""


class NonBinaryJudgementTemplate:
    @staticmethod
    def generate_non_binary_verdict(
        criteria: str, text: str, options: List[str]
    ):
        return f"""{criteria}

{text}

**
IMPORTANT: Please make sure to only return a json with two keys: 'verdict' {options} and 'reason' providing the reason.
Example JSON:
{{
    "verdict": {options},
    "reason": "..."
}}
**

JSON:
"""
