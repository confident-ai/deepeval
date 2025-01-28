from typing import List


class TaskNodeTemplate:
    @staticmethod
    def generate_task_output(instructions: str, text: str):
        return f"""{instructions}

{text}

Output:
"""


class BinaryJudgementTemplate:
    @staticmethod
    def generate_binary_verdict(criteria: str, text: str):
        return f"""{criteria}

{text}

Output:
"""


class NonBinaryJudgementTemplate:
    @staticmethod
    def generate_non_binary_verdict(
        criteria: str, text: str, options: List[str]
    ):
        return f"""{criteria}

{text}

Output:
"""
