from deepeval.benchmarks.hellaswag.task import HellaSwagTask


class HellaSwagTemplate:

    # Template for HellaSwag was heavily inspired by MMLU due to multiple-choice nature of benchmark
    # In the original HellaSwag paper, the models were fine-tuned using softmax layer. No prompts were used.
    # But GPT-4 topped the leaderboard using 10-shot prompting, though the prompt was not released.

    @staticmethod
    def generate_output(
        input: str, train_set: object, task: HellaSwagTask, n_shots: int
    ):
        prompt = "The following are multiple choice questions (with answers) are sentence completion problems about {}.\n\n"
        prompt = prompt.format(task.value)
        for i in range(n_shots):
            prompt += HellaSwagTemplate.format_question(train_set[i])
        prompt += input

        # define ouptut confinement
        prompt += "\n\nOutput 'A', 'B', 'C', or 'D'. Full answer not needed."
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer: bool = True):
        prompt = data["ctx"]
        choices = ["A", "B", "C", "D"]
        for j in range(len(choices)):
            choice = choices[j]
            prompt += "\n{}. {}".format(choice, data["endings"][j])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(choices[int(data["label"])])
        return prompt
