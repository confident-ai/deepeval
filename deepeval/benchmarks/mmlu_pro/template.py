from deepeval.benchmarks.mmlu_pro.task import MMLUProTask


class MMLUProTemplate:

    @staticmethod
    def generate_output(
        input: str, train_set: object, task: MMLUProTask, n_shots: int
    ):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n"
        prompt = prompt.format(task.value)
        for i in range(min(n_shots, len(train_set))):
            prompt += MMLUProTemplate.format_question(train_set[i])
        prompt += input
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer: bool = True):
        choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        prompt = data["question"]
        for j in range(len(data["options"])):
            prompt += "\n{}. {}".format(choices[j], data["options"][j])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(data["answer"])
        return prompt
