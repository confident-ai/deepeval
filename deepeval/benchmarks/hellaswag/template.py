from deepeval.benchmarks.hellaswag.task import HellaSwagTask


class HellaSwagTemplate:

    staticmethod
    def generate_output(input: str, train_set: object, task: HellaSwagTask, n_shots: int):
        prompt = "The following are multiple choice questions (with answers) are sentence completion problems about {}.\n\n"
        prompt = prompt.format(task.value)
        for i in range(n_shots):
            prompt += HellaSwagTemplate.format_question(train_set[i])
        prompt += input

        #define ouptut confinement
        prompt += "\n\nOutput 'A', 'B', 'C', or 'D'. Full answer not needed."
        return prompt
    
    @staticmethod
    def format_question(data: dict, include_answer: bool=True):
        prompt = data['ctx']
        choices = ["A", "B", "C", "D"]
        for j in range(len(choices)):
            choice = choices[j]
            prompt += "\n{}. {}".format(choice, data['endings'][j])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(choices[int(data['label'])])
        return prompt