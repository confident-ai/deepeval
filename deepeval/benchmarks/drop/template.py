class DROPTemplate:

    # Most of this template was taken from MMLU Github Repo
    # The output confinement is a novel addition, since the original code
    # outputted log_probabilties for each answer choice

    @staticmethod
    def generate_output(input: str, train_set: object, type: str, n_shots: int):
        prompt = "Answer the following question based on the passage.\n\n"
        # Examples
        if n_shots > 0:
            prompt += "Below are some examples:\n\n"
        for i in range(n_shots):
            prompt += DROPTemplate.format_question(train_set[i]) + "\n"
        # define output confinement
        prompt += input
        prompt += (
            "Output should be of type {type}. No explanation needed.".format(
                type=type
            )
        )
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer: bool = False):
        prompt = "Passage: " + data["passage"] + "\n"
        prompt += "Question: " + data["question"] + "\n"
        prompt += "Answer: "
        if include_answer:
            prompt += data["answers_spans"]["spans"][0] + "\n"
        return prompt
