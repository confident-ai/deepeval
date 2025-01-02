class LogiQATemplate:

    n_shot_examples = [
        "Input\nWrite a multi-choice question for the following article:\nArticle: David knows Mr. Zhang's friend Jack, and Jack knows David's friend Ms. Lin. Everyone of them who knows Jack has a master's degree, and everyone of them who knows Ms. Lin is from Shanghai.\nQuestion: \nWho is from Shanghai and has a master's degree?\nOptions:\nA David\nB Jack\nC Mr Zhang\nD Ms. Lin\nAnswer:\nA\n",
        "Input\nWrite a multi-choice question for the following article:\nArticle: Jimmy asked Hank to go to the mall the next day. Hank said, If it doesn't rain tomorrow, I'll go climbing. The next day, there was a drizzle. Jimmy thought that Hank would not go climbing, so he went to pick up Henry to the mall. Nevertheless, Hank went climbing the mountain. When the two met again, Jimmy blamed Hank for not keeping his word.\nQuestion: \nWhich of the following comments is appropriate?\nOptions:\nA This argument between Jimmy and Hank is meaningless\nB Jimmy's reasoning is illogical\nC Two people have different understandings of a drizzle\nD Hank broke his promise and caused the debate\nAnswer:\nB\n",
        "Input\nWrite a multi-choice question for the following article:\nArticle: Only if the government reinforce basic education can we improve our nation's education to a new stage. In order to stand out among other nations, we need to have a strong educational enterprise.\nQuestion: \nWhich can be inferred from the statement above?\nOptions:\nA The whole society should be focused on education\nB In order to stand out among nations, we should reinforce basic education\nC In order to improve our education to a new stage, it is necessary to increase the salary of college teachers\nD In order to reinforce basic education, all primary school teachers must have a bachelor degree or above.\nAnswer:\nB\n",
        "Input\nWrite a multi-choice question for the following article:\nArticle: Last night, Mark either went to play in the gym or visited his teacher Tony. If Mark drove last night, he didn't go to play in the gym. Mark would go visit his teacher Tony only if he and his teacher had an appointment. In fact, Mark had no appointment with his teacher Tony in advance.\nQuestion: \nWhich is true based on the above statement?\nOptions:\nA Mark went to the gym with his teacher Tony last night\nB Mark visited his teacher Tony last night\nC Mark didn't drive last night\nD Mark didn't go to the gym last night.\nAnswer:\nC\n",
        "Input\nWrite a multi-choice question for the following article:\nArticle: The coach of a national football team found that the best cooperative arrangement of the players U, V, W, X, Y, and Z during the training are: (1) V and X cannot be on the field at the same time, and neither can be off the field the same time. (2) V is not on the field only if U is not on the field. (3) If W is on the field, then X is on the field. (4) If Ｙ and Ｚ are on the field, then W must be on the field. This arrangement can yield the best performance.\nQuestion: \nIf U and Z are both on the field, for best performance, which of the following arrangement is appropriate?\nOptions:\nA X is on the eld and Y is not on the field\nB V is on the eld and Y is not on the field\nC V and W are both on the field\nD V and Y are not on the field\nAnswer:\nB\n",
    ]

    @staticmethod
    def generate_output(input: str, n_shots: int):
        prompt = ""
        for i in range(n_shots):
            prompt += LogiQATemplate.n_shot_examples[i] + "\n"
        prompt += input
        return prompt

    @staticmethod
    def format_question(data: dict):
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        article = data["text"]
        question = data["question"]
        options_old = data["options"]
        options = ""
        for j, option in enumerate(options_old):
            options += label_map[j] + " " + option + "\n"
        return (
            "Write a multi-choice question for the following article:\nArticle: "
            + article
            + "\nQuestion: "
            + question
            + "\nOptions: "
            + options
            + "Answer: "
        )

    @staticmethod
    def format_output(data: dict):
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        answer = data["answer"]
        return label_map[answer]
