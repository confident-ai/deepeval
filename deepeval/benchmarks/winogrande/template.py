import re


class WinograndeTemplate:

    n_shot_examples = [
        {
            "sentence": "Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine.",
            "option1": "Ian",
            "option2": "Dennis",
            "answer": "2",
        },
        {
            "sentence": "He never comes to my home, but I always go to his house because the _ is smaller.",
            "option1": "home",
            "option2": "house",
            "answer": "1",
        },
        {
            "sentence": "Kyle doesn't wear leg warmers to bed, while Logan almost always does. _ is more likely to live in a warmer climate.",
            "option1": "Kyle",
            "option2": "Logan",
            "answer": "1",
        },
        {
            "sentence": "The treasury workers took the gold bars off of the trolley and stacked them in the safe until the _ was empty.",
            "option1": "safe",
            "option2": "trolley",
            "answer": "2",
        },
        {
            "sentence": "Emily looked up and saw Patricia racing by overhead, as _ was on the ramp .",
            "option1": "Emily",
            "option2": "Patricia",
            "answer": "2",
        },
    ]

    @staticmethod
    def generate_output(input: str, n_shots: int):
        prompt = ""
        for i in range(n_shots):
            prompt += WinograndeTemplate.format_question(
                WinograndeTemplate.n_shot_examples[i]
            )
        prompt += input
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer: bool = True):
        sentence = data["sentence"]
        option1 = data["option1"]
        option2 = data["option2"]
        answer = data["answer"]
        prompt = f"Sentence: {sentence}\nA. {option1}\nB. {option2}\nAnswer:"
        if include_answer:
            prompt += f"{'A' if answer == '1' else 'B'}\n\n"
        return prompt

    @staticmethod
    def format_answer(data: dict):
        answer = data["answer"]
        return "A" if answer == "1" else "B"
