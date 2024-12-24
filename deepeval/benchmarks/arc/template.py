class ARCTemplate:
    n_shot_examples = [
        {
            "id": "Mercury_7220990",
            "question": "Which factor will most likely cause a person to develop a fever?",
            "choices": {
                "text": [
                    "a leg muscle relaxing after exercise",
                    "a bacterial population in the bloodstream",
                    "several viral particles on the skin",
                    "carbohydrates being digested in the stomach",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "MCAS_2007_8_5189",
            "question": "Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?",
            "choices": {
                "text": ["carbon dioxide", "food", "protection", "water"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "Mercury_SC_401169",
            "question": "When a switch is used in an electrical circuit, the switch can",
            "choices": {
                "text": [
                    "cause the charge to build.",
                    "increase and decrease the voltage.",
                    "cause the current to change direction.",
                    "stop and start the flow of current.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "D",
        },
        {
            "id": "MCAS_2004_8_27",
            "question": "Which of the following is an example of an assistive device?",
            "choices": {
                "text": [
                    "contact lens",
                    "motorcycle",
                    "raincoat",
                    "coffee pot",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "id": "NYSEDREGENTS_2006_8_10",
            "question": "Rocks are classified as igneous, metamorphic, or sedimentary according to",
            "choices": {
                "text": [
                    "their color",
                    "their shape",
                    "how they formed",
                    "the minerals they contain",
                ],
                "label": ["1", "2", "3", "4"],
            },
            "answerKey": "3",
        },
    ]

    @staticmethod
    def generate_output(input: str, n_shots: int):
        prompt = ""
        for i in range(n_shots):
            prompt += ARCTemplate.format_question(
                ARCTemplate.n_shot_examples[i]
            )
        prompt += input
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer: bool = True):
        prompt = data["question"]
        texts = data["choices"]["text"]
        labels = data["choices"]["label"]
        for i in range(len(labels)):
            prompt += "\n{}. {}".format(labels[i], texts[i])
        prompt += "\nAnswer: "
        if include_answer:
            prompt += " {}\n\n".format(data["answerKey"])
        return prompt

    @staticmethod
    def format_answer(data: dict):
        return data["answerKey"]
