class ARCTemplate:
    # ARC does not label its options consistently. Most items use "A"-"D", but a
    # subset labels them "1"-"4", a few offer five options ("A"-"E"), and at
    # least one offers only three. The gold `answerKey` uses whichever labelling
    # that item happens to carry, so a numerically labelled item produced an
    # `expected_output` of "3" — a value the answer schema cannot represent and
    # the confinement instructions never mention. Those items scored 0 no matter
    # what the model answered.
    #
    # Option labels are positionally aligned with `choices["text"]`, so both the
    # rendered prompt and the gold answer are normalized to letters by position.
    # This keeps the prompt and the expected answer in the same alphabet.
    option_letters = ["A", "B", "C", "D", "E"]

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
    def normalize_label(data: dict, label: str) -> str:
        """Map an item's own option label onto its positional letter.

        Returns `label` unchanged when it is not one of the item's options, or
        when the item has more options than there are letters, so an unexpected
        dataset shape degrades to the previous behaviour instead of raising.
        """
        labels = data["choices"]["label"]
        try:
            index = labels.index(label)
        except ValueError:
            return label
        if index >= len(ARCTemplate.option_letters):
            return label
        return ARCTemplate.option_letters[index]

    @staticmethod
    def format_question(data: dict, include_answer: bool = True):
        prompt = data["question"]
        texts = data["choices"]["text"]
        labels = data["choices"]["label"]
        for i in range(len(labels)):
            prompt += "\n{}. {}".format(
                ARCTemplate.normalize_label(data, labels[i]), texts[i]
            )
        prompt += "\nAnswer: "
        if include_answer:
            prompt += " {}\n\n".format(ARCTemplate.format_answer(data))
        return prompt

    @staticmethod
    def format_answer(data: dict):
        return ARCTemplate.normalize_label(data, data["answerKey"])
