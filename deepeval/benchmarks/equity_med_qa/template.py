class EquityMedQATemplate:

    @staticmethod
    def format_question(data: dict):
        items = list(data.items())
        question = items[0][-1]
        return question
