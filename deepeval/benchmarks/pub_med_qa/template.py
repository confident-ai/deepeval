from typing import Dict, List


class PubMedQATemplate:
    @staticmethod
    def generate_output(input: str) -> str:
        return input

    @staticmethod
    def format_question(data: Dict) -> str:
        context = data["context"]
        passages: List[str] = context["contexts"]
        labels: List[str] = context.get("labels", [])

        formatted_passages = []
        for index, passage in enumerate(passages):
            if index < len(labels) and labels[index]:
                formatted_passages.append(f"{labels[index]}: {passage}")
            else:
                formatted_passages.append(passage)

        abstract = "\n".join(formatted_passages)
        return (
            f"Question: {data['question']}\n"
            f"Abstract:\n{abstract}\n"
            "Answer (yes, no, or maybe): "
        )

    @staticmethod
    def format_answer(data: Dict) -> str:
        return data["final_decision"].lower()
