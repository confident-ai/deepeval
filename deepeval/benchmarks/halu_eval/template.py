from typing import List, Tuple

from deepeval.benchmarks.halu_eval.task import HaluEvalTask


class HaluEvalTemplate:
    """Prompt templates for HaluEval (Li et al., 2023).

    Every golden asks the model a single Yes/No question: does the given
    response contain hallucinated information? "Yes" means hallucinated.
    """

    instructions = {
        HaluEvalTask.QA: (
            "You are a hallucination detector. Given some knowledge, a question "
            "and an answer, decide whether the answer contains non-factual or "
            "hallucinated information that is not supported by the knowledge."
        ),
        HaluEvalTask.DIALOGUE: (
            "You are a hallucination detector. Given some knowledge, a dialogue "
            "history and a response, decide whether the response contains "
            "non-factual or hallucinated information that is not supported by "
            "the knowledge or the dialogue."
        ),
        HaluEvalTask.SUMMARIZATION: (
            "You are a hallucination detector. Given a document and a summary, "
            "decide whether the summary contains non-factual or hallucinated "
            "information that is not supported by the document."
        ),
        HaluEvalTask.GENERAL: (
            "You are a hallucination detector. Given a user query and a "
            "response, decide whether the response contains non-factual or "
            "hallucinated information."
        ),
    }

    question = (
        "Does the {subject} contain hallucinated information? "
        "Answer 'Yes' or 'No'."
    )

    @staticmethod
    def generate_output(input: str, task: HaluEvalTask) -> str:
        return f"{HaluEvalTemplate.instructions[task]}\n\n{input}"

    @staticmethod
    def format_samples(data: dict, task: HaluEvalTask) -> List[Tuple[str, str]]:
        """Turn one dataset row into (input, expected_output) pairs.

        QA, dialogue and summarization rows hold a right and a hallucinated
        response, so each row yields two goldens: the right response
        (expected "No") followed by the hallucinated one (expected "Yes").
        General rows hold one human-labelled ChatGPT response.
        """
        if task == HaluEvalTask.QA:
            context = (
                f"Knowledge: {data['knowledge']}\n"
                f"Question: {data['question']}\n"
            )
            pairs = [
                (f"{context}Answer: {data['right_answer']}", "No"),
                (f"{context}Answer: {data['hallucinated_answer']}", "Yes"),
            ]
            subject = "answer"
        elif task == HaluEvalTask.DIALOGUE:
            context = (
                f"Knowledge: {data['knowledge']}\n"
                f"Dialogue history: {data['dialogue_history']}\n"
            )
            pairs = [
                (f"{context}Response: {data['right_response']}", "No"),
                (f"{context}Response: {data['hallucinated_response']}", "Yes"),
            ]
            subject = "response"
        elif task == HaluEvalTask.SUMMARIZATION:
            context = f"Document: {data['document']}\n"
            pairs = [
                (f"{context}Summary: {data['right_summary']}", "No"),
                (f"{context}Summary: {data['hallucinated_summary']}", "Yes"),
            ]
            subject = "summary"
        elif task == HaluEvalTask.GENERAL:
            label = str(data["hallucination"]).strip().lower()
            pairs = [
                (
                    f"User query: {data['user_query']}\n"
                    f"Response: {data['chatgpt_response']}",
                    "Yes" if label == "yes" else "No",
                )
            ]
            subject = "response"
        else:
            raise ValueError(f"Unsupported HaluEval task: {task}")

        question = HaluEvalTemplate.question.format(subject=subject)
        return [
            (f"{input}\n\n{question}\nAnswer: ", expected)
            for input, expected in pairs
        ]

    @staticmethod
    def normalize_prediction(prediction: str) -> str:
        """Map free-text answers such as 'yes.' or 'No, because ...' to
        'Yes'/'No'. Anything else is returned unchanged (and scored wrong)."""
        text = str(prediction).strip()
        first_word = text.split()[0] if text else ""
        first_word = first_word.strip(".,:;!\"'()[]*").lower()
        if first_word == "yes":
            return "Yes"
        if first_word == "no":
            return "No"
        return text
