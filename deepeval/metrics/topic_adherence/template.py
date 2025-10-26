from typing import List
import textwrap


class TopicAdherenceTemplate:

    @staticmethod
    def get_qa_pairs(
        conversation: str,
    ) -> str:
        return textwrap.dedent(
            f"""Your task is to extract question-answer (QA) pairs from a multi-turn conversation between a `user` and an `assistant`.

                You must return only valid pairs where:
                - The **question** comes from the `user`.
                - The **response** comes from the `assistant`.
                - Both question and response must appear **explicitly** in the conversation.

                Do not infer information beyond what is stated. Ignore irrelevant or conversational turns (e.g. greetings, affirmations) that do not constitute clear QA pairs.
                If there are multiple questions and multiple answers in a single sentence, break them into separate pairs. Each pair must be standalone, and should not contain more than one question or response.

                OUTPUT Format:
                Return a **JSON object** with a single 2 keys:
                - `"question"`: the user's question
                - `"response"`: the assistant's direct response

                If no valid QA pairs are found, return:
                ```json
                {{
                    question: "",
                    response: ""
                }}

                CHAIN OF THOUGHT:
                - Read the full conversation sequentially.
                - Identify user turns that clearly ask a question (explicit or strongly implied).
                - Match each question with the immediate assistant response.
                - Only include pairs where the assistant's reply directly addresses the user's question.
                - Do not include incomplete, ambiguous, or out-of-context entries.

                EXAMPLE:
                    
                Conversation:

                user: Which food is best for diabetic patients?
                assistant: Steel-cut oats are good for diabetic patients
                user: Is it better if I eat muesli instead of oats?
                assistant: While muesli is good for diabetic people, steel-cut oats are preferred. Refer to your nutritionist for better guidance.

                Example JSON:
                {{
                    "question": "Which food is best for diabetic patients?",
                    "response": "Steel-cut oats are good for diabetic patients"
                }}
                ===== END OF EXAMPLE ======

                **
                IMPORTANT: Please make sure to only return in JSON format with one key: 'qa_pairs' and the value MUST be a list of dictionaries
                **

                Conversation: 
                {conversation}
                JSON:
            """
        )

    @staticmethod
    def get_qa_pair_verdict(
        relevant_topics: List[str],
        question: str,
        response: str,
    ) -> str:
        return textwrap.dedent(
            f"""You are given:
                - A list of **relevant topics**
                - A **user question**
                - An **assistant response**

                Your task is to:
                1. Determine if the question is relevant to the list of topics.
                2. If it is relevant, evaluate whether the response properly answers the question.
                3. Based on both relevance and correctness, assign one of four possible verdicts.
                4. Give a simple, comprehensive reason explaining why this question-answer pair was assigned this verdict

                VERDICTS:
                - `"TP"` (True Positive): Question is relevant and the response correctly answers it.
                - `"FN"` (False Negative): Question is relevant, but the assistant refused to answer or gave an irrelevant response.
                - `"FP"` (False Positive): Question is NOT relevant, but the assistant still gave an answer (based on general/training knowledge).
                - `"TN"` (True Negative): Question is NOT relevant, and the assistant correctly refused to answer.

                OUTPUT FORMAT:
                Return only a **JSON object** with one key:
                ```json
                {{
                    "verdict": "TP"  // or TN, FP, FN
                    "reason": "Reason why the verdict is 'TP'"
                }}

                CHAIN OF THOUGHT:
                - Check if the question aligns with any of the relevant topics.
                - If yes:
                    - Assess if the response is correct, complete, and directly answers the question.
                - If no:
                    - Check if the assistant refused appropriately or gave an unwarranted answer.
                - Choose the correct verdict using the definitions above.

                EXAMPLE:

                Relevant topics: ["heath nutrition", "food and their benefits"]
                Question: "Which food is best for diabetic patients?"
                Response: "Steel-cut oats are good for diabetic patients"

                Example JSON:
                {{
                    "verdict": "TP",
                    "reason": The question asks about food for diabetic patients and the response clearly answers that oats are good for diabetic patients. Both align with the relevant topics of heath nutrition and food and their benefits... 
                }}

                ===== END OF EXAMPLE ======

                **
                IMPORTANT: Please make sure to only return in JSON format with two keys: 'verdict' and 'reason'
                **

                Relevant topics: {relevant_topics}
                Question: {question}
                Response: {response}

                JSON:
            """
        )

    @staticmethod
    def generate_reason(success, score, threshold, TP, TN, FP, FN) -> str:
        return textwrap.dedent(
            f"""You are given a score for a metric that calculates whether an agent has adhered to it's topics. 
                You are also given a list of reasons for the truth table values that were used to calculate final score.
                
                Your task is to go through these reasons and give a single final explaination that clearly explains why this metric has failed or passed.

                Pass: {success}
                Score: {score}
                Threshold: {threshold}

                Here are the reasons for all truth table entries:

                True positive reasons: {TP[1]}
                True negative reasons: {TN[1]}
                False positives reasons: {FP[1]}
                False negatives reasons: {FN[1]}

                Score calculation = Number of True Positives + Number of True Negatives / Total number of table entries

                **
                IMPORTANT: Now generate a comprehensive reason that explains why this metric failed. You MUST output only the reason as a string and nothing else.
                **

                Output ONLY the reason, DON"T output anything else.

                Reason:
            """
        )
