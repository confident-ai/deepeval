# TODO: LLMEvalTemplate
evaluation_steps_template = """
You will be given 4 blocks of text labelled "Input", "Actual output", "Expected output", and "Context". Generate 3-4 concise evaluation steps based on the criteria below. Explicitly state to ignore any blocks of text that is not mentioned in the evaluation criteria.

Criteria:
{criteria}

**
IMPORTANT: Please make sure to only return in JSON format, with the "steps" key as a list of strings. No words or explaination is needed.
**

JSON:
"""

evaluation_results_template = """
Evaluation Steps:
{evaluation_steps}

Text:
{text}

Given the evaluation steps, please evaluate the provided Text. Some fields in text might be unavailable and will be labelled "N/A". Only return a JSON with two keys: 1) a `score` key ranging from 0 - 10, with 10 being that it follows the criteria and 0 being that it does not, and 2) a `reason` key, a reason for the given score. Be extra harsh and give as low a score as possible as it designed to penalize.

**
IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explaination is needed.
**

JSON:
"""

# TODO: summarization template
closed_end_questions_template = """
Based on the text below, please generate {n} closed-ended questions that can be answered with either a 'yes' or 'no'. Only return a JSON with a 'questions' key, which is a list of strings. The questions have to be STRICTLY closed ended.

Text:
{text}

JSON:
"""

closed_end_answers_template = """
Based on the given text, please provide either a 'yes', 'no', or 'idk' answer to the question presented. Only answer 'idk' IF the the answer cannot be deduced from the given text.

Question:
{question}

Text:
{text}

Answer:
"""


class FaithfulnessTemplate:
    @staticmethod
    def generate_truths(text):
        return f"""Based on the given text, please generate a comphrensive list of undisputed "truths" that can inferred from the provided text. You should NOT incorporate any knowledge you have, and take each truth at face value.

Example text: "Einstein won the noble prize in 1968 for his discovery of the photoelectric effect."
Example truths: ["Einstein won the noble prize for his discovery of the photoelectric effect.", "Einstein won the noble prize in 1968."]

**
IMPORTANT: Please make sure to only return in JSON format, with the "truths" key as a list of strings. No words or explaination is needed.
**

Text:
{text}

JSON:
"""

    @staticmethod
    def generate_verdicts(truths, text):
        return f"""Based on a list of strings, called contexts, please generate a list of JSON objects to indicate whether the given 'actual output' agrees with EACH context. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes', 'no', or 'idk', and states whether the given text agrees with the context. 
The 'reason' is the reason for the verdict. When the answer is 'no' or 'idk', try to provide a correction in the reason.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example contexts: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect.", "Einstein won the Nobel Prize in 1968."]
Example text: "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect."

Example:
"verdicts": [
    {{
        "verdict": "yes",
        "reason": "The context states that Einstein won the Nobel Prize for his discovery of the photoelectric effect."
    }},
    {{
        "verdict": "no",
        "reason": "The context states that Einstein won the Nobel Prize in 1968, not 1969."
    }}
]  

You should NOT incorporate any prior knowledge you have and take each context at face value. Since you are going to generate a verdict for each context, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of contexts.
**

Contexts:
{truths}

Actual Output:
{text}

JSON:
"""

    @staticmethod
    def generate_reason(score, contradiction_reasons):
        return f"""Below is a list of Contradictions. It explains why the 'actual output' does not align with the 'retrieval context'.
Given the faithfulness score, which is a 0-1 score indicating how faithful the `actual output` is the context (higher the better), concisely summarize the contradictions to justify the score. If there are no contradictions, just say something positive with an upbeat tone (but don't overdo it otherwise it gets annoying).

Faithfulness Score:
{score}

Contradictions:
{contradiction_reasons}

Example:
The score is <faithfulness_score> because <your_reason>.

Justification:
"""


class AnswerRelevancyTemplate:
    @staticmethod
    def generate_mock_questions(answer, retrieval_context):
        return f"""Given the answer and retrieval context, generate a list of possible questions that could lead to such as answer. The generated question should revolve around the answer, retrieval contexts are just there for additional information, in case you need it. Make these questions concise.

Answer:
{answer}

Retrieval Context:
{retrieval_context}

Questions:
"""

    @staticmethod
    def generate_meta_question(original_question, mock_question):
        return f"""Could the following mock question: {mock_question}, be be seeking the same information as the original question '{original_question}'?
"""

    @staticmethod
    def generate_verdicts(meta_questions):
        return f"""For each question in the provided list of questions, please generate and list of JSON with two keys: `verdict` and `reason`.
The 'verdict' key should STRICTLY be either a 'yes', 'no', or 'idk' answer to the question presented.  Only answer 'idk' IF the the answer cannot be reliably from the given text.
The 'reason' is the reason for the verdict.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example questions: ["Could the following question: 'Describe the journey of the protagonist in the first book in discovering his magical background' be addressing a similar point as 'Who discovers he is a wizard in the first book'?"]

Example:
"verdicts": [
    {{
        "verdict": "idk",
    }},
]  

Since you are going to generate a verdict for each question, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of questions.
**
           
Questions:
{meta_questions}

JSON:
"""

    @staticmethod
    def generate_reason(
        irrelevant_questions, ambiguous_questions, original_question, score
    ):
        pass
