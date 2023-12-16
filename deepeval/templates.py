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
