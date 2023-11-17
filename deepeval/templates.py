evaluation_steps_template = """
You will be given 4 blocks of text labelled "Input", "Actual output", "Expected output", and "Context". Generate 3-4 concise evaluation steps based on the criteria below. Explicitly state to ignore any blocks of text that is not mentioned in the evaluation criteria.

Criteria:
{criteria}
"""

evaluation_results_template = """
Evaluation Steps:
{evaluation_steps}

Text:
{text}

Given the evaluation steps, please evaluate the provided Text. Some fields in text might be unavailable and will be labelled "N/A". Return a `score` ranging from 0 - 5, with 5 being that it follows the criteria and 0 being that it does not. Be extra harsh and give as low a score as possible as it designed to penalize.

**
IMPORTANT: Please make sure to only score integer value between 0 - 5. No words or explaination is needed.
**

score:
"""
