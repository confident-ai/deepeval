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


class HallucinationTemplate:
    @staticmethod
    def generate_verdicts(actual_output, contexts):
        return f"""For each context in contexts, which is a list of strings, please generate a list of JSON objects to indicate whether the given 'actual output' agrees with EACH context. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the given text agrees with the context. 
The 'reason' is the reason for the verdict. When the answer is 'no', try to provide a correction in the reason. 

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example contexts: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect.", "Einstein won the Nobel Prize in 1968."]
Example actual output: "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect."

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "The actual output agrees with the provided context which states that Einstein won the Nobel Prize for his discovery of the photoelectric effect."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output contradicts the provided context which states that Einstein won the Nobel Prize in 1968, not 1969."
        }}
    ]  
}}

You should NOT incorporate any prior knowledge you have and take each context at face value. Since you are going to generate a verdict for each context, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of contexts.
You should FORGIVE cases where the actual output is lacking in detail, you should ONLY provide a 'no' answer if IT IS A CONTRADICTION.
**

Contexts:
{contexts}

Actual Output:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_reason(factual_alignments, contradictions, score):
        return f"""Given a list of factual alignments and contradictions, which highlights alignment/contradictions between the `actual output` and `contexts, use it to provide a reason for the hallucination score in a CONCISELY. Note that The hallucination score ranges from 0 - 1, and the lower the better.

Factual Alignments:
{factual_alignments}

Contradictions:
{contradictions}

Hallucination Score:
{score}

Example:
The score is <hallucination_score> because <your_reason>.

Reason:
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
You DON'T have to provide a reason if the answer is 'yes'.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example retrieval contexts: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect.", "Einstein won the Nobel Prize in 1968."]
Example actual output: "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect."

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "The node in the retrieval context states that Einstein won the Nobel Prize for his discovery of the photoelectric effect."
        }},
        {{
            "verdict": "no",
            "reason": "The node in the retrieval context states that Einstein won the Nobel Prize in 1968, not 1969."
        }}
    ]  
}}

You should NOT incorporate any prior knowledge you have and take each context at face value. Since you are going to generate a verdict for each context, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of contexts.
You DON'T have to provide a reason if the answer is 'yes'.
You should ONLY provide a 'no' answer if IT IS A CONTRADICTION.
**

Retrieval Contexts:
{truths}

Actual Output:
{text}

JSON:
"""

    @staticmethod
    def generate_reason(score, contradictions):
        return f"""Below is a list of Contradictions. It is a list of JSON with the `contradiction` and `rank` key.
The `contradiction` explains why the 'actual output' does not align with a certain node in the 'retrieval context'. Contradictions happen in the 'actual output', NOT the 'retrieval context'.
The `rank` tells you which node in the 'retrieval context' the actual output contradicted with.
Given the faithfulness score, which is a 0-1 score indicating how faithful the `actual output` is to the retrieval context (higher the better), CONCISELY summarize the contradictions to justify the score. 

Faithfulness Score:
{score}

Contradictions:
{contradictions}

Example:
The score is <faithfulness_score> because <your_reason>.

**
IMPORTANT: 
If there are no contradictions, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
Your reason MUST use information in `contradiction` and the node RANK (eg., first node of the retrieval context) in your reason.
Be sure in your reason, as if you know what the actual output is from the contradictions.
**

Reason:
"""


class AnswerRelevancyTemplate:
    @staticmethod
    def generate_key_points(answer, retrieval_context):
        return f"""Given the answer text, breakdown and generate a list of key points presented in the answer. In case the answer is ambigious to what it is talking about, you can use the retrieval contexts as additional information for more comphrensive key points. Make the key points concise.

Answer:
{answer}

Retrieval Context:
{retrieval_context}

**
IMPORTANT: Please make sure to only return in JSON format, with the "key_points" key as a list of strings. No words or explaination is needed.
**

JSON:
"""

    @staticmethod
    def generate_verdicts(original_question, key_points):
        return f"""For the provided list of key points, compare each key point with the question. Please generate a list of JSON with two keys: `verdict` and `reason`.
The 'verdict' key should STRICTLY be either a 'yes', 'no', or 'idk'. Answer 'yes' if it makes sense for the key point is relevant as an answer to the question, 'no' if the key point is irrelevant, and 'idk' if it is ambiguous (eg., not directly relevant but could be used as a supporting point to answer the question).
The 'reason' is the reason for the verdict.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example key points: ["Meditation offers a rich tapestry of benefits that touch upon various aspects of well-being.", "The practice of meditation has been around for centuries, evolving through various cultures and traditions, which underscores its timeless relevance.", "Improved sleep quality is another significant benefit, aiding in overall physical restoration."]

Example:
Question:
What are the primary benefits of meditation?

{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "Addresses the question directly, stating benefits of meditation."
        }},
        {{
            "verdict": "no",
            "reason": "The historical and cultural origins of meditation is not relevant to the question."
        }},
        {{
            "verdict": "yes",
            "reason": "Improved sleep quality is relevant a benefit of meditation."
        }}
    ]  
}}

Since you are going to generate a verdict for each question, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of `key points`.
**
           
Question:
{original_question}

Key Points:
{key_points}

JSON:
"""

    @staticmethod
    def generate_reason(irrelevant_points, original_question, answer, score):
        return f"""
Given the answer relevancy score, the list of irrelevant points, the list of ambiguous point, and the original question, summarize a CONCISE reason for the score. Explain why it is not higher, but also why it is at its current score.
The irrelevant points represent things in the original answer to the original question that is irrelevant to the question.
If there are nothing irrelevant, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).

Answer Relevancy Score:
{score}

Irrelevant Points:
{irrelevant_points}

Original Question:
{original_question}

Original Answer:
{answer}

Example:
The score is <answer_relevancy_score> because <your_reason>.

Reason:
"""


class ContextualRecallTemplate:
    @staticmethod
    def generate_reason(
        expected_output, supportive_reasons, unsupportive_reasons, score
    ):
        return f"""
Given the original expected output, a list of supportive reasons, and a list of unsupportive reasons (which is deduced directly from the 'expected output'), and a contextual recall score (closer to 1 the better), summarize a CONCISE reason for the score.
A supportive reason is the reason why a certain sentence in the original expected output can be attributed to the node in the retrieval context.
An unsupportive reason is the reason why a certain sentence in the original expected output cannot be attributed to anything in the retrieval context.
In your reason, you should related suportive/unsupportive reasons to the sentence number in expected output, and info regarding the node number in retrieval context to support your final reason. The first mention of "node(s)" should specify "node(s) in retrieval context)".

Contextual Recall Score:
{score}

Expected Output:
{expected_output}

Supportive Reasons:
{supportive_reasons}

Unsupportive Reasons:
{unsupportive_reasons}

Example:
The score is <contextual_recall_score> because <your_reason>.

**
IMPORTANT: DO NOT mention 'supportive reasons' and 'unsupportive reasons' in your reason, these terms are just here for you to understand the broader scope of things.
If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
**

Reason:
"""

    @staticmethod
    def generate_verdicts(expected_output, retrieval_context):
        return f"""
For EACH sentence in the given expected output below, determine whether the sentence can be attributed to the nodes of retrieval contexts. Please generate a list of JSON with two keys: `verdict` and `reason`.
The `verdict` key should STRICTLY be either a 'yes' or 'no'. Answer 'yes' if the sentence can be attributed to any parts of the retrieval context, else answer 'no'.
The `reason` key should provide a reason why to the verdict. In the reason, you should aim to include the node(s) count in the retrieval context (eg., 1st node, and 2nd node in the retrieval context) that is attributed to said sentence. You should also aim to quote the specific part of the retrieval context to justify your verdict, but keep it extremely concise and cut short the quote with an ellipsis if possible. 


**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects, each with two keys: `verdict` and `reason`.

{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "..."
        }},
        ...
    ]  
}}

Since you are going to generate a verdict for each sentence, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of sentences in of `expected output`.
**

Expected Output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""


class ContextualRelevancyTemplate:
    @staticmethod
    def generate_reason(input, irrelevant_sentences, score):
        return f"""Based on the given input, irrelevant sentences (list of JSON), and the contextual relevancy score (the closer to 1 the better), please generate a CONCISE reason for the score.
Irrelevant Sentences will contain JSONs with two keys: `sentence` and `node`. `sentence` is the actual sentence itself, and `node` is the node number from the `retrieval context` which it was drawn from. Specify that nodes are from retrieval context the first time you mention it.
In your reason, you should use data in the irrelevant sentences to support your point.

Contextual Relevancy Score:
{score}

Input:
{input}

Irrelevant Sentences:
{irrelevant_sentences}

Example:
The score is <contextual_relevancy_score> because <your_reason>.

** 
IMPORTANT:
If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
**

Reason:
"""

    @staticmethod
    def generate_verdicts(text, context):
        return f"""Based on the input and context, please generate a list of JSON objects to indicate whether each given sentence in the context relevant to the provided input. The JSON will have 1 mandatory field: 'verdict', and 1 optional field: 'sentence'.
The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the sentence is relevant to the text. 
Copy the sentence and supply the value to the 'sentence' key ONLY IF verdict is no.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example Context: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in 1968. There was a cat."
Example Input: "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect."

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "sentence": "There was a cat"
        }}
    ]  
}}
**

Input:
{text}

Context:
{context}

JSON:
"""


class ContextualPrecisionTemplate:
    @staticmethod
    def generate_verdicts(input, expected_output, retrieval_context):
        return f"""Given the input, expected output, and retrieval context, please generate a list of JSON objects to determine whether each node in the retrieval context was remotely useful in arriving at the expected output.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON. These JSON only contain the `verdict` key that outputs only 'yes' or 'no', and a `reason` key to justify the verdict. In your reason, you should aim to quote parts of the context.
Example Retrieval Context: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect", "He won the Nobel Prize in 1968.", "There was a cat."]
Example Input: "Who won the Nobel Prize in 1968 and for what?"
Example Expected Output: "Einstein won the Nobel Prize in 1968 for his discovery of the photoelectric effect."

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "It clearly addresses the question by stating that 'Einstein won the Nobel Prize for his discovery of the photoelectric effect.'"
        }},
        {{
            "verdict": "yes",
            "reason": "The text verifies that the prize was indeed won in 1968."
        }},
        {{
            "verdict": "no",
            "reason": "'There was a cat' is not at all relevant to the topic of winning a Nobel Prize."
        }}
    ]  
}}
Since you are going to generate a verdict for each context, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of the contexts.
**
        
Input:
{input}

Expected output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""

    @staticmethod
    def generate_reason(input, verdicts, score):
        # given the input and retrieval context for this input, where the verdict is whether ... and the node is the ..., give a reason for the score
        return f"""Given the input, retrieval contexts, and contextual precision score, provide a CONCISE summarize for the score. Explain why it is not higher, but also why it is at its current score.
The retrieval contexts is a list of JSON with three keys: `verdict`, `reason` (reason for the verdict) and `node`. `verdict` will be either 'yes' or 'no', which represents whether the corresponding 'node' in the retrieval context is relevant to the input. 
Contextual precision represents if the relevant nodes are ranked higher than irrelevant nodes. Also note that retrieval contexts is given IN THE ORDER OF THEIR RANKINGS.

Contextual Precision Score:
{score}

Input:
{input}

Retrieval Contexts:
{verdicts}

Example:
The score is <contextual_precision_score> because <your_reason>.

**
IMPORTANT: DO NOT mention 'verdict' in your reason, but instead phrase it as irrelevant nodes. The term 'verdict' are just here for you to understand the broader scope of things.
Also DO NOT mention there are `reason` fields in the retrieval contexts you are presented with, instead just use the information in the `reason` field.
In your reason, you MUST USE the `reason`, QUOTES in the 'reason', and the node RANK (starting from 1, eg. first node) to explain why the 'no' verdicts should be ranked lower than the 'yes' verdicts.
When addressing nodes, make it explicit that it is nodes in retrieval context.
If the score is 1, keep it short and say something positive with an upbeat tone (but don't overdo it otherwise it gets annoying).
**

Reason:
"""
