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


class SummarizationTemplate:
    @staticmethod
    def generate_reason(contradictions, redundancies, questions, score):
        return f"""You will be given the following: 1) information in the summary contradicting the original text, 2) extra information in the summary not mentioned in the original text, 3) [Optional] quesetions cannot be answered by the summary. Your task is to explain the quality of this summarization task.
Given the summarization score, which is a 0-1 score indicating how good the summary is to the original text (higher the better), CONCISELY summarize the provided information to justify the score.  

Example Reason:
The score is <summarization_score> because <your_reason>.

**
IMPORTANT: 
For 'None' values in contradictions, extra information, or questions that the original text can answer but not the summary, DON'T mention anything and instead offer some praise.
Be sure in your reason, as if you know what the summary and original text is.
**

Summarization Score:
{score}

Contradicting Information in the orignial text:
{contradictions}

Extra Information not mentioned in the original text:
{redundancies}


"""

    @staticmethod
    def generate_answers(questions, text):
        return f"""Based on the list of close-ended 'yes' or 'no' questions, generate a JSON with key 'answers', which is a list of strings that determines whether the provided text contains sufficient information to answer EACH question.
Answers should STRICTLY be either 'yes' or 'no'.
Answer 'no' if the provided text does not contain enough information to answer the question.
**
IMPORTANT: Please make sure to only return in JSON format, with the 'answers' key as a list of strings.

Example:
Example Text: Mario and Luigi were best buds but since Luigi had a crush on Peach Mario ended up killing him.
Example Questions: ["Are there enough information about Luigi and Mario?"]
Example Answers:
{{
    "answers": ["yes"]
}}

The length of 'answers' SHOULD BE STRICTLY EQUAL to that of questions.
===== END OF EXAMPLE ======

Text:
{text}

Questions:
{questions}

JSON:
"""

    @staticmethod
    def generate_alignment_verdicts(orignal_text, summary_claims):
        return f"""Based on the given summary claims, which is a list of strings, generate a list of JSON objects to indicate whether EACH piece of info contradicts any facts in the original text. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes', 'no', or 'idk', which states whether the given summary claim agrees with the original text. 
Provide a 'reason' ONLY if the answer is 'no' OR 'idk'. 
The provided summary claims is drawn from the summary. Try to provide a correction in the reason using the facts in the original text.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example Original Text: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
Example Summary Claims: ["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a Germen chef."]

Example:
{{
    "verdicts": [
        {{
            "verdict": "idk",
            "reason": "The original text does not mention Barack Obama at all, let alone his racial features.
        }},
        {{
            "verdict": "idk",
            "reason": "The original text does not mention Zurich, not does it mention Zurich being in London".
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "The summary claims Einstein won the Nobel Prize in 1969, which is untrue as the original text states it is 1968 instead."
        }},
        {{
            "verdict": "no",
            "reason": "The summary claims Einstein is a Germen chef, which is not correct as the original text states he was a German scientist instead."
        }},
    ]  
}}
===== END OF EXAMPLE ======

The length of 'verdicts' SHOULD BE STRICTLY EQUAL to that of summary claims.
You DON'T have to provide a reason if the answer is 'yes'.
ONLY provide a 'no' answer if the summary DIRECTLY CONTRADICTS the claims. YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
Claims made using vague, suggestive, speculative language such as 'may have', 'possibility due to', does NOT count as a contradiction.
Claims that is not backed up due to a lack of information/is not mentioned in the summary MUST be answered 'idk', otherwise I WILL DIE.
**

Original Text:
{orignal_text}

Summary Claims:
{summary_claims}

JSON:
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
    def generate_claims(text):
        return f"""Based on the given text, please generate a comphrensive list of FACTUAL claims that can inferred from the provided text.

Example:
Example Text: 
"Einstein won the noble prize in 1968 for his discovery of the photoelectric effect."

Example JSON: 
{{
    "claims": [
        "Einstein won the noble prize for his discovery of the photoelectric effect.",
        "Einstein won the noble prize in 1968."
    ]  
}}
===== END OF EXAMPLE ======

**
IMPORTANT: Please make sure to only return in JSON format, with the "claims" key as a list of strings. No words or explaination is needed.
Only include claims that are factual, and the claims you extract should include the full context it was presented in, NOT cherry picked facts.
You should NOT include any prior knowledge, and take the text at face value when extracting claims.
**

Text:
{text}

JSON:
"""

    @staticmethod
    def generate_truths(text):
        return f"""Based on the given text, please generate a comphrensive list of FACTUAL, undisputed truths that can inferred from the provided text.

Example:
Example Text: 
"Einstein won the noble prize in 1968 for his discovery of the photoelectric effect."

Example JSON: 
{{
    "truths": [
        "Einstein won the noble prize for his discovery of the photoelectric effect.",
        "Einstein won the noble prize in 1968."
    ]  
}}
===== END OF EXAMPLE ======

**
IMPORTANT: Please make sure to only return in JSON format, with the "truths" key as a list of strings. No words or explaination is needed.
Only include truths that are factual.
**

Text:
{text}

JSON:
"""

    @staticmethod
    def generate_verdicts(claims, retrieval_context):
        return f"""Based on the given claims, which is a list of strings, generate a list of JSON objects to indicate whether EACH claim contradicts any facts in the retrieval context. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes', 'no', or 'idk', which states whether the given claim agrees with the context. 
Provide a 'reason' ONLY if the answer is 'no'. 
The provided claim is drawn from the actual output. Try to provide a correction in the reason using the facts in the retrieval context.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example retrieval contexts: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
Example claims: ["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a Germen chef."]

Example:
{{
    "verdicts": [
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein won the Nobel Prize in 1969, which is untrue as the retrieval context states it is 1968 instead."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein is a Germen chef, which is not correct as the retrieval context states he was a German scientist instead."
        }},
    ]  
}}
===== END OF EXAMPLE ======

The length of 'verdicts' SHOULD BE STRICTLY EQUAL to that of claims.
You DON'T have to provide a reason if the answer is 'yes' or 'idk'.
ONLY provide a 'no' answer if the retrieval context DIRECTLY CONTRADICTS the claims. YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
Claims made using vague, suggestive, speculative language such as 'may have', 'possibility due to', does NOT count as a contradiction.
Claims that is not backed up due to a lack of information/is not mentioned in the retrieval contexts MUST be answered 'idk', otherwise I WILL DIE.
**

Retrieval Contexts:
{retrieval_context}

Claims:
{claims}

JSON:
"""

    @staticmethod
    def generate_reason(score, contradictions):
        return f"""Below is a list of Contradictions. It is a list of strings explaining why the 'actual output' does not align with the information presented in the 'retrieval context'. Contradictions happen in the 'actual output', NOT the 'retrieval context'.
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
Your reason MUST use information in `contradiction` in your reason.
Be sure in your reason, as if you know what the actual output is from the contradictions.
**

Reason:
"""


class AnswerRelevancyTemplate:
    @staticmethod
    def generate_statements(actual_output):
        return f"""Given the text, breakdown and generate a list of statements presented. Ambigious, single words, can also be statements.

Example:
Example text: Shoes. The shoes can be refunded at no extra cost. Thanks for asking the question!

{{
    "statements": ["Shoes.", "Shoes can be refunded at no extra cost", "Thanks for asking the question!"]
}}
===== END OF EXAMPLE ======
        
Text:
{actual_output}

**
IMPORTANT: Please make sure to only return in JSON format, with the "statements" key as a list of strings. No words or explaination is needed.
**

JSON:
"""

    @staticmethod
    def generate_verdicts(input, actual_output, retrieval_context):
        return f"""For the provided list of statements, see whether each statement is relevant to address the input.
Please generate a list of JSON with two keys: `verdict` and `reason`.
The 'verdict' key should STRICTLY be either a 'yes', 'idk' or 'no'. Answer 'yes' if the statement is relevant to addressing the original input, 'no' if the statement is irrelevant, and 'idk' if it is ambiguous (eg., not directly relevant but could be used as a supporting point to address the input). You can use the information in the retrieval context to support your deicison.
The 'reason' is the reason for the verdict.
Provide a 'reason' ONLY if the answer is 'no'. 
The provided statements are statements made in the actual output.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example statements: ["Shoes.", "Thanks for asking the question!", "Is there anything else I can help you with?", "Duck and hide"]
Example retrieval context: ["In the unlikely event of an earthquake, you should duck and hide under a table."]

Example:
Input: What should I do if there is an earthquake?

{{
    "verdicts": [
        {{
            "verdict": "no",
            "reason": "The 'Shoes.' statement made in the acutal output is completely irrelvant to the input, which asks about what to do in the event of an earthquake."
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }}
    ]  
}}

Since you are going to generate a verdict for each statement, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of `statements`.
**

Retrieval Context:
{retrieval_context}           

Input:
{input}

Statements:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_reason(irrelevant_statements, input, score):
        return f"""Given the answer relevancy score, the list of reasons of irrelevant statements made in the actual output, and the input, provide a CONCISE reason for the score. Explain why it is not higher, but also why it is at its current score.
The irrelevant statements represent things in the actual output that is ireelevant to addressing whatever is asked/talked about in the input.
If there are nothing irrelevant, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).

Answer Relevancy Score:
{score}

Reasons why the score can't be higher based on irrelevant statements from actual output:
{irrelevant_statements}

Input:
{input}

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
