class SummarizationTemplate:
    @staticmethod
    def generate_reason(contradictions, redundancies, questions, score):
        return f"""You will be given the following: 1) information in the summary contradicting the original text, 2) extra information in the summary not mentioned in the original text, 3) [Optional] questions cannot be answered by the summary. Your task is to explain the quality of this summarization task.
Given the summarization score, which is a 0-1 score indicating how good the summary is to the original text (higher the better), CONCISELY summarize the provided information to justify the score.  

** 
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <summarization_score> because <your_reason>."
}}

For 'None' values in contradictions, extra information, or questions that the original text can answer but not the summary, DON'T mention anything and instead offer some praise.
Be sure in your reason, as if you know what the summary and original text is.
**

Summarization Score:
{score}

Contradicting Information in the original text:
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
    def generate_questions(text, n):
        return f"""Based on the given text, generate {n} closed-ended questions that can be answered with either a 'yes' or 'no'. 
The questions generated should ALWAYS result in a 'yes' based on the given text. 
        
** IMPORTANT
Only return a JSON with a 'questions' key, which is a list of strings. 
The questions have to be STRICTLY closed ended.
The given text should be able to answer 'yes' for each question.
**
Text:
{text}

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
