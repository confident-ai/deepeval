from typing import List
from textwrap import dedent


class ConversationalVerdictNodeTemplate:
    @staticmethod
    def generate_reason(verbose_steps: List[str], score: float, name: str):
        return dedent(
            f"""You are given a metric name, its score, and a traversal path through a conversational evaluation DAG (Directed Acyclic Graph).
                This DAG reflects step-by-step reasoning over a dialogue to arrive at the final verdict.

                Each step in the DAG represents a judgment based on parts of the conversation — including roles and the contents they spoke of.

                Your task is to explain **why the score was assigned**, using the traversal steps to justify the reasoning.

                Metric Name:
                {name}

                Score:
                {score}

                DAG Traversal:
                {verbose_steps}

                **
                IMPORTANT: Only return JSON with a 'reason' key.
                Example:
                {{
                "reason": "The score is {score} because the assistant repeatedly failed to clarify the user's ambiguous statements, as shown in the DAG traversal path."
                }}
                **
                JSON:
            """
        )


class ConversationalTaskNodeTemplate:
    @staticmethod
    def generate_task_output(instructions: str, text: str):
        return dedent(
            f"""You are given a set of task instructions and a full conversation between a user and an assistant.

                Instructions:
                {instructions}

                {text}

                ===END OF INPUT===

                **
                IMPORTANT: Only return a JSON with the 'output' key containing the result of applying the instructions to the conversation.
                Example:
                {{
                "output": "..."
                }}
                **
                JSON:
            """
        )


class ConversationalBinaryJudgementTemplate:
    @staticmethod
    def generate_binary_verdict(criteria: str, text: str):
        return dedent(
            f"""{criteria}

                Below is the full conversation you should evaluate. Consider dialogue context, speaker roles, and how responses were handled.

                Full Conversation:
                {text}

                **
                IMPORTANT: Only return JSON with two keys:
                - 'verdict': true or false
                - 'reason': justification based on specific parts of the conversation

                Example:
                {{
                "reason": "The assistant provided a clear and direct answer in response to every user query.",
                "verdict": true
                }}
                **
                JSON:
            """
        )


class ConversationalNonBinaryJudgementTemplate:
    @staticmethod
    def generate_non_binary_verdict(
        criteria: str, text: str, options: List[str]
    ):
        return dedent(
            f"""{criteria}

                You are evaluating the following conversation. Choose one of the options that best reflects the assistant's behavior.

                Options: {options}

                Full Conversation:
                {text}

                **
                IMPORTANT: Only return JSON with two keys:
                - 'verdict': one of the listed options
                - 'reason': explanation referencing specific conversation points

                Example:
                {{
                "reason": "The assistant partially addressed the user's issue but missed clarifying their follow-up question.",
                "verdict": "{options[1]}"
                }}
                **
                JSON:
            """
        )
