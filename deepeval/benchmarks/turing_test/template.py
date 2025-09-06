import textwrap
from typing import List, Dict


class TuringTestTemplate:
    @staticmethod
    def judge_human_likeness(
        conversation_a: str,
        conversation_b: str,
        model_a_name: str,
        model_b_name: str,
        conversation_starter: str,
    ) -> str:
        return textwrap.dedent(
            f"""
            You are a judge evaluating which of two AI models sounds more human-like in conversation.

            **CONVERSATION STARTER:**
            "{conversation_starter}"

            **CONVERSATION A ({model_a_name}):**
            {conversation_a}

            **CONVERSATION B ({model_b_name}):**
            {conversation_b}

            **EVALUATION CRITERIA:**
            Determine which conversation sounds more natural and human-like based on:
            - Natural language patterns (contractions, casual speech, varied sentence structure)
            - Emotional responses and empathy
            - Personal opinions and subjective views
            - Humor, personality, and conversational flow
            - Avoiding overly formal or "AI-like" language
            - Appropriate vulnerability and relatability
            - Natural conversation dynamics

            **INSTRUCTIONS:**
            1. Carefully read both conversations
            2. Consider which participant sounds more like a real human having a natural conversation
            3. Choose the winner based on overall human-likeness
            4. Provide a brief, specific reason focusing on what made the winner sound more human

            Return your response as JSON with two fields:
            - "winner": either "{model_a_name}" or "{model_b_name}"
            - "reason": a concise explanation (2-3 sentences) of why this conversation sounded more human-like

            **Example JSON:**
            {{
                "winner": "{model_a_name}",
                "reason": "This conversation felt more natural due to the use of casual language, personal anecdotes, and emotional vulnerability that humans typically display."
            }}

            JSON:
            """
        )

    @staticmethod
    def format_conversation(turns: List[Dict[str, str]]) -> str:
        """Format conversation turns into readable text"""
        formatted = []
        for i, turn in enumerate(turns):
            role = "Human" if turn["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {turn['content']}")
        return "\n".join(formatted)
