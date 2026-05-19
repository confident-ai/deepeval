import textwrap
from typing import List


class SimulationGraphTemplate:
    @staticmethod
    def classify_edge(assistant_reply: str, choices: List[str]) -> str:
        """Prompt the simulator model to pick the index of the outgoing edge
        that best describes the assistant's reply. A "None of the above"
        option is appended automatically.
        """
        numbered_choices = "\n".join(
            f"  {i}) {choice}" for i, choice in enumerate(choices, start=1)
        )
        none_index = len(choices) + 1
        prompt = textwrap.dedent(
            f"""You are routing a simulated conversation. The assistant just said:
            <<<
            {assistant_reply}
            >>>

            Which of the following best describes the assistant's reply?
            {numbered_choices}
              {none_index}) None of the above

            Pick exactly one option. Respond with the option's number.

            IMPORTANT: The output must be formatted as a JSON object with two keys:
            - `index`: the 1-based number of the option you chose, OR `null` if you chose "None of the above" (option {none_index}).
            - `reason`: a short rationale (one sentence).

            Example JSON Output for a match:
            {{
                "index": 1,
                "reason": "The reply explicitly approved the refund."
            }}

            Example JSON Output for no match:
            {{
                "index": null,
                "reason": "The reply asked a clarifying question that does not fit any option."
            }}

            JSON Output:
            """
        )
        return prompt
