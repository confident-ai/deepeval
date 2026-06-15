import textwrap


class ImageHelpfulnessTemplate:

    @staticmethod
    def evaluate_image_helpfulness(context_above, context_below):
        return textwrap.dedent(
            f"""
            # Task Description
            You are a multi-modal document evaluation assistant. You will receive an image and its textual context.
            Your task is to evaluate the helpfulness of the image in enabling human readers to comprehend the text (context above and below) it accompanies.

            # Context Above
            {context_above}

            # Context Below
            {context_below}

            # Image
            [The image is provided below this section.]

            # Scoring Criteria
            Evaluate how well the image helps human readers understand the content of its accompanying text, assigning a score from 0 to 10.
            A higher score indicates that the image significantly enhances comprehension of the text. Be precise when assigning the score.

            - A score from 0-3 means the image is minimally or not at all helpful for comprehension.
            - A score from 4-6 indicates the image provides some helpful context or information but may contain extraneous or less relevant details.
            - A score from 7-9 indicates the image is highly helpful in enabling comprehension of the text.
            - A score of 10 indicates the image perfectly enhances and clarifies the information provided in the text.

            Be rigorous and discerning when assigning your score.

            # Output Instructions
            Provide your evaluation in the following structured JSON format:
            {{
                "score": <integer between 0 and 10>,
                "reasoning": "<brief explanation for the assigned score>"
            }}

            # Image
            [Insert Image Here]
            """
        )
