import textwrap


class ImageReferenceTemplate:

    @staticmethod
    def evaluate_image_reference(context_above, context_below):
        return textwrap.dedent(
            f"""
            # Task Description
            You are a multi-modal document quality assessment assistant. You will receive an image and its accompanying textual context.
            Your task is to determine whether the image is explicitly referenced or explained within the surrounding text (both above and below the image).

            # Context Above
            {context_above}

            # Context Below
            {context_below}

            # Image
            [The image is provided below this section.]

            # Scoring Criteria
            Evaluate the extent to which the image is referenced or explained in the text, assigning a score from 0 to 10:
            - 0: The image is not mentioned or referenced in the context.
            - 1-3: The image is referenced implicitly, and the reference is improper or incorrect.
            - 4-6: The image is referenced explicitly but in an improper manner, or it is referenced implicitly.
            - 7-9: The image is referenced explicitly, with the reference being generally proper and correct.
            - 10: The image is referenced explicitly, with the placement and explanation being completely proper and correct.

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
