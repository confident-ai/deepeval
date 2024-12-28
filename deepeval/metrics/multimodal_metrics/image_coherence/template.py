import textwrap


class ImageCoherenceTemplate:

    @staticmethod
    def evaluate_image_coherence(context_above, context_below):
        return textwrap.dedent(
            f"""
            # Task Description
            You are a multi-modal document evaluation assistant. You will receive an image and its textual context. 
            Your task is to evaluate the coherence between the image and the text (context above and below) it accompanies.

            # Context Above
            {context_above}

            # Context Below
            {context_below}

            # Image
            [The image is provided below this section.]

            # Scoring Criteria
            Assess how coherent the image is in relation to its accompanying text, assigning a score from 0 to 10. 
            A higher score indicates stronger coherence between the image and the text. Be precise when assigning the score.

            - A score from 0-3 means that the image is minimally or not at all coherent with the text.
            - A score from 4-6 indicates that the image shows some coherence with the text but may include unrelated elements.
            - A score from 7-9 indicates that the image is highly coherent with the text.
            - A score of 10 indicates perfect coherence, where the image completely corresponds with and enhances the text.

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
