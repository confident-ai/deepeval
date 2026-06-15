import textwrap


class TextToImageTemplate:

    context = textwrap.dedent(
        """
        You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
        All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.
                              
        You will have to give your output in this way (Keep your reasoning concise and short.):
        {
            "score" : [...],
            "reasoning" : "..."
        }
    """
    )

    @staticmethod
    def generate_semantic_consistency_evaluation_results(text_prompt: str):
        return textwrap.dedent(
            f"""
            {TextToImageTemplate.context}

            RULES:
                            
            The image is an AI-generated image according to the text prompt.
            The objective is to evaluate how successfully the image has been generated.

            From scale 0 to 10: 
            A score from 0 to 10 will be given based on the success in following the prompt. 
            (0 indicates that the AI generated image does not follow the prompt at all. 10 indicates the AI generated image follows the prompt perfectly.)

            Put the score in a list such that output score = [score].

            Text Prompt: {text_prompt}
        """
        )

    @staticmethod
    def generate_perceptual_quality_evaluation_results():
        return textwrap.dedent(
            f"""
            {TextToImageTemplate.context}

            RULES:

            The image is an AI-generated image.
            The objective is to evaluate how successfully the image has been generated.

            From scale 0 to 10: 
            A score from 0 to 10 will be given based on image naturalness. 
            (
                0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
                10 indicates that the image looks natural.
            )
            A second score from 0 to 10 will rate the image artifacts. 
            (
                0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
                10 indicates the image has no artifacts.
            )
            Put the score in a list such that output score = [naturalness, artifacts]
        """
        )
