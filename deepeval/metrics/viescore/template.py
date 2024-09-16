from .task import VIEScoreTask
import textwrap


class VIEScoreTemplate:

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
    def generate_semantic_consistency_evaluation_results(
        text_prompt: str, task: VIEScoreTask
    ):
        if task == VIEScoreTask.TEXT_TO_IMAGE_GENERATION:
            return textwrap.dedent(
                f"""
                {VIEScoreTemplate.context}

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
        elif task == VIEScoreTask.TEXT_TO_IMAGE_EDITING:
            return textwrap.dedent(
                f"""
                {VIEScoreTemplate.context}

                RULES:
                                
                Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
                The objective is to evaluate how successfully the editing instruction has been executed in the second image.

                From scale 0 to 10: 
                A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
                A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
                Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

                Editing instruction: {text_prompt}
            """
            )

    @staticmethod
    def generate_perceptual_quality_evaluation_results():
        return textwrap.dedent(
            f"""
            {VIEScoreTemplate.context}

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
