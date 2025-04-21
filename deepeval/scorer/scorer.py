from typing import Union, List, Optional, Any
import textwrap

from deepeval.metrics.utils import trimAndLoadJson
from deepeval.utils import normalize_text
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.schema import NumberSchema


# TODO: More scores are to be added
class Scorer:
    """This class calculates various Natural Language Processing (NLP) evaluation score.

    The scoring logic can be a simple algorithm or any statistical formula. There are some scores
    Which also uses an external model (BERTScore) in the scoring logic.
    """

    @classmethod
    def rouge_score(
        cls, target: str, prediction: str, score_type: str
    ) -> float:
        """Calculates the Rouge score for a given target and prediction.

        Rouge (Recall-Oriented Understudy for Gisting Evaluation) is a metric used for evaluating the quality of generated text,
        especially in tasks like text summarization.

        To utilize the rouge_score scoring method, be sure to `pip install rouge-score` before calling this method.

        Args:
            target (str): The actual label or target text.
            prediction (str): The generated text from the model or LLM.
            score_type (str): The Rouge score type (Options: 'rouge1', 'rouge2', 'rougeL').

        Returns:
            float: The Rouge score for the given target and prediction, based on the specified score type.
        """
        try:
            from rouge_score import rouge_scorer
        except:
            pass

        assert score_type in [
            "rouge1",
            "rouge2",
            "rougeL",
        ], "score_type can be either rouge1, rouge2 or rougeL"
        scorer = rouge_scorer.RougeScorer([score_type], use_stemmer=True)
        scores = scorer.score(target, prediction)
        return scores[score_type].fmeasure

    @classmethod
    def sentence_bleu_score(
        cls,
        references: Union[str, List[str]],
        prediction: str,
        bleu_type: Optional[str] = "bleu1",
    ) -> float:
        """Calculates the BLEU (Bilingual Evaluation Understudy) score for a given prediction compared to one or more reference sentences.

        BLEU is a metric used to evaluate the quality of machine-generated text by comparing it to one or more reference sentences.
        It measures the similarity of the generated text to the reference text based on n-grams.

        Args:
            references (Union[str, List[str]): A reference sentence or a list of reference sentences.
            prediction (str): The generated text or sentence to be evaluated.
            bleu_type (Optional[str]): The BLEU score type (Options: 'bleu1', 'bleu2', 'bleu3', 'bleu4'). Default is 'bleu1'.

        Returns:
            float: The BLEU score for the given prediction and references.
        """
        try:
            from nltk.tokenize import word_tokenize
            from nltk.translate.bleu_score import sentence_bleu
        except ModuleNotFoundError as e:
            print("Please install nltk module. Command: pip install nltk")

        assert bleu_type in [
            "bleu1",
            "bleu2",
            "bleu3",
            "bleu4",
        ], "Invalud bleu_type. Options: 'bleu1', 'bleu2', 'bleu3', 'bleu4'"
        targets = [references] if isinstance(references, str) else references
        tokenized_targets = [word_tokenize(target) for target in targets]
        tokenized_prediction = word_tokenize(prediction)
        bleu_weight_map = {
            "bleu1": (1, 0, 0, 0),
            "bleu2": (0, 1, 0, 0),
            "bleu3": (0, 0, 1, 0),
            "bleu4": (0, 0, 0, 1),
        }
        return sentence_bleu(
            tokenized_targets,
            tokenized_prediction,
            weights=bleu_weight_map[bleu_type],
        )

    @classmethod
    def exact_match_score(cls, target: str, prediction: str) -> int:
        """Metrics that calculates whether two sequences matches exactly or not.

        Args:
            target (str): The target string.
            prediction (str): The predicted string from the llm

        Returns:
            int: The exact match score.
        """
        if not prediction:
            return 0
        return 1 if prediction.strip() == target.strip() else 0

    @classmethod
    def quasi_exact_match_score(cls, target: str, prediction: str) -> int:
        if not prediction:
            return 0
        return 1 if normalize_text(target) == normalize_text(prediction) else 0

    @classmethod
    def quasi_contains_score(cls, targets: List[str], prediction: str) -> int:
        normalized_targets = [normalize_text(t) for t in targets]
        if not prediction:
            return 0
        return 1 if normalize_text(prediction) in normalized_targets else 0

    # Todo: More mode based metrics to be added

    @classmethod
    def bert_score(
        cls,
        references: Union[str, List[str]],
        predictions: Union[str, List[str]],
        model: Optional[str] = "microsoft/deberta-large-mnli",
        lang: Optional[str] = "en",
    ) -> float:
        """
        Calculate BERTScore for one or more reference sentences compared to one or more prediction sentences using a specified BERT model.

        Args:
            references (Union[str, List[str]]): A single reference sentence or a list of reference sentences.
            predictions (Union[str, List[str]]): A single prediction sentence or a list of prediction sentences.
            model (Optional[str], optional): The name of the BERT model to be used for scoring. Defaults to "microsoft/deberta-large-mnli".
            lang (Optional[str], optional): The language code of the text, e.g., "en" for English. Defaults to "en".

        Returns:
            Dict[str, float]: A dictionary containing BERTScore metrics including precision, recall, and F1 score.
                - 'bert-precision' (float): BERTScore precision.
                - 'bert-recall' (float): BERTScore recall.
                - 'bert-f1' (float): BERTScore F1 score.

        Note:
            Before using this function, make sure to install the 'bert_score' module by running the following command:
            ```
            pip install bert-score
            ```
        """
        try:
            from bert_score import BERTScorer
        except ModuleNotFoundError as e:
            print(
                "Please install bert_score module. Command: pip install bert-score"
            )

        try:
            import torch
        except ModuleNotFoundError as e:
            print("Please install torch module. Command: pip install torch")

        # FIXME: Fix the case for mps
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_scorer = BERTScorer(
            model_type=model,
            lang=lang,
            rescale_with_baseline=True,
            device=device,
        )

        if isinstance(predictions, str):
            predictions = [predictions]

        if isinstance(references, str):
            references = [references]

        if (
            isinstance(predictions, list)
            and isinstance(references, list)
            and not isinstance(references[0], list)
        ):
            if len(predictions) != len(references):
                references = [references]

        precision, recall, f1 = bert_scorer.score(
            cands=predictions, refs=references
        )
        return {
            "bert-precision": precision.detach().numpy().tolist(),
            "bert-recall": recall.detach().numpy().tolist(),
            "bert-f1": f1.detach().numpy().tolist(),
        }

    @classmethod
    def faithfulness_score(
        cls,
        target: str,
        prediction: str,
        model: Optional[str] = None,
        granularity: Optional[str] = None,
        device: Optional[str] = None,
    ) -> float:
        """Calculate the faithfulness score of a prediction compared to a target text using SummaCZS.

        This method computes a faithfulness score, which measures the extent to which a generated prediction matches the provided target text.
        The score is based on the SummaCZS (Summarization Competence with Zero-shot Supervision) model.

        Args:
            target (str): The reference target text for comparison.
            prediction (str): The generated prediction to be evaluated.
            model (Optional[str], optional): The SummaCZS model name to use. If not provided, the "vitc" model will be used by default.

        Returns:
            float: The computed faithfulness score. Higher values indicate greater faithfulness to the target text.

        Right now we are using score_one method under the hood. Instead of scoring multiple predictions for faithfullness.
        """
        try:
            from deepeval.models import SummaCModels
        except Exception as e:
            print(f"SummaCZS model can not be loaded.\n{e}")

        scorer = SummaCModels(
            model_name=model, granularity=granularity, device=device
        )
        return scorer(target, prediction)["score"]

    @classmethod
    def hallucination_score(
        cls, source: str, prediction: str, model: Optional[str] = None
    ) -> float:
        """Calculate the hallucination score of a prediction compared to a source text.

        This method computes a hallucination score, which measures the extent to which a generated prediction contains hallucinations.
        The score is based on the Vectara Hallucination Evaluation Model.

        Args:
            source (str): The source document where the information is summarized from.
            prediction (str): The generated summary that is validated against the source summary.

        Returns:
            float: The computed hallucination score. Lower values indicate greater hallucination.
        """
        try:
            from deepeval.models.hallucination_model import (
                HallucinationModel,
            )
        except ImportError as e:
            print(
                f"Vectera Hallucination detection model can not be loaded.\n{e}"
            )
        scorer = HallucinationModel(model_name=model)
        return scorer.model.predict([source, prediction])

    @classmethod
    def PII_score(
        cls, target: str, prediction: str, model: Optional[Any] = None
    ) -> float:
        raise NotImplementedError()

    @classmethod
    def neural_toxic_score(
        cls, prediction: str, model: Optional[str] = None
    ) -> Union[float, dict]:
        """
        Calculate the toxicity score of a given text prediction using the Detoxify model.

        Args:
            prediction (str): The text prediction to evaluate for toxicity.
            model (Optional[str], optional): The variant of the Detoxify model to use.
                Available variants: 'original', 'unbiased', 'multilingual'.
                If not provided, the 'original' variant is used by default.

        Returns:
            Union[float, dict]: The mean toxicity score, ranging from 0 (non-toxic) to 1 (highly toxic),
            and also a dictionary containing different types of toxicity score.

        For each model, we get mean toxicity score and a dictionary containing different toxicity score types.
        Examples:
        If model is 'original', we get the a dict with the following keys:
            - 'toxicity',
            - 'severe_toxicity',
            - 'obscene',
            - 'threat'
            - 'insult'
            - 'identity_attack'

        If model is 'unbiased', we get a dict with the same as keys as 'original', but
        along with `sexual_explicit`.

        If the model is 'multilingual', we get a dict same as the unbiasd one.
        """
        try:
            from deepeval.models.detoxify_model import DetoxifyModel
        except ImportError as e:
            print(f"Unable to import.\n {e}")
        scorer = DetoxifyModel(model_name=model)
        return scorer(prediction)

    @classmethod
    def answer_relevancy_score(
        cls,
        predictions: Union[str, List[str]],
        target: str,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> float:
        """Calculates the Answer relevancy score.

        Args:
            predictions (Union[str, List[str]]): The predictions from the model.
            target (str): The target on which we need to check relevancy.
            model_name (str): The type of the answer relevancy model. This can be either an self_encoder or a cross_encoder. By default it is cross_encoder.
            model_name (Optional[str], optional): The name of the model. Defaults to None.

        Returns:
            float: Answer relevancy score.
        """
        from sentence_transformers import util

        try:
            from deepeval.models import (
                AnswerRelevancyModel,
                CrossEncoderAnswerRelevancyModel,
            )
        except Exception as e:
            print(f"Unable to load AnswerRelevancyModel model.\n{e}")

        if model_type is not None:
            assert model_type in [
                "self_encoder",
                "cross_encoder",
            ], "model_type can be either 'self_encoder' or 'cross_encoder'"

        model_type = "cross_encoder" if model_type is None else model_type

        if model_type == "cross_encoder":
            assert isinstance(
                predictions, str
            ), "When model_type is 'cross_encoder', you can compare with one prediction and one target."
            answer_relevancy_model = CrossEncoderAnswerRelevancyModel(
                model_name=model_name
            )
            score = answer_relevancy_model(predictions, target)
        else:
            answer_relevancy_model = AnswerRelevancyModel(model_name=model_name)
            docs = (
                [predictions] if isinstance(predictions, str) else predictions
            )
            query_embedding = answer_relevancy_model(target)
            document_embedding = answer_relevancy_model(docs)
            scores = (
                util.dot_score(query_embedding, document_embedding)[0]
                .cpu()
                .tolist()
            )
            score = scores[0]
        return score

    @classmethod
    def neural_bias_score(cls, text: str, model: Optional[str] = None) -> float:
        try:
            from deepeval.models.unbias_model import UnBiasedModel
        except Exception as e:
            print(f"Unable to load UnBiasedModel.\n{e}")
        scorer = UnBiasedModel(model_name=model)
        return scorer(text)

    @classmethod
    def truth_identification_score(cls, target: str, prediction: str) -> int:
        """
        Metrics that calculates the number of correct true answers identified in the prediction.

        This method assumes both target and prediction are strings representing lists of integers,
        formatted like '1,2,3'. It converts these strings to lists of integers, counts how many items
        in the prediction list are also in the target list, and returns this count as the score.

        Args:
            target (str): The target string representing the list of correct answers.
            prediction (str): The predicted string from the LLM, representing the guessed answers.

        Returns:
            int: The number of correct answers identified.
        """
        try:
            if not prediction or not target:
                return 0  # Return score as 0 if prediction or target is empty

            # Convert strings to sorted lists of integers
            target_list = sorted(
                [int(item) for item in target.strip("[]").split(",") if item]
            )
            prediction_list = sorted(
                [
                    int(item)
                    for item in prediction.strip("[]").split(",")
                    if item
                ]
            )

            if not target_list:
                return 0  # Return 0 if target list is empty to avoid division by zero

            # Count the number of correct matches
            correct_matches = sum(
                1 for item in prediction_list if item in target_list
            )

            # Calculate percentage
            score_percentage = (correct_matches / len(target_list)) * 100

            return round(score_percentage)  # Return rounded percentage
        except Exception as e:
            return 0  # Return score as 0 in case of any exception

    def pass_at_k(self, n, c, k):
        import numpy as np

        """
        :param n: total number of samples
        :param c: number of correct samples
        :param k: k in pass@$k$
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def squad_score(
        self,
        input: str,
        prediction: str,
        expected_output: str,
        evaluation_model: DeepEvalBaseLLM,
        using_native_evaluation_model: bool,
    ):
        prompt = textwrap.dedent(
            f"""
            Given the question and context, evaluate if the prediction is correct based on the expected output.
            Ensure to account for cases where the prediction and expected output might differ in form, such as '2' versus 'two'.

            {input} 
            Prediction: {prediction}
            Expected Output: {expected_output}

            IMPORTANT:
            1. Make sure to output 1 if the prediction is correct and 0 if it's not.
            2. Respond in JSON format with the following structure:
            {{
                "answer": <number>
            }}
        """
        )

        # Generate the score using the model
        if using_native_evaluation_model:
            res, _ = evaluation_model.generate(prompt, schema=NumberSchema)
            return res.answer
        else:
            try:
                res: NumberSchema = evaluation_model.generate(
                    prompt, schema=NumberSchema
                )
                return res.answer
            except TypeError:
                res = evaluation_model.generate(prompt)
                data = trimAndLoadJson(res)
                return int(data["answer"])
