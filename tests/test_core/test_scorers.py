import pytest

from deepeval.scorer import Scorer


class TestRougeScoreValidation:
    def test_invalid_score_type_raises_value_error(self):
        with pytest.raises(ValueError, match="score_type can be either"):
            Scorer.rouge_score("the cat sat", "the cat sat", "rougeX")

    def test_valid_score_type(self):
        pytest.importorskip("rouge_score")
        score = Scorer.rouge_score(
            "the cat sat on the mat", "the cat sat on the mat", "rouge1"
        )
        assert score == pytest.approx(1.0)


class TestSentenceBleuScoreValidation:
    def test_invalid_bleu_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid bleu_type"):
            Scorer.sentence_bleu_score(
                "the cat sat", "the cat sat", bleu_type="bleu9"
            )

    def test_valid_bleu_type(self):
        pytest.importorskip("nltk")
        try:
            score = Scorer.sentence_bleu_score(
                "the cat sat on the mat",
                "the cat sat on the mat",
                bleu_type="bleu1",
            )
        except LookupError:
            pytest.skip("nltk punkt tokenizer data not available")
        assert score == pytest.approx(1.0)


class TestAnswerRelevancyScoreValidation:
    def test_invalid_model_type_raises_value_error(self):
        with pytest.raises(ValueError, match="model_type can be either"):
            Scorer.answer_relevancy_score(
                "an answer", "a target", model_type="bogus"
            )

    def test_list_predictions_cross_encoder_raises_type_error(self):
        with pytest.raises(TypeError, match="cross_encoder"):
            Scorer.answer_relevancy_score(
                ["answer one", "answer two"],
                "a target",
                model_type="cross_encoder",
            )

    def test_list_predictions_default_model_type_raises_type_error(self):
        # model_type=None resolves to "cross_encoder", so a list of
        # predictions must be rejected the same way.
        with pytest.raises(TypeError, match="cross_encoder"):
            Scorer.answer_relevancy_score(
                ["answer one", "answer two"], "a target"
            )

    def test_valid_cross_encoder_input_reaches_model(self, monkeypatch):
        pytest.importorskip("sentence_transformers")
        from deepeval.models import answer_relevancy_model as arm

        class FakeCrossEncoder:
            def __init__(self, model_name=None):
                pass

            def __call__(self, prediction, target):
                return 0.42

        monkeypatch.setattr(
            arm, "CrossEncoderAnswerRelevancyModel", FakeCrossEncoder
        )
        score = Scorer.answer_relevancy_score(
            "an answer", "a target", model_type="cross_encoder"
        )
        assert score == 0.42
