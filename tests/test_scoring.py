"""Tests for metrics calculator
"""

import unittest
from deepeval.metrics.scoring import Scorer


class TestScorer(unittest.TestCase):
    # tests for exact_match_score

    def test_exact_equal_strings(self):
        target = "Winners of the FIFA world cup were the French national football team"
        prediction = "Winners of the FIFA world cup were the French national football team"
        self.assertEqual(Scorer.exact_match_score(target, prediction), 1)

    def test_exact_match_score_unequal_strings(self):
        target = "Hello, World!"
        prediction = "Goodbye, World!"
        self.assertEqual(Scorer.exact_match_score(target, prediction), 0)

    def test_exact_match_score_whitespace_difference(self):
        target = "Hello, World!"
        prediction = " Hello, World! "
        self.assertEqual(Scorer.exact_match_score(target, prediction), 1)

    def test_exact_match_score_empty_prediction(self):
        target = "Hello, World!"
        prediction = ""
        self.assertEqual(Scorer.exact_match_score(target, prediction), 0)

    def test_exact_match_score_empty_target(self):
        target = ""
        prediction = "Hello, World!"
        self.assertEqual(Scorer.exact_match_score(target, prediction), 0)

    # test for quasi_exact_match_score

    def test_exact_match(self):
        target = "The quick brown fox"
        prediction = "The quick brown fox"
        self.assertEqual(Scorer.quasi_exact_match_score(target, prediction), 1)

    def test_case_insensitive_match(self):
        target = "The quick brown fox"
        prediction = "the quick brown fox"
        self.assertEqual(Scorer.quasi_exact_match_score(target, prediction), 1)

    def test_partial_match(self):
        target = "The quick brown fox"
        prediction = "The brown fox"
        self.assertEqual(Scorer.quasi_exact_match_score(target, prediction), 0)

    def test_empty_prediction(self):
        target = "The quick brown fox"
        prediction = ""
        self.assertEqual(Scorer.quasi_exact_match_score(target, prediction), 0)

    # Testing for rouge score 1/2/L

    def test_rouge_score_rouge1(self):
        target = "The quick brown fox"
        prediction = "The quick brown fox"
        score_type = "rouge1"
        rouge_score = Scorer.rouge_score(target, prediction, score_type)
        self.assertAlmostEqual(rouge_score, 1.0, places=2)

    def test_rouge_score_rouge2(self):
        target = "The quick brown fox"
        prediction = "The quick brown fox"
        score_type = "rouge2"
        rouge_score = Scorer.rouge_score(target, prediction, score_type)
        self.assertAlmostEqual(rouge_score, 1.0, places=2)

    def test_rouge_score_rougeL(self):
        target = "The quick brown fox"
        prediction = "The quick brown fox"
        score_type = "rougeL"
        rouge_score = Scorer.rouge_score(target, prediction, score_type)
        self.assertAlmostEqual(rouge_score, 1.0, places=2)

    # Testing sentence BLEU score 1/4

    def test_sentence_bleu_score_bleu1(self):
        references = ["The quick brown fox jumps over the lazy dog"]
        prediction = "The quick brown fox jumps over the lazy dog"
        bleu_type = "bleu1"
        bleu_score = Scorer.sentence_bleu_score(
            references, prediction, bleu_type
        )
        self.assertAlmostEqual(bleu_score, 1.0, places=2)

    def test_sentence_bleu_score_bleu4(self):
        references = ["The quick brown fox jumps over the lazy dog"]
        prediction = "The quick brown fox jumps over the lazy dog"
        bleu_type = "bleu4"
        bleu_score = Scorer.sentence_bleu_score(
            references, prediction, bleu_type
        )
        self.assertAlmostEqual(bleu_score, 1.0, places=2)

    # Adding tests for mismatch for rouge and sentence BLEU

    def test_rouge_score_mismatch(self):
        target = "The quick brown fox"
        prediction = "The lazy dog"
        score_type = "rouge1"
        rouge_score = Scorer.rouge_score(target, prediction, score_type)
        self.assertNotAlmostEqual(rouge_score, 0.0, places=2)

    def test_sentence_bleu_score_mismatch(self):
        references = ["The quick brown fox jumps over the lazy dog"]
        prediction = "The lazy cat"
        bleu_type = "bleu1"
        bleu_score = Scorer.sentence_bleu_score(
            references, prediction, bleu_type
        )
        self.assertNotAlmostEqual(bleu_score, 0.0, places=2)

    # Tests for Bert score

    def test_bert_score_single_reference_single_prediction(self):
        reference = "The quick brown fox jumps over the lazy dog"
        prediction = "The quick brown fox jumps over the lazy dog"
        bert_scores = Scorer.bert_score(reference, prediction)
        self.assertTrue(isinstance(bert_scores, dict))
        self.assertIn("bert-precision", bert_scores)
        self.assertIn("bert-recall", bert_scores)
        self.assertIn("bert-f1", bert_scores)

    def test_bert_score_single_reference_multiple_predictions(self):
        reference = "The quick brown fox jumps over the lazy dog"
        predictions = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox jumps over a sleeping dog",
        ]
        bert_scores = Scorer.bert_score(reference, predictions)
        self.assertTrue(isinstance(bert_scores, dict))
        self.assertIn("bert-precision", bert_scores)
        self.assertIn("bert-recall", bert_scores)
        self.assertIn("bert-f1", bert_scores)

    def test_bert_score_multiple_references_single_prediction(self):
        references = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox jumps over a sleeping dog",
        ]
        prediction = "The quick brown fox jumps over the lazy dog"
        bert_scores = Scorer.bert_score(references, prediction)
        self.assertTrue(isinstance(bert_scores, dict))
        self.assertIn("bert-precision", bert_scores)
        self.assertIn("bert-recall", bert_scores)
        self.assertIn("bert-f1", bert_scores)

    def test_bert_score_multiple_references_multiple_predictions(self):
        references = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox jumps over a sleeping dog",
        ]
        predictions = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox jumps over a sleeping dog",
        ]
        bert_scores = Scorer.bert_score(references, predictions)
        self.assertTrue(isinstance(bert_scores, dict))
        self.assertIn("bert-precision", bert_scores)
        self.assertIn("bert-recall", bert_scores)
        self.assertIn("bert-f1", bert_scores)
