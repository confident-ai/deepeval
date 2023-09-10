# Sample Metric for BLEU
import numpy as np

from .metric import Metric


class BLEUMetric(Metric):
    def __init__(self, minimum_score: float = 0.5):
        self.minimum_score = minimum_score

    def compute_bleu(
        self, candidate: str, reference: str, weights=(0.25, 0.25, 0.25, 0.25)
    ):
        """
        Compute BLEU score for a candidate sentence given a reference sentence.

        :param candidate: The candidate sentence as a string.
        :param reference: The reference sentence as a string.
        :param weights: Weights for the n-gram precisions, default is uniform (0.25 for each).
        :return: BLEU score.
        """
        import nltk
        from nltk.util import ngrams

        candidate = (
            candidate.split()
        )  # Convert the candidate string to a list of tokens
        reference = (
            reference.split()
        )  # Convert the reference string to a list of tokens

        precisions = []

        for i in range(1, 5):  # Compute BLEU for 1 to 4-grams
            candidate_ngrams = ngrams(candidate, i)
            candidate_ngram_freq = nltk.FreqDist(candidate_ngrams)

            reference_ngrams = ngrams(reference, i)
            reference_ngram_freq = nltk.FreqDist(reference_ngrams)

            clipped_counts = {
                ngram: min(
                    candidate_ngram_freq[ngram], reference_ngram_freq[ngram]
                )
                for ngram in candidate_ngram_freq
            }
            precision = sum(clipped_counts.values()) / sum(
                candidate_ngram_freq.values()
            )
            precisions.append(precision)

        brevity_penalty = min(1, len(candidate) / len(reference))

        bleu = brevity_penalty * np.exp(
            np.mean([w * np.log(p) for w, p in zip(weights, precisions)])
        )

        return bleu
