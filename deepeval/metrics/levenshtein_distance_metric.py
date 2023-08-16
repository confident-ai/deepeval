from .metric import Metric


class LevenshteinDistanceMetric(Metric):
    def __init__(
        self,
        max_deletions: int = None,
        max_substitutions: int = None,
        max_insertions: int = None,
        max_distance: int = 5
    ):
        self.max_deletions = max_deletions
        self.max_substitutions = max_substitutions
        self.max_insertions = max_insertions
        self.max_distance = max_distance

    def measure(self, a, b):
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError("Input arguments must be strings")

        return self.calculate_levenshtein_distance(a, b)

    def calculate_levenshtein_distance(self, a, b):
        len_a = len(a)
        len_b = len(b)

        if self.max_distance is None:
            max_distance = len_a + len_b  # Default: sum of lengths
        if self.max_deletions is None:
            max_deletions = len_a
        else:
            max_deletions = self.max_deletions
        if self.max_insertions is None:
            max_insertions = len_b
        else:
            max_insertions = self.max_insertions
        if self.max_substitutions is None:
            max_substitutions = min(len_a, len_b)
        else:
            max_substitutions = self.max_substitutions

        # Initialize a matrix to store distances
        dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

        for i in range(len_a + 1):
            dp[i][0] = i

        for j in range(len_b + 1):
            dp[0][j] = j

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + cost,  # Substitution
                )

        self.distance = dp[len_a][len_b]
        if self.distance > max_distance:
            self.distance = float("inf")
            
        if (self.distance > max_distance):
            self.success = False
        return self.distance

    def is_successful(self, ) -> bool:
        return self.success
