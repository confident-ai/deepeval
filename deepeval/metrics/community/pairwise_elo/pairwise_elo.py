import math
from typing import Dict, List, Optional, Tuple


class PairwiseElo:
    """Run-level Elo ratings from pairwise model comparisons.

    Unlike per-test-case metrics, Elo aggregates wins across a run
    (for example the winners reported by ``ArenaGEval``): call
    ``record`` once per comparison, then read ``ratings`` or
    ``leaderboard``. ``win_rate`` is the plain share of non-draw games
    won. Fully **deterministic**: no LLM, no API key, zero token cost.
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
    ):
        if k_factor <= 0:
            raise ValueError(f"`k_factor` must be positive, got {k_factor}.")
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self._ratings: Dict[str, float] = {}
        self._games: Dict[str, int] = {}
        self._score: Dict[str, float] = {}

    def rating(self, model: str) -> float:
        return self._ratings.get(model, self.initial_rating)

    def games(self, model: str) -> int:
        return self._games.get(model, 0)

    def win_rate(self, model: str) -> float:
        played = self._games.get(model, 0)
        if not played:
            return 0.0
        return self._score.get(model, 0.0) / played

    @property
    def ratings(self) -> Dict[str, float]:
        return dict(self._ratings)

    def expected_score(self, model_a: str, model_b: str) -> float:
        diff = (self.rating(model_b) - self.rating(model_a)) / 400.0
        return 1.0 / (1.0 + math.pow(10, diff))

    def record(
        self, model_a: str, model_b: str, winner: Optional[str]
    ) -> Tuple[float, float]:
        """Record one comparison. ``winner`` is a model name or ``None``
        for a draw. Returns the new ``(rating_a, rating_b)``."""
        if winner is not None and winner not in (model_a, model_b):
            raise ValueError(
                f"`winner` must be {model_a!r}, {model_b!r} or None "
                f"for a draw, got {winner!r}."
            )
        if winner is None:
            actual_a, actual_b = 0.5, 0.5
        elif winner == model_a:
            actual_a, actual_b = 1.0, 0.0
        else:
            actual_a, actual_b = 0.0, 1.0
        expected_a = self.expected_score(model_a, model_b)
        expected_b = 1.0 - expected_a
        self._ratings[model_a] = self.rating(model_a) + self.k_factor * (
            actual_a - expected_a
        )
        self._ratings[model_b] = self.rating(model_b) + self.k_factor * (
            actual_b - expected_b
        )
        self._games[model_a] = self.games(model_a) + 1
        self._games[model_b] = self.games(model_b) + 1
        self._score[model_a] = self._score.get(model_a, 0.0) + actual_a
        self._score[model_b] = self._score.get(model_b, 0.0) + actual_b
        return self._ratings[model_a], self._ratings[model_b]

    def record_win(self, winner: str, loser: str) -> Tuple[float, float]:
        return self.record(winner, loser, winner)

    def record_draw(self, model_a: str, model_b: str) -> Tuple[float, float]:
        return self.record(model_a, model_b, None)

    def leaderboard(self) -> List[Dict[str, float]]:
        table = [
            {
                "model": model,
                "elo": rating,
                "games": float(self.games(model)),
                "win_rate": self.win_rate(model),
            }
            for model, rating in self._ratings.items()
        ]
        table.sort(key=lambda row: row["elo"], reverse=True)
        return table
