import pytest

from deepeval.metrics.community import PairwiseElo


class TestPairwiseElo:
    """PairwiseElo is deterministic, no API key needed."""

    def test_equal_ratings_split_sixteen(self):
        elo = PairwiseElo(k_factor=32.0)
        rating_a, rating_b = elo.record("a", "b", "a")
        assert rating_a == pytest.approx(1516.0)
        assert rating_b == pytest.approx(1484.0)

    def test_draw_changes_nothing_for_equals(self):
        elo = PairwiseElo()
        rating_a, rating_b = elo.record("a", "b", None)
        assert rating_a == pytest.approx(1500.0)
        assert rating_b == pytest.approx(1500.0)

    def test_underdog_win_moves_more(self):
        elo = PairwiseElo(k_factor=32.0)
        elo.record("strong", "weak", "strong")
        before = elo.rating("weak")
        elo.record("strong", "weak", "weak")
        assert elo.rating("weak") - before > 16.0

    def test_win_rate_tracks_results(self):
        elo = PairwiseElo()
        elo.record("a", "b", "a")
        elo.record("a", "b", None)
        assert elo.win_rate("a") == pytest.approx(0.75)
        assert elo.win_rate("b") == pytest.approx(0.25)
        assert elo.games("a") == 2

    def test_unknown_model_has_defaults(self):
        elo = PairwiseElo()
        assert elo.rating("new") == 1500.0
        assert elo.win_rate("new") == 0.0

    def test_leaderboard_is_ranked(self):
        elo = PairwiseElo()
        elo.record("a", "b", "b")
        board = elo.leaderboard()
        assert [row["model"] for row in board] == ["b", "a"]
        assert board[0]["games"] == 1.0

    def test_bad_winner_rejected(self):
        elo = PairwiseElo()
        with pytest.raises(ValueError):
            elo.record("a", "b", "c")

    def test_bad_k_factor_rejected(self):
        with pytest.raises(ValueError):
            PairwiseElo(k_factor=0)
