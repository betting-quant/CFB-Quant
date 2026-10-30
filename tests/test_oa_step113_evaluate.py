"""Unit tests for the locked Step 113 scoring calculations."""

import unittest
from types import SimpleNamespace

from scripts.oa_step113_evaluate import score_game


class TestProspectiveScoring(unittest.TestCase):

    def setUp(self):
        self.row = SimpleNamespace(
            game_id=401859185,
            away_team="Notre Dame",
            home_team="Navy",
            predicted_home=21.0,
            predicted_away=28.0,
        )
        self.columns = ["predicted_home", "predicted_away"]

    def evaluate(self, home, away):
        return score_game(
            self.row,
            home,
            away,
            "test_model",
            self.columns,
        )

    def test_exact_scoring(self):
        result = self.evaluate(21, 28)

        self.assertEqual(result["spread_ae"], 0)
        self.assertEqual(result["total_ae"], 0)
        self.assertEqual(result["team_score_ae"], 0)
        self.assertTrue(result["winner_correct"])

    def test_spread_total_and_team_score_errors(self):
        # Predicted: away 28, home 21.
        # Actual:    away 17, home 24.
        result = self.evaluate(24, 17)

        self.assertAlmostEqual(result["predicted_margin"], -7)
        self.assertAlmostEqual(result["actual_margin"], 7)
        self.assertAlmostEqual(result["spread_ae"], 14)

        self.assertAlmostEqual(result["predicted_total"], 49)
        self.assertAlmostEqual(result["actual_total"], 41)
        self.assertAlmostEqual(result["total_ae"], 8)

        self.assertAlmostEqual(
            result["team_score_ae"],
            (abs(21 - 24) + abs(28 - 17)) / 2,
        )
        self.assertFalse(result["winner_correct"])

    def test_actual_tie_excluded_from_winner_accuracy(self):
        result = self.evaluate(21, 21)

        self.assertIsNone(result["winner_correct"])

    def test_predicted_tie_excluded_from_winner_accuracy(self):
        self.row.predicted_home = 24.0
        self.row.predicted_away = 24.0

        result = self.evaluate(28, 21)

        self.assertIsNone(result["winner_correct"])

    def test_result_game_identity(self):
        result = self.evaluate(21, 28)

        self.assertEqual(result["game_id"], "401859185")
        self.assertEqual(result["home_team"], "Navy")
        self.assertEqual(result["away_team"], "Notre Dame")
        self.assertEqual(result["model"], "test_model")


if __name__ == "__main__":
    unittest.main()
