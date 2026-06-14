from unittest.mock import patch

from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse

from src.faceit_api import FaceitError

from . import services
from .models import PredictionLog

# A fake get_player_stats result with the keys the feature code needs.
FAKE_STATS = {
    "username": "x",
    "player_id": "x",
    "elo": 1500.0,
    "matches": 20,
    "kd": 1.0,
    "adr": 80.0,
    "hs_percent": 50.0,
    "win_rate": 50.0,
}

TEAM_A = ["a1", "a2", "a3", "a4", "a5"]
TEAM_B = ["b1", "b2", "b3", "b4", "b5"]


class ServiceTests(TestCase):
    def setUp(self):
        cache.clear()  # LocMemCache persists across tests in one process

    @patch("predictor.services.get_player_stats", return_value=FAKE_STATS)
    def test_run_prediction_returns_valid_probability(self, _mock):
        result = services.run_prediction(TEAM_A, TEAM_B)
        self.assertIn("prob_a", result)
        self.assertTrue(0.0 <= result["prob_a"] <= 1.0)
        self.assertEqual(set(result["avg_a"]), {"kd", "adr", "hs_percent", "win_rate"})

    @patch("predictor.services.get_player_stats", return_value=FAKE_STATS)
    def test_player_stats_are_cached(self, mock_stats):
        # Two lookups for the same nickname should hit the API only once.
        services.cached_player_stats("SamePlayer")
        services.cached_player_stats("SamePlayer")
        self.assertEqual(mock_stats.call_count, 1)

    @patch("predictor.services.get_player_stats", side_effect=FaceitError("not found"))
    def test_run_prediction_raises_on_bad_player(self, _mock):
        with self.assertRaises(FaceitError):
            services.run_prediction(TEAM_A, TEAM_B)


class ViewTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_get_renders_form(self):
        response = self.client.get(reverse("predictor:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Predict winner")

    @patch("predictor.services.get_player_stats", return_value=FAKE_STATS)
    def test_post_predicts_and_logs(self, _mock):
        payload = {f"a{i}": n for i, n in enumerate(TEAM_A, 1)}
        payload.update({f"b{i}": n for i, n in enumerate(TEAM_B, 1)})
        response = self.client.post(reverse("predictor:index"), payload)
        self.assertEqual(response.status_code, 200)
        # A result section always renders this, regardless of coin-flip vs favored.
        self.assertContains(response, "win probability")
        # Per-player Elo is shown in the result.
        self.assertContains(response, "1500")
        self.assertEqual(PredictionLog.objects.count(), 1)


class RandomLineupTests(TestCase):
    @patch("predictor.views.random_ranked_players", return_value=[f"p{i}" for i in range(10)])
    def test_returns_ten_named_fields(self, _mock):
        response = self.client.get(reverse("predictor:random_lineup"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(set(data), {f"{side}{i}" for side in "ab" for i in range(1, 6)})
        self.assertEqual(data["a1"], "p0")
        self.assertEqual(data["b5"], "p9")

    @patch("predictor.views.random_ranked_players", side_effect=FaceitError("boom"))
    def test_faceit_error_returns_502(self, _mock):
        response = self.client.get(reverse("predictor:random_lineup"))
        self.assertEqual(response.status_code, 502)
        self.assertIn("error", response.json())
