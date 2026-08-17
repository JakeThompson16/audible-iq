
import json

import polars as pl

# nflreadpy 0.1.5 (latest on PyPI) builds the dynastyprocess player-ID
# crosswalk URL via github.com/.../raw/master/..., which now 404s — GitHub
# dropped that redirect. raw.githubusercontent.com still serves the file.
# Patched here rather than in site-packages so it survives reinstalls.
import nflreadpy.downloader as _nflreadpy_downloader
_nflreadpy_downloader.NflverseDownloader.BASE_URLS["dynastyprocess"] = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/"
)

from clients.sleeper_client import get_user, get_user_leagues
from domain.scoring import ScoringSettings

from clients.nflreadpy.player_data import load_player_stats
from clients.nflreadpy.team_data import pull_team_games
from engine.scoring import calculate_points_vectorized

from projections.expected_points.features.opponent_skew import calculate_all_position_skews
from projections.expected_points.features.player_rolling import add_rolling_features
from engine.expected_points import calculate_expected_points
from engine.metrics import evaluate_projections

username = "jakethompson16"
season = "2026"

# 1. Resolve username -> user_id, pull a real league's scoring settings
user = get_user(username)
print(f"User: {user['display_name']} (id: {user['user_id']})")

leagues = get_user_leagues(user["user_id"], season)
league = leagues[1]
scoring_settings = ScoringSettings.from_dict(league["scoring_settings"])
print(f"Scoring settings from league: {league['name']}\n")

EVAL_SEASONS = [2024, 2025]
# 2023 is loaded only so 2024 has prior-season data to blend against
# (see CLAUDE.md "Early-season handling") — it is never scored itself.
LOAD_SEASONS = [2023] + EVAL_SEASONS

stats = load_player_stats(LOAD_SEASONS)
stats = calculate_points_vectorized(stats, scoring_settings)
stats = add_rolling_features(stats, window=8)

games = pull_team_games(LOAD_SEASONS)
skew = calculate_all_position_skews(stats, games)

predictions = calculate_expected_points(stats, skew)
predictions = predictions.filter(pl.col("season").is_in(EVAL_SEASONS))

results = evaluate_projections(predictions, stats)


def _fmt(x):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


print(f"\n===== Expected Points metrics: {EVAL_SEASONS} =====\n")

for position, pos_results in results["by_position"].items():
    print(f"--- {position} (n={pos_results['n']}) ---")

    proj = pos_results["projection"]
    base = pos_results["baseline_rolling_avg_only"]
    print(
        f"  projection:  MAE={_fmt(proj['mae'])}  RMSE={_fmt(proj['rmse'])}  "
        f"mean_error={_fmt(proj['mean_error'])}  R2={_fmt(proj['r2'])}  n={proj['n']}"
    )
    print(
        f"  baseline:    MAE={_fmt(base['mae'])}  RMSE={_fmt(base['rmse'])}  "
        f"mean_error={_fmt(base['mean_error'])}  R2={_fmt(base['r2'])}  n={base['n']}"
    )
    print(f"  skew_improves_mae: {pos_results['skew_improves_mae']}")

    spearman = pos_results["spearman_rank_correlation"]
    print(
        f"  spearman rank corr (mean across {len(spearman['by_week'])} weeks, "
        f"{spearman['weeks_skipped_too_few_players']} skipped): {_fmt(spearman['mean'])}"
    )

    calib = pos_results["calibration"]["by_tier"]
    print("  calibration (MAE by confidence tier):")
    for tier in ["high", "medium", "low"]:
        t = calib[tier]
        print(f"    {tier:8s} MAE={_fmt(t['mae'])}  n={t['n']}")
    print(f"  monotonic_decreasing_mae: {pos_results['calibration']['monotonic_decreasing_mae']}")
    print()

print("--- overall calibration (all positions combined) ---")
overall_calib = results["calibration_overall"]["by_tier"]
for tier in ["high", "medium", "low"]:
    t = overall_calib[tier]
    print(f"  {tier:8s} MAE={_fmt(t['mae'])}  n={t['n']}")
print(f"  monotonic_decreasing_mae: {results['calibration_overall']['monotonic_decreasing_mae']}")

print("\n--- meta ---")
for k, v in results["meta"].items():
    print(f"  {k}: {v}")
