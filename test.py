"""
End-to-end backtest of engine/expected_points.py with the production position
registry (RB: stat vector, QB/WR/TE: rolling_avg_prior + opponent_skew),
leave-one-season-out over 2019-2025 plus the partial 2026 season.

Run: python test.py
"""
import polars as pl

from clients.sleeper_client import get_user, get_user_leagues
from domain.scoring import ScoringSettings
from engine.expected_points import POSITION_IMPLEMENTATIONS
from engine.metrics import POSITIONS, evaluate_projections
from evaluation.backtest import load_inputs, loso_folds, predict_fold

username = "jakethompson16"
season = "2026"

user = get_user(username)
print(f"User: {user['display_name']} (id: {user['user_id']})")
league = get_user_leagues(user["user_id"], season)[1]
scoring_settings = ScoringSettings.from_dict(league["scoring_settings"])
print(f"Scoring settings from league: {league['name']}")
print(f"Implementations: {POSITION_IMPLEMENTATIONS}\n")


def _fmt(x):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


inputs = load_inputs(scoring_settings)

for label, train, test in loso_folds():
    preds, _ = predict_fold(inputs, train, test)
    results = evaluate_projections(preds, inputs.stats)

    print(f"===== test season {label} (stat-vector models fit on {', '.join(map(str, train))}) =====")
    for position in POSITIONS:
        r = results["by_position"][position]
        proj, base = r["projection"], r["baseline_rolling_avg_only"]
        method = preds.filter(pl.col("position") == position)["projection_method"].unique().to_list()
        print(f"  {position} [{', '.join(method)}] n={r['n']}  "
              f"MAE={_fmt(proj['mae'])} (rolling-only {_fmt(base['mae'])})  "
              f"R2={_fmt(proj['r2'])} (rolling-only {_fmt(base['r2'])})  "
              f"Spearman={_fmt(r['spearman_rank_correlation']['mean'])}  "
              f"calibration monotonic={r['calibration']['monotonic_decreasing_mae']}")
    calib = results["calibration_overall"]["by_tier"]
    print("  calibration (all positions, MAE by tier): " + ", ".join(
        f"{t}={_fmt(calib[t]['mae'])} (n={calib[t]['n']})" for t in calib))
    print(f"  meta: {results['meta']['n_scored']} scored, "
          f"{results['meta']['n_null_projection_dropped']} null projections dropped\n")
