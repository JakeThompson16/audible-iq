"""
Grid search of the continuous rolling-window length for rolling_plus_skew, per
position, on the LOSO seasons (no fitting involved, so each season is simply
evaluated). Selection rule: lowest mean MAE over the full-season folds;
Spearman/R² reported alongside. Result lives in POSITION_ROLLING_WINDOWS.

Run: python rolling_window_eval.py
"""
import polars as pl

from clients.nflreadpy.player_data import load_player_stats
from clients.sleeper_client import get_user, get_user_leagues
from domain.scoring import ScoringSettings
from engine.expected_points import POSITION_ROLLING_WINDOWS, ExpectedPointsContext, calculate_expected_points
from engine.metrics import evaluate_projections
from engine.scoring import calculate_points_vectorized
from evaluation.backtest import FULL_SEASONS, LOAD_SEASONS, PARTIAL_SEASON, load_inputs

WINDOWS = [8, 10, 12, 14, 16, 20]
POSITIONS = ["QB", "WR", "TE"]

user = get_user("jakethompson16")
league = get_user_leagues(user["user_id"], "2026")[1]
scoring = ScoringSettings.from_dict(league["scoring_settings"])
base = calculate_points_vectorized(load_player_stats(LOAD_SEASONS), scoring)


def _m(df, pos):
    r = evaluate_projections(df, base)["by_position"][pos]
    return r["projection"]["mae"], r["projection"]["r2"], r["spearman_rank_correlation"]["mean"]


rows = []
for w in WINDOWS:
    inputs = load_inputs(scoring, rolling_windows=w, with_epa=False, base_stats=base)
    preds = calculate_expected_points(
        inputs.stats, ExpectedPointsContext(skew_df=inputs.skew, rolling_windows=inputs.rolling_windows),
        {p: "rolling_plus_skew" for p in inputs.rolling_windows},
    )
    for pos in POSITIONS:
        per = [_m(preds.filter(pl.col("season") == s), pos) for s in FULL_SEASONS]
        early = _m(preds.filter(pl.col("season").is_in(FULL_SEASONS) & (pl.col("week") <= 3)), pos)
        partial = _m(preds.filter(pl.col("season") == PARTIAL_SEASON), pos)
        n = len(per)
        rows.append({
            "position": pos, "window": w,
            "mae": sum(p[0] for p in per) / n, "r2": sum(p[1] for p in per) / n,
            "spearman": sum(p[2] for p in per) / n,
            "wk1_3_mae": early[0], "partial_mae": partial[0], "partial_spearman": partial[2],
        })

res = pl.DataFrame(rows)
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(-1)
pl.Config.set_float_precision(3)
print(res)
for pos in POSITIONS:
    best = res.filter(pl.col("position") == pos).sort("mae").row(0, named=True)
    print(f"{pos}: MAE-optimal window {best['window']} (configured: {POSITION_ROLLING_WINDOWS[pos]})")
