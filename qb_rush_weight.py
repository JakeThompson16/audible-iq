"""Learn the QB rush-attempt weight relative to a pass attempt (pass attempt = 1.0).

OLS of QB weekly points on [attempts, carries]; weight = coef_carries / coef_attempts.
Scoring-dependent. Runs twice: the league's ScoringSettings, and nflreadpy's standard
fantasy_points_ppr column. Run: python qb_rush_weight.py
"""
import numpy as np
import nflreadpy as nfl
import polars as pl

import nflreadpy.downloader as _d
_d.NflverseDownloader.BASE_URLS["dynastyprocess"] = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/"
)

from clients.sleeper_client import get_user, get_user_leagues
from domain.scoring import ScoringSettings
from clients.nflreadpy.player_data import load_player_stats
from engine.scoring import calculate_points_vectorized

SEASONS = [2023, 2024, 2025]
MIN_ATTEMPTS = 10  # drop relief/garbage-time appearances
N_BOOT = 2000

user = get_user("jakethompson16")
league = get_user_leagues(user["user_id"], "2026")[1]
scoring = ScoringSettings.from_dict(league["scoring_settings"])

stats = calculate_points_vectorized(load_player_stats(SEASONS), scoring)
qb = stats.filter((pl.col("position") == "QB") & (pl.col("attempts") >= MIN_ATTEMPTS))


def fit(df: pl.DataFrame, target: str = "fantasy_points", intercept: bool = True) -> dict:
    X = df.select(["attempts", "carries"]).to_numpy()
    y = df[target].to_numpy()
    if intercept:
        X = np.column_stack([X, np.ones(len(X))])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return {"pass": beta[0], "rush": beta[1], "weight": beta[1] / beta[0],
            "r2": 1 - resid.var() / y.var(), "n": len(y)}


def report(label: str, df: pl.DataFrame, target: str) -> None:
    main = fit(df, target)
    rng = np.random.default_rng(0)
    Xy = df.select(["attempts", "carries", target]).to_numpy()
    boot = []
    for _ in range(N_BOOT):
        s = Xy[rng.integers(0, len(Xy), len(Xy))]
        A = np.column_stack([s[:, 0], s[:, 1], np.ones(len(s))])
        b = np.linalg.lstsq(A, s[:, 2], rcond=None)[0]
        boot.append(b[1] / b[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n=== {label} === QB player-weeks (attempts>={MIN_ATTEMPTS}): {main['n']}")
    print(f"pts/pass attempt={main['pass']:.4f}  pts/rush attempt={main['rush']:.4f}  R2={main['r2']:.3f}")
    print(f"WEIGHT (pass=1.0): rush = {main['weight']:.3f}   95% bootstrap CI [{lo:.3f}, {hi:.3f}]")
    print(f"no-intercept fit: rush = {fit(df, target, intercept=False)['weight']:.3f}")
    for s in SEASONS:
        f = fit(df.filter(pl.col("season") == s), target)
        print(f"  season {s}: weight={f['weight']:.3f}  n={f['n']}")


report("League scoring", qb, "fantasy_points")

ppr = (
    nfl.load_player_stats(SEASONS)
    .filter((pl.col("position") == "QB") & (pl.col("attempts") >= MIN_ATTEMPTS))
    .select(["season", "attempts", "carries", "fantasy_points_ppr"])
)
report("Standard PPR (nflreadpy fantasy_points_ppr)", ppr, "fantasy_points_ppr")
