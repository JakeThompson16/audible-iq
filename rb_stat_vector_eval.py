

import polars as pl

# Same nflreadpy crosswalk URL patch as test.py (see the comment there).
import nflreadpy.downloader as _nflreadpy_downloader
_nflreadpy_downloader.NflverseDownloader.BASE_URLS["dynastyprocess"] = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/"
)

from clients.sleeper_client import get_user, get_user_leagues
from domain.scoring import ScoringSettings

from clients.nflreadpy.player_data import load_player_stats
from clients.nflreadpy.team_data import pull_team_games
from clients.nflreadpy.pbp_data import load_pbp_data
from engine.scoring import calculate_points_vectorized
from engine.expected_points import calculate_expected_points
from engine.metrics import evaluate_projections

from projections.expected_points.features.opponent_skew import calculate_all_position_skews
from projections.expected_points.features.player_rolling import add_rolling_features
from projections.expected_points.features.epa_allowed import calculate_epa_allowed, join_epa_allowed
from projections.expected_points.features.stat_rolling import add_stat_rolling_features
from projections.expected_points.stat_vector.rb import (
    KEY_COLUMNS, LEGACY_RB_MODELS, RATE_TARGETS, VOLUME_TARGETS, apply_rate_priors,
    fit_rb_models, predict_rb_stat_vector, rate_priors, recompose_points, with_unfitted_rates,
)

# Standalone comparison: RB stat-vector projection vs. the current
# rolling_avg_prior + opponent_skew formula. Nothing here is used in production.

username = "jakethompson16"
season = "2026"

user = get_user(username)
leagues = get_user_leagues(user["user_id"], season)
league = leagues[1]
scoring_settings = ScoringSettings.from_dict(league["scoring_settings"])
print(f"Scoring settings from league: {league['name']}\n")

# 2018 is history only (prior-season blend + rolling windows for 2019).
# 2019-2025 are full seasons, each held out once (leave-one-season-out).
# 2026 is in progress: test-only fold, trained on 2019-2025.
LOAD_SEASONS = list(range(2018, 2027))
FULL_SEASONS = list(range(2019, 2026))
PARTIAL_SEASON = 2026
WINDOW = 8  # winning n from the first iteration's {4, 6, 8} grid

FOLDS = (
    [(f"{s} (LOSO)", [o for o in FULL_SEASONS if o != s], s) for s in FULL_SEASONS]
    + [(f"{PARTIAL_SEASON} wk1-3 (partial)", FULL_SEASONS, PARTIAL_SEASON)]
    + [("orig 2024->2025", [2024], 2025), ("orig 2025->2024", [2025], 2024)]
)

PBP_COLUMNS = ["season", "week", "defteam", "play_type", "sack", "epa",
               "yardline_100", "rushing_yards", "air_yards"]


def _fmt(x, d=3):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


# ---- shared inputs (same as test.py) ----
stats = load_player_stats(LOAD_SEASONS)
stats = calculate_points_vectorized(stats, scoring_settings)
stats = add_rolling_features(stats, window=8)

games = pull_team_games(LOAD_SEASONS)
skew = calculate_all_position_skews(stats, games)
current = calculate_expected_points(stats, skew)

pbp = pl.concat(
    [load_pbp_data(s).select(PBP_COLUMNS) for s in LOAD_SEASONS], how="vertical_relaxed"
)
epa = calculate_epa_allowed(pbp)
del pbp

rb = join_epa_allowed(add_stat_rolling_features(stats, window=WINDOW), epa) \
    .filter(pl.col("position") == "RB")

# load_player_metadata can fan out rows for players listed twice in the ID
# crosswalk (seen for two defensive players when loading from 2018). Make sure
# none of that reaches the RB rows.
assert not rb.select(["gsis_id", "season", "week"]).is_duplicated().any(), \
    "duplicate RB player-week rows (metadata fan-out)"

print("===== data by season (RB player-weeks, epa_allowed defense-weeks) =====")
print(rb.group_by("season").len().sort("season").join(
    epa.group_by("season").len().rename({"len": "epa_rows"}), on="season").rows())

# ---- variants ----
VARIANTS = {
    "v0 first iteration": "epa in volume, unweighted fitted rates",
    "v1 step 1": "no epa in volume, unweighted fitted rates",
    "v2 steps 1+3": "no epa in volume, attempt-weighted fitted rates",
    "v3 league-avg rates": "v2 volume, rates = pooled training RB rate",
    "v4 season-to-date rates": "v2 volume, rates = player season-to-date (league-avg fallback)",
}
BENCHMARKS = {
    "current (rolling+skew)": "current_projection",
    "rolling_avg_prior only": "rolling_avg_prior",
}

results = {}         # fold label -> {method: metrics}
volume_coefs = {}    # fold label -> {target: {term: coef}}
rate_epa_coefs = {}  # fold label -> {rate target: (coef, p)}

for label, train_seasons, test in FOLDS:
    tr = rb.filter(pl.col("season").is_in(train_seasons))
    te = rb.filter(pl.col("season") == test)

    priors = rate_priors(tr)
    tr = apply_rate_priors(tr, priors)
    te = apply_rate_priors(te, priors)

    v2 = fit_rb_models(tr)
    fits = {
        "v0 first iteration": fit_rb_models(tr, weight_rates=False, models=LEGACY_RB_MODELS),
        "v1 step 1": fit_rb_models(tr, weight_rates=False),
        "v2 steps 1+3": v2,
        "v3 league-avg rates": with_unfitted_rates(v2, priors, "league"),
        "v4 season-to-date rates": with_unfitted_rates(v2, priors, "season_to_date"),
    }

    volume_coefs[label] = {
        t: dict(zip(["intercept"] + v2[t].features, v2[t].coef.tolist())) for t in VOLUME_TARGETS
    }
    rate_epa_coefs[label] = {}
    for t in RATE_TARGETS:
        s = v2[t].summary().filter(pl.col("term").str.starts_with("epa_allowed"))
        rate_epa_coefs[label][t] = (s["coef"][0], s["p"][0])

    frame = (
        current.filter((pl.col("position") == "RB") & (pl.col("season") == test))
        .select(KEY_COLUMNS + ["projection", "rolling_avg_prior", "confidence"])
        .rename({"projection": "current_projection"})
    )
    for name, f in fits.items():
        pts = recompose_points(predict_rb_stat_vector(te, f), scoring_settings) \
            .select(KEY_COLUMNS + ["fantasy_points"]).rename({"fantasy_points": name})
        frame = frame.join(pts, on=KEY_COLUMNS, how="inner")

    methods = {**{v: v for v in VARIANTS}, **BENCHMARKS}
    frame = frame.drop_nulls(subset=list(methods.values()))

    results[label] = {}
    for method, col in methods.items():
        r = evaluate_projections(frame.with_columns(pl.col(col).alias("projection")), stats)
        r = r["by_position"]["RB"]
        results[label][method] = {
            "n": r["n"],
            "mae": r["projection"]["mae"],
            "r2": r["projection"]["r2"],
            "bias": r["projection"]["mean_error"],
            "spearman": r["spearman_rank_correlation"]["mean"],
        }

# ---- report ----
print("\n===== variants =====")
for v, desc in VARIANTS.items():
    print(f"  {v:24s} {desc}")

print("\n===== per-fold RB recomposed points (same rows per fold; bias = actual - predicted) =====")
for label, by_method in results.items():
    print(f"\n--- {label} (rows={by_method['v2 steps 1+3']['n']}) ---")
    for method, m in by_method.items():
        print(f"  {method:24s} MAE={_fmt(m['mae'])}  R2={_fmt(m['r2'])}  "
              f"bias={_fmt(m['bias'])}  Spearman={_fmt(m['spearman'])}")

loso = [label for label, *_ in FOLDS if "LOSO" in label]
print("\n===== LOSO summary over", len(loso), "full-season folds =====")
for method in list(VARIANTS) + list(BENCHMARKS):
    maes = [results[l][method]["mae"] for l in loso]
    sps = [results[l][method]["spearman"] for l in loso]
    r2s = [results[l][method]["r2"] for l in loso]
    print(f"  {method:24s} mean MAE={_fmt(sum(maes) / len(maes))}  mean R2={_fmt(sum(r2s) / len(r2s))}  "
          f"mean Spearman={_fmt(sum(sps) / len(sps))}")

print("\n===== head-to-head: folds (of", len(loso), "LOSO) where the variant beats the benchmark =====")
for v in VARIANTS:
    parts = []
    for b in BENCHMARKS:
        mae_w = sum(results[l][v]["mae"] < results[l][b]["mae"] for l in loso)
        sp_w = sum(results[l][v]["spearman"] > results[l][b]["spearman"] for l in loso)
        parts.append(f"vs {b}: MAE {mae_w}/{len(loso)}, Spearman {sp_w}/{len(loso)}")
    print(f"  {v:24s} " + " | ".join(parts))

print("\n===== v2 vs v3 vs v4 per LOSO fold (MAE / Spearman) =====")
for l in loso:
    print(f"  {l:12s} " + "  ".join(
        f"{v.split()[0]}={_fmt(results[l][v]['mae'])}/{_fmt(results[l][v]['spearman'])}"
        for v in ["v2 steps 1+3", "v3 league-avg rates", "v4 season-to-date rates"]))

print("\n===== volume model coefficients per fold (v2; no epa) =====")
for label, coefs in volume_coefs.items():
    print(f"  {label:22s} " + "  |  ".join(
        f"{t}: " + ", ".join(f"{k}={_fmt(c)}" for k, c in terms.items()) for t, terms in coefs.items()))

print("\n===== epa_allowed coefficient in each weighted rate model, per fold: coef (p) =====")
for label, coefs in rate_epa_coefs.items():
    print(f"  {label:22s} " + "  ".join(f"{t}={_fmt(c)} ({_fmt(p)})" for t, (c, p) in coefs.items()))
