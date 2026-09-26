

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
    RB_MODELS, KEY_COLUMNS, apply_rate_priors, fit_rb_models, predict_rb_stat_vector,
    rate_priors, recompose_points, vif,
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

EVAL_SEASONS = [2024, 2025]
LOAD_SEASONS = [2023] + EVAL_SEASONS
WINDOWS = [4, 6, 8]
DIRECTIONS = [(2024, 2025), (2025, 2024)]


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

stats = stats.sort(["gsis_id", "season", "week"]).with_columns(
    pl.col("fantasy_points").shift(1).over("gsis_id").alias("last_week_points")
)

pbp = load_pbp_data(LOAD_SEASONS)
epa = calculate_epa_allowed(pbp)

# ---- sanity checks ----
print("===== sanity: epa_allowed =====")
print(f"rows={epa.height}  unique keys asserted")
print(epa.select(["epa_allowed_pass", "epa_allowed_rush"]).describe())
print("n_games distribution (2024):",
      epa.filter(pl.col("season") == 2024)["n_games"].value_counts().sort("n_games").rows())
wk1 = epa.filter((pl.col("season") == 2024) & (pl.col("week") == 1))
print(f"2024 week 1: {wk1.height} defenses, nonzero pass={int((wk1['epa_allowed_pass'] != 0).sum())}"
      f" rush={int((wk1['epa_allowed_rush'] != 0).sum())} (should be nonzero: prior season blended in)")
print("2023 week 1 (no prior season loaded; should be all 0):",
      epa.filter((pl.col("season") == 2023) & (pl.col("week") == 1))["epa_allowed_pass"].abs().sum())
print("softest / toughest pass D, 2025 final week present:")
last = epa.filter(pl.col("season") == 2025).filter(pl.col("week") == pl.col("week").max().over("defteam"))
print(last.sort("epa_allowed_pass").select(["defteam", "week", "epa_allowed_pass", "n_eff_pass"]).head(3))
print(last.sort("epa_allowed_pass").select(["defteam", "week", "epa_allowed_pass", "n_eff_pass"]).tail(3))

print("\n===== sanity: season-boundary rolling (top-carry RB, 2024 tail -> 2025 head) =====")
probe = add_stat_rolling_features(stats, window=8)
top_rb = (stats.filter((pl.col("position") == "RB") & (pl.col("season") == 2024))
          .group_by("gsis_id").agg(pl.col("carries").sum()).sort("carries").tail(1)["gsis_id"][0])
print(probe.filter(pl.col("gsis_id") == top_rb)
      .filter(((pl.col("season") == 2024) & (pl.col("week") >= 15)) | ((pl.col("season") == 2025) & (pl.col("week") <= 3)))
      .select(["display_name", "season", "week", "carries", "roll_carries", "delta_carries", "roll_ypc"]))

print("\n===== sanity: recompose ACTUAL stats (what the excluded categories cost) =====")
rb_actual = stats.filter((pl.col("position") == "RB") & pl.col("season").is_in(EVAL_SEASONS))
recomp_actual = recompose_points(rb_actual, scoring_settings).rename({"fantasy_points": "recomposed"})
cmp = recomp_actual.join(rb_actual.select(KEY_COLUMNS + ["fantasy_points"]), on=KEY_COLUMNS)
gap = cmp["fantasy_points"] - cmp["recomposed"]
print(f"RB rows={cmp.height}  mean(actual - recomposed_actual)={_fmt(gap.mean())}  "
      f"mean|gap|={_fmt(gap.abs().mean())}  share exact={_fmt(float((gap.abs() < 1e-9).mean()))}")
print("  (this is the ceiling cost of not predicting 2pt / first downs / fumbles / threshold bonuses)")

# ---- grid ----
METHODS = {
    "stat_vector": "stat_vector_points",
    "stat_vector (WLS rates)": "stat_vector_wls_points",
    "current (rolling+skew)": "current_projection",
    "rolling_avg_prior only": "rolling_avg_prior",
    "naive mean": "naive_mean",
    "last week": "last_week_points",
}

results = {}      # (n, train, test) -> {method: rb metrics}
coef_tables = {}  # (n, train) -> {target: summary df}
vif_tables = {}   # (n, train) -> {target: vif dict}
wls_tables = {}   # (n, train) -> {rate target: (summary df, n_obs, weighted R2)}

for n in WINDOWS:
    feat = join_epa_allowed(add_stat_rolling_features(stats, window=n), epa)
    rb = feat.filter(pl.col("position") == "RB")

    for train, test in DIRECTIONS:
        tr = rb.filter(pl.col("season") == train)
        te = rb.filter(pl.col("season") == test)

        priors = rate_priors(tr)
        tr = apply_rate_priors(tr, priors)
        te = apply_rate_priors(te, priors)

        fits = fit_rb_models(tr)
        wls_fits = fit_rb_models(tr, weight_rates=True)
        coef_tables[(n, train)] = {t: f.summary() for t, f in fits.items()}
        coef_tables[(n, train)]["_n_r2"] = {t: (f.n, f.r2) for t, f in fits.items()}
        wls_tables[(n, train)] = {t: (wls_fits[t].summary(), wls_fits[t].n, wls_fits[t].r2)
                                  for t, (_, den) in RB_MODELS.items() if den is not None}
        vif_tables[(n, train)] = {
            t: vif(tr if den is None else tr.filter(pl.col(den) > 0), feats)
            for t, (feats, den) in RB_MODELS.items()
        }

        pts = recompose_points(predict_rb_stat_vector(te, fits), scoring_settings) \
            .select(KEY_COLUMNS + ["fantasy_points"]).rename({"fantasy_points": "stat_vector_points"})

        pts_wls = recompose_points(predict_rb_stat_vector(te, wls_fits), scoring_settings) \
            .select(KEY_COLUMNS + ["fantasy_points"]).rename({"fantasy_points": "stat_vector_wls_points"})

        naive = tr["fantasy_points"].mean()

        frame = (
            current.filter((pl.col("position") == "RB") & (pl.col("season") == test))
            .select(KEY_COLUMNS + ["projection", "rolling_avg_prior", "confidence"])
            .rename({"projection": "current_projection"})
            .join(pts, on=KEY_COLUMNS, how="inner")
            .join(pts_wls, on=KEY_COLUMNS, how="inner")
            .join(stats.select(KEY_COLUMNS + ["last_week_points"]), on=KEY_COLUMNS, how="left")
            .with_columns(pl.lit(naive).alias("naive_mean"))
            .drop_nulls(subset=list(METHODS.values()))
        )

        results[(n, train, test)] = {}
        for method, col in METHODS.items():
            preds = frame.with_columns(pl.col(col).alias("projection"))
            r = evaluate_projections(preds, stats)["by_position"]["RB"]
            results[(n, train, test)][method] = {
                "n": r["n"],
                "mae": r["projection"]["mae"],
                "rmse": r["projection"]["rmse"],
                "r2": r["projection"]["r2"],
                "mean_error": r["projection"]["mean_error"],
                "spearman": r["spearman_rank_correlation"]["mean"],
            }

# ---- report ----
print("\n===== RB recomposed-points comparison (same rows per cell, unmodified evaluate_projections) =====")
for (n, train, test), by_method in results.items():
    print(f"\n--- window n={n}, train {train} -> test {test} (rows={by_method['stat_vector']['n']}) ---")
    for method, m in by_method.items():
        print(f"  {method:24s} MAE={_fmt(m['mae'])}  RMSE={_fmt(m['rmse'])}  R2={_fmt(m['r2'])}  "
              f"bias={_fmt(m['mean_error'])}  Spearman={_fmt(m['spearman'])}")

print("\n===== winning n (stat_vector mean MAE across both directions; Spearman tie-break) =====")
summary = []
for n in WINDOWS:
    maes = [results[(n, a, b)]["stat_vector"]["mae"] for a, b in DIRECTIONS]
    sps = [results[(n, a, b)]["stat_vector"]["spearman"] for a, b in DIRECTIONS]
    summary.append((n, sum(maes) / 2, sum(sps) / 2))
    print(f"  n={n}: mean MAE={_fmt(summary[-1][1])}  mean Spearman={_fmt(summary[-1][2])}")
best_n = sorted(summary, key=lambda s: (round(s[1], 3), -s[2]))[0][0]
print(f"  winner: n={best_n}")

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_width_chars(140)
pl.Config.set_float_precision(4)
for n in WINDOWS:
    for train, _ in DIRECTIONS:
        print(f"\n===== coefficients: n={n}, fit on {train} =====")
        n_r2 = coef_tables[(n, train)]["_n_r2"]
        for target in RB_MODELS:
            nobs, r2 = n_r2[target]
            v = vif_tables[(n, train)][target]
            print(f"\n[{target}]  n_obs={nobs}  in-sample R2={_fmt(r2)}  "
                  f"VIF: " + ", ".join(f"{k}={_fmt(x, 2)}" for k, x in v.items()))
            print(coef_tables[(n, train)][target])

for n in WINDOWS:
    for train, _ in DIRECTIONS:
        print(f"\n===== WLS rate coefficients (weight = denominator): n={n}, fit on {train} =====")
        for target, (summ, nobs, r2) in wls_tables[(n, train)].items():
            print(f"\n[{target}]  n_obs={nobs}  weighted in-sample R2={_fmt(r2)}")
            print(summ)
