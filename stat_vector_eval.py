"""
Stat-vector backtest for one position: new model vs current formula
(rolling_avg_prior + opponent_skew) vs rolling-average-alone, leave-one-season-out.

Run: python stat_vector_eval.py RB [--candidates]
  --candidates  also fit a variant with the spec's candidate_terms switched on
                (terms evaluated but off by default, e.g. WR targets/ypr EPA)
"""
import sys
import polars as pl

from clients.sleeper_client import get_user, get_user_leagues
from domain.scoring import ScoringSettings
from engine.expected_points import ExpectedPointsContext, calculate_expected_points
from engine.metrics import evaluate_projections
from evaluation.backtest import load_inputs, loso_folds, predict_fold
from projections.expected_points.stat_vector.core import KEY_COLUMNS, recompose_points
from projections.expected_points.stat_vector.specs import STAT_VECTOR_SPECS

POSITION = sys.argv[1] if len(sys.argv) > 1 else "RB"
CANDIDATES = "--candidates" in sys.argv
SPEC = STAT_VECTOR_SPECS[POSITION]

user = get_user("jakethompson16")
league = get_user_leagues(user["user_id"], "2026")[1]
scoring = ScoringSettings.from_dict(league["scoring_settings"])
print(f"Position: {POSITION}   Scoring settings from league: {league['name']}\n")


def _fmt(x, d=3):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


inputs = load_inputs(scoring)
stats = inputs.stats

# Current formula for every position (the comparison baseline).
current = calculate_expected_points(
    stats, ExpectedPointsContext(skew_df=inputs.skew),
    {p: "rolling_plus_skew" for p in stats["position"].unique().to_list()},
).select(["gsis_id", "season", "week", "projection"]).rename({"projection": "current"})

# ---- sanity ----
print("===== sanity =====")
pos_rows = stats.filter(pl.col("position") == POSITION)
print("player-weeks by season:", pos_rows.group_by("season").len().sort("season").rows())
first_wk = stats.filter((pl.col("season") == 2026) & (pl.col("week") == 1) & (pl.col("position") == POSITION))
print(f"2026 week 1 {POSITION}: {first_wk.height} rows, rolling_avg_prior non-null "
      f"{first_wk['rolling_avg_prior'].is_not_null().sum()}, n_games_in_window distribution "
      f"{first_wk['n_games_in_window'].value_counts().sort('n_games_in_window').rows()}")
wk1_skew = inputs.skew.filter((pl.col("season") == 2026) & (pl.col("week") == 1) & (pl.col("position") == POSITION))
print(f"2026 week 1 skew rows for {POSITION}: {wk1_skew.height} (continuous window: prior-season games count)")
actual = pos_rows.filter(pl.col("season").is_in([2024, 2025]))
recomp = recompose_points(actual, SPEC, scoring).rename({"fantasy_points": "recomposed"}) \
    .join(actual.select(KEY_COLUMNS + ["fantasy_points"]), on=KEY_COLUMNS)
gap = recomp["fantasy_points"] - recomp["recomposed"]
print(f"recomposing ACTUAL {POSITION} stats (2024/25): mean gap={_fmt(gap.mean())}  "
      f"mean|gap|={_fmt(gap.abs().mean())}  (cost of unpredicted categories)")

# ---- folds ----
variant_specs = {}
if CANDIDATES and SPEC.candidate_terms:
    variant_specs["+candidates"] = SPEC.with_candidates()

impl = {POSITION: "stat_vector"}
results, models_by_fold = {}, {}
for label, train, test in loso_folds():
    preds, models = predict_fold(inputs, train, test, impl)
    models_by_fold[label] = models[POSITION]

    frame = (
        preds.filter(pl.col("position") == POSITION)
        .select(KEY_COLUMNS + ["projection", "rolling_avg_prior", "confidence"])
        .rename({"projection": "new model"})
        .join(current, on=["gsis_id", "season", "week"], how="left")
    )
    methods = {"new model": "new model", "current formula": "current", "rolling alone": "rolling_avg_prior"}

    for vname, vspec in variant_specs.items():
        vpreds, vmodels = predict_fold(inputs, train, test, impl, specs={**STAT_VECTOR_SPECS, POSITION: vspec})
        models_by_fold[f"{label}|{vname}"] = vmodels[POSITION]
        frame = frame.join(
            vpreds.filter(pl.col("position") == POSITION).select(["gsis_id", "season", "week", "projection"])
            .rename({"projection": vname}), on=["gsis_id", "season", "week"], how="left")
        methods[vname] = vname

    frame = frame.drop_nulls(subset=list(methods.values()))
    results[label] = {}
    for method, col in methods.items():
        r = evaluate_projections(frame.with_columns(pl.col(col).alias("projection")), stats)["by_position"][POSITION]
        results[label][method] = {
            "n": r["n"], "mae": r["projection"]["mae"], "r2": r["projection"]["r2"],
            "bias": r["projection"]["mean_error"], "spearman": r["spearman_rank_correlation"]["mean"],
        }

# ---- report ----
methods = list(next(iter(results.values())).keys())
print(f"\n===== {POSITION} per-fold (same rows per fold; bias = actual - predicted) =====")
for label, by_m in results.items():
    print(f"\n--- test {label} (rows={by_m['new model']['n']}) ---")
    for m, v in by_m.items():
        print(f"  {m:16s} MAE={_fmt(v['mae'])}  R2={_fmt(v['r2'])}  bias={_fmt(v['bias'])}  Spearman={_fmt(v['spearman'])}")

full = [l for l, _, _ in loso_folds() if "partial" not in l]
print(f"\n===== LOSO summary over {len(full)} full-season folds =====")
for m in methods:
    agg = {k: sum(results[l][m][k] for l in full) / len(full) for k in ["mae", "r2", "spearman", "bias"]}
    print(f"  {m:16s} mean MAE={_fmt(agg['mae'])}  mean R2={_fmt(agg['r2'])}  "
          f"mean Spearman={_fmt(agg['spearman'])}  mean bias={_fmt(agg['bias'])}")
for m in methods:
    if m in ("current formula", "rolling alone"):
        continue
    for b in ("current formula", "rolling alone"):
        w = {k: sum((results[l][m][k] < results[l][b][k]) if k == "mae" else (results[l][m][k] > results[l][b][k])
                    for l in full) for k in ["mae", "r2", "spearman"]}
        print(f"  {m} beats {b}: MAE {w['mae']}/{len(full)}, R2 {w['r2']}/{len(full)}, Spearman {w['spearman']}/{len(full)}")

print(f"\n===== coefficient stability across {len(full)} LOSO folds (new model) =====")
for target in SPEC.models:
    rows = []
    for l in full:
        s = models_by_fold[l].fits[target].summary()
        rows.append(s.with_columns(pl.lit(l).alias("fold")))
    allf = pl.concat(rows)
    print(f"\n[{target}]  (weighted by denominator)" if SPEC.models[target][1] else f"\n[{target}]")
    print(allf.group_by("term", maintain_order=True).agg(
        pl.col("coef").min().alias("min"), pl.col("coef").max().alias("max"),
        pl.col("coef").mean().alias("mean"),
        (pl.col("coef") > 0).sum().alias("n_pos"),
        (pl.col("p") < 0.05).sum().alias("n_p<.05"),
        pl.col("p").max().alias("max_p"),
    ))

for vname in variant_specs:
    for target, terms in SPEC.candidate_terms.items():
        for term in terms:
            print(f"\n===== variant {vname}: candidate {term} in [{target}] per fold =====")
            for l in full:
                s = models_by_fold[f"{l}|{vname}"].fits[target].summary().filter(pl.col("term") == term)
                print(f"  {l}: coef={_fmt(s['coef'][0])}  p={_fmt(s['p'][0])}")
