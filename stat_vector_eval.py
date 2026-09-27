"""
Stat-vector backtest for one position: new model vs current formula
(rolling_avg_prior + opponent_skew, tuned window) vs rolling-average-alone,
leave-one-season-out.

Run: python stat_vector_eval.py QB [--grid] [--candidates]
  --grid        grid-search the stat vector's own feature window
                (STAT_VECTOR_WINDOW_GRID), choose by select_by_policy, report at it
  --candidates  also fit a variant with the spec's candidate_terms switched on
"""
import sys
from dataclasses import replace

import polars as pl

from clients.sleeper_client import get_user, get_user_leagues
from domain.scoring import ScoringSettings
from engine.expected_points import ExpectedPointsContext, calculate_expected_points
from engine.metrics import evaluate_projections
from evaluation.backtest import (
    STAT_VECTOR_WINDOW_GRID, load_inputs, loso_folds, predict_fold, select_by_policy,
)
from projections.expected_points.stat_vector.core import KEY_COLUMNS, recompose_points
from projections.expected_points.stat_vector.specs import STAT_VECTOR_SPECS

POSITION = sys.argv[1] if len(sys.argv) > 1 else "RB"
GRID = "--grid" in sys.argv
CANDIDATES = "--candidates" in sys.argv
SPEC = STAT_VECTOR_SPECS[POSITION]
K3 = ["gsis_id", "season", "week"]

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
all_positions = stats["position"].unique().drop_nulls().to_list()

# Current formula (tuned per-position window) for every position.
current = calculate_expected_points(
    stats, ExpectedPointsContext(skew_df=inputs.skew, rolling_windows=inputs.rolling_windows),
    {p: "rolling_plus_skew" for p in all_positions},
).select(K3 + ["projection"]).rename({"projection": "current"})

# ---- sanity ----
print("===== sanity =====")
pos_rows = stats.filter(pl.col("position") == POSITION)
print("player-weeks by season:", pos_rows.group_by("season").len().sort("season").rows())
actual = pos_rows.filter(pl.col("season").is_in([2024, 2025]))
recomp = recompose_points(actual, SPEC, scoring).rename({"fantasy_points": "recomposed"}) \
    .join(actual.select(KEY_COLUMNS + ["fantasy_points"]), on=KEY_COLUMNS)
gap = recomp["fantasy_points"] - recomp["recomposed"]
print(f"recomposing ACTUAL {POSITION} stats (2024/25): mean gap={_fmt(gap.mean())}  "
      f"mean|gap|={_fmt(gap.abs().mean())}  (cost of unpredicted categories)")

# Only the evaluated position is fit; everything else runs the cheap formula.
impl = {**{p: "rolling_plus_skew" for p in all_positions}, POSITION: "stat_vector"}
FULL = [l for l, _, _ in loso_folds() if "partial" not in l]


def run(spec, extra_specs=None):
    """LOSO for one spec (+ optional variants). Returns (results, models_by_fold)."""
    extra_specs = extra_specs or {}
    results, models_by_fold = {}, {}
    for label, train, test in loso_folds():
        preds, models = predict_fold(inputs, train, test, impl, specs={**STAT_VECTOR_SPECS, POSITION: spec})
        models_by_fold[label] = models[POSITION]
        frame = (
            preds.filter(pl.col("position") == POSITION)
            .select(KEY_COLUMNS + ["projection", "rolling_avg_prior", "confidence"])
            .rename({"projection": "new model"})
            .join(current, on=K3, how="left")
        )
        methods = {"new model": "new model", "current formula": "current", "rolling alone": "rolling_avg_prior"}
        for vname, vspec in extra_specs.items():
            vpreds, vmodels = predict_fold(inputs, train, test, impl, specs={**STAT_VECTOR_SPECS, POSITION: vspec})
            models_by_fold[f"{label}|{vname}"] = vmodels[POSITION]
            frame = frame.join(vpreds.filter(pl.col("position") == POSITION).select(K3 + ["projection"])
                               .rename({"projection": vname}), on=K3, how="left")
            methods[vname] = vname
        frame = frame.drop_nulls(subset=list(methods.values()))
        results[label] = {}
        for method, col in methods.items():
            r = evaluate_projections(frame.with_columns(pl.col(col).alias("projection")), stats)["by_position"][POSITION]
            results[label][method] = {
                "n": r["n"], "mae": r["projection"]["mae"], "r2": r["projection"]["r2"],
                "bias": r["projection"]["mean_error"], "spearman": r["spearman_rank_correlation"]["mean"],
            }
    return results, models_by_fold


def means(results, method):
    return {k: sum(results[l][method][k] for l in FULL) / len(FULL) for k in ["mae", "r2", "spearman", "bias"]}


# ---- window grid ----
if GRID:
    print("\n===== stat-vector window grid (new model, LOSO means over full seasons) =====")
    grid = {}
    for w in STAT_VECTOR_WINDOW_GRID:
        res, _ = run(replace(SPEC, window=w))
        grid[w] = (res, means(res, "new model"))
        m = grid[w][1]
        p = res[f"{2026} partial"]["new model"]
        print(f"  window {w:2d}: MAE={_fmt(m['mae'])}  R2={_fmt(m['r2'])}  Spearman={_fmt(m['spearman'])}  "
              f"bias={_fmt(m['bias'])}  | 2026 partial MAE={_fmt(p['mae'])} Sp={_fmt(p['spearman'])}")
    best = select_by_policy({w: g[1] for w, g in grid.items()})
    print(f"  chosen by policy (Spearman, then R2, then MAE): window {best} (spec default {SPEC.window})")
    SPEC = replace(SPEC, window=best)

variants = {"+candidates": SPEC.with_candidates()} if CANDIDATES and SPEC.candidate_terms else {}
results, models_by_fold = run(SPEC, variants)

# ---- report ----
methods = list(next(iter(results.values())).keys())
print(f"\n===== {POSITION} per-fold at stat-vector window {SPEC.window} (same rows per fold; bias = actual - predicted) =====")
for label, by_m in results.items():
    print(f"\n--- test {label} (rows={by_m['new model']['n']}) ---")
    for m, v in by_m.items():
        print(f"  {m:16s} MAE={_fmt(v['mae'])}  R2={_fmt(v['r2'])}  bias={_fmt(v['bias'])}  Spearman={_fmt(v['spearman'])}")

print(f"\n===== LOSO summary over {len(FULL)} full-season folds =====")
for m in methods:
    a = means(results, m)
    print(f"  {m:16s} mean MAE={_fmt(a['mae'])}  mean R2={_fmt(a['r2'])}  "
          f"mean Spearman={_fmt(a['spearman'])}  mean bias={_fmt(a['bias'])}")
for m in methods:
    if m in ("current formula", "rolling alone"):
        continue
    for b in ("current formula", "rolling alone"):
        w = {k: sum((results[l][m][k] < results[l][b][k]) if k == "mae" else (results[l][m][k] > results[l][b][k])
                    for l in FULL) for k in ["mae", "r2", "spearman"]}
        print(f"  {m} beats {b}: MAE {w['mae']}/{len(FULL)}, R2 {w['r2']}/{len(FULL)}, Spearman {w['spearman']}/{len(FULL)}")

print(f"\n===== coefficient stability across {len(FULL)} LOSO folds (new model) =====")
for target in SPEC.models:
    allf = pl.concat([models_by_fold[l].fits[target].summary() for l in FULL])
    print(f"\n[{target}]  (weighted by denominator)" if SPEC.models[target][1] else f"\n[{target}]")
    print(allf.group_by("term", maintain_order=True).agg(
        pl.col("coef").min().alias("min"), pl.col("coef").max().alias("max"),
        pl.col("coef").mean().alias("mean"),
        (pl.col("coef") > 0).sum().alias("n_pos"),
        (pl.col("p") < 0.05).sum().alias("n_p<.05"),
        pl.col("p").max().alias("max_p"),
    ))

for vname in variants:
    for target, terms in SPEC.candidate_terms.items():
        for term in terms:
            print(f"\n===== variant {vname}: candidate {term} in [{target}] per fold =====")
            for l in FULL:
                s = models_by_fold[f"{l}|{vname}"].fits[target].summary().filter(pl.col("term") == term)
                print(f"  {l}: coef={_fmt(s['coef'][0])}  p={_fmt(s['p'][0])}")
