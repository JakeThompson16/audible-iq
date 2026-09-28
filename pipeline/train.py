"""
Retrain the production stat-vector models and regenerate their artifacts.

    python -m pipeline.train retrain [--start 2022] [--end 2026] [--dry-run]

Fits every position in STAT_VECTOR_SPECS on target rows from start_season
through the last fully completed week, using fit_stat_vector exactly as the
backtest does. Writes artifacts/stat_vector/<POS>.json (safe overwrite, one
.prev generation kept) and regenerates artifacts/MODEL_METRICS.md in full.
Idempotent: rerunning on the same data produces the same artifacts (only the
fit timestamp changes). Safe to schedule weekly, e.g. Tuesday morning after
Monday Night Football is final; see CLAUDE.md "Retraining".
"""
import argparse
import sys
from datetime import datetime, timezone

import polars as pl

from clients.sleeper_client import get_user, get_user_leagues
from domain.scoring import ScoringSettings
from engine.expected_points import POSITION_ROLLING_WINDOWS
from engine.metrics import POSITIONS, evaluate_projections
from engine.scoring import calculate_points_vectorized
from evaluation.backtest import BacktestInputs, predict_fold
from pipeline.artifacts import (
    STAT_VECTOR_DIR, coefficient_drift, load_reference_ranges, model_to_artifact, read_all, validate, write_all,
)
from pipeline.context import DEFAULT_START_SEASON, FeatureContext, get_context, through_mask
from projections.expected_points.features.epa_allowed import epa_allowed_from_weekly
from projections.expected_points.features.opponent_skew import calculate_all_position_skews
from projections.expected_points.features.player_rolling import add_rolling_features
from projections.expected_points.stat_vector.core import build_features, fit_stat_vector
from projections.expected_points.stat_vector.core import unpredicted_categories as _unpredicted
from projections.expected_points.stat_vector.specs import STAT_VECTOR_SPECS

# League whose scoring the out-of-sample section uses (same as test.py / the README).
METRICS_SLEEPER_USER = "jakethompson16"
METRICS_LEAGUE_INDEX = 1
WEIGHT_RATES = True


def default_metrics_scoring(season: int) -> ScoringSettings:
    user = get_user(METRICS_SLEEPER_USER)
    league = get_user_leagues(user["user_id"], str(season))[METRICS_LEAGUE_INDEX]
    return ScoringSettings.from_dict(league["scoring_settings"])


def unpredicted_categories(position: str) -> list[str]:
    return _unpredicted(STAT_VECTOR_SPECS[position])


def _cut(ctx: FeatureContext) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Player rows and epa_allowed restricted to fully completed weeks."""
    through = through_mask(ctx.data_through)
    rows = ctx.stats.filter(through)
    epa = epa_allowed_from_weekly(ctx.pass_weekly.filter(through), ctx.rush_weekly.filter(through))
    return rows, epa


def _trained_on(rows: pl.DataFrame, seasons: list[int]) -> dict:
    weeks = (rows.filter(pl.col("season").is_in(seasons)).group_by("season")
             .agg(pl.col("week").min().alias("first"), pl.col("week").max().alias("last"),
                  pl.col("week").n_unique().alias("n_weeks"))
             .sort("season"))
    return {"seasons": seasons,
            "weeks": {str(s): {"first": f, "last": l, "n_weeks": n} for s, f, l, n in weeks.iter_rows()}}


def train_models(ctx: FeatureContext, start_season: int) -> tuple[dict, dict, pl.DataFrame, pl.DataFrame]:
    """Fit every spec. Returns (models, artifacts, rows, epa) — artifacts without timestamps."""
    rows, epa = _cut(ctx)
    ds, dw = ctx.data_through
    seasons = [s for s in range(start_season, ds + 1)]
    models, artifacts = {}, {}
    for pos, spec in STAT_VECTOR_SPECS.items():
        model = fit_stat_vector(rows, epa, spec, seasons, weight_rates=WEIGHT_RATES)
        pos_rows = rows.filter((pl.col("position") == pos) & pl.col("season").is_in(seasons))
        meta = {
            "trained_on": _trained_on(pos_rows, seasons),
            "data_through": {"season": ds, "week": dw},
            "row_counts": {"player_weeks": pos_rows.height,
                           "sub_models": {t: int(f.n) for t, f in model.fits.items()}},
        }
        models[pos] = model
        artifacts[pos] = model_to_artifact(model, meta, weight_rates=WEIGHT_RATES)
    return models, artifacts, rows, epa


# ---------------------------------------------------------------- metrics

def _fmt(x, d=3):
    if x is None:
        return "–"
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def _oos(ctx, rows, epa, scoring, start_season) -> dict:
    """
    Leave-one-season-out inside the training window: each full season scored
    by a model fit on the window's other full seasons; the partial current
    season is a test-only fold fit on all of them.
    """
    ds, dw = ctx.data_through
    last_week = ctx.schedule.filter(pl.col("season") == ds)["week"].max()
    full = [s for s in range(start_season, ds + 1) if s < ds or dw >= last_week]
    partial = ds if ds not in full else None

    stats = add_rolling_features(calculate_points_vectorized(rows, scoring), window=POSITION_ROLLING_WINDOWS)
    games = ctx.schedule.filter(pl.col("game_type") == "REG").select(["team", "opponent", "week", "season"])
    inputs = BacktestInputs(stats=stats, skew=calculate_all_position_skews(stats, games), epa=epa,
                            scoring=scoring, rolling_windows=dict(POSITION_ROLLING_WINDOWS))

    folds = [(str(s), [o for o in full if o != s], s) for s in full]
    if partial is not None:
        folds.append((f"{partial} partial (through wk {dw})", full, partial))

    per_fold, full_preds = {}, []
    for label, train, test in folds:
        preds, _ = predict_fold(inputs, train, test)
        if test in full:
            full_preds.append(preds)
        res = {}
        for method, col in [("stat_vector", "projection"), ("rolling_alone", "rolling_avg_prior")]:
            r = evaluate_projections(preds.with_columns(pl.col(col).alias("projection")), stats)
            res[method] = {pos: {"n": r["by_position"][pos]["n"],
                                 "mae": r["by_position"][pos]["projection"]["mae"],
                                 "r2": r["by_position"][pos]["projection"]["r2"],
                                 "bias": r["by_position"][pos]["projection"]["mean_error"],
                                 "spearman": r["by_position"][pos]["spearman_rank_correlation"]["mean"]}
                           for pos in POSITIONS}
        per_fold[label] = res
    pooled = evaluate_projections(pl.concat(full_preds, how="diagonal_relaxed"), stats) if full_preds else None
    return {"full": full, "partial": partial, "per_fold": per_fold, "pooled": pooled}


def _null_rates(rows, epa, through) -> dict:
    out = {}
    for pos, spec in STAT_VECTOR_SPECS.items():
        feats = build_features(rows.filter(pl.col("position") == pos), epa, spec.window)
        latest = feats.filter((pl.col("season") == through[0]) & (pl.col("week") == through[1]))
        cols = sorted({f for fs, _ in spec.models.values() for f in fs})
        out[pos] = (latest.height, {c: (latest[c].null_count() / latest.height if latest.height else None) for c in cols})
    return out


def build_metrics_md(ctx, artifacts, oos, drift, null_rates, warnings, start_season, generated_at) -> str:
    ds, dw = ctx.data_through
    L = []
    L.append("# Stat-vector model metrics\n")
    L.append("_Regenerated in full by `python -m pipeline.train retrain`; do not edit by hand._\n")
    L.append(f"- Generated: {generated_at}")
    L.append(f"- Data through: {ds} week {dw} (last week whose scheduled games are all final)")
    L.append(f"- Training seasons: {start_season}-{ds}; history-only seasons loaded: "
             f"{', '.join(str(s) for s in ctx.seasons if s < start_season)}")
    L.append("- Stat-vector windows: " + ", ".join(f"{p} {s.window}" for p, s in STAT_VECTOR_SPECS.items())
             + "; rolling-alone baseline windows: " + ", ".join(f"{p} {w}" for p, w in POSITION_ROLLING_WINDOWS.items()))
    L.append(f"- Rate models attempt-weighted: {WEIGHT_RATES}\n")

    L.append("## Coefficients (production fit)\n")
    for pos, art in artifacts.items():
        L.append(f"### {pos}  ({art['row_counts']['player_weeks']} player-weeks)\n")
        L.append("| Sub-model | Term | Coef | SE | p | n |")
        L.append("|---|---|---:|---:|---:|---:|")
        for target, sub in art["sub_models"].items():
            terms = ["intercept"] + sub["features"]
            vals = [sub["intercept"]] + sub["coefficients"]
            for i, (term, v) in enumerate(zip(terms, vals)):
                L.append(f"| {target if i == 0 else ''} | {term} | {v:.5g} | {sub['std_errors'][i]:.3g} | "
                         f"{sub['p_values'][i]:.3g} | {sub['n'] if i == 0 else ''} |")
        L.append("")

    L.append("## In-sample fit (in-sample, not an accuracy claim)\n")
    L.append("| Pos | In-sample R² per sub-model (rate models: weighted R²) |")
    L.append("|---|---|")
    for pos, art in artifacts.items():
        L.append(f"| {pos} | " + ", ".join(f"{t} {sub['r2_in_sample']:.3f}" for t, sub in art["sub_models"].items()) + " |")
    L.append("")

    L.append("## Out-of-sample accuracy (leave-one-season-out inside the training window)\n")
    L.append(f"Each full season ({', '.join(map(str, oos['full']))}) is scored by models fit on the window's other "
             f"full seasons" + (f"; {oos['partial']} (partial) is test-only, fit on all of them" if oos["partial"] else "")
             + ". Same harness and league scoring as the README; fewer, shorter folds than the README's "
               "2019-2025 table, so numbers are not identical. Cells: MAE / R² / Spearman / bias (actual − predicted).\n")
    L.append("| Pos | Fold | n | Stat vector | Rolling alone |")
    L.append("|---|---|---:|---|---|")
    full_labels = [str(s) for s in oos["full"]]
    for pos in POSITIONS:
        def cell(m):
            return f"{_fmt(m['mae'])} / {_fmt(m['r2'])} / {_fmt(m['spearman'])} / {_fmt(m['bias'], 2)}"
        for label, res in oos["per_fold"].items():
            L.append(f"| {pos} | {label} | {res['stat_vector'][pos]['n']} | {cell(res['stat_vector'][pos])} | "
                     f"{cell(res['rolling_alone'][pos])} |")
        if full_labels:
            mean = {m: {k: sum(oos["per_fold"][l][m][pos][k] for l in full_labels) / len(full_labels)
                        for k in ["mae", "r2", "spearman", "bias"]} for m in ["stat_vector", "rolling_alone"]}
            L.append(f"| **{pos}** | **mean of full seasons** | | **{cell(mean['stat_vector'])}** | {cell(mean['rolling_alone'])} |")
    L.append("")

    if oos["pooled"]:
        L.append("### Confidence calibration (pooled full-season folds)\n")
        L.append("Per tier: MAE / mean actual points / normalized MAE. Judge on normalized MAE (Q-15).\n")
        L.append("| Pos | high | medium | low | insufficient | normalized monotonic |")
        L.append("|---|---|---|---|---|---|")
        for pos in POSITIONS + ["ALL"]:
            c = oos["pooled"]["calibration_overall"] if pos == "ALL" else oos["pooled"]["by_position"][pos]["calibration"]
            t = c["by_tier"]
            L.append(f"| {pos} | " + " | ".join(
                f"{_fmt(t[k]['mae'])} / {_fmt(t[k]['mean_actual'], 2)} / {_fmt(t[k]['normalized_mae'])}"
                for k in ["high", "medium", "low", "insufficient_data"]) + f" | {c['monotonic_decreasing_normalized_mae']} |")
        L.append("")

    L.append("## Coefficient drift vs 2019-2025 leave-one-season-out ranges\n")
    flagged = [(pos, d) for pos, rows in drift.items() for d in rows if d["status"] != "in range"]
    total = sum(len(r) for r in drift.values())
    L.append(f"{total - len(flagged)} of {total} coefficients inside their 2019-2025 fold range. Flagged:\n")
    if flagged:
        L.append("| Pos | Sub-model | Term | Value | 2019-2025 range | Status |")
        L.append("|---|---|---|---:|---|---|")
        for pos, d in flagged:
            rng = "–" if d["min"] is None else f"{d['min']:.4g} … {d['max']:.4g}"
            L.append(f"| {pos} | {d['target']} | {d['term']} | {d['value']:.4g} | {rng} | {d['status']} |")
    else:
        L.append("None.")
    L.append("")

    L.append("## Warnings\n")
    L.append(f"### Feature null rates, latest completed week ({ds} wk {dw})\n")
    L.append("Nulls before rate-prior filling; a null volume feature means no projection (no history).\n")
    for pos, (n, rates) in null_rates.items():
        nonzero = {c: r for c, r in rates.items() if r}
        L.append(f"- {pos} ({n} rows): " + (", ".join(f"{c} {r:.1%}" for c, r in nonzero.items()) or "no nulls"))
    L.append("\n### Unpredicted scoring categories (score 0 in every projection)\n")
    for pos in STAT_VECTOR_SPECS:
        L.append(f"- {pos}: {', '.join(unpredicted_categories(pos))}")
    if warnings:
        L.append("\n### Validation warnings\n")
        L.extend(f"- {w}" for w in warnings)
    L.append("")
    return "\n".join(L)


def train_all(
        start_season: int = DEFAULT_START_SEASON,
        end_season: int | None = None,
        save: bool = True,
        artifacts_dir=STAT_VECTOR_DIR,
        context: FeatureContext | None = None,
        metrics_scoring: ScoringSettings | None = None,
        reference: dict | None = None) -> dict:
    """
    Train every stat-vector position and (optionally) write artifacts.

    :param context: preloaded FeatureContext (default: get_context(start, end))
    :param metrics_scoring: scoring for the out-of-sample section (default: the
        README's Sleeper league)
    :return: {data_through, artifacts, warnings, errors, paths, metrics_md}.
        With save=True a validation error raises ArtifactValidationError and
        leaves existing artifacts untouched.
    """
    ctx = context or get_context(start_season, end_season)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reference = load_reference_ranges() if reference is None else reference

    models, artifacts, rows, epa = train_models(ctx, start_season)
    for art in artifacts.values():
        art["fit_timestamp"] = generated_at

    errors, warnings = validate(artifacts, read_all(artifacts_dir), reference)
    scoring = metrics_scoring or default_metrics_scoring(ctx.data_through[0])
    oos = _oos(ctx, rows, epa, scoring, start_season)
    drift = {pos: coefficient_drift(art, reference) for pos, art in artifacts.items()}
    metrics_md = build_metrics_md(ctx, artifacts, oos, drift, _null_rates(rows, epa, ctx.data_through),
                                  warnings, start_season, generated_at)

    paths = {}
    if save:
        warnings, paths = write_all(artifacts, metrics_md, artifacts_dir, reference)
    return {
        "data_through": ctx.data_through,
        "artifacts": artifacts,
        "models": models,
        "warnings": warnings,
        "errors": errors,
        "paths": {k: str(v) for k, v in paths.items()},
        "metrics_md": metrics_md,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline.train")
    sub = parser.add_subparsers(dest="command", required=True)
    rt = sub.add_parser("retrain", help="refit all positions and rewrite artifacts")
    rt.add_argument("--start", type=int, default=DEFAULT_START_SEASON)
    rt.add_argument("--end", type=int, default=None)
    rt.add_argument("--dry-run", action="store_true", help="fit and validate, don't write")
    args = parser.parse_args(argv)

    result = train_all(args.start, args.end, save=not args.dry_run)
    s, w = result["data_through"]
    print(f"data through {s} week {w}")
    for pos, art in result["artifacts"].items():
        print(f"  {pos}: {art['row_counts']['player_weeks']} player-weeks, sub-model rows "
              f"{art['row_counts']['sub_models']}")
    for e in result["errors"]:
        print(f"  ERROR {e}")
    for wmsg in result["warnings"]:
        print(f"  warning: {wmsg}")
    print("wrote:" if result["paths"] else "dry run, nothing written", *result["paths"].values(), sep="\n  ")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
