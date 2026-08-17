
import polars as pl


POSITIONS = ["QB", "RB", "WR", "TE"]
CONFIDENCE_TIERS = ["high", "medium", "low"]  # insufficient_data excluded: by
# construction (games_this_season == 0 => rolling window unfilled), it never
# has a non-null projection, so it never has a scoreable row.


def _regression_metrics(df: pl.DataFrame, error_col: str) -> dict:
    """
    :param df: rows with an `actual` column and a signed `error_col` (actual - predicted)
    """
    n = df.height
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "mean_error": None, "r2": None}

    abs_err = df[error_col].abs()
    sq_err = df[error_col] ** 2

    mae = abs_err.mean()
    rmse = sq_err.mean() ** 0.5
    mean_error = df[error_col].mean()

    actual_var = df["actual"].var(ddof=0)
    r2 = 1 - (sq_err.mean() / actual_var) if actual_var else None

    return {"n": n, "mae": mae, "rmse": rmse, "mean_error": mean_error, "r2": r2}


def _projection_vs_baseline(df: pl.DataFrame) -> dict:
    proj = df.with_columns((pl.col("actual") - pl.col("projection")).alias("error"))
    base = df.with_columns((pl.col("actual") - pl.col("rolling_avg_prior")).alias("error"))

    proj_metrics = _regression_metrics(proj, "error")
    base_metrics = _regression_metrics(base, "error")

    skew_improves_mae = (
        proj_metrics["mae"] is not None
        and base_metrics["mae"] is not None
        and proj_metrics["mae"] < base_metrics["mae"]
    )

    return {
        "projection": proj_metrics,
        "baseline_rolling_avg_only": base_metrics,
        "skew_improves_mae": skew_improves_mae,
    }


def _calibration(df: pl.DataFrame) -> dict:
    by_tier = {}
    for tier in CONFIDENCE_TIERS:
        sub = df.filter(pl.col("confidence") == tier).with_columns(
            (pl.col("actual") - pl.col("projection")).alias("error")
        )
        by_tier[tier] = _regression_metrics(sub, "error")

    maes = [by_tier[t]["mae"] for t in CONFIDENCE_TIERS]  # [high, medium, low]
    monotonic_decreasing_mae = (
        all(m is not None for m in maes) and maes[0] <= maes[1] <= maes[2]
    )

    return {
        "by_tier": by_tier,
        "monotonic_decreasing_mae": monotonic_decreasing_mae,
        "note": (
            "Confidence thresholds are provisional (see CLAUDE.md). Expect "
            "MAE(high) <= MAE(medium) <= MAE(low) once calibrated — if this is "
            "False, the tier thresholds need adjustment."
        ),
    }


def _spearman_by_week(df: pl.DataFrame, min_group_size: int = 3) -> dict:
    by_week = {}
    skipped = 0
    for (season, week), group in df.group_by(["season", "week"], maintain_order=True):
        if group.height < min_group_size:
            skipped += 1
            continue
        corr = group.select(pl.corr("projection", "actual", method="spearman")).item()
        by_week[f"{season}-wk{week}"] = corr

    values = [v for v in by_week.values() if v is not None]
    mean_corr = sum(values) / len(values) if values else None

    return {"by_week": by_week, "mean": mean_corr, "weeks_skipped_too_few_players": skipped}


def evaluate_projections(predictions_df: pl.DataFrame, actuals_df: pl.DataFrame) -> dict:
    """
    :param predictions_df: output of engine.expected_points.calculate_expected_points()
        — must include gsis_id, season, week, position, projection,
        rolling_avg_prior, confidence
    :param actuals_df: player-week stats with ground-truth fantasy_points —
        must include gsis_id, season, week, fantasy_points
    :return: dict of MAE/RMSE/mean_error/R2 (projection vs. rolling_avg_prior-only
        baseline), Spearman rank correlation per week, all segmented by position,
        plus a confidence-tier calibration check. MAPE intentionally omitted —
        unstable/undefined near zero-point fantasy outcomes.

    Metrics are segmented by position (QB/RB/WR/TE) because point-total scale
    differs enormously by position — an aggregate MAE/RMSE across positions is
    close to meaningless. Spearman rank correlation is computed within
    (season, week, position) groups since start/sit decisions rank players
    against same-position alternatives, not against the whole league.
    """
    n_total = predictions_df.height

    df = predictions_df.join(
        actuals_df.select(["gsis_id", "season", "week", "fantasy_points"])
                  .rename({"fantasy_points": "actual"}),
        on=["gsis_id", "season", "week"],
        how="inner",
    )

    n_other_position = df.filter(~pl.col("position").is_in(POSITIONS)).height
    df = df.filter(pl.col("position").is_in(POSITIONS))

    scoreable = df.filter(pl.col("projection").is_not_null() & pl.col("actual").is_not_null())
    n_null_projection = df.height - scoreable.height

    by_position = {}
    for position in POSITIONS:
        pos_df = scoreable.filter(pl.col("position") == position)
        by_position[position] = {
            "n": pos_df.height,
            **_projection_vs_baseline(pos_df),
            "spearman_rank_correlation": _spearman_by_week(pos_df),
            "calibration": _calibration(pos_df),
        }

    return {
        "by_position": by_position,
        "calibration_overall": _calibration(scoreable),
        "meta": {
            "n_predictions": n_total,
            "n_joined_to_actuals": df.height + n_other_position,
            "n_other_position_dropped": n_other_position,
            "n_null_projection_dropped": n_null_projection,
            "n_scored": scoreable.height,
            "note": "MAPE intentionally omitted — undefined/unstable near zero-point fantasy outcomes.",
        },
    }
