
from pydantic import BaseModel
import polars as pl

from domain.tool_result import ToolResult


class ExpectedPointsArgs(BaseModel):
    gsis_id: str
    week: int
    season: int


# How each engine implementation's projection is described to the agent.
# Keyed by projection_method, same names as engine IMPLEMENTATIONS.
_METHOD_EXPLANATIONS = {
    "rolling_plus_skew": lambda r: (
        f"projection = rolling_avg_prior ({r['rolling_avg_prior']:.2f}, last "
        f"{r.get('rolling_window') or '?'} games) + opponent_skew ({r['opponent_skew']:.2f}) "
        f"vs {r['opponent_team']}."
    ),
    "stat_vector": lambda r: (
        f"projection = this league's scoring applied to a predicted stat line "
        f"(stat vector: predicted volume — pass attempts, targets, carries — from "
        f"recent usage, times per-attempt rates adjusted for {r['opponent_team']}'s "
        f"EPA allowed). "
        f"opponent_skew is not part of this projection."
    ),
}


def _explain_method(method: str, row: dict) -> str:
    explain = _METHOD_EXPLANATIONS.get(method)
    return explain(row) if explain else f"projection produced by '{method}'."


def get_expected_points(args: ExpectedPointsArgs, projections_df: pl.DataFrame) -> ToolResult:
    """
    :param args: identifies which player/week/season to look up
    :param projections_df: output of engine.expected_points.calculate_expected_points()

    explanation/confidence are agent-facing only — a UI adapter displaying this
    projection to a user should read only .value off the returned ToolResult.
    """
    row = projections_df.filter(
        (pl.col("gsis_id") == args.gsis_id)
        & (pl.col("week") == args.week)
        & (pl.col("season") == args.season)
    )

    if row.is_empty():
        return ToolResult(
            value=None,
            confidence="insufficient_data",
            explanation=(
                f"No stats found for player {args.gsis_id}, week {args.week}, "
                f"season {args.season}."
            ),
            metadata={"gsis_id": args.gsis_id, "week": args.week, "season": args.season},
        )

    row = row.row(0, named=True)

    projection = row["projection"]
    games_this_season = row["games_this_season"]
    # Which engine implementation produced this row (engine/expected_points.py
    # POSITION_IMPLEMENTATIONS). Older outputs without the column were all
    # rolling_plus_skew.
    method = row.get("projection_method") or "rolling_plus_skew"

    metadata = {
        "gsis_id": args.gsis_id,
        "week": args.week,
        "season": args.season,
        "position": row["position"],
        "opponent_team": row["opponent_team"],
        "projection_method": method,
        "rolling_avg_prior": row["rolling_avg_prior"],
        "rolling_window": row.get("rolling_window"),
        "opponent_skew": row["opponent_skew"],
        "opponent_skew_n_games": row["opponent_skew_n_games"],
        "games_this_season": games_this_season,
    }

    if projection is None:
        return ToolResult(
            value=None,
            confidence=row["confidence"],
            explanation=(
                f"No projection available for player {args.gsis_id} at week "
                f"{args.week}, season {args.season} — no prior games to build "
                f"trailing features from ({method})."
            ),
            metadata=metadata,
        )

    explanation = (
        f"{_explain_method(method, row)} "
        f"Confidence reflects {games_this_season} game(s) played by this player "
        f"this season — player-side sample size only, not the matchup's sample "
        f"size or outcome volatility."
    )

    return ToolResult(
        value=round(projection, 2),
        confidence=row["confidence"],
        explanation=explanation,
        metadata=metadata,
    )
