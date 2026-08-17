
from pydantic import BaseModel
import polars as pl

from domain.tool_result import ToolResult


class ExpectedPointsArgs(BaseModel):
    gsis_id: str
    week: int
    season: int


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

    metadata = {
        "gsis_id": args.gsis_id,
        "week": args.week,
        "season": args.season,
        "position": row["position"],
        "opponent_team": row["opponent_team"],
        "rolling_avg_prior": row["rolling_avg_prior"],
        "opponent_skew": row["opponent_skew"],
        "opponent_skew_n_games": row["opponent_skew_n_games"],
        "games_this_season": games_this_season,
    }

    if projection is None:
        return ToolResult(
            value=None,
            confidence=row["confidence"],
            explanation=(
                f"No rolling average available yet for player {args.gsis_id} at "
                f"week {args.week}, season {args.season} — insufficient trailing "
                f"game history to compute rolling_avg_prior."
            ),
            metadata=metadata,
        )

    explanation = (
        f"projection = rolling_avg_prior ({row['rolling_avg_prior']:.2f}) + "
        f"opponent_skew ({row['opponent_skew']:.2f}) vs {row['opponent_team']}. "
        f"Confidence reflects {games_this_season} game(s) played by this player "
        f"this season — player-side sample size only, not opponent_skew's sample "
        f"size or outcome volatility."
    )

    return ToolResult(
        value=round(projection, 2),
        confidence=row["confidence"],
        explanation=explanation,
        metadata=metadata,
    )
