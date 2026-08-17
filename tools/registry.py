
from tools.expected_points import ExpectedPointsArgs, get_expected_points


TOOL_REGISTRY = {
    "expected_points": {
        "args_model": ExpectedPointsArgs,
        "func": get_expected_points,
        "description": (
            "Projects a player's fantasy points for a given week as rolling "
            "average plus an opponent-matchup adjustment. Confidence reflects "
            "player-side games-played sample size only."
        ),
    },
}
