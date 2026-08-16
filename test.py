
from clients.sleeper_client import get_user, get_user_leagues
from domain.scoring import ScoringSettings

username = "jakethompson16"
season = "2026"

# 1. Resolve username -> user_id
user = get_user(username)
print(f"User: {user['display_name']} (id: {user['user_id']})")

# 2. Load all leagues for this user this season
leagues = get_user_leagues(user["user_id"], season)
print(f"\nFound {len(leagues)} league(s):\n")

league = leagues[1]
scoring_settings = ScoringSettings.from_dict(league["scoring_settings"])

from clients.nflreadpy.player_data import load_player_stats
import polars as pl
from engine.scoring import calculate_points_vectorized

df = load_player_stats(2025)

df = df.filter(
    pl.col('display_name') == 'Justin Jefferson'
)

df = calculate_points_vectorized(df, scoring_settings)

for row in df.iter_rows(named=True):
    print(row['fantasy_points'])
