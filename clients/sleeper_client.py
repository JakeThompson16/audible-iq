
"""
Minimal example: pull leagues, accounts (users), and settings from Sleeper's API.
No auth needed, it's a free read-only API.
"""

import requests

from domain.scoring import ScoringSettings

BASE_URL = "https://api.sleeper.app/v1"


def get_user(username: str) -> dict:
    """Resolve a username to Sleeper's internal user_id + profile info."""
    resp = requests.get(f"{BASE_URL}/user/{username}")
    resp.raise_for_status()
    return resp.json()


def get_user_leagues(user_id: str, season: str, sport: str = "nfl") -> list:
    """Get all leagues a user is in for a given season."""
    resp = requests.get(f"{BASE_URL}/user/{user_id}/leagues/{sport}/{season}")
    resp.raise_for_status()
    return resp.json()


def get_league_users(league_id: str) -> list:
    """Get all accounts (members) in a league."""
    resp = requests.get(f"{BASE_URL}/league/{league_id}/users")
    resp.raise_for_status()
    return resp.json()


def get_league_rosters(league_id: str) -> list:
    """Get roster data (owner -> players) for a league."""
    resp = requests.get(f"{BASE_URL}/league/{league_id}/rosters")
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    username = "jakethompson16"
    season = "2026"

    # 1. Resolve username -> user_id
    user = get_user(username)
    print(f"User: {user['display_name']} (id: {user['user_id']})")

    # 2. Load all leagues for this user this season
    leagues = get_user_leagues(user["user_id"], season)
    print(f"\nFound {len(leagues)} league(s):\n")

    for league in leagues:
        print(f"League: {league['name']} (id: {league['league_id']})")

        # 3. Pull scoring settings straight from the league object
        scoring_settings = ScoringSettings.from_dict(league["scoring_settings"])
        print(f"  Scoring settings: {scoring_settings}")

        roster_positions = league["roster_positions"]
        print(f"  Roster positions: {roster_positions}")

        # 4. Pull accounts (members) in this league
        members = get_league_users(league["league_id"])
        print(f"  Members ({len(members)}):")
        for m in members:
            team_name = m.get("metadata", {}).get("team_name", "(no team name set)")
            print(f"    - {m['display_name']} | team: {team_name}")

        print()