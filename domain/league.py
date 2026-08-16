
from dataclasses import dataclass, field

from domain.team import Team
from domain.scoring import ScoringSettings


@dataclass
class League:
    league_id: str
    league_name: str
    season: str
    scoring: ScoringSettings
    roster_positions: list[str]
    teams: list[Team] = field(default_factory=list)

    def __eq__(self, other) -> bool:
        if not isinstance(other, League):
            return NotImplemented
        return self.league_id == other.league_id

    def __hash__(self) -> int:
        return hash(self.league_id)

    def standings(self) -> list[Team]:
        """Teams sorted by wins desc, then points_for desc (standard tiebreak)."""
        return sorted(self.teams, key=lambda t: (-t.wins, -t.points_for))

    def get_team(self, roster_id: str) -> Team | None:
        return next((t for t in self.teams if t.roster_id == roster_id), None)