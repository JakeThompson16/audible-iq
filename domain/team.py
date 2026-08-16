
from dataclasses import dataclass, field

from domain.player import Player


@dataclass
class Team:
    roster_id: str
    team_name: str
    owner_id: str

    wins: int = 0
    losses: int = 0
    points_for: float = 0.0
    points_against: float = 0.0

    players: list[Player] = field(default_factory=list)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Team):
            return NotImplemented
        return self.roster_id == other.roster_id and self.owner_id == other.owner_id

    def __hash__(self):
        return hash((self.roster_id, self.owner_id))



