"""In-memory state for Studio Ops core API.

This is a thin reference implementation of the service layer. Replace with DB-backed
repositories for production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PlayerProfile:
    player_id: str
    game_id: str
    created_at: str = field(default_factory=utc_now_iso)
    last_login_at: str = field(default_factory=utc_now_iso)
    level: int = 1
    xp: int = 0
    soft_currency: int = 0
    hard_currency: int = 0
    inventory: dict[str, int] = field(default_factory=dict)
    sync_version: int = 1


class StudioStore:
    def __init__(self) -> None:
        self.players_by_game: dict[str, dict[str, PlayerProfile]] = {}
        self.sessions: dict[str, dict[str, str]] = {}

    def _game_bucket(self, game_id: str) -> dict[str, PlayerProfile]:
        if game_id not in self.players_by_game:
            self.players_by_game[game_id] = {}
        return self.players_by_game[game_id]

    def get_or_create_player(self, game_id: str, player_id: str | None = None) -> PlayerProfile:
        bucket = self._game_bucket(game_id)
        if player_id and player_id in bucket:
            profile = bucket[player_id]
            profile.last_login_at = utc_now_iso()
            return profile

        new_player_id = player_id or str(uuid4())
        profile = PlayerProfile(player_id=new_player_id, game_id=game_id)
        bucket[new_player_id] = profile
        return profile

    def create_session(self, game_id: str, player_id: str) -> str:
        sid = str(uuid4())
        self.sessions[sid] = {"game_id": game_id, "player_id": player_id}
        return sid

    def validate_session(self, session_id: str, game_id: str, player_id: str) -> bool:
        session = self.sessions.get(session_id)
        if not session:
            return False
        return session["game_id"] == game_id and session["player_id"] == player_id

    def get_sync_payload(self, game_id: str, player_id: str, last_sync_token: str | None) -> dict[str, Any]:
        profile = self.get_or_create_player(game_id=game_id, player_id=player_id)
        has_changes = last_sync_token != str(profile.sync_version)
        return {
            "profile": {
                "player_id": profile.player_id,
                "game_id": profile.game_id,
                "level": profile.level,
                "xp": profile.xp,
                "created_at": profile.created_at,
                "last_login_at": profile.last_login_at,
            },
            "wallet": {
                "soft_currency": profile.soft_currency,
                "hard_currency": profile.hard_currency,
            },
            "inventory": profile.inventory,
            "active_events": [],
            "offers": [],
            "server_time": utc_now_iso(),
            "has_changes": has_changes,
            "sync_token": str(profile.sync_version),
        }


store = StudioStore()

