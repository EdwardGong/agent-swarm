from __future__ import annotations

from pydantic import BaseModel, Field


class GuestLoginRequest(BaseModel):
    game_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    player_id: str | None = None


class AuthTokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int


class GuestLoginResponse(BaseModel):
    player_id: str
    game_id: str
    session_id: str
    auth: AuthTokenPair


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class MeResponse(BaseModel):
    player_id: str
    game_id: str
    session_id: str


class SyncRequest(BaseModel):
    last_sync_token: str | None = None
    client_config_version: int | None = None
    client_time: str | None = None


class SyncResponse(BaseModel):
    profile: dict
    wallet: dict
    inventory: dict
    active_events: list
    offers: list
    server_time: str
    has_changes: bool
    sync_token: str

