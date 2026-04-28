from __future__ import annotations

from fastapi import Header, HTTPException, status

from apps.studio_ops.auth import verify_token
from apps.studio_ops.store import store


def require_access_token(authorization: str = Header(default="")) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    token = authorization.removeprefix("Bearer ").strip()
    try:
        claims = verify_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    if claims.get("typ") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    sid = claims.get("sid", "")
    game_id = claims.get("game_id", "")
    player_id = claims.get("sub", "")
    if not store.validate_session(session_id=sid, game_id=game_id, player_id=player_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalid",
        )
    return claims

