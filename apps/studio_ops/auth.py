"""Token utilities for Studio Ops API.

This intentionally uses only the Python stdlib so it works in minimal environments.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any


SECRET = os.getenv("STUDIO_OPS_JWT_SECRET", "change-me-in-prod")
ACCESS_TTL_SECONDS = 60 * 60
REFRESH_TTL_SECONDS = 60 * 60 * 24 * 30


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(token_part: str) -> bytes:
    padding = "=" * (-len(token_part) % 4)
    return base64.urlsafe_b64decode(token_part + padding)


def _sign(message: bytes) -> str:
    digest = hmac.new(SECRET.encode("utf-8"), message, hashlib.sha256).digest()
    return _b64url_encode(digest)


def create_token(payload: dict[str, Any], ttl_seconds: int) -> str:
    payload_copy = dict(payload)
    payload_copy["exp"] = int(time.time()) + ttl_seconds
    body = _b64url_encode(
        json.dumps(payload_copy, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    sig = _sign(body.encode("utf-8"))
    return f"{body}.{sig}"


def verify_token(token: str) -> dict[str, Any]:
    try:
        body, sig = token.split(".", 1)
    except ValueError as exc:
        raise ValueError("Malformed token") from exc

    expected_sig = _sign(body.encode("utf-8"))
    if not hmac.compare_digest(sig, expected_sig):
        raise ValueError("Invalid signature")

    payload = json.loads(_b64url_decode(body).decode("utf-8"))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("Token expired")
    return payload


def issue_auth_pair(player_id: str, game_id: str, session_id: str) -> dict[str, str]:
    base_claims = {
        "sub": player_id,
        "game_id": game_id,
        "sid": session_id,
    }
    access = create_token({**base_claims, "typ": "access"}, ACCESS_TTL_SECONDS)
    refresh = create_token({**base_claims, "typ": "refresh"}, REFRESH_TTL_SECONDS)
    return {"access_token": access, "refresh_token": refresh}

