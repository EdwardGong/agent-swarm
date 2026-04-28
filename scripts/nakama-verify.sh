#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.nakama"
EXAMPLE_ENV_FILE="${ROOT_DIR}/.env.nakama.example"

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${EXAMPLE_ENV_FILE}" "${ENV_FILE}"
  echo "Created ${ENV_FILE} from ${EXAMPLE_ENV_FILE}."
fi

docker compose --env-file "${ENV_FILE}" -f "${ROOT_DIR}/docker-compose.nakama.yml" config >/dev/null
echo "docker compose config validation passed."

NAKAMA_HTTP_PORT="$(grep '^NAKAMA_HTTP_PORT=' "${ENV_FILE}" | cut -d'=' -f2)"
if [[ -z "${NAKAMA_HTTP_PORT}" ]]; then
  NAKAMA_HTTP_PORT=7350
fi

curl --fail --silent "http://localhost:${NAKAMA_HTTP_PORT}/" >/dev/null
echo "Nakama HTTP endpoint reachable on port ${NAKAMA_HTTP_PORT}."

