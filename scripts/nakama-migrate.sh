#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.nakama"
EXAMPLE_ENV_FILE="${ROOT_DIR}/.env.nakama.example"

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${EXAMPLE_ENV_FILE}" "${ENV_FILE}"
  echo "Created ${ENV_FILE} from ${EXAMPLE_ENV_FILE}."
fi

set -a
source "${ENV_FILE}"
set +a

docker compose --env-file "${ENV_FILE}" -f "${ROOT_DIR}/docker-compose.nakama.yml" run --rm nakama \
  /nakama/nakama migrate up \
  --database.address "${NAKAMA_DB_USER}:${NAKAMA_DB_PASSWORD}@postgres:5432/${NAKAMA_DB_NAME}?sslmode=disable"

echo "Nakama migration completed."

