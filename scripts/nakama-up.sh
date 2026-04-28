#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.nakama"
EXAMPLE_ENV_FILE="${ROOT_DIR}/.env.nakama.example"

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${EXAMPLE_ENV_FILE}" "${ENV_FILE}"
  echo "Created ${ENV_FILE} from ${EXAMPLE_ENV_FILE}. Update secrets before shared/deployed use."
fi

docker compose --env-file "${ENV_FILE}" -f "${ROOT_DIR}/docker-compose.nakama.yml" up -d

echo "Nakama stack started."
echo "HTTP API:    http://localhost:7350"
echo "gRPC:        localhost:7349"
echo "Console:     http://localhost:7351"

