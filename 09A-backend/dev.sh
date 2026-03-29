#!/usr/bin/env bash
set -euo pipefail

echo "Stopping containers..."
docker compose down -v

echo "Starting containers..."
docker compose up -d

echo "Waiting for Postgres..."
ready=false
for i in $(seq 1 90); do
    if docker compose exec -T db pg_isready -U appuser -d appdb >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done

if [ "$ready" != true ]; then
    echo "Postgres did not become ready." >&2
    exit 1
fi

echo "Running main.py..."
python ../main.py

echo "Opening psql..."
docker compose exec db psql -U appuser -d appdb

