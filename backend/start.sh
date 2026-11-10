#!/bin/sh
# Fly [http_service] internal_port is 8080. Default here must match fly.toml / Dockerfile ENV.
# Render (and similar) may inject PORT; Fly sets PORT via fly.toml [env].
set -eu
port="${PORT:-8080}"
db="${TRUSTPAY_DB:-/tmp/trustpay.db}"
db_dir=$(dirname "$db")
mkdir -p "$db_dir"
export TRUSTPAY_DB="$db"
echo "trustpay: starting uvicorn on 0.0.0.0:${port} (db=${db})"
exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port "${port}"
