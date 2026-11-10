# Multi-stage: Vite dashboard → FastAPI serves /api and the built SPA (same origin).
FROM node:22-slim AS dashboard
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
ARG VITE_AG_STUDIO_LICENSE_KEY
ENV VITE_AG_STUDIO_LICENSE_KEY=$VITE_AG_STUDIO_LICENSE_KEY
# Leave VITE_API_BASE_URL unset so the SPA calls same-origin /api (Fly Path A / Docker).
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
COPY --from=dashboard /frontend/dist /app/frontend_dist
# Must match fly.toml [http_service] internal_port and [env] PORT.
# Default DB is ephemeral /tmp (Fly free trial has no volumes). Override TRUSTPAY_DB when using a volume.
ENV TRUSTPAY_DB=/tmp/trustpay.db \
    PORT=8080 \
    PYTHONUNBUFFERED=1
RUN chmod +x /app/start.sh \
 && sed -i 's/\r$//' /app/start.sh
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8080'), timeout=4)"
# Use start.sh so PORT is always expanded by a real shell (avoids stale ${PORT:-8000} images).
CMD ["/app/start.sh"]
