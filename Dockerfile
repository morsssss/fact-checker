FROM python:3.12-slim

# Grab the uv binary from Astral's official image rather than pip-installing it.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first so this layer is cached unless these files change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# App code, plus only the assets it actually reads at runtime - the source of
# truth for facts. (Test audio/video/transcript are dev-only, not needed here.)
COPY server/ server/
COPY web/ web/
COPY assets/facts.txt assets/facts.txt

# Most hosts (Railway, Render, Fly, Cloud Run) inject PORT themselves; this is
# just a sane default for a plain `docker run` without one.
ENV PORT=8001
EXPOSE 8001

CMD ["uv", "run", "server/web_server.py"]
