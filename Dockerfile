# Multi-stage build: uv resolves the locked environment in the builder, the
# runtime image carries only the venv, the code and the data. Non-root user.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 rwsat
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY app ./app
COPY data ./data
# Precomputed artifacts baked into the image: the container serves in seconds
# instead of retraining. .dockerignore admits only the three product files.
COPY artifacts ./artifacts
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /app/artifacts && chown -R rwsat:rwsat /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER rwsat
EXPOSE 8000 8501
CMD ["/entrypoint.sh"]
