# syntax=docker/dockerfile:1

# Stage 1 (builder): install the package and its dependencies into a throwaway virtualenv, so pip
# and any build tooling never reach the final image.
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
# README.md must be copied too: pyproject.toml declares it as the package readme, and the build
# fails without it. Copying the metadata before the source lets Docker cache this layer.
COPY pyproject.toml README.md ./
COPY src ./src
# A real (non-editable) install: the wheel includes the console under ui/ and the seed data under
# systems/data/, so the runtime image needs no source tree.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir .

# Stage 2 (runtime): a clean base plus only the finished virtualenv.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 agentops
COPY --from=builder /opt/venv /opt/venv

USER agentops
# Nothing is read from this directory. Settings would pick up a ./.env here, but none is baked into
# the image (see .dockerignore): configuration arrives as environment variables at run time.
WORKDIR /home/agentops
EXPOSE 8000

# Probes /healthz with the standard library, so no curl has to be installed in the image. Docker
# marks the container unhealthy after three consecutive failures.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

# The app factory reconfigures uvicorn's loggers onto the JSON formatter at import time.
CMD ["uvicorn", "agentops.main:app", "--host", "0.0.0.0", "--port", "8000"]
