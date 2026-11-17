"""`agentops` console entry point."""

from __future__ import annotations

import argparse

import uvicorn

from agentops.core.config import get_settings


def main() -> None:
    """Run the API with uvicorn. Installed as the `agentops` command.

    Flags: `--host`, `--port`, and `--reload` for development.
    """
    parser = argparse.ArgumentParser(prog="agentops", description="Run the multi-agent-ops API.")
    # Binds every interface, which a container needs. Use 127.0.0.1 to keep it local-only.
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Reload on code changes.")
    args = parser.parse_args()

    settings = get_settings()
    uvicorn.run(
        "agentops.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # Our own `configure_logging` already routes uvicorn through the JSON formatter; a uvicorn
        # log config here would fight it.
        log_config=None,
        # Per-request access lines duplicate the structured trace logs, so only emit them when
        # debugging.
        access_log=settings.log_level.upper() == "DEBUG",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
