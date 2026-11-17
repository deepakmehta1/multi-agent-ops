"""`agentops` console entry point."""

from __future__ import annotations

import argparse

import uvicorn

from agentops.core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentops", description="Run the multi-agent-ops API.")
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
        log_config=None,
        access_log=settings.log_level.upper() == "DEBUG",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
