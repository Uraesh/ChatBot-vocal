"""Entry point for running the API with `python -m nosql_project`."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Start uvicorn server."""
    host = os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0"
    port_raw = os.getenv("PORT", "8000").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 8000
    uvicorn.run("nosql_project.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
