"""Local launcher for the FastAPI service."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    """Start the API server with local source path injection."""
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    debug_value = os.getenv("DEBUG", "").strip().lower()
    is_debug = debug_value in {"1", "true", "yes", "on"}
    logging.basicConfig(
        level=logging.DEBUG if is_debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    host = os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0"
    port_raw = os.getenv("PORT", "8000").strip()
    try:
        port = int(port_raw)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid PORT value '%s'. Using 8000.", port_raw)
        port = 8000
    uvicorn.run("nosql_project.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
