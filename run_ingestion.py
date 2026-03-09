"""Local launcher for MongoDB ingestion."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Run MongoDB ingestion CLI with local source path injection."""
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from nosql_project.mongo_ingestion import (  # pylint: disable=import-outside-toplevel
        main as ingestion_main,
    )

    return ingestion_main()


if __name__ == "__main__":
    raise SystemExit(main())
