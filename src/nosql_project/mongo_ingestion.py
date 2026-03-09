"""MongoDB ingestion pipeline for OpenSubtitles dialogue documents."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import itertools
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from .config import Settings
from .ingestion import batched, iter_dialogue_documents
from .mongo_utils import create_ingestion_collection

LOGGER = logging.getLogger(__name__)


class DialogueRepository(Protocol):
    """Repository contract for dialogue document persistence."""

    def ensure_indexes(self) -> None:
        """Ensure indexes required by the ingestion flow."""
        raise NotImplementedError

    def insert_many(self, documents: Sequence[dict[str, Any]]) -> int:
        """Insert many dialogue documents and return inserted count."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Summary metrics returned by ingestion runs."""

    attempted_documents: int
    inserted_documents: int
    batch_count: int
    duration_seconds: float


class MongoDialogueRepository:
    """MongoDB-backed repository implementation."""

    def __init__(
        self,
        mongo_uri: str,
        mongo_database: str,
        mongo_collection: str,
        server_selection_timeout_ms: int = 5000,
    ) -> None:
        self._client, self._collection = create_ingestion_collection(
            mongo_uri=mongo_uri,
            mongo_database=mongo_database,
            mongo_collection=mongo_collection,
            server_selection_timeout_ms=server_selection_timeout_ms,
        )

    def ping(self) -> None:
        """Verify that MongoDB server is reachable."""
        self._client.admin.command("ping")

    def ensure_indexes(self) -> None:
        """Create useful indexes for dialogue lookups and sorting."""
        self._collection.create_index(
            [("conversation_id", 1), ("turn_id", 1)],
            unique=True,
        )
        self._collection.create_index("created_at")

    def insert_many(self, documents: Sequence[dict[str, Any]]) -> int:
        """Insert a document batch and return inserted count."""
        if not documents:
            return 0
        result = self._collection.insert_many(list(documents), ordered=False)
        return len(result.inserted_ids)

    def close(self) -> None:
        """Close MongoDB connection."""
        self._client.close()


def _limit_documents(
    documents: Iterator[dict[str, Any]],
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    if limit is None:
        return documents
    if limit <= 0:
        raise ValueError("limit must be strictly positive when provided.")
    return itertools.islice(documents, limit)


def ingest_dialogues(
    file_path: Path,
    repository: DialogueRepository,
    *,
    source: str = "OpenSubtitles",
    split: str = "train",
    batch_size: int = 1000,
    limit: int | None = None,
    log_every_batches: int = 10,
) -> IngestionReport:
    """Stream dialogue pairs from text file and insert batches into MongoDB."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive.")
    if log_every_batches <= 0:
        raise ValueError("log_every_batches must be strictly positive.")

    started_at = perf_counter()
    repository.ensure_indexes()

    base_stream = iter_dialogue_documents(
        file_path=file_path,
        source=source,
        split=split,
    )
    document_stream = _limit_documents(base_stream, limit=limit)

    attempted_documents = 0
    inserted_documents = 0
    batch_count = 0
    for batch_count, batch in enumerate(batched(document_stream, batch_size), start=1):
        attempted_documents += len(batch)
        inserted_documents += repository.insert_many(batch)
        if batch_count % log_every_batches == 0:
            LOGGER.info(
                "Ingestion progress: batches=%s attempted=%s inserted=%s",
                batch_count,
                attempted_documents,
                inserted_documents,
            )

    duration_seconds = perf_counter() - started_at
    return IngestionReport(
        attempted_documents=attempted_documents,
        inserted_documents=inserted_documents,
        batch_count=batch_count,
        duration_seconds=duration_seconds,
    )


def ingest_from_settings(settings: Settings, limit: int | None = None) -> IngestionReport:
    """Run ingestion with runtime settings."""
    repository = MongoDialogueRepository(
        mongo_uri=settings.mongo_uri,
        mongo_database=settings.mongo_database,
        mongo_collection=settings.mongo_collection,
    )
    try:
        repository.ping()
        return ingest_dialogues(
            file_path=Path(settings.subtitles_file_path),
            repository=repository,
            source="OpenSubtitles",
            split="train",
            batch_size=settings.mongo_batch_size,
            limit=limit,
        )
    finally:
        repository.close()


def _build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest OpenSubtitles dialogue pairs into MongoDB.",
    )
    parser.add_argument("--file", default=settings.subtitles_file_path)
    parser.add_argument("--uri", default=settings.mongo_uri)
    parser.add_argument("--database", default=settings.mongo_database)
    parser.add_argument("--collection", default=settings.mongo_collection)
    parser.add_argument("--batch-size", type=int, default=settings.mongo_batch_size)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", default="OpenSubtitles")
    parser.add_argument("--split", default="train")
    return parser


def main() -> int:
    """CLI entrypoint for Mongo ingestion."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    settings = Settings.from_env()
    parser = _build_parser(settings)
    args = parser.parse_args()

    repository = MongoDialogueRepository(
        mongo_uri=args.uri,
        mongo_database=args.database,
        mongo_collection=args.collection,
    )
    try:
        repository.ping()
        report = ingest_dialogues(
            file_path=Path(args.file),
            repository=repository,
            source=args.source,
            split=args.split,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Mongo ingestion failed.")
        return 1
    finally:
        repository.close()

    LOGGER.info(
        "Mongo ingestion completed attempted=%s inserted=%s batches=%s duration=%.2fs",
        report.attempted_documents,
        report.inserted_documents,
        report.batch_count,
        report.duration_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
