"""Tests for Mongo ingestion orchestration logic."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import pytest

from nosql_project.mongo_ingestion import ingest_dialogues


class FakeRepository:
    """Simple in-memory repository used for ingestion tests."""

    def __init__(self) -> None:
        self.indexes_created = False
        self.batches: list[list[dict[str, Any]]] = []

    def ensure_indexes(self) -> None:
        """Mark index creation as executed."""
        self.indexes_created = True

    def insert_many(self, documents: Sequence[dict[str, Any]]) -> int:
        """Store inserted documents in memory."""
        snapshot = [dict(doc) for doc in documents]
        self.batches.append(snapshot)
        return len(documents)


def _write_sample(path: Path) -> None:
    path.write_text(
        "Bonjour\nSalut\nCa va ?\nOui\nMerci\nAvec plaisir\n",
        encoding="utf-8",
    )


def test_ingest_dialogues_batches_and_report(tmp_path: Path) -> None:
    """Ingestion should batch and report counters correctly."""
    source_file = tmp_path / "sample.txt"
    _write_sample(source_file)
    repository = FakeRepository()

    report = ingest_dialogues(
        file_path=source_file,
        repository=repository,
        batch_size=2,
    )

    assert repository.indexes_created is True
    assert report.attempted_documents == 3
    assert report.inserted_documents == 3
    assert report.batch_count == 2
    assert len(repository.batches) == 2


def test_ingest_dialogues_with_limit(tmp_path: Path) -> None:
    """Ingestion limit should cap processed document count."""
    source_file = tmp_path / "sample.txt"
    _write_sample(source_file)
    repository = FakeRepository()

    report = ingest_dialogues(
        file_path=source_file,
        repository=repository,
        batch_size=10,
        limit=2,
    )

    assert report.attempted_documents == 2
    assert report.inserted_documents == 2
    assert report.batch_count == 1


def test_ingest_dialogues_invalid_batch_size(tmp_path: Path) -> None:
    """batch_size must stay strictly positive."""
    source_file = tmp_path / "sample.txt"
    _write_sample(source_file)
    repository = FakeRepository()

    with pytest.raises(ValueError, match="batch_size"):
        ingest_dialogues(
            file_path=source_file,
            repository=repository,
            batch_size=0,
        )


def test_ingest_dialogues_missing_file(tmp_path: Path) -> None:
    """Missing source file should raise FileNotFoundError."""
    missing_file = tmp_path / "missing.txt"
    repository = FakeRepository()

    with pytest.raises(FileNotFoundError):
        ingest_dialogues(
            file_path=missing_file,
            repository=repository,
        )
