"""Streaming helpers for large OpenSubtitles-style text files."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

_MULTI_SPACE_REGEX = re.compile(r"\s+")


def normalize_line(raw_line: str) -> str:
    """Normalize a raw text line into a compact sentence."""
    cleaned = raw_line.strip()
    if not cleaned:
        return ""
    cleaned = _MULTI_SPACE_REGEX.sub(" ", cleaned)
    return cleaned


def iter_clean_lines(
    file_path: Path,
    min_length: int = 2,
    max_length: int = 220,
) -> Iterator[str]:
    """Yield cleaned lines from a large text file in streaming mode."""
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            sentence = normalize_line(raw_line)
            if min_length <= len(sentence) <= max_length:
                yield sentence


def make_pairs(lines: Iterable[str]) -> Generator[tuple[str, str], None, None]:
    """Convert sequential lines into dialogue pairs."""
    previous: str | None = None
    for line in lines:
        if previous is None:
            previous = line
            continue
        if previous != line:
            yield previous, line
        previous = None


def batched(items: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield items in fixed-size lists."""
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive.")
    bucket: list[Any] = []
    for item in items:
        bucket.append(item)
        if len(bucket) == batch_size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def iter_dialogue_documents(
    file_path: Path,
    source: str = "OpenSubtitles",
    split: str = "train",
) -> Iterator[dict[str, Any]]:
    """Yield MongoDB-ready dialogue documents from a large text file."""
    timestamp = datetime.now(timezone.utc)
    cleaned_lines = iter_clean_lines(file_path=file_path)
    for turn_id, (input_text, response_text) in enumerate(make_pairs(cleaned_lines), start=1):
        yield {
            "conversation_id": str(uuid4()),
            "turn_id": turn_id,
            "input": input_text,
            "response": response_text,
            "lang": "fr",
            "source": source,
            "split": split,
            "quality_score": 0.7,
            "created_at": timestamp,
        }
