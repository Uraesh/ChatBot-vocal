"""Tests for ingestion helpers."""

from __future__ import annotations

from pathlib import Path

from nosql_project.ingestion import iter_clean_lines, iter_dialogue_documents, make_pairs


def test_make_pairs() -> None:
    """Consecutive lines should form dialogue pairs."""
    lines = ["Bonjour", "Salut", "Ca va ?", "Oui"]
    pairs = list(make_pairs(lines))
    assert pairs == [("Bonjour", "Salut"), ("Ca va ?", "Oui")]


def test_iter_dialogue_documents(tmp_path: Path) -> None:
    """Document stream should produce MongoDB-ready keys."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Bonjour\nSalut\nMerci\nAvec plaisir\n", encoding="utf-8")

    documents = list(iter_dialogue_documents(sample_file))
    assert len(documents) == 2
    assert {"input", "response", "conversation_id", "turn_id", "lang"} <= documents[0].keys()


def test_iter_clean_lines_filters_lengths(tmp_path: Path) -> None:
    """Short lines should be filtered out by default."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("A\nBonjour\n  Merci   bien \n", encoding="utf-8")
    cleaned = list(iter_clean_lines(sample_file))
    assert cleaned == ["Bonjour", "Merci bien"]
