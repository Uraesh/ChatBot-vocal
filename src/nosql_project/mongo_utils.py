"""Shared MongoDB utilities used by runtime and ingestion modules."""

from __future__ import annotations

from typing import Any


def create_mongo_collection(
    *,
    mongo_uri: str,
    mongo_database: str,
    mongo_collection: str,
    server_selection_timeout_ms: int,
    purpose: str,
) -> tuple[Any, Any]:
    """Create and return a `(client, collection)` MongoDB pair."""
    try:
        import pymongo  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"pymongo is required for {purpose}. "
            "Install with `python -m pip install pymongo`."
        ) from exc

    mongo_client_factory: Any = getattr(pymongo, "MongoClient")
    client: Any = mongo_client_factory(
        mongo_uri,
        serverSelectionTimeoutMS=server_selection_timeout_ms,
    )
    collection: Any = client[mongo_database][mongo_collection]
    return client, collection


def create_ingestion_collection(
    *,
    mongo_uri: str,
    mongo_database: str,
    mongo_collection: str,
    server_selection_timeout_ms: int,
) -> tuple[Any, Any]:
    """Create Mongo collection configured for ingestion use cases."""
    return create_mongo_collection(
        mongo_uri=mongo_uri,
        mongo_database=mongo_database,
        mongo_collection=mongo_collection,
        server_selection_timeout_ms=server_selection_timeout_ms,
        purpose="MongoDB ingestion",
    )


def create_interaction_collection(
    *,
    mongo_uri: str,
    mongo_database: str,
    mongo_collection: str,
    server_selection_timeout_ms: int,
) -> tuple[Any, Any]:
    """Create Mongo collection configured for interaction persistence."""
    return create_mongo_collection(
        mongo_uri=mongo_uri,
        mongo_database=mongo_database,
        mongo_collection=mongo_collection,
        server_selection_timeout_ms=server_selection_timeout_ms,
        purpose="Mongo interaction persistence",
    )
