"""Persistent research memory backed by ChromaDB + Ollama embeddings.

Stores research findings as vector embeddings so agents can recall
relevant past research when working on new queries.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import chromadb
import requests

# Persist research memory to disk so it survives restarts
MEMORY_DIR = Path(__file__).parent / ".research_memory"

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB persistent client (singleton)."""
    global _client
    if _client is None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(MEMORY_DIR))
    return _client


def _embed(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Ollama's nomic-embed-text model."""
    embeddings = []
    for text in texts:
        resp = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        embeddings.append(resp.json()["embeddings"][0])
    return embeddings


def _get_collection():
    """Get or create the research collection."""
    client = _get_client()
    return client.get_or_create_collection(
        name="research",
        metadata={"hnsw:space": "cosine"},
    )


def save_finding(content: str, source: str, topic: str) -> str:
    """Save a research finding to memory."""
    col = _get_collection()
    doc_id = hashlib.md5(content[:200].encode()).hexdigest()
    embedding = _embed([content[:2000]])[0]

    col.upsert(
        ids=[doc_id],
        documents=[content],
        embeddings=[embedding],
        metadatas=[{
            "source": source,
            "topic": topic,
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        }],
    )
    return f"Saved finding ({len(content)} chars) from {source} under topic '{topic}'"


def recall(query: str, k: int = 5, topic: str | None = None) -> list[dict]:
    """Recall relevant research findings from memory."""
    col = _get_collection()

    if col.count() == 0:
        return []

    query_embedding = _embed([query])[0]
    where = {"topic": topic} if topic else None

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(k, col.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    findings = []
    for i in range(len(results["ids"][0])):
        dist = results["distances"][0][i]
        findings.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "topic": results["metadatas"][0][i].get("topic", ""),
            "timestamp": results["metadatas"][0][i].get("timestamp", ""),
            "relevance": round(1 - dist, 3),
        })
    return findings


def recall_formatted(query: str, k: int = 5, topic: str | None = None) -> str:
    """Recall findings and format as readable text."""
    findings = recall(query, k=k, topic=topic)
    if not findings:
        return "No relevant past research found."

    lines = [f"Found {len(findings)} relevant past findings:\n"]
    for i, f in enumerate(findings, 1):
        lines.append(
            f"[{i}] (relevance: {f['relevance']}, source: {f['source']}, "
            f"date: {f['timestamp']})\n{f['content'][:500]}\n"
        )
    return "\n".join(lines)
