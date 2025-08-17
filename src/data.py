import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class PolicyRecord:
    id: int
    policy_text: str
    site_domain: Optional[str]
    year: Optional[int]
    phase: Optional[str]


def load_entities(entities_path: str) -> List[str]:
    with open(entities_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("entities.json must contain a JSON array of strings")
    return [str(x).strip() for x in items if str(x).strip()]


def load_policy_texts(
    sqlite_path: str,
    limit: Optional[int] = 5000,
    min_length: int = 200,
) -> List[PolicyRecord]:
    conn = sqlite3.connect(sqlite_path)
    try:
        query = (
            """
            SELECT ps.id as snapshot_id, pt.policy_text as policy_text, s.domain as domain, ps.year as year, ps.phase as phase
            FROM policy_snapshots ps
            LEFT JOIN policy_texts pt ON ps.policy_text_id = pt.id
            LEFT JOIN sites s ON ps.site_id = s.id
            WHERE pt.policy_text IS NOT NULL AND length(pt.policy_text) >= ?
            ORDER BY ps.id DESC
            """
        )
        if limit is not None:
            query += " LIMIT ?"
            params: Tuple[int, int] = (min_length, int(limit))
        else:
            params = (min_length,)

        df: pd.DataFrame = pd.read_sql_query(query, conn, params=params)
        records: List[PolicyRecord] = []
        for _, row in df.iterrows():
            records.append(
                PolicyRecord(
                    id=int(row["snapshot_id"]),
                    policy_text=str(row["policy_text"]),
                    site_domain=str(row["domain"]) if pd.notna(row["domain"]) else None,
                    year=int(row["year"]) if pd.notna(row["year"]) else None,
                    phase=str(row["phase"]) if pd.notna(row["phase"]) else None,
                )
            )
        return records
    finally:
        conn.close()


def chunk_text(text: str, max_tokens: int = 800) -> List[str]:
    # Backwards-compatible wrapper returning only chunk strings
    return [c for c, _ in chunk_text_with_offsets(text, max_tokens=max_tokens)]


def chunk_text_with_offsets(text: str, max_tokens: int = 800) -> List[tuple[str, int]]:
    # Simple sentence/paragraph chunker: split by double newlines, fallback to sentences
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[tuple[str, int]] = []
    search_start = 0
    def append_with_offset(chunk_str: str):
        nonlocal search_start
        idx = text.find(chunk_str, search_start)
        if idx == -1:
            # Fallback: search from beginning; may duplicate but avoids crash
            idx = text.find(chunk_str)
        if idx != -1:
            search_start = idx + len(chunk_str)
        else:
            idx = 0
        chunks.append((chunk_str, idx))

    for p in paragraphs:
        if len(p) <= max_tokens * 4:
            append_with_offset(p)
        else:
            sentences = [s.strip() for s in p.replace("\n", " ").split(". ") if s.strip()]
            current: List[str] = []
            current_len = 0
            for s in sentences:
                s_len = len(s)
                if current_len + s_len > max_tokens * 4 and current:
                    chunk_str = ". ".join(current)
                    append_with_offset(chunk_str)
                    current = [s]
                    current_len = s_len
                else:
                    current.append(s)
                    current_len += s_len
            if current:
                chunk_str = ". ".join(current)
                append_with_offset(chunk_str)
    if not chunks:
        chunks = [(text.strip(), 0)]
    return chunks

