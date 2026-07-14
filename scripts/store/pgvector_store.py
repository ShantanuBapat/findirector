"""PgVectorStore: a VectorStore backed by Postgres + pgvector."""

import json

import psycopg
from pgvector.psycopg import register_vector

from scripts.store.base import VectorStore
from scripts.store.db import get_dsn

# Columns we insert (everything except the auto-generated id).
_COLUMNS = [
    "ticker", "fiscal_year", "filing_type", "accession_number",
    "section", "content_type", "chunk_index", "n_tokens", "text", "embedding",
]


class PgVectorStore(VectorStore):
    """Store embedded chunks in Postgres; search by cosine distance + filters."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or get_dsn()

    def _connect(self):
        """Open a connection with the pgvector type adapter registered."""
        conn = psycopg.connect(self.dsn)
        register_vector(conn)  # teaches psycopg the vector type (list <-> vector)
        return conn

    def add(self, chunks: list[dict], batch_size: int = 500) -> int:
        """Bulk-insert embedded chunks. Returns the count inserted."""
        placeholders = ", ".join(["%s"] * len(_COLUMNS))
        insert_sql = (
            f"INSERT INTO chunks ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
        )

        inserted = 0
        with self._connect() as conn, conn.cursor() as cur:
            batch = []
            for chunk in chunks:
                batch.append(tuple(chunk.get(col) for col in _COLUMNS))
                if len(batch) >= batch_size:
                    cur.executemany(insert_sql, batch)
                    inserted += len(batch)
                    batch = []
            if batch:
                cur.executemany(insert_sql, batch)
                inserted += len(batch)
            conn.commit()
        return inserted

    def existing_accessions(self) -> set[str]:
        """Return the set of accession numbers already present in the store.

        The delta key for embedding: filings whose accession is already here
        have been embedded and loaded, so the embedding orchestration skips
        them (mirrors the delta-chunking pattern).
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT accession_number FROM chunks")
            return {row[0] for row in cur.fetchall()}

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Return the k chunks most similar to query_embedding, after filtering.

        Metadata filters are applied first (WHERE), then remaining rows are
        ranked by cosine distance (<=>). Empty result on an out-of-corpus filter
        is the signal the corpus-boundary guard uses to decline.
        """
        where_sql = ""
        params: list = []
        if filters:
            clauses = []
            for key, value in filters.items():
                clauses.append(f"{key} = %s")
                params.append(value)
            where_sql = "WHERE " + " AND ".join(clauses)

        sql = f"""
            SELECT ticker, fiscal_year, filing_type, accession_number,
                   section, content_type, chunk_index, n_tokens, text,
                   1 - (embedding <=> %s) AS score
            FROM chunks
            {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        params = [query_embedding] + params + [query_embedding, k]

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
