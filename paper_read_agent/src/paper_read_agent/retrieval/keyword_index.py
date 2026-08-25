"""SQLite FTS5 keyword index with CJK and identifier normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import unicodedata

from paper_read_agent.retrieval.vector_index import VectorRecord


@dataclass(frozen=True, slots=True)
class KeywordHit:
    chunk_id: str
    text: str
    score: float
    paper_id: str
    version_id: str
    content_type: str


class SQLiteKeywordIndex:
    def __init__(self, path: str | Path, *, index_version: str = "keyword-v1") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index_version = index_version
        self._initialize()

    def upsert(self, records: list[VectorRecord]) -> None:
        with self._connect() as connection:
            for record in records:
                chunk = record.chunk
                connection.execute("DELETE FROM keyword_fts WHERE chunk_id=?", (chunk.chunk_id,))
                connection.execute("DELETE FROM keyword_documents WHERE chunk_id=?", (chunk.chunk_id,))
                connection.execute(
                    "INSERT INTO keyword_documents VALUES (?,?,?,?,?,?)",
                    (chunk.chunk_id, chunk.text, record.paper_id, chunk.version_id,
                     chunk.content_type, self.index_version),
                )
                connection.execute(
                    "INSERT INTO keyword_fts(chunk_id,search_text) VALUES (?,?)",
                    (chunk.chunk_id, self.normalize(chunk.text)),
                )

    def query(
        self, query: str, *, limit: int = 50,
        paper_ids: list[str] | None = None,
        version_ids: list[str] | None = None,
        include_references: bool = False,
    ) -> tuple[KeywordHit, ...]:
        if limit <= 0:
            raise ValueError("Keyword query limit must be positive")
        terms = self._query_terms(query)
        if not terms:
            return ()
        match = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        clauses = ["keyword_fts MATCH ?"]
        values: list[object] = [match]
        if paper_ids:
            clauses.append("d.paper_id IN (" + ",".join("?" for _ in paper_ids) + ")")
            values.extend(paper_ids)
        if version_ids:
            clauses.append("d.version_id IN (" + ",".join("?" for _ in version_ids) + ")")
            values.extend(version_ids)
        if not include_references:
            clauses.append("d.content_type != 'reference'")
        values.append(limit)
        sql = f"""
            SELECT d.*, bm25(keyword_fts) AS rank
            FROM keyword_fts JOIN keyword_documents d USING(chunk_id)
            WHERE {' AND '.join(clauses)}
            ORDER BY rank, d.chunk_id LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(sql, values).fetchall()
        return tuple(
            KeywordHit(row["chunk_id"], row["text"], -float(row["rank"]),
                       row["paper_id"], row["version_id"], row["content_type"])
            for row in rows
        )

    def delete(self, *, paper_id: str | None = None, version_id: str | None = None) -> None:
        if (paper_id is None) == (version_id is None):
            raise ValueError("Specify exactly one paper_id or version_id")
        column, value = ("paper_id", paper_id) if paper_id is not None else ("version_id", version_id)
        with self._connect() as connection:
            ids = [row[0] for row in connection.execute(
                f"SELECT chunk_id FROM keyword_documents WHERE {column}=?", (value,)
            )]
            connection.executemany("DELETE FROM keyword_fts WHERE chunk_id=?", ((item,) for item in ids))
            connection.execute(f"DELETE FROM keyword_documents WHERE {column}=?", (value,))

    def rebuild(self, records: list[VectorRecord]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM keyword_fts")
            connection.execute("DELETE FROM keyword_documents")
            connection.execute("UPDATE keyword_metadata SET value=? WHERE key='index_version'",
                               (self.index_version,))
        self.upsert(records)

    @classmethod
    def normalize(cls, text: str) -> str:
        value = unicodedata.normalize("NFKC", text).casefold()
        identifiers = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)+|[a-z]+\d+[a-z0-9-]*", value)
        cjk_runs = re.findall(r"[\u3400-\u9fff]+", value)
        cjk_terms = [part for run in cjk_runs for size in (1, 2) for part in cls._ngrams(run, size)]
        base = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
        aliases = [item.replace("-", " ").replace("_", " ").replace(".", " ") for item in identifiers]
        return " ".join([base, *identifiers, *aliases, *cjk_terms])

    @classmethod
    def _query_terms(cls, query: str) -> list[str]:
        normalized = cls.normalize(query)
        terms = re.findall(r"[\w.-]+", normalized, flags=re.UNICODE)
        return list(dict.fromkeys(term for term in terms if term))

    @staticmethod
    def _ngrams(value: str, size: int) -> list[str]:
        return [value[index:index + size] for index in range(max(len(value) - size + 1, 0))]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS keyword_documents (
                    chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL, paper_id TEXT NOT NULL,
                    version_id TEXT NOT NULL, content_type TEXT NOT NULL, index_version TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_keyword_paper ON keyword_documents(paper_id);
                CREATE INDEX IF NOT EXISTS idx_keyword_version ON keyword_documents(version_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(
                    chunk_id UNINDEXED, search_text, tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TABLE IF NOT EXISTS keyword_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )
            row = connection.execute(
                "SELECT value FROM keyword_metadata WHERE key='index_version'"
            ).fetchone()
            if row is None:
                connection.execute("INSERT INTO keyword_metadata VALUES ('index_version',?)",
                                   (self.index_version,))
            elif row[0] != self.index_version:
                raise RuntimeError(
                    f"Keyword index version mismatch: stored={row[0]}, configured={self.index_version}"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection
