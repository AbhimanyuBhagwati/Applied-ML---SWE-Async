"""Every query and the answer it got, kept in SQLite.

SQLite rather than a list in memory: a history that empties on restart isn't
one, and it would differ per worker under more than one process. sqlite3 is in
the standard library, so it costs no dependency.

The driver is synchronous, so every call hops to a worker thread. aiosqlite
would avoid that; for writes measured in milliseconds the thread hop is the
simpler trade.
"""

import asyncio
import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    query         TEXT NOT NULL,
    response      TEXT NOT NULL,
    finish_reason TEXT,
    tool_plan     TEXT,
    tool_calls    TEXT NOT NULL DEFAULT '[]',
    citations     TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL
);
"""

# Stored as JSON in one column each. A history that kept only the query and the
# answer would throw away the grounding task 2 exists to produce, and searching
# inside these isn't something the endpoint offers, so columns would buy
# nothing.
JSON_COLUMNS = ("tool_calls", "citations")


class HistoryStore:
    """One connection, shared across worker threads, guarded by a lock.

    sqlite3 allows sharing a connection with check_same_thread=False but
    doesn't make it safe on its own, hence the lock.
    """

    def __init__(self, database_path: str) -> None:
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            database_path, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    async def record(self, answer: Any) -> None:
        """Store what the caller was given. Never raises.

        The answer has already been produced and paid for. Losing a history row
        is worth less than the response, so a storage failure is logged and
        swallowed rather than turned into a 500 on a request that worked.
        """

        def write() -> None:
            with self._lock:
                self._connection.execute(
                    "INSERT INTO turns (query, response, finish_reason, tool_plan,"
                    " tool_calls, citations, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        answer.query,
                        answer.response,
                        answer.finish_reason,
                        answer.tool_plan,
                        json.dumps([call.model_dump() for call in answer.tool_calls]),
                        json.dumps([cite.model_dump() for cite in answer.citations]),
                        datetime.now(UTC).isoformat(timespec="seconds"),
                    ),
                )

        try:
            await asyncio.to_thread(write)
        except Exception as exc:  # noqa: BLE001 - see below
            # Deliberately everything, not just sqlite3.Error. The point isn't
            # which layer failed, it's that the answer is already produced and
            # paid for: nothing here is worth turning a working request into a
            # 500. Serialising the grounding can fail too, not only the write.
            log.warning("Could not record the turn: %s: %s", type(exc).__name__, exc)

    async def page(self, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        """A slice of the history, newest first, with the total alongside.

        The total is what tells a caller the slice isn't everything; without it
        a short page and the end of the history look identical.
        """

        def read() -> tuple[list[dict[str, Any]], int]:
            with self._lock:
                total = self._connection.execute(
                    "SELECT COUNT(*) AS n FROM turns"
                ).fetchone()["n"]
                rows = self._connection.execute(
                    "SELECT * FROM turns ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [_entry(row) for row in rows], int(total)

        return await asyncio.to_thread(read)


def _entry(row: sqlite3.Row) -> dict[str, Any]:
    entry = dict(row)
    for column in JSON_COLUMNS:
        entry[column] = json.loads(entry[column])
    return entry
