import sqlite3
from pathlib import Path

import pytest

from app.db import Database


@pytest.mark.asyncio
async def test_db_init_creates_tables(tmp_path: Path):
    db_path = tmp_path / "schema.db"
    db = Database(str(db_path))
    await db.init()
    rows = await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in rows}
    expected = {
        "conversations",
        "runs",
        "messages",
        "events",
        "prompt_state",
        "conversation_state",
    }
    assert expected.issubset(tables)
    prompt_row = await db.fetchone("SELECT COUNT(*) as cnt FROM prompt_state")
    convo_row = await db.fetchone("SELECT COUNT(*) as cnt FROM conversation_state")
    assert prompt_row["cnt"] == 1
    assert convo_row["cnt"] == 1


@pytest.mark.asyncio
async def test_db_migration_backfills_conversation_id(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE runs(
            run_id TEXT PRIMARY KEY,
            created_at TEXT,
            user_question TEXT,
            reasoning_mode TEXT,
            status TEXT
        );
        CREATE TABLE messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO runs(run_id, created_at, user_question, reasoning_mode, status) VALUES (?,?,?,?,?)",
        ("run-1", "2024-01-01T00:00:00Z", "hello", "auto", "completed"),
    )
    conn.execute(
        "INSERT INTO messages(run_id, role, content, created_at) VALUES (?,?,?,?)",
        ("run-1", "user", "hello", "2024-01-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    db = Database(str(db_path))
    await db.init()

    run_row = await db.fetchone("SELECT conversation_id FROM runs WHERE run_id=?", ("run-1",))
    msg_row = await db.fetchone("SELECT conversation_id FROM messages WHERE run_id=?", ("run-1",))
    assert run_row["conversation_id"]
    assert msg_row["conversation_id"]
