import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .schemas import Artifact

import aiosqlite


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


class Database:
    def __init__(self, path: str):
        self.path = path

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS conversations(
                    id TEXT PRIMARY KEY,
                    created_at TEXT,
                    updated_at TEXT,
                    title TEXT,
                    model_tier TEXT,
                    reasoning_mode TEXT,
                    manual_level TEXT,
                    deep_mode TEXT,
                    archived INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS runs(
                    run_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    created_at TEXT,
                    user_question TEXT,
                    reasoning_mode TEXT,
                    router_decision_json TEXT,
                    final_answer TEXT,
                    confidence TEXT,
                    status TEXT
                );
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS tasks(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    task_type TEXT,
                    task_payload_json TEXT,
                    status TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS searches(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    lane TEXT,
                    query TEXT,
                    search_depth TEXT,
                    max_results INTEGER,
                    raw_response_json TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS extracts(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    lane TEXT,
                    url TEXT,
                    extract_depth TEXT,
                    raw_response_json TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS sources(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    lane TEXT,
                    url TEXT,
                    title TEXT,
                    publisher TEXT,
                    date_published TEXT,
                    snippet TEXT,
                    extracted_text TEXT
                );
                CREATE TABLE IF NOT EXISTS claims(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    claim TEXT,
                    support_urls_json TEXT,
                    confidence TEXT,
                    notes TEXT
                );
                CREATE TABLE IF NOT EXISTS drafts(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    draft_text TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS verifier_reports(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    verdict TEXT,
                    issues_json TEXT,
                    revised_answer TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    seq INTEGER,
                    event_type TEXT,
                    payload_json TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS configs(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    payload_json TEXT
                );
                CREATE TABLE IF NOT EXISTS step_plans(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    plan_json TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS step_runs(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    step_id INTEGER,
                    status TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    agent_profile TEXT,
                    prompt_text TEXT,
                    output_json TEXT,
                    error_text TEXT
                );
                CREATE TABLE IF NOT EXISTS artifacts(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    step_id INTEGER,
                    key TEXT,
                    artifact_type TEXT,
                    content_text TEXT,
                    content_json TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS uploads(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    filename TEXT,
                    original_name TEXT,
                    mime TEXT,
                    size_bytes INTEGER,
                    storage_path TEXT,
                    status TEXT,
                    summary_text TEXT,
                    summary_json TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS control_actions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    action_type TEXT,
                    payload_json TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS memory_items(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    updated_at TEXT,
                    kind TEXT,
                    title TEXT,
                    content TEXT,
                    tags_json TEXT,
                    pinned_bool INTEGER,
                    relevance_score REAL
                );
                CREATE TABLE IF NOT EXISTS run_memory_links(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    memory_item_id INTEGER,
                    reason TEXT
                );
                CREATE TABLE IF NOT EXISTS conversation_state(
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    reset_at TEXT,
                    default_conversation_id TEXT
                );
                INSERT OR IGNORE INTO conversation_state(id, reset_at) VALUES (1, NULL);
                CREATE TABLE IF NOT EXISTS prompt_state(
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    prompt_text TEXT,
                    run_id TEXT,
                    updated_at TEXT
                );
                INSERT OR IGNORE INTO prompt_state(id, prompt_text, run_id, updated_at) VALUES (1, NULL, NULL, NULL);
                """
            )
            async def column_exists(table: str, column: str) -> bool:
                cursor = await db.execute(f"PRAGMA table_info({table})")
                rows = await cursor.fetchall()
                await cursor.close()
                return any(row[1] == column for row in rows)

            async def ensure_column(table: str, column: str, decl: str) -> None:
                if not await column_exists(table, column):
                    await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

            await ensure_column("runs", "conversation_id", "TEXT")
            await ensure_column("messages", "conversation_id", "TEXT")
            await ensure_column("conversation_state", "default_conversation_id", "TEXT")

            cursor = await db.execute("SELECT id FROM conversations LIMIT 1")
            existing = await cursor.fetchone()
            await cursor.close()

            if not existing:
                convo_id = uuid.uuid4().hex
                cursor = await db.execute("SELECT MIN(created_at), MAX(created_at) FROM runs")
                row = await cursor.fetchone()
                await cursor.close()
                created_at = row[0] if row and row[0] else utc_now()
                updated_at = row[1] if row and row[1] else created_at
                cursor = await db.execute("SELECT user_question FROM runs ORDER BY created_at ASC LIMIT 1")
                row = await cursor.fetchone()
                await cursor.close()
                title = row[0] if row and row[0] else "Legacy chat"
                await db.execute(
                    "INSERT INTO conversations(id, created_at, updated_at, title, model_tier, reasoning_mode, manual_level, deep_mode, archived) "
                    "VALUES (?,?,?,?,?,?,?,?,0)",
                    (convo_id, created_at, updated_at, title, "pro", "auto", "MED", "auto"),
                )
                await db.execute(
                    "UPDATE conversation_state SET default_conversation_id=? WHERE id=1",
                    (convo_id,),
                )
                await db.execute(
                    "UPDATE runs SET conversation_id=? WHERE conversation_id IS NULL OR conversation_id=''",
                    (convo_id,),
                )
                await db.execute(
                    "UPDATE messages SET conversation_id=? WHERE conversation_id IS NULL OR conversation_id=''",
                    (convo_id,),
                )
            else:
                cursor = await db.execute("SELECT default_conversation_id FROM conversation_state WHERE id=1")
                row = await cursor.fetchone()
                await cursor.close()
                default_id = row[0] if row else None
                if not default_id:
                    cursor = await db.execute(
                        "SELECT id FROM conversations ORDER BY updated_at DESC, created_at DESC LIMIT 1"
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                    default_id = row[0] if row else None
                    if default_id:
                        await db.execute(
                            "UPDATE conversation_state SET default_conversation_id=? WHERE id=1",
                            (default_id,),
                        )
                if default_id:
                    await db.execute(
                        "UPDATE runs SET conversation_id=? WHERE conversation_id IS NULL OR conversation_id=''",
                        (default_id,),
                    )
                    await db.execute(
                        "UPDATE messages SET conversation_id=? WHERE conversation_id IS NULL OR conversation_id=''",
                        (default_id,),
                    )
            await db.commit()

    async def execute(self, query: str, params: Tuple[Any, ...] = ()) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(query, params)
            await db.commit()

    async def fetchall(self, query: str, params: Tuple[Any, ...] = ()) -> List[aiosqlite.Row]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()
            return rows

    async def fetchone(self, query: str, params: Tuple[Any, ...] = ()) -> Optional[aiosqlite.Row]:
        rows = await self.fetchall(query, params)
        return rows[0] if rows else None

    async def insert_run(
        self,
        run_id: str,
        conversation_id: str,
        question: str,
        reasoning_mode: str,
        status: str = "running",
    ) -> None:
        created_at = utc_now()
        await self.execute(
            "INSERT INTO runs(run_id, conversation_id, created_at, user_question, reasoning_mode, status) VALUES (?,?,?,?,?,?)",
            (run_id, conversation_id, created_at, question, reasoning_mode, status),
        )
        await self.touch_conversation(conversation_id, updated_at=created_at)

    async def update_run_router(self, run_id: str, router_decision_json: dict) -> None:
        await self.execute(
            "UPDATE runs SET router_decision_json=? WHERE run_id=?",
            (json.dumps(router_decision_json), run_id),
        )

    async def finalize_run(
        self,
        run_id: str,
        final_answer: str,
        confidence: str,
        status: str = "completed",
    ) -> None:
        await self.execute(
            "UPDATE runs SET final_answer=?, confidence=?, status=? WHERE run_id=?",
            (final_answer, confidence, status, run_id),
        )

    async def update_run_status(self, run_id: str, status: str) -> None:
        await self.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))

    async def add_message(self, run_id: str, conversation_id: str, role: str, content: str) -> dict:
        created_at = utc_now()
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "INSERT INTO messages(run_id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?)",
                (run_id, conversation_id, role, content, created_at),
            )
            await db.commit()
            await self.touch_conversation(conversation_id, updated_at=created_at)
            return {"id": cursor.lastrowid, "created_at": created_at}

    async def add_task(self, run_id: str, task_type: str, payload: dict, status: str) -> None:
        await self.execute(
            "INSERT INTO tasks(run_id, task_type, task_payload_json, status, created_at) VALUES (?,?,?,?,?)",
            (run_id, task_type, json.dumps(payload), status, utc_now()),
        )

    async def add_search(
        self,
        run_id: str,
        lane: str,
        query: str,
        search_depth: str,
        max_results: int,
        raw_response: dict,
    ) -> None:
        await self.execute(
            "INSERT INTO searches(run_id, lane, query, search_depth, max_results, raw_response_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, lane, query, search_depth, max_results, json.dumps(raw_response), utc_now()),
        )

    async def add_extract(
        self, run_id: str, lane: str, url: str, extract_depth: str, raw_response: dict
    ) -> None:
        await self.execute(
            "INSERT INTO extracts(run_id, lane, url, extract_depth, raw_response_json, created_at) VALUES (?,?,?,?,?,?)",
            (run_id, lane, url, extract_depth, json.dumps(raw_response), utc_now()),
        )

    async def add_source(
        self,
        run_id: str,
        lane: str,
        url: str,
        title: str,
        publisher: str,
        date_published: str,
        snippet: str,
        extracted_text: str,
    ) -> None:
        await self.execute(
            "INSERT INTO sources(run_id, lane, url, title, publisher, date_published, snippet, extracted_text) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, lane, url, title, publisher, date_published, snippet, extracted_text),
        )

    async def add_claim(
        self,
        run_id: str,
        claim: str,
        support_urls: List[str],
        confidence: str,
        notes: str = "",
    ) -> None:
        await self.execute(
            "INSERT INTO claims(run_id, claim, support_urls_json, confidence, notes) VALUES (?,?,?,?,?)",
            (run_id, claim, json.dumps(support_urls), confidence, notes),
        )

    async def add_draft(self, run_id: str, draft_text: str) -> None:
        await self.execute(
            "INSERT INTO drafts(run_id, draft_text, created_at) VALUES (?,?,?)",
            (run_id, draft_text, utc_now()),
        )

    async def add_verifier_report(
        self, run_id: str, verdict: str, issues_json: List[dict], revised_answer: Optional[str]
    ) -> None:
        await self.execute(
            "INSERT INTO verifier_reports(run_id, verdict, issues_json, revised_answer, created_at) VALUES (?,?,?,?,?)",
            (run_id, verdict, json.dumps(issues_json), revised_answer or "", utc_now()),
        )

    async def next_event_seq(self, run_id: str) -> int:
        row = await self.fetchone("SELECT MAX(seq) as max_seq FROM events WHERE run_id=?", (run_id,))
        max_seq = row["max_seq"] if row and row["max_seq"] is not None else 0
        return int(max_seq) + 1

    async def add_event(self, run_id: str, event_type: str, payload: dict) -> dict:
        seq = await self.next_event_seq(run_id)
        created_at = utc_now()
        await self.execute(
            "INSERT INTO events(run_id, seq, event_type, payload_json, created_at) VALUES (?,?,?,?,?)",
            (run_id, seq, event_type, json.dumps(payload), created_at),
        )
        return {"run_id": run_id, "seq": seq, "event_type": event_type, "payload": payload, "created_at": created_at}

    async def list_events(self, run_id: str, after_seq: int = 0) -> List[dict]:
        rows = await self.fetchall(
            "SELECT seq, event_type, payload_json, created_at FROM events WHERE run_id=? AND seq>? ORDER BY seq ASC",
            (run_id, after_seq),
        )
        return [
            {
                "seq": row["seq"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def save_config(self, payload: dict) -> None:
        await self.execute(
            "INSERT INTO configs(created_at, payload_json) VALUES (?,?)", (utc_now(), json.dumps(payload))
        )

    async def get_run_summary(self, run_id: str) -> Optional[dict]:
        row = await self.fetchone(
            "SELECT run_id, conversation_id, created_at, user_question, reasoning_mode, router_decision_json, final_answer, confidence, status "
            "FROM runs WHERE run_id=?",
            (run_id,),
        )
        if not row:
            return None
        return {
            "run_id": row["run_id"],
            "conversation_id": row["conversation_id"],
            "created_at": row["created_at"],
            "user_question": row["user_question"],
            "reasoning_mode": row["reasoning_mode"],
            "router_decision": json.loads(row["router_decision_json"] or "{}"),
            "final_answer": row["final_answer"],
            "confidence": row["confidence"],
            "status": row["status"],
        }

    async def get_latest_run(self, after: Optional[str] = None, conversation_id: Optional[str] = None) -> Optional[dict]:
        params: Tuple[Any, ...]
        clauses = []
        if conversation_id:
            clauses.append("conversation_id=?")
        if after:
            clauses.append("created_at > ?")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params_list: List[Any] = []
        if conversation_id:
            params_list.append(conversation_id)
        if after:
            params_list.append(after)
        params = tuple(params_list)
        row = await self.fetchone(
            f"SELECT run_id FROM runs {where} ORDER BY created_at DESC LIMIT 1",
            params,
        )
        if not row:
            return None
        return await self.get_run_summary(row["run_id"])

    async def get_sources(self, run_id: str) -> List[dict]:
        rows = await self.fetchall(
            "SELECT lane, url, title, publisher, date_published, snippet, extracted_text FROM sources WHERE run_id=?",
            (run_id,),
        )
        return [dict(row) for row in rows]

    async def get_claims(self, run_id: str) -> List[dict]:
        rows = await self.fetchall(
            "SELECT claim, support_urls_json, confidence, notes FROM claims WHERE run_id=?", (run_id,)
        )
        mapped = []
        for row in rows:
            mapped.append(
                {
                    "claim": row["claim"],
                    "support_urls": json.loads(row["support_urls_json"] or "[]"),
                    "confidence": row["confidence"],
                    "notes": row["notes"],
                }
            )
        return mapped

    async def get_latest_draft(self, run_id: str) -> Optional[str]:
        row = await self.fetchone(
            "SELECT draft_text FROM drafts WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,)
        )
        return row["draft_text"] if row else None

    async def get_verifier_report(self, run_id: str) -> Optional[dict]:
        row = await self.fetchone(
            "SELECT verdict, issues_json, revised_answer, created_at FROM verifier_reports WHERE run_id=? ORDER BY id DESC LIMIT 1",
            (run_id,),
        )
        if not row:
            return None
        return {
            "verdict": row["verdict"],
            "issues": json.loads(row["issues_json"] or "[]"),
            "revised_answer": row["revised_answer"],
            "created_at": row["created_at"],
        }

    async def get_artifacts(self, run_id: str) -> List[dict]:
        rows = await self.fetchall(
            "SELECT step_id, key, artifact_type, content_text, content_json, created_at FROM artifacts WHERE run_id=? ORDER BY id ASC",
            (run_id,),
        )
        out = []
        for row in rows:
            out.append(
                {
                    "step_id": row["step_id"],
                    "key": row["key"],
                    "artifact_type": row["artifact_type"],
                    "content_text": row["content_text"],
                    "content_json": json.loads(row["content_json"] or "{}"),
                    "created_at": row["created_at"],
                }
            )
        return out

    async def add_step_plan(self, run_id: str, plan_json: dict) -> None:
        await self.execute(
            "INSERT INTO step_plans(run_id, plan_json, created_at) VALUES (?,?,?)",
            (run_id, json.dumps(plan_json), utc_now()),
        )

    async def add_step_run(
        self,
        run_id: str,
        step_id: int,
        status: str,
        agent_profile: str,
        prompt_text: str,
    ) -> int:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "INSERT INTO step_runs(run_id, step_id, status, started_at, agent_profile, prompt_text) VALUES (?,?,?,?,?,?)",
                (run_id, step_id, status, utc_now(), agent_profile, prompt_text),
            )
            await db.commit()
            return cursor.lastrowid

    async def update_step_run(
        self,
        row_id: int,
        status: str,
        output_json: Optional[dict] = None,
        error_text: Optional[str] = None,
    ) -> None:
        await self.execute(
            "UPDATE step_runs SET status=?, ended_at=?, output_json=?, error_text=? WHERE id=?",
            (status, utc_now(), json.dumps(output_json or {}), error_text or "", row_id),
        )

    async def add_artifact(self, run_id: str, artifact: "Artifact") -> None:
        await self.execute(
            "INSERT INTO artifacts(run_id, step_id, key, artifact_type, content_text, content_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                run_id,
                artifact.step_id,
                artifact.key,
                artifact.artifact_type,
                artifact.content_text or "",
                json.dumps(artifact.content_json or {}),
                utc_now(),
            ),
        )

    async def add_upload(
        self,
        run_id: Optional[str],
        filename: str,
        original_name: str,
        mime: str,
        size_bytes: int,
        storage_path: str,
        status: str = "received",
    ) -> int:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "INSERT INTO uploads(run_id, filename, original_name, mime, size_bytes, storage_path, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (run_id or "", filename, original_name, mime, size_bytes, storage_path, status, utc_now()),
            )
            await db.commit()
            return cursor.lastrowid

    async def update_upload_status(
        self, upload_id: int, status: str, summary_text: Optional[str] = None, summary_json: Optional[dict] = None
    ) -> None:
        await self.execute(
            "UPDATE uploads SET status=?, summary_text=?, summary_json=? WHERE id=?",
            (status, summary_text or "", json.dumps(summary_json or {}), upload_id),
        )

    async def assign_upload_to_run(self, upload_id: int, run_id: str) -> None:
        await self.execute("UPDATE uploads SET run_id=? WHERE id=?", (run_id, upload_id))

    async def get_upload(self, upload_id: int) -> Optional[dict]:
        row = await self.fetchone(
            "SELECT id, run_id, filename, original_name, mime, size_bytes, storage_path, status, summary_text, summary_json, created_at FROM uploads WHERE id=?",
            (upload_id,),
        )
        if not row:
            return None
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "filename": row["filename"],
            "original_name": row["original_name"],
            "mime": row["mime"],
            "size_bytes": row["size_bytes"],
            "storage_path": row["storage_path"],
            "status": row["status"],
            "summary_text": row["summary_text"],
            "summary_json": json.loads(row["summary_json"] or "{}"),
            "created_at": row["created_at"],
        }

    async def list_uploads(self, run_id: str) -> List[dict]:
        rows = await self.fetchall(
            "SELECT id, run_id, filename, original_name, mime, size_bytes, storage_path, status, summary_text, summary_json, created_at FROM uploads WHERE run_id=? ORDER BY id ASC",
            (run_id,),
        )
        out = []
        for row in rows:
            out.append(
                {
                    "id": row["id"],
                    "run_id": row["run_id"],
                    "filename": row["filename"],
                    "original_name": row["original_name"],
                    "mime": row["mime"],
                    "size_bytes": row["size_bytes"],
                    "storage_path": row["storage_path"],
                    "status": row["status"],
                    "summary_text": row["summary_text"],
                    "summary_json": json.loads(row["summary_json"] or "{}"),
                    "created_at": row["created_at"],
                }
            )
        return out

    async def add_control_action(self, run_id: str, payload: dict) -> None:
        await self.execute(
            "INSERT INTO control_actions(run_id, action_type, payload_json, created_at) VALUES (?,?,?,?)",
            (run_id, payload.get("control"), json.dumps(payload), utc_now()),
        )

    async def add_memory_item(
        self, kind: str, title: str, content: str, tags: List[str], pinned: bool = False, relevance_score: float = 0.0
    ) -> int:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "INSERT INTO memory_items(created_at, updated_at, kind, title, content, tags_json, pinned_bool, relevance_score) VALUES (?,?,?,?,?,?,?,?)",
                (utc_now(), utc_now(), kind, title, content, json.dumps(tags), 1 if pinned else 0, relevance_score),
            )
            await db.commit()
            return cursor.lastrowid

    async def update_memory_item(
        self, item_id: int, title: Optional[str] = None, content: Optional[str] = None, pinned: Optional[bool] = None
    ) -> None:
        row = await self.fetchone("SELECT title, content, pinned_bool FROM memory_items WHERE id=?", (item_id,))
        if not row:
            return
        new_title = title if title is not None else row["title"]
        new_content = content if content is not None else row["content"]
        new_pinned = pinned if pinned is not None else row["pinned_bool"]
        await self.execute(
            "UPDATE memory_items SET title=?, content=?, pinned_bool=?, updated_at=? WHERE id=?",
            (new_title, new_content, 1 if new_pinned else 0, utc_now(), item_id),
        )

    async def delete_memory_item(self, item_id: int) -> None:
        await self.execute("DELETE FROM memory_items WHERE id=?", (item_id,))
        await self.execute("DELETE FROM run_memory_links WHERE memory_item_id=?", (item_id,))

    async def search_memory(self, query: str, limit: int = 10) -> List[dict]:
        pattern = f"%{query}%"
        rows = await self.fetchall(
            "SELECT id, kind, title, content, tags_json, pinned_bool, relevance_score, updated_at FROM memory_items "
            "WHERE title LIKE ? OR content LIKE ? ORDER BY pinned_bool DESC, relevance_score DESC, updated_at DESC LIMIT ?",
            (pattern, pattern, limit),
        )
        return [
            {
                "id": r["id"],
                "kind": r["kind"],
                "title": r["title"],
                "content": r["content"],
                "tags": json.loads(r["tags_json"] or "[]"),
                "pinned": bool(r["pinned_bool"]),
                "relevance_score": r["relevance_score"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    async def list_memory(self, limit: int = 50) -> List[dict]:
        rows = await self.fetchall(
            "SELECT id, kind, title, content, tags_json, pinned_bool, relevance_score, updated_at FROM memory_items "
            "ORDER BY pinned_bool DESC, relevance_score DESC, updated_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "id": r["id"],
                "kind": r["kind"],
                "title": r["title"],
                "content": r["content"],
                "tags": json.loads(r["tags_json"] or "[]"),
                "pinned": bool(r["pinned_bool"]),
                "relevance_score": r["relevance_score"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    async def link_memory_to_run(self, run_id: str, memory_item_id: int, reason: str) -> None:
        await self.execute(
            "INSERT INTO run_memory_links(run_id, memory_item_id, reason) VALUES (?,?,?)",
            (run_id, memory_item_id, reason),
        )

    async def get_run_memory(self, run_id: str) -> List[dict]:
        rows = await self.fetchall(
            "SELECT m.id, m.title, m.content, m.tags_json, m.pinned_bool FROM memory_items m "
            "JOIN run_memory_links l ON m.id = l.memory_item_id WHERE l.run_id=?",
            (run_id,),
        )
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "content": r["content"],
                "tags": json.loads(r["tags_json"] or "[]"),
                "pinned": bool(r["pinned_bool"]),
            }
            for r in rows
        ]

    async def set_default_conversation_id(self, conversation_id: str) -> None:
        await self.execute(
            "UPDATE conversation_state SET default_conversation_id=? WHERE id=1",
            (conversation_id,),
        )

    async def get_default_conversation_id(self) -> Optional[str]:
        row = await self.fetchone("SELECT default_conversation_id FROM conversation_state WHERE id=1")
        if row and row["default_conversation_id"]:
            return row["default_conversation_id"]
        latest = await self.fetchone(
            "SELECT id FROM conversations ORDER BY updated_at DESC, created_at DESC LIMIT 1"
        )
        if latest and latest["id"]:
            await self.set_default_conversation_id(latest["id"])
            return latest["id"]
        return None

    async def touch_conversation(self, conversation_id: Optional[str], updated_at: Optional[str] = None) -> Optional[str]:
        if not conversation_id:
            return None
        stamp = updated_at or utc_now()
        await self.execute("UPDATE conversations SET updated_at=? WHERE id=?", (stamp, conversation_id))
        return stamp

    async def create_conversation(
        self,
        title: Optional[str] = None,
        model_tier: str = "pro",
        reasoning_mode: str = "auto",
        manual_level: str = "MED",
        deep_mode: str = "auto",
    ) -> dict:
        convo_id = uuid.uuid4().hex
        created_at = utc_now()
        await self.execute(
            "INSERT INTO conversations(id, created_at, updated_at, title, model_tier, reasoning_mode, manual_level, deep_mode, archived) "
            "VALUES (?,?,?,?,?,?,?,?,0)",
            (convo_id, created_at, created_at, title or "New chat", model_tier, reasoning_mode, manual_level, deep_mode),
        )
        default_id = await self.get_default_conversation_id()
        if not default_id:
            await self.set_default_conversation_id(convo_id)
        return {
            "id": convo_id,
            "created_at": created_at,
            "updated_at": created_at,
            "title": title or "New chat",
            "model_tier": model_tier,
            "reasoning_mode": reasoning_mode,
            "manual_level": manual_level,
            "deep_mode": deep_mode,
            "archived": False,
        }

    async def get_conversation(self, conversation_id: str) -> Optional[dict]:
        row = await self.fetchone(
            "SELECT id, created_at, updated_at, title, model_tier, reasoning_mode, manual_level, deep_mode, archived "
            "FROM conversations WHERE id=?",
            (conversation_id,),
        )
        if not row:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "title": row["title"],
            "model_tier": row["model_tier"],
            "reasoning_mode": row["reasoning_mode"],
            "manual_level": row["manual_level"],
            "deep_mode": row["deep_mode"],
            "archived": bool(row["archived"]),
        }

    async def list_conversations(self, include_archived: bool = False, limit: int = 200) -> List[dict]:
        where = "" if include_archived else "WHERE archived=0"
        rows = await self.fetchall(
            "SELECT id, created_at, updated_at, title, model_tier, reasoning_mode, manual_level, deep_mode, archived, "
            "(SELECT run_id FROM runs WHERE conversation_id=conversations.id ORDER BY created_at DESC LIMIT 1) AS latest_run_id, "
            "(SELECT status FROM runs WHERE conversation_id=conversations.id ORDER BY created_at DESC LIMIT 1) AS latest_status, "
            "(SELECT content FROM messages WHERE conversation_id=conversations.id ORDER BY created_at DESC LIMIT 1) AS latest_message, "
            "(SELECT role FROM messages WHERE conversation_id=conversations.id ORDER BY created_at DESC LIMIT 1) AS latest_role, "
            "(SELECT created_at FROM messages WHERE conversation_id=conversations.id ORDER BY created_at DESC LIMIT 1) AS latest_message_at "
            f"FROM conversations {where} ORDER BY updated_at DESC, created_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "title": r["title"],
                "model_tier": r["model_tier"],
                "reasoning_mode": r["reasoning_mode"],
                "manual_level": r["manual_level"],
                "deep_mode": r["deep_mode"],
                "archived": bool(r["archived"]),
                "latest_run_id": r["latest_run_id"],
                "latest_status": r["latest_status"],
                "latest_message": r["latest_message"],
                "latest_role": r["latest_role"],
                "latest_message_at": r["latest_message_at"],
            }
            for r in rows
        ]

    async def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        model_tier: Optional[str] = None,
        reasoning_mode: Optional[str] = None,
        manual_level: Optional[str] = None,
        deep_mode: Optional[str] = None,
    ) -> Optional[dict]:
        row = await self.fetchone(
            "SELECT title, model_tier, reasoning_mode, manual_level, deep_mode FROM conversations WHERE id=?",
            (conversation_id,),
        )
        if not row:
            return None
        next_title = title if title is not None else row["title"]
        next_tier = model_tier if model_tier is not None else row["model_tier"]
        next_reasoning = reasoning_mode if reasoning_mode is not None else row["reasoning_mode"]
        next_manual = manual_level if manual_level is not None else row["manual_level"]
        next_deep = deep_mode if deep_mode is not None else row["deep_mode"]
        updated_at = utc_now()
        await self.execute(
            "UPDATE conversations SET title=?, model_tier=?, reasoning_mode=?, manual_level=?, deep_mode=?, updated_at=? WHERE id=?",
            (next_title, next_tier, next_reasoning, next_manual, next_deep, updated_at, conversation_id),
        )
        return await self.get_conversation(conversation_id)

    async def ensure_conversation_title(self, conversation_id: str, title: str) -> None:
        row = await self.fetchone("SELECT title FROM conversations WHERE id=?", (conversation_id,))
        if not row:
            return
        current = (row["title"] or "").strip()
        if current and current.lower() not in ("new chat", "legacy chat"):
            return
        await self.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title, utc_now(), conversation_id),
        )

    async def delete_conversation(self, conversation_id: str) -> None:
        await self.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        for table in (
            "tasks",
            "searches",
            "extracts",
            "sources",
            "claims",
            "drafts",
            "verifier_reports",
            "events",
            "step_plans",
            "step_runs",
            "artifacts",
            "uploads",
            "control_actions",
            "run_memory_links",
        ):
            await self.execute(
                f"DELETE FROM {table} WHERE run_id IN (SELECT run_id FROM runs WHERE conversation_id=?)",
                (conversation_id,),
            )
        await self.execute("DELETE FROM runs WHERE conversation_id=?", (conversation_id,))
        await self.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

    async def get_prompt_state(self) -> Optional[dict]:
        row = await self.fetchone("SELECT prompt_text, run_id, updated_at FROM prompt_state WHERE id=1")
        if not row or not row["prompt_text"]:
            return None
        return {
            "prompt_text": row["prompt_text"],
            "run_id": row["run_id"],
            "updated_at": row["updated_at"],
        }

    async def set_prompt_state(self, prompt_text: str, run_id: Optional[str] = None) -> dict:
        updated_at = utc_now()
        await self.execute(
            "INSERT OR REPLACE INTO prompt_state(id, prompt_text, run_id, updated_at) VALUES (1, ?, ?, ?)",
            (prompt_text, run_id, updated_at),
        )
        return {"prompt_text": prompt_text, "run_id": run_id, "updated_at": updated_at}

    async def clear_prompt_state(self, updated_at: Optional[str] = None) -> str:
        stamp = updated_at or utc_now()
        await self.execute(
            "UPDATE prompt_state SET prompt_text=NULL, run_id=NULL, updated_at=? WHERE id=1",
            (stamp,),
        )
        return stamp

    async def list_messages(self, conversation_id: Optional[str] = None, limit: int = 200) -> List[dict]:
        convo_id = conversation_id or await self.get_default_conversation_id()
        if not convo_id:
            return []
        rows = await self.fetchall(
            "SELECT id, run_id, conversation_id, role, content, created_at "
            "FROM messages WHERE conversation_id=? ORDER BY created_at ASC LIMIT ?",
            (convo_id, limit),
        )
        return [dict(r) for r in rows]

    async def reset_conversation(self, conversation_id: Optional[str] = None) -> str:
        reset_at = utc_now()
        convo_id = conversation_id or await self.get_default_conversation_id()
        async with aiosqlite.connect(self.path) as db:
            if convo_id:
                await db.execute("DELETE FROM messages WHERE conversation_id=?", (convo_id,))
            else:
                await db.execute("DELETE FROM messages")
            await db.execute(
                "INSERT OR REPLACE INTO conversation_state(id, reset_at, default_conversation_id) VALUES (1, ?, ?)",
                (reset_at, convo_id),
            )
            await db.execute(
                "UPDATE prompt_state SET prompt_text=NULL, run_id=NULL, updated_at=? WHERE id=1",
                (reset_at,),
            )
            await db.commit()
        return reset_at

    async def get_conversation_reset(self) -> Optional[str]:
        row = await self.fetchone("SELECT reset_at FROM conversation_state WHERE id=1")
        return row["reset_at"] if row and row["reset_at"] else None
