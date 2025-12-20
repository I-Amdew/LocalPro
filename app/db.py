import json
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
                CREATE TABLE IF NOT EXISTS runs(
                    run_id TEXT PRIMARY KEY,
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
                """
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
        question: str,
        reasoning_mode: str,
        status: str = "running",
    ) -> None:
        await self.execute(
            "INSERT INTO runs(run_id, created_at, user_question, reasoning_mode, status) VALUES (?,?,?,?,?)",
            (run_id, utc_now(), question, reasoning_mode, status),
        )

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

    async def add_message(self, run_id: str, role: str, content: str) -> None:
        await self.execute(
            "INSERT INTO messages(run_id, role, content, created_at) VALUES (?,?,?,?)",
            (run_id, role, content, utc_now()),
        )

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
            "SELECT run_id, created_at, user_question, reasoning_mode, router_decision_json, final_answer, confidence, status FROM runs WHERE run_id=?",
            (run_id,),
        )
        if not row:
            return None
        return {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "user_question": row["user_question"],
            "reasoning_mode": row["reasoning_mode"],
            "router_decision": json.loads(row["router_decision_json"] or "{}"),
            "final_answer": row["final_answer"],
            "confidence": row["confidence"],
            "status": row["status"],
        }

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
