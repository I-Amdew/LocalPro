import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

