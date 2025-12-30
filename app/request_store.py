import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class RequestStore:
    """Persistent request store for planner-time dispatch."""

    def __init__(self, path: str):
        self.path = path

    async def create(self, payload: Dict[str, Any]) -> str:
        request_id = str(payload.get("request_id") or uuid.uuid4())
        created_at = utc_now()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO plan_requests(request_id, plan_id, status, type, payload_json, expected_output_schema_json, "
                "result_refs_json, created_by, priority, created_at, updated_at, error_text) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    payload.get("plan_id") or "",
                    "pending",
                    payload.get("type") or "",
                    _json_dumps(payload.get("payload") or {}),
                    _json_dumps(payload.get("expected_output_schema") or {}),
                    _json_dumps([]),
                    payload.get("created_by") or "",
                    int(payload.get("priority") or 0),
                    created_at,
                    created_at,
                    "",
                ),
            )
            await db.commit()
        return request_id

    async def status(self, request_id: str) -> Optional[str]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT status FROM plan_requests WHERE request_id=?",
                (request_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return row["status"] if row else None

    async def result(self, request_id: str) -> Optional[List[Dict[str, Any]]]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT result_refs_json FROM plan_requests WHERE request_id=?",
                (request_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if not row:
            return None
        return _json_loads(row["result_refs_json"], [])

    async def fulfill(self, request_id: str, result_refs: List[Dict[str, Any]]) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "UPDATE plan_requests SET status=?, result_refs_json=?, updated_at=? WHERE request_id=?",
                ("done", _json_dumps(result_refs), utc_now(), request_id),
            )
            await db.commit()

    async def fail(self, request_id: str, error: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "UPDATE plan_requests SET status=?, error_text=?, updated_at=? WHERE request_id=?",
                ("failed", error, utc_now(), request_id),
            )
            await db.commit()

    async def list_pending(self, plan_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        clauses = ["status='pending'"]
        params: List[Any] = []
        if plan_id:
            clauses.append("plan_id=?")
            params.append(plan_id)
        where = " AND ".join(clauses)
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT request_id, plan_id, type, payload_json, expected_output_schema_json, priority FROM plan_requests "
                f"WHERE {where} ORDER BY priority DESC, created_at ASC LIMIT ?",
                (*params, limit),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [
            {
                "request_id": row["request_id"],
                "plan_id": row["plan_id"],
                "type": row["type"],
                "payload": _json_loads(row["payload_json"], {}),
                "expected_output_schema": _json_loads(row["expected_output_schema_json"], {}),
                "priority": int(row["priority"] or 0),
            }
            for row in rows
        ]
