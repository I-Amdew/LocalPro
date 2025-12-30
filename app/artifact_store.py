import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

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


class ArtifactStore:
    """Artifact storage with stable refs backed by SQLite blobs."""

    def __init__(self, path: str):
        self.path = path

    async def put(
        self,
        key: Optional[str],
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        kind: str = "text",
    ) -> Dict[str, Any]:
        ref_id = key or str(uuid.uuid4())
        created_at = utc_now()
        meta = dict(metadata or {})
        meta.setdefault("created_at", created_at)
        if kind == "json" and not isinstance(content, str):
            content_text = ""
            content_json = _json_dumps(content)
        elif isinstance(content, str):
            content_text = content
            content_json = ""
        else:
            content_text = str(content)
            content_json = ""
        size = len(content_text) if content_text else len(content_json or "")
        meta.setdefault("size", size)
        uri = f"db://artifact/{ref_id}"
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO artifact_refs(ref_id, kind, uri, metadata_json, created_at, step_id) "
                "VALUES (?,?,?,?,?,?)",
                (ref_id, kind, uri, _json_dumps(meta), created_at, str(meta.get("step_id") or "")),
            )
            await db.execute(
                "INSERT OR REPLACE INTO artifact_blobs(ref_id, content_text, content_json, created_at) "
                "VALUES (?,?,?,?)",
                (ref_id, content_text, content_json, created_at),
            )
            await db.commit()
        return {
            "ref_id": ref_id,
            "kind": kind,
            "uri": uri,
            "metadata": meta,
        }

    async def get(self, ref_id: str, range: Optional[Dict[str, int]] = None) -> Any:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT content_text, content_json FROM artifact_blobs WHERE ref_id=?",
                (ref_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if not row:
            return None
        content_text = row["content_text"] or ""
        content_json = row["content_json"] or ""
        if content_json:
            payload = _json_loads(content_json, {})
        else:
            payload = content_text
        if range and isinstance(payload, str):
            start = int(range.get("start", 0))
            end = int(range.get("end", len(payload)))
            return payload[start:end]
        return payload

    async def list(self, prefix: Optional[str] = None, query: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if prefix:
            clauses.append("ref_id LIKE ?")
            params.append(f"{prefix}%")
        if query:
            clauses.append("metadata_json LIKE ?")
            params.append(f"%{query}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT ref_id, kind, uri, metadata_json, created_at FROM artifact_refs {where} ORDER BY created_at DESC",
                tuple(params),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [
            {
                "ref_id": row["ref_id"],
                "kind": row["kind"],
                "uri": row["uri"],
                "metadata": _json_loads(row["metadata_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        filters = filters or {}
        clauses = ["content_text LIKE ?"]
        params: List[Any] = [f"%{query}%"]
        if filters.get("kind"):
            clauses.append("ref_id IN (SELECT ref_id FROM artifact_refs WHERE kind=?)")
            params.append(filters["kind"])
        if filters.get("step_id"):
            clauses.append("ref_id IN (SELECT ref_id FROM artifact_refs WHERE step_id=?)")
            params.append(str(filters["step_id"]))
        where = " AND ".join(clauses)
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT ref_id, content_text FROM artifact_blobs WHERE {where} LIMIT 100",
                tuple(params),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [{"ref_id": row["ref_id"], "preview": row["content_text"][:200]} for row in rows]

    async def summarize(
        self,
        refs: Iterable[Dict[str, Any]],
        token_budget: int,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chunks: List[str] = []
        for ref in refs:
            ref_id = ref.get("ref_id") if isinstance(ref, dict) else str(ref)
            content = await self.get(ref_id)
            if isinstance(content, dict):
                chunks.append(_json_dumps(content))
            else:
                chunks.append(str(content))
        joined = "\n".join(chunks)
        budget_chars = max(1, token_budget * 4)
        summary = joined[:budget_chars]
        return await self.put(None, summary, metadata={"schema": schema or {}, "source_count": len(chunks)}, kind="text")

    async def pack_context(
        self,
        step_id: str,
        prereq_refs: Iterable[Dict[str, Any]],
        token_budget: int,
    ) -> Dict[str, Any]:
        packed = await self.summarize(prereq_refs, token_budget, schema={"step_id": step_id})
        return packed
