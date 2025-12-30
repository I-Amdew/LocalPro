import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import aiosqlite

STEP_STATUSES = {
    "PENDING",
    "READY",
    "CLAIMED",
    "RUNNING",
    "DONE",
    "FAILED",
    "CANCELED",
    "STALE",
}

FINDING_SEVERITIES = {"INFO", "WARN", "ERROR", "CRITICAL"}
FINDING_CATEGORIES = {
    "ASSUMPTION_INVALID",
    "MISSING_DATA",
    "SCOPE_EXPANSION",
    "CONTRADICTION",
    "QUALITY_FAILURE",
    "TOOL_FAILURE",
    "DEPENDENCY_ERROR",
    "OTHER",
}
FINDING_STATUSES = {"OPEN", "ACKED", "IN_PROGRESS", "RESOLVED", "DISMISSED"}
PATCH_STATUSES = {"PROPOSED", "VALIDATED", "APPLIED", "REJECTED", "CONFLICT"}


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


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if value is None:
        return []
    return [str(value)]


def _extract_fields(record: Dict[str, Any], fields: Optional[Iterable[str]]) -> Dict[str, Any]:
    if not fields:
        return record
    filtered: Dict[str, Any] = {}
    for key in fields:
        if key in record:
            filtered[key] = record[key]
    return filtered


def _normalize_enum(value: Any, allowed: Iterable[str], default: str) -> str:
    cleaned = str(value or "").upper()
    return cleaned if cleaned in allowed else default


class PlanStore:
    """Persistent plan store with revisioned steps and diff support."""

    def __init__(self, path: str):
        self.path = path

    async def _record_plan_change(
        self,
        db: aiosqlite.Connection,
        plan_id: str,
        revision: int,
        change_type: str,
        entity_id: str,
        payload: Dict[str, Any],
    ) -> None:
        await db.execute(
            "INSERT INTO plan_changes(plan_id, revision, change_type, entity_id, payload_json, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (plan_id, revision, change_type, entity_id, _json_dumps(payload), utc_now()),
        )

    async def _bump_revision(self, db: aiosqlite.Connection, plan_id: str) -> int:
        cursor = await db.execute("SELECT revision FROM plans WHERE plan_id=?", (plan_id,))
        row = await cursor.fetchone()
        await cursor.close()
        current = int(row["revision"]) if row and row["revision"] is not None else 0
        revision = current + 1
        await db.execute(
            "UPDATE plans SET revision=?, updated_at=? WHERE plan_id=?",
            (revision, utc_now(), plan_id),
        )
        return revision

    async def _record_change(
        self,
        db: aiosqlite.Connection,
        plan_id: str,
        revision: int,
        step_id: str,
        changed_fields: List[str],
        old: Dict[str, Any],
        new: Dict[str, Any],
    ) -> None:
        await db.execute(
            "INSERT INTO plan_step_changes(plan_id, revision, step_id, changed_fields_json, old_json, new_json, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                plan_id,
                revision,
                step_id,
                _json_dumps(changed_fields),
                _json_dumps(old),
                _json_dumps(new),
                utc_now(),
            ),
        )

    async def create(self, metadata: Dict[str, Any]) -> str:
        plan_id = str(metadata.get("plan_id") or uuid.uuid4())
        created_at = utc_now()
        partitions = metadata.get("partitions") or []
        clean_meta = dict(metadata)
        clean_meta.pop("plan_id", None)
        clean_meta.pop("partitions", None)
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO plans(plan_id, created_at, updated_at, metadata_json, revision, partitions_json) "
                "VALUES (?,?,?,?,?,?)",
                (plan_id, created_at, created_at, _json_dumps(clean_meta), 0, _json_dumps(partitions)),
            )
            await db.commit()
        return plan_id

    async def get(self, plan_id: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT plan_id, created_at, updated_at, metadata_json, revision, partitions_json FROM plans WHERE plan_id=?",
                (plan_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if not row:
            return None
        return {
            "plan_id": row["plan_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": _json_loads(row["metadata_json"], {}),
            "revision": int(row["revision"] or 0),
            "partitions": _json_loads(row["partitions_json"], []),
        }

    async def add_steps(self, plan_id: str, steps: List[Dict[str, Any]]) -> List[str]:
        created_at = utc_now()
        new_ids: List[str] = []
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            revision = await self._bump_revision(db, plan_id)
            for step in steps:
                step_id = str(step.get("step_id") or uuid.uuid4())
                new_ids.append(step_id)
                title = str(step.get("title") or step.get("name") or f"Step {step_id}")
                description = str(step.get("description") or "")
                step_type = step.get("step_type")
                if step_type is None and "type" in step:
                    step_type = step.get("type")
                step_type = str(step_type).upper() if step_type else None
                status = str(step.get("status") or "PENDING").upper()
                if status not in STEP_STATUSES:
                    status = "PENDING"
                prereqs = _coerce_str_list(step.get("prereq_step_ids") or step.get("depends_on"))
                tags = _coerce_str_list(step.get("tags"))
                priority = int(step.get("priority") or 0)
                cost_hint = step.get("cost_hint") or {}
                partition_key = step.get("partition_key")
                attempt = int(step.get("attempt") or 0)
                max_retries = int(step.get("max_retries") or 1)
                created_by = step.get("created_by") or {"type": "planner", "id": "scaffold"}
                claimed_by = step.get("claimed_by")
                run_metadata = step.get("run_metadata") or {}
                input_refs = step.get("input_refs") or []
                output_refs = step.get("output_refs") or []
                notes = step.get("notes")
                await db.execute(
                    "INSERT INTO plan_steps(step_id, plan_id, title, description, step_type, status, prereq_step_ids_json, "
                    "tags_json, priority, cost_hint_json, partition_key, attempt, max_retries, created_by_json, "
                    "claimed_by, run_metadata_json, input_refs_json, output_refs_json, notes, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        step_id,
                        plan_id,
                        title,
                        description,
                        step_type,
                        status,
                        _json_dumps(prereqs),
                        _json_dumps(tags),
                        priority,
                        _json_dumps(cost_hint),
                        partition_key,
                        attempt,
                        max_retries,
                        _json_dumps(created_by),
                        claimed_by,
                        _json_dumps(run_metadata),
                        _json_dumps(input_refs),
                        _json_dumps(output_refs),
                        notes,
                        created_at,
                        created_at,
                    ),
                )
                await self._record_change(
                    db,
                    plan_id,
                    revision,
                    step_id,
                    ["insert"],
                    {},
                    {
                        "status": status,
                        "title": title,
                        "description": description,
                        "prereq_step_ids": prereqs,
                        "tags": tags,
                        "priority": priority,
                        "partition_key": partition_key,
                    },
                )
            await db.commit()
        return new_ids

    async def _fetch_step(self, db: aiosqlite.Connection, plan_id: str, step_id: str) -> Optional[aiosqlite.Row]:
        cursor = await db.execute(
            "SELECT * FROM plan_steps WHERE plan_id=? AND step_id=?",
            (plan_id, step_id),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row

    def _row_to_step(self, row: aiosqlite.Row) -> Dict[str, Any]:
        return {
            "step_id": row["step_id"],
            "plan_id": row["plan_id"],
            "title": row["title"],
            "description": row["description"],
            "step_type": row["step_type"] if "step_type" in row.keys() else None,
            "status": row["status"],
            "prereq_step_ids": _json_loads(row["prereq_step_ids_json"], []),
            "tags": _json_loads(row["tags_json"], []),
            "priority": int(row["priority"] or 0),
            "cost_hint": _json_loads(row["cost_hint_json"], {}),
            "partition_key": row["partition_key"],
            "attempt": int(row["attempt"] or 0),
            "max_retries": int(row["max_retries"] or 0),
            "created_by": _json_loads(row["created_by_json"], {}),
            "claimed_by": row["claimed_by"],
            "run_metadata": _json_loads(row["run_metadata_json"], {}),
            "input_refs": _json_loads(row["input_refs_json"], []),
            "output_refs": _json_loads(row["output_refs_json"], []),
            "notes": row["notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def update_step(self, plan_id: str, step_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {
            "title",
            "description",
            "step_type",
            "status",
            "prereq_step_ids",
            "tags",
            "priority",
            "cost_hint",
            "partition_key",
            "attempt",
            "max_retries",
            "created_by",
            "claimed_by",
            "run_metadata",
            "input_refs",
            "output_refs",
            "notes",
        }
        updates = {k: v for k, v in patch.items() if k in allowed}
        if not updates:
            return None
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            row = await self._fetch_step(db, plan_id, step_id)
            if not row:
                await db.execute("ROLLBACK")
                return None
            old = self._row_to_step(row)
            assignments: List[str] = []
            params: List[Any] = []
            for key, value in updates.items():
                column = key
                stored: Any = value
                if key in ("prereq_step_ids", "tags", "cost_hint", "created_by", "run_metadata", "input_refs", "output_refs"):
                    column = f"{key}_json" if not key.endswith("_json") else key
                    stored = _json_dumps(value)
                elif key == "step_type":
                    stored = str(value).upper() if value else None
                elif key == "status":
                    stored = str(value).upper()
                    if stored not in STEP_STATUSES:
                        stored = "PENDING"
                assignments.append(f"{column}=?")
                params.append(stored)
            params.extend([utc_now(), plan_id, step_id])
            sql = f"UPDATE plan_steps SET {', '.join(assignments)}, updated_at=? WHERE plan_id=? AND step_id=?"
            await db.execute(sql, tuple(params))
            revision = await self._bump_revision(db, plan_id)
            new_row = await self._fetch_step(db, plan_id, step_id)
            new = self._row_to_step(new_row) if new_row else {}
            changed_fields = list(updates.keys())
            await self._record_change(db, plan_id, revision, step_id, changed_fields, old, new)
            await db.commit()
        return new

    async def set_prereqs(self, plan_id: str, step_id: str, prereq_step_ids: List[str]) -> Optional[Dict[str, Any]]:
        return await self.update_step(plan_id, step_id, {"prereq_step_ids": prereq_step_ids})

    async def clear_notes(self, plan_id: str, step_id: str) -> Optional[Dict[str, Any]]:
        return await self.update_step(plan_id, step_id, {"notes": None})

    async def _prereqs_satisfied(
        self, db: aiosqlite.Connection, plan_id: str, prereqs: List[str]
    ) -> bool:
        if not prereqs:
            return True
        placeholders = ",".join("?" for _ in prereqs)
        cursor = await db.execute(
            f"SELECT step_id, status FROM plan_steps WHERE plan_id=? AND step_id IN ({placeholders})",
            (plan_id, *prereqs),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        status_map = {row["step_id"]: row["status"] for row in rows}
        for pid in prereqs:
            if status_map.get(pid) != "DONE":
                return False
        return True

    async def claim_step(
        self, plan_id: str, step_id: str, worker_id: str, lease_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            row = await self._fetch_step(db, plan_id, step_id)
            if not row:
                await db.execute("ROLLBACK")
                return {"ok": False, "status": "not_found"}
            status = row["status"]
            if status not in ("PENDING", "READY"):
                await db.execute("ROLLBACK")
                return {"ok": False, "status": "conflict"}
            prereqs = _json_loads(row["prereq_step_ids_json"], [])
            if not await self._prereqs_satisfied(db, plan_id, prereqs):
                await db.execute("ROLLBACK")
                return {"ok": False, "status": "blocked"}
            await db.execute(
                "UPDATE plan_steps SET status=?, claimed_by=?, updated_at=? WHERE plan_id=? AND step_id=?",
                ("CLAIMED", worker_id, utc_now(), plan_id, step_id),
            )
            revision = await self._bump_revision(db, plan_id)
            old = {"status": status, "claimed_by": row["claimed_by"]}
            new = {"status": "CLAIMED", "claimed_by": worker_id}
            await self._record_change(db, plan_id, revision, step_id, ["status", "claimed_by"], old, new)
            await db.commit()
        return {"ok": True}

    async def mark_running(
        self,
        plan_id: str,
        step_id: str,
        worker_id: str,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        run_metadata = run_metadata or {}
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            row = await self._fetch_step(db, plan_id, step_id)
            if not row:
                await db.execute("ROLLBACK")
                return None
            status = row["status"]
            if status not in ("READY", "CLAIMED"):
                await db.execute("ROLLBACK")
                return None
            attempt = int(row["attempt"] or 0) + 1
            await db.execute(
                "UPDATE plan_steps SET status=?, claimed_by=?, run_metadata_json=?, attempt=?, updated_at=? "
                "WHERE plan_id=? AND step_id=?",
                ("RUNNING", worker_id, _json_dumps(run_metadata), attempt, utc_now(), plan_id, step_id),
            )
            revision = await self._bump_revision(db, plan_id)
            old = {"status": status, "claimed_by": row["claimed_by"], "attempt": row["attempt"]}
            new = {"status": "RUNNING", "claimed_by": worker_id, "attempt": attempt}
            await self._record_change(db, plan_id, revision, step_id, ["status", "claimed_by", "attempt"], old, new)
            await db.commit()
        return {"status": "RUNNING", "attempt": attempt}

    async def mark_done(
        self,
        plan_id: str,
        step_id: str,
        output_refs: List[Dict[str, Any]],
        evidence: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            row = await self._fetch_step(db, plan_id, step_id)
            if not row:
                await db.execute("ROLLBACK")
                return {"ok": False, "status": "not_found"}
            status = row["status"]
            if status not in ("CLAIMED", "RUNNING"):
                await db.execute("ROLLBACK")
                return {"ok": False, "status": "conflict"}
            prereqs = _json_loads(row["prereq_step_ids_json"], [])
            if not await self._prereqs_satisfied(db, plan_id, prereqs):
                await db.execute("ROLLBACK")
                return {"ok": False, "status": "blocked"}
            run_metadata = _json_loads(row["run_metadata_json"], {})
            invalidate_on_complete = run_metadata.get("invalidate_on_complete")
            if invalidate_on_complete:
                behavior = str(run_metadata.get("invalidate_behavior") or "SET_STATUS_STALE").upper()
                new_status = "STALE" if behavior == "SET_STATUS_STALE" else "PENDING"
                reason = str(run_metadata.get("invalidate_reason") or "invalidated_on_complete")
                run_metadata.pop("invalidate_on_complete", None)
                run_metadata.pop("invalidate_behavior", None)
                run_metadata.pop("invalidate_reason", None)
                await db.execute(
                    "UPDATE plan_steps SET status=?, output_refs_json=?, run_metadata_json=?, error_json=?, updated_at=? "
                    "WHERE plan_id=? AND step_id=?",
                    (
                        new_status,
                        _json_dumps([]),
                        _json_dumps(run_metadata),
                        _json_dumps({"invalidated": True, "reason": reason}),
                        utc_now(),
                        plan_id,
                        step_id,
                    ),
                )
                revision = await self._bump_revision(db, plan_id)
                old = {"status": status, "output_refs": _json_loads(row["output_refs_json"], [])}
                new = {"status": new_status, "output_refs": [], "run_metadata": run_metadata}
                await self._record_change(
                    db, plan_id, revision, step_id, ["status", "output_refs", "run_metadata"], old, new
                )
            else:
                await db.execute(
                    "UPDATE plan_steps SET status=?, output_refs_json=?, updated_at=? WHERE plan_id=? AND step_id=?",
                    ("DONE", _json_dumps(output_refs), utc_now(), plan_id, step_id),
                )
                revision = await self._bump_revision(db, plan_id)
                old = {"status": status, "output_refs": _json_loads(row["output_refs_json"], [])}
                new = {"status": "DONE", "output_refs": output_refs}
                await self._record_change(db, plan_id, revision, step_id, ["status", "output_refs"], old, new)
            await db.commit()
        return {"ok": True}

    async def mark_failed(
        self,
        plan_id: str,
        step_id: str,
        error: str,
        retryable: bool,
        diagnostics_ref: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            row = await self._fetch_step(db, plan_id, step_id)
            if not row:
                await db.execute("ROLLBACK")
                return {"ok": False, "status": "not_found"}
            status = row["status"]
            await db.execute(
                "UPDATE plan_steps SET status=?, error_json=?, updated_at=? WHERE plan_id=? AND step_id=?",
                ("FAILED", _json_dumps({"error": error, "retryable": retryable, "diagnostics": diagnostics_ref}), utc_now(), plan_id, step_id),
            )
            revision = await self._bump_revision(db, plan_id)
            old = {"status": status}
            new = {"status": "FAILED"}
            await self._record_change(db, plan_id, revision, step_id, ["status"], old, new)
            await db.commit()
        return {"ok": True}

    async def get_overview(self, plan_id: str, verbosity: str = "default") -> Dict[str, Any]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT revision FROM plans WHERE plan_id=?",
                (plan_id,),
            )
            plan_row = await cursor.fetchone()
            await cursor.close()
            revision = int(plan_row["revision"] or 0) if plan_row else 0
            cursor = await db.execute(
                "SELECT status, COUNT(*) as cnt FROM plan_steps WHERE plan_id=? GROUP BY status",
                (plan_id,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            counts_by_status = {row["status"]: int(row["cnt"] or 0) for row in rows}
            cursor = await db.execute(
                "SELECT status, COUNT(*) as cnt FROM plan_findings WHERE plan_id=? GROUP BY status",
                (plan_id,),
            )
            finding_rows = await cursor.fetchall()
            await cursor.close()
            findings_by_status = {row["status"]: int(row["cnt"] or 0) for row in finding_rows}
            tag_counts: Dict[str, int] = {}
            try:
                cursor = await db.execute(
                    "SELECT json_each.value as tag, COUNT(*) as cnt "
                    "FROM plan_steps, json_each(plan_steps.tags_json) "
                    "WHERE plan_id=? GROUP BY json_each.value",
                    (plan_id,),
                )
                tag_rows = await cursor.fetchall()
                await cursor.close()
                tag_counts = {row["tag"]: int(row["cnt"] or 0) for row in tag_rows if row["tag"]}
            except Exception:
                tag_counts = {}
            cursor = await db.execute(
                "SELECT partition_key, COUNT(*) as cnt FROM plan_steps WHERE plan_id=? GROUP BY partition_key",
                (plan_id,),
            )
            partition_rows = await cursor.fetchall()
            await cursor.close()
            partition_stats = {
                row["partition_key"] or "default": int(row["cnt"] or 0) for row in partition_rows
            }
            ready_count = counts_by_status.get("READY", 0)
            running_count = counts_by_status.get("RUNNING", 0)
            failed_count = counts_by_status.get("FAILED", 0)
            top_blockers = []
            if verbosity in ("default", "verbose"):
                cursor = await db.execute(
                    "SELECT step_id, prereq_step_ids_json FROM plan_steps WHERE plan_id=? AND status IN ('PENDING','READY')",
                    (plan_id,),
                )
                blocker_rows = await cursor.fetchall()
                await cursor.close()
                blocker_counts: Dict[str, int] = {}
                for row in blocker_rows:
                    prereqs = _json_loads(row["prereq_step_ids_json"], [])
                    for pid in prereqs:
                        blocker_counts[pid] = blocker_counts.get(pid, 0) + 1
                top_blockers = sorted(
                    [{"step_id": sid, "blocked_count": cnt} for sid, cnt in blocker_counts.items()],
                    key=lambda item: item["blocked_count"],
                    reverse=True,
                )[:5]
            events = []
            if verbosity == "verbose":
                cursor = await db.execute(
                    "SELECT event_type, payload_json, created_at FROM plan_events WHERE plan_id=? ORDER BY id DESC LIMIT 5",
                    (plan_id,),
                )
                events = await cursor.fetchall()
                await cursor.close()
            return {
                "revision": revision,
                "counts_by_status": counts_by_status,
                "findings_by_status": findings_by_status,
                "counts_by_tag": tag_counts,
                "ready_count": ready_count,
                "running_count": running_count,
                "failed_count": failed_count,
                "top_blockers": top_blockers,
                "partitions_stats": partition_stats,
                "recent_events": [
                    {"event_type": row["event_type"], "payload": _json_loads(row["payload_json"], {}), "created_at": row["created_at"]}
                    for row in events
                ],
            }

    async def get_diff(
        self, plan_id: str, since_revision: int, limit: int = 200, cursor: Optional[int] = None
    ) -> Dict[str, Any]:
        offset = int(cursor or 0)
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor_db = await db.execute(
                "SELECT revision, step_id, changed_fields_json, old_json, new_json, created_at "
                "FROM plan_step_changes WHERE plan_id=? AND revision>? "
                "ORDER BY revision ASC, id ASC LIMIT ? OFFSET ?",
                (plan_id, since_revision, limit, offset),
            )
            rows = await cursor_db.fetchall()
            await cursor_db.close()
            cursor_db = await db.execute(
                "SELECT revision, change_type, entity_id, payload_json, created_at "
                "FROM plan_changes WHERE plan_id=? AND revision>? "
                "ORDER BY revision ASC, id ASC",
                (plan_id, since_revision),
            )
            change_rows = await cursor_db.fetchall()
            await cursor_db.close()
            cursor_db = await db.execute(
                "SELECT revision FROM plans WHERE plan_id=?",
                (plan_id,),
            )
            plan_row = await cursor_db.fetchone()
            await cursor_db.close()
        changes = [
            {
                "step_id": row["step_id"],
                "changed_fields": _json_loads(row["changed_fields_json"], []),
                "old": _json_loads(row["old_json"], {}),
                "new": _json_loads(row["new_json"], {}),
                "ts": row["created_at"],
            }
            for row in rows
        ]
        finding_changes: List[Dict[str, Any]] = []
        patch_changes: List[Dict[str, Any]] = []
        for row in change_rows:
            payload = _json_loads(row["payload_json"], {})
            entry = {
                "entity_id": row["entity_id"],
                "type": row["change_type"],
                "payload": payload,
                "ts": row["created_at"],
                "revision": row["revision"],
            }
            if row["change_type"] == "finding":
                finding_changes.append(entry)
            elif row["change_type"] == "patch":
                patch_changes.append(entry)
        next_cursor = offset + len(rows) if len(rows) == limit else None
        revision = int(plan_row["revision"] or 0) if plan_row else since_revision
        return {
            "revision": revision,
            "cursor_next": next_cursor,
            "changes": changes,
            "finding_changes": finding_changes,
            "patch_changes": patch_changes,
        }

    async def list_steps(
        self,
        plan_id: str,
        cursor: Optional[int] = None,
        limit: int = 200,
        fields: Optional[List[str]] = None,
        order_by: str = "priority_desc",
        partition_key: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        offset = int(cursor or 0)
        clauses = ["plan_id=?"]
        params: List[Any] = [plan_id]
        if partition_key is not None:
            clauses.append("partition_key=?")
            params.append(partition_key)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = " AND ".join(clauses)
        order_clause = "priority DESC, step_id ASC" if order_by == "priority_desc" else "step_id ASC"
        query = (
            "SELECT step_id, plan_id, title, description, step_type, status, prereq_step_ids_json, tags_json, priority, "
            "cost_hint_json, partition_key, attempt, max_retries, created_by_json, claimed_by, "
            "run_metadata_json, input_refs_json, output_refs_json, notes, created_at, updated_at "
            f"FROM plan_steps WHERE {where} ORDER BY {order_clause} LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, tuple(params))
            rows = await cursor.fetchall()
            await cursor.close()
            cursor = await db.execute(
                "SELECT revision FROM plans WHERE plan_id=?",
                (plan_id,),
            )
            plan_row = await cursor.fetchone()
            await cursor.close()
        steps = []
        for row in rows:
            record = self._row_to_step(row)
            steps.append(_extract_fields(record, fields))
        next_cursor = offset + len(rows) if len(rows) == limit else None
        revision = int(plan_row["revision"] or 0) if plan_row else 0
        return {"revision": revision, "cursor_next": next_cursor, "steps": steps}

    async def get_steps(
        self,
        plan_id: str,
        step_ids: List[str],
        fields: Optional[List[str]] = None,
        max_chars_per_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not step_ids:
            return {"revision": 0, "steps": []}
        placeholders = ",".join("?" for _ in step_ids)
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT * FROM plan_steps WHERE plan_id=? AND step_id IN ({placeholders})",
                (plan_id, *step_ids),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            cursor = await db.execute(
                "SELECT revision FROM plans WHERE plan_id=?",
                (plan_id,),
            )
            plan_row = await cursor.fetchone()
            await cursor.close()
        steps = []
        for row in rows:
            record = self._row_to_step(row)
            if max_chars_per_step and record.get("description"):
                record["description"] = record["description"][: max_chars_per_step]
            steps.append(_extract_fields(record, fields))
        revision = int(plan_row["revision"] or 0) if plan_row else 0
        return {"revision": revision, "steps": steps}

    async def resolve_prereqs(
        self,
        plan_id: str,
        mode: str = "title",
    ) -> Dict[str, Any]:
        changes: List[Dict[str, Any]] = []
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                "SELECT step_id, title, notes FROM plan_steps WHERE plan_id=? AND notes IS NOT NULL AND notes!=''",
                (plan_id,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            title_map: Dict[str, str] = {}
            tag_map: Dict[str, List[str]] = {}
            cursor = await db.execute(
                "SELECT step_id, title, tags_json FROM plan_steps WHERE plan_id=?",
                (plan_id,),
            )
            all_rows = await cursor.fetchall()
            await cursor.close()
            for row in all_rows:
                title_map[str(row["title"]).strip().lower()] = row["step_id"]
                tags = _json_loads(row["tags_json"], [])
                for tag in tags:
                    tag_map.setdefault(str(tag), []).append(row["step_id"])
            revision = await self._bump_revision(db, plan_id) if rows else None
            for row in rows:
                step_id = row["step_id"]
                notes = row["notes"]
                resolved: List[str] = []
                payload: Dict[str, Any] = {}
                if isinstance(notes, str):
                    payload = _json_loads(notes, {})
                    if not payload and notes.strip():
                        payload = {"prereq_titles": [notes.strip()]}
                if mode == "title":
                    for title in payload.get("prereq_titles", []) or []:
                        key = str(title).strip().lower()
                        if key in title_map:
                            resolved.append(title_map[key])
                for tag in payload.get("prereq_tags", []) or []:
                    resolved.extend(tag_map.get(str(tag), []))
                for step_ref in payload.get("prereq_step_ids", []) or []:
                    resolved.append(str(step_ref))
                if resolved:
                    resolved = sorted(set(resolved))
                    await db.execute(
                        "UPDATE plan_steps SET prereq_step_ids_json=?, notes=NULL, updated_at=? WHERE plan_id=? AND step_id=?",
                        (_json_dumps(resolved), utc_now(), plan_id, step_id),
                    )
                    await self._record_change(
                        db,
                        plan_id,
                        revision or 0,
                        step_id,
                        ["prereq_step_ids", "notes"],
                        {"notes": notes},
                        {"prereq_step_ids": resolved, "notes": None},
                    )
                    changes.append({"step_id": step_id, "prereq_step_ids": resolved})
            await db.commit()
        return {"changes": changes}

    def _row_to_finding(self, row: aiosqlite.Row) -> Dict[str, Any]:
        return {
            "finding_id": row["finding_id"],
            "plan_id": row["plan_id"],
            "source_step_id": row["source_step_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "severity": row["severity"],
            "category": row["category"],
            "summary": row["summary"],
            "details": row["details"] or "",
            "evidence_refs": _json_loads(row["evidence_refs_json"], []),
            "suggested_actions": _json_loads(row["suggested_actions_json"], []),
            "impacted_step_ids": _json_loads(row["impacted_step_ids_json"], []),
            "status": row["status"],
            "linked_patch_id": row["linked_patch_id"],
            "resolution_note": row["resolution_note"],
        }

    async def raise_finding(
        self,
        plan_id: str,
        source_step_id: Optional[str],
        finding_payload: Dict[str, Any],
    ) -> str:
        payload = dict(finding_payload or {})
        finding_id = str(payload.get("finding_id") or uuid.uuid4())
        created_at = utc_now()
        summary = str(payload.get("summary") or f"Finding from step {source_step_id or ''}".strip())
        details = str(payload.get("details") or "")
        severity = _normalize_enum(payload.get("severity"), FINDING_SEVERITIES, "INFO")
        category = _normalize_enum(payload.get("category"), FINDING_CATEGORIES, "OTHER")
        status = _normalize_enum(payload.get("status"), FINDING_STATUSES, "OPEN")
        evidence_refs = payload.get("evidence_refs") or []
        suggested_actions = payload.get("suggested_actions") or []
        impacted_step_ids = _coerce_str_list(payload.get("impacted_step_ids"))
        linked_patch_id = payload.get("linked_patch_id")
        resolution_note = payload.get("resolution_note")
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            revision = await self._bump_revision(db, plan_id)
            await db.execute(
                "INSERT INTO plan_findings("
                "finding_id, plan_id, source_step_id, created_at, updated_at, severity, category, summary, details, "
                "evidence_refs_json, suggested_actions_json, impacted_step_ids_json, status, linked_patch_id, resolution_note"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    finding_id,
                    plan_id,
                    source_step_id,
                    created_at,
                    created_at,
                    severity,
                    category,
                    summary,
                    details,
                    _json_dumps(evidence_refs),
                    _json_dumps(suggested_actions),
                    _json_dumps(impacted_step_ids),
                    status,
                    linked_patch_id,
                    resolution_note,
                ),
            )
            await self._record_plan_change(
                db,
                plan_id,
                revision,
                "finding",
                finding_id,
                {
                    "action": "created",
                    "finding_id": finding_id,
                    "status": status,
                    "severity": severity,
                    "category": category,
                    "summary": summary,
                    "source_step_id": source_step_id,
                },
            )
            await db.commit()
        return finding_id

    async def list_findings(
        self,
        plan_id: str,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[int] = None,
    ) -> Dict[str, Any]:
        offset = int(cursor or 0)
        clauses = ["plan_id=?"]
        params: List[Any] = [plan_id]
        if status:
            clauses.append("status=?")
            params.append(_normalize_enum(status, FINDING_STATUSES, status))
        if severity:
            clauses.append("severity=?")
            params.append(_normalize_enum(severity, FINDING_SEVERITIES, severity))
        where = " AND ".join(clauses)
        query = (
            "SELECT * FROM plan_findings WHERE "
            f"{where} ORDER BY created_at ASC, finding_id ASC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor_db = await db.execute(query, tuple(params))
            rows = await cursor_db.fetchall()
            await cursor_db.close()
        items = [self._row_to_finding(row) for row in rows]
        next_cursor = offset + len(rows) if len(rows) == limit else None
        return {"items": items, "cursor_next": next_cursor}

    async def get_finding(self, plan_id: str, finding_id: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM plan_findings WHERE plan_id=? AND finding_id=?",
                (plan_id, finding_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._row_to_finding(row) if row else None

    async def _update_finding_status(
        self,
        plan_id: str,
        finding_id: str,
        status: str,
        note: Optional[str] = None,
        actor: Optional[Dict[str, Any]] = None,
        linked_patch_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        status = _normalize_enum(status, FINDING_STATUSES, "OPEN")
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                "SELECT * FROM plan_findings WHERE plan_id=? AND finding_id=?",
                (plan_id, finding_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if not row:
                await db.execute("ROLLBACK")
                return None
            old = self._row_to_finding(row)
            resolution_note = note if note is not None else old.get("resolution_note")
            next_patch_id = linked_patch_id if linked_patch_id is not None else old.get("linked_patch_id")
            await db.execute(
                "UPDATE plan_findings SET status=?, resolution_note=?, linked_patch_id=?, updated_at=? "
                "WHERE plan_id=? AND finding_id=?",
                (status, resolution_note, next_patch_id, utc_now(), plan_id, finding_id),
            )
            revision = await self._bump_revision(db, plan_id)
            await self._record_plan_change(
                db,
                plan_id,
                revision,
                "finding",
                finding_id,
                {
                    "action": "updated",
                    "finding_id": finding_id,
                    "status": status,
                    "severity": old.get("severity"),
                    "category": old.get("category"),
                    "summary": old.get("summary"),
                    "actor": actor or {},
                },
            )
            await db.commit()
        return await self.get_finding(plan_id, finding_id)

    async def ack_finding(
        self,
        plan_id: str,
        finding_id: str,
        actor: Optional[Dict[str, Any]] = None,
        note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self._update_finding_status(plan_id, finding_id, "ACKED", note=note, actor=actor)

    async def resolve_finding(
        self,
        plan_id: str,
        finding_id: str,
        resolution_note: Optional[str],
        linked_patch_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self._update_finding_status(
            plan_id,
            finding_id,
            "RESOLVED",
            note=resolution_note,
            linked_patch_id=linked_patch_id,
        )

    async def dismiss_finding(
        self,
        plan_id: str,
        finding_id: str,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        return await self._update_finding_status(plan_id, finding_id, "DISMISSED", note=reason)

    async def get_dependents(self, plan_id: str, step_id: str) -> List[str]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT DISTINCT plan_steps.step_id as step_id "
                "FROM plan_steps, json_each(plan_steps.prereq_step_ids_json) "
                "WHERE plan_steps.plan_id=? AND json_each.value=?",
                (plan_id, step_id),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [row["step_id"] for row in rows]

    async def get_ancestors(self, plan_id: str, step_id: str) -> List[str]:
        pending = {str(step_id)}
        visited: set[str] = set()
        ancestors: set[str] = set()
        while pending:
            batch = list(pending)
            pending.clear()
            step_rows = await self.get_steps(
                plan_id,
                batch,
                fields=["step_id", "prereq_step_ids"],
            )
            for row in step_rows.get("steps") or []:
                for pid in row.get("prereq_step_ids") or []:
                    if pid not in ancestors and pid not in visited:
                        ancestors.add(pid)
                        pending.add(pid)
                visited.add(row.get("step_id"))
        return sorted(ancestors)

    async def compute_impact(
        self,
        plan_id: str,
        seed_step_ids: List[str],
        mode: str = "downstream",
    ) -> Dict[str, Any]:
        seeds = [str(s) for s in (seed_step_ids or []) if s]
        impacted: set[str] = set(seeds)
        include_downstream = mode in ("downstream", "all", "full", "both")
        include_upstream = mode in ("upstream", "all", "full", "both")
        if include_downstream:
            frontier = set(seeds)
            while frontier:
                next_frontier: set[str] = set()
                for sid in list(frontier):
                    dependents = await self.get_dependents(plan_id, sid)
                    for dep in dependents:
                        if dep not in impacted:
                            impacted.add(dep)
                            next_frontier.add(dep)
                frontier = next_frontier
        if include_upstream:
            for seed in seeds:
                ancestors = await self.get_ancestors(plan_id, seed)
                for anc in ancestors:
                    impacted.add(anc)
        impacted_list = sorted(impacted)
        step_rows = await self.get_steps(
            plan_id,
            impacted_list,
            fields=["step_id", "status", "partition_key"],
        )
        impacted_partitions: set[str] = set()
        recommended_invalidations: List[str] = []
        for row in step_rows.get("steps") or []:
            partition_key = row.get("partition_key")
            if partition_key:
                impacted_partitions.add(str(partition_key))
            if row.get("status") in ("DONE", "FAILED"):
                recommended_invalidations.append(row.get("step_id"))
        return {
            "impacted_step_ids": impacted_list,
            "impacted_partitions": sorted(impacted_partitions),
            "recommended_invalidations": recommended_invalidations,
        }

    async def invalidate_steps(
        self,
        plan_id: str,
        step_ids: List[str],
        reason: str,
        behavior: str = "SET_STATUS_STALE",
    ) -> Dict[str, Any]:
        target_ids = [str(s) for s in (step_ids or []) if s]
        if not target_ids:
            return {"count": 0}
        updated = 0
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            revision = await self._bump_revision(db, plan_id)
            for step_id in target_ids:
                row = await self._fetch_step(db, plan_id, step_id)
                if not row:
                    continue
                old = self._row_to_step(row)
                status = row["status"]
                run_metadata = _json_loads(row["run_metadata_json"], {})
                new_status = "STALE" if behavior == "SET_STATUS_STALE" else "PENDING"
                new_output_refs: List[Dict[str, Any]] = []
                changed_fields: List[str] = []
                if status in ("RUNNING", "CLAIMED"):
                    run_metadata = dict(run_metadata)
                    run_metadata["invalidate_on_complete"] = True
                    run_metadata["invalidate_reason"] = reason
                    run_metadata["invalidate_behavior"] = behavior
                    await db.execute(
                        "UPDATE plan_steps SET run_metadata_json=?, updated_at=? WHERE plan_id=? AND step_id=?",
                        (_json_dumps(run_metadata), utc_now(), plan_id, step_id),
                    )
                    changed_fields.append("run_metadata")
                    new = {**old, "run_metadata": run_metadata}
                else:
                    await db.execute(
                        "UPDATE plan_steps SET status=?, output_refs_json=?, error_json=?, claimed_by=?, updated_at=? "
                        "WHERE plan_id=? AND step_id=?",
                        (
                            new_status,
                            _json_dumps(new_output_refs),
                            _json_dumps({"invalidated": True, "reason": reason}),
                            None,
                            utc_now(),
                            plan_id,
                            step_id,
                        ),
                    )
                    changed_fields.extend(["status", "output_refs"])
                    new = {
                        **old,
                        "status": new_status,
                        "output_refs": new_output_refs,
                        "claimed_by": None,
                    }
                await self._record_change(db, plan_id, revision, step_id, changed_fields, old, new)
                updated += 1
            await db.commit()
        return {"count": updated, "revision": revision}

    def _row_to_patch(self, row: aiosqlite.Row) -> Dict[str, Any]:
        validation_ref = _json_loads(row["validation_report_ref_json"], None)
        return {
            "patch_id": row["patch_id"],
            "plan_id": row["plan_id"],
            "base_revision": int(row["base_revision"] or 0),
            "proposal_revision": int(row["proposal_revision"] or 0),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "created_by": _json_loads(row["created_by_json"], {}),
            "rationale": row["rationale"] or "",
            "linked_finding_ids": _json_loads(row["linked_finding_ids_json"], []),
            "status": row["status"],
            "validation_report_ref": validation_ref,
            "operations": _json_loads(row["operations_json"], []),
        }

    async def propose_patch(
        self,
        plan_id: str,
        base_revision: int,
        rationale: str,
        linked_finding_ids: List[str],
        operations: List[Dict[str, Any]],
        created_by: Optional[Dict[str, Any]] = None,
    ) -> str:
        patch_id = str(uuid.uuid4())
        created_at = utc_now()
        created_by = created_by or {"type": "planner", "id": "replan_patch"}
        status = "PROPOSED"
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            revision = await self._bump_revision(db, plan_id)
            await db.execute(
                "INSERT INTO plan_patches("
                "patch_id, plan_id, base_revision, proposal_revision, created_at, updated_at, created_by_json, "
                "rationale, linked_finding_ids_json, status, validation_report_ref_json, operations_json"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    patch_id,
                    plan_id,
                    int(base_revision),
                    int(revision),
                    created_at,
                    created_at,
                    _json_dumps(created_by),
                    rationale or "",
                    _json_dumps(_coerce_str_list(linked_finding_ids)),
                    status,
                    _json_dumps(None),
                    _json_dumps(operations or []),
                ),
            )
            await self._record_plan_change(
                db,
                plan_id,
                revision,
                "patch",
                patch_id,
                {
                    "action": "created",
                    "patch_id": patch_id,
                    "status": status,
                    "base_revision": int(base_revision),
                },
            )
            await db.commit()
        return patch_id

    async def get_patch(self, plan_id: str, patch_id: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM plan_patches WHERE plan_id=? AND patch_id=?",
                (plan_id, patch_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._row_to_patch(row) if row else None

    async def _update_patch_status(
        self,
        plan_id: str,
        patch_id: str,
        status: str,
        validation_report_ref: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        status = _normalize_enum(status, PATCH_STATUSES, "PROPOSED")
        updated_at = utc_now()
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                "SELECT * FROM plan_patches WHERE plan_id=? AND patch_id=?",
                (plan_id, patch_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if not row:
                await db.execute("ROLLBACK")
                return None
            revision = await self._bump_revision(db, plan_id)
            await db.execute(
                "UPDATE plan_patches SET status=?, validation_report_ref_json=?, updated_at=?, proposal_revision=? "
                "WHERE plan_id=? AND patch_id=?",
                (
                    status,
                    _json_dumps(validation_report_ref) if validation_report_ref is not None else row["validation_report_ref_json"],
                    updated_at,
                    revision,
                    plan_id,
                    patch_id,
                ),
            )
            await self._record_plan_change(
                db,
                plan_id,
                revision,
                "patch",
                patch_id,
                {
                    "action": "updated",
                    "patch_id": patch_id,
                    "status": status,
                    "reason": reason or "",
                },
            )
            await db.commit()
        return await self.get_patch(plan_id, patch_id)

    async def reject_patch(self, plan_id: str, patch_id: str, reason: str) -> Optional[Dict[str, Any]]:
        return await self._update_patch_status(plan_id, patch_id, "REJECTED", reason=reason)

    async def _resolve_bulk_match(
        self,
        db: aiosqlite.Connection,
        plan_id: str,
        match: Dict[str, Any],
    ) -> List[str]:
        tags = _coerce_str_list(match.get("tags"))
        partition_key = match.get("partition_key")
        status = match.get("status")
        params: List[Any] = [plan_id]
        if tags:
            tag_placeholders = ",".join("?" for _ in tags)
            clauses = ["plan_steps.plan_id=?", f"json_each.value IN ({tag_placeholders})"]
            params.extend(tags)
            if partition_key is not None:
                clauses.append("plan_steps.partition_key=?")
                params.append(partition_key)
            if status:
                clauses.append("plan_steps.status=?")
                params.append(status)
            sql = (
                "SELECT DISTINCT plan_steps.step_id as step_id "
                "FROM plan_steps, json_each(plan_steps.tags_json) "
                f"WHERE {' AND '.join(clauses)}"
            )
        else:
            clauses = ["plan_id=?"]
            if partition_key is not None:
                clauses.append("partition_key=?")
                params.append(partition_key)
            if status:
                clauses.append("status=?")
                params.append(status)
            sql = f"SELECT step_id FROM plan_steps WHERE {' AND '.join(clauses)}"
        cursor = await db.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        await cursor.close()
        return [row["step_id"] for row in rows]

    async def validate_patch(
        self,
        plan_id: str,
        patch_id: str,
        validation_report_ref: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        patch = await self.get_patch(plan_id, patch_id)
        if not patch:
            return {"ok": False, "errors": ["patch_not_found"], "warnings": [], "would_create_cycle": False}
        ops = patch.get("operations") or []
        errors: List[str] = []
        warnings: List[str] = []
        impacted: set[str] = set()
        new_edges: List[tuple[str, str]] = []
        new_edges_by_prereq: Dict[str, set[str]] = {}
        removed_edges_by_prereq: Dict[str, set[str]] = {}
        prereq_updates: Dict[str, List[str]] = {}

        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row

            async def ensure_existing(step_ids: List[str]) -> Dict[str, Dict[str, Any]]:
                if not step_ids:
                    return {}
                result = await self.get_steps(
                    plan_id,
                    step_ids,
                    fields=["step_id", "prereq_step_ids", "status"],
                )
                return {row["step_id"]: row for row in result.get("steps") or []}

            candidate_ids: set[str] = set()
            add_steps: List[Dict[str, Any]] = []
            bulk_ops: List[Dict[str, Any]] = []

            for op in ops:
                op_type = str(op.get("op") or op.get("type") or "").upper()
                if op_type == "ADD_STEP":
                    step = op.get("step") or {}
                    add_steps.append(step)
                    step_id = str(step.get("step_id") or "")
                    if step_id:
                        candidate_ids.add(step_id)
                elif op_type in ("UPDATE_STEP", "SET_PREREQS", "CANCEL_STEP", "INVALIDATE_STEP_OUTPUTS"):
                    step_id = str(op.get("step_id") or "")
                    if step_id:
                        candidate_ids.add(step_id)
                elif op_type == "BULK_OP":
                    bulk_ops.append(op)
                elif op_type in ("ADD_PARTITION", "UPDATE_PARTITION"):
                    continue
                else:
                    errors.append(f"unsupported_op:{op_type or 'UNKNOWN'}")

            existing_map = await ensure_existing(list(candidate_ids))

            for step in add_steps:
                step_id = str(step.get("step_id") or "")
                if not step_id:
                    errors.append("add_step_missing_step_id")
                    continue
                if step_id in existing_map:
                    errors.append(f"add_step_exists:{step_id}")
                    continue
                impacted.add(step_id)
                prereqs = _coerce_str_list(step.get("prereq_step_ids") or step.get("depends_on"))
                prereq_updates[step_id] = prereqs
                for pid in prereqs:
                    new_edges.append((step_id, pid))
                    new_edges_by_prereq.setdefault(pid, set()).add(step_id)

            for op in ops:
                op_type = str(op.get("op") or op.get("type") or "").upper()
                if op_type in ("UPDATE_STEP", "SET_PREREQS", "CANCEL_STEP", "INVALIDATE_STEP_OUTPUTS"):
                    step_id = str(op.get("step_id") or "")
                    if not step_id:
                        errors.append("missing_step_id")
                        continue
                    if step_id not in existing_map and step_id not in prereq_updates:
                        errors.append(f"unknown_step:{step_id}")
                        continue
                    impacted.add(step_id)
                    if op_type == "UPDATE_STEP":
                        patch_fields = op.get("patch") or {}
                        if "prereq_step_ids" in patch_fields:
                            prereqs = _coerce_str_list(patch_fields.get("prereq_step_ids"))
                            prereq_updates[step_id] = prereqs
                    if op_type == "SET_PREREQS":
                        prereqs = _coerce_str_list(op.get("prereq_step_ids"))
                        prereq_updates[step_id] = prereqs

            if prereq_updates:
                existing_prereqs = {
                    sid: existing_map.get(sid, {}).get("prereq_step_ids", []) for sid in prereq_updates.keys()
                }
                for step_id, prereqs in prereq_updates.items():
                    old_prereqs = set(existing_prereqs.get(step_id) or [])
                    new_prereqs = set(prereqs or [])
                    removed = old_prereqs - new_prereqs
                    for pid in removed:
                        removed_edges_by_prereq.setdefault(pid, set()).add(step_id)
                    for pid in new_prereqs:
                        new_edges.append((step_id, pid))
                        new_edges_by_prereq.setdefault(pid, set()).add(step_id)

            for bulk in bulk_ops:
                match = bulk.get("match") or {}
                action = bulk.get("action") or {}
                op_type = str(action.get("op") or action.get("type") or "").upper()
                if not op_type:
                    errors.append("bulk_missing_action")
                    continue
                targets = await self._resolve_bulk_match(db, plan_id, match)
                if not targets:
                    warnings.append("bulk_no_targets")
                for step_id in targets:
                    impacted.add(step_id)
                    if op_type in ("SET_PREREQS", "UPDATE_STEP"):
                        prereqs = _coerce_str_list(
                            action.get("prereq_step_ids") or action.get("patch", {}).get("prereq_step_ids")
                        )
                        prereq_updates[step_id] = prereqs

        would_create_cycle = False
        if new_edges:
            dependents_cache: Dict[str, List[str]] = {}

            async def _dependents_for(node: str) -> List[str]:
                if node in dependents_cache:
                    return dependents_cache[node]
                deps = await self.get_dependents(plan_id, node)
                dependents_cache[node] = deps
                return deps

            async def _path_exists(start: str, target: str) -> bool:
                stack = [start]
                visited: set[str] = set()
                while stack:
                    current = stack.pop()
                    if current == target:
                        return True
                    if current in visited:
                        continue
                    visited.add(current)
                    deps = await _dependents_for(current)
                    removed = removed_edges_by_prereq.get(current, set())
                    for dep in deps:
                        if dep in removed:
                            continue
                        if dep not in visited:
                            stack.append(dep)
                    for dep in new_edges_by_prereq.get(current, set()):
                        if dep not in visited:
                            stack.append(dep)
                return False

            for step_id, prereq_id in new_edges:
                if await _path_exists(step_id, prereq_id):
                    would_create_cycle = True
                    errors.append(f"cycle_detected:{step_id}->{prereq_id}")
                    break

        ok = not errors
        if ok:
            await self._update_patch_status(plan_id, patch_id, "VALIDATED", validation_report_ref=validation_report_ref)
        else:
            await self._update_patch_status(plan_id, patch_id, "REJECTED", validation_report_ref=validation_report_ref)
        return {
            "ok": ok,
            "errors": errors,
            "warnings": warnings,
            "would_create_cycle": would_create_cycle,
            "impacted_step_ids": sorted(impacted),
            "preview_revision_delta": len(ops),
        }

    async def apply_patch(self, plan_id: str, patch_id: str, approver_id: str) -> Dict[str, Any]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                "SELECT revision, partitions_json FROM plans WHERE plan_id=?",
                (plan_id,),
            )
            plan_row = await cursor.fetchone()
            await cursor.close()
            current_revision = int(plan_row["revision"] or 0) if plan_row else 0
            cursor = await db.execute(
                "SELECT * FROM plan_patches WHERE plan_id=? AND patch_id=?",
                (plan_id, patch_id),
            )
            patch_row = await cursor.fetchone()
            await cursor.close()
            if not patch_row:
                await db.execute("ROLLBACK")
                return {"conflict": True, "current_revision": current_revision}
            patch = self._row_to_patch(patch_row)
            if patch.get("status") != "VALIDATED":
                await db.execute("ROLLBACK")
                return {"conflict": True, "current_revision": current_revision}
            base_revision = int(patch.get("base_revision") or 0)
            if current_revision != base_revision:
                cursor = await db.execute(
                    "SELECT COUNT(*) as cnt FROM plan_step_changes WHERE plan_id=? AND revision>?",
                    (plan_id, base_revision),
                )
                row = await cursor.fetchone()
                await cursor.close()
                step_changes = int(row["cnt"] or 0) if row else 0
                if step_changes > 0:
                    await db.execute(
                        "UPDATE plan_patches SET status=?, updated_at=? WHERE plan_id=? AND patch_id=?",
                        ("CONFLICT", utc_now(), plan_id, patch_id),
                    )
                    await db.commit()
                    return {"conflict": True, "current_revision": current_revision}
            revision = await self._bump_revision(db, plan_id)

            async def _expand_ops(operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                expanded: List[Dict[str, Any]] = []
                for op in operations:
                    op_type = str(op.get("op") or op.get("type") or "").upper()
                    if op_type == "BULK_OP":
                        match = op.get("match") or {}
                        action = op.get("action") or {}
                        action_type = str(action.get("op") or action.get("type") or "").upper()
                        if not action_type:
                            continue
                        targets = await self._resolve_bulk_match(db, plan_id, match)
                        for step_id in targets:
                            expanded.append({**action, "op": action_type, "step_id": step_id})
                    else:
                        expanded.append(op)
                return expanded

            operations = await _expand_ops(patch.get("operations") or [])
            partitions = _json_loads(plan_row["partitions_json"], []) if plan_row else []

            for op in operations:
                op_type = str(op.get("op") or op.get("type") or "").upper()
                if op_type == "ADD_STEP":
                    step = op.get("step") or {}
                    step_id = str(step.get("step_id") or uuid.uuid4())
                    title = str(step.get("title") or step.get("name") or f"Step {step_id}")
                    description = str(step.get("description") or "")
                    step_type = step.get("step_type")
                    if step_type is None and "type" in step:
                        step_type = step.get("type")
                    step_type = str(step_type).upper() if step_type else None
                    status = str(step.get("status") or "PENDING").upper()
                    if status not in STEP_STATUSES:
                        status = "PENDING"
                    prereqs = _coerce_str_list(step.get("prereq_step_ids") or step.get("depends_on"))
                    tags = _coerce_str_list(step.get("tags"))
                    priority = int(step.get("priority") or 0)
                    cost_hint = step.get("cost_hint") or {}
                    partition_key = step.get("partition_key")
                    attempt = int(step.get("attempt") or 0)
                    max_retries = int(step.get("max_retries") or 1)
                    created_by = step.get("created_by") or {"type": "planner", "id": "patch"}
                    claimed_by = step.get("claimed_by")
                    run_metadata = step.get("run_metadata") or {}
                    input_refs = step.get("input_refs") or []
                    output_refs = step.get("output_refs") or []
                    notes = step.get("notes")
                    created_at = utc_now()
                    await db.execute(
                        "INSERT INTO plan_steps(step_id, plan_id, title, description, step_type, status, prereq_step_ids_json, "
                        "tags_json, priority, cost_hint_json, partition_key, attempt, max_retries, created_by_json, "
                        "claimed_by, run_metadata_json, input_refs_json, output_refs_json, notes, created_at, updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            step_id,
                            plan_id,
                            title,
                            description,
                            step_type,
                            status,
                            _json_dumps(prereqs),
                            _json_dumps(tags),
                            priority,
                            _json_dumps(cost_hint),
                            partition_key,
                            attempt,
                            max_retries,
                            _json_dumps(created_by),
                            claimed_by,
                            _json_dumps(run_metadata),
                            _json_dumps(input_refs),
                            _json_dumps(output_refs),
                            notes,
                            created_at,
                            created_at,
                        ),
                    )
                    await self._record_change(
                        db,
                        plan_id,
                        revision,
                        step_id,
                        ["insert"],
                        {},
                        {
                            "status": status,
                            "title": title,
                            "description": description,
                            "prereq_step_ids": prereqs,
                            "tags": tags,
                            "priority": priority,
                            "partition_key": partition_key,
                        },
                    )
                elif op_type in ("UPDATE_STEP", "SET_PREREQS", "CANCEL_STEP", "INVALIDATE_STEP_OUTPUTS"):
                    step_id = str(op.get("step_id") or "")
                    if not step_id:
                        continue
                    row = await self._fetch_step(db, plan_id, step_id)
                    if not row:
                        continue
                    old = self._row_to_step(row)
                    if op_type == "UPDATE_STEP":
                        patch_fields = op.get("patch") or {}
                        updates = {k: v for k, v in patch_fields.items()}
                        if updates:
                            assignments: List[str] = []
                            params: List[Any] = []
                            for key, value in updates.items():
                                column = key
                                stored: Any = value
                                if key in (
                                    "prereq_step_ids",
                                    "tags",
                                    "cost_hint",
                                    "created_by",
                                    "run_metadata",
                                    "input_refs",
                                    "output_refs",
                                ):
                                    column = f"{key}_json"
                                    stored = _json_dumps(value)
                                elif key == "step_type":
                                    stored = str(value).upper() if value else None
                                elif key == "status":
                                    stored = str(value).upper()
                                    if stored not in STEP_STATUSES:
                                        stored = "PENDING"
                                assignments.append(f"{column}=?")
                                params.append(stored)
                            params.extend([utc_now(), plan_id, step_id])
                            sql = f"UPDATE plan_steps SET {', '.join(assignments)}, updated_at=? WHERE plan_id=? AND step_id=?"
                            await db.execute(sql, tuple(params))
                            new_row = await self._fetch_step(db, plan_id, step_id)
                            new = self._row_to_step(new_row) if new_row else {}
                            await self._record_change(
                                db,
                                plan_id,
                                revision,
                                step_id,
                                list(updates.keys()),
                                old,
                                new,
                            )
                    elif op_type == "SET_PREREQS":
                        prereqs = _coerce_str_list(op.get("prereq_step_ids"))
                        await db.execute(
                            "UPDATE plan_steps SET prereq_step_ids_json=?, updated_at=? WHERE plan_id=? AND step_id=?",
                            (_json_dumps(prereqs), utc_now(), plan_id, step_id),
                        )
                        new = {**old, "prereq_step_ids": prereqs}
                        await self._record_change(
                            db,
                            plan_id,
                            revision,
                            step_id,
                            ["prereq_step_ids"],
                            old,
                            new,
                        )
                    elif op_type == "CANCEL_STEP":
                        reason = str(op.get("reason") or "canceled_by_patch")
                        run_metadata = dict(old.get("run_metadata") or {})
                        run_metadata["canceled_reason"] = reason
                        await db.execute(
                            "UPDATE plan_steps SET status=?, output_refs_json=?, error_json=?, claimed_by=?, run_metadata_json=?, updated_at=? "
                            "WHERE plan_id=? AND step_id=?",
                            (
                                "CANCELED",
                                _json_dumps([]),
                                _json_dumps({"canceled": True, "reason": reason}),
                                None,
                                _json_dumps(run_metadata),
                                utc_now(),
                                plan_id,
                                step_id,
                            ),
                        )
                        new = {**old, "status": "CANCELED", "output_refs": [], "claimed_by": None, "run_metadata": run_metadata}
                        await self._record_change(
                            db,
                            plan_id,
                            revision,
                            step_id,
                            ["status", "output_refs", "run_metadata"],
                            old,
                            new,
                        )
                    elif op_type == "INVALIDATE_STEP_OUTPUTS":
                        reason = str(op.get("reason") or "invalidated_by_patch")
                        behavior = str(op.get("behavior") or "SET_STATUS_STALE").upper()
                        status_now = old.get("status")
                        if status_now in ("RUNNING", "CLAIMED"):
                            run_metadata = dict(old.get("run_metadata") or {})
                            run_metadata["invalidate_on_complete"] = True
                            run_metadata["invalidate_reason"] = reason
                            run_metadata["invalidate_behavior"] = behavior
                            await db.execute(
                                "UPDATE plan_steps SET run_metadata_json=?, updated_at=? WHERE plan_id=? AND step_id=?",
                                (_json_dumps(run_metadata), utc_now(), plan_id, step_id),
                            )
                            new = {**old, "run_metadata": run_metadata}
                            await self._record_change(
                                db,
                                plan_id,
                                revision,
                                step_id,
                                ["run_metadata"],
                                old,
                                new,
                            )
                        else:
                            new_status = "STALE" if behavior == "SET_STATUS_STALE" else "PENDING"
                            await db.execute(
                                "UPDATE plan_steps SET status=?, output_refs_json=?, error_json=?, claimed_by=?, updated_at=? "
                                "WHERE plan_id=? AND step_id=?",
                                (
                                    new_status,
                                    _json_dumps([]),
                                    _json_dumps({"invalidated": True, "reason": reason}),
                                    None,
                                    utc_now(),
                                    plan_id,
                                    step_id,
                                ),
                            )
                            new = {**old, "status": new_status, "output_refs": [], "claimed_by": None}
                            await self._record_change(
                                db,
                                plan_id,
                                revision,
                                step_id,
                                ["status", "output_refs"],
                                old,
                                new,
                            )
                elif op_type == "ADD_PARTITION":
                    partition = op.get("partition") or {}
                    if partition:
                        partitions.append(partition)
                elif op_type == "UPDATE_PARTITION":
                    partition = op.get("partition") or {}
                    key = partition.get("partition_key")
                    if key is None:
                        continue
                    updated = False
                    for idx, existing in enumerate(partitions):
                        if existing.get("partition_key") == key:
                            partitions[idx] = {**existing, **partition}
                            updated = True
                            break
                    if not updated:
                        partitions.append(partition)

            await db.execute(
                "UPDATE plan_patches SET status=?, updated_at=? WHERE plan_id=? AND patch_id=?",
                ("APPLIED", utc_now(), plan_id, patch_id),
            )
            await db.execute(
                "UPDATE plans SET partitions_json=?, updated_at=? WHERE plan_id=?",
                (_json_dumps(partitions), utc_now(), plan_id),
            )
            await self._record_plan_change(
                db,
                plan_id,
                revision,
                "patch",
                patch_id,
                {
                    "action": "applied",
                    "patch_id": patch_id,
                    "status": "APPLIED",
                    "approver_id": approver_id,
                },
            )
            await db.commit()
        return {"new_revision": revision}

    async def log_event(self, plan_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO plan_events(plan_id, event_type, payload_json, created_at) VALUES (?,?,?,?)",
                (plan_id, event_type, _json_dumps(payload), utc_now()),
            )
            await db.commit()
