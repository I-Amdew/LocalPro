import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiosqlite


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ExecutorStateStore:
    """Persist small executor state per plan."""

    def __init__(self, path: str):
        self.path = path

    async def get(self, plan_id: str) -> Dict[str, Any]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT state_json FROM executor_states WHERE plan_id=?",
                (plan_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if not row:
            return {}
        try:
            return json.loads(row["state_json"] or "{}")
        except Exception:
            return {}

    async def set(self, plan_id: str, state: Dict[str, Any]) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO executor_states(plan_id, state_json, updated_at) VALUES (?,?,?)",
                (plan_id, json.dumps(state, ensure_ascii=True), utc_now()),
            )
            await db.commit()
