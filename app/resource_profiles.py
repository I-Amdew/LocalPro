import json
import time
from typing import Any, Dict, List, Optional

import aiosqlite


PROFILE_VERSION = 1


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ModelResourceProfileStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def upsert(self, profile: Dict[str, Any], ttl_seconds: int) -> None:
        backend_id = profile.get("backend_id") or ""
        model_key = profile.get("model_key") or ""
        config_signature = profile.get("config_signature") or ""
        updated_at = profile.get("profiled_at") or utc_now()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO model_resource_profiles("
                "backend_id, model_key, config_signature, profile_json, updated_at, ttl_seconds"
                ") VALUES (?,?,?,?,?,?)",
                (
                    backend_id,
                    model_key,
                    config_signature,
                    json.dumps(profile, ensure_ascii=True),
                    updated_at,
                    ttl_seconds,
                ),
            )
            await db.commit()

    async def get(self, backend_id: str, model_key: str, config_signature: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT profile_json FROM model_resource_profiles "
                "WHERE backend_id=? AND model_key=? AND config_signature=?",
                (backend_id, model_key, config_signature),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if not row:
            return None
        return json.loads(row["profile_json"] or "{}")

    async def get_latest(self, backend_id: str, model_key: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT profile_json FROM model_resource_profiles "
                "WHERE backend_id=? AND model_key=? "
                "ORDER BY updated_at DESC LIMIT 1",
                (backend_id, model_key),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if not row:
            return None
        return json.loads(row["profile_json"] or "{}")

    async def list_all(self) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT profile_json FROM model_resource_profiles ORDER BY updated_at DESC"
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [json.loads(row["profile_json"] or "{}") for row in rows]
