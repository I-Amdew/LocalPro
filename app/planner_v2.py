import json
import uuid
from typing import Any, Dict, List, Optional

from .plan_store import PlanStore


def normalize_reasoning_level(value: Optional[int]) -> int:
    try:
        level = int(value) if value is not None else 2
    except Exception:
        level = 2
    return max(1, min(level, 10))


def _expansion_count(level: int, extensive: bool, override: Optional[int] = None) -> int:
    if override is not None:
        return max(1, int(override))
    if extensive:
        return min(10000, max(50, level * 250))
    return max(5, level * 10)


async def scaffold_plan(
    plan_store: PlanStore,
    *,
    question: str,
    reasoning_mode: str,
    planning_mode: str,
    reasoning_level: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> str:
    overrides = overrides or {}
    level = normalize_reasoning_level(reasoning_level)
    metadata = {
        "query": question,
        "reasoning_mode": reasoning_mode,
        "planning_mode": planning_mode,
        "reasoning_level": level,
    }
    meta_overrides = overrides.get("metadata")
    if isinstance(meta_overrides, dict):
        metadata.update(meta_overrides)
    plan_id = await plan_store.create(metadata)
    steps: List[Dict[str, Any]] = []
    expand_needed = planning_mode == "extensive" or reasoning_mode == "extensive"
    expansion_step_id = str(uuid.uuid4())
    if expand_needed:
        steps.append(
            {
                "step_id": expansion_step_id,
                "title": "Expand partitions",
                "description": "Enumerate partitions for exhaustive coverage.",
                "status": "READY",
                "tags": ["phase:expand"],
                "priority": 10,
                "cost_hint": {
                    "required_capabilities": ["structured_output"],
                    "preferred_objective": "latency",
                    "estimated_tokens": 400,
                },
                "run_metadata": {
                    "expansion_count": _expansion_count(
                        level, True, overrides.get("expansion_count")
                    )
                },
                "created_by": {"type": "planner", "id": "scaffold"},
            }
        )
    resolve_step_id = str(uuid.uuid4())
    steps.append(
        {
            "step_id": resolve_step_id,
            "title": "Resolve prerequisites",
            "description": "Map placeholder prereqs to concrete step ids.",
            "status": "PENDING",
            "tags": ["phase:resolve"],
            "priority": 5,
            "prereq_step_ids": [expansion_step_id] if expand_needed else [],
            "cost_hint": {
                "required_capabilities": [],
                "preferred_objective": "latency",
                "estimated_tokens": 200,
            },
            "created_by": {"type": "planner", "id": "scaffold"},
        }
    )
    if not expand_needed:
        exec_step_id = str(uuid.uuid4())
        steps.append(
            {
                "step_id": exec_step_id,
                "title": "Execute core work",
                "description": "Carry out the main task.",
                "status": "READY",
                "tags": ["phase:execute"],
                "priority": 8,
                "cost_hint": {
                    "required_capabilities": ["tool_use"],
                    "preferred_objective": "quality",
                    "estimated_tokens": 800,
                },
                "created_by": {"type": "planner", "id": "scaffold"},
            }
        )
        steps.append(
            {
                "step_id": f"verify-{exec_step_id}",
                "title": "Verify core work",
                "description": "Verify outputs for core work.",
                "step_type": "VERIFIER",
                "status": "PENDING",
                "tags": ["phase:verify", f"verifies:{exec_step_id}"],
                "priority": 7,
                "prereq_step_ids": [exec_step_id],
                "run_metadata": {"verify_step_id": exec_step_id, "allow_fallback": True},
                "cost_hint": {
                    "required_capabilities": ["structured_output"],
                    "preferred_objective": "best_quality",
                    "estimated_tokens": 200,
                },
                "created_by": {"type": "planner", "id": "scaffold"},
            }
        )
    steps.append(
        {
            "step_id": str(uuid.uuid4()),
            "title": "Draftbook",
            "description": "Condense step outputs into a final draftbook.",
            "status": "PENDING",
            "tags": ["phase:draft"],
            "priority": 3,
            "notes": json.dumps({"prereq_tags": ["phase:execute"]}),
            "cost_hint": {
                "required_capabilities": [],
                "preferred_objective": "latency",
                "estimated_tokens": 600,
            },
            "created_by": {"type": "planner", "id": "scaffold"},
        }
    )
    steps.append(
        {
            "step_id": str(uuid.uuid4()),
            "title": "Finalize",
            "description": "Synthesize the final response from the draftbook.",
            "status": "PENDING",
            "tags": ["phase:finalize"],
            "priority": 1,
            "notes": json.dumps({"prereq_titles": ["Draftbook"]}),
            "cost_hint": {
                "required_capabilities": [],
                "preferred_objective": "quality",
                "estimated_tokens": 800,
            },
            "created_by": {"type": "planner", "id": "scaffold"},
        }
    )
    await plan_store.add_steps(plan_id, steps)
    return plan_id
