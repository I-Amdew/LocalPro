LocalPro Planning Overhaul (v2)

Overview
- Plan reasoning mode controls how exhaustive the plan must be. "extensive" interprets words like "all/every/each" literally and expands partitions accordingly.
- Planning mode controls the 2-layer planner. "normal" emits a compact plan; "extensive" scaffolds + expands into many concrete steps.
- Reasoning level is an integer that maps to plan granularity (step count / partition size), not "think harder."

PlanStore Views
- Overview: aggregated counts, ready/running/failed, partitions, and recent events for lightweight monitoring.
- Diff: incremental change stream since a revision for executor sync.
- list_steps/get_steps: paginated browsing and on-demand detail fetches to avoid loading huge plans.

Executor Scheduling Policy
- Uses overview + diff + paginated step lists; never loads full plan into context.
- Schedules by priority plus a "unblocks many dependents" heuristic.
- Uses ResourceManager reserve/release with headroom thresholds to avoid overcommit.
- Supports multiple concurrent backend instances chosen by capability requirements.

Prereq Resolution Pass
- Scaffold and expansion steps can emit notes in the form:
  - {"prereq_titles":[...]} or {"prereq_tags":[...]} or {"prereq_step_ids":[...]}
- A resolve step converts notes to concrete prereq ids and clears notes in a second pass.

Drafting / Narrowing Pipeline
- Draft steps gather upstream output refs and produce a draftbook artifact that fits the final synthesizer budget.
- Finalize steps read the draftbook ref, not raw outputs.

Request Mechanism
- Planners create requests in plan_requests.
- Executor fulfills requests into ArtifactStore and attaches result refs back to the request.

Self-Correcting Plans (Findings + Patches)
- Execution and verification steps can raise Findings with severity, category, evidence refs, and suggested actions.
- Executor triages Findings: INFO/WARN log only; ERROR/CRITICAL trigger impact analysis and containment.
- Impact analysis is code-driven (DAG queries) and marks affected steps STALE while allowing unaffected partitions to continue.
- Replan workflow emits REPLAN_PATCH and PATCH_VERIFY steps that propose and validate PlanPatch operations.
- PlanPatch applies atomically with revision checks; conflicts raise a Finding and reschedule replanning.
- Bulk ops allow partition/tag invalidation without enumerating large step sets.

How to Enable Extensive Modes
- In /api/run payloads:
  - plan_reasoning_mode: "normal" | "extensive"
  - planning_mode: "normal" | "extensive"
  - reasoning_level: integer granularity scale
- Defaults can be set via env:
  - PLAN_REASONING_MODE_DEFAULT, PLANNING_MODE_DEFAULT, REASONING_LEVEL_DEFAULT
