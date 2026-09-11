from __future__ import annotations
import json
from pathlib import Path
from .schema import ReactionContext, Participant, Conditions, Procedure


def _load_mapping(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML input requires: pip install pyyaml") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def build_context(data: dict) -> ReactionContext:
    participants = [Participant(**p) for p in data.get("participants", [])]
    conditions = Conditions(**data.get("conditions", {}))
    procedure = Procedure(**data.get("procedure", {}))
    return ReactionContext(
        experiment_id=str(data.get("experiment_id", "experiment")),
        participants=participants,
        conditions=conditions,
        procedure=procedure,
        metadata=data.get("metadata", {}) or {},
    )


def load_context(path: str | Path) -> ReactionContext:
    return build_context(_load_mapping(path))


def context_to_query(ctx: ReactionContext, visual_summary: dict | None = None) -> str:
    names = ", ".join(p.name for p in ctx.participants if p.name)
    roles = ", ".join(f"{p.name}:{p.role}" for p in ctx.participants if p.name)
    cond = []
    c = ctx.conditions
    if c.temperature_c is not None: cond.append(f"temperature {c.temperature_c} C")
    if c.pressure_kpa is not None: cond.append(f"pressure {c.pressure_kpa} kPa")
    if c.ph is not None: cond.append(f"pH {c.ph}")
    if c.stirring_rpm is not None: cond.append(f"stirring {c.stirring_rpm} rpm")
    if c.atmosphere: cond.append(f"atmosphere {c.atmosphere}")
    visual = ""
    if visual_summary:
        keys = ["color_change_rate", "motion_activity", "turbidity_proxy", "solid_proxy", "phase_boundary_proxy", "stability"]
        visual = "; ".join(f"{k}={visual_summary.get(k)}" for k in keys if k in visual_summary)
    return (
        f"experiment type {ctx.procedure.experiment_type}; current step {ctx.procedure.current_step}; "
        f"participants {names}; roles {roles}; conditions {'; '.join(cond)}; "
        f"objective {ctx.procedure.objective}; visual evidence {visual}"
    )
