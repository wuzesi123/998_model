from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Participant:
    name: str
    role: str = "unspecified"
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    amount: Optional[float] = None
    amount_unit: Optional[str] = None
    concentration: Optional[float] = None
    concentration_unit: Optional[str] = None
    notes: str = ""


@dataclass
class Conditions:
    temperature_c: Optional[float] = None
    pressure_kpa: Optional[float] = None
    ph: Optional[float] = None
    stirring_rpm: Optional[float] = None
    atmosphere: Optional[str] = None
    illumination: Optional[str] = None
    details: str = ""


@dataclass
class Procedure:
    experiment_type: str = "unknown"
    current_step: str = "unknown"
    objective: str = ""
    notes: str = ""


@dataclass
class ReactionContext:
    experiment_id: str
    participants: list[Participant] = field(default_factory=list)
    conditions: Conditions = field(default_factory=Conditions)
    procedure: Procedure = field(default_factory=Procedure)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedDocument:
    doc_id: str
    source: str
    score: float
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticReport:
    """Structured two-layer report.

    The three primary semantic axes are deliberately separated:

    - process_type: what kind of experiment/process the evidence supports
    - temporal_state: where the currently observed visual dynamics sit
    - active_event: what appears to be happening *now*, if anything

    Backward-compatible properties (current_state / visual_state /
    chemical_interpretation) are kept so the rest of the V1 project does not
    need to be edited when these three files are dropped in.
    """

    experiment_id: str
    process_type: str
    temporal_state: str
    active_event: str
    confidence: float
    visual_evidence: dict[str, Any]
    hypotheses: list[dict[str, Any]]
    retrieved_knowledge: list[dict[str, Any]]
    evidence: dict[str, Any]
    expected_next_events: list[str]
    uncertainty_notes: list[str]
    model_notes: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Backward-compatible aliases for the original V1 code/tests.
    # ------------------------------------------------------------------
    @property
    def current_state(self) -> str:
        # Original V1 semantics: top ranked chemical hypothesis label.
        if self.hypotheses:
            return str(self.hypotheses[0].get("label", self.process_type))
        return self.process_type

    @property
    def visual_state(self) -> str:
        # Preserve the old uppercase-style labels for legacy consumers.
        mapping = {
            "plateau_or_stable": "VISUAL_PLATEAU_OR_STABLE",
            "phase_formation_or_separation_visible": "VISIBLE_PHASE_BOUNDARY",
            "heterogeneous_or_solid_present": "VISIBLE_HETEROGENEITY_OR_SOLID",
            "active_change": "ACTIVE_VISUAL_CHANGE",
            "low_to_moderate_change": "LOW_TO_MODERATE_VISUAL_CHANGE",
        }
        return mapping.get(self.temporal_state, self.temporal_state)

    @property
    def chemical_interpretation(self) -> str:
        if self.hypotheses:
            return str(self.hypotheses[0].get("label", self.process_type))
        return self.process_type

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        # Include aliases in JSON output for compatibility with consumers of
        # demo_result.json from V1. New code should prefer the primary fields.
        data["current_state"] = self.current_state
        data["visual_state"] = self.visual_state
        data["chemical_interpretation"] = self.chemical_interpretation
        return data
