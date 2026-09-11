from __future__ import annotations

from typing import Any

from ..context.schema import ReactionContext, RetrievedDocument, SemanticReport


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _contains(text: str | None, *words: str) -> bool:
    s = (text or "").lower()
    return any(w.lower() in s for w in words)


def _trend(v: dict[str, Any], name: str) -> dict[str, float]:
    """Read a trend record while remaining compatible with old summaries."""
    raw = (v.get("trends") or {}).get(name) or {}
    return {
        "early_mean": float(raw.get("early_mean", 0.0) or 0.0),
        "late_mean": float(raw.get("late_mean", 0.0) or 0.0),
        "slope": float(raw.get("slope", 0.0) or 0.0),
        "recent_slope": float(raw.get("recent_slope", 0.0) or 0.0),
        "late_to_early_ratio": float(
            raw.get("late_to_early_ratio", 1.0) or 1.0
        ),
        "delta": float(raw.get("delta", 0.0) or 0.0),
    }


def _trend_strength(tr: dict[str, float], scale: float = 20.0) -> float:
    """Map a positive trend into a bounded support score.

    This is intentionally conservative: trend-derived scores support a
    hypothesis, but never identify a compound or reaction on their own.
    """
    delta = max(0.0, tr.get("delta", 0.0))
    slope = max(0.0, tr.get("slope", 0.0))
    ratio = max(1.0, tr.get("late_to_early_ratio", 1.0))
    return _clip(delta * scale + slope * scale * 2.0 + (ratio - 1.0) * 0.08)


def _visual_state(v: dict[str, Any]) -> str:
    stable = _clip(v.get("stability", 0.0))
    motion = _clip(v.get("motion_activity", 0.0))
    solid = _clip(v.get("solid_proxy", 0.0))
    boundary = _clip(v.get("phase_boundary_proxy", 0.0))
    rate = abs(float(v.get("color_change_rate", 0.0) or 0.0))

    solid_tr = _trend(v, "solid")
    turb_tr = _trend(v, "turbidity")
    boundary_tr = _trend(v, "phase_boundary")

    # Current dynamics take precedence over absolute proxy thresholds.
    recent_active = (
        abs(solid_tr["recent_slope"]) > 3e-4
        or abs(turb_tr["recent_slope"]) > 3e-4
        or abs(boundary_tr["recent_slope"]) > 2e-3
        or rate > 0.015
        or motion > 0.12
    )

    if stable > 0.78 and motion < 0.08 and not recent_active:
        return "plateau_or_stable"
    if boundary > 0.5:
        return "phase_formation_or_separation_visible"
    if solid > 0.35:
        return "heterogeneous_or_solid_present"
    if recent_active:
        return "active_change"
    return "low_to_moderate_change"


def _evidence_item(source: str, kind: str, text: str) -> dict[str, str]:
    return {"source": source, "type": kind, "text": text}


def _build_reasons(
    *,
    visual_support: list[str] | None = None,
    context_support: list[str] | None = None,
    rag_support: list[str] | None = None,
    contradictions: list[str] | None = None,
    missing: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build attributable evidence.

    Crucially, retrieval is never relabelled as explicit context. The output
    preserves where every reason came from.
    """
    out: list[dict[str, str]] = []
    for text in visual_support or []:
        out.append(_evidence_item("visual", "support", text))
    for text in context_support or []:
        out.append(_evidence_item("context", "support", text))
    for text in rag_support or []:
        out.append(_evidence_item("rag", "support", text))
    for text in contradictions or []:
        out.append(_evidence_item("visual", "contradiction", text))
    for text in missing or []:
        out.append(_evidence_item("system", "missing_evidence", text))
    return out


def _process_type_from_label(label: str) -> str:
    mapping = {
        "crystallization_or_nucleation": "crystallization",
        "precipitation_or_new_solid": "precipitation",
        "phase_separation_or_settling": "phase_separation_or_settling",
        "dissolution_or_homogenization": "dissolution_or_homogenization",
        "color_or_conversion_dynamics": "color_or_conversion_dynamics",
    }
    return mapping.get(label, label)


class SemanticReasoner:
    """Evidence-attributed two-layer chemical semantic reasoner.

    Visual evidence answers "what is observable?". Reaction context and RAG
    help answer "what could that observation mean chemically?". Retrieved text
    can support plausibility but is not allowed to masquerade as explicit
    experiment context or override contradictory observations.
    """

    def infer(
        self,
        ctx: ReactionContext,
        visual: dict,
        docs: list[RetrievedDocument],
    ) -> SemanticReport:
        v = visual["summary"] if "summary" in visual else visual

        experiment_type = (ctx.procedure.experiment_type or "").strip()
        current_step = (ctx.procedure.current_step or "").strip()
        proc = f"{experiment_type} {current_step}".lower()

        rag_text = " ".join(
            f"{d.title} {d.text}".lower() for d in docs[:5]
        )

        change = min(
            1.0,
            abs(float(v.get("color_change_rate", 0.0) or 0.0)) * 15.0
            + abs(float(v.get("brightness_change", 0.0) or 0.0)) * 2.0,
        )
        turb = _clip(v.get("turbidity_proxy", 0.0))
        solid = _clip(v.get("solid_proxy", 0.0))
        boundary = _clip(v.get("phase_boundary_proxy", 0.0))
        stable = _clip(v.get("stability", 0.0))
        motion = _clip(v.get("motion_activity", 0.0))

        solid_tr = _trend(v, "solid")
        turb_tr = _trend(v, "turbidity")
        boundary_tr = _trend(v, "phase_boundary")
        color_tr = _trend(v, "color_change")
        motion_tr = _trend(v, "motion")

        solid_growth = _trend_strength(solid_tr, scale=24.0)
        turb_growth = _trend_strength(turb_tr, scale=22.0)
        boundary_growth = _trend_strength(boundary_tr, scale=8.0)
        color_growth = _trend_strength(color_tr, scale=6.0)

        explicit = {
            "cryst": 1.0
            if _contains(proc, "crystal", "crystall", "nucleation")
            else 0.0,
            "precip": 1.0 if _contains(proc, "precip") else 0.0,
            "sep": 1.0
            if _contains(proc, "extraction", "separation", "settling")
            else 0.0,
            "dissol": 1.0 if _contains(proc, "dissol") else 0.0,
            "color": 1.0
            if _contains(proc, "color", "colour", "kinetic", "conversion")
            else 0.0,
        }

        rag_sup = {
            "cryst": 1.0
            if _contains(rag_text, "crystal", "nucleation", "supersatur")
            else 0.0,
            "precip": 1.0
            if _contains(rag_text, "precipitate", "precipitation", "insoluble")
            else 0.0,
            "sep": 1.0
            if _contains(rag_text, "phase separation", "immiscible", "settling")
            else 0.0,
            "dissol": 1.0
            if _contains(rag_text, "dissolution", "solubility")
            else 0.0,
            "color": 1.0
            if _contains(rag_text, "color", "colour", "chrom")
            else 0.0,
        }

        hypotheses: list[dict[str, Any]] = []

        def add(label: str, score: float, reasons: list[dict[str, str]]) -> None:
            hypotheses.append(
                {
                    "label": label,
                    "score": round(_clip(score), 4),
                    "reasons": reasons,
                }
            )

        # --------------------------------------------------------------
        # Crystallization / nucleation
        # --------------------------------------------------------------
        cryst_visual: list[str] = []
        if solid_tr["delta"] > 5e-4:
            cryst_visual.append(
                "solid evidence increased "
                f"(delta={solid_tr['delta']:.4f}, "
                f"late/early={solid_tr['late_to_early_ratio']:.2f}x)"
            )
        if turb_tr["delta"] > 5e-4:
            cryst_visual.append(
                "turbidity evidence increased "
                f"(delta={turb_tr['delta']:.4f})"
            )
        if boundary_tr["delta"] > 2e-3:
            cryst_visual.append(
                "phase-boundary evidence increased "
                f"(delta={boundary_tr['delta']:.4f})"
            )

        cryst_context = (
            ["experiment context explicitly specifies crystallization/nucleation"]
            if explicit["cryst"]
            else []
        )
        cryst_rag = (
            ["retrieved knowledge supports crystallization/nucleation plausibility under compatible conditions"]
            if rag_sup["cryst"]
            else []
        )
        cryst_contra = []
        if stable > 0.85 and abs(solid_tr["recent_slope"]) < 3e-4:
            cryst_contra.append(
                "current visual activity is low and solid evidence is no longer increasing strongly"
            )

        cryst_score = (
            0.17 * solid
            + 0.06 * turb
            + 0.16 * solid_growth
            + 0.08 * turb_growth
            + 0.06 * boundary_growth
            + 0.34 * explicit["cryst"]
            + 0.10 * rag_sup["cryst"]
            + 0.03 * (1.0 - stable)
        )
        add(
            "crystallization_or_nucleation",
            cryst_score,
            _build_reasons(
                visual_support=cryst_visual,
                context_support=cryst_context,
                rag_support=cryst_rag,
                contradictions=cryst_contra,
            ),
        )

        # --------------------------------------------------------------
        # Precipitation / new solid
        # --------------------------------------------------------------
        precip_visual: list[str] = []
        if solid_tr["delta"] > 5e-4:
            precip_visual.append(
                "new/increasing solid-like visual evidence is present"
            )
        if turb_tr["delta"] > 5e-4:
            precip_visual.append(
                "turbidity increased over the observed history"
            )

        precip_score = (
            0.20 * solid
            + 0.08 * turb
            + 0.18 * solid_growth
            + 0.10 * turb_growth
            + 0.33 * explicit["precip"]
            + 0.09 * rag_sup["precip"]
            + 0.02 * (1.0 - stable)
        )
        add(
            "precipitation_or_new_solid",
            precip_score,
            _build_reasons(
                visual_support=precip_visual,
                context_support=(
                    ["experiment context explicitly specifies precipitation"]
                    if explicit["precip"]
                    else []
                ),
                rag_support=(
                    ["retrieved knowledge supports precipitation/new-solid plausibility"]
                    if rag_sup["precip"]
                    else []
                ),
                missing=(
                    ["visual solid/turbidity evidence is weak"]
                    if solid_growth < 0.08 and turb_growth < 0.08 and solid < 0.08
                    else []
                ),
            ),
        )

        # --------------------------------------------------------------
        # Phase separation / settling
        # --------------------------------------------------------------
        sep_visual: list[str] = []
        if boundary > 0.15 or boundary_tr["delta"] > 2e-3:
            sep_visual.append(
                "visible phase-boundary evidence is present or increasing"
            )
        if motion < 0.05 and motion_tr["recent_slope"] <= 0:
            sep_visual.append("recent motion is low or decaying")

        sep_score = (
            0.28 * boundary
            + 0.13 * boundary_growth
            + 0.08 * (1.0 - motion)
            + 0.33 * explicit["sep"]
            + 0.09 * rag_sup["sep"]
            + 0.04 * stable
        )
        add(
            "phase_separation_or_settling",
            sep_score,
            _build_reasons(
                visual_support=sep_visual,
                context_support=(
                    ["experiment context explicitly specifies separation/extraction/settling"]
                    if explicit["sep"]
                    else []
                ),
                rag_support=(
                    ["retrieved knowledge supports phase-separation/settling plausibility"]
                    if rag_sup["sep"]
                    else []
                ),
            ),
        )

        # --------------------------------------------------------------
        # Dissolution / homogenization
        # --------------------------------------------------------------
        dissol_visual: list[str] = []
        if solid_tr["delta"] < -5e-4:
            dissol_visual.append("solid-like evidence decreased over time")
        if turb_tr["delta"] < -5e-4:
            dissol_visual.append("turbidity evidence decreased over time")
        if stable > 0.75:
            dissol_visual.append("current visual dynamics are relatively stable")

        # For dissolution, *decreasing* solid/turbidity trends are stronger
        # evidence than merely having low absolute proxies.
        solid_decay = _clip(max(0.0, -solid_tr["delta"]) * 24.0)
        turb_decay = _clip(max(0.0, -turb_tr["delta"]) * 22.0)
        dissol_score = (
            0.07 * (1.0 - solid)
            + 0.05 * (1.0 - turb)
            + 0.13 * solid_decay
            + 0.10 * turb_decay
            + 0.06 * stable
            + 0.38 * explicit["dissol"]
            + 0.08 * rag_sup["dissol"]
        )
        add(
            "dissolution_or_homogenization",
            dissol_score,
            _build_reasons(
                visual_support=dissol_visual,
                context_support=(
                    ["experiment context explicitly specifies dissolution/homogenization"]
                    if explicit["dissol"]
                    else []
                ),
                rag_support=(
                    ["retrieved knowledge supports dissolution/solubility plausibility"]
                    if rag_sup["dissol"]
                    else []
                ),
            ),
        )

        # --------------------------------------------------------------
        # Color / conversion-like visual kinetics
        # --------------------------------------------------------------
        color_visual: list[str] = []
        if change > 0.05 or color_growth > 0.05:
            color_visual.append(
                "a sustained color/brightness trajectory is observable"
            )

        color_score = (
            0.33 * change
            + 0.12 * color_growth
            + 0.08 * (1.0 - stable)
            + 0.34 * explicit["color"]
            + 0.09 * rag_sup["color"]
        )
        add(
            "color_or_conversion_dynamics",
            color_score,
            _build_reasons(
                visual_support=color_visual,
                context_support=(
                    ["experiment context explicitly specifies color/kinetic monitoring"]
                    if explicit["color"]
                    else []
                ),
                rag_support=(
                    ["retrieved knowledge supports color/chromophore kinetic relevance"]
                    if rag_sup["color"]
                    else []
                ),
            ),
        )

        hypotheses.sort(key=lambda h: h["score"], reverse=True)
        top = hypotheses[0]
        second = hypotheses[1]

        temporal_state = _visual_state(v)

        # If the user supplied a meaningful process identity, retain that as
        # process_type. The ranked hypotheses describe interpretation, not an
        # excuse to overwrite explicit experimental metadata.
        if experiment_type and experiment_type.lower() not in {"unknown", "unspecified"}:
            process_type = experiment_type
        else:
            process_type = _process_type_from_label(top["label"])

        # --------------------------------------------------------------
        # Active event: separate "what process is this?" from "what is
        # happening now?". This is deliberately conservative.
        # --------------------------------------------------------------
        active_event = "none_or_uncertain"

        if top["label"] == "crystallization_or_nucleation":
            if stable > 0.85 and abs(solid_tr["recent_slope"]) < 3e-4:
                if solid_tr["delta"] > 5e-4:
                    active_event = "crystal_growth_decelerating_or_plateau"
                else:
                    active_event = "crystallization_context_but_no_clear_active_visual_event"
            elif solid_tr["recent_slope"] > 3e-4:
                active_event = "solid_growth_or_nucleation"

        elif top["label"] == "precipitation_or_new_solid":
            if solid_tr["recent_slope"] > 3e-4 or turb_tr["recent_slope"] > 3e-4:
                active_event = "new_solid_or_turbidity_increasing"
            elif stable > 0.85:
                active_event = "new_solid_process_approaching_plateau"

        elif top["label"] == "phase_separation_or_settling":
            if boundary_tr["recent_slope"] > 2e-3:
                active_event = "phase_boundary_developing"
            elif motion < 0.05 and stable > 0.75:
                active_event = "settling_or_phase_separation_stabilising"

        elif top["label"] == "dissolution_or_homogenization":
            if solid_tr["recent_slope"] < -3e-4 or turb_tr["recent_slope"] < -3e-4:
                active_event = "dissolution_or_homogenization_progressing"
            elif stable > 0.85:
                active_event = "homogenization_or_dissolution_plateau"

        elif top["label"] == "color_or_conversion_dynamics":
            if abs(color_tr["recent_slope"]) > 3e-4:
                active_event = "color_trajectory_changing"
            elif stable > 0.85:
                active_event = "color_trajectory_plateau"

        confidence = _clip(
            0.50 * top["score"]
            + 0.18 * max(0.0, top["score"] - second["score"])
            + 0.12 * min(1.0, len(docs) / 3.0)
            + 0.10 * (1.0 if ctx.participants else 0.0)
            + 0.10 * (1.0 if any(explicit.values()) else 0.0)
        )

        # --------------------------------------------------------------
        # Expected next *observable* events, not guaranteed chemistry.
        # --------------------------------------------------------------
        next_events: list[str]
        if top["label"] in {
            "crystallization_or_nucleation",
            "precipitation_or_new_solid",
        }:
            if active_event in {
                "crystal_growth_decelerating_or_plateau",
                "new_solid_process_approaching_plateau",
            }:
                next_events = [
                    "solid-region growth may remain slow if the visual plateau persists",
                    "renewed solid/turbidity growth would indicate reactivation or a new event",
                ]
            else:
                next_events = [
                    "solid-region area/count may increase if the process remains active",
                    "visual activity may later decay toward a plateau",
                ]
        elif top["label"] == "phase_separation_or_settling":
            next_events = [
                "phase boundary may sharpen or move",
                "motion/turbidity may decrease",
            ]
        elif top["label"] == "dissolution_or_homogenization":
            next_events = [
                "solid/turbidity proxies may decrease",
                "visual stability may increase",
            ]
        else:
            next_events = [
                "continue monitoring the trajectory; sustained rate decay would support a plateau interpretation"
            ]

        uncertainty: list[str] = []
        if not ctx.participants:
            uncertainty.append(
                "No participant identities were provided; chemical interpretation is mostly visual."
            )
        if not docs:
            uncertainty.append(
                "RAG returned no relevant documents; interpretation is not ORD/knowledge grounded."
            )
        if temporal_state == "plateau_or_stable" and active_event not in {
            "homogenization_or_dissolution_plateau",
            "color_trajectory_plateau",
            "crystal_growth_decelerating_or_plateau",
            "new_solid_process_approaching_plateau",
        }:
            uncertainty.append(
                "The video is currently visually stable; process identity does not prove an ongoing molecular event."
            )
        if confidence < 0.55:
            uncertainty.append(
                "Chemical semantics remain ambiguous; treat the result as a ranked hypothesis, not a confirmed identity/state."
            )

        retrieved = [
            {
                "id": d.doc_id,
                "source": d.source,
                "score": round(d.score, 4),
                "title": d.title,
                "text": d.text[:800],
            }
            for d in docs
        ]

        # Evidence in the report is grouped by provenance so a human can audit
        # exactly why the top interpretation was selected.
        top_reasons = top.get("reasons", [])
        evidence = {
            "visual": [
                r
                for r in top_reasons
                if r.get("source") == "visual" and r.get("type") == "support"
            ],
            "context": [
                r
                for r in top_reasons
                if r.get("source") == "context" and r.get("type") == "support"
            ],
            "rag": [
                r
                for r in top_reasons
                if r.get("source") == "rag" and r.get("type") == "support"
            ],
            "contradictions": [
                r for r in top_reasons if r.get("type") == "contradiction"
            ],
            "missing_evidence": [
                r for r in top_reasons if r.get("type") == "missing_evidence"
            ],
        }

        return SemanticReport(
            experiment_id=ctx.experiment_id,
            process_type=process_type,
            temporal_state=temporal_state,
            active_event=active_event,
            confidence=confidence,
            visual_evidence=v,
            hypotheses=hypotheses[:5],
            retrieved_knowledge=retrieved,
            evidence=evidence,
            expected_next_events=next_events,
            uncertainty_notes=uncertainty,
            model_notes={
                "reasoner": "deterministic evidence-attributed v1.2",
                "tcpt_enabled": bool(
                    visual.get("tcpt", {}).get("enabled", False)
                ),
                "evidence_policy": (
                    "visual, explicit context, and RAG support are tracked separately"
                ),
            },
        )
