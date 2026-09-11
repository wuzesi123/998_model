import json, tempfile
from pathlib import Path
from chem_process_rag.context.context_builder import build_context, context_to_query
from chem_process_rag.rag.ord_rag import ORDRAG
from chem_process_rag.reasoning.semantic_reasoner import SemanticReasoner

def test_context_and_rag():
    ctx=build_context({"experiment_id":"x","participants":[{"name":"benzoic acid","role":"solute"},{"name":"water","role":"solvent"}],"procedure":{"experiment_type":"crystallization"}})
    assert "benzoic acid" in context_to_query(ctx)
    seed=Path(__file__).parents[1]/"data/knowledge/seed_chemistry.jsonl"
    rag=ORDRAG(seed_knowledge=seed); docs=rag.search(ctx,{"solid_proxy":.8,"turbidity_proxy":.7})
    assert docs and "crystall" in (docs[0].title+docs[0].text).lower()

def test_reasoner():
    ctx=build_context({"experiment_id":"x","participants":[{"name":"benzoic acid"}],"procedure":{"experiment_type":"crystallization"}})
    vis={"summary":{"color_change_rate":.01,"brightness_change":.02,"motion_activity":.01,"turbidity_proxy":.7,"solid_proxy":.8,"phase_boundary_proxy":.1,"stability":.35},"tcpt":{"enabled":False}}
    rep=SemanticReasoner().infer(ctx,vis,[])
    assert rep.current_state in {"crystallization_or_nucleation","precipitation_or_new_solid"}
