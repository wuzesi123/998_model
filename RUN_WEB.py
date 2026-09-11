from pathlib import Path
import json, tempfile
try:
    import streamlit as st
except ImportError:
    raise SystemExit("Install web UI dependencies first: pip install -r requirements-web.txt")
from chem_process_rag import ChemProcessPipeline

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="ChemProcessRAG",layout="wide")
st.title("ChemProcessRAG — Visual TCPT + Reaction Context + ORD RAG")
st.caption("Visual evidence is kept separate from chemical interpretation. RAG uses causal fields only by default.")
video=st.file_uploader("Reaction video",type=["mp4","avi","mov","mkv"])
ctx=st.file_uploader("Reaction context JSON",type=["json"])
checkpoint=st.text_input("Optional TCPT checkpoint path","")
sample=st.number_input("Sample every seconds",0.1,10.0,1.0,.1)
if st.button("Analyze",type="primary",disabled=not(video and ctx)):
    with tempfile.TemporaryDirectory() as td:
        vp=Path(td)/video.name; cp=Path(td)/ctx.name
        vp.write_bytes(video.getvalue()); cp.write_bytes(ctx.getvalue())
        pipe=ChemProcessPipeline(ROOT/"data/ord",ROOT/"data/knowledge/seed_chemistry.jsonl",sample_every_sec=sample,tcpt_checkpoint=checkpoint or None)
        res=pipe.run(vp,cp)
        rep=res["report"]
        c1,c2,c3=st.columns(3); c1.metric("Interpretation",rep["current_state"]); c2.metric("Confidence",f"{rep['confidence']:.2f}"); c3.metric("RAG documents",res["rag_stats"]["documents"])
        st.subheader("Visual evidence"); st.json(res["visual"]["summary"])
        st.subheader("Semantic hypotheses"); st.dataframe(rep["hypotheses"],use_container_width=True)
        st.subheader("Retrieved knowledge"); st.json(rep["retrieved_knowledge"])
        st.subheader("Full result"); st.download_button("Download JSON",json.dumps(res,ensure_ascii=False,indent=2),"chem_process_result.json","application/json")
