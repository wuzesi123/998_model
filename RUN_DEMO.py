from pathlib import Path
import json, cv2, numpy as np
from chem_process_rag import ChemProcessPipeline, save_result

ROOT=Path(__file__).resolve().parent

def make_demo_video(path:Path, seconds=18, fps=12):
    path.parent.mkdir(parents=True,exist_ok=True)
    size=(640,420)
    fourcc=cv2.VideoWriter_fourcc(*"MJPG")
    out=cv2.VideoWriter(str(path),fourcc,fps,size)
    rng=np.random.default_rng(7)
    crystals=[]
    for i in range(seconds*fps):
        p=i/(seconds*fps-1)
        img=np.zeros((size[1],size[0],3),np.uint8)
        img[:]=(145-int(15*p),125-int(8*p),95+int(12*p))
        cv2.rectangle(img,(80,60),(560,370),(170-int(12*p),145-int(8*p),115+int(10*p)),-1)
        if p>0.28 and i%6==0:
            crystals.append((int(rng.integers(120,520)),int(rng.integers(100,330)),float(rng.uniform(2,5))))
        for j,(x,y,r0) in enumerate(crystals):
            r=int(r0 + max(0,p-0.28)*18*(0.4+0.6*((j%5)/4)))
            pts=np.array([[x-r,y],[x,y-r//2],[x+r,y],[x,y+r//2]],np.int32)
            cv2.fillConvexPoly(img,pts,(225,225,220))
            cv2.polylines(img,[pts],True,(245,245,240),1)
        noise=rng.normal(0,1.8,img.shape).astype(np.int16)
        img=np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)
        cv2.putText(img,f"demo t={i/fps:04.1f}s",(18,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(250,250,250),1,cv2.LINE_AA)
        out.write(img)
    out.release()
    if not path.exists() or path.stat().st_size<1000: raise RuntimeError("Could not create demo video")

if __name__=="__main__":
    video=ROOT/"outputs"/"demo_reaction.avi"
    result_path=ROOT/"outputs"/"demo_result.json"
    make_demo_video(video)
    pipe=ChemProcessPipeline(ROOT/"data/ord",ROOT/"data/knowledge/seed_chemistry.jsonl",sample_every_sec=.5)
    result=pipe.run(video,ROOT/"examples/reaction_context.json",top_k=5)
    save_result(result,result_path)
    r=result["report"]
    print("\n=== ChemProcessRAG demo ===")
    print("state      :",r["current_state"])
    print("confidence :",round(r["confidence"],3))
    print("visual     :",json.dumps(result["visual"]["summary"],ensure_ascii=False,indent=2))
    print("RAG docs   :",result["rag_stats"]["documents"])
    print("top RAG    :",[x["title"] for x in r["retrieved_knowledge"][:3]])
    print("next       :",r["expected_next_events"])
    print("saved      :",result_path)
