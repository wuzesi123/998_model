from pathlib import Path
import subprocess, sys, json
ROOT=Path(__file__).resolve().parent
print('[1/3] compile')
subprocess.run([sys.executable,'-m','compileall','-q',str(ROOT/'chem_process_rag')],check=True)
print('[2/3] demo')
subprocess.run([sys.executable,str(ROOT/'RUN_DEMO.py')],check=True,cwd=ROOT)
print('[3/3] verify result')
r=json.loads((ROOT/'outputs/demo_result.json').read_text(encoding='utf-8'))
assert r['report']['visual_state']
assert r['report']['chemical_interpretation']
assert r['rag_stats']['documents']>=1
print('PASS: visual -> context -> RAG -> semantic report')
