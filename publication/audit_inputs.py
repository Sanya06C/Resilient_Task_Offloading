from pathlib import Path
import json, re, hashlib, csv, argparse
import numpy as np
import pypdf
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'publication'/'evidence'; OUT.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--trace',type=Path,required=True);args=parser.parse_args()
trace=args.trace
from eh_extension_v2.uclm_trace_v2 import load_uclm_trace_v2
t=load_uclm_trace_v2(trace)
splits={}
for s in ['train','val','test','all']:
 x=t.slice_split(s); d=np.diff(x.timestamp).astype('timedelta64[s]').astype(int)
 splits[s]={'records':len(x.timestamp),'start':str(x.timestamp[0]),'end':str(x.timestamp[-1]),'elapsed_seconds':int((x.timestamp[-1]-x.timestamp[0])/np.timedelta64(1,'s')),'cadence_seconds_counts':dict(zip(*[v.tolist() for v in np.unique(d,return_counts=True)]))}
splits['trace_sha256']=sha(trace)
splits['training_used_seconds']=1100
splits['evaluation_start_actual_timestamp']=str(t.slice_split('test').timestamp[540])
splits['training_end_actual_timestamp']=str(t.timestamp[18])
splits['test_first_two_minutes']=list(map(str,t.slice_split('test').timestamp[540:543]))
(OUT/'trace_audit.json').write_text(json.dumps(splits,indent=2))
print(json.dumps(splits,indent=2))
raw=json.loads((ROOT/'eh_extension_v2/results/core_evaluation_raw.json').read_text())
print('EH raw records',len(raw),'sample', {k:raw[0][k] for k in ['model_seed','episode_idx','mean_final_soc','harvested_charge_mah_per_ue','energy_induced_failures_total']})
manifest={}
for seed in [101,202,303,404,505]:
 p=ROOT/f'eh_extension_v2/checkpoints/seed_{seed}'; m=json.loads((p/'metadata.json').read_text())
 manifest[str(seed)]={'episodes':m['train_episodes'],'updates':m['total_gradient_steps'],'hashes_valid':all(sha(p/k)==v for k,v in m['checkpoint_hashes'].items())}
(OUT/'training_audit.json').write_text(json.dumps(manifest,indent=2)); print(manifest)
for p in [ROOT/'accounting_baseline/default_legacy/episodes/episode_0000.npz',ROOT/'network_stress_profiles/run_medium/episodes/episode_0000.npz']:
 a=np.load(p); print(str(p),a.files)
pdf=pypdf.PdfReader(r'D:\research\Resilient_Task_Offloading_IIoT_LINKS_ADDED.pdf')
text='\n'.join(p.extract_text() or '' for p in pdf.pages)
(OUT/'original_manuscript_text.txt').write_text(text,encoding='utf-8')
print('Original pages',len(pdf.pages))
