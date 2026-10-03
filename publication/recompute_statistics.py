"""Seed-level inference. All nested episodes are averaged before inference."""
from pathlib import Path
import json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'publication/evidence'
def stat(a):
 a=np.array(a,float);n=len(a);m=a.mean();sd=a.std(ddof=1)
 assert n==5, 'This routine reports five independent seed means.'
 w=2.7764451051977987*sd/np.sqrt(n)
 return {'n':n,'mean':float(m),'sd':float(sd),'ci95':[float(m-w),float(m+w)]}
def paired(a,b):
 d=np.array(a)-np.array(b);s=stat(d)
 s['seed_differences']=d.tolist();s['t_stat']=float(d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))) if np.std(d)>0 else None
 perms=np.array([abs(np.dot(d,sign)) for sign in itertools.product([-1,1],repeat=len(d))])
 s['exact_signflip_p']=float(np.mean(perms>=abs(d.sum())-1e-12))
 return s
def summarize(raw,seed_key,pol_key,pols,metrics):
 results={}
 seeds=sorted({r[seed_key] for r in raw});assert len(seeds)==5
 for stress in ['Normal','Medium','High']:
  results[stress]={}
  for metric in metrics:
   data={}
   for pol in pols:
    values=[]
    for seed in seeds:
     rows=[r for r in raw if r['stress_level']==stress and r[seed_key]==seed and r[pol_key]==pol]
     assert len(rows)==10,(stress,seed,pol,len(rows))
     values.append(np.mean([r[metric] for r in rows]))
    data[str(pol)]={**stat(values),'seed_means':list(map(float,values))}
   data['paired_difference']=paired(data[str(pols[0])]['seed_means'],data[str(pols[1])]['seed_means'])
   results[stress][metric]=data
 return results
metrics=['ue_only_total_energy','ue_transmission_energy','deadline_violation_fraction','offloaded_fraction','mean_final_soc','harvested_charge_mah_per_ue','discharged_charge_mah_per_ue']
eh_path=OUT/'eh_core_verified.json'
eh=json.loads(eh_path.read_text())['raw_records'] if eh_path.exists() else json.loads((ROOT/'eh_extension_v2/results/core_evaluation_raw.json').read_text())
result=summarize(eh,'model_seed','harvest_enabled',[True,False],metrics)
for stress in result:
 for metric in ['mean_final_soc','minimum_soc','harvested_charge_mah_per_ue','discharged_charge_mah_per_ue']:
  key='terminal_'+metric;data={}
  for enabled in [True,False]:
   vals=[next(r[metric] for r in eh if r['stress_level']==stress and r['model_seed']==s and r['harvest_enabled']==enabled and r['episode_idx']==9) for s in [101,202,303,404,505]]
   data[str(enabled)]={**stat(vals),'seed_means':list(map(float,vals))}
  data['paired_difference']=paired(data['True']['seed_means'],data['False']['seed_means']);result[stress][key]=data
(OUT/'eh_seed_statistics.json').write_text(json.dumps(result,indent=2))
combined_path=OUT/'combined_verified.json'
combined=json.loads(combined_path.read_text())['raw_records'] if combined_path.exists() else json.loads((ROOT/'combined_selector_eh_results.json').read_text())['raw_records']
combined_stats={}
for stress in ['Normal','Medium','High']:
 combined_stats[stress]={}
 for metric in metrics+['overrides']:
  vals=[np.mean([r[metric] for r in combined if r['stress_level']==stress and r['model_seed']==s]) for s in [101,202,303,404,505]]
  combined_stats[stress][metric]={**stat(vals),'seed_means':list(map(float,vals))}
  if metric in result[stress]:combined_stats[stress][metric]['vs_eh']=paired(vals,result[stress][metric]['True']['seed_means'])
 for metric in ['mean_final_soc','harvested_charge_mah_per_ue']:
  vals=[next(r[metric] for r in combined if r['stress_level']==stress and r['model_seed']==s and r['episode_idx']==9) for s in [101,202,303,404,505]]
  combined_stats[stress]['terminal_'+metric]={**stat(vals),'seed_means':list(map(float,vals))}
(OUT/'combined_seed_statistics.json').write_text(json.dumps(combined_stats,indent=2))
for stress in result:
 print(stress,'EH SOC delta',result[stress]['mean_final_soc']['paired_difference'],'combined UE',combined_stats[stress]['ue_only_total_energy'])
if (OUT/'main_multiseed_raw.json').exists():
 raw=json.loads((OUT/'main_multiseed_raw.json').read_text())['raw_records']
 if len(raw)==300:
  main=summarize(raw,'workload_seed','policy',['Selector','Original'],metrics[:4])
  (OUT/'main_seed_statistics.json').write_text(json.dumps(main,indent=2))
