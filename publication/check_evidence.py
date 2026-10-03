"""Scientific integrity checks on the final evidence, not implementation-mirroring tests."""
from pathlib import Path
import json,hashlib,re,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'publication/evidence'
def load(n):return json.loads((E/n).read_text(encoding='utf8'))
main=load('main_multiseed_raw.json');eh=load('eh_core_verified.json');combined=load('combined_verified.json')
assert len(main['raw_records'])==300
assert len(eh['raw_records'])==300 and len(combined['raw_records'])==150
assert eh['metadata']==combined['metadata']
for group in [eh,combined]:
 assert group['metadata']['nuisance_inputs_fixed']
 for row in group['raw_records']:
  assert row['energy_induced_failures_total']==0 and row['battery_depleted_ues']==0
  expected=.5+(.95*row['harvested_charge_mah_per_ue']-row['discharged_charge_mah_per_ue']/.95)/2000
  assert abs(row['mean_final_soc']-expected)<1e-9, ('Charge conservation',row)
  assert 0<=row['deadline_violation_fraction']<=1
stats=load('eh_seed_statistics.json')
for stress in stats:
 for key,item in stats[stress].items():
  assert item['paired_difference']['n']==5
  assert item['paired_difference']['exact_signflip_p']>=.0625-1e-12
  for mode in ['True','False']:
   assert len(item[mode]['seed_means'])==5
   np.testing.assert_allclose(np.mean(item[mode]['seed_means']),item[mode]['mean'])
for seed,m in load('training_audit.json').items():
 assert m['hashes_valid']
 folder=ROOT/f'eh_extension_v2/checkpoints/seed_{seed}'
 meta=json.loads((folder/'metadata.json').read_text())
 for filename,sha in meta['checkpoint_hashes'].items():assert hashlib.sha256((folder/filename).read_bytes()).hexdigest()==sha
for path,sha in main['checkpoint_hashes'].items():assert hashlib.sha256((ROOT/path.replace('\\','/')).read_bytes()).hexdigest()==sha
trace=load('trace_audit.json');assert sum(trace[x]['records'] for x in ['train','val','test'])==93438
cf=load('counterfactual_verified.json')
assert set(cf)=={'Normal','Medium','High'}
for stress in cf:
 for policy,r in cf[stress].items():
  assert r['episodes']==20 and r['deadline_s']==1
  assert 0<=r['local_predicted_feasible_fraction']<=1 and 0<=r['offload_observed_success_fraction']<=1
  if r['both_feasible_tasks']:
   assert r['local_mean_completion_s_both_feasible']<=1 and r['offload_mean_completion_s_both_feasible']<=1
source=(ROOT/'publication/Resilient_Task_Offloading_IIoT_FINAL.tex').read_text(encoding='utf8')
cited=set(k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',source) for k in group.split(','))
defined=set(re.findall(r'\\bibitem\{([^}]+)\}',source))
assert cited==defined, ('Citation mismatch',cited-defined,defined-cited)
assert '\\input{' not in source and '\\includegraphics' not in source
assert not any(x in source for x in ['Author Name','email@','[?]','p < 10^{-9}','p < 10^{-4}'])
print('PASS: complete runs; paired nuisance controls; seed units; charge conservation; checkpoint integrity; trace allocation; matched deadlines; complete citation coverage; standalone source.')
