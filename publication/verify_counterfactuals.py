"""Matched task audit on 100 paired episodes, using each policy's executed trajectory.

Local energy is full-work service energy; local completion is a no-future-arrivals
FIFO prediction censored at deadline+one slot. Offload time is observed terminal
time, including failed tasks. Offload full-work energy is TX plus observed task
post-transmission wait, a task attribution (not the union-based episode ledger).
"""
from pathlib import Path
import json,sys
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from MEC_Env import MEC
from accounting import EpisodeAccounting
from stress_aware_selector import StressAwareSelector
CAPS={'Normal':14.,'Medium':8.48769868957287,'High':6.326878361484239}
ORIG={'Normal':'accounting_baseline/default_legacy','Medium':'network_stress_profiles/run_medium','High':'network_stress_profiles/run_high'}
parser=argparse.ArgumentParser();parser.add_argument('--episodes',type=int,default=100);args=parser.parse_args()
results={}
for stress,cap in CAPS.items():
 results[stress]={}
 for policy,directory in [('Original',ORIG[stress]),('Selector','selective_offloading/run_'+stress.lower())]:
  env=MEC(20,2,110,1,10);env.tran_cap_ue[:]=cap*.1;env.accounting=EpisodeAccounting('power_time_v1');sel=StressAwareSelector()
  values=[]
  for ep in range(args.episodes):
   saved=np.load(ROOT/directory/'episodes'/f'episode_{ep:04d}.npz')
   sizes=saved['sizes'];dens=saved['densities'];actions=saved['actions'];arrived=sizes>0
   env.reset(sizes.copy(),dens.copy());done=False
   while not done:
    ti=env.time_count
    for u in np.flatnonzero(arrived[ti] & (actions[ti]>0)):
     loc=sel.estimate_local(env,u)
     full_tx=2.3*sizes[ti,u]/cap
     d=float(saved['delay'][ti,u])*.1;txd=float(saved['transmission_delay'][ti,u])*.1
     # Waiting is a per-task counterfactual attribution; whole-run union idle differs.
     wait=max(d-txd,0.) if txd>0 else 0.
     values.append([loc['ue_energy'],full_tx+.1*wait,loc['predicted_delay_slots']*.1,d,int(loc['feasible']),int(saved['failed'][ti,u]==0),float(saved['tx_volume'][ti,u])*2.3/cap])
    env.step(np.maximum(actions[ti],0));done=env.time_count>=110
   np.testing.assert_allclose(env.process_delay,saved['delay'],atol=1e-8)
   np.testing.assert_allclose(env.accounting.volumes['tx'],saved['tx_volume'],atol=1e-8)
  v=np.array(values);both=(v[:,4]==1)&(v[:,5]==1)
  results[stress][policy]={'episodes':args.episodes,'offloaded_tasks':len(v),'local_full_work_energy_j':float(v[:,0].mean()),'offload_full_work_tx_plus_task_wait_j':float(v[:,1].mean()),'local_predicted_terminal_time_s_censored':float(v[:,2].mean()),'offload_observed_terminal_time_s':float(v[:,3].mean()),'local_predicted_feasible_fraction':float(v[:,4].mean()),'offload_observed_success_fraction':float(v[:,5].mean()),'actual_tx_energy_per_task_j':float(v[:,6].mean()),'deadline_s':1.0,'both_feasible_tasks':int(both.sum()),'local_mean_completion_s_both_feasible':float(v[both,2].mean()) if both.any() else None,'offload_mean_completion_s_both_feasible':float(v[both,3].mean()) if both.any() else None}
  print(stress,policy,results[stress][policy],flush=True)
 (ROOT/'publication/evidence/counterfactual_verified.json').write_text(json.dumps(results,indent=2))
