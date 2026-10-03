"""Paired EH and selector evaluation with all nuisance observation inputs fixed."""
import os
for k,v in {'TF_USE_LEGACY_KERAS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','TF_CPP_MIN_LOG_LEVEL':'3','TF_ENABLE_ONEDNN_OPTS':'0'}.items():os.environ.setdefault(k,v)
from pathlib import Path
import sys,json,random,time,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eh_extension_v2.uclm_trace_v2 import load_uclm_trace_v2
from eh_extension_v2.eh_mec_env_v2 import EHMECV2
from eh_extension_v2.battery_v2 import BatteryConfigV2
from stress_aware_selector import StressAwareSelector
from D3QN import DuelingDoubleDeepQNetwork
import tensorflow as tf
def run(trace_path):
 trace=load_uclm_trace_v2(trace_path,'test')
 states=json.loads((ROOT/'network_stress_profiles/run_medium/metadata.json').read_text())['ue_energy_state']
 workloads=[]
 for ep in range(10):
  with np.load(ROOT/'accounting_baseline/default_legacy/episodes'/f'episode_{ep:04d}.npz') as s:workloads.append((s['sizes'].copy(),s['densities'].copy()))
 rows=[];combined=[];start=time.time()
 for seed in [101,202,303,404,505]:
  p=DuelingDoubleDeepQNetwork(3,8,2,110,learning_rate=.002,reward_decay=.9,e_greedy=.98,n_lstm_step=1)
  prefix=ROOT/f'eh_extension_v2/checkpoints/seed_{seed}/eh_d3qn';p.saver.restore(p.sess,str(prefix));p.epsilon=1.
  for stress in ['Normal','Medium','High']:
   for mode in ['EH','NoEH','CombinedEH']:
    np.random.seed(seed);random.seed(seed);p.lstm_history.clear();p.lstm_history.append(np.zeros(2));p.store_q_value.clear()
    env=EHMECV2(trace=trace,stress_level=stress,harvest_enabled=mode!='NoEH',persistent_battery=True,battery_config=BatteryConfigV2(),start_slot_offset=540*600)
    env.ue_energy_state=states.copy();selector=StressAwareSelector()
    for ep,(sizes,dens) in enumerate(workloads):
     obs,_=env.reset(sizes.copy(),dens.copy());selector.reset_episode();done=False;over=0
     while not done:
      actions=np.zeros(20,dtype=int)
      for u in np.flatnonzero(np.any(obs!=0,axis=1)):
       action=int(p.choose_action(obs[u]));p.store_q_value.clear()
       if mode=='CombinedEH':
        chosen,_=selector.select(env,u,action);over+=int(action!=chosen);action=chosen
       actions[u]=action
      obs,_,done=env.step(actions)
     r=env.accounting.summarize(env,ep);r.update(env.eh_summary());r.update(model_seed=seed,episode_idx=ep,stress_level=stress,harvest_enabled=mode!='NoEH',mode=mode,overrides=over)
     assert r['energy_induced_failures_total']==0 and r['battery_depleted_ues']==0
     (combined if mode=='CombinedEH' else rows).append(r)
    print(seed,stress,mode,'completed',round(time.time()-start,1),flush=True)
    metadata={'static_energy_preference':states,'start_test_row':540,'physical_seconds':110.,'initial_soc':.5,'persistent_battery':True,'training_seeds':[101,202,303,404,505],'evaluated_workloads':list(range(10)),'trace_sha256':hashlib.sha256(Path(trace_path).read_bytes()).hexdigest(),'nuisance_inputs_fixed':True}
    (ROOT/'publication/evidence/eh_core_verified.json').write_text(json.dumps({'metadata':metadata,'raw_records':rows},indent=2))
    (ROOT/'publication/evidence/combined_verified.json').write_text(json.dumps({'metadata':metadata,'raw_records':combined},indent=2))
  assert p.learn_step_counter==0;p.sess.close();tf.compat.v1.reset_default_graph()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--trace',type=Path,required=True);args=parser.parse_args();run(args.trace)
