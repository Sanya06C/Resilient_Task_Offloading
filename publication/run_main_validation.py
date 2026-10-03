"""Independent workload-seed validation of fixed checkpoints, not training-seed inference.

Five predeclared workload seeds, ten episodes each, three capacities and paired
Original/Selector conditions. Policies and original observation semantics stay fixed.
"""
import os
for k,v in {'TF_USE_LEGACY_KERAS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','TF_CPP_MIN_LOG_LEVEL':'3','TF_ENABLE_ONEDNN_OPTS':'0'}.items():os.environ.setdefault(k,v)
from pathlib import Path
import sys,json,hashlib,time,random
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Config import Config
from MEC_Env import MEC
from accounting import EpisodeAccounting
from stress_aware_selector import StressAwareSelector
from D3QN import DuelingDoubleDeepQNetwork
SEEDS=[1101,2202,3303,4404,5505]
CAPS={'Normal':14.,'Medium':8.48769868957287,'High':6.326878361484239}
OUT=ROOT/'publication/evidence/main_multiseed_raw.json'
def run():
 policies=[];hashes={}
 for u in range(20):
  prefix=ROOT/f'TrainedModel_20UE_2EN_PerformanceMode/800/{u}_X_model/model.ckpt-800'
  for suffix in ['.index','.data-00000-of-00001']:
   p=Path(str(prefix)+suffix);hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
  p=DuelingDoubleDeepQNetwork(3,6,2,110,learning_rate=Config.LEARNING_RATE,reward_decay=Config.REWARD_DECAY,e_greedy=Config.E_GREEDY,replace_target_iter=Config.N_NETWORK_UPDATE,memory_size=Config.MEMORY_SIZE)
  p.saver.restore(p.sess,str(prefix));p.epsilon=1.;policies.append(p)
 rows=[];start=time.time()
 fixed_state=json.loads((ROOT/'network_stress_profiles/run_medium/metadata.json').read_text())['ue_energy_state']
 for seed in SEEDS:
  rng=np.random.default_rng(seed);workloads=[]
  for ep in range(10):
   sizes=rng.uniform(1,7,(110,20));sizes*=rng.random((110,20))<.3;sizes[-10:]=0
   density=np.zeros_like(sizes);density[sizes>0]=rng.choice(Config.TASK_COMP_DENS,size=np.count_nonzero(sizes))
   workloads.append((sizes,density))
  for stress,cap in CAPS.items():
   for use_selector in [False,True]:
    # Reset only the recurrent memory between independent condition rollouts.
    for p in policies:
     p.lstm_history.clear()
     for _ in range(p.n_lstm_step):p.lstm_history.append(np.zeros(p.n_lstm_state))
    np.random.seed(seed);random.seed(seed)
    env=MEC(20,2,110,1,10);env.ue_energy_state=fixed_state.copy();env.tran_cap_ue[:]=cap*.1;env.accounting=EpisodeAccounting('power_time_v1')
    selector=StressAwareSelector()
    for ep,(sizes,dens) in enumerate(workloads):
     obs,lstm=env.reset(sizes.copy(),dens.copy());selector.reset_episode();done=False
     while not done:
      actions=np.zeros(20,dtype=int)
      for u,p in enumerate(policies):
       if np.sum(obs[u])!=0:actions[u]=int(p.choose_action(obs[u]))
      if use_selector:
       for u in np.flatnonzero(sizes[env.time_count]>0):actions[u],_=selector.select(env,u,actions[u])
      obs,lstm,done=env.step(actions)
      for u,p in enumerate(policies):p.update_lstm(lstm[u]);p.store_q_value.clear()
     r=env.accounting.summarize(env,ep);r.update(workload_seed=seed,stress_level=stress,policy='Selector' if use_selector else 'Original');rows.append(r)
    print(seed,stress,use_selector,'complete',round(time.time()-start,1),flush=True)
    OUT.write_text(json.dumps({'workload_seeds':SEEDS,'checkpoint_hashes':hashes,'episodes_per_seed':10,'raw_records':rows},indent=2))
 assert all(p.learn_step_counter==0 for p in policies)
 for p in policies:p.sess.close()
if __name__=='__main__':run()
