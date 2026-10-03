"""Capacity-only charge-ledger replay of fixed pre-depletion demand.

No battery-empty task-execution claims. Each demand trace repeats recorded
Always-Local CPU activity from 50 workloads; source current and voltage use the
same nominal 60-s sample grid as the inherited EH implementation.
"""
from pathlib import Path
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from MEC_Env import MEC
from accounting import EpisodeAccounting
from eh_extension_v2.uclm_trace_v2 import load_uclm_trace_v2
parser=argparse.ArgumentParser();parser.add_argument('--trace',type=Path,required=True);args=parser.parse_args()
t=load_uclm_trace_v2(args.trace,'test')
env=MEC(20,2,110,1,10);env.accounting=EpisodeAccounting('power_time_v1')
demand=[]
for ep in range(50):
 s=np.load(ROOT/'accounting_baseline/default_legacy/episodes'/f'episode_{ep:04d}.npz');env.reset(s['sizes'].copy(),s['densities'].copy())
 for _ in range(110):env.step(np.zeros(20,dtype=int))
 local=env.accounting.local_fraction;tx=env.accounting.tx_fraction
 watts=2*local+2.3*tx+.1*np.maximum(1-np.maximum(local,tx),0)
 demand.append(watts)
demand=np.concatenate(demand);eta=.95;rows=[]
for capacity in [1000.,2000.,3000.]:
 for enabled in [False,True]:
  charge=np.full(20,capacity*.5);first=None;first_zero=None;mean_cross=None
  for slot in range(6*36000):
   minute=480+slot//600;v=max(t.battery_voltage_v[minute],3.3)
   harvest=max(t.source_current_ma[minute],0) if enabled else 0
   draw=demand[slot%len(demand)]/v*1000*.1/3600/eta
   charge=np.clip(charge+eta*harvest*.1/3600-draw,0,capacity)
   if first is None and np.any(charge<=capacity*.0001):first=(slot+1)*.1/3600
   if first_zero is None and np.any(charge==0):first_zero=(slot+1)*.1/3600
   if charge.mean()<=capacity*.0001:mean_cross=(slot+1)*.1/3600;break
  rows.append({'capacity_mah':capacity,'harvest_enabled':enabled,'first_ue_zero_hours':first_zero,'first_ue_below_001percent_hours':first,'mean_soc_below_001percent_hours':mean_cross,'demand':'Repeated 50 Always-Local workload traces; no policy rerouting; starts test row 480','threshold_soc_fraction':.0001})
  print(rows[-1],flush=True)
alpha=[]
rho=.297;cstar=2.3*2.6/(2*rho)
for a in [1/3,.5,2/3,1.]:alpha.append({'alpha':a,'tx_only_break_even_capacity':cstar,'break_even_multiplier':(14/cstar)**(1/a)})
(ROOT/'publication/evidence/sensitivity_verified.json').write_text(json.dumps({'battery_capacity':rows,'analytical_exponent_crossover':alpha,'status':'Analytical TX-only crossover; deterministic fixed-demand battery sensitivity'},indent=2))
