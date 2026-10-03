"""Regenerate manuscript tables and figures from released numerical records only."""
from pathlib import Path
import json,csv,math,re,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'publication';E=PUB/'evidence';T=PUB/'tables';F=PUB/'figures'
T.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
def load(name):return json.loads((E/name).read_text(encoding='utf8'))
def readcsv(path):
 def value(v):
  if v in ('','None'):return None
  try:return float(v)
  except ValueError:return v
 with path.open(newline='',encoding='utf8') as f:return [{k:value(v) for k,v in r.items()} for r in csv.DictReader(f)]
def stats(x):
 a=np.array(x,float);sd=a.std(ddof=1);m=a.mean();w=1.962341461*sd/math.sqrt(len(a));return {'mean':m,'sd':sd,'ci95':[m-w,m+w]}
def fmt(v,n=2):return f'{v:.{n}f}'
def ci(s,scale=1,n=2):return '['+fmt(s['ci95'][0]*scale,n)+', '+fmt(s['ci95'][1]*scale,n)+']'
def ms(s,scale=1,n=2):return fmt(s['mean']*scale,n)+r' $\pm$ '+fmt(s['sd']*scale,n)
def table(name,caption,label,columns,header,rows,wide=False,note=''):
 env='table*' if wide else 'table';s='\\begin{'+env+'}[t]\n\\centering\\footnotesize\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{@{}'+columns+'@{}}\\toprule\n'+header+r'\\\midrule'+'\n'
 s+='\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}\n'
 if note:s+='\n\\par\\smallskip\\begin{minipage}{'+(r'\textwidth' if wide else r'\columnwidth')+'}\\footnotesize '+note+'\\end{minipage}\n'
 s+='\\end{'+env+'}\n';(T/name).write_text(s,encoding='utf8')
main={};all_summary={}
for st,od in [('Normal','accounting_baseline/default_power_time_v1'),('Medium','network_stress_profiles/run_medium'),('High','network_stress_profiles/run_high')]:
 main[st]={}
 for pol,path in [('Original',ROOT/od/'episode_metrics.csv'),('Selector',ROOT/'selective_offloading'/f'run_{st.lower()}'/'episode_metrics.csv')]:
  r=readcsv(path);assert len(r)==1000;main[st][pol]=r
  all_summary[(st,pol)]={k:stats([row[k] for row in r]) for k in ['ue_only_total_energy','total_system_energy','ue_transmission_energy','deadline_violation_fraction','offloaded_fraction']}
baseline=load('comparative_baselines_1000.json')
rows=[]
for st in ['Normal','Medium','High']:
 for pol in ['Original','Selector','Always-Local','Fixed-Edge-0','Greedy-Rule']:
  if pol in ['Original','Selector']:
   d=all_summary[(st,pol)];ue=d['ue_only_total_energy'];sy=d['total_system_energy'];tx=d['ue_transmission_energy'];vi=d['deadline_violation_fraction'];off=d['offloaded_fraction']
  else:
   d=baseline[pol][st];assert d['episodes']==1000;ue=d['ue_energy'];sy=d['total_system_energy'];tx=d['transmission_energy'];vi=d['deadline_violation_rate'];off=d['offloading_ratio']
  rows.append([st,pol,ms(ue),ci(ue),fmt(sy['mean']),fmt(tx['mean']),ms(vi,100),fmt(off['mean']*100)])
table('main_comparison.tex','1,000 saved-workload episodes per policy and condition. UE energy is mean $\\pm$ episode SD; intervals are conditional on the fixed checkpoints.','tab:main','llrrrrrr',r'Stress & Policy & UE energy (J) & UE 95\% CI & System (J) & TX (J) & Violations (\%) & Offload (\%)',rows,True)
rows=[]
for st in main:
 for metric,scale,title in [('ue_only_total_energy',1,'UE (J)'),('deadline_violation_fraction',100,'Viol. (pp)')]:
  a=np.array([r[metric] for r in main[st]['Selector']]);b=np.array([r[metric] for r in main[st]['Original']]);s=stats(a-b)
  rng=np.random.default_rng(20261004);d=a-b;obs=abs(d.mean());extreme=0
  for _ in range(100):
   perm=(rng.choice([-1,1],size=(100,len(d)))*d).mean(axis=1);extreme+=int((np.abs(perm)>=obs-1e-12).sum())
  p=(extreme+1)/10001
  rows.append([st,'Saved 1,000',title,fmt(s['mean']*scale),ci(s,scale),f'{p:.4f}'])
seed=load('main_seed_statistics.json')
for st in seed:
 for metric,scale,title in [('ue_only_total_energy',1,'UE (J)'),('deadline_violation_fraction',100,'Viol. (pp)')]:
  d=seed[st][metric]['paired_difference'];rows.append([st,'5 seeds',title,ms(d,scale),ci(d,scale),fmt(d['exact_signflip_p'],4)])
table('main_statistics.tex','Paired Selector minus Original. Five-workload-seed rows show mean difference $\\pm$ seed SD (ten episodes per seed). Saved-replay tests use 10,000 random sign flips with a plus-one correction.','tab:mainstats','lllrrr',r'Stress & Unit & Metric & Difference & 95\% CI & $p$',rows,True,note='Seeds vary workloads; supplied QECO checkpoints stay fixed. The exact five-seed test has coarse resolution. All tests are exploratory and unadjusted; 1,000-episode intervals estimate conditional workload variability.')
profiles=[]
with (E/'profiles.csv').open(newline='') as f:profiles=list(csv.DictReader(f))
rows=[[r['stress_level'],r['n_windows'],fmt(float(r['monotone_offered_load_multiplier']),4),fmt(float(r['effective_ue_transmission_capacity']),4)] for r in profiles]
table('stress.tex','Traffic-derived profile and simulated service.','tab:stress','lrrr',r'Label & Windows & $M$ & $C_{\rm eff}$',rows)
trace=load('trace_audit.json');rows=[]
for key,purpose in [('train','Training allocation; initial 19 rows consumed'),('val','Reserved; unused in model selection'),('test','Evaluation allocation; brief selected window')]:
 r=trace[key];sec=r['elapsed_seconds'];day=sec//86400;hour=sec%86400//3600;minute=sec%3600//60;seconds=sec%60
 start=r['start'].replace('T',' ')[:19];end=r['end'].replace('T',' ')[:19]
 rows.append([key.capitalize(),str(r['records']),start,end,f'{day}d {hour}h {minute}m {seconds}s',purpose])
table('split.tex','Exact UCLM row partitions. Elapsed durations are end timestamp minus start timestamp, not counts of full days.','tab:split','lrllll',r'Partition & Rows & First timestamp & Last timestamp & Elapsed & Purpose',rows,True)
eh=load('eh_seed_statistics.json');comb=load('combined_seed_statistics.json');rows=[]
for st in eh:
 for enabled,name in [('True','Enabled'),('False','Disabled')]:
  ue=eh[st]['ue_only_total_energy'][enabled];vi=eh[st]['deadline_violation_fraction'][enabled];off=eh[st]['offloaded_fraction'][enabled];soc=eh[st]['terminal_mean_final_soc'][enabled];har=eh[st]['terminal_harvested_charge_mah_per_ue'][enabled]
  rows.append([st,name,ms(ue),ci(ue),ms(vi,100),fmt(off['mean']*100),ms(soc,100,4),fmt(har['mean'],4)])
table('eh_metrics.tex','EH ablation: five independent training seeds, ten episodes per seed. SD and UE intervals are over seed means. Terminal SOC and cumulative harvest use the 110-second endpoint only.','tab:eh','llrrrrrr',r'Stress & Harvest & UE (J) & UE 95\% CI & Viol. (\%) & Offload (\%) & Terminal SOC (\%) & Harvest/UE (mAh)',rows,True,note='Harvest is the inferred external-source proxy. Total system harvest is 20 times the per-UE charge. Both conditions use the same checkpoint and eight-feature observation; Disabled sets harvest input/feature to zero.')
rows=[]
for st in eh:
 for key,label,scale,n in [('ue_only_total_energy','UE energy (J)',1,3),('ue_transmission_energy','TX energy (J)',1,3),('mean_final_soc','Mean endpoint SOC (pp)',100,4),('terminal_mean_final_soc','Terminal SOC (pp)',100,4),('deadline_violation_fraction','Violations (pp)',100,3)]:
  d=eh[st][key]['paired_difference'];rows.append([st,label,ms(d,scale,n),ci(d,scale,n),fmt(d['exact_signflip_p'],4)])
table('eh_differences.tex','Harvest Enabled minus Disabled: paired differences across five training seeds. Endpoint mean averages ten SOC readings; terminal SOC uses the last reading only.','tab:ehdiff','llrrr',r'Stress & Outcome & Difference $\pm$ SD & 95\% CI & Exact $p$',rows,True)
rows=[]
for st,d in comb.items():
 ue=d['ue_only_total_energy'];vi=d['deadline_violation_fraction'];soc=d['terminal_mean_final_soc'];over=d['overrides']
 rows.append([st,ms(ue),ci(ue),ms(vi,100),ms(soc,100,4),fmt(over['mean'],2)])
table('combined.tex',r'Selector + EH-QECO: five training seeds, ten episodes each. SD and energy CIs are seed-level; SOC is terminal after 110\,s.','tab:combined','lrrrrr',r'Stress & UE (J) & UE 95\% CI & Viol. (\%) & Terminal SOC (\%) & Overrides/ep.',rows,True)
cf=load('counterfactual_verified.json');rows=[]
for st,d in cf.items():
 for pol,r in d.items():
  def time(key):return '--' if r[key] is None else fmt(r[key],3)
  rows.append([st,pol,str(r['offloaded_tasks']),fmt(r['local_full_work_energy_j'],3),fmt(r['offload_full_work_tx_plus_task_wait_j'],3),fmt(r['actual_tx_energy_per_task_j'],3),time('local_mean_completion_s_both_feasible'),time('offload_mean_completion_s_both_feasible'),fmt(r['local_predicted_feasible_fraction']*100,1),fmt(r['offload_observed_success_fraction']*100,1)])
table('counterfactual.tex',r'Matched-task audit on 20 paired episodes per condition. Energy is per task; every deadline is 1.0\,s. Completion means use the same both-feasible task subset.','tab:counter','llrrrrrrrr',r'Stress & Policy & Tasks & $E_l^{\rm full}$ & $E_x^{\rm full}+E_w^{\rm obs}$ & Actual TX & $T_l$ & $T_o$ & Local feas. & Offload succ.',rows,True,note='$E_l^{\\rm full}$ is full local service energy in J. The offload column combines full-transmission requirement with observed per-task post-transmission waiting energy; it is not an observed successful full-offload requirement for expired tasks. Actual TX is spent energy, including expired transmissions. $T_l$ is a no-future-arrivals FIFO prediction and $T_o$ is observed successful completion (s), restricted to the intersection of local predicted feasibility and actual offload success. Feasibility/success columns (\\%) use all offloaded tasks. A dash means an empty both-feasible subset, not zero completion time.')
sens=load('sensitivity_verified.json');rows=[[fmt(r['alpha'],3),fmt(r['tx_only_break_even_capacity'],4),fmt(r['break_even_multiplier'],4)] for r in sens['analytical_exponent_crossover']]
table('sensitivity.tex','Analytical transmission-only thresholds at density 0.297.','tab:alpha','rrr',r'$\alpha$ & $C_x^*$ (units/s) & $M_x^*$',rows)
rows=[[fmt(r['capacity_mah'],0),'On' if r['harvest_enabled'] else 'Off',fmt(r['first_ue_zero_hours'],4),fmt(r['mean_soc_below_001percent_hours'],4)] for r in sens['battery_capacity']]
table('battery.tex',r'Deterministic fixed-demand ledger sensitivity at 50\% initial SOC.','tab:battery','rlrr',r'Capacity (mAh) & EH & First empty (h) & Mean low (h)',rows,note='First empty denotes the first UE with zero model charge; Mean low denotes mean SOC at or below 0.01\\%. Fifty Always-Local workload activity traces repeat under continuous demand. These are charge-ledger depletion/threshold times under assumed powers and inferred harvest, not field lifetime or post-depletion task-service measurements.')
# Keep both analytical sensitivities under the manuscript's single input.
with (T/'sensitivity.tex').open('a') as f:f.write('\n'+(T/'battery.tex').read_text())
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(3.5,4.0),layout='constrained');colors=['#323e50','#197a70','#a1783a','#b34946','#6271ae'];x=np.arange(3)
for j,pol in enumerate(['Original','Selector','Always-Local','Fixed-Edge-0','Greedy-Rule']):
 vals=[];vis=[];err=[];verr=[]
 for st in ['Normal','Medium','High']:
  if pol in ['Original','Selector']:ue=all_summary[(st,pol)]['ue_only_total_energy'];v=all_summary[(st,pol)]['deadline_violation_fraction']
  else:ue=baseline[pol][st]['ue_energy'];v=baseline[pol][st]['deadline_violation_rate']
  vals.append(ue['mean']);err.append(ue['ci95'][1]-ue['mean']);vis.append(v['mean']*100);verr.append((v['ci95'][1]-v['mean'])*100)
 axes[0].errorbar(x,vals,yerr=err,label=pol,color=colors[j],marker='o',ms=3,lw=1);axes[1].errorbar(x,vis,yerr=verr,color=colors[j],marker='o',ms=3,lw=1)
axes[0].set_ylabel('UE energy / episode (J)');axes[1].set_ylabel('Deadline violations (%)')
for ax in axes:ax.set_xticks(x,['Normal','Medium','High']);ax.grid(axis='y',alpha=.2)
fig.legend(loc='outside upper center',ncol=2,fontsize=8,frameon=False);fig.savefig(F/'comparison.pdf');fig.savefig(F/'comparison.png',dpi=220);plt.close(fig)
fig,ax=plt.subplots(figsize=(3.5,2.7),layout='constrained');cap=np.linspace(5,18,200)
for rho,col in zip([.197,.297,.397],colors):ax.plot(cap,2.3*2.6/(2*rho*cap),label=f'Density {rho:.3f}',color=col)
ax.axhline(1,color='black',ls='--',lw=.8);ax.axvspan(6.326878,14,color='#d7e6e3',alpha=.4)
ax.set(xlabel='Effective capacity (simulator units/s)',ylabel='Full TX / full local energy',ylim=(.35,3.2));ax.legend(fontsize=8,frameon=False,loc='upper right');fig.savefig(F/'boundary.pdf');fig.savefig(F/'boundary.png',dpi=220);plt.close(fig)
source=PUB/'Resilient_Task_Offloading_IIoT_FINAL.tex';text=source.read_text();text=text.replace('0.35, 0.20, 0.10, 0.10, 0.15, and 0.10','0.40, 0.15, 0.10, 0.10, 0.15, and 0.10').replace('first 100 paired','first 20 paired');source.write_text(text,encoding='utf8')
print('Built',len(list(T.glob('*.tex'))),'table fragments and two figures from released evidence.')
