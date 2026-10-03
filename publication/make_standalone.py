"""Inline tables, verified bibliography and vector plots for the native LaTeX editor."""
from pathlib import Path
import json,re,unicodedata,csv,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'publication';E=PUB/'evidence'
source=(PUB/'manuscript_template.tex').read_text(encoding='utf8')
source=source.replace('0.35, 0.20, 0.10, 0.10, 0.15, and 0.10','0.40, 0.15, 0.10, 0.10, 0.15, and 0.10').replace('first 100 paired','first 20 paired')
source=re.sub(r'\\input\{(tables/[^}]+)\}',lambda m:(PUB/m[1]).read_text(),source)
source=source.replace(r'\usepackage{microtype}',r'\usepackage{microtype}'+'\n'+r'\usepackage{pgfplots}'+'\n'+r'\pgfplotsset{compat=1.18}'+'\n'+r'\usepgfplotslibrary{groupplots}')
source=source.replace(r'\documentclass[conference]{IEEEtran}',r'\documentclass[conference]{IEEEtran}'+'\n'+r'\usepackage[T1]{fontenc}'+'\n'+r'\usepackage{mathptmx}')
def esc(s):
 s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode()
 return s.replace('&',r'\&').replace('_',r'\_').replace('%',r'\%').replace('#',r'\#')
registry=json.loads((E/'cited_reference_metadata.json').read_text(encoding='utf8'));bib=[]
for e in registry:
 m=e['metadata'];aa=m.get('author',[]);names=[]
 for a in aa[:3] if len(aa)>4 else aa:
  initials='~'.join(w[0]+'.' for w in a.get('given','').split() if w);names.append(esc(initials)+'~'+esc(a.get('family','')))
 author=', '.join(names)+(r' \emph{et al.}' if len(aa)>4 else '')
 year=m.get('published-print',m.get('published',m['issued']))['date-parts'][0][0]
 line='\\bibitem{'+e['key']+'} '+author+', ``'+esc(m['title'][0])+'\'\', '
 line+=('in ' if m['type']=='proceedings-article' else '')+'\\emph{'+esc(m['container-title'][0])+'}'
 if m.get('volume'):line+=', vol.~'+esc(m['volume'])
 if m.get('issue'):line+=', no.~'+esc(m['issue'])
 if m.get('page'):line+=', '+('pp.~' if '-' in m['page'] else 'Art. no.~')+esc(m['page']).replace('-','--')
 elif m.get('article-number'):line+=', Art. no.~'+esc(m['article-number'])
 line+=', '+str(year)+'. DOI: \\href{https://doi.org/'+m['DOI']+'}{\\nolinkurl{'+m['DOI']+'}}.';bib.append(line)
for key,author,title,url in [('UCLMData','UCLM-ARCO','Energy-harvesting dataset','https://github.com/UCLM-ARCO/energy-harvesting-dataset'),('QECOCode','I. Rahmati','QECO implementation','https://github.com/ImanRHT/QECO'),('StudyCode','S. Chavan','Resilient Task Offloading: reproducibility package','https://github.com/Sanya06C/Resilient_Task_Offloading/tree/codex/audited-paper-20261004')]:
 bib.append('\\bibitem{'+key+'} '+author+', ``'+title+'\'\', GitHub repository. [Online]. Available: \\url{'+url+'}. Accessed: Oct. 4, 2026.')
source=source.replace('\\bibliographystyle{IEEEtran}\n\\bibliography{references}','\\begin{thebibliography}{99}\n'+'\n\n'.join(bib)+'\n\\end{thebibliography}')
# Native editor receives no external figure/table files; use pgfplots vector graphics.
baseline=json.loads((E/'comparative_baselines_1000.json').read_text());stress=['Normal','Medium','High'];policies=['Original','Selector','Always-Local','Fixed-Edge-0','Greedy-Rule'];colors=['blue!65!black','teal!80!black','orange!75!black','red!65!black','violet!70!black']
metrics={}
for st,od in [('Normal','accounting_baseline/default_power_time_v1'),('Medium','network_stress_profiles/run_medium'),('High','network_stress_profiles/run_high')]:
 for pol,path in [('Original',ROOT/od/'episode_metrics.csv'),('Selector',ROOT/'selective_offloading'/f'run_{st.lower()}'/'episode_metrics.csv')]:
  with path.open(newline='') as f:rows=list(csv.DictReader(f))
  for k in ['ue_only_total_energy','deadline_violation_fraction']:
   a=np.array([float(r[k]) for r in rows]);metrics[(st,pol,k)]=(a.mean(),1.962341461*a.std(ddof=1)/math.sqrt(len(a)))
plot=['\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=1 by 2,vertical sep=0.9cm},width=\\columnwidth,height=3.6cm,scale only axis=false,xtick={1,2,3},xticklabels={Normal,Medium,High},tick label style={font=\\scriptsize},label style={font=\\scriptsize},ymajorgrids=true,grid style={gray!20},legend style={font=\\scriptsize,draw=none,at={(0.5,1.10)},anchor=south},legend columns=2]']
for k,blkey,scale,label in [('ue_only_total_energy','ue_energy',1,'UE energy / episode (J)'),('deadline_violation_fraction','deadline_violation_rate',100,'Deadline violations (\\%)')]:
 plot.append('\\nextgroupplot[ylabel={'+label+'}]')
 for pol,col in zip(policies,colors):
  coords=[]
  for i,st in enumerate(stress,1):
   if pol in ['Original','Selector']:m,w=metrics[(st,pol,k)]
   else:d=baseline[pol][st][blkey];m=d['mean'];w=d['ci95'][1]-m
   coords.append(f'({i},{m*scale:.8f}) +- (0,{w*scale:.8f})')
  plot.append('\\addplot+[color='+col+',mark=*,mark size=1.2pt,line width=0.65pt,error bars/.cd,y dir=both,y explicit] coordinates {'+' '.join(coords)+'};')
  if k=='ue_only_total_energy':plot.append('\\addlegendentry{'+pol+'}')
plot.append('\\end{groupplot}\n\\end{tikzpicture}')
source=source.replace(r'\includegraphics[width=\columnwidth]{comparison.pdf}','\n'.join(plot))
boundary=r'''\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=5.3cm,xmin=5,xmax=18,ymin=0.35,ymax=3.2,xlabel={Effective capacity (simulator units/s)},ylabel={Full TX / full local energy},label style={font=\scriptsize},tick label style={font=\scriptsize},legend style={font=\scriptsize,draw=none,at={(0.98,0.98)},anchor=north east}]
\addplot[draw=none,fill=teal!8,forget plot] coordinates {(6.326878,0.35) (14,0.35) (14,3.2) (6.326878,3.2)} \closedcycle;
\addplot[black,dashed,domain=5:18,forget plot]{1};
\addplot[blue!65!black,domain=5:18,samples=100]{15.177665/ x};\addlegendentry{Density 0.197}
\addplot[teal!80!black,domain=5:18,samples=100]{10.067340/ x};\addlegendentry{Density 0.297}
\addplot[orange!75!black,domain=5:18,samples=100]{7.531486/ x};\addlegendentry{Density 0.397}
\end{axis}\end{tikzpicture}'''
source=source.replace(r'\includegraphics[width=\columnwidth]{boundary.pdf}',boundary)
source=source.replace('the inherited evaluator randomized','the inherited evaluator randomized')
source=source.replace('Each enabled/disabled condition starts from a new battery/trace instance, and evaluation is greedy.','Each enabled/disabled condition starts from a new battery/trace instance, and evaluation is greedy. The static six-feature energy-preference vector is fixed to the frozen replay vector in all enabled, disabled, and combined conditions. This repairs the inherited evaluator\'s unpaired random allocation of that nuisance observation.')
eh=json.loads((E/'eh_seed_statistics.json').read_text())
txps=[eh[s]['ue_transmission_energy']['paired_difference']['exact_signflip_p'] for s in stress];socps=[eh[s]['terminal_mean_final_soc']['paired_difference']['exact_signflip_p'] for s in stress]
if any(p!=.0625 for p in txps+socps):
 source=source.replace('while every two-sided exact seed-level test of the reported EH energy and reserve effects has $p=0.0625$','while two-sided exact seed-level tests do not establish the reported EH energy and reserve effects at the 0.05 threshold')
 source=source.replace('but exact paired tests have $p=0.0625$','but exact paired tests do not cross the 0.05 threshold')
source=source.replace('The offload full-work attribution adds','The full-transmission-plus-observed-wait attribution adds').replace('full-work local/offload requirements','full-work local and full-transmission-plus-observed-wait requirements')
source=source.replace('full-offload requirement','full-offload requirement')
source=source.replace('gives the first-device and mean-SOC threshold times at 0.01\\% reserve','gives the first-device zero-charge depletion time and mean-SOC threshold time at 0.01\\% reserve')
(PUB/'Resilient_Task_Offloading_IIoT_FINAL.tex').write_text(source,encoding='utf8')
print('Standalone source: tables, citations and plots inlined; no project-file dependencies.')
