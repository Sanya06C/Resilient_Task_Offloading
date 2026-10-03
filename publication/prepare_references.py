from pathlib import Path
import json,re,requests,unicodedata
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'publication/evidence'
registry=json.loads((OUT/'reference_registry_audit.json').read_text(encoding='utf8'))
keys=['Mali2024_s3','Islam2021_supp11','Wang2020_supp14','Yang2023_supp18','Mao2016_s2','Min2017_s18','Mei2023_s23','Qin2021_s31','DoDuy2022_s17','Jiang2023_s30','Chi2024_s26','Donovan2022_supp8','Singh2024_supp3','Mirani2024_supp7','QECO','Traffic','UCLM']
selected=[]
for key in keys:
 e=next(x for x in registry if x['key']==key)
 if 'metadata' not in e:
  r=requests.get('https://api.crossref.org/works/'+requests.utils.quote(e['doi'],safe=''),timeout=45);r.raise_for_status();e['metadata']=r.json()['message'];e['verification']='DOI metadata retrieved on retry'
 selected.append(e)
def esc(x):
 x=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode()
 for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:x=x.replace(a,b)
 return x.replace('–','--').replace('—','--')
lines=[]
for e in selected:
 m=e['metadata'];authors=' and '.join(esc(a.get('family',''))+', '+esc(a.get('given','')) for a in m.get('author',[]))
 # Use the assigned publication year; an online-first date is not the volume year.
 year=m.get('published-print',m.get('published',m['issued']))['date-parts'][0][0]
 typ='inproceedings' if m['type']=='proceedings-article' else 'article'
 fields={'author':authors,'title':'{'+esc(m['title'][0])+'}','year':year,'doi':m['DOI'],'url':'https://doi.org/'+m['DOI']}
 fields['booktitle' if typ=='inproceedings' else 'journal']=esc(m.get('container-title',[''])[0])
 for key,field in [('volume','volume'),('issue','number'),('page','pages')]:
  if m.get(key):fields[field]=esc(m[key]).replace('-','--') if field=='pages' else esc(m[key])
 if m.get('article-number') and not m.get('page'):fields['pages']=m['article-number']
 lines.append('@'+typ+'{'+e['key']+',\n'+',\n'.join('  '+k+' = {'+str(v)+'}' for k,v in fields.items())+'\n}\n')
lines.extend([
'@misc{UCLMData, author={UCLM-ARCO}, title={{Energy-harvesting dataset}}, howpublished={GitHub repository}, year={2018}, url={https://github.com/UCLM-ARCO/energy-harvesting-dataset}, note={Accessed: October 4, 2026}}',
'@misc{QECOCode, author={Rahmati, Iman}, title={{QECO implementation}}, howpublished={GitHub repository}, url={https://github.com/ImanRHT/QECO}, note={Accessed: October 4, 2026}}',
'@misc{StudyCode, author={Chavan, Sanya}, title={{Resilient Task Offloading: reproducibility package}}, howpublished={GitHub repository}, url={https://github.com/Sanya06C/Resilient_Task_Offloading}, note={Accessed: October 4, 2026}}'])
(ROOT/'publication/references.bib').write_text('\n'.join(lines),encoding='utf8')
(OUT/'cited_reference_metadata.json').write_text(json.dumps(selected,indent=2),encoding='utf8')
print('Verified cited references',len(selected),'+ 3 software/dataset entries')
