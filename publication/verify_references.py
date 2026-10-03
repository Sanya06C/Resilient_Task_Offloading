"""Verify the inherited registry against DOI registration metadata; never guess fields."""
from pathlib import Path
import json,re,time,requests
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'publication/evidence'; OUT.mkdir(exist_ok=True)
text=Path(r'D:\research\clean_build_test\references_zotero_full.bib').read_text(encoding='utf-8')
entries=[]
for block in re.split(r'(?=@\w+\{)',text):
 m=re.match(r'@(\w+)\{([^,]+),',block)
 if not m: continue
 fields={k.lower():v.strip() for k,v in re.findall(r'^\s*(\w+)\s*=\s*\{(.*)\},?\s*$',block,re.M)}
 entries.append({'key':m[2],**fields})
entries += [{'key':'QECO','title':'QECO: A QoE-Oriented Computation Offloading Algorithm Based on Deep Reinforcement Learning for Mobile Edge Computing','doi':'10.1109/TNSE.2025.3556809'}, {'key':'Traffic','title':'Malware communication in smart factories: A network traffic data set','doi':'10.1016/j.comnet.2024.110804'}, {'key':'UCLM','title':'A Testbed and an Experimental Public Dataset for Energy-Harvested IoT Solutions','doi':'10.1109/INDIN41052.2019.8972219'}]
def get(e):
 doi=e.get('doi','')
 if not doi:return {**e,'verification':'no DOI in inherited registry'}
 try:
  r=requests.get('https://api.crossref.org/works/'+requests.utils.quote(doi,safe=''),timeout=40)
  r.raise_for_status(); meta=r.json()['message']
  from difflib import SequenceMatcher
  norm=lambda t:re.sub('[^a-z0-9]','',t.lower())
  similarity=SequenceMatcher(None,norm(e['title']),norm(meta['title'][0])).ratio()
  return {**e,'verification':'DOI metadata retrieved','title_similarity':similarity,'metadata':meta,'source':'https://api.crossref.org/works/'+doi}
 except Exception as ex:return {**e,'verification':'retrieval failed','error':str(ex)}
with ThreadPoolExecutor(max_workers=6) as pool:
 results=list(pool.map(get,entries))
(OUT/'reference_registry_audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
for e in results:
 print(e['key'],e['verification'],round(e.get('title_similarity',0),3),e.get('metadata',{}).get('title',[''])[0])
print('Registry entries',len(results))
