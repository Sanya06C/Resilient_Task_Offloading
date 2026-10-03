from pathlib import Path
import json
PUB=Path(__file__).resolve().parent
code='''from pathlib import Path
import os, subprocess, sys, json
root = Path.cwd()
if root.name == "publication":
    root = root.parent
assert (root / "publication").is_dir(), "Run from the repository root or publication directory"
os.chdir(root)
for script in ["recompute_statistics.py", "build_tables.py", "make_standalone.py", "check_evidence.py"]:
    subprocess.run([sys.executable, str(root / "publication" / script)], check=True)
'''
cells=[{'cell_type':'markdown','metadata':{},'source':['# Reproduce the audited paper tables\n','This notebook averages nested EH episodes within independent model seeds. It reads the corrected released records, regenerates tables and figures, and verifies evidence integrity. It does not retrain models or run expensive simulations.']}, {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(keepends=True)}, {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['results = json.loads((root / "publication/evidence/eh_seed_statistics.json").read_text())\n','for stress, metrics in results.items():\n','    print(stress, metrics["terminal_mean_final_soc"]["paired_difference"])\n']}]
for i,c in enumerate(cells):c['id']='paper-repro-'+str(i)
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
(PUB/'Reproduce_Paper_Tables.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf8')
p=PUB/'README.md';s=p.read_text();s=s.replace('python publication/battery_capacity_sensitivity.py\n','python publication/battery_capacity_sensitivity.py --trace "C:/path/to/dataset.csv"\n');p.write_text(s,encoding='utf8')
print('Reproduction notebook created.')
