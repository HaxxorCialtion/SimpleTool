from pathlib import Path
import hashlib,json
d=Path(__file__).resolve().parent
j=json.loads((d/'OVERLAY_RELATIVE.json').read_text());out={}
for k,v in j.items():
 p=d/v['relative_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==v['overlay_sha256']
 out[k]={**v,'path':str(p)}
p=d/'runtime/prefill-only-20261110/overlay/manifest.json'
p.write_text(json.dumps(out,indent=2)+'\n')
print('Private overlay rebased; installed vLLM sources untouched')
