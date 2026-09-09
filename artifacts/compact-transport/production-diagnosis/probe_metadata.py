import hashlib,json
from pathlib import Path
import httpx
import home_cortex.ollama as module
from home_cortex.config import Settings
s=Settings(); package=Path(module.__file__).parent
root=Path('/app/benchmarks/fixtures/semantic-contract')
files=sorted(root.rglob('*.json'))
fixture_hash=hashlib.sha256(''.join(str(p.relative_to(root))+':'+hashlib.sha256(p.read_bytes()).hexdigest()+'\n' for p in files).encode()).hexdigest()
print(json.dumps({'ollama_version':httpx.get(s.ollama_url.rstrip('/')+'/api/version').json().get('version'),'source_hashes':{name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in ['ollama.py','semantic_transport.py','semantic_facts.py','semantic_ontology.py']},'synthetic_fixture_manifest_sha256':fixture_hash,'synthetic_fixture_files':len(files)}))
