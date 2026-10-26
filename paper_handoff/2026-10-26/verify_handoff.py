"""Verify this writing handoff, not the separately referenced original archives."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
expected=json.loads((root/'HANDOFF_SHA256.json').read_text(encoding='utf8'))
actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file() and p.name!='HANDOFF_SHA256.json' and '__pycache__' not in p.parts}
assert actual==expected, {'missing':sorted(set(expected)-set(actual)),
                         'extra':sorted(set(actual)-set(expected)),
                         'changed':[k for k in expected.keys()&actual.keys() if expected[k]!=actual[k]]}
print(f'PASS: {len(actual)} files verified')
