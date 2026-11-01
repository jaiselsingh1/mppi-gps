from pathlib import Path
import hashlib,json
v=Path('/Users/jaisel/Documents/Obsidian/labspace/MPPI GPS');b=Path('/Users/jaisel/Documents/workbench/obsidian-backups/2026-11-01-k1-2-closure');art=Path('/Users/jaisel/Documents/mppi-gps/artifacts/2026-11-01_motion_diagnosis_k1_2_001')
m=json.loads((b/'manifest.json').read_text());old='6ae18cd1d0d846d000a4d44929a25d0316bef75c';new='daf2345782427f765b88b3d1ad1c5e335420895d'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert set(m['files'])=={p.name for p in v.glob('*.md')}
for n in ['Core Idea.md','Conversation Points.md']:assert (v/n).read_text()==(b/n).read_text().replace(old,new)
assert (v/'Research Thoughts.md').read_bytes()==(b/'Research Thoughts.md').read_bytes()
s=(b/'Research ideas.md').read_text();before,rest=s.split('[S1.1]',1);_,after=rest.split('\n\n## What should change the teacher?',1)
current=(v/'Research ideas.md').read_text();assert current.startswith(before.replace(old,new));assert current.endswith(('\n\n## What should change the teacher?'+after).replace(old,new));assert '[K1.2 inspected the actual motions]' in current and '[H5.0 proposal]' in current
r={'schema':'walker-motion-obsidian-brief-update-v1','backup':str(b),'link_commit':new,'changed':['Core Idea.md','Conversation Points.md','Research ideas.md'],'notes_after':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(v.glob('*.md'))},'scope':'Four-note structure retained. Core/Conversation changed only commit links; Research ideas changes only one result/recommendation paragraph and links; Research Thoughts exact bytes preserved.','first_verification_failure':{'source':'update_obsidian.py','type':'AssertionError','reason':'Final post-write check assumed the user scratch marker OT -> was in Core Idea; it is in Research Thoughts. All intended writes had completed. No note content was restored/overwritten to pass the check.','verification':'Read-only exact-backup comparisons now prove intended changes only.','wall_seconds':None},'verification_source_sha256':sha(Path(__file__))}
with (art/'obsidian_update.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r,indent=2))
