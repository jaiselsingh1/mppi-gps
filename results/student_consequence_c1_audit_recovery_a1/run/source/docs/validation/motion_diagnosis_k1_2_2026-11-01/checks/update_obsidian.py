from pathlib import Path
import json,hashlib,os
v=Path('/Users/jaisel/Documents/Obsidian/labspace/MPPI GPS');b=Path('/Users/jaisel/Documents/workbench/obsidian-backups/2026-11-01-k1-2-closure')
m=json.loads((b/'manifest.json').read_text());old='6ae18cd1d0d846d000a4d44929a25d0316bef75c';new='daf2345782427f765b88b3d1ad1c5e335420895d'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert {p.name for p in v.glob('*.md')}==set(m['files'])
for name,record in m['files'].items():assert sha(v/name)==record['sha256'],name+' changed since backup'
texts={name:(v/name).read_text() for name in m['files']}
name='Research ideas.md';s=texts[name];start=s.index('[S1.1]');end=s.index('\n\n## What should change the teacher?',start)
texts[name]=s[:start]+f"[S1.1](https://github.com/jaiselsingh1/mppi-gps/blob/{new}/docs/student_source_s1_1_results.md) still has no competent student. [K1.2 inspected the actual motions](https://github.com/jaiselsingh1/mppi-gps/blob/{new}/docs/motion_diagnosis_k1_2_results.md): MPPI walks with more irregular local motion; the reference repeats more clearly, yet its demonstrations still do not reliably transfer. Recurrence is not a teaching gate. The next [H5.0 proposal](https://github.com/jaiselsingh1/mppi-gps/blob/{new}/docs/teacher_proposal_h5_0_hypothesis.md) tests guidance inside proposal optimization, preserving independent discovery and judging fixed-update student execution. It is unimplemented and unlaunched; S1.2 is not the default."+s[end:]
changed=[]
for name,text in texts.items():
 if name!='Research Thoughts.md':text=text.replace(old,new)
 if text==(v/name).read_text():continue
 p=v/(name+'.k12.tmp')
 with p.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
 os.replace(p,v/name);changed.append(name)
assert sha(v/'Research Thoughts.md')==m['files']['Research Thoughts.md']['sha256']
assert len(list(v.glob('*.md')))==4
assert texts['Core Idea.md'].split('OT ->',1)[-1]==(v/'Core Idea.md').read_text().split('OT ->',1)[-1]
r={'schema':'walker-motion-obsidian-brief-update-v1','backup':str(b),'link_commit':new,'changed':changed,'notes_after':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(v.glob('*.md'))},'scope':'Four-note structure retained; only one result/recommendation paragraph replaced plus links updated. Scratch OT text and Research Thoughts preserved. Full evidence stays in repository.'}
p=Path('/Users/jaisel/Documents/mppi-gps/artifacts/2026-11-01_motion_diagnosis_k1_2_001/obsidian_update.json')
with p.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r,indent=2))
