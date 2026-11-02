from pathlib import Path
import hashlib,json,os,time
started=time.perf_counter();root=Path('/Users/jaisel/Documents/mppi-gps');art=root/'artifacts/2026-11-01_teacher_proposal_h5_0_r1_001';prep=art/'obsidian_preparation_001';m=json.loads((prep/'manifest.json').read_text());v=Path(m['vault']);backup=Path('/Users/jaisel/Documents/workbench/obsidian-backups/2026-11-01-h5-r1-closure-001')
def sha(data):return hashlib.sha256(data).hexdigest()
notes={p.name:p.read_bytes() for p in v.glob('*.md')};name=m['file'];assert sha(notes[name])==m['before_sha256'];new=(prep/name).read_bytes();assert sha(new)==m['after_sha256']
backup.mkdir(parents=True,exist_ok=False)
for n,data in notes.items():
 with (backup/n).open('xb') as f:f.write(data)
manifest={'files':{n:{'bytes':len(data),'sha256':sha(data)} for n,data in notes.items()},'vault':str(v),'scope':'Exact backup before one-paragraph H5 closure update.'}
with (backup/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
assert all((v/n).read_bytes()==data for n,data in notes.items())
tmp=v/(name+'.h5-closure-001.tmp')
with tmp.open('xb') as f:f.write(new);f.flush();os.fsync(f.fileno())
assert (v/name).read_bytes()==notes[name]
os.replace(tmp,v/name)
assert {p.name for p in v.glob('*.md')}==set(notes)
for n,data in notes.items():assert (v/n).read_bytes()==(new if n==name else data)
assert b'OT ->' in notes['Research Thoughts.md'] and (v/'Research Thoughts.md').read_bytes()==notes['Research Thoughts.md']
r={'schema':'walker-h5-brief-obsidian-update-v1','backup':str(backup),'vault':str(v),'changed':[name],'link_commit':m['link_commit'],'before':manifest['files'],'after':{n:{'bytes':(v/n).stat().st_size,'sha256':sha((v/n).read_bytes())} for n in notes},'note_count_before':len(notes),'note_count_after':len(notes),'single_existing_paragraph_replaced':True,'Kendall_Ben_core_and_scratch_points_preserved':True,'all_other_notes_exact_bytes_unchanged':True,'old_paragraph_words':m['old_words'],'new_paragraph_words':m['new_words'],'new_scientific_work':0,'wall_seconds':time.perf_counter()-started,'scope':'One brief conclusion/link replacement; full evidence stays in repository. Read/write/fsync/backup work only, no research. File IO counted by explicit payload sizes; library/stat/interactive overhead not fully metered.'}
with (art/'obsidian_update_001.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r,indent=2))
