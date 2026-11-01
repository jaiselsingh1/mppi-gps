"""Make ordered overview PNGs of every atlas page; no scientific arrays read."""

import csv
import hashlib
import io
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT/'artifacts/2026-11-01_motion_diagnosis_k1_2_001'
QA = ART/'visual_qa'
started = time.perf_counter()
source = ART/'report_001/trajectories.csv'
source_data = source.read_bytes()
rows = list(csv.DictReader(io.StringIO(source_data.decode())))
assert len(rows)==34
out = QA/'contact_sheets_001'
out.mkdir(exist_ok=False)
font = ImageFont.load_default(size=16)
small = ImageFont.load_default(size=14)
inputs = {str(source):dict(bytes=len(source_data),sha256=hashlib.sha256(source_data).hexdigest())}
read_bytes = len(source_data)
outputs = {}
for kind in ('raw','derived'):
    folder = QA/(kind+'_pages_001')
    expected = json.loads((folder/'receipt.json').read_text())['outputs']
    for offset in range(0,34,6):
        sheet = Image.new('RGB',(1392,1042),'white')
        draw = ImageDraw.Draw(sheet)
        stop = min(offset+6,34)
        heading = ('FULL RAW MOTION' if kind=='raw' else 'SAVED BLOCK MEAN / MIN / MAX')
        draw.text((16,12),f'{heading} - pages {offset+1}-{stop} of34',fill='black',font=font)
        for position,index in enumerate(range(offset,stop)):
            page = folder/f'page-{index+1:02d}.png'
            data = page.read_bytes()
            binding = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            assert binding==expected[page.name]
            inputs[str(page)]=binding
            read_bytes += len(data)
            image = Image.open(io.BytesIO(data)).convert('RGB')
            image.thumbnail((448,448),Image.Resampling.LANCZOS)
            x,y = 16+(position%3)*464,48+(position//3)*492
            sheet.paste(image,(x,y))
            draw.rectangle((x,y,x+448,y+448),outline='#999999',width=1)
            row = rows[index]
            draw.text((x,y+451),f'p{index+1:02d} {row["source"]}',fill='black',font=small)
            draw.text((x,y+470),f'{row["id"]}; N={row["N"]}',fill='black',font=small)
        path = out/f'{kind}_{offset+1:02d}_{stop:02d}.png'
        sheet.save(path)
        outputs[path.name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
receipt = dict(schema='walker-motion-atlas-contact-sheets-v1',
               scope='Ordered visual overview of all68 rasterized pages; unchanged source PDF/PNG bytes. Thumbnails are for layout review, not numeric reconstruction.',
               pages_per_atlas=34,sheets=len(outputs),inputs=inputs,outputs=outputs,
               logical_source_image_read_bytes=read_bytes,
               wall_seconds=time.perf_counter()-started,
               scientific_calls=dict(network_forwards=0,physics_controls=0,optimizer_steps=0,mppi_queries=0))
with (out/'receipt.json').open('x') as stream:
    json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps({key:receipt[key] for key in ('pages_per_atlas','sheets','logical_source_image_read_bytes','wall_seconds')},indent=2))
