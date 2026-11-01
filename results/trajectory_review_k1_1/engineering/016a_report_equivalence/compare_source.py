"""Compare generated report bytes only; never read or recompute scientific arrays."""

import hashlib
import json
import re
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path('/Users/jaisel/Documents/mppi-gps')
art = root / 'artifacts/2026-11-01_trajectory_review_k1_1_001'
first, second = art / 'report_001', art / 'report_002'
original = json.loads((first / 'receipt.json').read_text())
corrected = json.loads((second / 'receipt.json').read_text())
assert original['inputs'] == corrected['inputs']
assert original['tables'] == corrected['tables']
assert corrected['scientific_calls'] == original['scientific_calls'] == 0
assert corrected['source_sha256'] == digest(
    art / 'analysis_source_002/src/gps/mppi_gps/reporting/trajectory_review.py'
)
assert original['source_sha256'] == digest(
    root / 'runs/2026-11-01_trajectory_review_k1_1_001/source/src/gps/mppi_gps/reporting/trajectory_review.py'
)
comparisons = {}
for name, binding in corrected['outputs'].items():
    a, b = first / name, second / name
    assert digest(b) == binding['sha256'] and b.stat().st_size == binding['bytes']
    pair = {'byte_identical': a.read_bytes() == b.read_bytes()}
    if b.suffix == '.pdf':
        pattern = rb'/(?:CreationDate|ModDate) \([^)]*\)'
        pair['identical_ignoring_pdf_dates'] = re.sub(pattern, b'', a.read_bytes()) == re.sub(
            pattern, b'', b.read_bytes()
        )
    comparisons[name] = pair
csv_names = [name for name in comparisons if name.endswith('.csv')]
assert len(csv_names) == 7 and all(comparisons[name]['byte_identical'] for name in csv_names)
png_changes = [
    name for name in comparisons if name.endswith('.png') and not comparisons[name]['byte_identical']
]
assert png_changes == ['cumulative_agreement.png']
assert comparisons['report.md']['byte_identical']
assert all(
    value['identical_ignoring_pdf_dates']
    for name, value in comparisons.items()
    if name.endswith('.pdf') and name != 'cumulative_agreement.pdf'
)
print(json.dumps({
    'passed': True,
    'original_report': str(first),
    'corrected_report': str(second),
    'csv_files_exact': csv_names,
    'changed_pngs': png_changes,
    'report_source_sha256': corrected['source_sha256'],
    'comparisons': comparisons,
}, indent=2))
