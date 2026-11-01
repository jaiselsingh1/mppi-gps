"""Create-only H5 seed freshness review over top-level local JSON configs.

No NPZ, checkpoint, raw ledger, policy, optimizer or simulator is imported or
opened. The receipt is evidence of the stated local metadata census, not proof
about unavailable hosts, unrecorded resets or actual draws in old ledgers.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT / 'artifacts/2026-11-01_teacher_proposal_h5_0_preparation_001'
RECIPE = ROOT / 'configs/studies/teacher_proposal_h5_0.json'
MAX_READ_BYTES = 128 * 1024**2
MAX_FILE_BYTES = 8 * 1024**2
MAX_WALL_SECONDS = 60
NATIVE_STOP = 2**32
ROOT_RESERVATION = 1_000_000


def main():
    started = time.perf_counter()
    output = ART / 'seed_review.json'
    if output.exists():
        raise FileExistsError(output)
    forbidden = ('numpy', 'torch', 'mujoco', 'h5py', 'scipy', 'src.envs', 'src.mppi')
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden for name in sys.modules):
        raise RuntimeError('The seed review must have no numerical/ML/simulator imports')
    read_bytes = dict(source=0, recipe=0, historical_configs=0)

    def read(path, role):
        before = path.stat()
        if before.st_size > MAX_FILE_BYTES:
            raise ValueError('Metadata file exceeds the declared per-file limit: ' + str(path))
        if sum(read_bytes.values()) + before.st_size > MAX_READ_BYTES:
            raise ValueError('Metadata read budget exhausted before file read')
        if time.perf_counter() - started > MAX_WALL_SECONDS:
            raise ValueError('Metadata review wall budget exhausted')
        data = path.read_bytes()
        read_bytes[role] += len(data)
        after = path.stat()
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity(before) != identity(after) or len(data) != before.st_size:
            raise ValueError('Metadata file changed during review: ' + str(path))
        return data, dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    _, source_binding = read(Path(__file__).resolve(), 'source')
    data, recipe_binding = read(RECIPE, 'recipe')
    cfg = json.loads(data)
    roster = dict(
        streams=cfg['streams'], primary_seeds=cfg['primary_seeds'],
        confirmation_seeds=cfg['confirmation_seeds'], bootstrap_seed=cfg['bootstrap_seed'],
        global_torch_seed=cfg['learning']['global_torch_seed'],
    )
    proposed = []
    for stream in roster['streams']:
        prefix = '/streams/' + str(stream['id'])
        proposed.append(dict(address=prefix+'/policy_seed', seed=stream['policy_seed'], role='policy_initialization', interval_check=False))
        proposed.append(dict(address=prefix+'/fit_seed', seed=stream['fit_seed'], role='fit_pcg64', interval_check=True))
        for name in ('acquisition_seeds', 'gap_seeds', 'protected_planner_seeds', 'second_planner_seeds', 'gap_protected_planner_seeds', 'gap_second_planner_seeds'):
            for index, seed in enumerate(stream[name]):
                proposed.append(dict(address=prefix+'/'+name+'/'+str(index), seed=seed, role=name, interval_check=True))
    for name in ('primary_seeds', 'confirmation_seeds'):
        proposed.extend(dict(address='/'+name+'/'+str(i), seed=seed, role=name, interval_check=True) for i, seed in enumerate(roster[name]))
    for name in ('bootstrap_seed', 'global_torch_seed'):
        proposed.append(dict(address='/'+name, seed=roster[name], role=name, interval_check=True))
    if any(type(row['seed']) is not int or row['seed'] < 0 for row in proposed):
        raise ValueError('All proposed seeds must be explicit nonnegative Python integers')
    if any(row['interval_check'] and row['seed'] >= NATIVE_STOP for row in proposed):
        raise ValueError('A proposed native/reset/planner seed exceeds uint32')
    duplicates = []
    seen = {}
    for row in proposed:
        if row['seed'] in seen:
            duplicates.append(dict(seed=row['seed'], addresses=[seen[row['seed']], row['address']]))
        seen[row['seed']] = row['address']
    policy = [row for row in proposed if not row['interval_check']]
    native = [row for row in proposed if row['interval_check']]
    files, extracted, collisions = [], [], []
    paths = sorted((ROOT / 'runs').glob('*/config.json'))
    if not paths:
        raise ValueError('No top-level local historical configurations are available')

    def visit(value, address, owner):
        if type(value) is int:
            matches = [row for row in proposed if row['seed'] == value]
            for match in matches:
                collisions.append(dict(file=owner, address=address, historical_value=value, proposed=match, reason='exact integer match anywhere in historical configuration'))
            if 0 <= value < NATIVE_STOP and any(word in address.lower() for word in ('seed', 'root')):
                stop = min(value + ROOT_RESERVATION, NATIVE_STOP)
                extracted.append(dict(file=owner, address=address, value=value, interval_start_inclusive=value, interval_stop_exclusive=stop))
                for match in native:
                    if value < match['seed'] < stop:
                        collisions.append(dict(file=owner, address=address, historical_value=value, reserved_stop_exclusive=stop, proposed=match, reason='conservative seed/root plus one-million-offset reservation'))
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, address+'/'+str(key), owner)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, address+'/'+str(index), owner)

    for path in paths:
        data, binding = read(path, 'historical_configs')
        files.append(binding)
        visit(json.loads(data), '', binding['path'])
    result = dict(
        schema='walker-h5-seed-review-v1', passed=not collisions and not duplicates,
        metadata_only=True, seed_roster=roster, reviewed_utc=datetime.now(timezone.utc).isoformat(),
        source=source_binding, source_sha256=source_binding['sha256'], recipe_snapshot=recipe_binding,
        historical_configurations=files, historical_configuration_count=len(files),
        proposed_seed_bindings=proposed, proposed_seed_count=len(proposed),
        policy_initialization_exact_check=policy, native_seed_interval_check=native,
        historical_seed_root_intervals=extracted, collisions=collisions, proposed_exact_duplicates=duplicates,
        extraction_rules=dict(
            inventory='Sorted local runs/*/config.json only; no nested frozen-source duplicate configurations',
            exact='Every type(value) is int anywhere in each historical JSON is compared with every proposed seed; booleans are excluded',
            intervals='Historical uint32 integers with seed or root anywhere in their JSON-pointer address reserve [value,min(value+1000000,2**32)); exact base matches are reported by the exact rule',
            policy='Policy initialization seeds exceed uint32 and receive a separate exact-integer check only; they are not passed to native-reset interval logic',
            pairing='Shared paired/control panels are represented once in seed_roster; no internal million-offset rule is applied between distinct intentionally adjacent proposed seeds',
        ),
        availability_limits='Only the enumerated top-level local configurations were available to this review. No remote configuration inventory, raw ledger, checkpoint, NPZ, original trajectory or unrecorded-reset history was opened. A passing receipt proves absence of collisions under these explicit local metadata rules, not universal historical freshness.',
        read_bytes=sum(read_bytes.values()), read_bytes_by_role=read_bytes,
        max_read_bytes=MAX_READ_BYTES, max_file_bytes=MAX_FILE_BYTES, max_wall_seconds=MAX_WALL_SECONDS,
        scope='Metadata-only seed/root freshness review; no array decoding, ML/simulator imports, policy initialization, inference, fitting or native physics.',
        work=dict(npz_decodes=0, checkpoint_loads=0, raw_ledger_reads=0, network_forwards=0, native_controls=0, optimizer_steps=0),
        wall_seconds=time.perf_counter()-started,
    )
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=result['passed'], configs=len(files), proposed_seeds=len(proposed), collisions=len(collisions), source_sha256=result['source_sha256'], read_bytes=result['read_bytes'], wall_seconds=result['wall_seconds'])))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
