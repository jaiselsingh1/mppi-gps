# Additional saved-prefix check

This was a post-run saved-evidence check, outside the preregistered numerical
audit. It used no physics, network inference or learning. The receipt records
its actual time and file-byte upper bound. The following is the executed check's
core reduction; it reads the existing archive and never regenerates a bank.

```python
for path in sorted(run.glob("stream_*/r*/search_*/iterations.npz")):
    rows = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as saved:
        for i, row in enumerate(rows):
            lengths = saved[f"p{i:03d}_lengths"]
            full = np.flatnonzero(lengths == 1000)
            expected = np.array([0]) if i == 0 else np.array([], dtype=int)
            assert np.array_equal(full, expected)
            assert row["candidate_survives"] == len(full)
```

All 1,843,200 stored acquisition prefix lengths were checked. Exactly 72
reached H1000, each at the supplied pass0/sample0 incumbent. This verifies
consistency of the preserved prefix evidence and logged census; it does not
independently reconstruct the physical states of unpreserved sample banks.
