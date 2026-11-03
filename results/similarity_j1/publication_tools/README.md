# J1 compact publication collector 002

This version extends the sealed 001 publication collector; its base file is
byte-identical (`39fd627ed345916b5acf03ab326be91e91431226a2e5bdabf66bb9545a25053b`).
The transfer verifier, preview tar streamer and fresh materialization helper are
unchanged. No scientific implementation, array, checkpoint or outcome changes.

Deploy these manifest-listed files unchanged into a fresh artifact tools
directory outside `RUN`. The CLI remains:

```text
CPU_PYTHON -B TOOLS/collector.py metadata --run RUN --out ART/publication_compact_002 --source-commit 8c678e507dad1e22708f02a121886d7951d1da44 --config-sha256 2d436b3ff0d51bfe4afd41a15c1dfe49bd9a0473aa8d7cdda916388570369876
```

It first checks the original actual closed worker/audit/failed-render owners.
Before selecting any 002 atlas/report, it additionally requires the repair's
actual clean parent/render/report exits, matching parent PIDs and argv, private
nonces, unchanged wrapper/protocol and source/config/input/admission/audit/
failed001 hashes, the same 96-item roster and done-codec receipt, original source
reporter hash, combined two-GiB rendering limit, and complete shared eight-hour
wall accounting. An active or unclosed repair is rejected. A collection pass
means byte collection, never scientific success. A failed repair is preserved
at its remote location and cannot become a successful publication alias.

The successful `render_002` metadata uses `atlas/`, and `report_002` uses
`report/` in the compact capsule. Original failed `render_001` metadata uses
`failed_render_001/`; any earlier report uses `prior_report_001/`. Old operator
receipts and every new repair source/protocol/ownership/actual-exit/result/log
are retained. Original scientific evidence remains under `run/`. The 001
generic nonzero-exit uncertainty and its zero-work atlas evidence remain intact.

All existing compact limits remain: metadata reads reserve 112 MiB, metadata plus
preview hashing share 128 MiB, each remote/local compact copy is at most 56 MiB,
and ignored local previews reserve 8 MiB. Raw arrays, checkpoints, ledgers,
indices and full atlas stay remote. Existing closed scientific audit hashes are
reused with explicit provenance; unknown checksums stay unknown. Full curves
and CSV are preserved losslessly, with both stored and original SHA-256 hashes.
There is no permanent tar file or remote preview-video copy.

After successful metadata closure, invoke the same fixed-role preview catalog:

```text
CPU_PYTHON -B TOOLS/collector.py previews --run RUN --out ART/publication_preview_catalog_002 --metadata-capsule ART/publication_compact_002 --source-commit 8c678e507dad1e22708f02a121886d7951d1da44 --config-sha256 2d436b3ff0d51bfe4afd41a15c1dfe49bd9a0473aa8d7cdda916388570369876
```

The exact same three declared roles now read from the admitted `render_002`.
Stream metadata tar directly under 64 MiB, and stream previews using
`stream_previews.py --catalog CATALOG --manifest-sha256 SHA --exit-sha256 SHA`
under 8 MiB. Preserve actual stream exits/stdout/stderr and verify extracted files
with `verify_capsule.py --capsule LOCAL --manifest-sha256 SHA --exit-sha256 SHA
--receipt FRESH_RECEIPT.json`. Keep a single local compact content copy; move a
verified staging directory instead of duplicating it.

For a runnable report replay, use `materialize.py --capsule LOCAL_COMPACT --out
FRESH_REPLAY --manifest-sha256 SHA --exit-sha256 SHA`. It verifies stored/original
hashes and bounded gzip expansion into a fresh ignored directory, with no source
capsule mutation. Then run the original source reporter manually:

```text
LOCAL_PYTHON -B -m src.gps.mppi_gps.reporting.similarity --run FRESH_REPLAY --out FRESH_REPORT --audit FRESH_REPLAY/audit/result --atlas FRESH_REPLAY/atlas/result
```

Fresh decoded/report directories are separate publication engineering outputs;
count their actual bytes and costs. Full remote atlas media are deliberately
absent from the compact replay. Present the verified fixed-role local previews
and explicit external evidence location alongside the compact report.

The 001 base's 17 synthetic checks and materializer remain frozen. This version's
additional synthetic joins/namespace/closed-gate/preview checks plus the copied
materializer tests run without any actual scientific input, physics or fitting.
