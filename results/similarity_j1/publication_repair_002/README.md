# J1 publication repair 002

This is a publication-only envelope. The original scientific source/run, audit,
admission and failed `render_001` remain immutable. The sole data adapter checks
that every saved `done` array is one-dimensional `uint8` with values exactly
0/1 and converts it to boolean. Every other array is passed through unchanged.
No policies, fits, evaluations, teacher acquisition or control steps are repeated.

Deploy these sealed files to `ART/publication_repair_002`, then invoke the pinned
CPU interpreter once after review/calibration:

```text
CPU_PYTHON -B ART/publication_repair_002/repair.py launch --run RUN --execution ART/publication_repair_002/execution_001
```

The create-only `execution_001` directory is the retry gate. The external launch
process waits for and preserves the **actual** publication parent's exit in
`parent_exit.json`; that parent waits for the actual renderer and original
saved-report CLI. Every child binds its actual parent PID, private nonce,
wrapper/protocol hashes and closed scientific/config/input/admission/audit/
failed-render hashes. Root's invocation must preserve the outer launch's actual
exit/stdout/stderr too. Cleanup permits only same-UID, exact argv or pinned
encoder, matching nonce, matching input/code hashes and process-group leaders.

Before rendering, the wrapper checks the original clean scientific admission,
actual reaped failed-render parents/children, the exact zero-frame boolean-codec
error, zero recorded offline/physics/neural/update work, and the unchanged
96-entry roster saved by the failed renderer. Its generic nonzero-exit
uncertainty remains recorded. Both the wrapper and children import renderer,
trace reader, roster selector and reporter from original `RUN/source` at
`8c678e507dad1e22708f02a121886d7951d1da44`. Active source fixes are not used.

The same preflight-pinned mode, XML, encoder, timing, role order and renderer
limits apply. The render child has 7,200 seconds, 2 GiB RSS and the original
64,000-frame cap. The 2 GiB source-read reservation leaves 64 MiB for frozen
source/input/metadata gates; saved-array reads are capped at the remaining
1,984 MiB. Original failed-render bytes, 4 MiB closing/roster metadata and 64 KiB
render stdout are subtracted from the combined 2 GiB render-output cap.

Only after the renderer's actual clean exit, passed atlas and matching child
receipt does the parent call the original frozen saved-only report module for
`ART/report_002`, with `ART/render_002/result` and the original passed audit.
The report has its original saved-input/output limits and a 600-second outer
watchdog. The wrapper checks its source hash. Both phases, all new overhead and
the original worker and failed-render operator-stage times share 28,800 seconds.
Actual earlier receipts are authoritative; approximate preparation values are
not substituted for them. Original audit/finalization costs stay separately
recorded; no scientific audit is rerun.

Outputs are `ART/render_002/result`, `ART/report_002`, actual render/report exits,
ownerships, merged stdout/stderr, process usage, source-read/codec accounting and
publication receipts. The original 16 GiB retained-output ceiling is checked
before launch. Null scientific outcomes stay unchanged. Any gate, bound, codec,
hash, ownership or process failure stops this version without retry.
