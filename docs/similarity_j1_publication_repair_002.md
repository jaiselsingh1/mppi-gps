# J1 publication repair 002

The scientific worker, independent audit and admission are closed. Scientific source remains `8c678e507dad1e22708f02a121886d7951d1da44`. J1 stopped at its development variation gate; the protected test was never acquired or evaluated. This repair changes only the saved-data publication boundary. It cannot launch acquisition, inference, gradients, optimizer updates, physics controls or a new predictive test.

The original `render_001` failed before its first frame because C1 stores `done` as uint8 while the atlas requires boolean flags. Its receipt records zero offline restores, position forwards, native controls, physics substeps, neural rows and optimizer steps. Preserve its nonzero actual exits and generic partial-work uncertainty; do not overwrite or reinterpret them as a clean exit.

## Frozen repair

Decode only an authenticated uint8 `done` array whose values are all zero or one into an equal boolean array. Already-boolean flags remain unchanged. Reject other dtypes, missing flags or values outside zero/one. Keep the renderer's shape, timestamp and early-fall checks unchanged. All observations, states, actions, contacts, forces, warmstarts and integration contexts retain their original values and hashes. No state is synthesized after a fall.

Use the original frozen atlas/report implementations, XML and preflight encoder binding. Rendering mode remains `kinematic_schematic`; ffmpeg SHA-256 remains `ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41`. The new external adapter and owner are frozen separately before execution; the current library receives the same explicit storage-to-renderer decoding fix for future invocations. No source inside the preserved run changes.

The roster remains the original role-based selection: all 16 teachers, 16 incoming executions, 32 updated same-context executions and the first predeclared standalone reset for each of the 32 cases: 96 traces. Retain actual timing, partial strides and failures. Do not select views based on results.

## Limits and closure

Create only `render_002` and `report_002` beneath the original artifact namespace. Bind the configuration, source/input manifests, scientific worker/audit/admission, original failed rendering and new adapter hashes. Require clean actual exits, correct ownership and absence of live owned children before admitting publication.

Rendering remains bounded to two hours, 2 GiB RSS, 64,000 frames and a combined 2 GiB output allocation across both rendering attempts. Reserve at least 64 KiB from the second output cap for the failed first attempt. Bound source reads to 2 GiB, including a 64 MiB metadata/source reserve. The unchanged saved-only report retains its original read/RSS/output limits and a 600-second outer wall cap.

Charge every failed and successful publication phase against the original eight-hour shared worker/render/report allowance, including the worker's 3,017.794043475 seconds and failed render operator's 0.318964455 seconds. The original independent audit/finalizer used 18.822127508 seconds of its separate three-hour allocation. Nested measurements are not additive.

Run synthetic C1-uint8-to-atlas and owner/receipt/budget checks before launch. Stop and preserve a second publication failure; never retry science or relax a gate. Publish both the original failure and the repair's actual inputs, adapter, costs, output bindings and exits. A successful publication repair does not alter the scientific verdict or admit guidance.
