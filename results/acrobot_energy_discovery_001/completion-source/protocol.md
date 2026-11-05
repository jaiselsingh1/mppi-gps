# Near-upright recording completion 1.1R

The [unchanged 1.1 probe](acrobot_energy_discovery_protocol.md) exhausted its
128 MiB near-upright recording allocation after two complete episodes and 429
controls of the third. This is an engineering failure, preserved as
`STOPPED_RESOURCES`; neither its status nor its immutable output is replaced.
The energy worker continues unchanged. No scientific outcome selects a retry.

This storage-only completion retains the two complete episodes and executes
only original start indices 2 and 3. The interrupted third episode has no
resumable controller checkpoint, so its 429-control prefix must be repeated.
Charge that duplication explicitly. Use the original source export at
`05fe64d62f1a5b31c9b4d110067c81404f6f7b66`, original configuration, shared passing
preflight, native environment, full states, planner seeds and all scientific
settings. A separately bound wrapper imports that frozen source. It writes a
create-only `near-upright-completion` directory; it does not edit the old source.

Before start 3, compare the repeated start-2 prefix against the saved original:
states, warmstarts, integration context, pre-action observations, actions and
costs, with `rtol=0, atol=1e-10`; initial state remains byte-exact. Any mismatch,
invalid binding or exhausted allocation stops completion. There is no new
preflight or additional seed. Ordinary height-hold failures remain outcomes.

## Frozen accounting amendment

The original paid near work is 1,429 executed controls, 1,436 model queries,
55,142,400 forecast controls and 110,287,658 physics substeps. Its unknown partial
physics count is zero; its final diagnostic bank may be incomplete. That bank's
missing fields remain missing and cannot be independently certified.

Completion is limited to 635 CPU seconds, 1,400 wall seconds, 128 MiB output,
4 GiB resident memory, 1,000 executed controls, 1,004 queries, 38,553,600 forecast
controls and 77,109,200 physics substeps. Original near CPU 559.860394085 plus
635 remains below its original 1,200-second CPU allocation. The shared
**11,160 CPU-second** and **2 GiB total output** limits remain in force. The
extra output allocation uses unused transport headroom: worker outputs at most
896 MiB and mirrors at most 896 MiB; source, reports and metadata have the
remaining 256 MiB. Transfers stream without retaining a third raw archive.

The original physics ceiling is preserved as a historical limit, not claimed
unchanged. The repeated partial attempt adds at most 16,588,800 forecast controls
and 429 executed controls, or **33,178,458 physics substeps**. The explicitly
amended shared ceiling is **1,543,589,874 physics substeps**, including the
already-paid preflight. No network forwards, gradients or fitting are allowed.

The canonical four-start near-upright result combines old starts 0/1 and completed
starts 2/3. Accounting includes all eleven episode attempts across both probes,
including the interrupted attempt. Independent saved review checks the overlay,
prefix and all paid work. The original height success test and energy kill rule
remain unchanged. No qualification expansion, learning or loop is authorized.
