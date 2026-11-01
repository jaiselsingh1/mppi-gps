# H5.0 saved-data reporting validation

The [receipt](receipt.json) binds the postprocessor source and successful **1,003-test** suite, with the existing one skip and expected failure. Sixteen synthetic report cases cover missing cohorts, checkpoint aliases, reconstruction/held joins, full/partial curves, prefix labels, consumed-input hashes and true paired gaps. Earlier synthetic-fixture failures are retained.

The postprocessor reads saved JSON only and adds no policy, simulator or optimizer calls to the live experiment. It separates minibatch curves, full-data action MSE (mean across six actions), teacher agreement (sum across six actions, then time mean), walking, full survival, complete cost and true paired teacher–child gaps. Missing children and post-fall observations are not zero-filled. Mechanical selection counts are descriptive, not a usefulness or execution gate.

The scientific worker remains the independently sealed `ff67495` source. This separately versioned reporting code does not change it. Current execution status is in the [launch snapshot](../launch/README.md); no scientific result or audit conclusion is established by this reporting validation.
