# Discarded Round 5 runs

## S1_unclipped_diverged

The fully unclipped S1 regime (`clip_norm=0`, `lr=0.1`). **Diverged**: activation
radius grows 45 -> 253 by task ~10, activations go non-finite, and the remaining
~130 tasks log chance accuracy (0.0875). ~430 of 450 spectral metrics per run are
NaN as a result.

Discarded, not deleted, because the divergence is itself a finding: at lr=0.1 the
gradient clip is load-bearing for stability, so you cannot simply remove clipping
to expose the penalty's magnitude effect. Replaced by the `loose` regime
(`clip_norm=10`), which almost never binds but still catches the rare gradient
spikes that cause the blow-up.

Never read these for any result.
