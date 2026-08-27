"""Where the two engine actors are placed on the cluster.

Topology: the small model on one GPU node, the large model on another. Ray
routes each actor by a *node-level resource* that uniquely identifies its node
group, so each actor owns its node's GPUs cleanly — no shared memory, no SM
partitioning.

This module holds only the resource keys. `run_pipeline.py` constructs
the actors (`prorouter/engine.py`) and resolves placement through
`hardware.placement_kwargs`, which drops a constraint the cluster does not
advertise rather than leaving the actor pending forever.

Both keys are empty by default, so placement is unconstrained until you name a
resource your cluster actually advertises. Two ways to set them:

  1. Your cluster already labels its nodes. Point the keys at those labels:

       DRAFT_RESOURCE=<small-model node label> \\
       TARGET_RESOURCE=<large-model node label> python run_pipeline.py ...

     On an autoscaled cluster, prefer the labels already present: adding new
     resource keys with `ray start --resources` after a node is up can make
     the autoscaler treat it as failed and re-provision it.

  2. Self-managed Ray. Tag the workers yourself:

       ray start --address=<head> --resources='{"draft_gpu": 1}'
       ray start --address=<head> --resources='{"target_gpu": 1}'

     then set DRAFT_RESOURCE=draft_gpu TARGET_RESOURCE=target_gpu.

Either way, `run_pipeline.py --small-resource/--large-resource` overrides the
env vars for a single run.

Sizing note: the small model is worth running at tensor-parallel > 1 when its
node allows it. On a single-GPU small-model node the small tier took roughly a
third of end-to-end wall time in our runs; spreading it over four GPUs brought
that into the 12-20% range. Pass `--small-tp` explicitly -- auto-detection
reads the GPUs visible to the *driver*, which on a CPU-only head node is none.
"""

from __future__ import annotations

import os

# Node-level resource keys; see the module docstring.
DRAFT_RESOURCE = os.getenv("DRAFT_RESOURCE", "")
TARGET_RESOURCE = os.getenv("TARGET_RESOURCE", "")

# Conventional names for the self-managed setup above, so callers can fall back
# to them without hardcoding the strings.
LEGACY_DRAFT_RESOURCE = "draft_gpu"
LEGACY_TARGET_RESOURCE = "target_gpu"
