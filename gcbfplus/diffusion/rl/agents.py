"""Actor-type registry.

Actors listed here produce deterministic actions; the rollout generator uses
this to decide whether guidance needs stochastic-policy handling.
"""

DETERMINISTIC_ACTORS = ["td3_bc"]
