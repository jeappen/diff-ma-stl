import jax.lax

from ..util import *


def exponential_schedule(step, total_steps, start_step=0, max_coeff=1.0, k=5.0):
    """
    Computes the coefficient for the achievable loss based on an exponential schedule.

    Args:
        step (int): The current training step.
        total_steps (int): The total number of training steps.
        max_coeff (float): The maximum coefficient value at the end of training.
        k (float): The steepness factor. Larger k means a faster ramp-up.

    Returns:
        float: The current achievable loss coefficient.
    """
    # Normalize step to a value between 0 and 1.
    t = (step - start_step) / total_steps
    # Exponential ramp-up: starts at 0 and asymptotically approaches max_coeff.
    coeff = max_coeff * (1 - jnp.exp(-k * t))
    return coeff


def get_achievable_loss_fn(type_of_loss="env-sync", env=None, rollout_fn=None,
                           rollout_len=None, group_agents_as=None, rng=None,
                           dense_supervision=True):
    """Creates various achievable loss functions for the diffusion model

    :param type_of_loss: which loss to build. ``None`` disables the auxiliary loss
        (returns a constant 0.0). ``"env-sync-train"`` and ``"env-sync-test"`` both
        build the same GCBF+ rollout-tracking loss; the name only records whether it
        is used during training or as sampling-time achievable guidance.
        ``"env-sync-single"`` is unfinished and raises NotImplementedError. Anything
        else raises NotImplementedError.
    :param env: the environment object
    :param rollout_fn: the rollout function
    :param rollout_len: the rollout length
    :param group_agents_as: how many agents to group in calculation (env and rollout_fn must match this)
    :param rng: the random number generator
    :param dense_supervision: if True, use all rollout steps with interpolated targets; if False, use final step only
    """
    if type_of_loss is None:
        return lambda x, *args, **kwargs: 0.0
    if rollout_len is None:
        # Default the achievable loss rollout length to the goal sample interval
        rollout_len = env.goal_sample_interval
    num_agents = env.num_agents
    if rng is None:
        rng = jax.random.PRNGKey(0)

    if "env-sync" in type_of_loss:

        if "env-sync-single" == type_of_loss:
            raise NotImplementedError("finish this with new features")
        else:
            @jax.jit
            @jax.named_scope("achievable_loss_fn")
            def achievable_loss_fn(y, *args, **kwargs):
                """This loss runs the MA controller for a few steps and minimizes the distance to the goal.
                Note: keep y the same shape to prevent JIT errors/recompilation"""

                # Note: It may help to run this loss in the final steps of diffusion since it is more expensive
                # Perhaps a stochastic guidance such as 30% of the time
                # Create graphs from the observations

                def make_graph(states, goals, key):
                    # rng fixed at closure creation: obstacle sampling is identical every
                    # call (deterministic under jit, no recompile).
                    key, obstacles = env.create_all_obstacles(key)
                    states = jnp.concatenate([states, jnp.zeros((num_agents, env.state_dim - env.goal_dim))],
                                             axis=1)
                    # Goals need no zero padding: the EnvState wrapper adds it.
                    env_states = env.EnvState(states, goals, obstacles)
                    return env.get_graph(env_states)

                # Make a graph for each step in the plan with the goal as the next state
                num_graphs = y.shape[1] - 1  # Number of graphs to make (plan length - 1)
                keys = jax.random.split(rng, num=num_graphs)
                graphs = jax.vmap(make_graph, in_axes=(1, 1, 0))(y[:, :-1], y[:, 1:], keys)
                # NOTE: obstacles are re-sampled via env.create_all_obstacles with a fixed
                # key, not taken from the live episode (README default --obs 0).
                # Now rollout the observation graphs as initial state using the MA controller
                # Make a single step plan
                # Goal is (batch, N, 4). Plan should be (batch, Plan len, N, 4)
                current_plan = jnp.expand_dims(graphs.env_states.goal[:, :, :env.goal_dim], (1, 2))
                graphs = graphs._replace(global_time=jnp.zeros(num_graphs),
                                         current_time=jnp.zeros((num_graphs, num_agents),
                                                                dtype=jnp.int32),
                                         current_plan=current_plan)

                sampled_rollout = jax.vmap(rollout_fn, in_axes=(0, None, 0))(graphs, rollout_len, keys)

                extract_nextstate_from_rollout = jax.vmap(
                    lambda x: x.next_graph.type_states_rollout(env.AGENT, num_agents))
                sampled_path = extract_nextstate_from_rollout(sampled_rollout)
                sampled_path = jax.vmap(jax.vmap(env.filter_state))(sampled_path)
                # sampled_path: (num_transitions, rollout_len, num_agents, goal_dim)

                if dense_supervision:
                    # Dense supervision: compare ALL rollout steps against interpolated plan targets
                    # This gives rollout_len× richer gradient signal than final-step-only
                    start_pos = y[:, :-1].transpose(1, 0, 2)  # (T, N, D)
                    end_pos = y[:, 1:].transpose(1, 0, 2)     # (T, N, D)
                    t_fracs = jnp.linspace(0, 1, rollout_len + 1)[1:]  # (RL,) fractions 1/RL..1.0
                    # Interpolated targets: (T, RL, N, D)
                    interp_targets = (start_pos[:, None] +
                                      t_fracs[None, :, None, None] * (end_pos - start_pos)[:, None])
                    achievable_loss = jnp.mean(jnp.square(interp_targets - sampled_path)) / 2
                else:
                    # Sparse supervision: only use final rollout step (original behavior)
                    subsampled_path = sampled_path[:, -1::-rollout_len].squeeze(1)
                    single_step_plan = y[:, 1:].transpose(1, 0, 2)
                    achievable_loss = jnp.mean(jnp.square(single_step_plan - subsampled_path), axis=-1).mean() / 2

                return achievable_loss

        if type_of_loss == "env-sync-train":
            print("Using env-sync-train loss")
            return achievable_loss_fn

    else:
        raise NotImplementedError(f"Loss type {type_of_loss} not implemented")

    if group_agents_as is not None:
        # This groups agents by group_agents_as to reduce costly agent interactions
        def achievable_loss_grouped(_plan):
            """Reshape plan into mini-batches and compute achievable loss."""
            grouped_plan = jnp.reshape(_plan, (-1, group_agents_as, _plan.shape[-2], _plan.shape[-1]))
            return jax.vmap(achievable_loss_fn)(grouped_plan).mean(axis=-1)

        return achievable_loss_grouped

    return achievable_loss_fn
