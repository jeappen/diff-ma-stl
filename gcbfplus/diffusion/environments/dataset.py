import h5py

from ..util import *


def load_dataset(args, normalize, val_split=0.0, unbounded_actions=False):
    """Load and normalize train and validation datasets"""
    trajs, val_trajs = _load_dataset(args, val_split=val_split, debug=args.debug)
    if normalize:
        transformed_traj = trajs._replace(
            # Note: This action normalization is not compatible with actions with norm greater than 1.0
            action=jnp.arctanh(jnp.clip(trajs.action, -0.999, 0.999))) if not unbounded_actions else trajs
        trajs, trajectory_norm_stats = _normalize_dataset(
            transformed_traj)
        if val_trajs is not None:
            transformed_val_traj = val_trajs._replace(
                # Note: This action normalization is not compatible with actions with norm greater than 1.0
                action=jnp.arctanh(jnp.clip(val_trajs.action, -0.999, 0.999))) if not unbounded_actions else val_trajs
            val_trajs = _normalize_from_stats(transformed_val_traj, trajectory_norm_stats, )
    obs_dim, num_actions = trajs.obs.shape[-1], trajs.action.shape[-1]
    if normalize:
        return trajs, val_trajs, trajectory_norm_stats, (obs_dim, num_actions)
    return trajs, val_trajs, (obs_dim, num_actions)


def _load_gcbfplus_data(args):
    """Load a GCBF+ HDF5 trajectory dataset in Jax Numpy format, split on done flags."""

    # Dataset files are named "<EnvName>_N<...>_<timestamp>.h5" (see
    # RolloutToDataset.rollouts_to_d4rl_dataset), so the env prefix comes from the
    # requested dataset name; with no --dataset_name we fall back to the release
    # default and pick the first matching file.
    env_name = args.dataset_name.split('_')[0] if args.dataset_name else 'DubinsCar'
    # find all files starting with the env_name in the datasets folder
    files_list = os.listdir('datasets')
    files_list = [f for f in files_list if f.startswith(env_name) and f.endswith('.h5')]
    if args.dataset_name is not None:
        print(f"Loading the file {args.dataset_name}")
        file_name = args.dataset_name
    else:
        print(f"Loading the first file {files_list[0]}")
        file_name = files_list[0]
    with h5py.File(f'datasets/{file_name}', 'r') as dataset_h5py:
        # TODO: Consider replacing actions with goals
        KEYS2LOAD = ['observations', 'actions', 'rewards', 'terminals', 'timeouts']
        # Dataset keys
        print(f"Dataset keys: {dataset_h5py.keys()}")
        print(f"Loading keys {KEYS2LOAD}")

        # Replace dataset of type h5py to a dictionary with Jax Numpy arrays and keys oppo
        dataset = {k: jnp.array(dataset_h5py[k]) for k in KEYS2LOAD}

        # if dataset time out has >2D shape, assume it is the goal and terminals given
        def _check_if_goals_given_as_timeouts(dataset):
            if len(dataset['timeouts'].shape) > 1:
                return True
            return False

        if _check_if_goals_given_as_timeouts(dataset):
            print("Goals given as timeouts, replacing actions with goals")
            print("Assuming no timeouts")
            if args.diffusion_trajectory_mode == 'sg':
                # Assuming goals have the same dimension as actions (x, y)
                dataset["actions"] = dataset["timeouts"][:, :dataset['actions'].shape[1]]
            dataset["timeouts"] = jnp.zeros_like(dataset['terminals'])
        else:
            assert not args.diffusion_trajectory_mode == 'sg', "Goals not given as timeouts"
            # otherwise, assume time outs are ignored (old dataset where terminals are not given)
            # Legacy datasets store the episode-end flag under 'terminals'; swap so
            # _assemble_dataset treats it as a timeout, not a terminal.
            dataset["terminals"], dataset["timeouts"] = dataset["timeouts"], dataset["terminals"]

    # TODO: load this into policy diff

    trajs = {
        attr: dataset[attr][:-1]
        for attr in ["observations", "actions", "rewards", "terminals", "timeouts"]
    }
    trajs["next_observations"] = dataset["observations"][1:]
    trajs["done"] = jnp.logical_or(dataset["terminals"][:-1], dataset["timeouts"][:-1])
    trajs = jtu.tree_map(jnp.array, trajs)

    # --- Split data on terminal or timeout flags ---
    # reshape(-1), not squeeze(): a single-episode dataset has one done flag and squeeze()
    # would collapse it to a 0-d array.
    split_idxs = jnp.argwhere(trajs["done"]).reshape(-1) + 1
    # Omit final index if present
    if split_idxs.size and split_idxs[-1] == len(trajs["done"]):
        split_idxs = split_idxs[:-1]
    trajs = jtu.tree_map(lambda x: jnp.array_split(x, split_idxs), trajs)

    # --- Return list of episode dicts ---
    return [{k: v[i] for k, v in trajs.items()} for i in range(len(split_idxs) + 1)]


def _subsample_dataset(rollout_dict, num_samples, sample_at_goals=False):
    """Subsample dataset with given rate"""
    if sample_at_goals:
        # Subsample at goal changes (actions are goals in this case when diffusion_trajectory_mode is sg)
        change = jnp.linalg.norm(rollout_dict['actions'][1:] - rollout_dict['actions'][:-1], axis=1)
        change_points = jnp.where(change > CLOSE_TO_ZERO)[0] + 1
        # Add the first and last points
        change_points = jnp.concatenate([jnp.array([0]), change_points, jnp.array([len(rollout_dict['actions']) - 1])])
        subsample_rate = len(rollout_dict['actions']) // num_samples

        # Make a list of goal change filling in the gaps with equally spaced points
        #
        def true_fn(_change_points, _rollout_dict):
            equally_spaced_points = jnp.arange(len(_rollout_dict['actions']))[::-subsample_rate]
            equally_spaced_points_to_add = jax_setdiff1d(equally_spaced_points, _change_points)[
                                           :num_samples - len(_change_points)]
            return jnp.unique(jnp.concatenate([_change_points, equally_spaced_points_to_add]))

        def false_fn(_change_points, _rollout_dict):
            return jnp.unique(_change_points)

        if len(change_points) < num_samples:
            # Python if, not lax.cond: jax_setdiff1d returns a data-dependent shape.
            goal_change_idxs = true_fn(change_points, rollout_dict)
        else:
            goal_change_idxs = false_fn(change_points, rollout_dict)

        return jtu.tree_map(lambda x: x[goal_change_idxs], rollout_dict)
    else:
        subsample_rate = len(rollout_dict['actions']) // num_samples
        # Take from the end of the episode
        return {k: v[::-subsample_rate][:num_samples][::-1] for k, v in rollout_dict.items()}


def _load_dataset(args, val_split=0.0, debug=False):
    """
    Loads a flattened HDF5 trajectory dataset.

    Episodes are concatenated together,
    then split into args.trajectory_length around done flags.
    """
    # --- Load training and validation episodes ---
    print("Loading GCBFPlus dataset", end="...")
    eps = _load_gcbfplus_data(args)

    if debug or args.tiny_dataset:
        # Make tiny dataset for testing
        print(f"Number of episodes: {len(eps)}, Making tiny dataset with 100 episodes")
        eps = eps[:100]
    # TODO: Fix subsampling and dataset vs generation
    _subsample_dataset_val = partial(_subsample_dataset, num_samples=args.stl_train_traj_len,
                                     sample_at_goals=args.sample_goal_change)
    eps = list(map(_subsample_dataset_val, eps))  # last episode buggy
    if debug:
        # Plot all X-Y trajectories into a single plot
        import matplotlib.pyplot as plt
        for i, ep in enumerate(eps):
            if i % 10 == 0:
                plt.plot(ep['observations'][:, 0], ep['observations'][:, 1], alpha=0.3)
                # Also mark the starting point with a red dot
                plt.plot(ep['observations'][0, 0], ep['observations'][0, 1], 'ro')
        plt.title('Trajectories Over Time')
        plt.xlabel('X Coordinates')
        plt.ylabel('Y Coordinates')
        plt.axis('equal')  # Equal scaling for X and Y axes
        plt.grid(True)
        wandb.log({"trajectories": plt})
        plt.savefig('trajectories.png')

    if val_split > 0.0:
        num_val_eps = int(val_split * len(eps))
        print(
            f"found {len(eps)} episodes, splitting off {num_val_eps} for validation.",
        )
        assert (
                num_val_eps > 0
        ), f"Val split {val_split} too small given {len(eps)} episodes"
        val_ep_idxs = jax.random.choice(
            jax.random.PRNGKey(args.seed),
            len(eps),
            shape=(num_val_eps,),
            replace=False,
        )
        val_eps = [eps[i] for i in val_ep_idxs]
        eps = [ep for i, ep in enumerate(eps) if i not in val_ep_idxs]
    else:
        print(f"found {len(eps)} episodes, no validation set.")

    def _assemble_dataset(eps):
        """
        Assemble subtrajectory dataset from list of episodes.

        Subtrajectories have length args.trajectory_length,
        with args.dataset_stride stride across dataset.

        Subtrajectories never reset at intermediate steps, or timeout at
        any step (done flag corresponds to terminal only).
        """
        if args.trajectory_length > 1:
            # --- Concatenate episodes and find global episode start indices ---
            print("Assembling dataset", end="...")
            flat_done = jnp.concatenate([ep["done"] for ep in eps], axis=0)
            done_idxs = jnp.argwhere(flat_done).squeeze(axis=-1)
            if done_idxs[-1] == len(flat_done) - 1:
                done_idxs = done_idxs[:-1]
            init_idxs = jnp.concatenate([jnp.zeros(1), done_idxs + 1], axis=0)

            # --- Compute subtrajectory indices without intermediate episode resets ---
            any_done = jax.jit(partial(jnp.convolve, mode="valid"))(
                a=jnp.ones(args.trajectory_length - 1), v=flat_done[:-1]
            )
            valid_start_idxs = jnp.argwhere(any_done == 0).squeeze(axis=-1)

            # --- Compute subtrajecories ending with terminal or timeout ---
            flat_term = jnp.concatenate([ep["terminals"] for ep in eps], axis=0)
            term_idxs = jnp.argwhere(flat_term).squeeze(axis=-1)
            flat_timeout = jnp.concatenate([ep["timeouts"] for ep in eps], axis=0)
            timeout_idxs = jnp.argwhere(flat_timeout).squeeze(axis=-1)
            print(
                f"{len(term_idxs)} terminal, {len(timeout_idxs)} timeout flags found",
                end="...",
            )
            term_idxs -= args.trajectory_length - 1
            timeout_idxs -= args.trajectory_length - 1

            # --- Compute subtrajectory indices ---
            # Add strided subtrajectories
            start_idxs = set(valid_start_idxs[:: args.dataset_stride].tolist())
            # Add the start and end (final step terminal) of episodes
            start_idxs |= set(valid_start_idxs.tolist()) & set(term_idxs.tolist())
            start_idxs |= set(valid_start_idxs.tolist()) & set(init_idxs.tolist())
            # Remove subtrajectories ending in timeout
            start_idxs -= set(timeout_idxs.tolist())
            # Compute index array from list of start positions
            start_idxs = jnp.array(list(start_idxs), dtype=jnp.int32)
            subtraj_idxs = jax.jit(
                jax.vmap(lambda x: jnp.arange(args.trajectory_length) + x)
            )(start_idxs)
        else:
            # --- Remove timeout transitions ---
            flat_timeout = jnp.concatenate([ep["timeouts"] for ep in eps], axis=0)
            subtraj_idxs = jnp.argwhere(~flat_timeout).squeeze(axis=-1)

        # --- Construct subtrajectories from indices ---
        def _construct_tensor(data, add_singleton=False):
            # --- Construct Jax Numpy array from subtrajectory indices ---
            ret = jnp.concatenate(data, axis=0)
            ret = jnp.take(ret, subtraj_idxs, axis=0)
            if add_singleton:
                # Add singleton dimension
                return jnp.expand_dims(ret, axis=-1)
            return ret

        trajectories = Transition(
            obs=_construct_tensor([ep["observations"] for ep in eps]),
            action=_construct_tensor([ep["actions"] for ep in eps]),
            reward=_construct_tensor([ep["rewards"] for ep in eps], add_singleton=True),
            next_obs=_construct_tensor([ep["next_observations"] for ep in eps]),
            done=_construct_tensor([ep["terminals"] for ep in eps], add_singleton=True),
            value=None,
            log_prob=None,
            info=None,
        )
        print(f"done ({len(subtraj_idxs)} subtrajectories constructed).")
        print(f"Number of terminals: {jnp.sum(trajectories.done)}")
        assert ~jnp.any(
            trajectories.done[:, :-1]
        ), "Done flags in the middle of subtrajectory"
        return trajectories

    # --- Return assembled training and validation datasets ---
    return (
        _assemble_dataset(eps),
        _assemble_dataset(val_eps) if val_split > 0.0 else None,
    )


def _normalize_dataset(trajs):
    """Normalize observations, actions, rewards and done flags"""
    obs, obs_norm_mean, obs_norm_std = normalise_traj(trajs.obs)
    obs_stats = {"mean": obs_norm_mean, "std": obs_norm_std}
    next_obs = normalise_traj(trajs.next_obs, obs_stats)
    action, action_norm_mean, action_norm_std = normalise_traj(trajs.action)
    reward, reward_norm_mean, reward_norm_std = normalise_traj(trajs.reward)
    done, done_norm_mean, done_norm_std = normalise_traj(trajs.done)
    trajectory_norm_stats = {
        "obs": obs_stats,
        "action": {"mean": action_norm_mean, "std": action_norm_std},
        "reward": {"mean": reward_norm_mean, "std": reward_norm_std},
        "done": {"mean": done_norm_mean, "std": done_norm_std},
    }
    return (
        trajs._replace(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            next_obs=next_obs,
        ),
        trajectory_norm_stats,
    )


def _normalize_from_stats(trajs, stats):
    """Normalize observations, actions, rewards and done flags with given statistics"""
    return trajs._replace(
        obs=normalise_traj(trajs.obs, stats["obs"]),
        next_obs=normalise_traj(trajs.next_obs, stats["obs"]),
        action=normalise_traj(trajs.action, stats["action"]),
        reward=normalise_traj(trajs.reward, stats["reward"]),
        done=normalise_traj(trajs.done, stats["done"]),
    )
