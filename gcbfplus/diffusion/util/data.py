import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

DIFFUSION_TRAJECTORY_MODES = ['sard', 'sa', 'sg']  # sard: state, action, reward, done (default)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def stack_transitions(trajectory: Transition, diffusion_mode='sard'):
    if diffusion_mode == 'sard':
        return jnp.concatenate(
            [trajectory.obs, trajectory.action, trajectory.reward, trajectory.done], axis=-1
        )
    elif diffusion_mode == 'sa' or diffusion_mode == 'sg':
        return jnp.concatenate(
            [trajectory.obs, trajectory.action], axis=-1
        )
    else:
        raise ValueError(f"Unknown diffusion mode {diffusion_mode}")


def unstack_transitions(dataset: jnp.ndarray, obs_dim: int, action_dim: int, diffusion_mode='sard'):
    if diffusion_mode == 'sard':
        return Transition(
            obs=dataset[..., :-1, :obs_dim],
            action=dataset[..., :-1, obs_dim: obs_dim + action_dim],
            reward=dataset[..., :-1, obs_dim + action_dim: obs_dim + action_dim + 1],
            done=dataset[..., :-1, obs_dim + action_dim + 1: obs_dim + action_dim + 2],
            next_obs=dataset[..., 1:, :obs_dim],
            info=None,
            log_prob=None,
            value=None,
        )
    elif diffusion_mode == 'sa' or diffusion_mode == 'sg':
        return Transition(
            obs=dataset[..., :-1, :obs_dim],
            action=dataset[..., :-1, obs_dim: obs_dim + action_dim],
            reward=None,
            done=None,
            next_obs=dataset[..., 1:, :obs_dim],
            info=None,
            log_prob=None,
            value=None,
        )
    else:
        raise ValueError(f"Unknown diffusion mode {diffusion_mode}")


@partial(jax.vmap, in_axes=-1, out_axes=-1)
def normalise_traj(trajectories, stats=None):
    """Normalize trajectory dimension and return statistics if not provided"""
    if stats is None:
        mean = jnp.mean(trajectories)
        std = jnp.std(trajectories)
        std = jnp.where(std == 0.0, 1.0, std)
        return (trajectories - mean) / std, mean, std
    return (trajectories - stats["mean"]) / stats["std"]


@partial(jax.vmap, in_axes=-1, out_axes=-1)
def unnormalise_traj(trajectories, stats):
    return trajectories * stats["std"] + stats["mean"]


def construct_rollout(
        denoised_traj,
        denoiser_norm_stats,
        obs_dim,
        action_dim,
        diffusion_mode='sard',
        unbounded_action=True  # Default was false for original environments
):
    rollout = unstack_transitions(denoised_traj, obs_dim, action_dim, diffusion_mode)
    action = unnormalise_traj(rollout.action, denoiser_norm_stats["action"])
    if not unbounded_action:
        action = jnp.tanh(action)
    if diffusion_mode == 'sard':
        done = unnormalise_traj(rollout.done, denoiser_norm_stats["done"])
        done = jnp.greater(done, 0.5).astype(jnp.float32)
        return Transition(
            obs=unnormalise_traj(rollout.obs, denoiser_norm_stats["obs"]),
            action=action,
            reward=unnormalise_traj(rollout.reward, denoiser_norm_stats["reward"]),
            done=done,
            next_obs=unnormalise_traj(rollout.next_obs, denoiser_norm_stats["obs"]),
            value=None,
            log_prob=None,
            info=None,
        )
    elif diffusion_mode == 'sa' or diffusion_mode == 'sg':
        return Transition(
            obs=unnormalise_traj(rollout.obs, denoiser_norm_stats["obs"]),
            action=action,
            reward=None,
            done=None,
            next_obs=unnormalise_traj(rollout.next_obs, denoiser_norm_stats["obs"]),
            value=None,
            log_prob=None,
            info=None,
        )
    else:
        raise ValueError(f"Unknown diffusion mode {diffusion_mode}")
