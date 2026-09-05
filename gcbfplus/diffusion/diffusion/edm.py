"""Single-agent EDM sampler (``--diffusion-method edm``: per-agent sequential sampling).

NOTE: the EDM preconditioning / noise-schedule / denoise scaffolding here is
intentionally duplicated in edm_ma.py (the batched multi-agent sampler, which
extends it with joint acceptance, task repair, and candidate selection). Kept
separate for release stability; factoring the shared helpers into one module
is a known follow-up.
"""
import functools
from distrax import Normal
from flax import struct
from typing import Callable, Optional

from ..util import *

DIFFUSION_TRAINING_CONFIG = TRAINING_CONFIG["diffusion"]

# sample_trajectory is re-traced on every jit cache miss, so its configuration dump
# would print repeatedly. Opt in with GCBF_DIFFUSION_VERBOSE=1.
VERBOSE = os.environ.get("GCBF_DIFFUSION_VERBOSE", "0") == "1"


@struct.dataclass
class DenoiserHyperparams:
    p_mean: float = -1.2  # Mean of log-normal noise distribution
    p_std: float = 1.2  # Standard deviation of log-normal noise distribution
    sigma_data: float = 1.0  # Standard deviation of data distribution
    sigma_min: float = 0.002  # Minimum noise level
    sigma_max: int = 80  # Maximum noise level
    rho: float = 7.0  # Sampling schedule
    edm_first_order: bool = False  # Disable second order Heun integration
    diffusion_timesteps: int = 200  # Number of diffusion timesteps for sampling
    # Stochastic sampling coefficients
    s_tmin: float = 0.05
    s_tmax: float = 50.0
    s_churn: float = 80.0
    s_noise: float = 1.003


# Derived preconditioning params - EDM Table 1
def c_skip(sigma, sigma_data):
    return (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)


def c_out(sigma, sigma_data):
    return sigma * sigma_data * ((sigma_data ** 2 + sigma ** 2) ** -0.5)


def c_in(sigma, sigma_data):
    return (sigma ** 2 + sigma_data ** 2) ** -0.5


def c_noise(sigma):
    return jnp.log(sigma) * 0.25


def train_step(
        rng,
        batch,
        denoiser_state,
        denoiser_hyperparams,
):
    """
    Params:
        data: (batch, seq_len, obs_dim + action_dim + 2)
        ts: (batch,)
    """

    def loss_weight(sigma):
        return (sigma ** 2 + denoiser_hyperparams.sigma_data ** 2) * (
                sigma * denoiser_hyperparams.sigma_data
        ) ** -2

    def seq_loss(denoiser_params, rng, seq):
        # Implements EDM from https://openreview.net/pdf?id=k7FuTOWMOc7
        rng, _rng = jax.random.split(rng)
        sigma = jnp.exp(
            (
                    denoiser_hyperparams.p_mean
                    + denoiser_hyperparams.p_std * jax.random.normal(_rng)
            )
        )

        rng, _rng = jax.random.split(rng)
        noise = jax.random.normal(_rng, shape=seq.shape)
        noised_seq = seq + sigma * noise  # alphas are 1. in the paper
        noise_pred = denoiser_state.apply_fn(
            denoiser_params,
            c_in(sigma, denoiser_hyperparams.sigma_data) * noised_seq,
            c_noise(sigma),
        )
        denoised_pred = (
                c_skip(sigma, denoiser_hyperparams.sigma_data) * noised_seq
                + c_out(sigma, denoiser_hyperparams.sigma_data) * noise_pred
        )
        return jnp.square(denoised_pred - seq) * loss_weight(sigma)

    def batch_loss(denoiser_params):
        _rng = jax.random.split(rng, batch.shape[0])
        return jnp.mean(
            jax.vmap(seq_loss, in_axes=(None, 0, 0))(denoiser_params, _rng, batch)
        )

    loss_val, grad = jax.value_and_grad(batch_loss)(denoiser_state.params)
    denoiser_state = denoiser_state.apply_gradients(grads=grad)
    return denoiser_state, loss_val


def sample_trajectory(
        rng,
        denoiser_state,
        seq_len,
        obs_dim,
        action_dim,
        denoiser_norm_stats,
        denoiser_hyperparams,
        policy_guidance_coeff=0.0,
        policy_guidance_delay_steps=0,
        policy_guidance_cosine_coeff=0.0,
        normalize_action_guidance: bool = True,
        denoised_guidance: bool = False,
        det_guidance: bool = False,
        agent_apply_fn=None,
        agent_params: Optional[dict] = None,
        stl_form_eval: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        stl_guidance: bool = False,
        max_iter_count: int = MAX_SAMPLING_ITER,
        goal_dim: int = 2,
        diffusion_mode: str = "sard",
        use_post_hack: bool = False,  # HACK: Use post hack to apply STL guidance at the end (not recommended)
        resample_batch_size: int = 1,
        skip_resample: bool = False,
):
    """
    Sample a trajectory using EDM diffusion model (for each agent)

    :param rng: PRNGKey
    :param denoiser_state: Denoiser state
    :param seq_len: Length of the trajectory
    :param obs_dim: Observation dimension
    :param action_dim: Action dimension
    :param denoiser_norm_stats: Normalization statistics for denoiser
    :param denoiser_hyperparams: Denoiser hyperparameters
    :param policy_guidance_coeff: Policy guidance coefficient
    :param policy_guidance_delay_steps: Delay steps for policy guidance
    :param policy_guidance_cosine_coeff: Cosine coefficient for policy guidance
    :param normalize_action_guidance: Normalize action guidance
    :param denoised_guidance: Use denoised guidance
    :param det_guidance: Use deterministic guidance (not related to STL guidance)
    :param agent_apply_fn: Agent apply function
    :param agent_params: Agent parameters
    :param stl_form_eval: STL form evaluation function ``(x) -> robustness``, where x is a
        batch of goal trajectories of shape (1, T, goal_dim)
    :param stl_guidance: Use STL guidance
    :param max_iter_count: Maximum iteration count to retry sampling
    :param goal_dim: Goal dimension
    :param diffusion_mode: Diffusion mode to use (sard, sa, sg)
    :param use_post_hack: Use post hack for STL guidance

    """
    if VERBOSE:
        print(f"[edm] STL Guidance: {stl_guidance}")

    # --- Compute noise schedule ---
    # Karras/EDM sigma schedule with num_diffusion_timesteps + 1 entries; the last is
    # overwritten to 0 so the final scan step lands on clean data. gammas below are the
    # per-step stochastic churn, non-zero only for sigma in [s_tmin, s_tmax].
    def _get_noise_schedule(num_diffusion_timesteps):
        inv_rho = 1 / denoiser_hyperparams.rho
        sigmas = (
                         denoiser_hyperparams.sigma_max ** inv_rho
                         + (jnp.arange(num_diffusion_timesteps + 1) / (num_diffusion_timesteps - 1))
                         * (
                                 denoiser_hyperparams.sigma_min ** inv_rho
                                 - denoiser_hyperparams.sigma_max ** inv_rho
                         )
                 ) ** denoiser_hyperparams.rho
        return sigmas.at[-1].set(0.0)  # last step has sigma value of 0.

    sigmas = _get_noise_schedule(denoiser_hyperparams.diffusion_timesteps)
    gammas = jnp.where(
        (sigmas >= denoiser_hyperparams.s_tmin)
        & (sigmas <= denoiser_hyperparams.s_tmax),
        jnp.minimum(
            denoiser_hyperparams.s_churn / denoiser_hyperparams.diffusion_timesteps,
            jnp.sqrt(2) - 1,
        ),
        0.0,
    )

    # Set the trajectory length
    traj_len = obs_dim + action_dim
    if diffusion_mode == "sard":
        # Add 2 dimensions for reward and done
        traj_len += 2

    # --- Construct guidance function ---
    do_apply_guidance = (
                                agent_apply_fn is not None
                                and agent_params is not None
                                and policy_guidance_coeff != 0.0
                        ) or (stl_form_eval is not None)

    def _multi_step_stl_updates(stl_form_eval: Callable, obs, num_stl_iterations, lr=1e-1):

        def update_step(obs):
            return obs - jax.grad(lambda x: -stl_form_eval(x).mean())(obs) * lr

        # Insert a dimension at beginning for batch size

        # jax.scan version
        scan_output = jax.lax.scan(
            lambda obs, _: (update_step(obs), 0),
            obs[None],
            jnp.arange(num_stl_iterations),
        )

        obs = scan_output[0]

        return obs.squeeze(0)

    def _stl_guidance(_stl_form_eval: Callable, traj, num_stl_iterations, lr=1e-3):
        '''Take a step towards the STL objective for one agent.

        traj : jnp.ndarray, (T, obs_dim + action_dim [+ 2])
            the agent's normalised trajectory. Guidance is computed on the goal
            channels, batched up to (1, T, goal_dim) for the STL evaluator.
        '''
        obs = traj[:, :obs_dim]
        obs = unnormalise_traj(obs, denoiser_norm_stats["obs"])
        reshaped_x = obs[:, :goal_dim][None]
        trajectory_guidance = jax.grad(lambda x: _stl_form_eval(x).mean())(reshaped_x)  # Softer guidance
        # --- Normalize and return guidance ---
        if normalize_action_guidance:
            trajectory_guidance = trajectory_guidance / (jnp.linalg.norm(trajectory_guidance) + 1e-8)
        return trajectory_guidance

    def _compute_action_guidance(traj):
        # --- Unnormalize observation ---
        obs = traj[:, :obs_dim]
        obs = unnormalise_traj(obs, denoiser_norm_stats["obs"])

        # --- Compute guidance from policy ---
        pi = agent_apply_fn(agent_params, obs)
        if det_guidance:
            # Apply guidance to unit Gaussian around deterministic action
            agent_action = pi.sample(seed=jax.random.PRNGKey(0))
            pi = Normal(agent_action, 1.0)

        def _transformed_action_log_prob(action):
            action = unnormalise_traj(action, denoiser_norm_stats["action"])
            action = jnp.tanh(action)
            return pi.log_prob(action).sum()

        action = traj[:, obs_dim: obs_dim + action_dim]
        action_guidance = jax.grad(_transformed_action_log_prob)(action)

        # --- Normalize and return guidance ---
        if normalize_action_guidance:
            action_guidance = action_guidance / (jnp.linalg.norm(action_guidance) + 1e-8)
        return action_guidance

    def denoise_step(runner_state, step_coeffs, denoiser_norm_stats, agent_params):
        rng, noised_traj, step_idx = runner_state
        sigma, next_sigma, gamma = step_coeffs

        if do_apply_guidance:
            # --- Compute guidance coefficient ---
            # lambd decays linearly over the denoise steps plus a sine bump
            # (policy_guidance_cosine_coeff), scaled by policy_guidance_coeff.
            n_steps = denoiser_hyperparams.diffusion_timesteps
            lambd = 1.0 - (step_idx / n_steps)
            cosine_adjustment = jnp.sin(jnp.pi * ((step_idx + 1) / n_steps))
            lambd += policy_guidance_cosine_coeff * cosine_adjustment
            do_apply_guidance_this_step = jnp.logical_and(
                step_idx >= policy_guidance_delay_steps,
                step_idx < n_steps - 1,
            )
            lambd = jnp.where(
                do_apply_guidance_this_step, policy_guidance_coeff * lambd, 0.0
            )

            # --- Compute denoised trajectory for guidance ---
            guidance_traj = noised_traj
            if denoised_guidance:
                noise_pred = denoiser_state.apply_fn(
                    denoiser_state.params,
                    c_in(sigma, denoiser_hyperparams.sigma_data) * noised_traj,
                    c_noise(sigma),
                )
                guidance_traj = (
                        c_skip(sigma, denoiser_hyperparams.sigma_data) * noised_traj
                        + c_out(sigma, denoiser_hyperparams.sigma_data) * noise_pred
                )

            # --- Apply guidance ---
            if stl_guidance:
                # Change the state part of the trajectory using STL guidance
                guidance_traj = _stl_guidance(stl_form_eval, guidance_traj, 30, 1e-1)
                state = noised_traj[:, : goal_dim]
                guidance_traj = guidance_traj.squeeze(0)  # Remove batch dimension
                guided_state = state + lambd * guidance_traj  # Take a step towards the guidance

                noised_traj = noised_traj.at[:, : goal_dim].set(
                    guided_state
                )
            else:
                # Below code just changes the action part of the trajectory
                action_guidance = _compute_action_guidance(guidance_traj)
                action = noised_traj[:, obs_dim: obs_dim + action_dim]
                guided_action = action + lambd * action_guidance
                noised_traj = noised_traj.at[:, obs_dim: obs_dim + action_dim].set(
                    guided_action
                )

        # --- Compute first-order EDM denoise step ---
        rng, _rng = jax.random.split(rng)
        eps = denoiser_hyperparams.s_noise * jax.random.normal(_rng, noised_traj.shape)
        sigma_hat = sigma + gamma * sigma
        # gamma == 0: sigma_hat == sigma, so guard sqrt(sigma_hat^2 - sigma^2) against a
        # tiny negative rounding error.
        traj_hat = jnp.where(
            gamma > 0,
            noised_traj + jnp.sqrt(sigma_hat ** 2 - sigma ** 2) * eps,
            noised_traj,
        )
        noise_pred = denoiser_state.apply_fn(
            denoiser_state.params,
            c_in(sigma_hat, denoiser_hyperparams.sigma_data) * traj_hat,
            c_noise(sigma_hat),
        )
        denoised_pred = (
                c_skip(sigma_hat, denoiser_hyperparams.sigma_data) * traj_hat
                + c_out(sigma_hat, denoiser_hyperparams.sigma_data) * noise_pred
        )
        denoised_over_sigma = (traj_hat - denoised_pred) / sigma_hat

        # --- Apply first-order EDM denoise step ---
        denoised_traj = noised_traj + (next_sigma - sigma_hat) * denoised_over_sigma

        # --- Compute EDM second-order correction ---
        if not denoiser_hyperparams.edm_first_order:
            next_noise_pred = denoiser_state.apply_fn(
                denoiser_state.params,
                c_in(next_sigma, denoiser_hyperparams.sigma_data) * denoised_traj,
                c_noise(next_sigma),
            )
            next_denoised_pred = (
                    c_skip(next_sigma, denoiser_hyperparams.sigma_data) * denoised_traj
                    + c_out(next_sigma, denoiser_hyperparams.sigma_data) * next_noise_pred
            )
            denoised_prime_over_sigma = (denoised_traj - next_denoised_pred) / (
                    next_sigma + 1e-9
            )

            # --- Apply second-order EDM denoise step ---
            denoised_traj = jnp.where(
                next_sigma != 0,
                traj_hat
                + 0.5
                * (next_sigma - sigma_hat)
                * (denoised_over_sigma + denoised_prime_over_sigma),
                denoised_traj,
            )

        # Pin index 0 of the plan to the agent's current state (in normalised space).
        normalized_start_state = normalise_traj(
            jnp.zeros_like(denoiser_norm_stats['obs']['mean']).at[:goal_dim].add(agent_params['x0']),
            denoiser_norm_stats['obs'])
        denoised_traj = denoised_traj.at[0, :obs_dim].set(normalized_start_state)

        return (rng, denoised_traj, step_idx + 1), None

    denoised_traj, info, num_resampling_iters, rng = sample_stl_guided_trajectory(_multi_step_stl_updates, agent_params,
                                                                                  denoise_step, denoiser_norm_stats,
                                                                                  gammas, goal_dim, obs_dim,
                                                                                  rng, seq_len, sigmas, stl_form_eval,
                                                                                  stl_guidance, traj_len, use_post_hack,
                                                                                  max_iter_count)

    info.update({'num_resampling_iters': num_resampling_iters})
    # --- Construct rollout ---
    return construct_rollout(
        denoised_traj,
        denoiser_norm_stats,
        obs_dim,
        action_dim,
        diffusion_mode=diffusion_mode
    ), info


def sample_stl_guided_trajectory(_multi_step_stl_updates, agent_params, denoise_step, denoiser_norm_stats, gammas,
                                 goal_dim, obs_dim, rng, seq_len, sigmas, stl_form_eval, stl_guidance,
                                 traj_len, use_post_hack, max_iter_count=MAX_SAMPLING_ITER):
    """Consolidate the resampling of trajectories with STL guidance for the entire system of agents"""

    # Best-so-far merge (GCBF_KEEP_BEST=1, default): keep the previous draw unless the
    # new one strictly improves robustness — same monotone anytime property as edm-ma,
    # so the plan returned at the iteration cap is the least-violating one found.
    keep_best = os.environ.get('GCBF_KEEP_BEST', '1') == '1'

    def body_fn(val):
        """Samples a new denoised trajectory"""
        denoised_traj, guidance_info, rng_arg, iter_count = val
        # --- Sample random noise trajectory ---
        rng_arg, _rng_arg = jax.random.split(rng_arg)
        init_noise = jax.random.normal(_rng_arg, (seq_len, traj_len))
        init_noise *= sigmas[0]

        new_traj, new_info, rng_out = _sample_denoised_trajectory(_multi_step_stl_updates, agent_params, denoise_step,
                                                                  denoiser_norm_stats, gammas, goal_dim, init_noise,
                                                                  rng_arg, sigmas, stl_form_eval, obs_dim,
                                                                  stl_guidance=stl_guidance,
                                                                  use_post_hack=use_post_hack)
        if keep_best and stl_guidance:
            better = new_info['stl_loss'] > guidance_info['stl_loss']
            new_traj = jnp.where(better, new_traj, denoised_traj)
            new_info = jax.tree_util.tree_map(lambda a, b: jnp.where(better, a, b), new_info, guidance_info)
        return new_traj, new_info, rng_out, iter_count + 1

    def condition_fn(val):
        """Condition to continue sampling, while robustness is below STL_RESAMPLE_THRESHOLD"""
        denoised_traj, guidance_info, rng, iter_count = val
        return (guidance_info['stl_loss'] < STL_RESAMPLE_THRESHOLD) & (iter_count < max_iter_count)

    init_guidance_info = {'stl_loss': jnp.array(-1e6), 'stl_loss_pre_hack': jnp.array(-1e6)}
    rng, _rng = jax.random.split(rng)
    init_denoised_traj = jax.random.normal(_rng, (seq_len, traj_len))

    if stl_guidance:
        # Keep running while robustness is below STL_RESAMPLE_THRESHOLD
        # (NOTE: not reverse mode differentiable so remove if training)
        denoised_traj, info, _, num_resampling_iters = jax.lax.while_loop(condition_fn, body_fn,
                                                                          (init_denoised_traj, init_guidance_info, rng,
                                                                           0))
    else:
        denoised_traj, info, _, num_resampling_iters = body_fn(
            (init_denoised_traj, init_guidance_info, rng, 0))
    return denoised_traj, info, num_resampling_iters, rng


def _sample_denoised_trajectory(_multi_step_stl_updates, agent_params, denoise_step, denoiser_norm_stats, gammas,
                                goal_dim, init_noise, rng, sigmas, stl_form_eval, obs_dim, stl_guidance=False,
                                use_post_hack=False):
    # --- Denoise trajectory ---
    denoise_step_scan = functools.partial(denoise_step, denoiser_norm_stats=denoiser_norm_stats,
                                          agent_params=agent_params)
    (rng, denoised_traj, _), _ = jax.lax.scan(
        denoise_step_scan,
        (rng, init_noise, 0),
        (sigmas[:-1], sigmas[1:], gammas[:-1]),

    )
    guidance_info = {}

    if stl_guidance:
        def _calc_stl_loss(x):
            """Helper function to calculate STL loss"""
            return jnp.mean(stl_form_eval(x[:, :goal_dim][None]))

        obs = denoised_traj[:, :obs_dim]
        obs = unnormalise_traj(obs, denoiser_norm_stats["obs"])

        guidance_info = {'stl_loss_pre_hack': _calc_stl_loss(obs)}
        if use_post_hack:
            hacked_traj = _multi_step_stl_updates(stl_form_eval, denoised_traj[:, :goal_dim], 30, 1e-1)
            # normalize and add hacked trajectory
            hacked_traj_zeros = jnp.zeros_like(obs)
            hacked_traj_zeros = hacked_traj_zeros.at[:hacked_traj.shape[0], :hacked_traj.shape[1]].set(hacked_traj)
            hacked_traj = normalise_traj(hacked_traj_zeros, denoiser_norm_stats["obs"])[:, :goal_dim]
            denoised_traj = denoised_traj.at[:, :goal_dim].set(hacked_traj)
        guidance_info.update({'stl_loss': _calc_stl_loss(obs)})
    return denoised_traj, guidance_info, rng
