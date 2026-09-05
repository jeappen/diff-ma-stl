import functools
import os
import jax.lax
from flax import struct
from typing import Callable, NamedTuple, Optional

from .losses import get_achievable_loss_fn, exponential_schedule
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
    achievable_loss_coeff: float = 1.0  # Achievable loss coefficient


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
        auxiliary_training_loss=None,
        env=None, sample_rollout_fn=None,
        exponential_loss_k=5.0,
        total_training_steps=1000,
        auxiliary_loss_last_fraction=0.4,
        auxiliary_loss_rollout_length=None,
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
        return denoised_pred, jnp.square(denoised_pred - seq) * loss_weight(sigma)

    # NOTE: the training-time achievable loss rolls out all agents jointly;
    # --auxiliary_group_agents_as grouping is sampling-only.
    achievable_loss = get_achievable_loss_fn(auxiliary_training_loss, env, sample_rollout_fn,
                                             rollout_len=auxiliary_loss_rollout_length,
                                             rng=rng)

    def batch_loss_w_aux(denoiser_params, step=0, _batch=batch):
        _rng = jax.random.split(rng, _batch.shape[0])
        denoised_outputs, losses = jax.vmap(seq_loss, in_axes=(None, 0, 0))(denoiser_params, _rng, _batch)
        _rng, _ = jax.random.split(rng, 2)
        denoised_outputs = denoised_outputs[:, :, :2]  # goal channels are assumed to be the first 2 (x, y)
        achievable_loss_val = achievable_loss(denoised_outputs, denoiser_params, _rng)

        exponential_coeff = exponential_schedule(step, total_training_steps,
                                                 start_step=int(
                                                     total_training_steps * (1 - auxiliary_loss_last_fraction)),
                                                 max_coeff=denoiser_hyperparams.achievable_loss_coeff,
                                                 k=exponential_loss_k)

        return jnp.mean(losses) + exponential_coeff * achievable_loss_val

    def batch_loss_no_aux(denoiser_params, step=0, _batch=batch):
        _rng = jax.random.split(rng, _batch.shape[0])
        denoised_outputs, losses = jax.vmap(seq_loss, in_axes=(None, 0, 0))(denoiser_params, _rng, _batch)
        return jnp.mean(losses)

    def batch_loss(params, step, _batch):
        # Use jax.lax.cond to choose the loss function based on the traced step value.
        return jax.lax.cond(
            step > (1 - auxiliary_loss_last_fraction) * total_training_steps,
            batch_loss_w_aux, batch_loss_no_aux, params, step, _batch
        )

    loss_val, grad = jax.value_and_grad(batch_loss, argnums=0)(denoiser_state.params, denoiser_state.step, batch)

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
        stl_form_eval: Optional[Callable[[jnp.ndarray, int], jnp.ndarray]] = None,
        stl_guidance: bool = False,
        max_iter_count: int = MAX_SAMPLING_ITER,
        goal_dim: int = 2,
        diffusion_mode: str = "sard",
        use_post_hack: bool = False,  # BROKEN — do not enable (wrong-axis slice + stale-obs scoring; guarded below)
        rollout_fn: Optional[Callable] = None,
        achievable_guidance: bool = False,
        achievable_loss_coeff: float = 0.0,
        agent_mask_noise: bool = False,
        env: Optional = None,
        ma_stl_form_eval: Optional[Callable] = None,
        ma_stl_loss_coeff: float = 1.0,
        auxiliary_training_loss: Optional[str] = None,
        auxiliary_loss_rollout_length: Optional[int] = None,
        alternate_update: bool = False,
        alternate_every: int = 8,
        resample_batch_size: int = 1,
        skip_resample: bool = False,
        ach_guidance_after_fraction: float = ACH_GUIDANCE_AFTER_FRACTION,
        smooth_loss_coeff: float = 0.0,
        dense_achievable_supervision: bool = True,
        guidance_hardness: float = 100.0,
        segment_stl_check: bool = False,
        segment_stl_coeff: float = 0.5,
        segment_interp_points: int = 5,
        avoid_regions: list = None,
        global_stl_only: bool = False,
        task_inner_eval_fns: Optional[tuple] = None,
        task_ms: Optional[tuple] = None,
        task_branches: Optional[tuple] = None,
        ma_stl_gate_eval: Optional[Callable] = None,
):
    """
    Sample a trajectory using EDM-Multi Agent diffusion model

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
    :param stl_form_eval: STL form evaluation function ``(x, agent_id) -> robustness``, where x
        is a batch of goal trajectories of shape (1, T, goal_dim) and agent_id selects the
        per-agent specification
    :param stl_guidance: Use STL guidance
    :param max_iter_count: Maximum iteration count to retry sampling
    :param goal_dim: Goal dimension
    :param diffusion_mode: Diffusion mode to use (sard, sa, sg)
    :param use_post_hack: Use post hack for STL guidance
    :param rollout_fn: Rollout function to use (if provided). Useful for achievable loss computation
    :param achievable_guidance: Use achievable guidance
    :param achievable_loss_coeff: Achievable loss coefficient
    :param agent_mask_noise: Mask noise for agents
    :param env: Environment
    :param ma_stl_form_eval: Multi-agent STL specification
    :param ma_stl_loss_coeff: Multi-agent STL loss coefficient
    :param auxiliary_training_loss: Auxiliary training loss type
    :param alternate_update: Alternate update between STL and MA STL guidance
    :param alternate_every: Alternate every n steps
    :param resample_batch_size: accepted for compatibility with logged commands; it is a
        no-op (the in-loop batch path it once selected has been removed)
    :param skip_resample: Skip the resampling loop and just do one pass (for speed)

    """
    if VERBOSE:
        if guidance_hardness != 100.0:
            print(f"[edm-ma] Segment STL guidance hardness: {guidance_hardness} "
                  f"(applied via custom softmin, not global)")
        if rollout_fn is not None:
            print("[edm-ma] Got a rollout function")
        print(f"[edm-ma] STL Guidance: {stl_guidance} | agent_mask_noise: {agent_mask_noise} | "
              f"skip_resample: {skip_resample}")

    auxiliary_group_agents_as = None  # If None, then no grouping in aux loss calculation
    if achievable_guidance:
        if VERBOSE:
            print(f"[edm-ma] Achievable Guidance: {achievable_guidance} | Achievable Loss Coeff: "
                  f"{achievable_loss_coeff} | Auxiliary Training Loss: {auxiliary_training_loss} | "
                  f"Auxiliary Loss Rollout Length: {auxiliary_loss_rollout_length}")

        if auxiliary_loss_rollout_length is None:
            auxiliary_loss_rollout_length = env.goal_sample_interval
        assert rollout_fn is not None, "Rollout function must be provided for achievable guidance"
        assert achievable_loss_coeff > 0.0, "Achievable loss coefficient must be positive to minimize"
        # Simple check for grouped agent mode below (agent_params) is from env
        num_agents = agent_params['x0'].shape[0]
        if env.num_agents != num_agents:
            if VERBOSE:
                print("[edm-ma] NOTE: Number of agents in environment and agent_params do not match; "
                      f"assuming grouped {num_agents} agents for achievable guidance with "
                      f"{env.num_agents} agents in group")
            auxiliary_group_agents_as = env.num_agents

    if ma_stl_form_eval is not None:
        assert ma_stl_loss_coeff >= 0.0, "MA STL loss coefficient must be positive to maximize"
        if VERBOSE:
            print(f"[edm-ma] MA STL Spec: {ma_stl_form_eval} | MA STL Loss Coeff: {ma_stl_loss_coeff} | "
                  f"Alternate Update: {alternate_update} | Alternate Every: {alternate_every}")

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

    num_agents = agent_params['x0'].shape[0]

    # Counting-aware repair context (CaTL+ disjunctive mode). Static per spec.
    task_ctx = None
    if task_inner_eval_fns is not None:
        n_tasks = len(task_inner_eval_fns)
        if task_branches:
            branch_masks = tuple(tuple(q in ids for q in range(n_tasks)) for ids in task_branches)
        else:
            branch_masks = (tuple(True for _ in range(n_tasks)),)
        task_ctx = TaskRepairCtx(tuple(task_inner_eval_fns), tuple(task_ms), branch_masks)

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

    def _stl_guidance(_stl_form_eval: Callable, traj, num_stl_iterations, lr=1e-3, rollout_fn=rollout_fn,
                      env=env,
                      _ma_stl_form_eval=None, _time2alternate_update=False, _time2achievable_guidance=False,
                      _agent_assign=None):
        '''Take a step towards the STL objective for every agent at once.

        traj : jnp.ndarray, (N, T, obs_dim + action_dim [+ 2])
            the normalised joint trajectory. Guidance is computed on the (N, T, goal_dim)
            goal channels and returned with that shape.
        '''
        obs = traj[:, :, :obs_dim]  # Extract observation part
        obs = unnormalise_traj(obs, denoiser_norm_stats["obs"])  # Unnormalize the observation
        reshaped_x = obs[:, :, :goal_dim]  # Extract goal part for guidance (X-Y coordinates)
        # Bottom line allows different stl forms for each agent
        loss_fns = []
        loss_fn_coeffs = []
        loss_names = []
        loss_values = {}
        # Per-agent STL loss (skip if global_stl_only — rely on CaTL+ only for guidance)
        if global_stl_only:
            # Global-only mode: CaTL+ gradient is the sole guidance signal
            loss_values.update({'stl': jnp.nan})
        elif not (_time2alternate_update and _ma_stl_form_eval is not None):
            if _agent_assign is not None:
                # Assignment-targeted per-agent loss (counting-aware repair): an
                # assigned agent's gradient pulls toward its CLAIMED task's inner
                # STL; free agents keep the disjunctive OR. Each task's inner form
                # is agent-batch independent, so the gradient stays agent-local.
                def plan_stl_loss(y):
                    task_rho = jnp.stack([fn(y) for fn in task_ctx.inner_eval_fns])
                    or_vals = jax.vmap(_stl_form_eval)(y[:, None, :, :], jnp.arange(y.shape[0])).reshape(-1)
                    return _assigned_loss(_agent_assign, task_rho, or_vals).mean()
            else:
                # STL loss for each agent
                plan_stl_loss = lambda y: jax.vmap(_stl_form_eval)(y[:, None, :, :], jnp.arange(y.shape[0])).mean()
            loss_fns.append(plan_stl_loss)
            loss_fn_coeffs.append(1.0)  # Positive because we want to maximize
            loss_names.append('stl')
        else:
            # Dummy loss for consistent dictionary size
            loss_values.update({'stl': jnp.nan})
        if achievable_guidance:
            # Now achievable stl loss
            if _time2achievable_guidance:
                achievable_stl_loss = get_achievable_loss_fn(auxiliary_training_loss, env, rollout_fn,
                                                             rollout_len=auxiliary_loss_rollout_length,
                                                             rng=rng, group_agents_as=auxiliary_group_agents_as,
                                                             dense_supervision=dense_achievable_supervision)
                loss_fns.append(achievable_stl_loss)
                loss_fn_coeffs.append(-achievable_loss_coeff)  # Negative because we want to minimize
                loss_names.append('achievable')
            else:
                loss_values.update({'achievable': jnp.nan})
        if _ma_stl_form_eval is not None:
            if not _time2alternate_update and alternate_update:
                # Dummy loss for consistent dictionary size while updating Plan STL loss
                loss_values.update({'ma_stl': jnp.nan})
            else:
                # jnp.mean reduction: jax.grad needs a scalar loss, and the acceptance
                # path already tolerates (1,)-shaped CaTL+ robustness the same way.
                ma_scalar_loss = lambda y: jnp.mean(_ma_stl_form_eval(y))
                loss_fns.append(ma_scalar_loss)
                loss_fn_coeffs.append(ma_stl_loss_coeff)  # Positive because we want to maximize
                loss_names.append('ma_stl')
        if smooth_loss_coeff > 0:
            # Smoothness regularizer: penalizes trajectory curvature (acceleration)
            # Complements env-sync-test with global gradient coverage across entire plan horizon
            def smooth_loss_fn(y):
                dy = y[:, 1:] - y[:, :-1]         # velocity proxy (N, T-1, D)
                ddy = dy[:, 1:] - dy[:, :-1]      # acceleration proxy (N, T-2, D)
                return jnp.mean(ddy ** 2) / 2
            loss_fns.append(smooth_loss_fn)
            loss_fn_coeffs.append(-smooth_loss_coeff)  # Negative because we want to minimize curvature
            loss_names.append('smooth')
        if segment_stl_check and avoid_regions is not None and any(len(r) > 0 for r in avoid_regions):
            # Segment-based avoidance: interpolate between consecutive waypoints
            # and penalize points that enter avoid regions. Uses custom softmin with
            # controllable hardness for distributed gradients (bypasses JIT-baked HARDNESS).
            #
            # avoid_regions: list[list[(center, size)]] per agent. Pad to uniform shape.
            max_regions = max(len(r) for r in avoid_regions)
            n_ag = len(avoid_regions)
            goal_d = reshaped_x.shape[-1]
            # Pad: (N, R_max, D) for centers and sizes; mask for valid regions
            _centers = jnp.zeros((n_ag, max_regions, goal_d))
            _sizes = jnp.ones((n_ag, max_regions, goal_d))  # ones so invalid regions have large margin
            _mask = jnp.zeros((n_ag, max_regions), dtype=bool)
            for ai, regions in enumerate(avoid_regions):
                for ri, (c, s) in enumerate(regions):
                    _centers = _centers.at[ai, ri].set(jnp.array(c[:goal_d]))
                    _sizes = _sizes.at[ai, ri].set(jnp.array(s[:goal_d]))
                    _mask = _mask.at[ai, ri].set(True)

            def segment_avoid_loss_fn(y):
                n_interp = segment_interp_points
                t_fracs = jnp.linspace(0, 1, n_interp + 2)[1:-1]
                starts = y[:, :-1]  # (N, T-1, D)
                ends = y[:, 1:]
                interp = (starts[:, :, None, :] +
                          t_fracs[None, None, :, None] * (ends - starts)[:, :, None, :])
                N, Tm1, K, D = interp.shape
                interp_flat = interp.reshape(N, Tm1 * K, D)  # (N, P, D)

                # Per-agent, per-region robustness: (N, P, R)
                diff = jnp.abs(interp_flat[:, :, None, :] - _centers[:, None, :, :])
                margin = diff - _sizes[:, None, :, :] / 2
                per_region_rob = jnp.max(margin, axis=-1)  # (N, P, R) max over dims

                # Mask invalid regions with large positive value (safe)
                per_region_rob = jnp.where(_mask[:, None, :], per_region_rob, 1e6)

                # Min over regions per point (worst violation)
                per_point_rob = jnp.min(per_region_rob, axis=-1)  # (N, P)

                # Softmin over interpolated points with controllable hardness
                H = guidance_hardness
                weights = jax.nn.softmax(-per_point_rob * H, axis=-1)
                soft_min_vals = jnp.sum(per_point_rob * weights, axis=-1)  # (N,)
                return soft_min_vals.mean()

            loss_fns.append(segment_avoid_loss_fn)
            loss_fn_coeffs.append(segment_stl_coeff)  # Positive: maximize avoidance robustness
            loss_names.append('segment_stl')

        # Single fused forward+backward: each loss term is evaluated exactly once, and
        # its value is reported through has_aux rather than recomputed for loss_values —
        # a second forward would double the expensive achievable rollout per denoise step.
        def _total_with_parts(y):
            parts = {name: fn(y) for name, fn in zip(loss_names, loss_fns)}
            total = sum([coeff * parts[name] for name, coeff in zip(loss_names, loss_fn_coeffs)])
            return total, parts
        (_, part_values), trajectory_guidance = jax.value_and_grad(_total_with_parts, has_aux=True)(reshaped_x)
        loss_values.update(part_values)

        # A non-finite guidance gradient must never poison the denoise scan: drop
        # guidance for that step instead. (XLA GPU priority-fusion in jaxlib 0.5.0
        # miscompiles the CaTL+ softmax-approx VJP at some vmap batch shapes — e.g.
        # num_candidates in {9,11,12,13,15,16} with the N=16 team formula on RTX
        # 3070; the resulting NaN otherwise nukes every candidate, keep-best never
        # accepts (NaN > prev is False), and the resample loop silently delivers
        # the raw init-noise carry.)
        trajectory_guidance = jnp.nan_to_num(trajectory_guidance)

        # --- Normalize and return guidance ---
        if normalize_action_guidance:
            # Normalize in a per agent manner to avoid agent-agent interference (IMPORTANT FOR MIXED SPECS)
            trajectory_guidance = (trajectory_guidance /
                                   (jnp.linalg.norm(trajectory_guidance, axis=(1, 2), keepdims=True) + 1e-8))
        return trajectory_guidance, loss_values

    def get_dummy_guidance_values(init_val=0.0):
        """Get dummy guidance initial values"""
        dummy_guidance_values = {'stl': init_val}
        if achievable_guidance:
            dummy_guidance_values.update({'achievable': init_val})
        if ma_stl_form_eval is not None:
            dummy_guidance_values.update({'ma_stl': init_val})
        if smooth_loss_coeff > 0:
            dummy_guidance_values.update({'smooth': init_val})
        if segment_stl_check:
            dummy_guidance_values.update({'segment_stl': init_val})
        return dummy_guidance_values

    def denoise_step(runner_state, step_coeffs, denoiser_norm_stats, agent_params, agent_mask_arg=None,
                     agent_assign_arg=None):
        rng, noised_traj, step_idx = runner_state
        sigma, next_sigma, gamma = step_coeffs

        guidance_values = {}

        if do_apply_guidance:
            # --- Compute guidance coefficient ---
            # lambd decays linearly over the denoise steps plus a sine bump scaled by
            # policy_guidance_cosine_coeff, the whole thing scaled by policy_guidance_coeff.
            # Despite the names (inherited from the single-agent policy-guided sampler) this
            # is the step size of the STL gradient step: edm-ma has no policy/action guidance.
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
                guidance_values, noised_traj = apply_modified_stl_guidance(agent_mask_arg, guidance_traj,
                                                                           guidance_values, lambd, n_steps, noised_traj,
                                                                           step_idx, agent_assign_arg=agent_assign_arg)
            else:
                # No other guidance implemented yet
                pass

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

        # Pin index 0 of every agent's plan to its current state (in normalised space).
        agent_start_states = agent_params['x0']  # N x goal_dim
        normalized_start_state = normalise_traj(
            jnp.zeros_like(denoised_traj[:, 0, :obs_dim]).at[:, :goal_dim].add(agent_start_states),
            stats=denoiser_norm_stats['obs'])
        # Set initial state to the start state for each agent
        denoised_traj = denoised_traj.at[:, 0, :obs_dim].set(normalized_start_state)

        return (rng, denoised_traj, step_idx + 1), guidance_values

    def apply_modified_stl_guidance(agent_mask_arg, guidance_traj, guidance_values, lambd, n_steps, noised_traj,
                                    step_idx, agent_assign_arg=None):
        """Choose between alternate STL update and achievable guidance based on the step index

        NOTE: with achievable guidance the denoise scan embeds a GCBF+ rollout per plan
        segment; first-call compile grows with N and rollout_len (minutes at N=32).
        """
        alternate_stl_update = functools.partial(apply_stl_guidance, _time2alternate_update=True)
        achievable_stl_update = functools.partial(apply_stl_guidance, _time2achievable_guidance=True)

        def achievable_guidance_switch(_args, _noised_traj):
            should_alternate, should_achievable, guidance_traj, lambd, agent_mask_arg, agent_assign_arg = _args
            return jax.lax.cond(
                should_achievable,
                achievable_stl_update,
                apply_stl_guidance,
                guidance_traj, lambd, agent_mask_arg, agent_assign_arg,
                noised_traj
            )

        def alternating_guidance_switch(_args, _noised_traj):
            should_alternate, should_achievable, guidance_traj, lambd, agent_mask_arg, agent_assign_arg = _args
            return jax.lax.cond(
                should_alternate,
                alternate_stl_update,
                apply_stl_guidance,
                guidance_traj, lambd, agent_mask_arg, agent_assign_arg,
                noised_traj
            )

        if alternate_update:
            guidance_fn = alternating_guidance_switch
        elif achievable_guidance:
            guidance_fn = achievable_guidance_switch
        else:
            # skip should_alternate and should_achievable arguments
            guidance_fn = lambda args, z: apply_stl_guidance(*args[2:], z)
        if alternate_update and achievable_guidance:
            raise NotImplementedError("Alternate STL update with achievable guidance not implemented yet")
        # Guidance gates, all decided per denoise step inside the scan:
        #  - alternate_update toggles between the per-agent STL gradient and the team
        #    CaTL+ gradient every alternate_every steps;
        #  - the costly achievable rollout only runs after ach_guidance_after_fraction
        #    (default ACH_GUIDANCE_AFTER_FRACTION) of the steps;
        #  - STL guidance itself only runs after STL_GUIDANCE_AFTER_FRACTION of the steps.
        # Both lax.cond branches must return identical pytrees, hence the dummy loss
        # entries (jnp.nan inside _stl_guidance, 0.0 from get_dummy_guidance_values).
        should_alternate = alternate_update & ((step_idx // alternate_every) % 2 == 1)
        should_achievable = achievable_guidance & (step_idx > (ach_guidance_after_fraction * n_steps))
        noised_traj, guidance_values = jax.lax.cond(step_idx > (STL_GUIDANCE_AFTER_FRACTION * n_steps),
                                                    guidance_fn,
                                                    lambda _, z: (z, get_dummy_guidance_values()),
                                                    (should_alternate, should_achievable, guidance_traj, lambd,
                                                     agent_mask_arg, agent_assign_arg),
                                                    noised_traj)
        return guidance_values, noised_traj

    def apply_stl_guidance(guidance_traj, lambd, agent_mask_arg, agent_assign_arg, noised_traj,
                           _time2alternate_update=False, _time2achievable_guidance=False):
        guidance_traj, loss_values = _stl_guidance(stl_form_eval, guidance_traj, 30, 1e-1,
                                                   _ma_stl_form_eval=ma_stl_form_eval,
                                                   _time2alternate_update=_time2alternate_update,
                                                   _time2achievable_guidance=_time2achievable_guidance,
                                                   _agent_assign=agent_assign_arg)
        state = noised_traj[:, :, : goal_dim]
        guided_state = state + lambd * guidance_traj  # Take a step towards the guidance
        # Only agents still in agent_mask (i.e. still resampling) receive the guidance step.
        # Accepted agents are re-denoised untouched, and the keep-best merge in
        # _sample_denoised_trajectory then discards that fresh draw.
        partly_guided_state = jnp.where(agent_mask_arg[:, None, None], guided_state[..., :goal_dim],
                                        noised_traj[..., :goal_dim])
        noised_traj = noised_traj.at[..., :goal_dim].set(partly_guided_state)
        return noised_traj, loss_values

    denoised_traj, info, num_resampling_iters, rng = sample_stl_guided_trajectory(_multi_step_stl_updates, agent_params,
                                                                                  denoise_step, denoiser_norm_stats,
                                                                                  gammas, goal_dim, num_agents, obs_dim,
                                                                                  rng, seq_len, sigmas, stl_form_eval,
                                                                                  stl_guidance, traj_len, use_post_hack,
                                                                                  max_iter_count, agent_mask_noise,
                                                                                  skip_resample=skip_resample,
                                                                                  # Acceptance gate: exact (non-smoothed)
                                                                                  # eval when provided; the smoothed
                                                                                  # ma_stl_form_eval closure stays
                                                                                  # gradient-only in _stl_guidance.
                                                                                  ma_stl_form_eval=(
                                                                                      ma_stl_gate_eval
                                                                                      if ma_stl_gate_eval is not None
                                                                                      else ma_stl_form_eval),
                                                                                  task_ctx=task_ctx)

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
                                 goal_dim, num_agents, obs_dim, rng, seq_len, sigmas, stl_form_eval, stl_guidance,
                                 traj_len, use_post_hack, max_iter_count=MAX_SAMPLING_ITER, agent_mask_noise=False,
                                 skip_resample=False, ma_stl_form_eval=None, task_ctx=None):
    """Consolidate the resampling of trajectories with STL guidance for the entire system of agents

    With ``ma_stl_form_eval`` (a CaTL+ team formula, whose satisfaction cannot be split
    by agent) and GCBF_MA_ACCEPT=1 (default), acceptance is JOINT: the loop terminates
    only when every agent clears its own threshold AND the team robustness of the kept
    joint plan clears GCBF_MA_ACCEPT_THRESH (default = STL_RESAMPLE_THRESHOLD). While
    some agent fails individually, only those agents resample (accept-and-freeze); once
    all pass individually but the team score fails, ALL agents redraw jointly and the
    new joint plan replaces the kept one only if it improves the team robustness while
    keeping every agent above threshold. Monotonicity, precisely: the team robustness
    is monotone non-decreasing in the joint phase, and no accepted agent ever drops
    below its acceptance threshold — but a whole-plan swap MAY reduce an accepted
    agent's margin (toward, never past, the threshold) to buy team-robustness gain.

    With ``task_ctx`` (per-task inner evals of a CaTL+ spec) and GCBF_TASK_REPAIR=1
    (default), the loop additionally runs counting-aware repair: after each keep-best
    merge, exactly m_q agents are claimed per active task from the kept plan's
    per-task robustness matrix, an agent's acceptance threshold applies to its
    ASSIGNED task (not "any task"), and only deficit agents redraw. Allocation is
    read off the sampler's own draws — no upfront task-to-agent allocation.
    NOTE: under task repair, per-agent kept robustness is monotone only PER FIXED
    ASSIGNMENT — a counting-driven reassignment changes the agent's measuring stick
    (OR-value -> assigned-task rho), so its stored loss can legitimately drop and
    re-open the agent for resampling. Keep-best comparisons remain like-for-like
    because the stored loss is always recomputed under the assignment that scores
    the next iteration's draws."""
    if use_post_hack:
        # The path has two latent bugs (denoised_traj[:, :goal_dim] slices the TIME
        # axis, and _multi_step_stl_updates calls stl_form_eval without agent_id) and
        # its stl_loss is computed from pre-hack obs, so the hack never influenced
        # acceptance anyway. Fail loudly instead of silently misbehaving.
        raise NotImplementedError("use_post_hack is broken (wrong-axis slice, missing agent_id, "
                                  "stale-obs scoring) — do not enable")
    ma_accept_on = (ma_stl_form_eval is not None and not use_post_hack
                    and os.environ.get('GCBF_MA_ACCEPT', '1') == '1'
                    and os.environ.get('GCBF_KEEP_BEST', '1') == '1')
    ma_accept_thresh = float(os.environ.get('GCBF_MA_ACCEPT_THRESH', str(STL_RESAMPLE_THRESHOLD)))
    if not ma_accept_on:
        ma_stl_form_eval = None  # keep guidance_info pytree consistent with the legacy paths
    repair_on = (task_ctx is not None and ma_accept_on
                 and os.environ.get('GCBF_TASK_REPAIR', '1') == '1')
    if not repair_on:
        task_ctx = None  # trace-time off-switch: legacy pytree and code paths exactly

    # while_loop carry, in order:
    #   denoised_traj  (N, T, traj_len)  the plan kept so far (best-so-far under keep-best)
    #   guidance_info  dict              per-agent bookkeeping, see init_guidance_info below
    #   rng_arg                          PRNG state
    #   agent_mask_arg (N,) bool         agents scheduled to redraw this iteration
    #   _init_noise    (N, T, traj_len)  last iteration's init noise (reused under
    #                                    agent_mask_noise so accepted agents restart from
    #                                    the SAME noise and stay close to the kept plan)
    #   iter_count     int               resample iterations so far, capped at max_iter_count
    #
    # guidance_info['stl_loss'] is per-agent STL ROBUSTNESS despite the name: higher is
    # better, an agent is accepted once it reaches STL_RESAMPLE_THRESHOLD, and the -1e6
    # initial value makes the first draw always win the keep-best comparison.
    # 'cum_agent_mask' starts at 1 and accumulates the redraw flags, i.e. how many times
    # each agent has been scheduled to resample.
    def body_fn(val: tuple[jnp.ndarray, dict, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]):
        """Samples a new denoised trajectory. For EDM_MA, give the option of only sampling a subset of the agents"""
        denoised_traj, guidance_info, rng_arg, agent_mask_arg, _init_noise, iter_count = val

        # --- Sample random noise trajectory ---
        rng_arg, _rng_arg = jax.random.split(rng_arg)
        init_noise = jax.random.normal(_rng_arg, (num_agents, seq_len, traj_len))
        init_noise *= sigmas[0]

        if agent_mask_noise:
            # Accepted agents restart from the SAME init noise as the last iteration, so
            # their re-denoised trajectories stay close to the plan being kept.
            init_noise = jnp.where(agent_mask_arg[:, None, None], init_noise, _init_noise)

        return _sample_denoised_trajectory(
            _multi_step_stl_updates, agent_params, denoise_step,
            denoiser_norm_stats, gammas, goal_dim, init_noise, rng_arg,
            sigmas, stl_form_eval, obs_dim, stl_guidance=stl_guidance,
            use_post_hack=use_post_hack, agent_mask=agent_mask_arg, guidance_info=guidance_info,
            denoised_traj=denoised_traj, ma_stl_form_eval=ma_stl_form_eval,
            ma_accept_thresh=ma_accept_thresh, task_ctx=task_ctx) + (iter_count + 1,)

    def condition_fn(val: tuple[jnp.ndarray, dict, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]):
        """Condition to continue sampling, while robustness is below STL_RESAMPLE_THRESHOLD"""
        denoised_traj, guidance_info, rng_arg, agent_mask_arg, _init_noise, iter_count = val
        stl_losses = guidance_info['stl_loss']
        # only check for agent in the mask
        stl_losses = jnp.where(agent_mask_arg, stl_losses, jnp.ones_like(stl_losses))
        cont = jnp.any(stl_losses < STL_RESAMPLE_THRESHOLD)
        if ma_stl_form_eval is not None:
            # Joint CaTL+ acceptance: also keep sampling while the team robustness of
            # the kept joint plan is below threshold (satisfaction not agent-splittable).
            cont = cont | (guidance_info['ma_stl_rho'] < ma_accept_thresh)
        return cont & (iter_count < max_iter_count)

    init_guidance_info = {'stl_loss': jnp.ones(num_agents) * -1e6, 'stl_loss_pre_hack': jnp.ones(num_agents) * 1e6,
                          'cum_agent_mask': jnp.ones(num_agents, dtype=jnp.int32)}
    if ma_stl_form_eval is not None:
        init_guidance_info.update({'ma_stl_rho': jnp.asarray(-1e6)})
    if task_ctx is not None:
        # All agents start free (assign == Q, disjunctive form); branch unchosen.
        init_guidance_info.update({
            'task_assign': jnp.full(num_agents, len(task_ctx.inner_eval_fns), dtype=jnp.int32),
            'branch_choice': jnp.asarray(-1, dtype=jnp.int32)})

    # Placeholders kept so the reported info dict has a stable set of keys.
    init_guidance_info.update({
        'resample_batch_stl_scores': jnp.array([-1e6]),  # Single element array
        'best_resample_batch_idx': 0,
    })

    rng, _rng = jax.random.split(rng)
    init_denoised_traj = jax.random.normal(_rng, (num_agents, seq_len, traj_len))
    # Boolean array to determine which agents to sample (initially sample all)
    agent_mask = jnp.ones(num_agents, dtype=bool)
    if stl_guidance and not skip_resample:
        # Keep running while any agent's robustness is below STL_RESAMPLE_THRESHOLD
        # (NOTE: not reverse mode differentiable so remove if training)
        denoised_traj, info, _, agent_mask, init_noise, num_resampling_iters = jax.lax.while_loop(condition_fn, body_fn,
                                                                                                  (init_denoised_traj,
                                                                                                   init_guidance_info,
                                                                                                   rng, agent_mask,
                                                                                                   init_denoised_traj,
                                                                                                   0))
    else:
        denoised_traj, info, _, agent_mask, init_noise, num_resampling_iters = body_fn(
            (init_denoised_traj, init_guidance_info, rng, agent_mask, init_denoised_traj, 0))
    return denoised_traj, info, num_resampling_iters, rng


class TaskRepairCtx(NamedTuple):
    """Static context for counting-aware repair of CaTL+ team specs.

    inner_eval_fns: Q callables, each (num_agents, T, goal_dim) -> (num_agents,)
        per-agent robustness of task q's inner STL (avoid conjunct included).
    ms: Q ints, agents required per task (task_m_override applied).
    branch_masks: (B, Q) nested bool tuples; DNF branch -> which tasks belong to
        it. Single all-True branch when the spec is a plain conjunction.
    """
    inner_eval_fns: tuple
    ms: tuple
    branch_masks: tuple


def _task_R(task_ctx, y):
    """(Q, N) per-task per-agent robustness of unnormalised goal trajectories y."""
    return jnp.stack([fn(y) for fn in task_ctx.inner_eval_fns])


def _assigned_loss(assign, R, or_vals):
    """Per-agent acceptance loss: assigned-task rho for assigned agents (assign < Q),
    the disjunctive OR-form value for free agents (assign == Q)."""
    num_tasks, num_agents = R.shape
    own = R[jnp.clip(assign, 0, num_tasks - 1), jnp.arange(num_agents)]
    return jnp.where(assign < num_tasks, own, or_vals)


def _repair_assignment(R, assign_prev, branch_choice_prev, thresh, task_ms,
                       branch_masks, sticky=1e3):
    """Counting-aware repair: claim exactly m_q agents per active task.

    The sticky bonus keeps an agent that is already assigned to a task AND
    satisfies it ahead of any newcomer, so frozen agents never churn and the
    greedy claim is a fixed point once every active task is met. Tasks are
    claimed in task_id order; unclaimed agents stay free (assign == Q) under
    the disjunctive form. Oversubscribed specs short-fill later tasks, whose
    agents then redraw until the iteration cap (honest failure).

    R: (Q, N) robustness of the kept plan. assign_prev: (N,) int32 in [0..Q].
    branch_choice_prev: () int32, -1 = unset (chosen once, then frozen).
    task_ms / branch_masks: static tuples. Returns (assign_new, branch_choice).
    """
    num_tasks, num_agents = R.shape
    neg = jnp.float32(-1e9)
    bm = jnp.asarray(branch_masks)

    # Branch choice: rho_b = min over branch tasks of the m_q-th best agent rho.
    task_rho = jnp.stack([jax.lax.top_k(R[q], task_ms[q])[0][-1]
                          for q in range(num_tasks)])
    branch_rho = jnp.stack([jnp.min(jnp.where(bm[b], task_rho, jnp.inf))
                            for b in range(len(branch_masks))])
    branch_choice = jnp.where(branch_choice_prev < 0,
                              jnp.argmax(branch_rho).astype(jnp.int32),
                              branch_choice_prev).astype(jnp.int32)
    active = bm[branch_choice]

    sat = R > thresh
    prev_is_q = assign_prev[None, :] == jnp.arange(num_tasks, dtype=jnp.int32)[:, None]
    score = R + sticky * (sat & prev_is_q)
    claimed = jnp.zeros(num_agents, dtype=bool)
    assign_new = jnp.full(num_agents, num_tasks, dtype=jnp.int32)
    for q in range(num_tasks):
        s_q = jnp.where(claimed | ~active[q], neg, score[q])
        _, idx = jax.lax.top_k(s_q, task_ms[q])
        sel = jnp.zeros(num_agents, dtype=bool).at[idx].set(True) & (s_q > neg)
        assign_new = jnp.where(sel, jnp.int32(q), assign_new)
        claimed = claimed | sel
    return assign_new, branch_choice


def _sample_denoised_trajectory(_multi_step_stl_updates, agent_params, denoise_step, denoiser_norm_stats, gammas,
                                goal_dim, init_noise, rng, sigmas, stl_form_eval, obs_dim, stl_guidance=False,
                                use_post_hack=False, agent_mask=None, guidance_info=None, denoised_traj=None,
                                ma_stl_form_eval=None, ma_accept_thresh=STL_RESAMPLE_THRESHOLD, task_ctx=None):
    # --- Denoise trajectory ---
    # With counting-aware repair, the traced per-agent task assignment also targets
    # the guidance GRADIENT (separately ablatable via GCBF_TASK_REPAIR_GRAD).
    repair_grad = (task_ctx is not None and guidance_info is not None
                   and 'task_assign' in guidance_info
                   and os.environ.get('GCBF_TASK_REPAIR_GRAD', '1') == '1')
    denoise_step_scan = functools.partial(denoise_step, denoiser_norm_stats=denoiser_norm_stats,
                                          agent_params=agent_params, agent_mask_arg=agent_mask,
                                          agent_assign_arg=guidance_info['task_assign'] if repair_grad else None)
    (rng, new_denoised_traj, _), loss_info = jax.lax.scan(
        denoise_step_scan,
        (rng, init_noise, 0),
        (sigmas[:-1], sigmas[1:], gammas[:-1]),

    )
    if guidance_info is None:
        guidance_info = {}

    if stl_guidance:
        def _calc_stl_loss(x, agent_id):
            """Helper function to calculate STL loss"""
            return jnp.mean(stl_form_eval(x[:, :goal_dim][None], agent_id))

    # Best-so-far merge (GCBF_KEEP_BEST=1, default): a retried agent keeps its previous
    # trajectory unless the new draw scores strictly better, so its kept robustness is
    # monotone non-decreasing across iterations UNDER A FIXED assignment/measuring
    # stick. Exceptions, both by design: (a) task repair can reassign an agent, which
    # rescored the stored loss (see sample_stl_guided_trajectory docstring); (b) the
    # team-phase whole-plan swap may reduce an accepted agent's margin, never below
    # threshold. The -1e6 init sentinel makes the first draw always win.
    # The post-hack path mutates the trajectory after scoring, so it keeps the
    # legacy replace-if-masked merge.
    keep_best = os.environ.get('GCBF_KEEP_BEST', '1') == '1'
    if stl_guidance and keep_best and not use_post_hack:
        new_obs = unnormalise_traj(new_denoised_traj[:, :, :obs_dim], denoiser_norm_stats["obs"])
        new_loss = jax.vmap(_calc_stl_loss)(new_obs, jnp.arange(new_obs.shape[0]))
        if ma_stl_form_eval is not None and task_ctx is not None:
            # Counting-aware repair (CaTL+ disjunctive mode). An agent's acceptance
            # loss is the robustness of its ASSIGNED task (OR-form only while free),
            # so "satisfies ANY task" can no longer freeze a lopsided allocation.
            # Invariant: the stored stl_loss was recomputed under this same assignment
            # at the end of the previous iteration, so keep-best compares like for like.
            assign = guidance_info['task_assign']
            new_goal = new_obs[:, :, :goal_dim]
            new_loss = _assigned_loss(assign, _task_R(task_ctx, new_goal), new_loss)
            prev_loss = guidance_info['stl_loss']
            take_new = agent_mask & (new_loss > prev_loss)
            merged_traj = jnp.where(take_new[:, None, None], new_denoised_traj, denoised_traj)
            # Same two-phase merge as the legacy joint gate: per-agent keep-best while
            # any agent fails; whole-plan monotone swap once individuals are settled.
            team_phase = jnp.all(prev_loss >= STL_RESAMPLE_THRESHOLD)
            new_ma = jnp.mean(ma_stl_form_eval(new_goal))
            prev_ma = guidance_info['ma_stl_rho']
            take_whole = team_phase & (new_ma > prev_ma) & jnp.all(new_loss >= STL_RESAMPLE_THRESHOLD)
            denoised_traj = jnp.where(take_whole, new_denoised_traj,
                                      jnp.where(team_phase, denoised_traj, merged_traj))
            kept_obs = unnormalise_traj(denoised_traj[:, :, :obs_dim], denoiser_norm_stats["obs"])
            kept_goal = kept_obs[:, :, :goal_dim]
            kept_ma = jnp.mean(ma_stl_form_eval(kept_goal))
            R_kept = _task_R(task_ctx, kept_goal)
            or_kept = jax.vmap(_calc_stl_loss)(kept_obs, jnp.arange(kept_obs.shape[0]))
            assign, branch_choice = _repair_assignment(
                R_kept, assign, guidance_info['branch_choice'],
                STL_RESAMPLE_THRESHOLD, task_ctx.ms, task_ctx.branch_masks)
            kept_loss = _assigned_loss(assign, R_kept, or_kept)
            guidance_info.update({'stl_loss_pre_hack': kept_loss, 'stl_loss': kept_loss,
                                  'ma_stl_rho': kept_ma, 'task_assign': assign,
                                  'branch_choice': branch_choice})
            ind_bad = kept_loss < STL_RESAMPLE_THRESHOLD
            team_bad = kept_ma < ma_accept_thresh
            # Deficit agents redraw (targeted repair); the joint redraw-all fires only
            # as endgame when every count is met but the soft team rho is still short.
            new_agent_mask = ind_bad | (team_bad & ~jnp.any(ind_bad))
            guidance_info.update({'cum_agent_mask': guidance_info['cum_agent_mask']
                                                    + new_agent_mask.astype(jnp.int32)})
            return denoised_traj, guidance_info, rng, new_agent_mask, init_noise
        prev_loss = guidance_info['stl_loss']
        take_new = agent_mask & (new_loss > prev_loss)
        merged_traj = jnp.where(take_new[:, None, None], new_denoised_traj, denoised_traj)
        merged_loss = jnp.where(take_new, new_loss, prev_loss)
        if ma_stl_form_eval is not None:
            # Joint CaTL+ acceptance. Two phases, decided by whether every agent already
            # cleared its OWN threshold on the previously kept plan:
            #  - individual phase: per-agent keep-best merge as usual (accept-and-freeze);
            #  - joint (team) phase: individuals are settled but the team robustness
            #    rho(Psi) of the kept plan is still below threshold. All agents were
            #    redrawn jointly; the new JOINT plan replaces the kept one only if it
            #    improves rho(Psi) while keeping every agent above threshold (whole-plan
            #    swap — no per-agent mixing, so the delivered plan is always an actually
            #    sampled joint configuration in this phase and rho(Psi) is monotone).
            team_phase = jnp.all(prev_loss >= STL_RESAMPLE_THRESHOLD)
            new_ma = jnp.mean(ma_stl_form_eval(new_obs[:, :, :goal_dim]))
            prev_ma = guidance_info['ma_stl_rho']
            take_whole = team_phase & (new_ma > prev_ma) & jnp.all(new_loss >= STL_RESAMPLE_THRESHOLD)
            denoised_traj = jnp.where(take_whole, new_denoised_traj,
                                      jnp.where(team_phase, denoised_traj, merged_traj))
            kept_loss = jnp.where(take_whole, new_loss, jnp.where(team_phase, prev_loss, merged_loss))
            kept_obs = unnormalise_traj(denoised_traj[:, :, :obs_dim], denoiser_norm_stats["obs"])
            kept_ma = jnp.mean(ma_stl_form_eval(kept_obs[:, :, :goal_dim]))
            guidance_info.update({'stl_loss_pre_hack': kept_loss, 'stl_loss': kept_loss,
                                  'ma_stl_rho': kept_ma})
            ind_bad = kept_loss < STL_RESAMPLE_THRESHOLD
            team_bad = kept_ma < ma_accept_thresh
            # Team-only failure -> everyone redraws jointly next iteration.
            new_agent_mask = ind_bad | (team_bad & ~jnp.any(ind_bad))
            guidance_info.update({'cum_agent_mask': guidance_info['cum_agent_mask']
                                                    + new_agent_mask.astype(jnp.int32)})
            return denoised_traj, guidance_info, rng, new_agent_mask, init_noise
        denoised_traj = merged_traj
        kept_loss = merged_loss
        guidance_info.update({'stl_loss_pre_hack': kept_loss, 'stl_loss': kept_loss})
        new_agent_mask = kept_loss < STL_RESAMPLE_THRESHOLD
        guidance_info.update({'cum_agent_mask': guidance_info['cum_agent_mask'] + new_agent_mask.astype(jnp.int32)})
        return denoised_traj, guidance_info, rng, new_agent_mask, init_noise

    # GCBF_KEEP_BEST=0 (and the post-hack path): plain replace-if-masked merge. The new
    # draw always overwrites a masked agent, so kept robustness is NOT monotone and the
    # plan delivered at the iteration cap is simply the last draw, not the best one seen.
    denoised_traj = jnp.where(agent_mask[:, None, None], new_denoised_traj, denoised_traj)
    if stl_guidance:
        obs = denoised_traj[:, :, :obs_dim]
        obs = unnormalise_traj(obs, denoiser_norm_stats["obs"])

        guidance_info.update({'stl_loss_pre_hack': jax.vmap(_calc_stl_loss)(obs, jnp.arange(obs.shape[0]))})
        if use_post_hack:
            hacked_traj = _multi_step_stl_updates(stl_form_eval, denoised_traj[:, :goal_dim], 30, 1e-1)
            # normalize and add hacked trajectory
            hacked_traj_zeros = jnp.zeros_like(obs)
            hacked_traj_zeros = hacked_traj_zeros.at[:hacked_traj.shape[0], :hacked_traj.shape[1]].set(hacked_traj)
            hacked_traj = normalise_traj(hacked_traj_zeros, denoiser_norm_stats["obs"])[:, :goal_dim]
            denoised_traj = denoised_traj.at[:, :goal_dim].set(hacked_traj)
        guidance_info.update({'stl_loss': jax.vmap(_calc_stl_loss)(obs, jnp.arange(obs.shape[0]))})
        new_agent_mask = guidance_info['stl_loss'] < STL_RESAMPLE_THRESHOLD
        guidance_info.update({'cum_agent_mask': guidance_info['cum_agent_mask'] + new_agent_mask.astype(jnp.int32)})
        agent_mask = new_agent_mask
    return denoised_traj, guidance_info, rng, agent_mask, init_noise


def sample_trajectory_batched(
        rng,
        denoiser_state,
        seq_len,
        obs_dim,
        action_dim,
        denoiser_norm_stats,
        denoiser_hyperparams,
        num_candidates: int = 4,  # Number of candidate plans to generate
        selection_criterion: str = "stl_score",  # How to select best plan: "stl_score", "ma_stl_score", "achievable_score"
        stl_weight= 1.0,  # Weight for STL score in combined selection
        ma_stl_weight= 1.0,  # Weight for MA STL score in
        achievable_weight= 0.5,  # Weight for achievable score in combined selection
        **kwargs  # All other arguments from sample_trajectory
):
    """
    Sample multiple trajectory candidates using EDM-Multi Agent diffusion model and select the best one

    :param num_candidates: Number of candidate trajectories to generate in parallel
    :param selection_criterion: Criterion for selecting the best trajectory
    :param kwargs: All other parameters from sample_trajectory function
    :return: Best trajectory and info including all candidate scores
    """

    # Generate multiple candidates in parallel
    candidate_keys = jax.random.split(rng, num_candidates)

    # Vectorize the single trajectory sampling function
    batched_sample_fn = jax.vmap(
        lambda key: sample_trajectory(
            key, denoiser_state, seq_len, obs_dim, action_dim,
            denoiser_norm_stats, denoiser_hyperparams, **kwargs
        ),
        in_axes=(0,)
    )

    # Generate all candidates
    candidate_rollouts, candidate_infos = batched_sample_fn(candidate_keys)

    # Select the best candidate based on the specified criterion
    select_best_kwargs = {
        'stl_weight': stl_weight,
        'ma_stl_weight': ma_stl_weight,
        'achievable_weight': achievable_weight
    }
    best_idx, selection_info = select_best_candidate(
        candidate_rollouts, candidate_infos, selection_criterion, **kwargs, **select_best_kwargs
    )

    # Extract the best rollout and info
    best_rollout = jtu.tree_map(lambda x: x[best_idx], candidate_rollouts)
    best_info = jtu.tree_map(lambda x: x[best_idx], candidate_infos)

    # Add selection information to the best info
    best_info.update({
        'num_candidates': num_candidates,
        'best_candidate_idx': best_idx,
        # selection_criterion is a Python str, so it stays out of the traced info dict.
        'all_candidate_scores': selection_info['all_scores'],
        'candidate_selection_info': selection_info
    })

    return best_rollout, best_info


def select_best_candidate(candidate_rollouts, candidate_infos, selection_criterion, **kwargs):
    """
    Select the best candidate trajectory based on the specified criterion

    Outer best-of-N: each candidate is a COMPLETE accept/resample loop (a full run of
    sample_stl_guided_trajectory), not a single denoise pass. Candidates are ranked by
    the mean per-agent robustness of the plan each loop kept; the team robustness
    rho(Psi) is folded in only when GCBF_TEAM_SELECT_OUTER=1.

    :param candidate_rollouts: Array of candidate rollouts [num_candidates, ...]
    :param candidate_infos: Dict with arrays where first dim is num_candidates
    :param selection_criterion: Criterion for selection
    :return: (best_index, selection_info)
    """
    stl_scores = candidate_infos["stl_loss"].mean(axis=1)  # Average STL loss across agents
    if "ma_stl_rho" in candidate_infos and os.environ.get('GCBF_TEAM_SELECT_OUTER', '0') == '1':
        # Opt-in: also rank outer candidates by the team robustness of the kept joint
        # plan. A/B on crowded CaTL+ specs (choiceseq3 N=32, hungarian guard) showed
        # this dilutes the per-agent avoid margins and costs safety, so default off;
        # the within-loop resample-batch selection stays team-aware regardless.
        stl_scores = stl_scores + candidate_infos["ma_stl_rho"]

    if selection_criterion == "stl_score":
        scores = stl_scores
    elif selection_criterion == "combined_score":
        stl_w = kwargs.get('stl_weight', 1.0)
        ach_w = kwargs.get('achievable_weight', 0.5)
        ma_w = kwargs.get('ma_stl_weight', 1.0)
        scores = stl_w * stl_scores
        if "achievable_loss" in candidate_infos:
            scores = scores - ach_w * candidate_infos["achievable_loss"].mean(axis=1)
        if "ma_stl_loss" in candidate_infos:
            scores = scores + ma_w * candidate_infos["ma_stl_loss"].mean(axis=1)
    else:
        scores = stl_scores

    best_idx = jnp.argmax(scores)

    selection_info = {
        'all_scores': scores,
        'best_score': scores[best_idx],
        'score_std': jnp.std(scores),
        'score_range': jnp.max(scores) - jnp.min(scores)
    }

    return best_idx, selection_info
