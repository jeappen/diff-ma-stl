import optax
from flax.training.train_state import TrainState
from typing import Optional

import gcbfplus.diffusion.diffusion.edm as edm
import gcbfplus.diffusion.diffusion.edm_ma as edm_ma
from ..models.diffusion import UNet
from ..util import *


def create_denoiser_train_state(rng, obs_dim, action_dim, args, dataset_len):
    # --- Create U-Net model ---
    denoiser = UNet(args.num_features, args.num_blocks)
    placeholder_batch, placeholder_seq = 2, 64
    traj_dim = obs_dim + action_dim
    if args.diffusion_trajectory_mode == "sard":
        traj_dim += 2
    denoiser_params = denoiser.init(
        rng,
        jnp.ones((placeholder_batch, placeholder_seq, traj_dim)),
        jnp.ones((1,)),
    )

    # --- Create cosine decay schedule ---
    num_steps_per_epoch = dataset_len // args.batch_size
    total_steps = num_steps_per_epoch * args.num_epochs
    warmup_steps = total_steps // 10
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.lr * 0.1,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
    )

    # --- Return train state ---
    return TrainState.create(
        apply_fn=denoiser.apply,
        params=denoiser_params,
        tx=optax.adam(learning_rate=lr_schedule),
    )


def get_denoiser_hypers(args):
    if args.diffusion_method == "edm":
        return edm.DenoiserHyperparams(
            p_mean=args.edm_p_mean,
            p_std=args.edm_p_std,
            sigma_data=args.edm_sigma_data,
            sigma_min=args.edm_sigma_min,
            sigma_max=args.edm_sigma_max,
            rho=args.edm_rho,
            edm_first_order=args.edm_first_order,
            diffusion_timesteps=args.diffusion_timesteps,
            s_tmin=args.edm_s_tmin,
            s_tmax=args.edm_s_tmax,
            s_churn=args.edm_s_churn,
            s_noise=args.edm_s_noise,
        )
    elif args.diffusion_method == "edm-ma":
        return edm_ma.DenoiserHyperparams(
            p_mean=args.edm_p_mean,
            p_std=args.edm_p_std,
            sigma_data=args.edm_sigma_data,
            sigma_min=args.edm_sigma_min,
            sigma_max=args.edm_sigma_max,
            rho=args.edm_rho,
            edm_first_order=args.edm_first_order,
            diffusion_timesteps=args.diffusion_timesteps,
            s_tmin=args.edm_s_tmin,
            s_tmax=args.edm_s_tmax,
            s_churn=args.edm_s_churn,
            s_noise=args.edm_s_noise,
            achievable_loss_coeff=args.achievable_loss_coeff,
        )
    raise ValueError(f"Unknown diffusion method {args.diffusion_method}.")


def make_train_step(args, env=None, rollout_fn=None, sample_rollout_fn=None):
    # rollout_fn is accepted (train_diffusion.py passes it) but the edm-ma train step
    # only uses sample_rollout_fn for the achievable loss.
    hypers = get_denoiser_hypers(args)
    if args.diffusion_method == "edm":
        return partial(edm.train_step, denoiser_hyperparams=hypers)
    elif args.diffusion_method == "edm-ma":
        return partial(edm_ma.train_step, denoiser_hyperparams=hypers,
                       auxiliary_training_loss=args.auxiliary_training_loss, env=env,
                       sample_rollout_fn=sample_rollout_fn, total_training_steps=args.num_epochs,
                       auxiliary_loss_last_fraction=args.auxiliary_loss_last_fraction,
                       auxiliary_loss_rollout_length=args.auxiliary_loss_rollout_length,
                       exponential_loss_k=args.exponential_loss_k)

    raise ValueError(f"Unknown diffusion method {args.diffusion_method}.")


def make_sample_fn(
        args,
        normalize_action_guidance,
        denoised_guidance,
        det_guidance,
        stl_guidance=False,
        achievable_guidance=False,
        achievable_loss_coeff=0.0,
        ma_stl_loss_coeff=0.0,
        agent_mask_noise=False,
        auxiliary_training_loss: Optional[str] = None,
        auxiliary_loss_rollout_length: Optional[int] = None,
        alternate_update: bool = False,
        alternate_every: int = 8,
        resample_batch_size: int = 1,  # Add resample_batch_size parameter
        skip_resample: bool = False,  # Skip resampling if True
        ach_guidance_after_fraction: Optional[float] = None,
        smooth_loss_coeff: float = 0.0,
        dense_achievable_supervision: bool = True,
        guidance_hardness: float = 100.0,
        segment_stl_check: bool = False,
        segment_stl_coeff: float = 0.5,
        segment_interp_points: int = 5,
        global_stl_only: bool = False,
):
    hypers = get_denoiser_hypers(args)
    if args.diffusion_method == "edm":
        return partial(
            edm.sample_trajectory,
            denoiser_hyperparams=hypers,
            normalize_action_guidance=normalize_action_guidance,
            denoised_guidance=denoised_guidance,
            det_guidance=det_guidance,
            stl_guidance=stl_guidance,
            diffusion_mode=args.diffusion_trajectory_mode,
            skip_resample=skip_resample
        )
    elif args.diffusion_method == "edm-ma":
        sample_kwargs = dict(
            denoiser_hyperparams=hypers,
            normalize_action_guidance=normalize_action_guidance,
            denoised_guidance=denoised_guidance,
            det_guidance=det_guidance,
            stl_guidance=stl_guidance,
            diffusion_mode=args.diffusion_trajectory_mode,
            achievable_guidance=achievable_guidance,
            achievable_loss_coeff=achievable_loss_coeff,
            agent_mask_noise=agent_mask_noise,
            ma_stl_loss_coeff=ma_stl_loss_coeff,
            auxiliary_training_loss=auxiliary_training_loss,
            auxiliary_loss_rollout_length=auxiliary_loss_rollout_length,
            alternate_update=alternate_update,
            alternate_every=alternate_every,
            resample_batch_size=resample_batch_size,
            skip_resample=skip_resample,
            smooth_loss_coeff=smooth_loss_coeff,
            dense_achievable_supervision=dense_achievable_supervision,
            guidance_hardness=guidance_hardness,
            segment_stl_check=segment_stl_check,
            segment_stl_coeff=segment_stl_coeff,
            segment_interp_points=segment_interp_points,
            global_stl_only=global_stl_only,
        )
        if ach_guidance_after_fraction is not None:
            sample_kwargs['ach_guidance_after_fraction'] = ach_guidance_after_fraction
        return partial(edm_ma.sample_trajectory, **sample_kwargs)
    raise ValueError(f"Unknown diffusion method {args.diffusion_method}.")


def make_batched_sample_fn(
        args,
        normalize_action_guidance,
        denoised_guidance,
        det_guidance,
        stl_guidance=False,
        achievable_guidance=False,
        achievable_loss_coeff=0.0,
        ma_stl_loss_coeff=0.0,
        agent_mask_noise=False,
        auxiliary_training_loss: Optional[str] = None,
        auxiliary_loss_rollout_length: Optional[int] = None,
        alternate_update: bool = False,
        alternate_every: int = 8,
        num_candidates: int = 4,
        selection_criterion: str = "stl_score",
        resample_batch_size: int = 1,  # Add resample_batch_size parameter
        skip_resample: bool = False,  # Skip resampling if True
        ach_guidance_after_fraction: Optional[float] = None,
        smooth_loss_coeff: float = 0.0,
        dense_achievable_supervision: bool = True,
        guidance_hardness: float = 100.0,
        segment_stl_check: bool = False,
        segment_stl_coeff: float = 0.5,
        segment_interp_points: int = 5,
        global_stl_only: bool = False,
):
    """Create a batched sampling function that generates multiple candidates and selects the best one."""
    hypers = get_denoiser_hypers(args)
    if args.diffusion_method == "edm":
        raise NotImplementedError("Batched sampling not implemented for EDM yet")
    elif args.diffusion_method == "edm-ma":
        batched_kwargs = dict(
            denoiser_hyperparams=hypers,
            normalize_action_guidance=normalize_action_guidance,
            denoised_guidance=denoised_guidance,
            det_guidance=det_guidance,
            stl_guidance=stl_guidance,
            diffusion_mode=args.diffusion_trajectory_mode,
            achievable_guidance=achievable_guidance,
            achievable_loss_coeff=achievable_loss_coeff,
            agent_mask_noise=agent_mask_noise,
            ma_stl_loss_coeff=ma_stl_loss_coeff,
            auxiliary_training_loss=auxiliary_training_loss,
            auxiliary_loss_rollout_length=auxiliary_loss_rollout_length,
            alternate_update=alternate_update,
            alternate_every=alternate_every,
            num_candidates=num_candidates,
            selection_criterion=selection_criterion,
            resample_batch_size=resample_batch_size,
            skip_resample=skip_resample,
            smooth_loss_coeff=smooth_loss_coeff,
            dense_achievable_supervision=dense_achievable_supervision,
            guidance_hardness=guidance_hardness,
            segment_stl_check=segment_stl_check,
            segment_stl_coeff=segment_stl_coeff,
            segment_interp_points=segment_interp_points,
            global_stl_only=global_stl_only,
        )
        if ach_guidance_after_fraction is not None:
            batched_kwargs['ach_guidance_after_fraction'] = ach_guidance_after_fraction
        return partial(edm_ma.sample_trajectory_batched, **batched_kwargs)
    raise ValueError(f"Unknown diffusion method {args.diffusion_method}.")
