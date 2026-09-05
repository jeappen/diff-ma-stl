import argparse
import sys

from gcbfplus.stl.utils import TRAINING_CONFIG, PLANNER_CONFIG
from .data import DIFFUSION_TRAJECTORY_MODES

DIFFUSION_TRAINING_CONFIG = TRAINING_CONFIG["diffusion"]
DIFFUSION_TESTING_CONFIG = PLANNER_CONFIG["diffusion"]
DIFFUSION_TRAINING_KEY = "diff_training"
DIFFUSION_TESTING_KEY = "diff_testing"


def parse_diffusion_args(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_nans", action="store_true")

    # Experiment
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Offline dataset name",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num_epochs", type=int, default=10000, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--eval_rate", type=int, default=50, help="Number of steps per evaluation"
    )

    # Dataset
    parser.add_argument(
        "--val_ratio", type=float, default=0.05, help="Validation ratio"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--trajectory_length",
        type=int,
        default=32,
        help="Trajectory length that the diffusion model will generate",
    )
    parser.add_argument(
        "--dataset_stride",
        type=int,
        default=8,
        help="Index stride for dataset sub-trajectory generation",
    )

    # Diffusion
    parser.add_argument(
        "--diffusion_method",
        type=str,
        default="edm",
        choices=["edm", "edm-ma"],
        help="Diffusion method",
    )
    parser.add_argument(
        "--diffusion_timesteps",
        type=int,
        default=256,
        help="Number of timesteps for diffusion sampling",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.995,
        help="Exponential moving average decay for model parameters",
    )
    parser.add_argument(
        "--ema_update_every",
        type=int,
        default=10,
        help="Number of steps between EMA updates",
    )
    parser.add_argument(
        "--diffusion_trajectory_mode",
        type=str,
        default="sard",
        choices=DIFFUSION_TRAJECTORY_MODES,
        help="Diffusion trajectory format, sard (default) or sa or sg",
    )

    # EDM
    parser.add_argument(
        "--edm_p_mean",
        type=float,
        default=-1.2,
        help="Mean of log-normal noise distribution",
    )
    parser.add_argument(
        "--edm_p_std",
        type=float,
        default=1.2,
        help="Standard deviation of log-normal noise distribution",
    )
    parser.add_argument(
        "--edm_sigma_data",
        type=float,
        default=1.0,
        help="Standard deviation of data distribution",
    )
    parser.add_argument(
        "--edm_sigma_min",
        type=float,
        default=0.002,
        help="Minimum noise level",
    )
    parser.add_argument(
        "--edm_sigma_max",
        type=float,
        default=80,
        help="Maximum noise level",
    )
    parser.add_argument(
        "--edm_rho",
        type=float,
        default=7.0,
        help="Sampling schedule",
    )
    parser.add_argument(
        "--edm_s_tmin",
        type=float,
        default=0.05,
        help="Stochastic sampling coefficients",
    )
    parser.add_argument(
        "--edm_s_tmax",
        type=float,
        default=50.0,
        help="Stochastic sampling coefficients",
    )
    parser.add_argument(
        "--edm_s_churn",
        type=float,
        default=80,
        help="Stochastic sampling coefficients",
    )
    parser.add_argument(
        "--edm_s_noise",
        type=float,
        default=1.003,
        help="Stochastic sampling coefficients",
    )
    parser.add_argument(
        "--edm_first_order",
        action="store_true",
        help="Use first-order Euler integration (disables second-order Heun)",
    )

    # U-Net
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=3,
        help="Number of blocks in the diffusion U-Net model",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=1024,
        help="Number of features in the diffusion U-Net model",
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=0,
        help="Number of epochs between checkpoints",
    )

    parser.add_argument(
        "--checkpoints_to_save",
        type=int,
        default=5,
        help="Number of epochs between checkpoints",
    )

    # Optimization
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")

    # Logging
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--tiny_dataset", action="store_true", help="Use tiny dataset for debugging")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project")
    parser.add_argument("--wandb_team", type=str, default=None, help="WandB team")
    parser.add_argument("--wandb_group", type=str, default="debug", help="WandB group")

    parser_add_ma_args(parser)

    args, rest_args = parser.parse_known_args(cmd_args)
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    # Checks for appropriate diffusion trajectory mode
    if args.sample_goal_change:
        assert args.diffusion_trajectory_mode == "sg", "Goal change sampling requires sg trajectory mode"

    return args


def parser_add_ma_args(parser):
    """Add multi-agent and STL relevant arguments to the parser."""

    # STL options
    parser.add_argument(
        "--stl_train_traj_len",
        type=int,
        default=8,  # Should be a divisor of trajectory_length
        help="Sample these many time-steps from the dataset trajectory",
    )
    parser.add_argument("--sample_goal_change", action="store_true", help="Sample data at goal change")
    parser.add_argument("--skip_guidance", action="store_true", help="Disable STL guidance during sampling")
    parser.add_argument("--stl_guidance", action="store_true", help="Diffusion STL guidance")
    parser.add_argument("--agent_mask_noise", action="store_true", help="Mask Noise in STL guidance")
    parser.add_argument("--achievable_guidance", action="store_true", help="Achievable guidance")
    parser.add_argument("--achievable_loss_coeff", type=float,  # Named achievable_loss_coeff for consistency
                        default=DIFFUSION_TRAINING_CONFIG["auxiliary_loss_default_weight"],
                        help="Achievable loss coefficient")
    parser.add_argument("--ach_guidance_after_fraction", type=float,
                        default=ACH_GUIDANCE_AFTER_FRACTION,
                        help="Fraction of diffusion steps after which achievable guidance is applied (0-1)")
    parser.add_argument("--auxiliary_loss_last_fraction", type=float,
                        default=DIFFUSION_TRAINING_CONFIG["auxiliary_loss_last_fraction"],
                        help="Achievable loss last fraction")
    parser.add_argument("--exponential_loss_k", type=float,
                        default=DIFFUSION_TRAINING_CONFIG["exponential_loss_k"],
                        help="K value for exponential loss")
    parser.add_argument("--ma_stl_loss_coeff", type=float, default=1.0, help="MA STL loss coefficient")
    parser.add_argument("--ma_stl_alternate_guidance", action="store_true", help="Alternate MA STL guidance")
    parser.add_argument("--ma_stl_alternate_every", type=int,
                        default=DIFFUSION_TESTING_CONFIG["ma_stl_alternate_every"],
                        help="Alternate MA STL guidance every")
    parser.add_argument("--smooth_loss_coeff", type=float, default=0.0,
                        help="Smoothness regularizer coefficient (curvature penalty on plan trajectory)")
    parser.add_argument("--dense_achievable_supervision", action="store_true", default=True,
                        help="Use dense (all rollout steps) vs sparse (final step only) supervision in achievable loss")
    parser.add_argument("--guidance_hardness", type=float, default=100.0,
                        help="Softmin hardness for STL guidance during diffusion (lower=smoother gradients, default 100)")
    parser.add_argument("--segment_stl_check", action="store_true", default=False,
                        help="Evaluate STL at interpolated points between waypoints for denser avoidance gradients")
    parser.add_argument("--segment_stl_coeff", type=float, default=0.5,
                        help="Coefficient for segment-based STL loss")
    parser.add_argument("--segment_interp_points", type=int, default=5,
                        help="Number of interpolation points per segment for segment STL check")
    parser.add_argument("--no_dense_achievable_supervision", dest="dense_achievable_supervision",
                        action="store_false",
                        help="Disable dense achievable supervision (use sparse final-step-only)")
    parser.add_argument("--global_stl_only", action="store_true", default=False,
                        help="Use only CaTL+ global guidance (no per-agent STL loss). "
                             "CaTL+ gradient handles both allocation and trajectory optimization.")
    # Training options
    parser.add_argument(
        "--auxiliary_loss_rollout_length",
        type=int,
        default=None,
        help="Rollout length for auxiliary loss, if None, uses goal_sample_interval",
    )

    parser.add_argument(
        "--auxiliary_training_loss",
        type=str,
        default=None,  # Achievable loss to use
        choices=["env-sync-train", "env-sync-test", None],
        help="Training loss modifications for diffusion model",
    )
    parser.add_argument(
        "--auxiliary_group_agents_as",
        type=int,
        default=None,
        help="How many agents to group in calculation of auxiliary loss",
    )

    # Batched sampling options
    parser.add_argument("--use_batched_sampling", action="store_true", default=PLANNER_CONFIG["diffusion"]["use_batched_sampling"],
                        help="Use batched sampling to generate multiple candidates and select the best")
    parser.add_argument("--num_candidates", type=int, default=PLANNER_CONFIG["diffusion"]["num_candidates"],
                        help="Number of candidate plans to generate when using batched sampling")
    parser.add_argument("--selection_criterion", type=str, default="stl_score",
                        choices=["stl_score", "ma_stl_score", "achievable_score", "combined_score"],
                        help="Criterion for selecting the best candidate plan")
    parser.add_argument("--stl_weight", type=float, default=1.0,
                        help="Weight for STL score in combined selection criterion")
    parser.add_argument("--ma_stl_weight", type=float, default=1.0,
                        help="Weight for MA STL score in combined selection criterion")
    parser.add_argument("--achievable_weight", type=float, default=0.5,
                        help="Weight for achievable score in combined selection criterion")
    parser.add_argument("--resample_batch_size", type=int, default=PLANNER_CONFIG["diffusion"]["resample_batch_size"],
                        help="No-op, accepted for compatibility with logged commands")
    parser.add_argument("--skip_resample", action="store_true", default=PLANNER_CONFIG["diffusion"]["skip_resample"],
                        help="Skip the STL reject/resample loop and accept the first denoised draw "
                             "(faster, no acceptance guarantee)")


def parse_agent_args(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_nans", action="store_true")

    # Experiment
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Offline dataset name",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=1_000_000,
        help="Number of epochs or agent train steps",
    )
    parser.add_argument(
        "--eval_rate",
        type=int,
        default=1,
        help="Number of train steps between evaluations",
    )
    parser.add_argument(
        "--num_env_workers",
        type=int,
        default=16,
        help="Number of environment workers for evaluation",
    )
    parser.add_argument("--batch_size", type=int, default=256)

    # Synthetic experience
    parser.add_argument("--synthetic_experience", action="store_true")
    parser.add_argument(
        "--num_synth_workers",
        type=int,
        default=32,
        help="Number of parallel workers for synthetic rollout",
    )
    parser.add_argument(
        "--num_synth_rollouts",
        type=int,
        default=256,
        help="Number of synthetic rollouts per worker",
    )
    parser.add_argument(
        "--synth_dataset_lifetime",
        type=int,
        default=10000,
        help="Number of steps before synthetic dataset is resampled",
    )
    parser.add_argument(
        "--synth_batch_size",
        type=int,
        default=240,
        help="Number of synthetic samples to use per-batch",
    )
    parser.add_argument(
        # This should be loaded by the saved denoiser config usually, but we allow edm-ma sampling here
        "--diffusion_method",
        type=str,
        default="edm",
        choices=["edm", "edm-ma"],
        help="Diffusion method",
    )
    parser.add_argument(
        "--synth_batch_lifetime",
        type=int,
        default=1,
        help="Number of epochs before a synthetic batch is resampled",
    )
    parser.add_argument("--diffusion_timesteps", type=int, default=None)
    parser.add_argument("--denoiser_checkpoint", type=str, default=None)

    # Policy guidance
    parser.add_argument("--policy_guidance_coeff", type=float, default=0.0)
    parser.add_argument("--policy_guidance_cosine_coeff", type=float, default=0.3)
    parser.add_argument(
        "--normalize_action_guidance",
        action="store_true",
        help="Normalize action guidance",
    )
    parser.add_argument(
        "--denoised_guidance",
        action="store_true",
        help="Apply guidance to denoised trajectory",
    )

    # Agent
    parser.add_argument(
        "--agent", type=str, default="iql", choices=["iql", "td3_bc"], help="Agent type"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function for actor critic",
    )
    parser.add_argument(
        "--num_rollout_steps",
        type=int,
        default=128,
        help="Number of rollout steps per agent update",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument(
        "--value_loss_coef", type=float, default=0.5, help="Value loss coefficient"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--polyak_step_size",
        type=float,
        default=0.005,
        help="Target update step size",
    )

    # Optimization
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lr_schedule", type=str, default="constant")

    # TD3+BC
    parser.add_argument(
        "--policy_noise", type=float, default=0.2, help="Policy noise parameter"
    )
    parser.add_argument(
        "--noise_clip", type=float, default=0.5, help="Noise clip parameter"
    )
    parser.add_argument("--a_max", type=float, default=1.0, help="Maximum action value")
    parser.add_argument(
        "--num_critic_updates_per_step",
        type=int,
        default=2,
        help="Number of critic updates per step",
    )
    parser.add_argument(
        "--td3_alpha", type=float, default=2.5, help="TD3 alpha parameter"
    )
    parser.add_argument(
        "--normalize_obs", action="store_true", help="Normalize observations"
    )

    # IQL
    parser.add_argument(
        "--iql_tau", type=float, default=0.7, help="Asymmetric L2 loss parameter"
    )
    parser.add_argument(
        "--iql_beta", type=float, default=3.0, help="Advantage scaling parameter"
    )

    # Logging
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--wandb_project", default=None, type=str, help="WandB project")
    parser.add_argument("--wandb_team", default=None, type=str, help="WandB team")
    parser.add_argument("--wandb_group", type=str, default="debug", help="Wandb group")

    parser_add_ma_args(parser)

    args, rest_args = parser.parse_known_args(cmd_args)
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    assert (
            not args.synthetic_experience
            or args.num_train_steps % args.synth_dataset_lifetime == 0
    ), "Number of train steps must be a multiple of the synthetic dataset lifetime"
    args.env_name = args.dataset_name
    return args


# Default diffusion models: (run id, plan length). The bundled qkmvppvt checkpoint
# (diff_checkpoints/qkmvppvt, DubinsCar, plan length 15) loads locally without wandb;
# add your own (run_id, plan_length) entries for wandb-hosted checkpoints.
DEFAULT_DIFFUSION_MODEL_LOAD = [('qkmvppvt', 15)]
DEFAULT_DIFFUSION_IND = 0  # Default diffusion model to load
DEFAULT_WANDB_TEAM = TRAINING_CONFIG['wandb']['default_wandb_team']
DEFAULT_WANDB_PROJECT = TRAINING_CONFIG['wandb']['default_wandb_project']
DEFAULT_LOADER_ARGS = ['--synthetic_experience', '--wandb_team', DEFAULT_WANDB_TEAM,
                       '--wandb_project', DEFAULT_WANDB_PROJECT, '--log', '--denoised_guidance',
                       '--policy_guidance_coeff', '1.0', '--normalize_action_guidance',
                       '--stl_guidance', '--num_synth_rollouts', '0']
STL_GUIDANCE_AFTER_FRACTION = PLANNER_CONFIG['diffusion']['stl_guidance_after_fraction']
ACH_GUIDANCE_AFTER_FRACTION = PLANNER_CONFIG['diffusion']['ach_guidance_after_fraction']
STL_RESAMPLE_THRESHOLD = PLANNER_CONFIG['diffusion']['stl_resample_threshold']
MAX_SAMPLING_ITER = PLANNER_CONFIG['diffusion']['max_sampling_iter']
PLAN_FROM_STATE = PLANNER_CONFIG['diffusion']['plan_from_state']


def pull_args_from_wandb(args=None, wandb_run_id=None):
    """Pull relevant args from the run config (local checkpoint first, wandb fallback).

    Ensures consistency between training and evaluation. A self-contained local
    checkpoint (see ``local_checkpoint_config``) is preferred so public users never
    need access to the wandb run the model came from.
    """
    if args is None:
        args = parse_agent_args(cmd_args=DEFAULT_LOADER_ARGS)

    # Set wandb_run_id to the default diffusion model to load if not provided
    if wandb_run_id is None:
        if args.denoiser_checkpoint is None:
            wandb_run_id = DEFAULT_DIFFUSION_MODEL_LOAD[DEFAULT_DIFFUSION_IND][0]
        else:
            wandb_run_id = args.denoiser_checkpoint
    from .logging import local_checkpoint_config
    local = local_checkpoint_config(wandb_run_id)
    if local is not None:
        ckpt_config, ckpt_source = local[0], local[1]
    else:
        import wandb
        api = wandb.Api()
        ckpt_run = api.run(
            f"{args.wandb_team}/{args.wandb_project}/{wandb_run_id}"
        )
        ckpt_config, ckpt_source = ckpt_run.config, ckpt_run.url
    args_to_pull = ['stl_train_traj_len']  # Pull these args from the run config
    map_args_to_pull = [['spec_len', 'plan_len']]  # Map these args to the pulled args
    dict_of_args = {}
    for arg, map_args in zip(args_to_pull, map_args_to_pull):
        for map_arg in map_args:
            dict_of_args[map_arg] = ckpt_config.get(arg)
    dict_of_args['wandb_run_id'] = wandb_run_id  # Setting in case default diffusion model is loaded
    print(f"Pulled args for run {wandb_run_id} ( {ckpt_source} ) : {dict_of_args}")
    return dict_of_args
