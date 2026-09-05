import tempfile
from argparse import Namespace
from orbax.checkpoint import PyTreeCheckpointer

from .diffusion import create_denoiser_train_state, make_sample_fn
from ..environments.offline_rollout import DatasetRolloutGenerator
from ..rl.agents import DETERMINISTIC_ACTORS
from ..util import *


class SyntheticRolloutGenerator(DatasetRolloutGenerator):
    def __init__(
            self,
            rng,
            denoiser_args,
            obs_shape,
            action_dim,
            action_lims,
            num_env_steps,
            agent_apply_fn=None,
            batch_size=None,
            goal_dim=None
    ):
        self.num_env_steps = num_env_steps
        self.agent_apply_fn = agent_apply_fn
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.action_lims = action_lims
        self.policy_guidance_coeff = denoiser_args.policy_guidance_coeff
        self.policy_guidance_cosine_coeff = denoiser_args.policy_guidance_cosine_coeff
        self.num_synth_workers = denoiser_args.num_synth_workers
        self.num_synth_rollouts = denoiser_args.num_synth_rollouts
        if goal_dim is None:
            goal_dim = obs_shape[0]

        if not denoiser_args.denoiser_checkpoint:
            raise ValueError(
                "Must specify generator checkpoint to use synthetic experience"
            )
        self._restore_diffusion_model(denoiser_args)
        if denoiser_args.diffusion_method != self.denoiser_config.diffusion_method:
            # Update diffusion method if changed from the checkpoint
            print(f"Diffusion method changed to {denoiser_args.diffusion_method} from "
                  f"{self.denoiser_config.diffusion_method}")
            self.denoiser_config.diffusion_method = denoiser_args.diffusion_method
        self.obs_stats = self.denoiser_norm_stats["obs"]
        det_guidance = denoiser_args.agent in DETERMINISTIC_ACTORS
        self.diffusion_sample_fn = partial(
            make_sample_fn(
                self.denoiser_config,
                denoiser_args.normalize_action_guidance,
                denoiser_args.denoised_guidance,
                det_guidance,
                denoiser_args.stl_guidance,
                agent_mask_noise=denoiser_args.agent_mask_noise,
                achievable_loss_coeff=denoiser_args.achievable_loss_coeff,
                achievable_guidance=denoiser_args.achievable_guidance,
                ma_stl_loss_coeff=denoiser_args.ma_stl_loss_coeff,
                auxiliary_training_loss=denoiser_args.auxiliary_training_loss,
                auxiliary_loss_rollout_length=denoiser_args.auxiliary_loss_rollout_length,
                alternate_update=denoiser_args.ma_stl_alternate_guidance,
                alternate_every=denoiser_args.ma_stl_alternate_every,
                resample_batch_size=denoiser_args.resample_batch_size,  # Pass resample_batch_size from config
                skip_resample=denoiser_args.skip_resample,
                ach_guidance_after_fraction=getattr(denoiser_args, 'ach_guidance_after_fraction', None),
                smooth_loss_coeff=getattr(denoiser_args, 'smooth_loss_coeff', 0.0),
                dense_achievable_supervision=getattr(denoiser_args, 'dense_achievable_supervision', True),
                guidance_hardness=getattr(denoiser_args, 'guidance_hardness', 100.0),
                segment_stl_check=getattr(denoiser_args, 'segment_stl_check', False),
                segment_stl_coeff=getattr(denoiser_args, 'segment_stl_coeff', 0.5),
                segment_interp_points=getattr(denoiser_args, 'segment_interp_points', 5),
                global_stl_only=getattr(denoiser_args, 'global_stl_only', False),
            ),
            denoiser_state=self.denoiser_state,
            # num_env_steps is the checkpoint's training horizon; the +1 is the pinned
            # start state, since denoise_step forces index 0 to the agent's current state.
            seq_len=self.num_env_steps + 1,
            obs_dim=self.obs_shape[0],
            action_dim=self.action_dim,
            denoiser_norm_stats=self.denoiser_norm_stats,
            policy_guidance_coeff=self.policy_guidance_coeff,
            policy_guidance_cosine_coeff=self.policy_guidance_cosine_coeff,
            goal_dim=goal_dim,
        )

        # Create batched sampling function for multiple candidates
        self.diffusion_batched_sample_fn = None  # Will be set when needed

        # Store parameters for creating batched sampling function
        self.batched_sample_params = {
            'denoiser_config': self.denoiser_config,
            'denoiser_args': denoiser_args,
            'det_guidance': det_guidance,
            'goal_dim': goal_dim,
        }

        # Generate unguided synthetic dataset
        if denoiser_args.num_synth_rollouts > 0:
            self.update_synthetic_dataset(rng, None)
            if batch_size is None:
                batch_size = denoiser_args.batch_size
            super().__init__(self._dataset, batch_size)

    def set_apply_fn(self, agent_apply_fn):
        self.agent_apply_fn = agent_apply_fn

    def _generate_single_rollout(self, rng, agent_params, stl_form_eval=None, **kwargs):
        return self.diffusion_sample_fn(
            rng=rng, agent_params=agent_params, agent_apply_fn=self.agent_apply_fn,
            stl_form_eval=stl_form_eval
        )

    def generate_single_rollout_ma(self, rng, agent_params, stl_form_eval=None, rollout_fn=None, env=None, **kwargs):
        """Generate a single rollout with multiple agents. Includes a rollout function to do more informative sampling."""
        return self.diffusion_sample_fn(
            rng=rng, agent_params=agent_params, agent_apply_fn=self.agent_apply_fn,
            stl_form_eval=stl_form_eval, rollout_fn=rollout_fn, env=env, **kwargs
        )

    def _create_batched_sample_fn(self, num_candidates=4, selection_criterion="stl_score"):
        """Create the batched sampling function if it doesn't exist."""
        if self.diffusion_batched_sample_fn is None:
            from .diffusion import make_batched_sample_fn

            self.diffusion_batched_sample_fn = partial(
                make_batched_sample_fn(
                    self.batched_sample_params['denoiser_config'],
                    self.batched_sample_params['denoiser_args'].normalize_action_guidance,
                    self.batched_sample_params['denoiser_args'].denoised_guidance,
                    self.batched_sample_params['det_guidance'],
                    self.batched_sample_params['denoiser_args'].stl_guidance,
                    agent_mask_noise=self.batched_sample_params['denoiser_args'].agent_mask_noise,
                    achievable_loss_coeff=self.batched_sample_params['denoiser_args'].achievable_loss_coeff,
                    achievable_guidance=self.batched_sample_params['denoiser_args'].achievable_guidance,
                    ma_stl_loss_coeff=self.batched_sample_params['denoiser_args'].ma_stl_loss_coeff,
                    auxiliary_training_loss=self.batched_sample_params['denoiser_args'].auxiliary_training_loss,
                    auxiliary_loss_rollout_length=self.batched_sample_params[
                        'denoiser_args'].auxiliary_loss_rollout_length,
                    alternate_update=self.batched_sample_params['denoiser_args'].ma_stl_alternate_guidance,
                    alternate_every=self.batched_sample_params['denoiser_args'].ma_stl_alternate_every,
                    num_candidates=num_candidates,
                    selection_criterion=selection_criterion,
                    resample_batch_size=self.batched_sample_params['denoiser_args'].resample_batch_size,
                    skip_resample=self.batched_sample_params['denoiser_args'].skip_resample,
                    ach_guidance_after_fraction=getattr(self.batched_sample_params['denoiser_args'],
                                                       'ach_guidance_after_fraction', None),
                ),
                denoiser_state=self.denoiser_state,
                # +1 for the start state pinned at index 0 by denoise_step
                seq_len=self.num_env_steps + 1,
                obs_dim=self.obs_shape[0],
                action_dim=self.action_dim,
                denoiser_norm_stats=self.denoiser_norm_stats,
                policy_guidance_coeff=self.policy_guidance_coeff,
                policy_guidance_cosine_coeff=self.policy_guidance_cosine_coeff,
                goal_dim=self.batched_sample_params['goal_dim'],
            )

    def generate_batched_rollout_ma(self, rng, agent_params, num_candidates=4, selection_criterion="stl_score",
                                    stl_form_eval=None, rollout_fn=None, env=None, **kwargs):
        """Generate multiple candidate rollouts with multiple agents and select the best one."""
        self._create_batched_sample_fn(num_candidates, selection_criterion)
        return self.diffusion_batched_sample_fn(
            rng=rng, agent_params=agent_params, agent_apply_fn=self.agent_apply_fn,
            stl_form_eval=stl_form_eval, rollout_fn=rollout_fn, env=env, **kwargs
        )

    def update_synthetic_dataset(self, rng, agent_params=None):
        # Regenerate synthetic dataset from the current agent state
        synth_rollouts = []
        batch_rollout_fn = jax.jit(
            jax.vmap(self._generate_single_rollout, in_axes=(0, None))
        )
        for _ in range(self.num_synth_rollouts):
            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, self.num_synth_workers)
            synth_rollouts.append(batch_rollout_fn(_rng, agent_params))
        # Stack and flatten rollouts
        self._dataset = jax.jit(
            lambda x: jtu.tree_map(
                lambda y: y.reshape((-1, y.shape[-1])), tree_stack(x)
            )
        )(synth_rollouts)

    def _restore_diffusion_model(self, args):
        # Self-contained local checkpoint first (e.g. the bundled diff_checkpoints/qkmvppvt):
        # config + norm stats come from its config.yaml and the orbax blobs from the same
        # directory, so no wandb access is needed. Falls back to downloading from wandb.
        local = local_checkpoint_config(args.denoiser_checkpoint)
        if local is not None:
            ckpt_dict, ckpt_root = local
            ckpt_source = os.path.abspath(ckpt_root)
        else:
            # Download checkpoint from wandb
            api = wandb.Api()
            ckpt_run = api.run(
                f"{args.wandb_team}/{args.wandb_project}/{args.denoiser_checkpoint}"
            )
            ckpt_root = os.path.join(tempfile.gettempdir(), args.denoiser_checkpoint)
            if not os.path.exists(ckpt_root):
                # Not downloaded already
                for file in ckpt_run.files():
                    with file.download(root=ckpt_root, exist_ok=True) as f:
                        # Download the file
                        pass
            ckpt_dict = ckpt_run.config
            ckpt_source = ckpt_run.url
        # Create placeholder train state
        if 'diffusion_trajectory_mode' not in ckpt_dict:
            # Default to SARD
            ckpt_dict['diffusion_trajectory_mode'] = 'sard'
        if 'achievable_loss_coeff' not in ckpt_dict:
            # Default to 1.0 (training hyperparameter that may not be used in sampling)
            ckpt_dict['achievable_loss_coeff'] = 1.0
        if args.diffusion_method == "edm-ma":
            # Set the loaded diffusion sampling method to EDM-MA if specified by the planner
            # Note: This assumes compatibility between training with EDM and sampling with EDM-MA
            ckpt_dict["diffusion_method"] = "edm-ma"
        self.denoiser_config = Namespace(**ckpt_dict)
        if args.diffusion_timesteps is not None:
            self.denoiser_config.diffusion_timesteps = args.diffusion_timesteps
        placeholder_train_state = create_denoiser_train_state(
            jax.random.PRNGKey(0),
            self.obs_shape[0],
            self.action_dim,
            self.denoiser_config,
            10000,  # Random dataset length to create LR schedule
        )
        # Restore checkpoint into placeholder train state
        ckptr = PyTreeCheckpointer()
        self.denoiser_state = ckptr.restore(
            os.path.abspath(os.path.join(ckpt_root, CHECKPOINT_DIR)),
            item=placeholder_train_state,
        )

        # Restore normalization statistics
        # wandb serialises numpy arrays in config.yaml as str(); parse them back.
        def conv_str(s):
            s = s.replace("\n", "")
            s = s.replace("[", "")
            s = s.replace("]", "")
            return [float(x) for x in s.split(" ") if x != ""]

        ckpt_dict["norm_stats"] = {
            k: {k1: v if not isinstance(v, str) else conv_str(v) for k1, v in x.items()}
            for k, x in ckpt_dict["norm_stats"].items()
        }
        self.denoiser_norm_stats = {
            attr: {
                stat_name: jnp.array(v, dtype=jnp.float32)
                for stat_name, v in attr_stats.items()
            }
            for attr, attr_stats in ckpt_dict["norm_stats"].items()
        }
        self.denoiser_norm_stats = jtu.tree_map(
            lambda x: jnp.expand_dims(x, 0) if len(x.shape) == 0 else x,
            self.denoiser_norm_stats,
        )
        print(f"Restored synthetic rollout generator from {args.denoiser_checkpoint} ({ckpt_source})")
