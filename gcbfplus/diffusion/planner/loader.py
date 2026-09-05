"""Loader for the diffusion planner."""
import functools
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from typing import Any

from gcbfplus.algo.module.planner.base import MultiAgentPlanner
from ..diffusion.rollout_generator import SyntheticRolloutGenerator
from ..util.stl_fastpath import shared_structural_override
from ..util.args import parse_agent_args as DiffusionArgs, DEFAULT_LOADER_ARGS, DEFAULT_DIFFUSION_MODEL_LOAD, \
    DEFAULT_DIFFUSION_IND, DIFFUSION_TESTING_CONFIG, DIFFUSION_TRAINING_CONFIG, DIFFUSION_TRAINING_KEY, \
    DIFFUSION_TESTING_KEY, PLAN_FROM_STATE
from ...env import MultiAgentEnv
from ...utils.graph import GraphsTuple


class DiffusionMAPlanner(MultiAgentPlanner):
    """Generates a plan for all agents in the environment using the diffusion model."""

    def init(self, key, obs: GraphsTuple, n_agents: int):
        # Stateless: plans are sampled fresh on every forward call.
        pass

    @property
    def args_of_interest(self) -> list:
        return ["achievable_guidance", "achievable_loss_coeff", "agent_mask_noise",
                "ma_stl_loss_coeff", "auxiliary_training_loss", "auxiliary_loss_rollout_length",
                "ma_stl_alternate_guidance", "ma_stl_alternate_every", "auxiliary_group_agents_as",
                "num_candidates", "selection_criterion", "use_batched_sampling",
                "ach_guidance_after_fraction", "smooth_loss_coeff", "global_stl_only"] + \
            super().args_of_interest

    @property
    def planner_all_info_dict(self) -> dict:
        info_dict = super().planner_all_info_dict

        additional_training_info = DIFFUSION_TRAINING_CONFIG
        additional_testing_info = DIFFUSION_TESTING_CONFIG

        info_dict[DIFFUSION_TRAINING_KEY] = additional_training_info
        info_dict[DIFFUSION_TESTING_KEY] = additional_testing_info
        return info_dict

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, action_dim: int, goal_dim: int,
                 agent_apply_fn=None, rng=None, state_dim: int = None, stl_form=None, diffusion_method='edm',
                 achievable_guidance=False, achievable_loss_coeff=1.0, ma_stl_loss_coeff=1.0,
                 auxiliary_training_loss=None, auxiliary_loss_rollout_length=None,
                 agent_mask_noise=False,
                 ma_stl_alternate_guidance=False, ma_stl_alternate_every=8,
                 skip_guidance=False,
                 ach_guidance_after_fraction=None,
                 smooth_loss_coeff=0.0,
                 dense_achievable_supervision=True,
                 global_stl_only=False,
                 guidance_hardness=100.0,
                 segment_stl_check=False,
                 segment_stl_coeff=0.5,
                 segment_interp_points=5,
                 **kwargs):
        """Load the diffusion model."""

        super().__init__(num_agents, node_dim, edge_dim, action_dim, goal_dim)

        # Load the diffusion model from wandb
        wandb_run_id = kwargs.get("wandb_run_id", None)
        stl_train_traj_len = kwargs.get("plan_length", None)  # Plan length of diffusion model
        if wandb_run_id is None:
            # Default diffusion model if not provided
            wandb_run_id, stl_train_traj_len = DEFAULT_DIFFUSION_MODEL_LOAD[DEFAULT_DIFFUSION_IND]

        diffusion_args = ['--denoiser_checkpoint', wandb_run_id, '--stl_train_traj_len', f"{stl_train_traj_len}",
                          '--diffusion_method', f"{diffusion_method}",
                          "--achievable_loss_coeff", f"{achievable_loss_coeff}",
                          "--ma_stl_loss_coeff", f"{ma_stl_loss_coeff}",
                          ] + DEFAULT_LOADER_ARGS
        if ach_guidance_after_fraction is not None:
            diffusion_args += ["--ach_guidance_after_fraction", f"{ach_guidance_after_fraction}"]
        if agent_mask_noise:
            diffusion_args += ["--agent_mask_noise"]
        if achievable_guidance:
            if auxiliary_training_loss is None:
                auxiliary_training_loss = "env-sync-test"
            diffusion_args += ["--achievable_guidance"]
            if auxiliary_training_loss is not None:
                diffusion_args += ["--auxiliary_training_loss",
                                   f"{auxiliary_training_loss}"]
            if auxiliary_loss_rollout_length is not None:
                diffusion_args += ["--auxiliary_loss_rollout_length",
                                   f"{auxiliary_loss_rollout_length}"]
        if ma_stl_alternate_guidance:
            diffusion_args += ["--ma_stl_alternate_guidance"]
            diffusion_args += ["--ma_stl_alternate_every", f"{ma_stl_alternate_every}"]
        if smooth_loss_coeff > 0:
            diffusion_args += ["--smooth_loss_coeff", f"{smooth_loss_coeff}"]
        if not dense_achievable_supervision:
            diffusion_args += ["--no_dense_achievable_supervision"]
        if global_stl_only:
            diffusion_args += ["--global_stl_only"]
        if guidance_hardness != 100.0:
            diffusion_args += ["--guidance_hardness", f"{guidance_hardness}"]
        if segment_stl_check:
            diffusion_args += ["--segment_stl_check"]
            diffusion_args += ["--segment_stl_coeff", f"{segment_stl_coeff}"]
            diffusion_args += ["--segment_interp_points", f"{segment_interp_points}"]

        # Add batched sampling arguments from kwargs
        use_batched_sampling = kwargs.get('use_batched_sampling', False)
        num_candidates = kwargs.get('num_candidates', 4)
        selection_criterion = kwargs.get('selection_criterion', 'stl_score')
        stl_weight = kwargs.get('stl_weight', 1.0)
        ma_stl_weight = kwargs.get('ma_stl_weight', 1.0)
        achievable_weight = kwargs.get('achievable_weight', 0.5)
        resample_batch_size = kwargs.get('resample_batch_size', 1)
        skip_resample = kwargs.get('skip_resample', False)

        if use_batched_sampling:
            diffusion_args += ["--use_batched_sampling"]
        diffusion_args += ["--num_candidates", f"{num_candidates}"]
        diffusion_args += ["--selection_criterion", f"{selection_criterion}"]
        diffusion_args += ["--stl_weight", f"{stl_weight}"]
        diffusion_args += ["--ma_stl_weight", f"{ma_stl_weight}"]
        diffusion_args += ["--achievable_weight", f"{achievable_weight}"]
        diffusion_args += ["--resample_batch_size", f"{resample_batch_size}"]
        if skip_resample:
            diffusion_args += ["--skip_resample"]

        if skip_guidance:
            print("\nSKIPPING GUIDANCE SINCE skip_guidance IS SET\n")
            diffusion_args.remove("--stl_guidance")
        args = DiffusionArgs(cmd_args=diffusion_args)
        self.args = args

        # Store batched sampling parameters as instance variables
        self.use_batched_sampling = args.use_batched_sampling
        self.num_candidates = args.num_candidates
        self.selection_criterion = args.selection_criterion
        self.stl_weight = args.stl_weight
        self.ma_stl_weight = args.ma_stl_weight
        self.achievable_weight = args.achievable_weight

        # seq_len is fixed by the checkpoint's training horizon (stl_train_traj_len);
        # test.py overrides --spec-len with it so evaluation matches how the model trained.
        num_env_steps = stl_train_traj_len
        rng, _rng = jax.random.split(rng)

        self.obs_shape = (node_dim, edge_dim)  # This is graph observation
        self.state_dim = (state_dim, 1)
        self.action_lims = (-1.0, 1.0)
        assert 0 < args.synth_batch_size <= args.batch_size
        self.synth_batch_size = args.synth_batch_size
        self.real_batch_size = args.batch_size - self.synth_batch_size
        self.synth_batch_lifetime = args.synth_batch_lifetime
        assert self.synth_batch_size % self.synth_batch_lifetime == 0
        self.synth_batch_size = self.synth_batch_size // self.synth_batch_lifetime
        self.diffusion_method = args.diffusion_method

        self.rollout_gen = SyntheticRolloutGenerator(_rng,
                                                     args,
                                                     self.state_dim,
                                                     self.action_dim,
                                                     self.action_lims,
                                                     num_env_steps,
                                                     agent_apply_fn,
                                                     self.synth_batch_size,
                                                     goal_dim)

    # jit with self/stl_forms static: recompiles per (spec structure, N); goal_centers and
    # goal_sizes are traced, so per-episode goal layouts do not force a retrace.
    @functools.partial(jax.jit,
                       static_argnames=('self', 'stl_forms_eval', 'stl_forms', 'rollout_fn', 'env', 'ma_stl_spec_eval',
                                        'ma_stl_gate_eval', 'task_stl_specs', 'task_ms', 'task_branches'))
    def forward(self, params, obs: GraphsTuple, key=None, stl_form_eval=None, stl_forms=None,
                rollout_fn=None, env=None, ma_stl_spec_eval=None, ma_stl_gate_eval=None, goal_centers=None,
                goal_sizes=None, task_stl_specs=None, task_ms=None, task_branches=None,
                **kwargs) -> tuple[Any, Any]:
        """Forward pass of the planner.

        :param goal_centers: optional TRACED per-agent goal-center array, shape
            ``(num_agents, n_goals, 2)``. When provided, the STL guidance/selection
            reads goal coordinates from it (via ``cent_override``) instead of the
            baked predicate centers in ``stl_forms``. Because ``goal_centers`` is a
            dynamic (non-static) jit argument while ``stl_forms`` stays a fixed
            structural object, changing the goal layout no longer recompiles this
            forward pass. ``None`` reproduces the original baked-center behaviour.
        """
        # For each agent run the rollout generator with guidance
        if key is None:
            rng = jax.random.PRNGKey(0)
        else:
            rng = key

        # Matching STL eval fn signature with time as 2nd arg. With goal_centers, close
        # over a per-agent traced center slice so the spec structure stays static. With
        # goal_sizes too, the override is a (cents, sizes) tuple the RectReachPredicate
        # leaf unpacks — both dynamic jit args, still zero recompile.
        #
        # O(N) fast path (traced AND fixed-grid): when all N per-agent forms share one
        # AST, evaluate the single structural form with a per-agent indexed override
        # instead of lax.switch (see _shared_structural_override). Falls back to the
        # switch closures whenever exactness cannot be proven.
        shared_struct_eval = shared_structural_override(stl_forms, goal_centers, goal_sizes)
        if goal_centers is not None:
            ovr = [goal_centers[i] if goal_sizes is None else (goal_centers[i], goal_sizes[i])
                   for i in range(self.num_agents)]
            stl_form_fns = [(lambda x, i=i: stl_forms[i].eval(x, cent_override=ovr[i]))
                            for i in range(self.num_agents)]
            stl_form_fns_train = [(lambda x, i=i: stl_forms[i].eval_train(x, cent_override=ovr[i]))
                                  for i in range(self.num_agents)]
        else:
            stl_form_fns = [stl_forms[i].eval for i in range(self.num_agents)]
            stl_form_fns_train = [stl_forms[i].eval_train for i in range(self.num_agents)]

        flat_obs = obs.type_states(MultiAgentEnv.AGENT, self.num_agents)
        x_0s = stop_gradient(flat_obs[:, :self.goal_dim])

        if self.diffusion_method == 'edm-ma':
            if shared_struct_eval is not None:
                stl_form_eval, stl_form_eval_train = shared_struct_eval
            else:
                def stl_form_eval_train(x, agent_id):
                    return jax.lax.switch(agent_id, stl_form_fns_train, x)

                def stl_form_eval(x, agent_id):
                    return jax.lax.switch(agent_id, stl_form_fns, x)

            agent_params = {'x0': x_0s}  # Set initial state to the observation
            # Pass avoid regions for segment-based avoidance in diffusion guidance
            avoid_regions = kwargs.pop('avoid_regions', None)

            # Counting-aware repair for CaTL+ (disjunctive mode): per-task inner STL
            # evals, each mapping (num_agents, T, goal_dim) -> per-agent robustness.
            # Static tuples built once per spec, so no per-episode recompile.
            # NOTE: repair needs inner forms whose predicate centers are baked in. With
            # traced goal_centers those centers are not overridden here, so repair is
            # skipped and the resample loop falls back to the plain joint CaTL+
            # acceptance gate.
            task_kwargs = {}
            if task_stl_specs is not None and goal_centers is None:
                task_inner_eval_fns = tuple((lambda y, _f=f: _f.eval(y)) for f in task_stl_specs)
                task_kwargs = dict(task_inner_eval_fns=task_inner_eval_fns,
                                   task_ms=task_ms, task_branches=task_branches)

            # Choose between batched and single sampling using instance variables
            if self.use_batched_sampling:
                print("\nUsing batched sampling in diffusion planner\n")
                generated_transitions, info = self.rollout_gen.generate_batched_rollout_ma(
                    rng, agent_params=agent_params,
                    stl_form_eval=stl_form_eval,
                    rollout_fn=rollout_fn,
                    env=env,
                    ma_stl_form_eval=ma_stl_spec_eval,
                    ma_stl_gate_eval=ma_stl_gate_eval,
                    num_candidates=self.num_candidates,
                    selection_criterion=self.selection_criterion,
                    stl_weight=self.stl_weight,
                    ma_stl_weight=self.ma_stl_weight,
                    achievable_weight=self.achievable_weight,
                    avoid_regions=avoid_regions,
                    **task_kwargs,
                )
            else:
                generated_transitions, info = self.rollout_gen.generate_single_rollout_ma(
                    rng, agent_params=agent_params,
                    stl_form_eval=stl_form_eval,
                    rollout_fn=rollout_fn,
                    env=env,
                    ma_stl_form_eval=ma_stl_spec_eval,
                    ma_stl_gate_eval=ma_stl_gate_eval,
                    avoid_regions=avoid_regions,
                    **task_kwargs,
                )
            # PLAN_FROM_STATE selects the denoised STATE channel (x, y) as the plan;
            # otherwise the ACTION channel is used, which in the 'sg' trajectory mode
            # holds the goal the demonstration was tracking.
            if PLAN_FROM_STATE:
                plan = generated_transitions.obs[:, :, :self.goal_dim]
            else:
                plan = generated_transitions.action[:, :, :self.goal_dim]
        else:
            # Run the rollout generator for each agent independently
            # Note: Batched sampling not implemented for non-edm-ma methods yet

            all_key = jax.random.split(key, self.num_agents + 1)
            agent_keys, key = all_key[:-1], all_key[-1]

            @jax.jit
            def run_diffusion_static(x_0, key, agent_id):
                def stl_form_eval_train(x):
                    if shared_struct_eval is not None:
                        return shared_struct_eval[1](x, agent_id)
                    return jax.lax.switch(agent_id, stl_form_fns_train, x)

                def stl_form_eval(x):
                    # Same O(N) shared-structural fast path as edm-ma: one structural
                    # form + per-agent indexed override instead of lax.switch (which
                    # evaluates all N branches under vmap -> O(N^2) guidance evals)
                    if shared_struct_eval is not None:
                        return shared_struct_eval[0](x, agent_id)
                    return jax.lax.switch(agent_id, stl_form_fns, x)

                agent_params = {'x0': x_0}  # Set initial state to the observation

                def one_candidate(k):
                    generated_transitions, info = self.rollout_gen._generate_single_rollout(
                        k, agent_params=agent_params, stl_form_eval=stl_form_eval)
                    plan = generated_transitions.obs[:, :self.goal_dim]
                    return plan, info

                # Per-agent best-of-N (mirrors edm-ma's --use_batched_sampling +
                # --num_candidates): draw N independent candidates per agent, keep the
                # one with the best exact robustness ('stl_loss' from the accept loop,
                # computed with the same stl_form_eval used for acceptance).
                if self.use_batched_sampling and self.num_candidates > 1:
                    cand_keys = jax.random.split(key, self.num_candidates)
                    plans, infos = jax.vmap(one_candidate)(cand_keys)
                    best = jnp.argmax(infos['stl_loss'])
                    plan = plans[best]
                    info = jax.tree_util.tree_map(lambda a: a[best], infos)
                else:
                    plan, info = one_candidate(key)
                return plan, info

            # Use vmap to vectorize over x_0s, agent_keys and the agent index. An
            # in_axes=(0, 0, None) variant with a constant agent_id traces slightly
            # faster but cannot select a different STL form per agent.
            plan, info = jax.vmap(run_diffusion_static, in_axes=(0, 0, 0))(x_0s, agent_keys,
                                                                           jnp.arange(self.num_agents))

        # Update info for consistency
        if 'num_resampling_iters' in info:
            if isinstance(info['num_resampling_iters'], int):
                # To handle case where guidance is not used
                info['max_num_resampling_iters'] = info['num_resampling_iters']
            else:
                info['max_num_resampling_iters'] = info['num_resampling_iters'].max()

        return plan, info
