"""Wrappers/Mixins for any environment."""
import jax
import jax.numpy as jnp
import logging
import numpy as np
import os
import pathlib
import string
from jax.lax import stop_gradient
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from typing import Any, Optional
from typing import Tuple

import ds
from gcbfplus.utils.graph import GraphsTuple

os.environ["DIFF_STL_BACKEND"] = "jax"
from ds.stl import StlpySolver
import ds.stl_jax
from gcbfplus.env import MultiAgentEnv
from gcbfplus.env.base import RolloutResult
from gcbfplus.stl.utils import STL_INFO_KEYS, TRAINING_CONFIG, ENV_CONFIG, STL_PY_NAME, GLOBAL_STL_PY_NAME, \
    LARGE_HARDNESS, DEFAULT_TtR, STL_EVAL_TOLERANCE, \
    DIFFUSION_NAME, PLANNER_CONFIG, MA_STL_EVAL_TOLERANCE
from gcbfplus.utils.typing import Action, Array, Cost, Done, Info, Reward

from contextlib import contextmanager


ds.stl.HARDNESS = TRAINING_CONFIG['ds_params']['HARDNESS']
ds.stl_jax.HARDNESS = TRAINING_CONFIG['ds_params']['HARDNESS']  # Set hardness for easier backpropagation


@contextmanager
def set_stl_jax_hardness(hardness: float):
    """Set the hardness of the softmax function for the duration of the context.
    Useful for making evaluation strict while allowing gradients to pass through during training.
    Note: using ds.stl_jax.set_hardness does not set global HARDNESS for ds.stl_jax

    :param hardness: hardness of the softmax function
    :type hardness: float
    """
    # global HARDNESS
    old_hardness = ds.stl_jax.HARDNESS
    ds.stl_jax.HARDNESS = hardness
    yield
    ds.stl_jax.HARDNESS = old_hardness


from gcbfplus.env.wrapper.stl_mixin import MASTLMixin
from gcbfplus.env.wrapper.base import PlannerWrapper
from gcbfplus.diffusion.planner import DiffusionMAPlanner


class STLWrapper(MASTLMixin, PlannerWrapper):
    """Adds an STL specification and a waypoint planner (per-agent STLPY MILP, STLPY-Global joint MILP,
    or the diffusion planner) to a multi-agent environment, and scores executed trajectories against it."""
    planner: StlpySolver | DiffusionMAPlanner | None = None

    def __init__(self, *args, stl_solver=STL_PY_NAME, spec=None, spec_len=400, device=None, max_step=None,
                 goal_sample_interval=1, plan_length=None, set_agent_goals_args=None, planner_config=None,
                 key=None, ma_stl_spec=None, ma_stl_exp_rob=True, skip_guidance=False,
                 allocation_strategy='greedy', team_disjunctive=False, team_avoid=False, avoid_expansion=1.2,
                 stlpy_u_bound=None, **kwargs):
        """Initialize the environment. Choose between the STLpy and diffusion solvers.

        :param planner_config: specific configuration for the planner
        :type planner_config: dict
        :param ma_stl_spec: Optional Multi-agent STL spec (on the whole system such as CATL+)
        :param allocation_strategy: 'greedy' | 'random' | 'greedy_last' | 'oracle_hungarian' for team specs
        :param team_disjunctive: if True, give every agent the same disjunctive spec (OR over
            tasks) so CaTL+ gradient discovers allocation during diffusion sampling
        """
        self.max_step = int(spec_len) if max_step is None else max_step
        self.max_episode_steps = self.max_step  # Backward compatibility
        self.spec_len = spec_len
        self.plan_length = self.spec_len if plan_length is None else plan_length
        self.stl_solver = stl_solver
        self.planner_config = planner_config
        self.skip_guidance = skip_guidance
        self.allocation_strategy = allocation_strategy
        self.team_disjunctive = team_disjunctive
        self.team_avoid = team_avoid
        self.avoid_expansion = avoid_expansion
        self.global_stlpy_time_limit = PLANNER_CONFIG.get('global_stlpy_time_limit', 600)
        # Load stlpy_u_bound: CLI overrides config default
        if stlpy_u_bound is not None:
            self.stlpy_u_bound = stlpy_u_bound
        else:
            self.stlpy_u_bound = PLANNER_CONFIG.get('stlpy_u_bound', 2.0)

        self.logger = logging.getLogger(__name__)
        self.spec = f'seq2' if spec is None else spec

        super().__init__(*args, device=device, max_step=self.max_step, goal_sample_interval=goal_sample_interval,
                         plan_length=self.plan_length, key=key, **kwargs)
        self._set_goal_set()
        set_agent_goals_args = {} if set_agent_goals_args is None else set_agent_goals_args
        self.ma_stl_spec = ma_stl_spec  # Multi-agent STL spec
        self.ma_stl_exp_rob = ma_stl_exp_rob  # Multi-agent STL robustness
        self._init_plan(spec=self.stl_string, agent_goals=self._set_agent_goals(**set_agent_goals_args),
                        ma_stl_spec=ma_stl_spec, key=key)

        self.env._render_extra = self._render_extra  # Since this is just a wrapper, we need to set the render extra

        all_stl_specs = set([stl_form.__repr__() for stl_form in self.stl_forms])
        if len(all_stl_specs) > 1:
            self.logger.info(f"Per-agent STL formulas differ ({len(all_stl_specs)} distinct forms)")
        agent_id = 0

        # Env steps per plan step: a plan has end_time() waypoints spread over max_step env steps.
        # The sync wrapper scores an executed trajectory on histories[goal_step-1::goal_step]
        # (one state per plan step); the async wrapper scores the recorded goal-change states instead.
        self.goal_step = self.max_step // self.stl_forms[agent_id].end_time()
        self.logger.info(f"Using {self.__class__.__name__} wrapper with spec {self.stl_string}")

    @property
    def args_of_interest_all(self):
        """Return the arguments of interest for the planner. This is used to set the arguments for the planner."""
        return ['_xy_min', '_xy_max'] + super().args_of_interest_all

    @property
    def stl_string(self):
        """Get the STL string corresponding to the spec and time length"""
        return f'{self.spec}_t{self.spec_len}'

    def get_all_predicates(self):
        """Get all predicates in the STL spec"""
        if self.spec[0] == 'm':
            # set of all stl_forms
            output = list(sorted(set([pred for stl_form in self.stl_forms for pred in stl_form.get_all_predicates()])))
        else:
            # Homogeneous spec, sorting for consistent labels with mixed
            output = list(sorted(self.stl_form.get_all_predicates()))

        # Filter out any boundary predicates (assumed to be set with name: int < 0 )
        output = [pred for pred in output if pred.name >= 0]
        return output

    def reset(self, key: Array, plan_settings=None, **kwargs) -> GraphsTuple:
        """Reset the environment."""
        if plan_settings is None:
            plan_settings = {'time_limit': 15 * 60 // self.env.num_agents,  # 15 min per run
                             'plan_length': self.plan_length}

        retval = super(PlannerWrapper, self).reset(key)
        # **kwargs are for any additional settings like the rollout fn
        self._init_plan_with_env(retval, key=key, **plan_settings, **kwargs)
        # Can change plot limits to display STL predicates
        shifted_cents = [x.cent - (x.size / 2) for x in self.get_all_predicates()]
        pred_sizes = [x.size for x in self.get_all_predicates()]
        # Check rectangles around predicates
        self._xy_min = [0, 0]
        self._xy_max = [self.area_size, self.area_size]
        for pred_size, cent in zip(pred_sizes, shifted_cents):
            lbc = cent
            ruc = cent + pred_size
            self._xy_min = np.minimum(self._xy_min, lbc)
            self._xy_max = np.maximum(self._xy_max, ruc)
        self.env._xy_min = self._xy_min
        self.env._xy_max = self._xy_max
        self.path = []  # Have made functional option by using graph.histories
        return self._init_stl_data(retval)

    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False) -> Tuple[
        Any, Reward, Cost, Done, Info]:
        """Take a step in the environment.

        :param graph: current graph
        :type graph: GraphsTuple"""

        graph = self.change_goals(graph)
        retval = super(PlannerWrapper, self).step(graph, action)

        info_dict = retval[-1]
        # Add STL specific data to observation
        new_graph, reward, cost, env_done, _ = retval
        # Can change done based on spec satisfaction if using graph.histories feature
        done = (graph.current_time >= self.max_step).all()  # or self.check_spec_satisfaction(self.path)
        new_retval = self._add_stl_data_to_obs(new_graph, graph), reward, cost, done, info_dict
        return new_retval

    def _add_stl_data_to_obs(self, new_graph: GraphsTuple, graph: GraphsTuple):
        """Add any STL specific data to observation"""
        new_graph = new_graph._replace(current_time=graph.current_time, current_plan=graph.current_plan,
                                       global_time=graph.global_time)
        return new_graph  # No STL data to add

    def _score_set_of_histories(self, histories, subsample=True):
        """Score a set of histories based on the STL spec"""
        if subsample:
            subsampled_hist = histories[self.goal_step - 1::self.goal_step]
        else:
            subsampled_hist = histories
        subsampled_hist = subsampled_hist.transpose((1, 0, 2))

        goal_centers = getattr(self, 'goal_centers', None)
        goal_sizes = getattr(self, 'goal_sizes', None)

        def eval_stl_form(history, stl_form, cent_override):
            return stl_form.eval(history, cent_override=cent_override)

        # Run each STL formula on the subsampled history for that agent. In --traced-goals
        # mode self.stl_forms is a fixed structural form, so the real per-episode goal
        # centers are supplied via cent_override (None otherwise -> baked centers, unchanged);
        # with random sizes the override is a (cents, sizes) tuple the predicate leaf unpacks.
        def _ovr(i):
            if goal_centers is None:
                return None
            return goal_centers[i] if goal_sizes is None else (goal_centers[i], goal_sizes[i])

        output = [eval_stl_form(subsampled_hist[i, None], self.stl_forms[i], _ovr(i))
                  for i in range(len(self.stl_forms))]
        return output

    def score_plan_robustness(self):
        """Per-agent exact robustness of the DELIVERED plan (planner-agnostic).

        Negative = the planner handed execution a spec-violating plan: the
        resample loop returned its least-violating draw at the iteration cap
        (diffusion) or the MILP was infeasible / returned a violating incumbent
        (stlpy) — the 'resampling exhaustion' failure channel. Mirrors the
        short-path branch of ``_score_path`` (plans are already at plan-step
        resolution, so no subsampling).
        """
        if getattr(self, 'plan', None) is None:
            return None
        score_tensor = jnp.array(self.plan).transpose((1, 0, 2))
        with set_stl_jax_hardness(LARGE_HARDNESS):
            scores = self._score_set_of_histories(score_tensor, subsample=False)
        return jnp.array(stop_gradient(scores)).reshape(-1)

    def _score_path(self, path, skip_subsample=False):
        path_scores = []
        tensor_part = jnp.stack(path)
        # A path of <= 2 states means no rollout was run (plan-only mode): score the plan itself.
        do_score_calculations = not (path.shape[0] <= 2)
        all_paths_tensor = stop_gradient(tensor_part)
        if do_score_calculations:
            # Convert to (T x N x D) format
            score_tensor = all_paths_tensor
            subsample = not skip_subsample
        else:
            self.logger.info("Skipping score calculations due to short path length")
            # Calculate scores on the plan directly if not rolling out
            score_tensor = jnp.array(self.plan).transpose((1, 0, 2))
            subsample = False
        with set_stl_jax_hardness(LARGE_HARDNESS):
            # Evaluate the STL spec on the path with true values
            path_scores = jnp.array(stop_gradient(self._score_set_of_histories(score_tensor, subsample=subsample)))
        max_path_score = path_scores.max()
        min_path_score = path_scores.min()
        mean_path_score = path_scores.mean()
        # Whether agent finished spec
        finish_rate = (path_scores > -STL_EVAL_TOLERANCE).squeeze(1)

        # Initialize score dictionary with basic STL metrics in correct order
        full_stl_score_dict = {}
        full_stl_score_dict[STL_INFO_KEYS[0]] = max_path_score  # max_path_score
        full_stl_score_dict[STL_INFO_KEYS[1]] = min_path_score  # min_path_score
        full_stl_score_dict[STL_INFO_KEYS[2]] = mean_path_score  # mean_path_score
        full_stl_score_dict[STL_INFO_KEYS[3]] = finish_rate  # finish_rate
        full_stl_score_dict[STL_INFO_KEYS[4]] = jnp.array(DEFAULT_TtR)  # TtR

        if self.ma_stl_spec is not None and do_score_calculations:
            # Calculate the multi-agent STL score — always the full eval form
            # (avoid_expansion=1.0). Branch reduction for OR specs lives on
            # ma_stl_form_guidance, not here.
            eval_form = self.ma_stl_form
            transpose_paths = all_paths_tensor.transpose((1, 0, 2))
            ma_stl_score = eval_form.eval(transpose_paths)
            full_stl_score_dict[STL_INFO_KEYS[5]] = ma_stl_score  # ma_stl_score
            full_stl_score_dict[STL_INFO_KEYS[6]] = ma_stl_score > MA_STL_EVAL_TOLERANCE  # ma_stl_satisfaction

            # Team task counting (for all team specs)
            # Use CaTL+ eval_whole_path as the authoritative metric — it evaluates the
            # full STL formula with proper time windows, reach, and avoid semantics.
            # Eval uses avoid_expansion=1.0 (actual goal size) for fair scoring.
            if self.team_spec_obj is not None:
                tasks_completed = 0
                per_task_completed = {}

                for task_obj in eval_form.get_all_predicates():
                    task_rob = task_obj.eval_whole_path(transpose_paths)
                    completed = bool(task_rob > MA_STL_EVAL_TOLERANCE)
                    per_task_completed[task_obj.name] = completed
                    if completed:
                        tasks_completed += 1

                # For OR specs: count best branch (tasks_total = tasks per branch)
                if self.team_spec_obj.branches is not None:
                    best_branch_completed = 0
                    tasks_per_branch = len(self.team_spec_obj.branches[0])
                    for branch_ids in self.team_spec_obj.branches:
                        branch_done = sum(
                            1 for tid in branch_ids
                            if per_task_completed.get(f"team_task_{tid}", False))
                        best_branch_completed = max(best_branch_completed, branch_done)
                    tasks_completed = best_branch_completed
                    tasks_total = tasks_per_branch
                else:
                    tasks_total = len(self.team_spec_obj.tasks)
                task_rate = tasks_completed / max(tasks_total, 1)
                full_stl_score_dict['tasks_completed'] = tasks_completed
                full_stl_score_dict['tasks_total'] = tasks_total
                full_stl_score_dict['task_rate'] = task_rate
                full_stl_score_dict['per_task_completed'] = per_task_completed
                if self.team_allocation is not None:
                    full_stl_score_dict['task_to_agents'] = dict(self.team_allocation.task_to_agents)

                # Override ma_stl_satisfaction for MofN (relaxed criterion)
                if self.team_spec_obj.spec_type == 'team_mofn':
                    full_stl_score_dict[STL_INFO_KEYS[6]] = tasks_completed >= self.team_spec_obj.m_required

                # Override per-agent finish_rate with task-level completion for team specs.
                # Per-agent finish_rate penalizes unassigned agents whose "stay in place" spec
                # may fail due to safety-induced drift. The team objective is task completion,
                # not individual agent plan tracking.
                # success_rate = safe_rate × (tasks_completed / tasks_total)
                team_finish = jnp.ones(self.num_agents) * (tasks_completed / tasks_total)
                full_stl_score_dict[STL_INFO_KEYS[3]] = team_finish  # finish_rate
                full_stl_score_dict['team_spec_success_mode'] = True  # flag for logging
        else:
            # For consistency in table (score, satisfaction, strict satisfaction)
            full_stl_score_dict[STL_INFO_KEYS[5]] = jnp.nan  # ma_stl_score
            full_stl_score_dict[STL_INFO_KEYS[6]] = jnp.nan  # ma_stl_satisfaction
            full_stl_score_dict[STL_INFO_KEYS[7]] = jnp.nan  # strict_ma_stl_satisfaction

        # Calculate trajectory-plan deviation if plan is available
        if hasattr(self, 'plan') and len(self.plan) > 0 and do_score_calculations:
            # Convert plan to proper format (T x N x D)

            if isinstance(self.plan, list):
                # Plan is list of paths per agent: convert to (T x N x D)
                plan_array = jnp.stack([jnp.array(agent_plan) for agent_plan in self.plan], axis=1)
            else:
                # Plan is already in array format
                plan_array = jnp.array(self.plan).transpose((1, 0, 2))
                if len(plan_array.shape) == 4:  # (1, T, N, D)
                    plan_array = plan_array[0]
                elif len(plan_array.shape) == 3 and plan_array.shape[0] == 1:  # (1, T, D) single agent
                    plan_array = plan_array[0]

            # Calculate deviation metrics
            deviation_metrics = self._calculate_trajectory_plan_deviation(all_paths_tensor, plan_array,
                                                                          skip_subsample=skip_subsample)
            full_stl_score_dict.update(deviation_metrics)
        else:
            # No plan available, set trajectory deviation metrics to NaN
            from gcbfplus.stl.utils import TRAJECTORY_DEVIATION_KEYS
            for key in TRAJECTORY_DEVIATION_KEYS:
                full_stl_score_dict[key] = jnp.nan

        # Set team info keys to NaN if not using team specs
        if not hasattr(self, 'team_spec_obj') or self.team_spec_obj is None:
            from gcbfplus.stl.utils import TEAM_INFO_KEYS
            for key in TEAM_INFO_KEYS:
                full_stl_score_dict[key] = jnp.nan

        return full_stl_score_dict, path_scores

    def _calculate_trajectory_plan_deviation(self, actual_trajectory, plan, skip_subsample=False):
        """Calculate the deviation between actual trajectory and planned path

        :param actual_trajectory: Actual trajectory taken by agents (T x N x D)
        :param plan: Planned trajectory (T x N x D) or (1 x T x N x D)
        :return: Dictionary with deviation metrics
        """
        # Handle plan shape - remove batch dimension if present
        if len(plan.shape) == 4:  # (1, T, N, D)
            plan = plan[0]  # Remove batch dimension

        # For synchronous planning, subsample actual trajectory at goal change intervals
        # This matches how goals are updated every goal_sample_interval steps
        if hasattr(self, 'goal_sample_interval') and not skip_subsample:
            # Subsample actual trajectory at goal change points
            # Start from goal_sample_interval-1 to align with when goals are actually reached
            subsample_indices = jnp.arange(self.goal_sample_interval - 1,
                                           actual_trajectory.shape[0],
                                           self.goal_sample_interval)
            actual_trajectory_subsampled = actual_trajectory[subsample_indices]
        else:
            # Fallback: use the entire trajectory
            actual_trajectory_subsampled = actual_trajectory

        # Ensure both trajectories have the same length by truncating to minimum
        min_length = min(actual_trajectory_subsampled.shape[0], plan.shape[0])
        actual_traj_truncated = actual_trajectory_subsampled[:min_length]
        plan_truncated = plan[:min_length]

        # Calculate Euclidean distance at each timestep for each agent
        # Shape: (T_subsampled, N)
        euclidean_distances = jnp.linalg.norm(actual_traj_truncated - plan_truncated, axis=-1)

        # Calculate per-agent metrics
        mean_deviation_per_agent = jnp.mean(euclidean_distances, axis=0)  # (N,)
        max_deviation_per_agent = jnp.max(euclidean_distances, axis=0)  # (N,)

        # Calculate overall metrics
        mean_deviation = jnp.mean(mean_deviation_per_agent)
        max_deviation = jnp.max(max_deviation_per_agent)
        std_deviation = jnp.std(mean_deviation_per_agent)

        # Calculate cumulative deviation (area under the deviation curve)
        # Note: This is now cumulative over the subsampled points
        cumulative_deviation_per_agent = jnp.sum(euclidean_distances, axis=0)  # (N,)
        mean_cumulative_deviation = jnp.mean(cumulative_deviation_per_agent)

        deviation_metrics = {
            'mean_traj_plan_deviation': mean_deviation,
            'max_traj_plan_deviation': max_deviation,
            'std_traj_plan_deviation': std_deviation,
            'cumulative_traj_plan_deviation': mean_cumulative_deviation,
        }

        return deviation_metrics

    def score_path(self, path):
        """Score the path based on the STL spec and add to info dict"""
        stl_score_dict, path_scores = self._score_path(path)
        stl_score_dict[STL_INFO_KEYS[4]] = jnp.array(DEFAULT_TtR)  # TtR for non-async
        return stl_score_dict

    def extra_summary_metrics(self, summary_metrics=None, finish_infos=None):
        """Get extra summary metrics to log"""
        if summary_metrics is None:
            summary_metrics = {}
        extra_metrics = super().extra_summary_metrics(summary_metrics, finish_infos)
        if self.ma_stl_spec is not None:
            # Add the strict satisfaction rate
            is_unsafes = jnp.any(summary_metrics['is_unsafe'], axis=-1)
            ma_stl_sat = jnp.stack([l[f'eval/{STL_INFO_KEYS[6]}'] for l in finish_infos])
            strict_ma_stl_sat = (1 - is_unsafes) * ma_stl_sat

            # Update the finish_infos with the strict satisfaction rate (to fit with existing pipeline)
            finish_infos = [{**l, f'eval/{STL_INFO_KEYS[7]}': strict_ma_stl_sat[i]} for i, l in enumerate(finish_infos)]

        return finish_infos, extra_metrics

    @property
    def aux_node_dim(self):
        """Get the dimension of the auxiliary nodes if add_aux_features is set"""
        return 1

    def _init_stl_data(self, data):
        """Add any initial STL specific data to observation"""
        dummy_plan = jnp.zeros(
            (1, self.plan_length, self.env.num_agents, self.env.goal_dim))  # to play nice with jax jit
        if len(self.plan) > 0:
            # Set fixed plan if generated (by MILP planner)
            dummy_plan = jnp.array(self.plan).transpose(1, 0, 2)[None]
        data = data._replace(current_time=jnp.tile(self.init_time, self.env.num_agents), current_plan=dummy_plan,
                             global_time=self.init_time)
        return data  # No STL data to add

    def _init_plan_with_env(self, obs: GraphsTuple, key=None, time_limit=20, plan_length=None, **kwargs):
        """Run planner from a given start state and make a plan for each agent

        :param obs: initial observation
        :type obs: GraphsTuple
        :param time_limit: time limit for planner
        additional kwargs could be the rollout_fn for the diffusion planner
        """

        flat_obs = obs.type_states(MultiAgentEnv.AGENT, self.env.num_agents)
        if key is None:
            key = jax.random.PRNGKey(0)

        # Re-allocate team tasks with real agent positions (always, for consistent evaluation)
        # Skip only for global solver (MILP handles allocation internally)
        if self.team_spec_obj is not None and self.stl_solver != GLOBAL_STL_PY_NAME:
            agent_positions = stop_gradient(flat_obs[:, :self.env.goal_dim]).__array__()
            self._reallocate_team_tasks(agent_positions)

        if self.stl_solver == STL_PY_NAME:
            solver = StlpySolver(space_dim=self.env.goal_dim)
            # Run for each agent
            x_0s = stop_gradient(flat_obs[:, :self.env.goal_dim])
            total_time = self.max_step if plan_length is None else plan_length  # Must be trajectory length
            self.plan = []
            self.init_time = 0
            import time

            u_bound = (-self.stlpy_u_bound, self.stlpy_u_bound)

            t0 = time.time()
            # In --traced-goals mode self.stl_forms is the fixed structural form (same component
            # assignment as the diffusion run); the real per-episode random goals come via
            # cent_override = goal_centers[agent], so STLPY solves the IDENTICAL map + spec.
            stlpy_goal_centers = getattr(self, 'goal_centers', None)
            stlpy_goal_sizes = getattr(self, 'goal_sizes', None)
            for agent_id, x_0 in enumerate(x_0s):
                if stlpy_goal_centers is None:
                    _cov = None
                elif stlpy_goal_sizes is None:
                    _cov = stlpy_goal_centers[agent_id]
                else:
                    # (cents, sizes) tuple: MILP boxes get the same per-episode random sizes.
                    _cov = (stlpy_goal_centers[agent_id], stlpy_goal_sizes[agent_id])
                stlpy_form = self.stl_forms[agent_id].get_stlpy_form(cent_override=_cov)
                path, info = solver.solve_stlpy_formula(stlpy_form, x0=x_0.__array__(), total_time=total_time,
                                                        time_limit=time_limit, u_bound=u_bound)
                self.logger.info(f"STLpy for agent {agent_id} x0 {x_0} info: {info}")
                if path is None:
                    self.logger.warning(f"Path not found for spec {self.stl_forms[agent_id]} agent {agent_id} x0 {x_0}")
                    path = np.tile(x_0, (total_time + 1, 1))  # Just stay in place
                self.plan.append(path[1:])  # Remove the initial state

            t1 = time.time()
            total_time = t1 - t0
            self.plan_info = {'plan_time': total_time}
        elif self.stl_solver == DIFFUSION_NAME:
            # Run for each agent
            total_time = self.max_step if plan_length is None else plan_length  # Must be trajectory length
            self.plan = []
            info = {}
            self.init_time = 0
            import time
            if self.ma_stl_spec is not None:
                # Use guidance CaTL+ (with expansion) for planner gradient
                guidance_catl = getattr(self, 'ma_stl_form_guidance', self.ma_stl_form)
                if self.ma_stl_exp_rob:
                    ma_stl_spec_eval = guidance_catl.eval_train
                else:
                    ma_stl_spec_eval = guidance_catl.eval
            else:
                ma_stl_spec_eval = None

            t0 = time.time()
            # Pass avoid regions for segment-based avoidance check in diffusion guidance
            avoid_regions = getattr(self, 'agent_avoid_regions', None)
            # Use disjunctive forms for guidance if available, else allocated forms
            guidance_forms = getattr(self, 'stl_forms_guidance', None) or self.stl_forms
            # Counting-aware repair (disjunctive CaTL+ only): per-task inner specs let
            # the sampler claim m_q agents per task and retarget only deficit agents.
            task_repair_kwargs = {}
            if (getattr(self, 'team_disjunctive', False)
                    and getattr(self, 'team_task_specs_guidance', None)
                    and getattr(self, 'goal_centers', None) is None):
                task_repair_kwargs = dict(
                    task_stl_specs=self.team_task_specs_guidance,
                    task_ms=self.team_task_ms,
                    task_branches=self.team_task_branches)
            # Traced goal centers (set by resample_goals in --traced-goals mode) flow as a
            # dynamic arg so changing the goal layout does not recompile forward. None in
            # the normal path -> baked predicate centers, unchanged behaviour.
            self.plan, info = self.planner.forward(None, obs, key=key,
                                                   stl_forms=tuple(guidance_forms), plan_length=total_time,
                                                   ma_stl_spec_eval=ma_stl_spec_eval,
                                                   ma_stl_gate_eval=(getattr(self, 'ma_stl_gate_eval', None)
                                                                     if ma_stl_spec_eval is not None else None),
                                                   avoid_regions=avoid_regions,
                                                   goal_centers=getattr(self, 'goal_centers', None),
                                                   goal_sizes=getattr(self, 'goal_sizes', None),
                                                   **task_repair_kwargs,
                                                   **kwargs)
            t1 = time.time()
            total_time = t1 - t0
            self.plan_info = {'plan_time': total_time}
            self.plan_info.update(info)

        elif self.stl_solver == GLOBAL_STL_PY_NAME:
            from gcbfplus.stl.global_stlpy_solver import GlobalStlpySolver

            assert self.team_spec_obj is not None, "stlpy_global requires a team spec"

            x_0s = stop_gradient(flat_obs[:, :self.env.goal_dim])
            agent_positions = np.array(x_0s)
            total_time = self.max_step if plan_length is None else plan_length
            goal_size = np.array([1, 1]) * ENV_CONFIG['goal_size']

            self.plan = []
            self.init_time = 0
            import time

            t0 = time.time()
            solver = GlobalStlpySolver(
                n_agents=self.num_agents, space_dim=self.env.goal_dim)
            global_u_bound = (-self.stlpy_u_bound, self.stlpy_u_bound)
            trajectories, info = solver.solve(
                team_spec=self.team_spec_obj,
                agent_positions=agent_positions,
                goal_size=goal_size,
                shrink_factor=self.stl_shrink_factor,
                total_time=total_time,
                time_limit=self.global_stlpy_time_limit,
                u_bound=global_u_bound,
                add_avoidance=self.team_avoid,
                avoid_expansion=self.avoid_expansion,
            )
            self.plan = [traj[1:] for traj in trajectories]  # Remove initial state
            t1 = time.time()
            self.plan_info = {'plan_time': t1 - t0}
            self.plan_info.update(info)

        else:
            raise NotImplementedError(f"MILP solver {self.stl_solver} not implemented")

        # Allocation happens at three points: (1) _load_team_spec from dummy positions (structure only);
        # (2) _reallocate_team_tasks from the real start positions at each reset -> pre_planned eval forms;
        # (3) with --team-disjunctive every agent was GUIDED by the OR-of-tasks form, so the allocation the
        # planner actually chose is inferred from the plan (Hungarian on per-task robustness) and the eval
        # forms are rebuilt from it. Evaluation never uses the disjunctive form itself.
        has_plan = hasattr(self, 'plan') and self.plan is not None and len(self.plan) > 0
        if (self.team_disjunctive and self.team_spec_obj is not None
                and has_plan
                and PLANNER_CONFIG.get('eval_allocation_mode', 'post_hoc') == 'post_hoc'):
            self._infer_and_update_eval_allocation()

    def _infer_and_update_eval_allocation(self):
        """Infer allocation from planned trajectories and rebuild eval specs."""
        from gcbfplus.stl.team_allocator import (infer_allocation_from_trajectories,
                                                  build_per_agent_stl_forms)
        try:
            traj_array = np.stack(self.plan)  # (N, T, D)
            goal_size = np.array([1, 1]) * ENV_CONFIG['goal_size']

            inferred_alloc = infer_allocation_from_trajectories(
                self.team_spec_obj, traj_array, goal_size, self.stl_shrink_factor,
                goal_sample_interval=self.goal_sample_interval)

            # Rebuild eval forms (no expansion) and guidance regions (with expansion)
            self.stl_forms_eval, _ = build_per_agent_stl_forms(
                self.team_spec_obj, inferred_alloc, goal_size, self.stl_shrink_factor,
                self.num_agents, add_avoidance=self.team_avoid,
                avoid_expansion=1.0)
            _, self.agent_avoid_regions = build_per_agent_stl_forms(
                self.team_spec_obj, inferred_alloc, goal_size, self.stl_shrink_factor,
                self.num_agents, add_avoidance=self.team_avoid,
                avoid_expansion=self.avoid_expansion)
            self.stl_forms = self.stl_forms_eval
            self.stl_form = self.stl_forms[0]
            self.team_allocation = inferred_alloc

            self.logger.info(f"Post-hoc allocation inferred: {inferred_alloc.task_to_agents}")
        except Exception as e:
            self.logger.warning(f"Post-hoc allocation inference failed: {e}. Using pre-planned allocation.")

    def _set_agent_goals(self, **kwargs):
        """Set agent goals for the environment if needed"""
        return self.GOAL_SET

    def _init_plan(self, spec=None, agent_goals=None, ma_stl_spec=None, key=None):
        """Init planner/specification for a given spec string without environment specific data like start state"""
        goal_list = None
        if agent_goals is not None:
            goal_list = np.array(agent_goals)[:, :2].tolist()  # Reshape to list of goals
        if self.stl_solver in [STL_PY_NAME, DIFFUSION_NAME, GLOBAL_STL_PY_NAME]:
            # Team spec intercept (team_cover / team_mofn / team_redun / team_choiceseq3)
            if self._is_team_spec(spec):
                self._load_team_spec(spec, goal_list,
                                     allocation_strategy=self.allocation_strategy)
                self.ma_stl_spec = spec          # triggers existing scoring in _score_path
                self.ma_stl_exp_rob = True        # use exponential robustness for gradient-based planning
                self.logger.info(f"Loaded team spec: {spec} (alloc={self.allocation_strategy})")
                return
            if agent_goals is not None and spec[0] == 'm':
                # Different goals for each agent (use agent_goals)
                self.stl_forms = []
                for i in range(self.num_agents):
                    stl_form, key = self._load_diff_spec(spec=spec, goal_list=goal_list, agent_id=i, key=key)
                    self.stl_forms.append(stl_form)
                self.stl_form = self.stl_forms[0]  # For consistency
            else:
                # Same default goals for all agents
                self.stl_form, _key = self._load_diff_spec(spec=spec, goal_list=goal_list, key=key)
                self.stl_forms = [self.stl_form] * self.num_agents

            self.logger.info(f"Loaded spec: {spec}")

        else:
            raise NotImplementedError(f"MILP solver {self.stl_solver} not implemented")

    def _setup_planner(self, key=None):
        """Setup any MILP planner module"""
        if key is None:
            key = jax.random.PRNGKey(0)
        if self.stl_solver == STL_PY_NAME:
            self.planner = StlpySolver(space_dim=2)
        elif self.stl_solver == DIFFUSION_NAME:
            from gcbfplus.diffusion.planner.loader import DiffusionMAPlanner
            planner_config = dict(num_agents=self.num_agents,
                                  node_dim=self.node_dim,
                                  edge_dim=self.edge_dim,
                                  state_dim=self.env.state_dim,
                                  action_dim=self.action_dim,
                                  goal_dim=self.goal_dim,
                                  planner_config=self.params.copy(),
                                  filter_state=self.filter_state,
                                  rng=key,
                                  plan_length=self.plan_length,
                                  skip_guidance=self.skip_guidance)
            planner_config.update(self.planner_config)  # Update with wandb config
            self.planner = DiffusionMAPlanner(**planner_config)
        elif self.stl_solver == GLOBAL_STL_PY_NAME:
            self.planner = None  # Created on-the-fly in _init_plan_with_env
        else:
            raise NotImplementedError(f"MILP solver {self.stl_solver} not implemented")

    def process_finished_rollouts(self):
        """Returns function to run on rollout to get satisfaction rates"""

        # Change graphs to states
        filter_goal = jax.vmap(self.filter_state)

        def rollout2score(rollout):
            """Checks min score > 0"""
            return self.score_path(filter_goal(rollout.type_states_rollout(self.env.AGENT, self.num_agents)))[
                STL_INFO_KEYS[1]] > 0

        return rollout2score

    def process_finished_rollouts_info(self):
        """Returns function to run on rollout to get any finished rollout metrics. Separate function from satisfaction"""

        # Change graphs to states
        filter_goal = jax.vmap(self.filter_state)

        def rollout2score(rollout):
            return self.score_path(filter_goal(rollout.type_states_rollout(self.env.AGENT, self.num_agents)))

        return rollout2score

    @property
    def extra_run_info(self) -> Optional[str]:
        # If has attr ma_stl_form, then it is a multi-agent STL spec
        if hasattr(self, 'ma_stl_spec'):
            if len(super().extra_run_info) == 0:
                # No extra info
                extra_info = []
            else:
                extra_info = [super().extra_run_info]
            return ";".join(extra_info + [f"ma_stl_spec: {self.ma_stl_spec}", f"ma_stl_exp_rob: {self.ma_stl_exp_rob}"])
        return super().extra_run_info

    def _render_extra(self, ax=None, rollout=None, viz_opts=None, key=None):
        """Render extra STL information on the plot"""
        if viz_opts is None:
            viz_opts = {}
        goal_color = 'green'
        # To add any ellipses for goals
        # Sort to get consistent order and only unique predicates
        predicates = sorted(set(self.get_all_predicates()))

        if viz_opts.get('plot_plan_and_quit', False):
            plan2plot = rollout.Tp1_graph.current_plan[0]
            plan2plot = plan2plot.squeeze(0)
            plan2plot = plan2plot.transpose((1, 0, 2))
            ax.plot(*plan2plot.T, 'g-', alpha=0.5, label="Plan")
            # Plot circles at all waypoints with a small border
            ax.scatter(*plan2plot.T, s=100, edgecolor='gray', facecolor=goal_color, alpha=0.5,
                       linewidth=2, zorder=2)

        # Remove all predicates that have the same center (predicate.cent)
        def remove_duplicates(predicates):
            # Dedupe by centre: RectReachPredicate has no value-based __eq__/__hash__, so key on the printed centre.
            seen = set()
            return [x for x in predicates if f"{x.cent}" not in seen and not seen.add(f"{x.cent}")]

        predicates = remove_duplicates(predicates)

        shifted_cents = [x.cent - (x.size / 2) for x in predicates]
        pred_sizes = [x.size for x in predicates]

        if not viz_opts.get('async_planner', False) and viz_opts.get('plot_changed_goal', False):
            state_trajectory = rollout.Tp1_graph.type_states_rollout(MultiAgentEnv.AGENT, self.num_agents)
            # Only plot the change points for failed agents
            finish_metrics = self.process_finished_rollouts_info()(rollout.Tp1_graph)
            num_agents_to_plot = min(self.num_agents, viz_opts.get('num_agents_to_plot', self.num_agents))
            failed_agents = ~finish_metrics["finish_rate"]
            agents_to_plot = failed_agents.copy()
            if sum(agents_to_plot) < num_agents_to_plot:
                # flip some agents to plot
                remaining_agents = num_agents_to_plot - sum(failed_agents)
                success_agents = jnp.where(failed_agents == False)[0]
                # select random remaining agents from success agents
                random_remaining_agents = jax.random.choice(
                    key, success_agents, shape=(remaining_agents,), replace=False)
                agents_to_plot = agents_to_plot.at[random_remaining_agents].set(True)

            self.logger.debug(f"Agents to plot: {jnp.where(agents_to_plot)[0]}")
            state_trajectory = state_trajectory[:, agents_to_plot]
            ax.plot(state_trajectory[:, :, 0], state_trajectory[:, :, 1], c='k', linestyle='-', linewidth=1,
                    alpha=0.5)

            # Plot X markers at plan points
            # NxHx(goal_dim)
            n_plans = rollout.Tp1_graph.current_plan[0].squeeze(0).transpose((1, 0, 2))
            n_plans = n_plans[agents_to_plot,:]
            ax.plot(n_plans[:, :, 0].T, n_plans[:, :, 1].T, 'b-', alpha=0.2)
            # Also include X markers for each step
            ax.plot(n_plans[:, :, 0].T, n_plans[:, :, 1].T, 'bx', alpha=0.3)
        # Plot the path from a rollout
        if viz_opts.get('plot_debug', False):
            max_to_plot = 2
            # NxHx(goal_dim)
            n_plans = rollout.Tp1_graph.current_plan[0].squeeze(0).transpose((1, 0, 2))
            # Only plot unsuccessful plans
            if 'changed_goal' in rollout.T_info:  # async wrapper
                finish_metrics = self.process_finished_rollouts_info()(rollout.Tp1_graph, rollout.T_info['changed_goal'])
            else:
                finish_metrics = self.process_finished_rollouts_info()(rollout.Tp1_graph)
            failed_agents = ~finish_metrics["finish_rate"]
            n_plans = n_plans[failed_agents]

            # plot line from start to end for each of N agents
            ax.plot(n_plans[:, :, 0].T, n_plans[:, :, 1].T, 'k-', alpha=0.5)
            # Also include X markers for each step
            ax.plot(n_plans[:, :, 0].T, n_plans[:, :, 1].T, 'kx', alpha=0.4)

            # Show failed agent indices
            failed_agents_indices = np.where(failed_agents)[0]
            ax.text(0.5, 0.95, f"Failed agents: {failed_agents_indices}", ha='center', va='center', fontsize=16,
                    color='gray', alpha=0.9, transform=ax.transAxes)

            # Also plot spec
            for _i, stl_form in enumerate(self.stl_forms):
                if _i > max_to_plot:
                    break
                stl_form_text = stl_form._extract_repr(print_rich=True)
                # Plot one after the other
                ax.text(0.5, 0.90 - 0.05 * _i, stl_form_text, ha='center', va='center', fontsize=16, color='gray',
                        alpha=0.9,
                        transform=ax.transAxes)

        # Plot rectangles around predicates
        ep_cents = viz_opts.get('episode_goal_centers')  # [A, n_goals, 2] traced random layout
        if ep_cents is not None:
            # Traced random goals: the forms' baked predicate centers are stale (the fixed
            # grid), so draw this episode's layout instead — union of goal regions faint,
            # each highlighted agent's OWN goals bold with visit-order labels.
            ep_sizes = viz_opts.get('episode_goal_sizes')  # [A, n_goals, 2] or None
            named_preds = [p for p in predicates if getattr(p, 'name', -1) >= 0]
            default_size = named_preds[0].size if named_preds else np.array([1.0, 1.0])
            uniq = {}
            for a in range(ep_cents.shape[0]):
                for j in range(ep_cents.shape[1]):
                    c = ep_cents[a, j]
                    s = ep_sizes[a, j] if ep_sizes is not None else default_size
                    uniq[(round(float(c[0]), 3), round(float(c[1]), 3))] = (c, s)
            for c, s in uniq.values():
                ax.add_patch(plt.Rectangle(c - s / 2, s[0], s[1], edgecolor='darkgray',
                                           facecolor=to_rgba('gray', 0.08), linewidth=1.0, zorder=0))
            highlight_colors = ['crimson', 'royalblue', 'darkorange', 'purple']
            for hi, k in enumerate(viz_opts.get('highlight_agents', [])):
                if k >= ep_cents.shape[0]:
                    continue
                agent_form = self.stl_forms[k] if self.stl_forms is not None and k < len(self.stl_forms) else None
                names = sorted({p.name for p in (agent_form.get_all_predicates() if agent_form is not None
                                                 else named_preds) if getattr(p, 'name', -1) >= 0})
                color = highlight_colors[hi % len(highlight_colors)]
                for order, j in enumerate(names):
                    if j >= ep_cents.shape[1]:
                        continue
                    c = ep_cents[k, j]
                    s = ep_sizes[k, j] if ep_sizes is not None else default_size
                    ax.add_patch(plt.Rectangle(c - s / 2, s[0], s[1], edgecolor=color,
                                               facecolor=to_rgba(color, 0.25), linewidth=2.5, zorder=1))
                    ax.text(c[0], c[1], f"{order + 1}", ha='center', va='center', fontsize=32,
                            color=color, alpha=0.9, zorder=2, weight='bold')
        else:
            for i, (pred_size, cent) in enumerate(zip(pred_sizes, shifted_cents)):
                rect = plt.Rectangle(cent, pred_size[0], pred_size[1], edgecolor='gray', facecolor=to_rgba(goal_color, 0.4),
                                     linewidth=2)
                ax.add_patch(rect)
                center_x = rect.get_x() + rect.get_width() / 2
                center_y = rect.get_y() + rect.get_height() / 2

                pred_txt = string.ascii_uppercase[i]  # Use letters for predicates
                # Add a text box in the center of the circle
                ax.text(center_x, center_y, pred_txt, ha='center', va='center', fontsize=48, color='gray', alpha=0.9,
                        zorder=2)

        # Change the xlim, ylim from (0, self.area_size) to include the STL predicates
        ax.set_xlim(self._xy_min[0], self._xy_max[0])
        ax.set_ylim(self._xy_min[1], self._xy_max[1])

    def _render_extra_update(self, ax=None, rollout=None, viz_opts=None, kk=None, safe_text=None):
        """Render any dynamic STL information on the plot"""
        pass

    def render_video(self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None,
                     dpi: int = 100, key=None, **kwargs):
        """Render video with STL predicates"""
        self.env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi, render_extra=self._render_extra,
                              render_extra_update=self._render_extra_update, stl_form=self.stl_form,
                              stl_spec_name=self.spec_name, key=key, **kwargs)
