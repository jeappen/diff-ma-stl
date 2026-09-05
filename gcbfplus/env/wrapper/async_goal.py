"""Holds the AsyncWrapper class. This class is used to wrap a synchronous
planner to make it asynchronous. Each agent changes its goal only when the
previous goal is reached and not based on the global time."""

import os
from typing import Any
from typing import Tuple

import jax
import jax.numpy as jnp

from gcbfplus.utils.graph import GraphsTuple

os.environ["DIFF_STL_BACKEND"] = "jax"
import ds.stl
import ds.stl_jax
from gcbfplus.env import MultiAgentEnv
from gcbfplus.stl.utils import STL_INFO_KEYS, TRAINING_CONFIG, ENV_CONFIG
from gcbfplus.utils.typing import Action, Cost, Done, Info, Reward

ds.stl.HARDNESS = TRAINING_CONFIG['ds_params']['HARDNESS']
ds.stl_jax.HARDNESS = TRAINING_CONFIG['ds_params']['HARDNESS']  # Set hardness for easier backpropagation

from .wrapper import STLWrapper


class AsyncPlannerMixin:
    """Mixin for all wrappers with asynchronous planning b/w agents."""

    def process_finished_rollouts_info(self):
        """Returns function to run on rollout to get any finished rollout metrics. Separate function from satisfaction"""

        # Change graphs to states
        filter_goal = jax.vmap(self.filter_state)

        def rollout2score(rollout, changed_goals):
            """Score one rollout on the states recorded at goal changes, padded to spec_len samples per agent."""
            goal_space = filter_goal(rollout.type_states_rollout(self.env.AGENT, self.env.num_agents))

            # The formula needs exactly spec_len samples per agent, but an agent that reached fewer
            # waypoints has fewer goal-change states. Pad the change mask deterministically: every
            # goal_sample_interval AFTER the last change, then every interval BEFORE it, then every
            # step backwards/forwards from the last change until spec_len entries are set.
            goals_left = jnp.maximum(self.spec_len - changed_goals.sum(axis=0), 0)
            max_indices = changed_goals.shape[0] - jnp.argmax(changed_goals[::-1, :], axis=0) - 1
            for i in range(self.env.num_agents):
                # Fill with strided samples after the last change
                changed_goals[max_indices[i]::self.goal_sample_interval, i][::-1][:goals_left[i]] = 1

            # Fill with strided samples before the last change
            goals_left = jnp.maximum(self.spec_len - changed_goals.sum(axis=0), 0)
            for i in range(self.env.num_agents):
                changed_goals[max_indices[i]::-self.goal_sample_interval, i][::-1][:goals_left[i]] = 1

            # Last resort, fill from end and beginning for any more samples
            goals_left = jnp.maximum(self.spec_len - changed_goals.sum(axis=0), 0)
            for i in range(self.env.num_agents):
                changed_goals[max_indices[i]::-1, i][:goals_left[i]] = 1
            goals_left = jnp.maximum(self.spec_len - changed_goals.sum(axis=0), 0)
            for i in range(self.env.num_agents):
                changed_goals[:max_indices[i], i][:goals_left[i]] = 1

            # Filter the change points states with padding (keep the last state of goal_space)
            flattened_indices = jnp.where(changed_goals)
            row_indices = flattened_indices[0]
            col_indices = flattened_indices[1]

            # Extract change points and map to correct size output
            op_subsampled_row_indices = jnp.zeros_like(col_indices, dtype=jnp.int32)

            # Keep track of the last index seen for each column
            last_indices = jnp.full(col_indices.max() + 1, -1, dtype=jnp.int32)

            for i, col in enumerate(col_indices):
                last_indices = last_indices.at[col].set(last_indices[col] + 1)  # Increment index for this column
                op_subsampled_row_indices = op_subsampled_row_indices.at[i].set(last_indices[col])

            change_points = jnp.zeros((self.spec_len,) + goal_space.shape[1:])
            change_points = change_points.at[op_subsampled_row_indices, col_indices].set(
                goal_space[row_indices, col_indices])
            score_dict = self.score_path(change_points, changed_goals)
            if TRAINING_CONFIG['ds_params']['include_start_state']:
                # concat initial state (crucial for loop stlpy spec)
                # But what about trajectories that reach the goal state in the final step?
                change_points = jnp.concatenate([goal_space[:1], change_points], axis=0)
                score_dict2 = self.score_path(change_points, changed_goals)  # Alternate score with start state
                score_dict = self._merge_score_dict([score_dict, score_dict2])
            # Set last state as the last change point (since this may be missing)
            return score_dict

        return rollout2score

    def _merge_score_dict(self, score_dicts):
        """Merge the score dictionaries for each agent based on the highest mean finish rate.

        For team specs: also take the best task metrics across variants, since async
        change_point timing can cause false negatives in one variant but not the other.
        The per-agent finish_rate picks the best trajectory reconstruction; the task
        metrics should reflect the best CaTL+ evaluation across all reconstructions.
        """
        max_finish_rate = -1
        max_score_dict = None
        for score_dict in score_dicts:
            finish_rate = score_dict[STL_INFO_KEYS[3]].mean()
            if finish_rate > max_finish_rate:
                max_finish_rate = finish_rate
                max_score_dict = score_dict

        # For team specs: take the best task_rate and tasks_completed across variants.
        # Different trajectory reconstructions (with/without start state) can have different
        # CaTL+ timing alignment. If ANY variant shows the task as completed, it's completed.
        if max_score_dict is not None and 'tasks_completed' in max_score_dict:
            best_tasks = max_score_dict.get('tasks_completed', 0)
            best_task_rate = max_score_dict.get('task_rate', 0)
            best_per_task = max_score_dict.get('per_task_completed', {})
            for score_dict in score_dicts:
                if score_dict is max_score_dict:
                    continue
                tc = score_dict.get('tasks_completed', 0)
                if tc > best_tasks:
                    best_tasks = tc
                    best_task_rate = score_dict.get('task_rate', 0)
                    best_per_task = score_dict.get('per_task_completed', {})
                elif tc == best_tasks:
                    # Merge per_task: if either variant says a task is completed, it is
                    other_per_task = score_dict.get('per_task_completed', {})
                    for k, v in other_per_task.items():
                        if v and not best_per_task.get(k, False):
                            best_per_task[k] = True

            if best_tasks > max_score_dict.get('tasks_completed', 0):
                max_score_dict['tasks_completed'] = best_tasks
                max_score_dict['task_rate'] = best_task_rate
                max_score_dict['per_task_completed'] = best_per_task
                # Re-compute finish_rate override for team specs
                tasks_total = max_score_dict.get('tasks_total', 1)
                if 'team_spec_success_mode' in max_score_dict:
                    import jax.numpy as jnp
                    team_finish = jnp.ones_like(max_score_dict[STL_INFO_KEYS[3]]) * (best_tasks / max(tasks_total, 1))
                    max_score_dict[STL_INFO_KEYS[3]] = team_finish

        return max_score_dict

    def score_path(self, path, changed_goals=None):
        full_stl_score_dict, path_scores = super()._score_path(path, skip_subsample=True)
        # Add the changed goal information
        # Index of last entry in changed goals
        num_timesteps = changed_goals.shape[0]
        last_change = num_timesteps - jnp.argmax(changed_goals[::-1, :], axis=0)

        last_change_w_nan = jnp.where((jnp.array(path_scores) > 0)[:, 0], last_change, jnp.nan)
        # Team/CaTL+ specs: success is joint (safe_rate x task_rate), but the per-agent
        # gate above tests each agent against its own allocated eval form, so agents can
        # report a reach time on episodes where the TEAM spec failed (e.g. STLPY-Global
        # stay-in-place fallback trivially "reaching" waypoints). TtR is undefined
        # unless the joint spec is satisfied.
        if self.ma_stl_spec is not None and STL_INFO_KEYS[6] in full_stl_score_dict:
            last_change_w_nan = jnp.where(full_stl_score_dict[STL_INFO_KEYS[6]],
                                          last_change_w_nan, jnp.nan)
        full_stl_score_dict[STL_INFO_KEYS[4]] = last_change_w_nan

        return full_stl_score_dict

    def _increment_times(self, graph: GraphsTuple):
        """Increment times based on reaching goals"""

        def _check_goal_reached(graph_arg: GraphsTuple):
            """Only check goal reached if global time is a multiple of goal_sample_interval"""
            agent = graph_arg.type_states(type_idx=self.AGENT, n_type=self.num_agents)
            agent_position = self.filter_state(agent)
            goal = graph_arg.type_states(type_idx=self.GOAL, n_type=self.num_agents)
            goal_position = self.filter_state(goal)
            error = goal_position - agent_position
            reached_agents = jnp.linalg.norm(error, axis=-1, keepdims=True) < self.async_reach_radius
            goal_outside = jnp.any(
                (goal_position > jnp.array(self.env._xy_max)) | (goal_position < jnp.array(self.env._xy_min)),
                axis=-1, keepdims=True)
            # A waypoint outside [_xy_min, _xy_max] (planner left the map) counts as reached so the
            # agent skips it instead of stalling on an unreachable goal.
            reached_agents = reached_agents | goal_outside
            reached_agents = reached_agents.squeeze(1)
            # Advance rule (evaluated every goal_sample_interval env steps): an agent's plan index moves
            # forward by one when it is within async_reach_radius of its current waypoint, but never ahead
            # of the sync clock (global_time // goal_sample_interval) and never past spec_len.
            new_time = graph_arg.current_time + reached_agents * self.time_increment
            new_global_time = graph_arg.global_time + self.time_increment
            new_time = jnp.minimum(new_time, new_global_time // self.goal_sample_interval)  # Do not exceed global time
            new_time = jnp.minimum(new_time, self.spec_len)  # Do not exceed spec length to capture completion
            new_time = new_time.astype(jnp.int32)
            return graph_arg._replace(current_time=new_time, global_time=new_global_time)

        return jax.lax.cond(graph.global_time % self.goal_sample_interval == 0, _check_goal_reached,
                            lambda graph_arg: graph_arg._replace(
                                global_time=graph_arg.global_time + self.time_increment), graph)

    @property
    def args_of_interest_all(self):
        """Return the arguments of interest for the planner. This is used to set the arguments for the planner."""
        return ['async_reach_radius'] + super().args_of_interest_all

    def step(
            self, graph: GraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[Any, Reward, Cost, Done, Info]:
        """Take a step in the environment and log when goals are changed."""

        old_time = graph.current_time
        retval = super().step(graph, action, get_eval_info)
        changed_goal = retval[0].current_time - old_time
        info_dict = retval[-1]
        info_dict['changed_goal'] = changed_goal
        # For Async planner, if current time >= spec_len then last goal reached.
        if ENV_CONFIG.get('vanish_on_end', False):
            # If vanish_on_end is True, then shift agents out of map when done
            agent_done = (graph.current_time >= self.spec_len)
            # Shift each agent based on agent id to a final resting place out of map using range(1, num_agents+1)
            x_displacement = jnp.arange(2, self.num_agents + 2) * self.env.area_size * agent_done
            # Shift y by a fixed amount (e.g., 1) to move agents out of the map
            y_displacement = jnp.ones_like(x_displacement) * self.env.area_size
            # Combine x and y displacement
            total_displacement = jnp.zeros_like(retval[0].type_states(MultiAgentEnv.AGENT, self.num_agents))
            total_displacement = total_displacement.at[:, 0].set(x_displacement)
            total_displacement = total_displacement.at[:, 1].set(y_displacement)
            new_states = retval[0].type_states(MultiAgentEnv.AGENT, self.num_agents) + total_displacement
            retval = (self._set_new_state(retval[0], new_states),) + retval[1:]

        done = (graph.current_time >= self.spec_len).all() | retval[3]
        return retval[:3] + (done, info_dict)

    def _set_new_state(self, graph_arg: GraphsTuple, new_state):
        """Replace the agent states (used by vanish_on_end) and rebuild the graph."""
        new_goal_envstate = self.env.EnvState(new_state, graph_arg.env_states.goal,
                                              graph_arg.env_states.obstacle)
        # Need to calculate new goal features
        return self.get_graph(new_goal_envstate)._replace(current_time=graph_arg.current_time,
                                                          current_plan=graph_arg.current_plan,
                                                          global_time=graph_arg.global_time,
                                                          history=graph_arg.history, aux_nodes=graph_arg.aux_nodes)

    def _score_set_of_histories(self, histories, subsample=None):
        """Score a set of histories based on the STL spec"""
        # Do not subsample during eval for async planner since histories already subsampled
        return super()._score_set_of_histories(histories, subsample=False)

class AsyncSTLWrapper(AsyncPlannerMixin, STLWrapper):
    """Allows Asynchronous plan b/w agents. Changes the goal only when reached and not based on a global clock."""

    def __init__(self, *args, async_reach_radius=0.1, **kwargs):
        """Initialize the environment.

        :param async_reach_radius: distance to the current waypoint below which the agent advances
        """

        super().__init__(*args, **kwargs)
        self.async_reach_radius = async_reach_radius


ASYNC_WRAPPER_LIST = [AsyncSTLWrapper]
