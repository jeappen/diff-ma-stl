"""Wrappers/Mixins for any environment."""
import jax
import jax.numpy as jnp
import logging
import numpy as np
import tqdm
from jax import lax
from typing import Any, Optional
from typing import Callable

from gcbfplus.env import MultiAgentEnv, SingleIntegrator
from gcbfplus.env.base import RolloutResult, StepResult
from gcbfplus.stl.utils import PLANNER_CONFIG
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.typing import Action, Array
from gcbfplus.utils.typing import PRNGKey
from gcbfplus.utils.utils import jax2np, jax_jit_np, tree_stack, tree_concat_at_front, turn_dict_into_comment_str


class BaseWrapper(object):
    """Base wrapper for all environments similar to OpenAI Gym."""

    def __init__(self, env: MultiAgentEnv, **kwargs):
        """Initialize the environment."""
        self.env = env
        self.plan_info = {}

    def reset(self, key: Array, plan_settings=None, **kwargs):
        """Reset the environment."""
        return self.env.reset(key)

    @property
    def extra_run_info(self) -> Optional[str]:
        return None

    def reset_np(self, key: Array, **kwargs) -> GraphsTuple:
        """Reset, but without the constraint that it has to be jittable."""
        return self.reset(key, **kwargs)

    def step(self, graph: Any, action: Action, get_eval_info: bool = False) -> StepResult:
        """Take a step in the environment.

            :param graph: current graph
            :type graph: EnvGraphsTuple"""
        return self.env.step(graph, action)

    def filter_state(self, state):
        """Filter the state to only contain the goal space projection"""
        # Goal space = the first goal_dim state coordinates (x, y for every shipped env).
        return state[:, :self.env.goal_dim]

    def render(self, *args, **kwargs):
        """Render the environment."""
        return self.env.render(*args, **kwargs)

    @property
    def extra_config(self):
        """Extra configuration for the environment from the wrapper."""
        return {}

    @property
    def config(self):
        """Get the configuration of the environment."""
        return self.env.config | self.extra_config

    def close(self):
        """Close the environment."""
        return self.env.close()

    def seed(self, seed=None):
        """Seed the environment."""
        return self.env.seed(seed)

    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self.env.unwrapped

    @property
    def num_agents(self):
        return self.env.num_agents

    def EnvState(self, agent, goals, obstacle):
        """Format the goals appropriately before passing to EnvState"""
        if not isinstance(self.env, SingleIntegrator):
            # For DubinsCar, DoubleIntegrator
            # Non-SingleIntegrator envs carry a 4-D goal (x, y, 0, 0). 3-D envs are not supported by the STL wrapper.
            goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)
        return self.env.EnvState(agent, goals, obstacle)

    def rollout_fn(self, policy: Callable, rollout_length: int = None, preprocess_graph: Callable = lambda x: x) -> \
            Callable[[PRNGKey], RolloutResult]:
        """Redefining to use wrapper functions."""
        rollout_length = rollout_length or self.max_episode_steps

        def body(graph, _):
            graph = preprocess_graph(graph)
            action = policy(graph)
            graph_new, reward, cost, done, info = self.step(graph, action, get_eval_info=True)
            return graph_new, (graph_new, action, reward, cost, done, info)

        def fn(key: PRNGKey) -> RolloutResult:
            graph0 = self.reset(key)
            graph_final, (T_graph, T_action, T_reward, T_cost, T_done, T_info) = lax.scan(body, graph0, None,
                                                                                          length=rollout_length)
            Tp1_graph = tree_concat_at_front(graph0, T_graph, axis=0)

            return RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)

        return fn

    def rollout_fn_jitstep(self, policy: Callable, rollout_length: int = None, noedge: bool = False,
                           nograph: bool = False, preprocess_graph: Callable = lambda x: x,
                           end_when_done: bool = False,
                           reset_args: dict = None):
        rollout_length = rollout_length or self.max_episode_steps
        if reset_args is None:
            reset_args = {}

        def body(graph: GraphsTuple, _):
            graph = preprocess_graph(graph)  # Important to change plan
            action = policy(graph)
            graph_new, reward, cost, done, info = self.step(graph, action, get_eval_info=True)
            return graph_new, (graph_new, action, reward, cost, done, info)

        jit_body = jax.jit(body)

        is_unsafe_fn = jax_jit_np(self.collision_mask)
        is_finish_fn = jax_jit_np(self.finish_mask)

        def fn(key: PRNGKey) -> [RolloutResult, Array, Array]:
            graph0 = self.reset_np(key, **reset_args)
            graph = graph0
            T_output = []
            is_unsafes = []
            is_finishes = []

            is_unsafes.append(is_unsafe_fn(graph0))
            is_finishes.append(is_finish_fn(graph0))
            graph0 = jax2np(graph0)
            plan_info = self.plan_info

            for kk in tqdm.trange(rollout_length, ncols=80):
                graph, output = jit_body(graph, None)

                is_unsafes.append(is_unsafe_fn(graph))
                is_finishes.append(is_finish_fn(graph))

                output = jax2np(output)
                if noedge:
                    output = (output[0].without_edge(), *output[1:])
                if nograph:
                    output = (None, *output[1:])
                T_output.append(output)

            # Concatenate everything together.
            T_graph = [o[0] for o in T_output]
            if noedge:
                T_graph = [graph0.without_edge()] + T_graph
            else:
                T_graph = [graph0] + T_graph
            del graph0
            T_action = [o[1] for o in T_output]
            T_reward = [o[2] for o in T_output]
            T_cost = [o[3] for o in T_output]
            T_done = [o[4] for o in T_output]
            T_info = [o[5] for o in T_output]
            del T_output

            if nograph:
                T_graph = None
            else:
                T_graph = tree_stack(T_graph)
            T_action = tree_stack(T_action)
            T_reward = tree_stack(T_reward)
            T_cost = tree_stack(T_cost)
            T_done = tree_stack(T_done)
            T_info = tree_stack(T_info)

            Tp1_graph = T_graph

            rollout_result = jax2np(RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info))
            if end_when_done:
                # Don't consider safety and finish after the episode is done
                done_time = T_done.argmax()
                if done_time != 0:
                    # Remove all the extra steps. If done_time is 0, then the episode is not done
                    is_unsafes = is_unsafes[:done_time + 1]
                    is_finishes = is_finishes[:done_time + 1]

            return rollout_result, np.stack(is_unsafes, axis=0), np.stack(is_finishes, axis=0), plan_info

        return fn

    def calc_safety_success_finish(self, is_unsafe, is_finish, rollout=None, ignore_on_finish=False,
                                    team_info=None):
        """Calculate safety and success rates for the set of rollouts.

        :param is_unsafe: Array of shape (T, N) where T is the number of timesteps and N is the number of agents
        :param is_finish: Array of shape (T, N) where T is the number of timesteps and N is the number of agents
        :param rollout: RolloutResult object
        :param ignore_on_finish: Ignore safety and success rates after the episode is done (for that agent)
        :type ignore_on_finish: bool
        :param team_info: Optional dict with 'per_task_completed', 'task_to_agents', 'tasks_total'
            for computing task_success_rate
        :return: safety_rate, finish_rate, success_rate, final_is_unsafe, task_success_rate
        """

        finish_rate = is_finish.max(axis=0).mean()
        final_is_unsafe = is_unsafe
        if ignore_on_finish:
            # Ignore safety and success rates after the episode is done (for that agent)
            done_time = rollout.Tp1_graph.current_time.argmax(axis=0)
            finished_spec_at_time = done_time * is_finish.max(axis=0)
            # If finished spec at time is 0, then the episode is not done and the value should be len(is_unsafe)
            done_time = jnp.where(finished_spec_at_time == 0, len(is_unsafe), done_time)
            # Now only consider the safety and success rates till done_time for each agent
            agent_unsafe = np.zeros(self.env.num_agents)
            for i in range(self.env.num_agents):
                agent_unsafe[i] = is_unsafe[:done_time[i] + 1, i].max()

            safe_rate = 1 - agent_unsafe.mean()
            success_rate = ((1 - agent_unsafe) * is_finish.max(axis=0)).mean()

            final_is_unsafe = agent_unsafe.reshape((1, -1)).astype(bool)

            self.logger.info(f"Truncated safety and success rates till done_time: {done_time}")
            if any(finished_spec_at_time == 0):
                self.logger.info(f"Unfinished agents: {jnp.where(finished_spec_at_time == 0)}")
            self.logger.info(f"Safe rate: {safe_rate}, Success rate: {success_rate}, Finish rate: {finish_rate}")
        else:
            # Classic safety and success rates
            safe_rate = 1 - is_unsafe.max(axis=0).mean()
            success_rate = ((1 - is_unsafe.max(axis=0)) * is_finish.max(axis=0)).mean()

        # Task success rate for team specs
        task_success_rate = None
        if team_info is not None:
            per_task_completed = team_info.get('per_task_completed', {})
            task_to_agents = team_info.get('task_to_agents', {})
            tasks_total = team_info.get('tasks_total', 1)
            # Per-agent safety: max collision over trajectory (or truncated if ignore_on_finish)
            agent_is_unsafe = np.array(final_is_unsafe).flatten()
            safely_completed = 0
            for task_name, completed in per_task_completed.items():
                if completed:
                    # Extract task_id from name like "team_task_0"
                    try:
                        task_id = int(task_name.split('_')[-1])
                    except (ValueError, IndexError):
                        task_id = task_name
                    agents = task_to_agents.get(task_id, [])
                    if len(agents) > 0 and all(not agent_is_unsafe[a] for a in agents):
                        safely_completed += 1
            task_success_rate = safely_completed / max(tasks_total, 1)

        return safe_rate, finish_rate, success_rate, final_is_unsafe, task_success_rate

    def extra_summary_metrics(self, summary_metrics=None, finish_infos=None) -> tuple[list, dict]:
        """Extra summary metrics and updated finish_infos for the environment that uses the rollout results.

        :param summary_metrics: Dictionary of summary metrics like finish_rate, safety_rate, success_rate
        :param finish_infos: List of finish_infos for each agent like satisfaction, time_to_finish
        """
        if finish_infos is None:
            finish_infos = []
        if summary_metrics is None:
            summary_metrics = {}
        return finish_infos, summary_metrics

    def __getattr__(self, name):
        """Get any other attribute."""
        return getattr(self.env, name)


class PlannerWrapper(BaseWrapper):
    """Wrapper for the environments with a Planner."""

    def __init__(self, *args, key=None, device=None, max_step=None, goal_sample_interval=1, plan_length=None, **kwargs):
        """Initialize the planner-wrapped environment."""
        super().__init__(*args, **kwargs)
        self.init_time = 0
        self.path = None

        self.logger = logging.getLogger(__name__)
        self.max_step = plan_length if max_step is None else max_step
        self.max_episode_steps = self.max_step  # Backward compatibility

        self._setup_planner(key=key)

        self.time_increment = 1
        self.goal_sample_interval = goal_sample_interval
        self.plan_length = 10 if plan_length is None else plan_length
        self.plan = []  # Used by MILP planner
        self.plan_info = {}  # Any planner information

        self.change_goal_immediately = PLANNER_CONFIG[
            'change_goal_immediately']  # Change goals immediately or after the goal_sample_interval

        self.device = 'cpu' if device is None else device

    @property
    def args_of_interest(self) -> list:
        """List of arguments of interest to log."""
        return ["goal_sample_interval", "plan_length", "max_step"]

    @property
    def args_of_interest_all(self) -> list:
        """List of arguments of interest to log including all (not needed for each run)."""
        return self.args_of_interest + ["change_goal_immediately"]

    @property
    def extra_run_info(self) -> Optional[str]:
        """Extra run info for the environment."""
        return turn_dict_into_comment_str(self.extra_run_info_dict, self.args_of_interest)

    @property
    def extra_run_info_dict(self) -> Optional[dict]:
        """Extra run info for the environment."""
        return self.args_of_interest_dict(all_args=False)

    @property
    def extra_run_info_full_dict(self) -> Optional[dict]:
        """Extra run info for the environment for wandb and non-text logs"""
        return self.args_of_interest_dict(all_args=True)

    def args_of_interest_dict(self, all_args: bool = False) -> dict:
        """Dictionary of arguments of interest to log."""
        if all_args:
            args_of_interest = self.args_of_interest_all
        else:
            args_of_interest = self.args_of_interest
        parts = {}
        for key in args_of_interest:
            # Only include the argument if it has a value (e.g., not None)
            value = getattr(self, key)
            if value is not None:
                parts[key] = value
        return parts

    def _increment_times(self, graph: GraphsTuple):
        """Increment the current time for each agent"""
        new_global_time = graph.global_time + self.time_increment
        new_time = (new_global_time // self.goal_sample_interval).astype(jnp.int32)
        return graph._replace(current_time=jnp.tile(new_time, self.env.num_agents), global_time=new_global_time)

    def change_goals(self, graph: GraphsTuple, a_all=None, flat_obs=None, init_path=None, o=None, threshold=None,
                     # Not used
                     **kwargs):
        """Set each agent's goal node from the stored plan (any planner) at its current plan index."""

        def get_new_goal(graph_arg):
            # current_plan: [1, plan_len, N, goal_dim]; current_time[i] is agent i's plan index.
            # Select current_plan[0, min(t_i, plan_len-1), i] per agent and rebuild the goal node features.
            batch_ind = 0

            # Select different goals for different agents based on time
            selected_plan = graph_arg.current_plan[batch_ind]
            reshaped_times = jnp.expand_dims(graph_arg.current_time, axis=tuple(range(1, selected_plan.ndim)))
            reshaped_times = jnp.minimum(reshaped_times, selected_plan.shape[0] - 1)  # Do not exceed plan length
            # Transpose to shape (Num_agents, Num_goals, Goal_dim) for easy selection of goals
            new_goals = jnp.take_along_axis(selected_plan.transpose(1, 0, 2), reshaped_times, axis=1).squeeze(1)

            new_goal_envstate = self.EnvState(graph_arg.env_states.agent, new_goals, graph_arg.env_states.obstacle)
            # Need to calculate new goal features
            return self.env.get_graph(new_goal_envstate)._replace(current_time=graph_arg.current_time,
                                                                  current_plan=graph_arg.current_plan,
                                                                  global_time=graph_arg.global_time)

        if self.change_goal_immediately:
            graph = get_new_goal(graph)
        else:
            goal_change = (graph.global_time % self.goal_sample_interval == 0)
            graph = jax.lax.cond(goal_change, get_new_goal, lambda graph_: graph_, graph)

        graph = self._increment_times(graph)  # Increment time after changing goals
        return graph

    def _setup_planner(self, key=None):
        """Setup the planner for the environment."""
        pass

    def forward_graph(self, graph: GraphsTuple, action, **kwargs):
        if 'current_time' in graph:
            graph = self._increment_times(graph)
        # Careful not to increment twice since env.forward_graph() calls env.forward()
        graph_next = self.env.forward_graph(graph, action, **kwargs)
        if 'current_time' in graph:
            graph_next = graph_next._replace(current_time=graph.current_time)
        return graph_next
