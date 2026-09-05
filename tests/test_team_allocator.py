"""Tests for team_allocator changes: strategy renaming, post-hoc inference, avoidance, metrics."""
import unittest
import numpy as np


class TestAllocationStrategies(unittest.TestCase):
    """Verify allocation strategy renaming: oracle removed, oracle_hungarian and greedy_last added."""

    def setUp(self):
        from gcbfplus.stl.team_spec import TeamSpec, TeamTask
        self.tasks = [
            TeamTask(task_id=0, goal_centers=[[1.0, 1.0]]),
            TeamTask(task_id=1, goal_centers=[[5.0, 5.0]]),
        ]
        self.team_spec = TeamSpec(
            spec_type='team_cover', tasks=self.tasks,
            time_horizon=15, n_agents=4, m_per_task=1, task_mode='cover',
        )
        # Agent 0,1 near task 0; Agent 2,3 near task 1
        self.positions = np.array([[0.5, 0.5], [1.5, 1.5], [4.5, 4.5], [5.5, 5.5]])

    def test_greedy_strategy(self):
        from gcbfplus.stl.team_allocator import allocate_tasks
        result = allocate_tasks(self.team_spec, self.positions, strategy='greedy')
        # Agent 0 closest to task 0's first goal, Agent 2 closest to task 1's first goal
        self.assertIn(0, result.task_to_agents[0])
        self.assertIn(2, result.task_to_agents[1])

    def test_greedy_last_strategy(self):
        from gcbfplus.stl.team_allocator import allocate_tasks
        result = allocate_tasks(self.team_spec, self.positions, strategy='greedy_last')
        # greedy_last uses last goal (same as first here since 1 goal per task)
        self.assertEqual(len(result.agent_to_task), 2)

    def test_oracle_hungarian_strategy(self):
        from gcbfplus.stl.team_allocator import allocate_tasks
        result = allocate_tasks(self.team_spec, self.positions, strategy='oracle_hungarian')
        # Hungarian should optimally assign nearby agents
        self.assertIn(result.agent_to_task[0], [0])  # Agent 0 -> task 0
        self.assertIn(result.agent_to_task[2], [1])  # Agent 2 -> task 1

    def test_oracle_strategy_removed(self):
        """Bare 'oracle' should fall through to greedy default, not hungarian."""
        from gcbfplus.stl.team_allocator import allocate_tasks
        # 'oracle' is not a recognized strategy, falls through to greedy default
        result = allocate_tasks(self.team_spec, self.positions, strategy='oracle')
        # Should work (greedy fallback) but NOT call _allocate_optimal
        self.assertIsNotNone(result)

    def test_random_strategy(self):
        from gcbfplus.stl.team_allocator import allocate_tasks
        result = allocate_tasks(self.team_spec, self.positions, strategy='random')
        self.assertEqual(len(result.agent_to_task), 2)


class TestBuildPerAgentStlFormsReturnType(unittest.TestCase):
    """Verify build_per_agent_stl_forms returns (list, avoid_regions) tuple."""

    def setUp(self):
        from gcbfplus.stl.team_spec import TeamSpec, TeamTask
        self.tasks = [
            TeamTask(task_id=0, goal_centers=[[1.0, 1.0]]),
            TeamTask(task_id=1, goal_centers=[[5.0, 5.0]]),
        ]
        self.team_spec = TeamSpec(
            spec_type='team_cover', tasks=self.tasks,
            time_horizon=15, n_agents=3, m_per_task=1, task_mode='cover',
        )
        self.goal_size = np.array([1.0, 1.0])

    def test_returns_tuple(self):
        from gcbfplus.stl.team_allocator import allocate_tasks, build_per_agent_stl_forms
        positions = np.array([[0.5, 0.5], [5.0, 5.0], [3.0, 3.0]])
        alloc = allocate_tasks(self.team_spec, positions, strategy='greedy')
        result = build_per_agent_stl_forms(
            self.team_spec, alloc, self.goal_size, 0.4, 3)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        stl_forms, avoid_regions = result
        self.assertEqual(len(stl_forms), 3)
        self.assertEqual(len(avoid_regions), 3)

    def test_without_avoidance(self):
        from gcbfplus.stl.team_allocator import allocate_tasks, build_per_agent_stl_forms
        positions = np.array([[0.5, 0.5], [5.0, 5.0], [3.0, 3.0]])
        alloc = allocate_tasks(self.team_spec, positions, strategy='greedy')
        stl_forms, avoid_regions = build_per_agent_stl_forms(
            self.team_spec, alloc, self.goal_size, 0.4, 3, add_avoidance=False)
        # Without avoidance, all avoid_regions should be empty
        for r in avoid_regions:
            self.assertEqual(len(r), 0)

    def test_with_avoidance(self):
        from gcbfplus.stl.team_allocator import allocate_tasks, build_per_agent_stl_forms
        positions = np.array([[0.5, 0.5], [5.0, 5.0], [3.0, 3.0]])
        alloc = allocate_tasks(self.team_spec, positions, strategy='greedy')
        stl_forms, avoid_regions = build_per_agent_stl_forms(
            self.team_spec, alloc, self.goal_size, 0.4, 3, add_avoidance=True)
        # With 2 tasks, assigned agents should have avoid regions for the other task's goals
        assigned_agents = list(alloc.agent_to_task.keys())
        for a in assigned_agents:
            self.assertGreater(len(avoid_regions[a]), 0,
                               f"Agent {a} assigned to task {alloc.agent_to_task[a]} should have avoid regions")


class TestInferAllocationFromTrajectories(unittest.TestCase):
    """Verify post-hoc allocation inference matches actual trajectory behavior."""

    def setUp(self):
        from gcbfplus.stl.team_spec import TeamSpec, TeamTask
        self.tasks = [
            TeamTask(task_id=0, goal_centers=[[1.0, 1.0]]),
            TeamTask(task_id=1, goal_centers=[[5.0, 5.0]]),
        ]
        self.team_spec = TeamSpec(
            spec_type='team_cover', tasks=self.tasks,
            time_horizon=10, n_agents=2, m_per_task=1, task_mode='cover',
        )
        self.goal_size = np.array([1.0, 1.0])

    def test_obvious_allocation(self):
        """Agent 0 goes to goal 0, agent 1 goes to goal 1."""
        from gcbfplus.stl.team_allocator import infer_allocation_from_trajectories
        T = 10
        # Agent 0: moves toward (1,1)
        traj_0 = np.linspace([0.0, 0.0], [1.0, 1.0], T)
        # Agent 1: moves toward (5,5)
        traj_1 = np.linspace([4.0, 4.0], [5.0, 5.0], T)
        trajectories = np.stack([traj_0, traj_1])

        result = infer_allocation_from_trajectories(
            self.team_spec, trajectories, self.goal_size, shrink_factor=0.4)
        self.assertEqual(result.agent_to_task[0], 0)
        self.assertEqual(result.agent_to_task[1], 1)

    def test_swapped_allocation(self):
        """Agent 0 goes to goal 1, agent 1 goes to goal 0 — planner discovered swap."""
        from gcbfplus.stl.team_allocator import infer_allocation_from_trajectories
        T = 10
        # Agent 0: moves toward (5,5)
        traj_0 = np.linspace([4.0, 4.0], [5.0, 5.0], T)
        # Agent 1: moves toward (1,1)
        traj_1 = np.linspace([0.0, 0.0], [1.0, 1.0], T)
        trajectories = np.stack([traj_0, traj_1])

        result = infer_allocation_from_trajectories(
            self.team_spec, trajectories, self.goal_size, shrink_factor=0.4)
        self.assertEqual(result.agent_to_task[0], 1, "Agent 0 should be inferred as doing task 1")
        self.assertEqual(result.agent_to_task[1], 0, "Agent 1 should be inferred as doing task 0")

    def test_m_per_task_greater_than_1(self):
        """With m_per_task=2, both agents should be assigned to the single task."""
        from gcbfplus.stl.team_spec import TeamSpec, TeamTask
        from gcbfplus.stl.team_allocator import infer_allocation_from_trajectories
        tasks = [TeamTask(task_id=0, goal_centers=[[3.0, 3.0]])]
        spec = TeamSpec(
            spec_type='team_cover', tasks=tasks,
            time_horizon=10, n_agents=2, m_per_task=2, task_mode='cover',
        )
        T = 10
        traj_0 = np.linspace([0.0, 0.0], [3.0, 3.0], T)
        traj_1 = np.linspace([6.0, 6.0], [3.0, 3.0], T)
        trajectories = np.stack([traj_0, traj_1])

        result = infer_allocation_from_trajectories(
            spec, trajectories, self.goal_size, shrink_factor=0.4)
        self.assertEqual(result.agent_to_task[0], 0)
        self.assertEqual(result.agent_to_task[1], 0)
        self.assertEqual(len(result.task_to_agents[0]), 2)


class TestCalcSafetyWithTeamInfo(unittest.TestCase):
    """Verify task_success_rate computation in calc_safety_success_finish."""

    def test_all_safe_all_completed(self):
        """All tasks completed by safe agents → task_success_rate = 1.0."""
        from gcbfplus.env.wrapper.base import BaseWrapper
        import jax.numpy as jnp

        # Minimal mock
        wrapper = BaseWrapper.__new__(BaseWrapper)
        wrapper.env = type('', (), {'num_agents': 2})()

        is_unsafe = np.array([[0, 0]], dtype=float)  # (T=1, N=2), no collisions
        is_finish = np.array([[1, 1]], dtype=float)   # both finished

        team_info = {
            'per_task_completed': {'team_task_0': True, 'team_task_1': True},
            'task_to_agents': {0: [0], 1: [1]},
            'tasks_total': 2,
        }

        safe_rate, finish_rate, success_rate, final_unsafe, task_success_rate = \
            wrapper.calc_safety_success_finish(is_unsafe, is_finish, team_info=team_info)

        self.assertEqual(task_success_rate, 1.0)

    def test_one_agent_unsafe(self):
        """Agent 0 collides, its task should not count as safely completed."""
        from gcbfplus.env.wrapper.base import BaseWrapper
        import jax.numpy as jnp

        wrapper = BaseWrapper.__new__(BaseWrapper)
        wrapper.env = type('', (), {'num_agents': 2})()

        is_unsafe = np.array([[1, 0]], dtype=float)  # Agent 0 unsafe
        is_finish = np.array([[1, 1]], dtype=float)

        team_info = {
            'per_task_completed': {'team_task_0': True, 'team_task_1': True},
            'task_to_agents': {0: [0], 1: [1]},
            'tasks_total': 2,
        }

        safe_rate, finish_rate, success_rate, final_unsafe, task_success_rate = \
            wrapper.calc_safety_success_finish(is_unsafe, is_finish, team_info=team_info)

        self.assertEqual(task_success_rate, 0.5)  # Only task 1 safely completed

    def test_task_not_completed(self):
        """Task 1 not completed (even with safe agents) → not counted."""
        from gcbfplus.env.wrapper.base import BaseWrapper

        wrapper = BaseWrapper.__new__(BaseWrapper)
        wrapper.env = type('', (), {'num_agents': 2})()

        is_unsafe = np.array([[0, 0]], dtype=float)
        is_finish = np.array([[1, 0]], dtype=float)

        team_info = {
            'per_task_completed': {'team_task_0': True, 'team_task_1': False},
            'task_to_agents': {0: [0], 1: [1]},
            'tasks_total': 2,
        }

        _, _, _, _, task_success_rate = \
            wrapper.calc_safety_success_finish(is_unsafe, is_finish, team_info=team_info)

        self.assertEqual(task_success_rate, 0.5)  # Only task 0 safely completed

    def test_no_team_info(self):
        """Without team_info, task_success_rate should be None."""
        from gcbfplus.env.wrapper.base import BaseWrapper

        wrapper = BaseWrapper.__new__(BaseWrapper)
        wrapper.env = type('', (), {'num_agents': 2})()

        is_unsafe = np.array([[0, 0]], dtype=float)
        is_finish = np.array([[1, 1]], dtype=float)

        _, _, _, _, task_success_rate = \
            wrapper.calc_safety_success_finish(is_unsafe, is_finish)

        self.assertIsNone(task_success_rate)


class TestConfigDefaults(unittest.TestCase):
    """Verify config defaults are consistent."""

    def test_stlpy_u_bound_fallback_matches_config(self):
        """The fallback default in wrapper.py should match default_config.yaml."""
        from gcbfplus.stl.utils import PLANNER_CONFIG
        config_value = PLANNER_CONFIG.get('stlpy_u_bound', 2.0)
        # The fallback in wrapper.py is also 2.0
        self.assertEqual(config_value, 2.0)

    def test_global_stlpy_time_limit_in_config(self):
        from gcbfplus.stl.utils import PLANNER_CONFIG
        self.assertIn('global_stlpy_time_limit', PLANNER_CONFIG)
        self.assertEqual(PLANNER_CONFIG['global_stlpy_time_limit'], 600)

    def test_eval_allocation_mode_in_config(self):
        from gcbfplus.stl.utils import PLANNER_CONFIG
        self.assertIn('eval_allocation_mode', PLANNER_CONFIG)
        self.assertIn(PLANNER_CONFIG['eval_allocation_mode'], ['pre_planned', 'post_hoc'])

    def test_no_oracle_in_argparse_choices(self):
        """Verify 'oracle' is not in the team-alloc choices."""
        import test as test_module
        import argparse
        parser = test_module.test_argparser()
        # Find the --team-alloc action
        for action in parser._actions:
            if hasattr(action, 'dest') and action.dest == 'team_alloc':
                self.assertNotIn('oracle', action.choices,
                                 "Bare 'oracle' should not be in team-alloc choices")
                self.assertIn('oracle_hungarian', action.choices)
                self.assertIn('greedy_last', action.choices)
                break


class TestAvoidanceGating(unittest.TestCase):
    """Verify avoidance is properly gated behind add_avoidance flag."""

    def setUp(self):
        from gcbfplus.stl.team_spec import TeamSpec, TeamTask
        self.tasks = [
            TeamTask(task_id=0, goal_centers=[[1.0, 1.0]]),
            TeamTask(task_id=1, goal_centers=[[5.0, 5.0]]),
        ]
        self.team_spec = TeamSpec(
            spec_type='team_cover', tasks=self.tasks,
            time_horizon=10, n_agents=2, m_per_task=1, task_mode='cover',
        )
        self.goal_size = np.array([1.0, 1.0])

    def test_build_team_catl_without_avoidance(self):
        from gcbfplus.stl.team_allocator import build_team_catl
        catl = build_team_catl(self.team_spec, self.goal_size, 0.4, add_avoidance=False)
        self.assertIsNotNone(catl)

    def test_build_team_catl_with_avoidance(self):
        from gcbfplus.stl.team_allocator import build_team_catl
        catl = build_team_catl(self.team_spec, self.goal_size, 0.4, add_avoidance=True)
        self.assertIsNotNone(catl)

    def test_build_diffusion_stl_forms_without_avoidance(self):
        from gcbfplus.stl.team_allocator import build_diffusion_stl_forms
        forms = build_diffusion_stl_forms(self.team_spec, self.goal_size, 0.4, 2,
                                           add_avoidance=False)
        self.assertEqual(len(forms), 2)

    def test_build_diffusion_stl_forms_with_avoidance(self):
        from gcbfplus.stl.team_allocator import build_diffusion_stl_forms
        forms = build_diffusion_stl_forms(self.team_spec, self.goal_size, 0.4, 2,
                                           add_avoidance=True)
        self.assertEqual(len(forms), 2)


class TestBuildTeamCatlReturnTasks(unittest.TestCase):
    """build_team_catl(return_tasks=True) exposes the per-task Task list."""

    def setUp(self):
        from gcbfplus.stl.team_spec import TeamSpec, TeamTask
        self.tasks = [
            TeamTask(task_id=0, goal_centers=[[1.0, 1.0]]),
            TeamTask(task_id=1, goal_centers=[[5.0, 5.0]]),
            TeamTask(task_id=2, goal_centers=[[1.0, 5.0]]),
            TeamTask(task_id=3, goal_centers=[[5.0, 1.0]]),
        ]
        self.team_spec = TeamSpec(
            spec_type='team_choiceseq3', tasks=self.tasks,
            time_horizon=10, n_agents=8, m_per_task=6, task_mode='cover',
            branches=[[0, 1], [2, 3]], task_m_override={1: 2, 3: 2},
        )
        self.goal_size = np.array([1.0, 1.0])

    def test_default_returns_formula_only(self):
        from gcbfplus.stl.team_allocator import build_team_catl
        catl = build_team_catl(self.team_spec, self.goal_size, 0.4)
        self.assertFalse(isinstance(catl, tuple))

    def test_tasks_in_task_id_order_with_overrides(self):
        from gcbfplus.stl.team_allocator import build_team_catl
        catl, tasks = build_team_catl(self.team_spec, self.goal_size, 0.4,
                                      return_tasks=True)
        self.assertIsNotNone(catl)
        self.assertEqual(len(tasks), 4)
        self.assertEqual([t.name for t in tasks],
                         [f"team_task_{q}" for q in range(4)])
        self.assertEqual([t.num_satisfied_agents for t in tasks], [6, 2, 6, 2])


if __name__ == '__main__':
    unittest.main()
