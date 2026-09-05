"""Global STLPY baseline: joint multi-agent MILP for team specs.

Solves a single MILP over ALL agents simultaneously, with the global CaTL+
objective as the optimization target. The MILP implicitly discovers the
optimal task-to-agent assignment via binary variables. No inter-agent
constraints (no collision avoidance, no temporal coupling).

The joint state vector is [x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}]
with dimension 2*N. Each agent's predicates reference its own indices
in the joint state via inside_rectangle_formula(bounds, y1=2*a, y2=2*a+1, d=2*N).

The formula encodes, for each task q, that at least m_q agents satisfy φ_q (m_q = 1 for
cover specs; the per-task count for redundant / choiceseq3 specs, encoded over agent subsets),
with OR-branch specs encoded as a disjunction over their branches.
"""

import logging
import functools as ft
import numpy as np
from typing import Tuple, List

from ds.stl import GurobiMICPSolver, outside_rectangle_formula
from ds.utils import inside_rectangle_formula
from stlpy.systems import LinearSystem

from gcbfplus.stl.team_spec import TeamSpec, TeamTask

logger = logging.getLogger(__name__)


class GlobalStlpySolver:
    """Multi-agent MILP solver that optimizes the global CaTL+ objective.

    Jointly assigns agents to tasks and plans trajectories using a single MILP.
    No inter-agent constraints (no collision avoidance, no temporal coupling).
    """

    def __init__(self, n_agents: int, space_dim: int = 2):
        self.n_agents = n_agents
        self.space_dim = space_dim
        self.joint_dim = space_dim * n_agents

    @staticmethod
    def _get_joint_ctrl_system(n_agents: int, space_dim: int = 2) -> LinearSystem:
        """Create a joint linear system for all agents (single integrator)."""
        d = space_dim * n_agents
        A = np.eye(d)
        B = np.eye(d)
        C = np.eye(d)
        D = np.zeros((d, d))
        return LinearSystem(A, B, C, D)

    @staticmethod
    def _make_agent_reach_formula(center, goal_size, shrink_factor,
                                  agent_id, n_agents, space_dim, pred_name):
        """Create an stlpy inside-rectangle formula for a specific agent."""
        d = space_dim * n_agents
        y1_index = space_dim * agent_id
        y2_index = space_dim * agent_id + 1

        cent = np.asarray(center, dtype=np.float64)[:space_dim]
        size = np.asarray(goal_size, dtype=np.float64)[:space_dim]

        bounds = np.stack([
            cent - size * shrink_factor / 2,
            cent + size * shrink_factor / 2,
        ]).T.flatten()  # (y1_min, y1_max, y2_min, y2_max)

        return inside_rectangle_formula(bounds, y1_index, y2_index, d, name=pred_name)

    @staticmethod
    def _make_agent_avoid_formula(center, goal_size, expansion,
                                  agent_id, n_agents, space_dim, pred_name):
        """Create an stlpy outside-rectangle formula for a specific agent."""
        d = space_dim * n_agents
        y1_index = space_dim * agent_id
        y2_index = space_dim * agent_id + 1

        cent = np.asarray(center, dtype=np.float64)[:space_dim]
        size = np.asarray(goal_size, dtype=np.float64)[:space_dim] * expansion

        bounds = np.stack([
            cent - size / 2,
            cent + size / 2,
        ]).T.flatten()

        return outside_rectangle_formula(bounds, y1_index, y2_index, d, name=pred_name)

    @classmethod
    def _build_agent_task_formula(cls, task: TeamTask, time_horizon: int,
                                  goal_size, shrink_factor, agent_id: int,
                                  n_agents: int, space_dim: int,
                                  task_mode: str, pred_id_offset: int):
        """Build STLpy formula for one agent satisfying one task.

        Cover mode:  F[0,T] Reach(g0) & F[0,T] Reach(g1) & ...
        Sequential:  F[0,T/K] Reach(g0) & F[T/K,2T/K] Reach(g1) & ...

        Returns:
            stlpy STLTree formula
        """
        n_goals = len(task.goal_centers)

        if task_mode == 'cover':
            # Cover mode: all goals must be reached, any order within [0, T]
            # Note: stlpy uses inclusive time bounds [a,b], so T-1 covers steps 0..T-1.
            # This matches ds.stl_jax which uses exclusive upper bound [a,b), so T covers 0..T-1.
            formulas = []
            for g_idx, goal_center in enumerate(task.goal_centers):
                pred_name = f"t{task.task_id}_a{agent_id}_g{g_idx}"
                reach = cls._make_agent_reach_formula(
                    goal_center, goal_size, shrink_factor,
                    agent_id, n_agents, space_dim, pred_name)
                formulas.append(reach.eventually(0, time_horizon - 1))
            return ft.reduce(lambda a, b: a & b, formulas)

        else:
            # Sequential mode: ordered time windows
            interval_length = time_horizon // n_goals
            formulas = []
            for g_idx, goal_center in enumerate(task.goal_centers):
                pred_name = f"t{task.task_id}_a{agent_id}_g{g_idx}"
                reach = cls._make_agent_reach_formula(
                    goal_center, goal_size, shrink_factor,
                    agent_id, n_agents, space_dim, pred_name)
                t_start = g_idx * interval_length
                t_end = (g_idx + 1) * interval_length if g_idx < n_goals - 1 else time_horizon
                formulas.append(reach.eventually(t_start, t_end - 1))
            return ft.reduce(lambda a, b: a & b, formulas)

    @classmethod
    def build_joint_stlpy_formula(cls, team_spec: TeamSpec, goal_size,
                                  shrink_factor: float, n_agents: int,
                                  space_dim: int = 2,
                                  add_avoidance: bool = False,
                                  avoid_expansion: float = 1.2):
        """Build the joint STLpy formula: ∧_q ∨_{a=0}^{N-1} φ_q(x_a).

        For each task q, at least m agents must satisfy the task's STL spec.
        With avoidance, each agent's task formula includes avoid predicates
        for other tasks' goals (time-windowed for sequential, spatial for cover).

        The MILP solver discovers the optimal assignment via binary variables.

        Args:
            team_spec: parsed team specification
            goal_size: goal region size array
            shrink_factor: conservative shrink factor for reach predicates
            n_agents: total number of agents
            space_dim: state dimension per agent (2 for 2D)
            add_avoidance: include avoid predicates for other tasks' goals
            avoid_expansion: expansion factor for avoid regions

        Returns:
            stlpy STLTree formula
        """
        T = team_spec.time_horizon
        is_sequential = (team_spec.task_mode == 'sequential')
        m_overrides = team_spec.task_m_override or {}
        task_goal_sets = {task.task_id: set(tuple(g[:2]) for g in task.goal_centers)
                          for task in team_spec.tasks}

        task_formulas = []
        pred_id_offset = 0
        avoid_pred_id = 1000  # offset to avoid name collisions with reach preds

        for task in team_spec.tasks:
            # Build φ_q(x_a) for each agent a, optionally with avoid
            agent_formulas = []
            for agent_id in range(n_agents):
                phi_q_a = cls._build_agent_task_formula(
                    task=task,
                    time_horizon=T,
                    goal_size=goal_size,
                    shrink_factor=shrink_factor,
                    agent_id=agent_id,
                    n_agents=n_agents,
                    space_dim=space_dim,
                    task_mode=team_spec.task_mode,
                    pred_id_offset=pred_id_offset,
                )

                # Add avoid predicates for other tasks' goals
                if add_avoidance:
                    own_goals = task.goal_centers
                    own_goal_tuples = task_goal_sets[task.task_id]
                    n_goals = len(own_goals)
                    interval = max(1, T // n_goals) if is_sequential else T

                    for other_task in team_spec.tasks:
                        if other_task.task_id == task.task_id:
                            continue
                        for g_idx, other_goal in enumerate(other_task.goal_centers):
                            other_goal_tuple = tuple(other_goal[:2])
                            if is_sequential:
                                other_start = g_idx * interval
                                other_end = min((g_idx + 1) * interval, T)
                                skip = False
                                for my_idx, my_goal in enumerate(own_goals):
                                    if tuple(my_goal[:2]) == other_goal_tuple:
                                        my_start = my_idx * interval
                                        my_end = min((my_idx + 1) * interval, T)
                                        if my_start < other_end and other_start < my_end:
                                            skip = True
                                            break
                                if skip:
                                    continue
                                avoid_name = f"av_t{task.task_id}_a{agent_id}_o{other_task.task_id}g{g_idx}"
                                avoid_f = cls._make_agent_avoid_formula(
                                    other_goal, goal_size, avoid_expansion,
                                    agent_id, n_agents, space_dim, avoid_name)
                                phi_q_a = phi_q_a & avoid_f.always(
                                    other_start, max(other_start, other_end - 1))
                            else:
                                if other_goal_tuple not in own_goal_tuples:
                                    avoid_name = f"av_t{task.task_id}_a{agent_id}_o{other_task.task_id}g{g_idx}"
                                    avoid_f = cls._make_agent_avoid_formula(
                                        other_goal, goal_size, avoid_expansion,
                                        agent_id, n_agents, space_dim, avoid_name)
                                    phi_q_a = phi_q_a & avoid_f.always(0, T - 1)

                agent_formulas.append(phi_q_a)

            pred_id_offset += len(task.goal_centers)

            # Task q formula: need at least m_q agents to satisfy
            # (task_m_override honored, e.g. choiceseq3 main 3N/4 vs support N/4)
            m = m_overrides.get(task.task_id, team_spec.m_per_task)
            if m == 1:
                # ∨_a φ_q(x_a) — at least one agent satisfies it
                task_formula = ft.reduce(lambda a, b: a | b, agent_formulas)
            else:
                # For m > 1: we need at least m agents. Encode as:
                # ∨ over all C(N,m) subsets of agents, AND within each subset.
                # Gate on the subset COUNT (C(N,m) is symmetric in m — C(8,6)=28),
                # not on m itself.
                from itertools import combinations
                from math import comb
                if comb(n_agents, m) <= 128:
                    # Exact: OR over all m-subsets, AND within each subset
                    subset_formulas = []
                    for subset in combinations(range(n_agents), m):
                        subset_f = ft.reduce(lambda a, b: a & b,
                                             [agent_formulas[a] for a in subset])
                        subset_formulas.append(subset_f)
                    task_formula = ft.reduce(lambda a, b: a | b, subset_formulas)
                else:
                    # Fallback: just OR (at least 1 agent) — MILP will be too large otherwise
                    logger.warning(f"m_per_task={m} with N={n_agents} too large for exact "
                                   f"encoding. Using m=1 relaxation for global solver.")
                    task_formula = ft.reduce(lambda a, b: a | b, agent_formulas)

            task_formulas.append(task_formula)

        # Global formula follows the outer structure of the team spec:
        # OR between branches (AND within), else ∧_q task_q.
        if team_spec.branches is not None:
            branch_formulas = []
            for branch_task_ids in team_spec.branches:
                branch_f = ft.reduce(lambda a, b: a & b,
                                     [task_formulas[tid] for tid in branch_task_ids])
                branch_formulas.append(branch_f)
            return ft.reduce(lambda a, b: a | b, branch_formulas)
        global_formula = ft.reduce(lambda a, b: a & b, task_formulas)
        return global_formula

    def solve(
        self,
        team_spec: TeamSpec,
        agent_positions: np.ndarray,
        goal_size: np.ndarray,
        shrink_factor: float,
        total_time: int,
        u_bound: tuple = (-20.0, 20.0),
        rho_min: float = 0.1,
        time_limit: float = 120.0,
        threads: int = 0,
        verbose: bool = False,
        add_avoidance: bool = False,
        avoid_expansion: float = 1.2,
    ) -> Tuple[List[np.ndarray], dict]:
        """Solve the joint multi-agent MILP.

        Args:
            team_spec: parsed team specification
            agent_positions: (N, 2+) initial positions
            goal_size: goal region size
            shrink_factor: predicate shrink factor
            total_time: time horizon (number of steps)
            u_bound: control input bounds
            rho_min: minimum robustness
            time_limit: Gurobi time limit in seconds
            threads: Gurobi threads (0=auto)
            verbose: print solver output

        Returns:
            (trajectories, info) where trajectories is a list of (T+1, 2) arrays
        """
        n_agents = self.n_agents
        space_dim = self.space_dim

        # Check feasibility
        n_tasks = len(team_spec.tasks)
        if n_agents < n_tasks:
            logger.warning(
                f"Not enough agents ({n_agents}) for {n_tasks} tasks. "
                f"MILP will be infeasible.")

        # Build joint initial state
        x0 = np.concatenate([
            agent_positions[a, :space_dim] for a in range(n_agents)
        ])  # (2*N,)

        # Build joint system
        sys = self._get_joint_ctrl_system(n_agents, space_dim)

        # Build joint STL formula
        logger.info(f"Building joint MILP formula for {n_agents} agents, "
                     f"{n_tasks} tasks, T={total_time}...")
        spec = self.build_joint_stlpy_formula(
            team_spec, goal_size, shrink_factor, n_agents, space_dim,
            add_avoidance=add_avoidance, avoid_expansion=avoid_expansion)

        # Create and configure solver
        logger.info(f"Setting up Gurobi solver (joint dim={self.joint_dim}, "
                     f"T={total_time}, time_limit={time_limit}s)...")
        solver = GurobiMICPSolver(
            spec, sys, x0, total_time, verbose=verbose)

        # Energy minimization: zero state cost, unit control cost
        Q = np.zeros((self.joint_dim, self.joint_dim))
        R = np.eye(self.joint_dim)
        solver.AddQuadraticCost(Q, R)
        solver.AddControlBounds(*u_bound)
        solver.AddRobustnessConstraint(rho_min=rho_min)

        # Solve — GurobiMICPSolver.Solve() only returns solutions for OPTIMAL status.
        # For large joint MILPs, Gurobi may time out with a feasible (but suboptimal)
        # solution. We extract it directly from the model in that case.
        logger.info("Solving joint MILP...")
        x, u, rho, solve_t = solver.Solve(time_limit=time_limit, threads=threads)

        # If the solver returned None but has feasible solutions (e.g. TIME_LIMIT),
        # extract the best feasible solution directly from the Gurobi model.
        import gurobipy as gp
        model = solver.model
        if x is None and model.SolCount > 0:
            logger.info(f"Gurobi status {model.status} with {model.SolCount} feasible "
                        f"solutions. Extracting best feasible solution.")
            x = solver.x.X
            u = solver.u.X
            rho = solver.rho.X[0]
            solve_t = model.Runtime

        info = {
            'rho': float(rho) if rho is not None else float('-inf'),
            'solve_t': float(solve_t) if solve_t is not None else 0.0,
            'feasible': x is not None,
            'optimal': model.status == gp.GRB.OPTIMAL,
            'joint_dim': self.joint_dim,
            'n_tasks': n_tasks,
        }

        if x is not None:
            # x shape: (joint_dim, T) from Gurobi
            joint_x = np.array(x)  # (2*N, T)
            trajectories = self._extract_per_agent_trajectories(
                joint_x, n_agents, space_dim)
            status = "optimal" if info['optimal'] else "suboptimal (time limit)"
            logger.info(f"Joint MILP solved ({status}): rho={rho:.4f}, "
                        f"time={solve_t:.2f}s")
        else:
            logger.warning(f"Joint MILP infeasible (Gurobi status {model.status}). "
                           f"Falling back to stay-in-place.")
            trajectories = []
            for a in range(n_agents):
                stay = np.tile(agent_positions[a, :space_dim], (total_time + 1, 1))
                trajectories.append(stay)

        return trajectories, info

    @staticmethod
    def _extract_per_agent_trajectories(
        joint_x: np.ndarray,
        n_agents: int,
        space_dim: int = 2,
    ) -> List[np.ndarray]:
        """Extract per-agent trajectories from joint MILP solution.

        Args:
            joint_x: (joint_dim, T) joint trajectory from Gurobi
            n_agents: number of agents
            space_dim: state dimension per agent

        Returns:
            List of (T, space_dim) arrays, one per agent
        """
        joint_x_T = joint_x.T  # (T, joint_dim)
        trajectories = []
        for a in range(n_agents):
            start = space_dim * a
            end = space_dim * (a + 1)
            agent_traj = joint_x_T[:, start:end]  # (T, space_dim)
            trajectories.append(agent_traj)
        return trajectories
