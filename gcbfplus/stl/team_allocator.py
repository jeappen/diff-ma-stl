"""Team task allocation and STL formula construction.

Handles:
- Task allocation: greedy (nearest first/last goal), random, Hungarian (oracle_hungarian), and
  post-hoc inference from planned trajectories (disjunctive mode)
- CaTLPlus team-level formula construction (for scoring and joint guidance)
- Per-agent STL form construction (for planner guidance)
"""

import functools as ft
import numpy as np
from dataclasses import dataclass

from ds.stl_jax import STL, RectReachPredicate, RectAvoidPredicate
from ds.ma_stl_jax import Task, CaTLPlus

from gcbfplus.stl.team_spec import TeamSpec, TeamTask


@dataclass
class AllocationResult:
    """Result of task allocation."""
    agent_to_task: dict    # agent_id → task_id
    task_to_agents: dict   # task_id → list of agent_ids
    unassigned_agents: list  # agent ids not assigned to any task


def allocate_tasks(team_spec: TeamSpec, agent_positions: np.ndarray,
                   strategy: str = 'greedy') -> AllocationResult:
    """Allocate agents to tasks using the specified strategy.

    Args:
        team_spec: parsed team specification
        agent_positions: (n_agents, 2) array of agent positions
        strategy: 'greedy' (nearest to first waypoint), 'greedy_last' (nearest to final goal),
                  'oracle_hungarian' (min-cost assignment) or 'random'

    Returns:
        AllocationResult with agent-task mapping
    """
    # For OR specs: pick the best branch first, then allocate within it
    if team_spec.branches is not None:
        return _allocate_branch_aware(team_spec, agent_positions, strategy)

    if strategy == 'random':
        return _allocate_random(team_spec, agent_positions)
    elif strategy == 'oracle_hungarian':
        return _allocate_optimal(team_spec, agent_positions)
    elif strategy == 'greedy_last':
        return _allocate_by_goal(team_spec, agent_positions, goal_index=-1)
    else:  # greedy (default)
        return _allocate_by_goal(team_spec, agent_positions, goal_index=0)


def _allocate_branch_aware(team_spec: TeamSpec, agent_positions: np.ndarray,
                            strategy: str) -> AllocationResult:
    """For OR specs: evaluate each branch independently, pick the best one.

    Tries allocating to each branch separately (as if only that branch's tasks exist),
    then picks the branch with lowest total assignment cost.
    """
    import copy
    best_alloc = None
    best_cost = float('inf')

    for branch_ids in team_spec.branches:
        # Create a sub-spec with only this branch's tasks
        branch_tasks = [t for t in team_spec.tasks if t.task_id in branch_ids]
        sub_spec = copy.copy(team_spec)
        sub_spec.tasks = branch_tasks
        sub_spec.branches = None  # treat as regular spec

        # Allocate using the requested strategy
        if strategy == 'oracle_hungarian':
            alloc = _allocate_optimal(sub_spec, agent_positions)
        elif strategy == 'random':
            alloc = _allocate_random(sub_spec, agent_positions)
        elif strategy == 'greedy_last':
            alloc = _allocate_by_goal(sub_spec, agent_positions, goal_index=-1)
        else:
            alloc = _allocate_by_goal(sub_spec, agent_positions, goal_index=0)

        # Compute total cost (sum of distances to assigned task goals)
        cost = 0.0
        for task in branch_tasks:
            goal = task.goal_centers[-1][:2]  # final destination
            for aid in alloc.task_to_agents.get(task.task_id, []):
                cost += np.linalg.norm(agent_positions[aid, :2] - goal)

        if cost < best_cost:
            best_cost = cost
            best_alloc = alloc

    return best_alloc


def _allocate_by_goal(team_spec: TeamSpec, agent_positions: np.ndarray,
                      goal_index: int = 0) -> AllocationResult:
    """Greedy nearest-agent allocation to a specific goal in each task.

    Args:
        team_spec: parsed team specification
        agent_positions: (n_agents, 2) array of agent positions
        goal_index: which goal_center to use for distance (0=first, -1=last)
    """
    n_agents = team_spec.n_agents
    m_default = team_spec.m_per_task
    m_overrides = team_spec.task_m_override or {}
    available = set(range(n_agents))
    agent_to_task = {}
    task_to_agents = {}

    for task in team_spec.tasks:
        task_center = task.goal_centers[goal_index]
        assigned = []
        m_this_task = m_overrides.get(task.task_id, m_default)

        for _ in range(m_this_task):
            if not available:
                break

            # Find nearest available agent
            best_agent = None
            best_dist = float('inf')
            for agent_id in available:
                dist = np.linalg.norm(agent_positions[agent_id, :2] - task_center[:2])
                if dist < best_dist:
                    best_dist = dist
                    best_agent = agent_id

            agent_to_task[best_agent] = task.task_id
            assigned.append(best_agent)
            available.discard(best_agent)

        task_to_agents[task.task_id] = assigned

    return AllocationResult(
        agent_to_task=agent_to_task,
        task_to_agents=task_to_agents,
        unassigned_agents=sorted(available),
    )


def _allocate_optimal(team_spec: TeamSpec, agent_positions: np.ndarray) -> AllocationResult:
    """Globally optimal assignment using the Hungarian algorithm.

    Builds a cost matrix of agent-to-task distances (using the mean distance
    to all goal centers in each task) and solves the minimum-cost assignment.
    Each task offers m_per_task slots (the spec-wide default; per-task overrides
    such as choiceseq3's smaller support tasks are enforced at evaluation by the
    CaTL+ counting, not by the slot count here). This is the behaviour behind the
    reported oracle_hungarian numbers.
    """
    from scipy.optimize import linear_sum_assignment

    n_agents = team_spec.n_agents
    m_per_task = team_spec.m_per_task
    tasks = team_spec.tasks
    n_tasks = len(tasks)
    n_slots = n_tasks * m_per_task  # total agent slots

    # Build cost matrix: (n_agents, n_slots)
    # Each task is replicated m_per_task times so the assignment picks M agents per task
    cost = np.full((n_agents, n_slots), fill_value=1e6)
    for t_idx, task in enumerate(tasks):
        # Mean distance to all goal centers in this task
        goal_centers = np.array([g[:2] for g in task.goal_centers])
        for a_idx in range(n_agents):
            dist = np.mean(np.linalg.norm(agent_positions[a_idx, :2] - goal_centers, axis=1))
            for m in range(m_per_task):
                cost[a_idx, t_idx * m_per_task + m] = dist

    # Solve assignment (handles n_agents >= n_slots)
    row_ind, col_ind = linear_sum_assignment(cost)

    agent_to_task = {}
    task_to_agents = {task.task_id: [] for task in tasks}

    for a_idx, slot in zip(row_ind, col_ind):
        if slot < n_slots:
            t_idx = slot // m_per_task
            task_id = tasks[t_idx].task_id
            agent_to_task[a_idx] = task_id
            task_to_agents[task_id].append(a_idx)

    unassigned = sorted(set(range(n_agents)) - set(agent_to_task.keys()))
    return AllocationResult(
        agent_to_task=agent_to_task,
        task_to_agents=task_to_agents,
        unassigned_agents=unassigned,
    )


def _allocate_random(team_spec: TeamSpec, agent_positions: np.ndarray) -> AllocationResult:
    """Random assignment of M agents per task."""
    n_agents = team_spec.n_agents
    m_per_task = team_spec.m_per_task
    agent_ids = list(range(n_agents))
    np.random.shuffle(agent_ids)

    agent_to_task = {}
    task_to_agents = {}
    idx = 0

    for task in team_spec.tasks:
        assigned = []
        for _ in range(m_per_task):
            if idx >= len(agent_ids):
                break
            agent_id = agent_ids[idx]
            agent_to_task[agent_id] = task.task_id
            assigned.append(agent_id)
            idx += 1
        task_to_agents[task.task_id] = assigned

    unassigned = sorted(agent_ids[idx:])
    return AllocationResult(
        agent_to_task=agent_to_task,
        task_to_agents=task_to_agents,
        unassigned_agents=unassigned,
    )


def _make_reach_pred(center, goal_size, pred_id, shrink_factor):
    """Create a reach predicate for a goal center."""
    return STL(RectReachPredicate(
        np.array(center, dtype=np.float64),
        np.array(goal_size, dtype=np.float64),
        pred_id,
        shrink_factor=shrink_factor,
    ))


def _make_avoid_pred(center, goal_size, pred_id, expansion=1.2):
    """Create an avoidance predicate for a goal center.

    Args:
        expansion: Scale factor for the avoid region size relative to goal_size.
            1.0 = same size as goal, 1.2 = 20% larger (conservative default).
    """
    return STL(RectAvoidPredicate(
        np.array(center, dtype=np.float64),
        np.array(goal_size, dtype=np.float64) * expansion,
        pred_id,
    ))


def _build_task_stl(task: TeamTask, time_horizon: int, goal_size, shrink_factor,
                    pred_id_offset: int = 0, task_mode: str = 'sequential'):
    """Build STL formula for a single task.

    - Single goal: F[0, T] Reach(goal)
    - Multiple goals, sequential: F[0, T/K] Reach(g0) & F[T/K, 2T/K] Reach(g1) & ...
    - Multiple goals, cover: F[0, T] Reach(g0) & F[0, T] Reach(g1) & ... (any order)

    Args:
        task: the team task
        time_horizon: total time horizon
        goal_size: goal region size array
        shrink_factor: shrink factor for reach predicates
        pred_id_offset: offset for predicate naming (to avoid collisions)
        task_mode: 'sequential' (ordered time windows) or 'cover' (any order within [0,T])

    Returns:
        STL formula for this task
    """
    n_goals = len(task.goal_centers)

    if n_goals == 1:
        # Single goal reach
        pred = _make_reach_pred(task.goal_centers[0], goal_size, pred_id_offset, shrink_factor)
        return pred.eventually(0, time_horizon)

    if task_mode == 'cover':
        # Cover mode: all goals must be reached, any order within [0, T]
        pred = _make_reach_pred(task.goal_centers[0], goal_size, pred_id_offset, shrink_factor)
        stl_form = pred.eventually(0, time_horizon)
        for g_idx, goal_center in enumerate(task.goal_centers[1:], start=1):
            pred = _make_reach_pred(goal_center, goal_size, pred_id_offset + g_idx, shrink_factor)
            stl_form = stl_form & pred.eventually(0, time_horizon)
        return stl_form

    # Sequential mode: ordered time windows
    interval_length = time_horizon // n_goals

    # Build from last goal backwards (same pattern as STLMixin._load_seq_stl_form)
    last_idx = n_goals - 1
    pred = _make_reach_pred(task.goal_centers[last_idx], goal_size,
                            pred_id_offset + last_idx, shrink_factor)
    stl_form = pred.eventually(max(0, (n_goals - 1) * interval_length), time_horizon)

    for i, goal_center in enumerate(reversed(task.goal_centers[:last_idx])):
        time_ind = last_idx - i
        pred = _make_reach_pred(goal_center, goal_size,
                                pred_id_offset + (time_ind - 1), shrink_factor)
        stl_form = pred.eventually(max(0, (time_ind - 1) * interval_length),
                                   time_ind * interval_length) & stl_form

    return stl_form


def _build_avoid_forms_for_task(team_spec, task, goal_size, avoid_expansion, pred_id_start):
    """Build avoidance STL forms for a given task — shared by all builders.

    For sequential tasks: time-windowed avoidance (avoid other tasks' goals during conflicting phases)
    For cover tasks: spatial avoidance (avoid other tasks' non-shared goals for full horizon)

    Returns:
        (avoid_forms: list of STL forms, next_pred_id: int, regions: list of (center, size))
    """
    T = team_spec.time_horizon
    is_sequential = (team_spec.task_mode == 'sequential')
    own_goals = task.goal_centers
    own_goal_tuples = set(tuple(g[:2]) for g in own_goals)
    n_goals = len(own_goals)
    interval = max(1, T // n_goals) if is_sequential else T

    avoid_forms_list = []
    regions = []
    avoid_pred_id = pred_id_start

    # For OR specs (branches != None): cross-branch avoidance.
    # Tasks avoid goals from OTHER branches (not their own).
    # This enforces spatial separation between alternative branches in DNF specs.
    if team_spec.branches is not None:
        own_branch = None
        for branch in team_spec.branches:
            if task.task_id in branch:
                own_branch = set(branch)
                break
        # Avoid all tasks NOT in own branch
        all_task_ids = {t.task_id for t in team_spec.tasks}
        avoid_task_ids = all_task_ids - (own_branch if own_branch else set())
    else:
        avoid_task_ids = {t.task_id for t in team_spec.tasks} - {task.task_id}

    for other_task in team_spec.tasks:
        if other_task.task_id not in avoid_task_ids:
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
                avoid_pred = _make_avoid_pred(other_goal, goal_size, avoid_pred_id,
                                               expansion=avoid_expansion)
                avoid_forms_list.append(avoid_pred.always(other_start, other_end))
            else:
                if other_goal_tuple not in own_goal_tuples:
                    avoid_pred = _make_avoid_pred(other_goal, goal_size, avoid_pred_id,
                                                   expansion=avoid_expansion)
                    avoid_forms_list.append(avoid_pred.always(0, T))
                else:
                    continue

            avoid_pred_id += 1
            regions.append((
                np.array(other_goal[:2], dtype=np.float64),
                np.array(goal_size, dtype=np.float64) * avoid_expansion
            ))

    return avoid_forms_list, avoid_pred_id, regions


def build_team_catl(team_spec: TeamSpec, goal_size, shrink_factor,
                    add_avoidance=False, avoid_expansion=1.2,
                    return_tasks=False) -> CaTLPlus:
    """Build CaTLPlus team-level formula for scoring.

    Creates: ∧_q CaTLPlus(Task(q, stl_q, num_satisfied_agents=m_per_task))

    When add_avoidance=True, each task's STL formula includes avoidance of other
    tasks' goals, so agents that violate avoid get negative scores and don't count
    toward the M required agents.

    Args:
        team_spec: parsed team specification
        goal_size: goal region size array (e.g. [1, 1])
        shrink_factor: shrink factor for reach predicates
        add_avoidance: if True, add avoid predicates to each task's formula
        avoid_expansion: expansion factor for avoid regions
        return_tasks: if True, return (formula, [Task, ...]) with the Task list
            in task_id order (task_m_override applied via num_satisfied_agents)

    Returns:
        CaTLPlus formula representing the team objective, or (formula, tasks)
    """
    catl_list = []
    ma_tasks = []
    pred_id_offset = 0
    m_default = team_spec.m_per_task  # 1 for cover/mofn, >1 for redundant and choiceseq3
    m_overrides = team_spec.task_m_override or {}

    for task in team_spec.tasks:
        stl_form = _build_task_stl(task, team_spec.time_horizon, goal_size, shrink_factor,
                                   pred_id_offset=pred_id_offset, task_mode=team_spec.task_mode)
        pred_id_offset += len(task.goal_centers)

        # Add avoidance to the task's STL formula
        if add_avoidance:
            avoid_forms, pred_id_offset, _ = _build_avoid_forms_for_task(
                team_spec, task, goal_size, avoid_expansion, pred_id_offset)
            if avoid_forms:
                avoid_form = ft.reduce(lambda x, y: x & y, avoid_forms)
                stl_form = stl_form & avoid_form

        m = m_overrides.get(task.task_id, m_default)
        ma_task = Task(f"team_task_{task.task_id}", stl_form, num_satisfied_agents=m, capability=None)
        ma_tasks.append(ma_task)
        catl_list.append(CaTLPlus(ma_task))

    # Build formula according to branch structure
    if team_spec.branches is not None:
        # OR between branches, AND within each branch
        # branches = [[0,1], [2,3]] means (Task0∧Task1) | (Task2∧Task3)
        branch_formulas = []
        for branch_task_ids in team_spec.branches:
            branch_catls = [catl_list[tid] for tid in branch_task_ids]
            branch_formulas.append(ft.reduce(lambda x, y: x & y, branch_catls))
        formula = ft.reduce(lambda x, y: x | y, branch_formulas)
    else:
        # Default: conjoin all tasks: ∧_q CaTLPlus(task_q)
        formula = ft.reduce(lambda x, y: x & y, catl_list)
    return (formula, ma_tasks) if return_tasks else formula


def build_per_agent_stl_forms(team_spec: TeamSpec, allocation: AllocationResult,
                               goal_size, shrink_factor, n_agents: int,
                               agent_positions=None, add_avoidance=False,
                               avoid_expansion=1.2) -> list:
    """Convert allocation to per-agent STL forms for the planner.

    Assigned agents get their task's STL form.
    Unassigned agents get a stay-in-place spec (reach own position) to avoid
    drifting toward goal regions and accidentally satisfying CaTLPlus evaluation.

    Args:
        team_spec: parsed team specification
        allocation: result of allocate_tasks
        goal_size: goal region size array
        shrink_factor: shrink factor for reach predicates
        n_agents: total number of agents
        agent_positions: (n_agents, 2+) array of agent positions, used for
            unassigned agent stay-in-place specs. If None, defaults to area center.
        add_avoidance: if True, each agent's spec includes G[0,T] Avoid(g) for
            goals belonging to other tasks (excluding shared goals)

    Returns:
        List of STL forms, one per agent
    """
    # Pre-build STL forms for each task
    task_stl_forms = {}
    pred_id_offset = 0
    for task in team_spec.tasks:
        task_stl_forms[task.task_id] = _build_task_stl(
            task, team_spec.time_horizon, goal_size, shrink_factor,
            pred_id_offset=pred_id_offset, task_mode=team_spec.task_mode)
        pred_id_offset += len(task.goal_centers)

    # Pre-compute per-task goal sets (as tuples for set ops)
    task_goal_sets = {}
    for task in team_spec.tasks:
        task_goal_sets[task.task_id] = set(tuple(g[:2]) for g in task.goal_centers)

    # Build avoidance forms per task using the shared helper
    task_avoid_forms = {}
    task_avoid_regions = {}
    if add_avoidance:
        avoid_pred_id = pred_id_offset
        for task in team_spec.tasks:
            avoid_forms_list, avoid_pred_id, regions = _build_avoid_forms_for_task(
                team_spec, task, goal_size, avoid_expansion, avoid_pred_id)
            if avoid_forms_list:
                task_avoid_forms[task.task_id] = ft.reduce(lambda x, y: x & y, avoid_forms_list)
            task_avoid_regions[task.task_id] = regions

    stl_forms = [None] * n_agents
    agent_avoid_regions = [[] for _ in range(n_agents)]

    # Assigned agents get their task's STL form (+ avoidance if enabled)
    for agent_id, task_id in allocation.agent_to_task.items():
        form = task_stl_forms[task_id]
        if add_avoidance and task_id in task_avoid_forms:
            form = form & task_avoid_forms[task_id]
        stl_forms[agent_id] = form
        if add_avoidance and task_id in task_avoid_regions:
            agent_avoid_regions[agent_id] = task_avoid_regions[task_id]

    # Unassigned agents: stay in place (reach own position) to avoid drifting toward goals
    for agent_id in allocation.unassigned_agents:
        if agent_positions is not None:
            stay_pos = agent_positions[agent_id, :2]
        else:
            # Fallback (only hit by the structural dummy allocation in _load_team_spec): area centre
            stay_pos = np.array([3.0, 3.0])
        pred = _make_reach_pred(stay_pos, goal_size, 0, shrink_factor)
        stl_forms[agent_id] = pred.eventually(0, team_spec.time_horizon)

    return stl_forms, agent_avoid_regions


def build_diffusion_stl_forms(team_spec: TeamSpec, goal_size, shrink_factor, n_agents: int,
                               add_avoidance: bool = False, avoid_expansion: float = 1.2) -> list:
    """Build per-agent STL forms for diffusion planner (no pre-allocation).

    Every agent gets the same disjunctive spec:
      Without avoid: φ_task0 | φ_task1 | ... | φ_taskQ  ("complete ANY one task")
      With avoid:    (φ_task0 & avoid_others_0) | (φ_task1 & avoid_others_1) | ...
                     ("pick ONE task and avoid all other tasks' goals")

    The CaTL+ gradient during diffusion sampling handles allocation by
    differentiating which agent focuses on which task.

    Args:
        team_spec: parsed team specification
        goal_size: goal region size array
        shrink_factor: shrink factor for reach predicates
        n_agents: total number of agents
        add_avoidance: if True, each disjunct includes avoidance of other tasks' goals
        avoid_expansion: expansion factor for avoid regions

    Returns:
        List of STL forms (all identical), one per agent
    """
    T = team_spec.time_horizon
    is_sequential = (team_spec.task_mode == 'sequential')

    # Build each task's STL formula
    task_stl_forms = []
    pred_id_offset = 0
    for task in team_spec.tasks:
        task_form = _build_task_stl(
            task, T, goal_size, shrink_factor,
            pred_id_offset=pred_id_offset, task_mode=team_spec.task_mode)
        task_stl_forms.append(task_form)
        pred_id_offset += len(task.goal_centers)

    if add_avoidance:
        # For each task, build: φ_task_i & G[...] Avoid(other_goals)
        # Agent picks ONE task and avoids all other tasks' goals
        task_goal_sets = {task.task_id: set(tuple(g[:2]) for g in task.goal_centers)
                          for task in team_spec.tasks}
        avoid_pred_id = pred_id_offset

        disjuncts = []
        for t_idx, task in enumerate(team_spec.tasks):
            own_goals = task_goal_sets[task.task_id]
            n_goals = len(task.goal_centers)
            interval = max(1, T // n_goals) if is_sequential else T

            # For OR specs: cross-branch avoidance (same as _build_avoid_forms_for_task)
            if team_spec.branches is not None:
                own_branch = None
                for branch in team_spec.branches:
                    if task.task_id in branch:
                        own_branch = set(branch)
                        break
                # Avoid all tasks NOT in own branch
                all_task_ids = {t.task_id for t in team_spec.tasks}
                disjunct_avoid_ids = all_task_ids - (own_branch if own_branch else set())
            else:
                disjunct_avoid_ids = {t.task_id for t in team_spec.tasks} - {task.task_id}

            avoid_forms_list = []
            for other_task in team_spec.tasks:
                if other_task.task_id not in disjunct_avoid_ids:
                    continue
                for g_idx, other_goal in enumerate(other_task.goal_centers):
                    other_goal_tuple = tuple(other_goal[:2])

                    if is_sequential:
                        # Time-windowed avoidance (same logic as build_per_agent_stl_forms)
                        other_start = g_idx * interval
                        other_end = min((g_idx + 1) * interval, T)
                        skip = False
                        for my_idx, my_goal in enumerate(task.goal_centers):
                            if tuple(my_goal[:2]) == other_goal_tuple:
                                my_start = my_idx * interval
                                my_end = min((my_idx + 1) * interval, T)
                                if my_start < other_end and other_start < my_end:
                                    skip = True
                                    break
                        if skip:
                            continue
                        avoid_pred = _make_avoid_pred(other_goal, goal_size, avoid_pred_id,
                                                       expansion=avoid_expansion)
                        avoid_forms_list.append(avoid_pred.always(other_start, other_end))
                    else:
                        # Spatial avoidance for full horizon
                        if other_goal_tuple not in own_goals:
                            avoid_pred = _make_avoid_pred(other_goal, goal_size, avoid_pred_id,
                                                           expansion=avoid_expansion)
                            avoid_forms_list.append(avoid_pred.always(0, T))
                    avoid_pred_id += 1

            # Combine: φ_task_i & avoid(other_goals)
            branch = task_stl_forms[t_idx]
            if avoid_forms_list:
                avoid_form = ft.reduce(lambda x, y: x & y, avoid_forms_list)
                branch = branch & avoid_form
            disjuncts.append(branch)

        disjunctive_form = ft.reduce(lambda a, b: a | b, disjuncts)
    else:
        # Simple disjunction without avoidance
        disjunctive_form = ft.reduce(lambda a, b: a | b, task_stl_forms)

    return [disjunctive_form] * n_agents


def infer_allocation_from_trajectories(team_spec: TeamSpec, trajectories: np.ndarray,
                                        goal_size, shrink_factor,
                                        goal_sample_interval: int = 1) -> AllocationResult:
    """Infer agent-task allocation from planned trajectories using robustness matching.

    Scores each agent's trajectory against each task's STL formula, then solves
    a maximum-weight matching to find the best agent-task assignment.

    Args:
        team_spec: parsed team specification
        trajectories: (N, T, D) array of planned positions, or list of (T, D) arrays
        goal_size: goal region size array
        shrink_factor: shrink factor for reach predicates
        goal_sample_interval: subsample trajectories at this interval for STL eval

    Returns:
        AllocationResult with inferred agent-task mapping
    """
    from scipy.optimize import linear_sum_assignment

    # Build per-task STL forms
    task_stl_forms = {}
    pred_id_offset = 0
    for task in team_spec.tasks:
        task_stl_forms[task.task_id] = _build_task_stl(
            task, team_spec.time_horizon, goal_size, shrink_factor,
            pred_id_offset=pred_id_offset, task_mode=team_spec.task_mode)
        pred_id_offset += len(task.goal_centers)

    # Convert trajectories to array
    if isinstance(trajectories, list):
        traj_array = np.stack(trajectories)  # (N, T, D)
    else:
        traj_array = np.array(trajectories)

    n_agents = traj_array.shape[0]
    n_tasks = len(team_spec.tasks)
    m_per_task = team_spec.m_per_task

    # Subsample trajectories for STL evaluation
    if goal_sample_interval > 1:
        traj_sub = traj_array[:, ::goal_sample_interval, :]
    else:
        traj_sub = traj_array

    # Build robustness matrix: (N_agents, N_tasks)
    import jax.numpy as jnp
    rob_matrix = np.zeros((n_agents, n_tasks))
    for t_idx, task in enumerate(team_spec.tasks):
        stl_form = task_stl_forms[task.task_id]
        for a_idx in range(n_agents):
            # STL eval expects (batch, time, dim)
            agent_traj = jnp.array(traj_sub[a_idx][None])
            rob_val = stl_form.eval(agent_traj)
            rob_matrix[a_idx, t_idx] = float(np.array(rob_val).flatten()[0])

    # Solve assignment: each task offers m_per_task slots (spec-wide default, see _allocate_optimal);
    # agents are matched to the task they satisfy best, and task completion is then counted by the
    # CaTL+ formula with its per-task requirements.
    # Replicate tasks m_per_task times for the assignment
    n_slots = n_tasks * m_per_task
    cost_matrix = np.full((n_agents, n_slots), 1e6)
    for t_idx in range(n_tasks):
        for m in range(m_per_task):
            slot = t_idx * m_per_task + m
            cost_matrix[:, slot] = -rob_matrix[:, t_idx]  # negate for min-cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    agent_to_task = {}
    task_to_agents = {task.task_id: [] for task in team_spec.tasks}

    for a_idx, slot in zip(row_ind, col_ind):
        if slot < n_slots:
            t_idx = slot // m_per_task
            task_id = team_spec.tasks[t_idx].task_id
            agent_to_task[a_idx] = task_id
            task_to_agents[task_id].append(a_idx)

    unassigned = sorted(set(range(n_agents)) - set(agent_to_task.keys()))
    return AllocationResult(
        agent_to_task=agent_to_task,
        task_to_agents=task_to_agents,
        unassigned_agents=unassigned,
    )
