"""Team-level STL specification parsing and data structures.

Supports:
- team_cover<N>      : N goals, each must be reached by any 1 agent
- team_<M>of<N>      : N goals, at least M must be reached (strategic partial completion)
- team_redun<N>x<M>  : N goals, each must be reached by M agents (redundant coverage; paper "Redundant")
- team_choiceseq3    : two-branch OR spec, each branch a 3-stage sweep + support task (paper "Choice")
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TeamTask:
    """A single task in a team specification."""
    task_id: int
    goal_centers: list  # list of np.ndarray; len=1 for cover/redundant, len>1 for choiceseq3 sweeps


@dataclass
class TeamSpec:
    """Parsed team specification."""
    spec_type: str           # 'team_cover', 'team_mofn', 'team_redundant', 'team_choiceseq3'
    tasks: list              # list of TeamTask
    time_horizon: int
    n_agents: int
    m_required: Optional[int] = None   # For team_mofn: how many tasks must be completed
    m_per_task: int = 1                # For team_redundant: agents required per task
    task_mode: str = 'sequential'      # 'sequential' (ordered time windows); 'cover' (any order) is not produced by any release spec
    branches: Optional[list] = None    # For OR specs: list of lists of task_ids per branch
                                       # e.g. [[0,1], [2,3]] means (Task0∧Task1) | (Task2∧Task3)
    task_m_override: Optional[dict] = None  # Per-task m override: {task_id: m_value}
                                            # If set, overrides m_per_task for specific tasks


# Regex patterns — match the stl_string format: {spec}_t{spec_len}
TEAM_COVER_REGEX = re.compile(r"team_cover(\d+)_t(\d+)")
TEAM_MOFN_REGEX = re.compile(r"team_(\d+)of(\d+)_t(\d+)")
TEAM_REDUN_REGEX = re.compile(r"team_redun(\d+)x(\d+)_t(\d+)")
TEAM_CHOICESEQ3_REGEX = re.compile(r"team_choiceseq3_t(\d+)")


def is_team_spec(spec_string: str) -> bool:
    """Check if a spec string is a team specification."""
    return any(rx.match(spec_string) is not None for rx in
               (TEAM_COVER_REGEX, TEAM_MOFN_REGEX, TEAM_REDUN_REGEX, TEAM_CHOICESEQ3_REGEX))


def parse_team_spec(spec_string: str, n_agents: int, goal_set: list) -> TeamSpec:
    """Parse a team spec string into a TeamSpec dataclass.

    Args:
        spec_string: e.g. "team_cover4_t400", "team_3of6_t15", "team_redun4x2_t15"
        n_agents: number of agents in the environment
        goal_set: list of goal positions [[x,y], ...] to draw from

    Returns:
        TeamSpec with parsed tasks and parameters
    """
    cover_match = TEAM_COVER_REGEX.match(spec_string)
    if cover_match:
        n_tasks = int(cover_match.group(1))
        time_horizon = int(cover_match.group(2))

        if n_tasks > len(goal_set):
            raise ValueError(
                f"team_cover{n_tasks} requires {n_tasks} goals but goal_set has {len(goal_set)}")

        tasks = []
        for q in range(n_tasks):
            tasks.append(TeamTask(
                task_id=q,
                goal_centers=[np.array(goal_set[q], dtype=np.float64)],
            ))

        return TeamSpec(
            spec_type='team_cover',
            tasks=tasks,
            time_horizon=time_horizon,
            n_agents=n_agents,
        )

    mofn_match = TEAM_MOFN_REGEX.match(spec_string)
    if mofn_match:
        m_required = int(mofn_match.group(1))
        n_goals = int(mofn_match.group(2))
        time_horizon = int(mofn_match.group(3))

        if n_goals > len(goal_set):
            raise ValueError(
                f"team_{m_required}of{n_goals} requires {n_goals} goals but goal_set has {len(goal_set)}")
        if m_required > n_goals:
            raise ValueError(
                f"team_{m_required}of{n_goals}: m_required ({m_required}) > n_goals ({n_goals})")

        tasks = []
        for q in range(n_goals):
            tasks.append(TeamTask(
                task_id=q,
                goal_centers=[np.array(goal_set[q], dtype=np.float64)],
            ))

        return TeamSpec(
            spec_type='team_mofn',
            tasks=tasks,
            time_horizon=time_horizon,
            n_agents=n_agents,
            m_required=m_required,
        )

    redun_match = TEAM_REDUN_REGEX.match(spec_string)
    if redun_match:
        n_goals = int(redun_match.group(1))
        m_per_task = int(redun_match.group(2))
        time_horizon = int(redun_match.group(3))

        if n_goals > len(goal_set):
            raise ValueError(
                f"team_redun{n_goals}x{m_per_task} requires {n_goals} goals but goal_set has {len(goal_set)}")
        if n_goals * m_per_task > n_agents:
            raise ValueError(
                f"team_redun{n_goals}x{m_per_task} requires {n_goals * m_per_task} agent slots "
                f"but only {n_agents} agents available")

        tasks = []
        for q in range(n_goals):
            tasks.append(TeamTask(
                task_id=q,
                goal_centers=[np.array(goal_set[q], dtype=np.float64)],
            ))

        return TeamSpec(
            spec_type='team_redundant',
            tasks=tasks,
            time_horizon=time_horizon,
            n_agents=n_agents,
            m_per_task=m_per_task,
        )

    choiceseq3_match = TEAM_CHOICESEQ3_REGEX.match(spec_string)
    if choiceseq3_match:
        return _parse_choiceseq3_spec(choiceseq3_match, n_agents, goal_set)

    raise ValueError(f"Unrecognized team spec: {spec_string}")


def _parse_choiceseq3_spec(match, n_agents: int, goal_set: list) -> TeamSpec:
    """Parse team_choiceseq3_tT spec.

    DNF spec: (A0 ∧ A1) | (B0 ∧ B1)

    Two branches, each with a main sweep task (3N/4 agents, 3 goals) and a
    support task (N/4 agents, 2 goals). All agents are assigned — no idle agents.
    Cross-branch avoidance enforces spatial separation (top vs bottom half).
    Goal indices below refer to DEFAULT_GOALS['empty'] order (GCBF_GOAL_SCALE preserves it;
    random goal locations do not, hence the guard in resample_goals).

    CaTL+ formula:
      (Task_A0(m=3N/4) ∧ Task_A1(m=N/4)) | (Task_B0(m=3N/4) ∧ Task_B1(m=N/4))

    Branch A (top):
      A0: F[0,T/3] Reach([2,2]) & F[T/3,2T/3] Reach([0,4]) & F[2T/3,T] Reach([4,4])
      A1: F[0,T/2] Reach([2,2]) & F[T/2,T] Reach([2,4])
      Cross-branch avoid: G Avoid([4,0]), G Avoid([0,0]), G Avoid([2,0])

    Branch B (bottom):
      B0: F[0,T/3] Reach([2,2]) & F[T/3,2T/3] Reach([4,0]) & F[2T/3,T] Reach([0,0])
      B1: F[0,T/2] Reach([2,2]) & F[T/2,T] Reach([2,0])
      Cross-branch avoid: G Avoid([0,4]), G Avoid([4,4]), G Avoid([2,4])

    Goal set layout:
      idx0=[0,0] idx2=[2,0] idx5=[4,0]
      idx3=[0,2] idx1=[2,2] idx7=[4,2]
      idx6=[0,4] idx8=[2,4] idx4=[4,4]

    N/4 agents per support task creates diversity within each branch — not all
    agents cluster at the same goals. DIFF-MA coordinates the split via CaTL+
    gradient. STLPY must pre-allocate both tasks within the chosen branch.
    """
    import math
    time_horizon = int(match.group(1))

    if len(goal_set) < 9:
        raise ValueError(
            f"team_choiceseq3 requires at least 9 goals but goal_set has {len(goal_set)}")

    m_main = math.ceil(3 * n_agents / 4)
    m_support = max(1, n_agents // 4)
    gathering = np.array(goal_set[1], dtype=np.float64)  # [2,2]

    tasks = [
        # Branch A (top)
        # A0: main sweep — [2,2] → [0,4] → [4,4] (gather → NW → NE)
        TeamTask(task_id=0, goal_centers=[
            gathering,
            np.array(goal_set[6], dtype=np.float64),   # [0,4] NW
            np.array(goal_set[4], dtype=np.float64),    # [4,4] NE
        ]),
        # A1: support — [2,2] → [2,4] (gather → top-center)
        TeamTask(task_id=1, goal_centers=[
            gathering,
            np.array(goal_set[8], dtype=np.float64),    # [2,4] top-center
        ]),
        # Branch B (bottom)
        # B0: main sweep — [2,2] → [4,0] → [0,0] (gather → SE → SW)
        TeamTask(task_id=2, goal_centers=[
            gathering,
            np.array(goal_set[5], dtype=np.float64),    # [4,0] SE
            np.array(goal_set[0], dtype=np.float64),    # [0,0] SW
        ]),
        # B1: support — [2,2] → [2,0] (gather → bottom-center)
        TeamTask(task_id=3, goal_centers=[
            gathering,
            np.array(goal_set[2], dtype=np.float64),    # [2,0] bottom-center
        ]),
    ]

    return TeamSpec(
        spec_type='team_choiceseq3',
        tasks=tasks,
        time_horizon=time_horizon,
        n_agents=n_agents,
        m_per_task=m_main,  # default for main tasks
        branches=[[0, 1], [2, 3]],  # (A0∧A1) | (B0∧B1)
        task_m_override={1: m_support, 3: m_support},  # support tasks need fewer agents
    )
