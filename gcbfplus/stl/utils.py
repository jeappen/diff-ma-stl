import os
import yaml

import gcbfplus.utils.configs as gcbf_module_configs

"""Explanation of STL info keys below:

- max_path_score: Maximum STL score of the paths
- min_path_score: Minimum STL score of the paths
- mean_path_score: Mean STL score of the paths
- finish_rate: Fraction of paths that satisfy the STL constraint
- TtR: Time to reach the STL constraint
- ma_stl_score: Score of the multi-agent STL constraint (applicable when using a multi-agent STL like CaTL+)
- ma_stl_satisfaction: Fraction of paths that satisfy the multi-agent STL constraint
- strict_ma_stl_satisfaction: Fraction of paths above that have no collisions
- plan_time: Time taken to plan the path
- max_num_resampling_iters: Maximum number of diffusion resampling iterations (different for each agent)
- *_traj_plan_deviation: distance between the executed trajectory and the plan (TRAJECTORY_DEVIATION_KEYS)
- tasks_completed / tasks_total / task_rate / task_success_rate: team-spec task counting (TEAM_INFO_KEYS)
"""
# Trajectory deviation metrics
TRAJECTORY_DEVIATION_KEYS = ['mean_traj_plan_deviation', 'max_traj_plan_deviation', 'std_traj_plan_deviation', 'cumulative_traj_plan_deviation']
# Team spec metrics
TEAM_INFO_KEYS = ['tasks_completed', 'tasks_total', 'task_rate', 'task_success_rate']
# Keys for STL info to be logged
STL_INFO_KEYS = ['max_path_score', 'min_path_score', 'mean_path_score', 'finish_rate', 'TtR', 'ma_stl_score',
                 'ma_stl_satisfaction', 'strict_ma_stl_satisfaction'] + TRAJECTORY_DEVIATION_KEYS + TEAM_INFO_KEYS
COMMON_PLANNER_INFO_KEYS = ['plan_time']  # Keys for planner info o be logged (for consistency across planners)
PLANNER_INFO_KEYS_TO_KEEP = ['plan_time', 'max_num_resampling_iters']  # Keys to keep in planner info (for logging)

STL_PY_NAME = "stlpy"
GLOBAL_STL_PY_NAME = "stlpy_global"
DIFFUSION_NAME = "diffusion"

LARGE_HARDNESS = 100

DEFAULT_TtR = -1.0
STL_EVAL_TOLERANCE = 2 * 1e-2  # Tolerance for satisfying STL constraints
MA_STL_EVAL_TOLERANCE = - 1e-1  # Tolerance for satisfying STL constraints


def load_yaml(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def load_config(currentpath: str = None) -> dict:
    config = yaml.safe_load(open(os.path.join(
        *((list(gcbf_module_configs.__path__) if currentpath is None else currentpath) + ['default_config.yaml'])),
        'r'))
    training_config = config['training_params']
    planner_config = config['planner_params']
    env_config = config['env_params']

    return {'training': training_config, 'planner': planner_config, 'env': env_config}


CONFIGS = load_config()
TRAINING_CONFIG = CONFIGS['training']
PLANNER_CONFIG = CONFIGS['planner']
ENV_CONFIG = CONFIGS['env']
