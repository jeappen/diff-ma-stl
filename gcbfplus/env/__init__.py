import jax
from typing import Optional, Callable, Tuple

from .base import MultiAgentEnv
from .crazyflie import CrazyFlie
from .double_integrator import DoubleIntegrator
from .dubins_car import DubinsCar
from .linear_drone import LinearDrone
from .single_integrator import SingleIntegrator
from .wrapper import BaseWrapper

ENV = {
    'SingleIntegrator': SingleIntegrator,
    'DoubleIntegrator': DoubleIntegrator,
    'LinearDrone': LinearDrone,
    'DubinsCar': DubinsCar,
    'CrazyFlie': CrazyFlie,
}

DEFAULT_MAX_STEP = 256


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
        wrapper_fn: Optional[BaseWrapper] = None,
        load_map: Optional[Tuple[str, Callable[[jax.random.PRNGKey], None]]] = None,

) -> MultiAgentEnv:
    """ Create environment instance with given parameters.

    :param env_id: Environment ID.
    :param num_agents: Number of agents.
    :param area_size: Size of the area.
    :param max_step: Maximum number of steps.
    :param max_travel: Maximum travel distance.
    :param num_obs: Number of obstacles.
    :param n_rays: Number of rays in LiDAR sensor.
    :param wrapper_fn: Wrapper function.
    :param load_map: Optional (map name, fn(key) -> (key, obs_pos, obs_len, obs_theta)) custom obstacle map.
                     No built-in map ships with this release and no release command sets it, so it is
                     None everywhere and obstacles are sampled randomly.
    """
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = ENV[env_id].PARAMS
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        params['n_obs'] = num_obs
    if n_rays is not None:
        params['n_rays'] = n_rays
    env_obj = ENV[env_id](
        num_agents=num_agents,
        area_size=area_size,
        max_step=max_step,
        max_travel=max_travel,
        dt=0.03,
        params=params,
        load_map=load_map,
    )

    if wrapper_fn is not None:
        env_obj = wrapper_fn(env_obj)
    return env_obj
