import os

import jax
# Spec related
import jax.random as jr
import numpy as np
import yaml

from gcbfplus.algo import make_algo
from gcbfplus.diffusion.environments.create_dataset_helpers import RolloutToDataset
from gcbfplus.env import make_env
from gcbfplus.env.wrapper import STLWrapper
from test import create_rollout_fn, test_args, print_metrics_and_log, rollout_and_log_metrics, load_fixed_plan_wrapper


def create_dataset(args):
    print(f"> Running create_dataset.py {args}")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)  # Disable JIT for good measure
    np.random.seed(args.seed)

    config = {}

    # load config
    if not args.u_ref and args.path is not None:
        print(f"Loading config from {args.path}")
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Spec related
    # Update params from config
    max_step = args.max_step
    params = {}
    planner = args.planner
    spec = args.spec
    spec_len = args.spec_len
    end_when_done = args.async_planner  # End when done for async planner

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    stl_wrapper = None
    if args.planner in ['stlpy', 'diffusion', 'stlpy_global']:
        max_step, spec_len, stl_wrapper = load_fixed_plan_wrapper(args, params, spec, key=test_key)
    else:
        print("Not using STL wrapper")

    # create environments
    env_id = config.env if args.env is None else args.env
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=env_id,
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=max_step,
        max_travel=args.max_travel,
        wrapper_fn=stl_wrapper,
    )

    # No need to save graph if no rollout metrics
    nograph = args.no_video and not isinstance(env, STLWrapper)

    if not args.u_ref:
        if args.path is not None:
            path = args.path
            model_path = os.path.join(path, "models")
            if args.step is None:
                models = os.listdir(model_path)
                step = max([int(model) for model in models if model.isdigit()])
            else:
                step = args.step
            print("step: ", step)

            algo = make_algo(
                algo=config.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                gnn_layers=config.gnn_layers,
                batch_size=config.batch_size,
                buffer_size=config.buffer_size,
                horizon=config.horizon,
                lr_actor=config.lr_actor,
                lr_cbf=config.lr_cbf,
                alpha=config.alpha,
                eps=0.02,
                inner_epoch=8,
                loss_action_coef=config.loss_action_coef,
                loss_unsafe_coef=config.loss_unsafe_coef,
                loss_safe_coef=config.loss_safe_coef,
                loss_h_dot_coef=config.loss_h_dot_coef,
                max_grad_norm=2.0,
                seed=config.seed,
                params=params,
            )
            algo.load(model_path, step)
            act_fn = jax.jit(algo.act)
            preprocess_graph = jax.jit(algo.preprocess_graph)
        else:
            algo = make_algo(
                algo=args.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                alpha=args.alpha,
            )
            act_fn = jax.jit(algo.act)
            preprocess_graph = jax.jit(algo.preprocess_graph)
            path = os.path.join(f"./logs/{args.env}/{args.algo}")
            if not os.path.exists(path):
                os.makedirs(path)
            step = None
    else:
        assert args.env is not None
        path = os.path.join(f"./logs/{args.env}/nominal")
        if not os.path.exists("./logs"):
            os.mkdir("./logs")
        if not os.path.exists(os.path.join("./logs", args.env)):
            os.mkdir(os.path.join("./logs", args.env))
        if not os.path.exists(path):
            os.mkdir(path)
        algo = None
        preprocess_graph = lambda x: x  # No preprocess for nominal
        act_fn = jax.jit(env.u_ref)
        step = 0

    env, _get_bb_cbf_fn, rollout_fn, _sample_rollout_fn = create_rollout_fn(act_fn, algo, args, end_when_done, env,
                                                                            nograph,
                                                                            preprocess_graph)

    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    is_finishes = []
    finish_infos = []
    finish_info_str = ""
    info_metrics = []
    planner_str = planner + ('_uref' if args.u_ref else '')
    # Same five fields, in the same order, as test.py's stl_info -- they fill the trailing
    # planner/spec_len/spec/async_planner/team_alloc columns of print_metrics_and_log's header.
    stl_info = f'{planner_str},{spec_len},{spec},{args.async_planner},{args.team_alloc}'.split(',')
    for i_epi in range(args.epi):

        key_x0, _ = jr.split(test_keys[i_epi], 2)
        finish_info_str, finish_rate, rollout, safe_rate, success_rate = rollout_and_log_metrics(args, env,
                                                                                                 finish_info_str,
                                                                                                 finish_infos,
                                                                                                 info_metrics,
                                                                                                 is_finishes,
                                                                                                 is_unsafes, key_x0,
                                                                                                 rollout_fn)
        epi_reward = rollout.T_reward.sum()
        epi_cost = rollout.T_cost.sum()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)

        print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, "
              f"safe rate: {safe_rate * 100:.3f}%,"
              f"finish rate: {finish_rate * 100:.3f}%, "
              f"success rate: {success_rate * 100:.3f}%"
              f"{finish_info_str}")

    print_metrics_and_log(args, costs, env, finish_info_str, finish_infos, info_metrics, is_finishes, is_unsafes, path,
                          rewards, stl_info)
    r2dataset = RolloutToDataset(rollouts, env_id=env_id, predicate_centers=env.GOAL_SET, env=env)
    r2dataset.rollouts_to_d4rl_dataset()


def main():
    args = test_args()  # Same as test.py
    create_dataset(args)


if __name__ == "__main__":
    main()
