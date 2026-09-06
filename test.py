import argparse
import datetime
import functools as ft
import os
import pathlib
import subprocess
import sys
from typing import Union

VERSION = "0.2.1"
try:
    _GIT_HASH = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=pathlib.Path(__file__).parent,
        stderr=subprocess.DEVNULL,
    ).decode().strip()
except Exception:
    _GIT_HASH = "unknown"

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import wandb
import yaml

from gcbfplus.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from gcbfplus.diffusion.planner import DiffusionMAPlanner
from gcbfplus.diffusion.util.args import pull_args_from_wandb, DIFFUSION_TRAINING_KEY, DIFFUSION_TESTING_KEY
from gcbfplus.env import make_env, BaseWrapper
from gcbfplus.env.base import RolloutResult
from gcbfplus.env.wrapper import STLWrapper, AsyncSTLWrapper
from gcbfplus.stl.utils import PLANNER_CONFIG, STL_INFO_KEYS, STL_PY_NAME, DIFFUSION_NAME, GLOBAL_STL_PY_NAME
from gcbfplus.trainer.utils import get_bb_cbf, rollout
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, jax_vmap, COMMENT_SEPARATOR, FLOAT_FORMAT, FLOAT_FORMAT_LESS, \
    get_git_diff, format_for_wandb


def test(args, wandb_run_id=None, plan_len=None, test_log_suffix="", test_debug_rollout=False,
         comment_suffix=""):
    """Evaluate a controller (optionally behind an STL planner) over ``args.epi`` episodes.

    Prints per-episode and aggregate safe/finish/success rates, and with ``--log`` appends
    one row to ``<path>/test_log<test_log_suffix>.csv`` (see :func:`print_metrics_and_log`).

    :param args: argparse args (see :func:`test_argparser`)
    :param wandb_run_id: diffusion checkpoint id; ``--wandb-run-id`` takes precedence
    :param plan_len: plan length for the diffusion planner; the checkpoint config overrides it
    :param test_log_suffix: suffix for the test log csv filename
    :param test_debug_rollout: return (env, get_bb_cbf_fn, rollout_fn, sample_rollout_fn)
        instead of running episodes (used to build the env-sync training losses)
    :param comment_suffix: additional comment string for logging
    """
    print(f"> Running test.py {args}")

    # Override the change-goal timing (read directly from PLANNER_CONFIG by the env
    # wrapper); no other CLI exposes it. Must happen before make_env().
    if getattr(args, "change_goal_immediately", False):
        PLANNER_CONFIG["change_goal_immediately"] = True
        print("Overriding PLANNER_CONFIG['change_goal_immediately'] = True (--change_goal_immediately)")

    # mixed_spec_mode is bound as a class attr of STLMixin at import (from
    # TRAINING_CONFIG['ds_params']); override the class attr before make_env.
    if getattr(args, "stl_mixed_spec_mode", None) is not None:
        from gcbfplus.env.wrapper.stl_mixin import STLMixin
        STLMixin.stl_mixed_spec_mode = args.stl_mixed_spec_mode
        print(f"Overriding STLMixin.stl_mixed_spec_mode = {args.stl_mixed_spec_mode} (--stl_mixed_spec_mode)")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"
        jax.config.update("jax_disable_jit", True)  # Disable JIT for good measure
    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)

    np.random.seed(args.seed)

    config = {}

    # load config
    if not args.u_ref and args.path is not None:
        print(f"Loading config from {args.path}")
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Environment related variables
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    wandb_run_id = args.wandb_run_id if args.wandb_run_id is not None else wandb_run_id

    # Spec related
    # Update params from config
    max_step = args.max_step
    params = {}
    planner = args.planner
    spec = args.spec
    spec_len = args.spec_len
    end_when_done = args.async_planner  # End when done for async planner
    planner_config = None  # Config specific to type of planner
    args_to_load = {}
    aux_loss_env = None
    aux_loss_rollout_fn = None

    if args.planner == 'diffusion':
        # Source the planner config from the denoiser checkpoint the run will actually load:
        # pull_args_from_wandb() reads diff_checkpoints/<run_id>/config.yaml when that
        # directory is self-contained (the bundled model), and only falls back to the wandb
        # run when it is absent. Among the pulled values is plan_len (the model's training
        # horizon); it overrides --spec-len below so evaluation matches how the model was
        # trained -- passing a different --spec-len to a diffusion run has no effect.
        manual_arg_loads = ['plan_len', 'wandb_run_id']
        pulled_args = pull_args_from_wandb(wandb_run_id=wandb_run_id)
        for arg in pulled_args:
            if arg in args:
                print(f"Overriding {arg} from checkpoint config {pulled_args[arg]} "
                      f"instead of given {getattr(args, arg)}")
                setattr(args, arg, pulled_args[arg])
            if arg in manual_arg_loads:
                print(f"Setting {arg} from checkpoint config : {pulled_args[arg]}")
                args_to_load[arg] = pulled_args[arg]
        planner_config = {'wandb_run_id': args_to_load.get('wandb_run_id'), 'diffusion_method': args.diffusion_method,
                          'agent_mask_noise': args.agent_mask_noise,
                          # Achievable guidance
                          'achievable_guidance': args.achievable_guidance,
                          'achievable_loss_coeff': args.achievable_loss_coeff,
                          'ach_guidance_after_fraction': args.ach_guidance_after_fraction,
                          'auxiliary_training_loss': args.auxiliary_training_loss,
                          'auxiliary_loss_rollout_length': args.auxiliary_loss_rollout_length,
                          'smooth_loss_coeff': args.smooth_loss_coeff,
                          'dense_achievable_supervision': args.dense_achievable_supervision,
                          'guidance_hardness': args.guidance_hardness,
                          'segment_stl_check': args.segment_stl_check,
                          'segment_stl_coeff': args.segment_stl_coeff,
                          'segment_interp_points': args.segment_interp_points,
                          'global_stl_only': args.global_stl_only,
                          # MA-STL specific
                          'ma_stl_loss_coeff': args.ma_stl_loss_coeff,
                          'ma_stl_alternate_guidance': args.ma_stl_alternate_guidance,
                          'ma_stl_alternate_every': args.ma_stl_alternate_every,
                          # Diffusion Sampling
                          'use_batched_sampling': args.use_batched_sampling,
                          'num_candidates': args.num_candidates,
                          'resample_batch_size': args.resample_batch_size,
                          'skip_resample': args.skip_resample,
                          }
        if args.auxiliary_group_agents_as is not None:
            num_grouped_agents = args.auxiliary_group_agents_as

            assert num_agents % num_grouped_agents == 0, f"Number of agents {num_agents} should be divisible by " \
                                                         f"num_grouped_agents {num_grouped_agents}"
            args_copy = argparse.Namespace(**vars(args))
            args_copy.num_agents = num_grouped_agents  # New environment with grouped agents
            args_copy.auxiliary_group_agents_as = None  # Reset group agents as to avoid infinite loop
            # Loading again to get all necessary params and functions matching the grouped agents
            aux_loss_env, _get_bb_cbf_fn, _unused_rollout_fn, aux_loss_rollout_fn = test(args_copy,
                                                                                         test_debug_rollout=True)

    # Set from loaded args if available
    wandb_run_id = args_to_load.get('wandb_run_id', wandb_run_id)
    plan_len = args_to_load.get('plan_len', plan_len)
    # Dummy episode (diffusion only) to exclude compile time from avg planning time. It consumes
    # i_epi=0, shifting which episodes are scored; --no-dummy-compile disables it so a diffusion run
    # scores the SAME i_epi (=> same starts and same --random-goals goals) as a stlpy run, for a fair
    # same-instance comparison.
    dummy_episode_to_compile = (args.planner == 'diffusion') and not args.no_dummy_compile

    test_key = jr.PRNGKey(args.seed)
    run_eps = (dummy_episode_to_compile + args.epi)
    test_keys = jr.split(test_key, 1_000)[: run_eps]
    test_keys = test_keys[args.offset:]
    if args.wandb_log:
        init_log_to_wandb(args, config.env if args.env is None else args.env)

    stl_wrapper = None
    if args.planner in ['stlpy', 'diffusion', 'stlpy_global']:
        max_step, spec_len, stl_wrapper = load_fixed_plan_wrapper(args, params, spec, planner_config=planner_config,
                                                                  plan_len=plan_len, key=test_key)
    else:
        print("Not using STL wrapper")

    if args.plot_plan_and_quit:
        print("Setting max_step to 1 for quick plotting of the plan")
        max_step = 1

    # create environments
    env = make_env(
        env_id=config.env if args.env is None else args.env,
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

    algo_is_cbf = isinstance(algo, (CentralizedCBF, DecShareCBF))

    env, get_bb_cbf_fn, rollout_fn, sample_rollout_fn = create_rollout_fn(act_fn, algo, args, end_when_done, env,
                                                                          nograph, preprocess_graph,
                                                                          grouped_env=aux_loss_env,
                                                                          grouped_rollout_fn=aux_loss_rollout_fn,
                                                                          max_step=max_step)
    if test_debug_rollout:
        return env, get_bb_cbf_fn, rollout_fn, sample_rollout_fn

    wandb_object_tracker = {}

    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    is_finishes = []
    rates = []
    cbfs = []
    # Per-episode traced goal layouts (centers/sizes), parallel to `rollouts`: with
    # --random-goals the wrapper's goal_centers hold only the LAST episode by video time
    goal_layouts = []
    finish_infos = []
    finish_info_str = ""
    info_metrics = []
    # planner is None for runs without --planner (e.g. raw CBF-QP baselines like
    # dec_share_cbf); fall back to the algo name so the label/logging path works.
    planner_str = (planner if planner is not None else (args.algo or 'none')) + ('_uref' if args.u_ref else '')
    # Order and arity must match the five trailing CSV columns of print_metrics_and_log's
    # header: planner, spec_len, spec, async_planner, team_alloc. create_dataset.py builds
    # the same list, so change both together.
    stl_info = f'{planner_str},{spec_len},{spec},{args.async_planner},{args.team_alloc}'.split(',')
    comments: Union[None, str] = None
    compilation_time = -1  # Compilation time for the first episode
    # (if optimally coded then only first episode should be slow)
    for i_epi in range(run_eps):
        # The dummy compile episode (i_epi=0) is a throwaway warm-up: it REUSES instance 0
        # (same start + same --random-goals goals as the first scored episode) so it only pays
        # the JIT compile, which is then excluded from every scored episode's planning time.
        # Scored episodes thus run instances 0..epi-1 -- identical to a stlpy run (which has no
        # dummy) -- so diffusion-vs-stlpy stays a fair same-instance comparison.
        eff_idx = max(0, i_epi - (1 if dummy_episode_to_compile else 0))
        key_x0, _ = jr.split(test_keys[eff_idx], 2)
        if args.random_goals:
            # Resample predicate locations from the whole area, in place: reuses the env and the
            # diffusion model/planner (no reconstruction), so only stl_forms + the forward recompile
            # repeat. rollout_fn closes over `env`, which we mutate, so the new goals are picked up at
            # reset. Per-episode key keeps goal layouts distinct yet reproducible across runs.
            if not hasattr(env, 'resample_goals'):
                raise ValueError("--random-goals requires an STL wrapper (use --planner diffusion/stlpy with a spec)")
            goal_key = jr.fold_in(jr.PRNGKey(args.random_goals_seed), eff_idx)
            env.resample_goals(goal_key, margin=args.random_goals_margin, traced=args.traced_goals,
                               spacing=args.random_goals_spacing,
                               region=args.random_goals_region, size_range=args.random_goals_size)

        finish_info_str, finish_rate, rollout, safe_rate, success_rate = rollout_and_log_metrics(args, env,
                                                                                                 finish_info_str,
                                                                                                 finish_infos,
                                                                                                 info_metrics,
                                                                                                 is_finishes,
                                                                                                 is_unsafes, key_x0,
                                                                                                 rollout_fn)
        epi_reward = rollout.T_reward.sum()
        epi_cost = rollout.T_cost.sum()

        team_note = " [team_spec: success=safe×(tasks/total)]" if hasattr(env, 'team_spec_obj') and env.team_spec_obj is not None else ""
        print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, "
              f"safe rate: {safe_rate * 100:.3f}%,"
              f"finish rate: {finish_rate * 100:.3f}%, "
              f"success rate: {success_rate * 100:.3f}%"
              f"{team_note}"
              f"{finish_info_str}")
        if i_epi == 0 and dummy_episode_to_compile:
            # This logs the planning + JIT compilation time for the first episode as part of the comments
            # Useful for understanding the compilation time of the diffusion planner
            print("NOTE: Skipping first episode for compilation costs")
            compilation_time = info_metrics[0]['plan_time']
            compilation_time_str = f"compilation_time:{compilation_time:4.4f}"
            # Drop this episode's metrics entirely: they never enter the CSV row or the
            # wandb averages (only its planning time survives, as the compilation_time comment).
            info_metrics.pop()
            is_finishes.pop()
            is_unsafes.pop()
            finish_infos.pop()

            if comments is None:
                comments = compilation_time_str
            else:
                comments = COMMENT_SEPARATOR.join([comments, compilation_time_str])
            continue

        # Log wandb result here per episode
        what2log_to_wandb = {'finish_rate': is_finishes[-1].max(axis=0).mean(),
                             'safe_rate': 1 - is_unsafes[-1].max(axis=0).mean(),
                             'success_rate': success_rate}
        if hasattr(env, 'team_spec_obj') and env.team_spec_obj is not None:
            what2log_to_wandb['team_spec_success_mode'] = True  # flag: success=safe×(tasks/total)
        if finish_infos:
            # Failure-channel fractions into per-episode wandb history (plots use history means)
            what2log_to_wandb.update({k: v for k, v in finish_infos[-1].items()
                                      if k.startswith('eval/fail_') or k == 'eval/plan_negative_rob_frac'})
        what2log_to_wandb.update(info_metrics[-1])
        log_to_wandb(args, env, wandb_log_file=what2log_to_wandb, summary={'compilation_time': compilation_time},
                     wandb_object_tracker=wandb_object_tracker)

        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)
        _viz_cents = getattr(env, 'goal_centers', None)
        if _viz_cents is None and getattr(env, 'stl_forms', None):
            # Fixed-goal runs: extract per-agent baked reach centers (spec-used goals
            # only, incl. shared first1/last1 goals) so the video overlay can label
            # the highlighted agent's goals just like traced runs.
            from gcbfplus.diffusion.util.stl_fastpath import extract_baked_goal_centers
            _viz_cents = extract_baked_goal_centers(env.stl_forms)
        goal_layouts.append((_viz_cents, getattr(env, 'goal_sizes', None)))

        if args.cbf is not None:
            cbfs.append(get_bb_cbf_fn(rollout.Tp1_graph))
        else:
            cbfs.append(None)

        rates.append(np.array([safe_rate, finish_rate, success_rate]))
    print_metrics_and_log(args, costs, env, finish_info_str, finish_infos, info_metrics, is_finishes, is_unsafes,
                          path, rewards, stl_info,
                          wandb_run_id=wandb_run_id, traj_len=plan_len, test_log_suffix=test_log_suffix,
                          comments=comments, comment_suffix=comment_suffix)

    # make video
    if args.no_video:
        return

    # For any random plotting (a key of its own: test_keys only holds the scored episodes)
    key_x0, _ = jr.split(jr.fold_in(test_key, run_eps + 1), 2)

    make_video(algo_is_cbf, args, cbfs, costs, env, is_unsafes, is_finishes, num_agents, path, rates, rewards, rollouts,
               stamp_str, step, stl_info, key=key_x0, goal_layouts=goal_layouts)


def make_video(algo_is_cbf, args, cbfs, costs, env, is_unsafes, is_finishes, num_agents, path, rates, rewards, rollouts,
               stamp_str, step, stl_info, key=None, goal_layouts=None):
    """Make video from rollouts"""
    videos_dir = pathlib.Path(path) / "videos"
    videos_dir.mkdir(exist_ok=True, parents=True)
    _, viz_opts = _set_viz_opts(args, "", None)
    for ii, (rollout, Ta_is_unsafe, Ta_is_finish, cbf) in enumerate(zip(rollouts, is_unsafes, is_finishes, cbfs)):
        key, _key = jr.split(key, 2)
        if algo_is_cbf or hasattr(env, 'planner'):
            safe_rate, finish_rate, success_rate = rates[ii] * 100
            video_name = f"n{num_agents}_epi{ii:02}_sr{safe_rate:.0f}_fr{finish_rate:.0f}_sr{success_rate:.0f}"
        else:
            video_name = f"n{num_agents}_step{step}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}"
        video_name += f"{','.join(stl_info)}"
        video_name, viz_opts = _set_viz_opts(args, video_name, cbf)
        if goal_layouts is not None and ii < len(goal_layouts) and goal_layouts[ii][0] is not None:
            # This episode's traced random goal layout (stl_forms keep stale baked centers)
            viz_opts['episode_goal_centers'] = np.asarray(goal_layouts[ii][0])
            if goal_layouts[ii][1] is not None:
                viz_opts['episode_goal_sizes'] = np.asarray(goal_layouts[ii][1])
        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        if args.plot_debug and (not (Ta_is_unsafe.astype(bool).any() or not Ta_is_finish.astype(bool).all())):
            print(f"Skipping video {video_path} as no unsafe states or  all states are finished")
            continue
        else:
            print(f"Rendering video with unsafe agents {jnp.where(Ta_is_unsafe.astype(bool))[1]} "
                  f"and unfinished agents {jnp.where(~Ta_is_finish.astype(bool))[1]} "
                  f"and speed factor {viz_opts.get('speed_factor', 1)}")
        env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi, key=_key)
        if args.wandb_log and wandb.run is not None and not args.plot_only_img:
            print(f"Uploading video {video_path} to wandb")
            wandb.log({f"rollouts/{video_name}": wandb.Video(str(video_path))})
        # If image exists with same name, log that too
        img_path = video_path.with_suffix('.png')
        if img_path.exists() and args.wandb_log and wandb.run is not None:
            print(f"Uploading image {img_path} to wandb")
            wandb.log({f"rollouts/{video_name}_img": wandb.Image(str(img_path))})


def _set_viz_opts(args, video_name=None, cbf=None):
    """Set viz options for video rendering from the arguments"""
    viz_opts = {}
    if args.cbf is not None:
        video_name += f"_cbf{args.cbf}"
        viz_opts["cbf"] = [*cbf, args.cbf]

    if args.highlight_agent is not None:
        # Parse comma-separated list of agent IDs
        highlight_agents = [int(agent_id) for agent_id in args.highlight_agent.split(',')]
        viz_opts['highlight_agents'] = highlight_agents

    if args.async_planner:
        viz_opts['async_planner'] = True
    if args.plot_snapshot:
        viz_opts['plot_snapshot'] = True
        if args.plot_pgf:
            viz_opts['plot_pgf'] = True
        if args.plot_svg:
            viz_opts['plot_svg'] = True
    if args.plot_only_img or args.plot_plan_and_quit:
        viz_opts['plot_only_img'] = True
    if args.plot_plan_and_quit:
        viz_opts['plot_plan_and_quit'] = True
    if args.video_speed_factor is not None:
        viz_opts['speed_factor'] = args.video_speed_factor

    if args.plot_debug:
        viz_opts['plot_debug'] = True
    return video_name, viz_opts


def load_fixed_plan_wrapper(args, params, spec, plan_len=None, planner_config=None,
                            key=None):
    """Load fixed plan wrapper for STL or diffusion planner

    :param args: argparse args
    :param params: planner params (common to all planners)
    :param spec: STL spec
    :type spec: str
    :param plan_len: plan length
    :type plan_len: int:
    :param planner_config: planner config (specific to planner such as wandb run id for diffusion)
    :type planner_config: dict
    :param key: random key
    :type key: jax.random.PRNGKey
    """
    # Seed `params` from default_config.yaml's planner_params. Note the values always come
    # from PLANNER_CONFIG, never from args -- a matching CLI flag only selects which keys get
    # copied. Of everything copied here the wrapper consumes exactly two below:
    # goal_sample_interval (steps between waypoint advances) and async_reach_radius (async
    # reach threshold); the rest ride along in `params` and are ignored by make_algo.
    param_keys = ['async_reach_radius']
    for k in PLANNER_CONFIG.keys():
        if k in args:
            print(f"Loading {k} from default_config.yaml planner_params: {PLANNER_CONFIG[k]}")
            params[k] = PLANNER_CONFIG[k]
        if k in param_keys:
            print(f"Loading {k} from default_config.yaml planner_params: {PLANNER_CONFIG[k]}")
            params[k] = PLANNER_CONFIG[k]
    if args.goal_sample_interval is not None:
        print(f"Setting goal_sample_interval to {args.goal_sample_interval} for planner ")
        params['goal_sample_interval'] = args.goal_sample_interval

    assert (args.spec_len is not None) or (plan_len is not None), "spec_len must be specified for STL planner"
    assert (args.spec is not None), "spec must be specified for STL planner"
    spec_len = plan_len if plan_len is not None else args.spec_len

    # Episode budget = goal_sample_interval * spec_len * max_step_factor. factor 1 is the
    # nominal plan duration (one waypoint per goal_sample_interval steps), which is all a
    # synchronous run needs; async execution advances waypoints only on reach, so it needs
    # slack, tuned per spec family below. Larger N gets a smaller factor (crowding is
    # penalised rather than waited out). --max-step overrides the whole computation.
    def get_max_step_factor(spec):
        """Tuned max step factor for different STL specs"""
        _factor = 5  # Default factor
        if 'loop' in spec:
            _factor = max(16, _factor)
        if 'signal' in spec:
            # signal = cover-loop(s) + a final goal, so it is loop-like and needs
            # at least as long a rollout budget as a loop spec.
            _factor = max(16, _factor)
        if 'reach' in spec:
            _factor = max(4, _factor)
        if 'seq' in spec:
            _factor = max(12, _factor)  # Was 10 for spec len 15
        if 'cover' in spec:
            _factor = max(8, _factor)
        if 'branch' in spec:
            _factor = max(10, _factor)
        if 'choice' in spec:
            _factor = max(12, _factor)  # Sequential tasks in OR specs
        if 'until' in spec:
            # Avoid-until plans are detour-shaped: waypoints advance slower than the
            # 1-per-gsi design rate, so the default budget truncates episodes (mauntil
            # was 12.5% success until run with the equivalent of factor 12; the
            # published expressive-stl cells passed --max-step 2400 explicitly).
            _factor = max(12, _factor)

        # Scale by number of agents for more leniency
        if args.num_agents is not None:
            def _scaling_function(x):
                """
                This function returns a scaled value based on the input x, with interpolation.
                - Returns 1 for an input of 32
                - Returns 1.5 for an input of 16
                - Returns 2 for an input of 8
                - Interpolates for other values.
                """
                # Define the known points
                xp = [8, 16, 32]
                fp = [1.5, 1.25, 1]

                # Use numpy's interpolation function
                return np.interp(x, xp, fp)

            _factor = int(_factor / _scaling_function(args.num_agents))
        # signal specs need a near-full rollout budget even at low N (N=8 should
        # reach ~100% success), so the N-scaling must not drop them below 16.
        if 'signal' in spec:
            _factor = max(_factor, 16)
        return _factor

    max_step_factor = get_max_step_factor(spec) if args.async_planner else 1
    max_step = int(params['goal_sample_interval'] * spec_len * max_step_factor)  # Longer max step for leniency
    print(f"Setting max_step to {max_step} for planner "
          f"= goal_sample_interval({params['goal_sample_interval']}) * "
          f"spec_len({spec_len}) * max_step_factor({max_step_factor})")
    if args.max_step is not None:
        max_step = int(args.max_step)
        print(f"Overriding max_step with --max-step {max_step}")
    if args.async_planner:
        print("Using async planner")
        stl_wrapper = AsyncSTLWrapper
    else:
        stl_wrapper = STLWrapper
    if args.planner == 'stlpy':
        planner_name = STL_PY_NAME
    elif args.planner == 'stlpy_global':
        planner_name = GLOBAL_STL_PY_NAME
    else:
        planner_name = DIFFUSION_NAME
    stl_wrapper = ft.partial(stl_wrapper, spec=args.spec, spec_len=spec_len, max_step=max_step,
                             goal_sample_interval=params['goal_sample_interval'],
                             async_reach_radius=params['async_reach_radius'],
                             stl_solver=planner_name,  # STL solver handles planner type
                             planner_config=planner_config,
                             key=key,
                             ma_stl_exp_rob=PLANNER_CONFIG['ma_stl']['exp_rob'],
                             skip_guidance=args.skip_guidance,
                             allocation_strategy=args.team_alloc,
                             team_disjunctive=args.team_disjunctive,
                             team_avoid=args.team_avoid,
                             avoid_expansion=args.avoid_expansion,
                             stlpy_u_bound=args.stlpy_u_bound)
    return max_step, spec_len, stl_wrapper


def rollout_and_log_metrics(args, env, finish_info_str, finish_infos, info_metrics, is_finishes, is_unsafes, key_x0,
                            rollout_fn):
    """Run a rollout and log metrics"""
    if args.nojit_rollout:
        rollout: RolloutResult
        rollout, is_unsafe, is_finish, rollout_info = rollout_fn(key_x0)
        is_unsafes.append(is_unsafe)
        is_finishes.append(is_finish)
        info_metrics.append(rollout_info)
        print('plan info', rollout_info)
    else:
        # Fully jit-ed rollout: it returns only the trajectory, so the per-step collision and
        # finish masks are evaluated here on the stacked (T+1) graph.
        if isinstance(env, STLWrapper):
            raise NotImplementedError(
                "An STL planner replans in Python between steps and cannot be traced: "
                "run planner evaluations with --nojit-rollout.")
        rollout: RolloutResult = rollout_fn(key_x0)
        is_unsafe_fn, is_finish_fn = _jit_eval_masks(env)
        is_unsafes.append(is_unsafe_fn(rollout.Tp1_graph))
        is_finishes.append(is_finish_fn(rollout.Tp1_graph))
        info_metrics.append({})  # no planner, so no plan info
    if isinstance(env, STLWrapper):
        # Get STL related satisfaction metrics (async wrappers also need the goal-change mask)
        if isinstance(env, AsyncSTLWrapper):
            finish_metrics = env.process_finished_rollouts_info()(rollout.Tp1_graph, rollout.T_info['changed_goal'])
        else:
            finish_metrics = env.process_finished_rollouts_info()(rollout.Tp1_graph)
        # Overwrite is_finishes with STL related satisfaction: on an STL env 'finish' means
        # the agent's spec was realized by the executed trajectory, not that it reached a goal,
        # so per-agent success = safe AND finished. For a team spec the wrapper reports the same
        # finish = tasks_completed / tasks_total for every agent, making success = safe x
        # (tasks / total); task_success_rate below is stricter still -- a task counts only if
        # every agent assigned to it stayed collision-free.
        is_finishes[-1] = jnp.array([finish_metrics[STL_INFO_KEYS[3]]])  # Shape appropriately
        # Filter out non-numeric values (per_task_completed, task_to_agents are dicts)
        numeric_metrics = {k: v for k, v in finish_metrics.items()
                           if not isinstance(v, (dict, bool))}
        finish_info = {f"eval/{k}": jnp.nanmean(v) for k, v in numeric_metrics.items()}
        finish_info_str = ', ' + ', '.join([f'{k}: {jnp.nanmean(v) :4.2f}' for k, v in numeric_metrics.items()])

        if finish_info:
            finish_infos.append(finish_info)

        # Build team_info for task_success_rate computation
        team_info = None
        if 'per_task_completed' in finish_metrics and 'task_to_agents' in finish_metrics:
            team_info = {
                'per_task_completed': finish_metrics['per_task_completed'],
                'task_to_agents': finish_metrics['task_to_agents'],
                'tasks_total': finish_metrics.get('tasks_total', 1),
            }

        # Vary the safety calculation depending on the mode (also change the is_unsafes to reflect different ep lengths)
        safe_rate, finish_rate, success_rate, is_unsafes[-1], task_success_rate = env.calc_safety_success_finish(
            is_unsafes[-1], is_finishes[-1], rollout,
            ignore_on_finish=args.ignore_on_finish, team_info=team_info)

        # Add task_success_rate to finish_info for logging
        if task_success_rate is not None:
            finish_info['eval/task_success_rate'] = task_success_rate

        # Failure-channel decomposition (STL violation vs collision vs
        # tracking error vs resampling exhaustion). Per-agent, exclusive by priority:
        # collision > resample-exhaustion (delivered plan already violates the spec at
        # plan time) > tracking (plan satisfies the spec, execution failed to realize
        # it). 'finish' is realized-STL satisfaction, so 1 - finish = the STL-violation
        # umbrella that channels 2+3 decompose.
        plan_rob = env.score_plan_robustness() if hasattr(env, 'score_plan_robustness') else None
        if plan_rob is not None and finish_info:
            agent_unsafe = np.asarray(is_unsafes[-1]).max(axis=0).astype(bool).reshape(-1)
            agent_finish = np.asarray(is_finishes[-1]).max(axis=0).astype(bool).reshape(-1)
            plan_bad = np.asarray(plan_rob).reshape(-1) < 0
            n = min(agent_unsafe.shape[0], agent_finish.shape[0], plan_bad.shape[0])
            u, f, b = agent_unsafe[:n], agent_finish[:n], plan_bad[:n]
            finish_info['eval/fail_collision_frac'] = float(u.mean())
            finish_info['eval/fail_resample_exhaustion_frac'] = float((~u & ~f & b).mean())
            finish_info['eval/fail_tracking_frac'] = float((~u & ~f & ~b).mean())
            finish_info['eval/plan_negative_rob_frac'] = float(b.mean())
    else:
        # Regular finish metrics
        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        finish_rate = is_finishes[-1].max(axis=0).mean()
        success_rate = ((1 - is_unsafes[-1].max(axis=0)) * is_finishes[-1].max(axis=0)).mean()
    return finish_info_str, finish_rate, rollout, safe_rate, success_rate


def init_log_to_wandb(args, env_name):
    """Initialize logging test results to wandb along with git diff and config diff"""
    algo = args.algo if args.algo is not None else os.path.basename(
        args.path.strip('/')) if args.path is not None else "u-ref"
    wandb.init(  # EDIT: Add your own wandb project name
        project="gcbfplus-stl-test",  # EDIT: your wandb project name
        name=f"{env_name}_{algo}_{args.planner}_{args.spec}_{args.spec_len}_N{args.num_agents}_Nepi{args.epi}",
        tags=getattr(args, 'wandb_tags', None),
    )
    # GCBF_* env-var knobs are invisible in args — record them in config so runs are
    # self-describing and filterable (goal scale, sampler-arm flags, gate thresholds).
    _env_knobs = {f"env_{k.lower()}": os.environ[k] for k in
                  ('GCBF_GOAL_SCALE', 'GCBF_TEAM_SELECT_OUTER', 'GCBF_MA_ACCEPT_THRESH',
                   'GCBF_TASK_REPAIR', 'GCBF_TASK_REPAIR_GRAD', 'GCBF_KEEP_BEST',
                   'GCBF_MA_ACCEPT')
                  if k in os.environ}
    if _env_knobs:
        wandb.config.update(_env_knobs, allow_val_change=True)

    # Ignore per episode metrics
    metrics_to_skip_summary = ['num_resampling_iters', 'plan_time', 'stl_loss', 'max_num_resampling_iters',
                               'stl_loss_pre_hack', 'finish_rate', 'safe_rate']
    for metric in metrics_to_skip_summary:
        wandb.define_metric(metric, summary="none")

    wandb.config.update(args)

    # Get git diff and config diff and log to wandb
    diff_str, diff_config = get_git_diff()

    with open("git_diff.patch", "w") as f:
        f.write(diff_str)

    with open("config_diff.patch", "w") as f:
        f.write(diff_config)

    artifact = wandb.Artifact("git_diff", type="diff")
    artifact.add_file("git_diff.patch")
    artifact.add_file("config_diff.patch")
    wandb.log_artifact(artifact)


def log_to_wandb(args, env, wandb_log_file: dict = None, summary: dict = None,
                 wandb_object_tracker: dict = None):
    """Log to wandb with any summary and per episode logs"""
    if args.wandb_log and (wandb.run is not None):
        if summary is not None:
            # Any summary metrics to log like compilation time
            wandb.summary.update(summary)

        if wandb_log_file is not None:
            # Per episode logs
            # Move array to table
            filtered_wandb_log_file = {}
            filtered_wandb_log_tables = {} if wandb_object_tracker is None else wandb_object_tracker
            for k, v in wandb_log_file.items():
                if (isinstance(v, np.ndarray) or isinstance(v, jnp.ndarray)) and len(v.shape) >= 1:
                    if k in wandb_object_tracker:
                        # If the key is already in the tracker, append to it
                        l = v.tolist()
                        filtered_wandb_log_tables[k].add_data(*l)
                        # Reinitialize the table with the new data
                        table = wandb.Table(data=filtered_wandb_log_tables[k].data,
                                            columns=[f"{k}_{i}" for i in range(len(l))])
                        filtered_wandb_log_tables[k] = table

                    else:
                        l = v.tolist()
                        table_row = l
                        table = wandb.Table(data=[table_row], columns=[f"{k}_{i}" for i in range(len(l))])
                        filtered_wandb_log_tables[k] = table

                elif isinstance(v, str) and 'plan_time' in k:
                    # Convert plan time to float
                    filtered_wandb_log_file[k] = float(v)
                elif isinstance(v, jax.Array):
                    filtered_wandb_log_file[k] = v.item()
                else:
                    filtered_wandb_log_file[k] = v
            wandb.log(dict(filtered_wandb_log_file, **filtered_wandb_log_tables))


def print_metrics_and_log(args, costs, env, finish_info_str, finish_infos, info_metrics, is_finishes, is_unsafes, path,
                          rewards, stl_info, wandb_run_id=None, traj_len=None, test_log_suffix="", comments=None,
                          comment_suffix=""):
    """Print metrics and log to csv if required

    :param args: argparse args
    :param costs: list of costs
    :param env: environment
    :param finish_info_str: finish info string
    :param finish_infos: list of finish infos
    :param info_metrics: list of info metrics
    :param is_finishes: list of is_finishes
    :param is_unsafes: list of is_unsafes
    :param path: path to save log
    :param rewards: list of rewards
    :param stl_info: STL info
    :param wandb_run_id: wandb run id
    :param traj_len: trajectory length
    :param test_log_suffix: test log suffix
    :param comments: comments to add (optional)
    """

    # is_unsafes may have different lengths, stack them to get the max over all
    is_unsafe = np.stack(list(map(lambda x: x.max(axis=0), is_unsafes)))
    is_finish = np.max(np.stack(is_finishes), axis=1)
    success_matrix = (1 - is_unsafe) * is_finish
    if PLANNER_CONFIG["mean_over_trajectories"]:
        # Get the mean over all trajectories
        is_unsafe = is_unsafe.mean(axis=-1)
        is_finish = is_finish.mean(axis=-1)
        success_matrix = success_matrix.mean(axis=-1)

    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
    finish_mean, finish_std = is_finish.mean(), is_finish.std()
    success_mean, success_std = success_matrix.mean(), success_matrix.std()
    # Get the mean of a list of dictionaries finish_metrics if they exist
    final_finish_metrics = {}
    if finish_infos:
        # Add extra summary metrics here
        summary_metrics = {'is_unsafe': is_unsafe, 'is_finish': is_finish}
        if isinstance(env, BaseWrapper):
            # Add extra metrics from the environment like strict ma-stl satisfaction checking for collision
            finish_infos, extra_metrics = env.extra_summary_metrics(summary_metrics=summary_metrics,
                                                                    finish_infos=finish_infos)
        final_finish_metrics_mean = {f"{k}": jnp.nanmean(jnp.stack([v[k] for v in finish_infos])) for k in
                                     finish_infos[0].keys()}
        final_finish_metrics_std = {f"{k}_std": jnp.nanstd(jnp.stack([v[k] for v in finish_infos])) for k in
                                    finish_infos[0].keys()}
        final_finish_metrics = {**final_finish_metrics_mean, **final_finish_metrics_std}
        finish_info_str = ', Mean scores: ' + ', '.join(
            [f'{k}: {v :4.2f}' for k, v in final_finish_metrics_mean.items()])
    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%, "
        f"finish_rate: {finish_mean * 100:.3f}%, "
        f"success_rate: {success_mean * 100:.3f}%"
        f"{finish_info_str}"
    )
    # save results
    if args.log:
        from gcbfplus.stl.utils import PLANNER_INFO_KEYS_TO_KEEP, COMMON_PLANNER_INFO_KEYS
        # calculate mean and std of keys of list of dicts info_metrics using numpy
        info_metrics = [{k: v for k, v in info.items() if k in PLANNER_INFO_KEYS_TO_KEEP} for info in info_metrics]
        info_metrics_mean = {f"{k}_mean": f"{jnp.nanmean(jnp.stack([v[k] for v in info_metrics])):{FLOAT_FORMAT}}" for k
                             in
                             info_metrics[0].keys()}
        info_metrics_std = {f"{k}_std": f"{jnp.nanstd(jnp.stack([v[k] for v in info_metrics])):{FLOAT_FORMAT}}" for k in
                            info_metrics[0].keys()}
        info_metrics = {**info_metrics_mean, **info_metrics_std}
        # Now separate the keys into common and specific keys (move to comments)
        comment_info_metrics = {k: v for k, v in info_metrics.items() if
                                not any(map(lambda x: x in k, COMMON_PLANNER_INFO_KEYS))}
        info_metrics = {k: v for k, v in info_metrics.items() if any(map(lambda x: x in k, COMMON_PLANNER_INFO_KEYS))}

        # consistent ordering of keys and vals
        info_keys = list(sorted(info_metrics))
        info_vals = [info_metrics[k] for k in info_keys]

        # Schema contract for test_log*.csv. The plotting bundle reads these columns BY NAME
        # (scripts/plot_team_spec_results.py, scripts/release/plot_reproduced.py and
        # gcbfplus/utils/spec_bar_figure.py), and a changed header renames the existing file
        # (below) instead of appending to it, so keep the names and the column order stable.
        # The five entries after the STL_INFO_KEYS blocks are filled from `stl_info`.
        header = ["num_agents", "epi", "max_step", "area_size", "n_obs",
                  "safe_mean", "safe_std", "finish_mean",
                  "finish_std", "success_mean", "success_std"] + STL_INFO_KEYS + [f"{k}_std" for k in STL_INFO_KEYS] + [
                     "planner", "spec_len", "spec", "async_planner", "team_alloc"] + info_keys + ["comments"]
        csv2write = os.path.join(path, f"test_log{test_log_suffix}.csv")

        # An existing file whose header differs is archived rather than appended to
        header_changed = False
        if os.path.exists(csv2write):
            with open(csv2write, "r") as f:
                existing_header = f.readline().strip().split(',')
                if existing_header != header:
                    header_changed = True
                    # Rename old file with timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    old_file = os.path.join(path, f"test_log{test_log_suffix}_old_{timestamp}.csv")
                    os.rename(csv2write, old_file)
                    print(f"Header schema changed. Renamed old file to: {old_file}")

        if not os.path.exists(csv2write) or header_changed:
            # write header
            with open(csv2write, "w") as f:
                f.write(','.join(header) + '\n')

        with (open(csv2write, "a") as f):
            # Add extra info which may not be consistent across all runs as comments
            comments_list = [f"version:{VERSION}@{_GIT_HASH}"]
            if comments is not None:
                comments_list += [comments]
            if wandb_run_id is not None and args.planner == 'diffusion':
                # Add wandb run id and trajectory length to comments if provided (easy to track diffusion model used)
                comments_list += [f"wandb_run_id: {wandb_run_id}", f"traj_len: {traj_len}",
                                  f"diffusion_method: {args.diffusion_method}"]
                if hasattr(env, 'planner') and env.planner.planner_info is not None:
                    # Planner info like guidance used
                    comments_list += [env.planner.planner_info]
            if hasattr(env, 'extra_run_info') and env.extra_run_info is not None:
                # Goal sample interval and other info
                comments_list += [env.extra_run_info]
            if comment_info_metrics:
                # Add any extra info from the planner
                comments_list += [f"{k}: {v}" for k, v in comment_info_metrics.items()]
            if args.team_alloc and args.team_alloc != 'greedy':
                comments_list += [f"team_alloc:{args.team_alloc}"]
            if hasattr(env, 'team_spec_obj') and env.team_spec_obj is not None:
                comments_list += ["team_spec_success_mode:True"]
            if args.team_disjunctive:
                comments_list += ["team_disjunctive:True"]
            if args.team_avoid:
                comments_list += ["team_avoid:True"]
            if getattr(args, 'global_stl_only', False):
                comments_list += ["global_stl_only:True"]
            if comment_suffix:
                # Add comment suffix if provided (account for leading and trailing quotes)
                comments_list += [comment_suffix]
            if args.wandb_log and wandb.run is not None:
                # Add wandb run id to comments if provided
                comments_list += [f"test_wandb_run_id: {wandb.run.id}", f"wandb_run_url: {wandb.run.get_url()}"]

            # Consolidate all comments
            comments = COMMENT_SEPARATOR.join(comments_list).replace(' ', '')
            if not comments:
                comments = "None"
            else:
                # Add quotes to comments if provided
                comments = f'"{comments}"'
            info_to_write = [f'{env.num_agents}', f'{args.epi}',
                             f'{env.max_episode_steps}',
                             f'{env.area_size}', f'{env.params["n_obs"]}', f'{safe_mean * 100:.3f}',
                             f'{safe_std * 100:.3f}', f'{finish_mean * 100:.3f}', f'{finish_std * 100:.3f}',
                             f'{success_mean * 100:.3f}', f'{success_std * 100:.3f}']
            # Loop over STL_INFO_KEYS for consistent ordering

            stl_info_metrics = (
                    [f'{final_finish_metrics[f"eval/{k}"] :{FLOAT_FORMAT_LESS}}' if f"eval/{k}" in final_finish_metrics
                     else 'NaN' for k in STL_INFO_KEYS] +
                    [f'{final_finish_metrics[f"eval/{k}_std"] :{FLOAT_FORMAT_LESS}}' if f"eval/{k}_std" in final_finish_metrics
                     else 'NaN' for k in STL_INFO_KEYS] + stl_info + info_vals)
            line_to_write = (','.join(info_to_write + stl_info_metrics + [comments]) + '\n').replace(' ', '')
            f.write(line_to_write)

        if args.wandb_log and wandb.run is not None:
            # means wandb.init() succeeded
            final_log2wandb(comment_suffix, env, final_finish_metrics, finish_mean, finish_std, header, info_metrics,
                            info_to_write, safe_mean, safe_std, stl_info_metrics, success_mean, success_std,
                            test_log_suffix)


def final_log2wandb(comment_suffix, env, final_finish_metrics, finish_mean, finish_std, header, info_metrics,
                    info_to_write,
                    safe_mean, safe_std, stl_info_metrics, success_mean, success_std, test_log_suffix):
    """Log final config and summary metrics to wandb"""

    def keep_key_in_summary(k):
        # Prevent logging of all keys in summary
        return (k in ['compilation_time']) or k.startswith('eval/') or k.endswith('_mean') or k.endswith('_std')

    summary_metric_keys = ['safe_mean', 'safe_std', 'finish_mean', 'finish_std', 'success_mean', 'success_std']
    summary_metric_vals = map(float, [safe_mean, safe_std, finish_mean, finish_std, success_mean, success_std])
    main_summary_metrics = dict(zip(summary_metric_keys, summary_metric_vals))
    all_row_entries = ",".join(info_to_write + stl_info_metrics).split(',')
    # Log to wandb without duplicates
    wandb_config = {k: v for k, v in zip(header, all_row_entries) if
                    not keep_key_in_summary(k) and (k not in info_metrics.keys()) and (
                            f"eval/{k}" not in final_finish_metrics.keys())}
    # Add extra config dict to wandb config like async_reach_radius
    extra_config_dict = getattr(env, 'extra_run_info_full_dict', {})
    wandb_config.update(extra_config_dict)
    if isinstance(env.planner, DiffusionMAPlanner):
        # Add default config dict to wandb config
        # This may have default values which were updated in the config
        config_to_update = {}
        for k, v in env.planner.planner_all_info_dict.items():
            if k not in wandb_config and k not in wandb.config:
                # Only add the keys which are not in the wandb config
                config_to_update[k] = v
            if k in [DIFFUSION_TRAINING_KEY, DIFFUSION_TESTING_KEY]:
                # Separate the training and testing config dicts
                inner_config_to_update = {}
                for kk, vv in v.items():
                    if kk not in wandb_config and kk not in wandb.config:
                        inner_config_to_update[kk] = vv
                config_to_update[k] = inner_config_to_update
        wandb_config.update(config_to_update)
    wandb.config.update(format_for_wandb(wandb_config), allow_val_change=True)
    info_metrics_correct_format = {k: float(v) for k, v in info_metrics.items()}
    tag_information = {}
    if len(comment_suffix) > 0:
        tag_information['comment_suffix'] = comment_suffix
    if len(test_log_suffix) > 0:
        tag_information['test_log_suffix'] = test_log_suffix
        wandb.run.tags = wandb.run.tags + (test_log_suffix,)
    summary_dict = {**info_metrics_correct_format, **final_finish_metrics,
                    **main_summary_metrics, **tag_information}
    keys_to_remove = [k for k in summary_dict.keys() if not keep_key_in_summary(k)]
    summary_dict = {k: v for k, v in summary_dict.items() if not (k in keys_to_remove)}
    # Add final summary metrics to wandb
    wandb.summary.update(summary_dict)


@ft.lru_cache(maxsize=None)
def _jit_eval_masks(env):
    """jit-ed (collision, finish) masks over a stacked rollout graph, for the jit-rollout path.

    ``env.rollout_fn`` returns only the trajectory (unlike ``rollout_fn_jitstep``, which also
    returns the per-step masks), so they are evaluated afterwards on ``rollout.Tp1_graph``.
    Cached per env object so every episode reuses the same compilation.
    """
    return jax_jit_np(jax_vmap(env.collision_mask)), jax_jit_np(env.process_finished_rollouts())


def create_rollout_fn(act_fn, algo, args, end_when_done, env, nograph, preprocess_graph,
                      max_step=None, grouped_env=None, grouped_rollout_fn=None):
    """Helper function to create JIT-ed rollout function"""

    if args.cbf is not None:
        assert isinstance(algo, GCBF) or isinstance(algo, GCBFPlus) or isinstance(algo, CentralizedCBF)
        get_bb_cbf_fn_ = ft.partial(get_bb_cbf, algo.get_cbf, env, agent_id=args.cbf, x_dim=0, y_dim=1)
        get_bb_cbf_fn_ = jax_jit_np(get_bb_cbf_fn_)

        def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h
    else:
        get_bb_cbf_fn = None

    # Create reset args (needed for achievable loss)
    reset_args = {}
    sample_rollout_fn = None
    if args.diffusion_method == 'edm-ma':
        # Make a (differentiable) rollout fn to be used in the diffusion phase

        def sample_rollout_fn(graph, max_length, sample_key):
            """Helper function compatible with vmap"""
            return rollout(env=env,
                           actor=lambda graph, k: (
                               act_fn(graph), None),
                           key=sample_key,
                           preprocess_graph=ft.partial(preprocess_graph, stop_grad=False),
                           single_goal=True,
                           init_graph=graph,
                           max_length=max_length)

        if grouped_env is not None and grouped_rollout_fn is not None:
            # Load env in minibatch form
            print(f"Using grouped rollout fn for env {grouped_env}")
            reset_args['rollout_fn'] = grouped_rollout_fn
            reset_args['env'] = grouped_env
        else:
            reset_args['rollout_fn'] = sample_rollout_fn
            reset_args['env'] = env

    if args.nojit_rollout:
        print("Only jit step, no jit rollout!")
        rollout_fn = env.rollout_fn_jitstep(act_fn, max_step, noedge=True, nograph=nograph,
                                            preprocess_graph=preprocess_graph,
                                            end_when_done=end_when_done,
                                            reset_args=reset_args)
    else:
        print("jit rollout!")
        rollout_fn = jax_jit_np(env.rollout_fn(act_fn, args.max_step, preprocess_graph=preprocess_graph))
    return env, get_bb_cbf_fn, rollout_fn, sample_rollout_fn


def test_argparser():
    """Parse test arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=0)
    parser.add_argument("--area-size", type=float, required=True)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--cbf", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--u-ref", action="store_true", default=False)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--video-speed-factor", type=float, default=4.0, help="Video speed multiplier")
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--test-log-suffix", type=str, default="",
                        help="Suffix appended to the test_log CSV filename (before .csv). "
                             "Give each run a unique suffix when several runs append to the "
                             "same directory.")
    parser.add_argument("--dpi", type=int, default=100)
    # Spec and planner related
    parser.add_argument('--spec', type=str, default=None)  # Use loaded config unless specified
    parser.add_argument('--spec-len', type=int, default=None)
    parser.add_argument("--goal-sample-interval", type=int, default=None,
                        help="Number of steps between sampling the planner")
    parser.add_argument('--planner', type=str, default=None, help="If specified, sets the planner to use",
                        choices=['stlpy', 'diffusion', 'stlpy_global'])
    parser.add_argument("--async-planner", action="store_true", default=False, help="Use async planner")
    parser.add_argument('--ignore-on-finish', action='store_true', default=False,
                        help="Ignore safety on finish for STL")
    # Random predicate locations per episode (sampled from the whole area, not the fixed grid).
    # Reuses the env+planner and only resamples goals in place, so the diffusion model/network is
    # never reconstructed (only the lightweight stl_forms + forward recompile). Tests whether the
    # diffusion planner generalizes to arbitrary predicate placements rather than a fixed grid.
    parser.add_argument("--random-goals", action="store_true", default=False,
                        help="Resample predicate/goal positions uniformly in the area each episode")
    parser.add_argument("--random-goals-seed", type=int, default=0,
                        help="Base seed for --random-goals (per-episode key = fold_in(seed, episode))")
    parser.add_argument("--random-goals-margin", type=float, default=None,
                        help="Keep random goals this far from the area border (default: goal_size)")
    parser.add_argument("--random-goals-spacing", type=float, default=None,
                        help="With --traced-goals: place each agent's loop goals as a rigid randomly "
                             "rotated/translated copy of the fixed signal demo's L-shape with this side "
                             "length (=2 matches the {0,2,4} grid). Preserves the demo's exact inter-goal "
                             "spacing {s,s,s*sqrt2} (no artificial grouping).")
    parser.add_argument("--no-dummy-compile", action="store_true", default=False,
                        help="Disable the diffusion dummy compile episode so the scored i_epi range "
                             "matches a stlpy run (identical starts + --random-goals goals) for fair comparison.")
    parser.add_argument("--random-goals-region", type=float, default=None,
                        help="With --random-goals-spacing: confine all agents' loop anchors AND finals "
                             "to a central box of this half-width, so different agents' clusters overlap "
                             "(induces crowding/coordination). Loop spacing stays rigid. Smaller = more crowded.")
    parser.add_argument("--random-goals-size", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="With --random-goals: give each goal rectangle its own per-episode size "
                             "= goal_size * U(LO, HI), e.g. 0.5 1.5. Tests generalization to predicate "
                             "SIZE, not just location. Works traced (sizes flow with centers as a "
                             "dynamic (cents, sizes) override -> zero recompile; uniform sampling only) "
                             "and non-traced (sizes baked into the AST -> per-episode recompile).")
    parser.add_argument("--traced-goals", action="store_true", default=False,
                        help="With --random-goals: feed goal centers as a dynamic cent_override and "
                             "keep one structural stl_forms, so changing the goal layout does NOT "
                             "recompile the diffusion forward (compiles once). diff-spec Approach A.")
    # Diffusion related
    parser.add_argument(
        # Must match the checkpoint being loaded (the bundled model is edm-ma)
        "--diffusion-method", type=str, default="edm-ma", choices=["edm", "edm-ma"], help="Diffusion method")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="Wandb run id for diffusion method")

    # Extra debug options
    parser.add_argument("--plot-snapshot", action="store_true", default=False, help="Plot trajectory snapshot")
    parser.add_argument("--plot-pgf", action="store_true", default=False, help="Plot trajectory snapshot in pgf")
    parser.add_argument("--plot-svg", action="store_true", default=False, help="Plot trajectory snapshot as svg")
    parser.add_argument("--plot-debug", action="store_true", default=False, help="Plot any debug info")
    parser.add_argument("--plot-only-img", action="store_true", default=False, help="Plot only images and skip video")
    parser.add_argument("--plot-plan-and-quit", action="store_true", default=False,
                        help="Plot plan and quit, no rollout")
    parser.add_argument("--highlight-agent", type=str, default=None,
                        help="Comma-separated list of agent IDs to highlight with circles (e.g., '5,1')")
    parser.add_argument("--debug-nans", action="store_true", default=False, help="Debug nans")

    parser.add_argument("--wandb-log", action="store_true", default=False,
                        help="Log inference results to wandb along with .patch file")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=None,
                        help="Tags for the wandb run, to separate experiment groups (e.g. random-size)")
    parser.add_argument("--change_goal_immediately", action="store_true", default=False,
                        help="Advance to the next goal immediately on reach (overrides "
                             "PLANNER_CONFIG['change_goal_immediately']) instead of waiting for the interval")
    parser.add_argument("--stl_mixed_spec_mode", type=str, default=None,
                        help="Override ds_params.mixed_spec_mode (e.g. first1/last1/None) — which goals "
                             "are common across agents in a mixed spec. Default: keep default_config.yaml value.")
    parser.add_argument("--team-alloc", type=str, default='greedy',
                        choices=['greedy', 'random', 'greedy_last', 'oracle_hungarian'],
                        help="Allocation strategy for team specs "
                             "(greedy/random/greedy_last/oracle_hungarian)")
    parser.add_argument("--team-disjunctive", action="store_true", default=False,
                        help="Give every agent the same disjunctive spec (OR over tasks) "
                             "so CaTL+ gradient discovers allocation during diffusion sampling")
    parser.add_argument("--team-avoid", action="store_true", default=False,
                        help="Add avoidance of other tasks' non-shared goals to per-agent STL specs")
    parser.add_argument("--avoid-expansion", type=float, default=1.2,
                        help="Expansion factor for avoid regions (1.0=same as goal, 1.2=20%% larger)")
    parser.add_argument("--stlpy-u-bound", type=float, default=None,
                        help="STLPy control bound per plan step. Default: default_config.yaml "
                             "planner_params.stlpy_u_bound (2.0). "
                             "Set to 20 for original (unconstrained) behavior.")

    from gcbfplus.diffusion.util.args import parser_add_ma_args
    #  Add arguments for EDM-MA planner
    parser_add_ma_args(parser)
    return parser


def test_args(cmd_args=sys.argv[1:]):
    parser = test_argparser()
    args, rest_args = parser.parse_known_args(cmd_args)
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")
    return args


def main():
    args = test_args()
    # Fast CLI check for unsupported team-spec + random-goals combinations (the wrapper
    # raises the same errors later, after model load — this fails in milliseconds instead).
    if args.random_goals and args.spec and str(args.spec).startswith('team'):
        if args.traced_goals:
            raise ValueError("--traced-goals is not supported with team specs "
                             "(CaTL+ team objective does not receive cent_override)")
        if args.random_goals_size is not None:
            raise ValueError("--random-goals-size is not supported with team specs "
                             "(team formula builders ignore goal_size_factors)")
    test(args, test_log_suffix=args.test_log_suffix)


if __name__ == "__main__":
    main()
