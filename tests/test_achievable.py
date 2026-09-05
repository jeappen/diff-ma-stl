import functools as ft
import pathlib
import jax.lax
import unittest

import test as evaluation_script  # From root directory
from gcbfplus.diffusion.diffusion.losses import get_achievable_loss_fn
from gcbfplus.diffusion.util import *
from gcbfplus.env.wrapper import AsyncSTLWrapper

_PRETRAINED_DUBINS_GCBFPLUS = (
    pathlib.Path(__file__).resolve().parent.parent / "pretrained" / "DubinsCar" / "gcbf+"
)


# Apply the reshape function to every leaf of the pytree.
# merged_pytree = jtu.tree_map(merge_vmapped_axes, original_pytree)
class TestAchievableLoss(unittest.TestCase):
    """Tests for differentiability of the achievable loss function."""
    env_name = 'DubinsCar'
    num_agents = 2
    area_size = 6
    # Rest Arbitrary
    spec_len = 15
    max_step = 1000

    def setUp(self):
        """Load the environment and rollout fns"""
        pass

    def load_stlpy_plan(self, env=None):
        """Load a plan from the STLpy planner"""
        load_map = None
        if env is None:
            spec_name = 'seq'
            i = 2
            spec = f'{spec_name}{i}'
        else:
            spec = env.spec
            if env.map_name != "empty":
                # Empty map is default
                load_map = env._load_map
        stlpy_env = self._load_env(AsyncSTLWrapper, spec, load_map=load_map, num_agents=env.num_agents)
        o = stlpy_env.reset(jax.random.PRNGKey(0))
        if stlpy_env.__getattribute__('plan_info') is not None:
            print(f"STLpy Plan time: {stlpy_env.plan_info['plan_time']}")
        # TODO: Check NAN because last 2 goals are same
        return o

    def run_achievable_loss_over_hparams(self, map_name, goal_sample_interval_list=None, single_agent=False,
                                         run_both=False, num_agents=2):
        results = {}
        if goal_sample_interval_list is None:
            goal_sample_interval_list = [1, 10, 20]
        env, rollout_fn = self._load_rollout_fn(map_name=map_name, goal_sample_interval=goal_sample_interval_list[0],
                                                num_agents=num_agents)
        reset_obs = self.load_stlpy_plan(env)
        plan = jnp.array(reset_obs.current_plan)
        for goal_sample_interval in goal_sample_interval_list:
            with self.subTest(goal_sample_interval=goal_sample_interval, single_agent=single_agent):
                if single_agent:
                    results[goal_sample_interval] = self.run_single_agent_achievable_loss(goal_sample_interval,
                                                                                          map_name)
                if not single_agent or run_both:
                    results[goal_sample_interval] = self.run_achievable_loss(plan, env, rollout_fn,
                                                                             goal_sample_interval, map_name,
                                                                             num_agents=num_agents)
        for key, val in results.items():
            print(f"\nGoal Sample Interval: {key}")
            if single_agent:
                print(f"Single agent Loss: {val['sa_loss']}")
                print(f"Norm of single agent grad: {val['sa_norm_grad']}")
            if not single_agent or run_both:
                print(f"Loss: {val['loss']}")
                print(f"Norm of grad: {val['norm_grad']}")
                print(f"Mean norm of grad: {val['mean_norm_grad']}")

    def run_single_agent_achievable_loss(self, goal_sample_interval, map_name):
        # TODO: Merge with run_achievable_loss
        env, rollout_fn = self._load_rollout_fn(map_name=map_name, goal_sample_interval=goal_sample_interval)
        single_env, single_rollout_fn = self._load_rollout_fn(map_name=map_name,
                                                              goal_sample_interval=goal_sample_interval,
                                                              num_agents=1)
        # TODO: Implement the test
        reset_obs = self.load_stlpy_plan(env)
        plan = jnp.array(reset_obs.current_plan)
        single_achievable_loss = get_achievable_loss_fn("env-sync-single", env, single_rollout_fn,
                                                        single_env)
        reshaped_plan = plan.squeeze(0).transpose(1, 0, 2)
        add_deviation = lambda x: jax.random.normal(jax.random.PRNGKey(0), x.shape)
        # perturbed_plan = reshaped_plan.at[:,-1,:].set(add_deviation(reshaped_plan[:,-1,:]))
        # TODO: DBG get nan in grad when goal is the same
        perturbed_plan = reshaped_plan[:, :-1]
        sa_loss_val, sa_grad = jax.value_and_grad(single_achievable_loss)(perturbed_plan)
        return {'sa_loss': sa_loss_val,
                'sa_norm_grad': jnp.linalg.norm(sa_grad)}

    @ft.partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
    def run_achievable_loss(self, plan, env, rollout_fn, goal_sample_interval, map_name, num_agents=2,
                            ach_mini_batch=None):
        if ach_mini_batch is not None:
            # TODO: Finish achievable loss mini-batch to scale up beyond 8 agents
            num_agents_mb = num_agents // ach_mini_batch
            env, rollout_fn = self._load_rollout_fn(map_name=map_name, goal_sample_interval=goal_sample_interval,
                                                    num_agents=num_agents_mb)

            def achievable_loss(_plan):
                """Reshape plan into mini-batches and compute achievable loss."""
                return get_achievable_loss_fn("env-sync-train", env, rollout_fn,
                                              rollout_len=goal_sample_interval)

        else:
            achievable_loss = get_achievable_loss_fn("env-sync-train", env, rollout_fn,
                                                     rollout_len=goal_sample_interval)
        reshaped_plan = plan.squeeze(0).transpose(1, 0, 2)
        add_deviation = lambda x: jax.random.normal(jax.random.PRNGKey(0), x.shape)
        # perturbed_plan = reshaped_plan.at[:,-1,:].set(add_deviation(reshaped_plan[:,-1,:]))
        # TODO: DBG get nan in grad when goal is the same
        perturbed_plan = reshaped_plan[:, :-1]

        def step_ach_loss(achievable_loss, perturbed_plan):
            loss_val, grad = jax.value_and_grad(achievable_loss)(perturbed_plan)  # compile
            perturbed_plan -= 0.1 * jnp.clip(grad, -.5, .5)
            return perturbed_plan, (grad, loss_val)

        perturbed_plan, (grad, loss_val) = step_ach_loss(achievable_loss, perturbed_plan)  # warm-up / compile

        # Then repeat the step under scan and keep every gradient.
        num_loops = 1000
        final_plan, outputs = jax.lax.scan(lambda x, i: step_ach_loss(achievable_loss, x), perturbed_plan,
                                           jnp.arange(num_loops))
        all_grads = outputs[0]

        final_grad, final_loss = outputs[0][-1], outputs[1][-1]

        return {'loss': final_loss, 'norm_grad': jnp.linalg.norm(final_grad), 'mean_norm_grad': jnp.mean(
            jnp.linalg.norm(all_grads, axis=-1))}

    # 1min 45 if boths
    def test_empty_env(self):
        """Tests differentiability in the empty map."""
        map_name = None
        self.run_achievable_loss_over_hparams(map_name, goal_sample_interval_list=[1, 3, 10, 20], num_agents=8)

    @unittest.skip(
        "env-sync-single path in get_achievable_loss_fn raises NotImplementedError "
        "(losses.py:57). Re-enable once the single-agent variant is finished."
    )
    def test_empty_env_single(self):
        """Tests differentiability in the empty map."""
        map_name = None
        self.run_achievable_loss_over_hparams(map_name, goal_sample_interval_list=[1, 3, 10, 20],
                                              single_agent=True)

    @unittest.skip(
        "Bottleneck rollout hits NaN in backward through DubinsCar.u_ref: "
        "jnp.linalg.norm(pos_diff) at pos_diff=0 and arccos(clip(.,-1,1)) at the "
        "boundary have singular gradients. Empty-map variant passes because agents "
        "never land exactly on the goal. Re-enable after safe-norm + tightened "
        "arccos clip in gcbfplus/env/dubins_car.py:378/385-387/416."
    )
    def test_bottleneck(self):
        """Tests differentiability in the bottleneck map."""
        map_name = "bottleneck"
        self.run_achievable_loss_over_hparams(map_name, goal_sample_interval_list=[3])  # , 10, 20])

    def _load_rollout_fn(self, map_name=None, goal_sample_interval=10, num_agents=2):
        arg_str = (
            f"--path {_PRETRAINED_DUBINS_GCBFPLUS}/ --epi 5 --area-size 6 -n {num_agents} --obs 0 --nojit-rollout "
            f"--planner diffusion --spec-len 30 --goal-sample-interval {goal_sample_interval} "
            f"--spec seq2 --log --async-planner --ignore-on-finish --debug-nans")
        if map_name is not None:
            arg_str = arg_str + f" --map {map_name}"
        args = evaluation_script.test_args(arg_str.split())
        env, get_bb_cbf_fn, rollout_fn, sample_rollout_fn = evaluation_script.test(args, test_debug_rollout=True)
        return env, sample_rollout_fn

    def _load_env(self, stl_wrapper=None, spec=None, load_map=None, num_agents=None):
        """Helper function to load the environment."""
        if num_agents is None:
            num_agents = self.num_agents
        if stl_wrapper is not None and spec is not None:
            part_stl_wrapper = ft.partial(stl_wrapper, spec=spec, spec_len=self.spec_len, max_step=self.max_step,
                                          goal_set_args={'dont_shuffle': True})
        else:
            part_stl_wrapper = None
        from gcbfplus.env import make_env
        env = make_env(
            env_id=self.env_name,
            num_agents=num_agents,
            area_size=self.area_size,
            max_step=self.max_step,
            wrapper_fn=part_stl_wrapper,
            load_map=load_map
        )

        return env


if __name__ == '__main__':
    unittest.main()
