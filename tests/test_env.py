import functools as ft
import unittest

from gcbfplus.env.wrapper import AsyncSTLWrapper


class TestEnvLoading(unittest.TestCase):
    """Tests for loading the environment with the async STL wrapper."""
    env_name = 'DubinsCar'
    num_agents = 1
    area_size = 4
    # Rest Arbitrary
    spec_len = 18
    max_step = 1000

    def test_loading_env(self):
        """Tests valid loading of the environment for seq2..seq9 and mseq2..mseq9."""
        for i in range(2, 10):
            for spec in ['seq', 'mseq']:
                with self.subTest(spec=f'{spec}{i}'):
                    env = self._load_env(AsyncSTLWrapper, f'{spec}{i}')
                    self.assertTrue(env is not None)

    def _load_env(self, stl_wrapper=None, spec=None):
        """Helper function to load the environment."""
        if stl_wrapper is not None and spec is not None:
            part_stl_wrapper = ft.partial(stl_wrapper, spec=spec, spec_len=self.spec_len, max_step=self.max_step,
                                          goal_set_args={'dont_shuffle': True})
        else:
            part_stl_wrapper = None
        from gcbfplus.env import make_env
        env = make_env(
            env_id=self.env_name,
            num_agents=self.num_agents,
            area_size=self.area_size,
            max_step=self.max_step,
            wrapper_fn=part_stl_wrapper,
        )

        return env


if __name__ == '__main__':
    unittest.main()
