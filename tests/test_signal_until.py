"""Tests for the until-based Signal spec.

Intended behavior of ``<L>signal<G>`` (e.g. ``2signal3``):
  loop over G goals L times (cover), THEN reach a DISTINCT final goal, and the
  final goal only counts AFTER the L loops are up.

Built in ``STLMixin._load_signal_stl_form`` as:
    loop_form & (progress  U[loop_horizon, T]  final_goal)
where loop_form is a cover-`always` over the G loop goals (forces ~L covers in
[0, loop_horizon)), and the until requires the final goal in [loop_horizon, T).

Goal grid (in-distribution {0,2,4}^2), row-major letters:
    A=(0,0) B=(2,0) C=(4,0)
    D=(0,2) E=(2,2) F=(4,2)
    G=(0,4) H=(2,4) I=(4,4)
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from gcbfplus.env.wrapper.stl_mixin import STLMixin, SIGNAL_GOAL_LETTERS
from ds.stl_jax import STL, RectReachPredicate

GS = 1.0
SPEC = "m2signal3_t15"           # 2 loops, 3 loop goals, T forced to 15
PER_VISIT = 2                    # 15 // (2*3+1)
LOOP_HORIZON = 12                # num_loops * num_goals * per_visit
T = 15

# Controlled distinct goals for behavioral tests: loop A,B,D + far final I.
GOALS = {0: [0., 0.], 1: [2., 0.], 2: [0., 2.], 3: [4., 4.]}  # A,B,D,I


def _pred(i):
    return STL(RectReachPredicate(np.array(GOALS[i]), np.array([GS, GS]), i))


def _path(step_goals):
    """(1, T, 2) path; goal indices per step, padded with the last."""
    s = (step_goals + [step_goals[-1]] * T)[:T]
    return jnp.asarray(np.array([GOALS[i] for i in s], float))[None]


class TestSignalSpecBehavior(unittest.TestCase):
    """The spec must require: 2 covers, THEN a distinct final goal (late)."""

    def setUp(self):
        preds = [_pred(i) for i in range(4)]
        _, self.form, _ = STLMixin()._load_signal_stl_form(preds, preds, SPEC)

    def _rob(self, p):
        return float(self.form.eval(p, approx_method="true")[0])

    def test_two_loops_then_distinct_final_passes(self):
        # A,B,D twice in [0,12), then I in [12,15)
        p = _path([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 3, 3, 3])
        r = self._rob(p)
        print(f"\n[2loops+finalI] {r:+.3f}")
        self.assertGreater(r, 0.0)

    def test_only_looping_fails(self):
        # Loops the 3 goals forever, NEVER visits the distinct final I -> must FAIL.
        # (This is the bug the videos showed: looping satisfied a non-distinct g4.)
        p = _path([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 2, 2, 2])
        r = self._rob(p)
        print(f"[only-loop]     {r:+.3f}")
        self.assertLess(r, 0.0)

    def test_one_loop_then_final_fails(self):
        # One loop then I -> fails the 2-loop (loop_form) requirement.
        p = _path([0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        r = self._rob(p)
        print(f"[1loop+finalI]  {r:+.3f}")
        self.assertLess(r, 0.0)

    def test_final_only_early_fails(self):
        # Visits I early (during the loop window) but not after -> final must be late.
        p = _path([0, 0, 3, 1, 2, 2, 0, 0, 1, 1, 2, 2, 2, 2, 2])
        r = self._rob(p)
        print(f"[finalI-early]  {r:+.3f}")
        self.assertLess(r, 0.0)


class TestSignalGoalAssignment(unittest.TestCase):
    """Per-agent goal assignment must give 4 DISTINCT in-distribution goals with
    the final goal NOT in the loop."""

    def _agent_goals(self, agent_id):
        m = STLMixin()
        form, _ = m._load_diff_spec(SPEC, goal_list=m.DEFAULT_GOALS['empty'],
                                    agent_id=agent_id, key=jax.random.PRNGKey(0))
        by_name = {}
        for p in form.get_all_predicates():
            nm = getattr(p, 'name', None)
            if nm is not None and nm >= 0:
                by_name[nm] = tuple(np.array(p.cent).tolist())
        return [by_name[i] for i in sorted(by_name)][:4]  # g0,g1,g2,final

    def test_final_distinct_from_loop_all_agents(self):
        grid = set(SIGNAL_GOAL_LETTERS.keys())
        for aid in range(9):
            with self.subTest(agent=aid):
                g = self._agent_goals(aid)
                self.assertEqual(len(g), 4, "need loop(3)+final(1)")
                loop, final = g[:3], g[3]
                self.assertEqual(len(set(loop)), 3, "loop goals distinct")
                self.assertNotIn(final, loop, "final goal must NOT be in the loop")
                for pt in g:
                    self.assertIn(pt, grid, "all goals in-distribution {0,2,4}^2")

    def test_letter_mapping(self):
        self.assertEqual(SIGNAL_GOAL_LETTERS[(0., 0.)], 'A')
        self.assertEqual(SIGNAL_GOAL_LETTERS[(2., 2.)], 'E')
        self.assertEqual(SIGNAL_GOAL_LETTERS[(4., 4.)], 'I')
        self.assertEqual(len(set(SIGNAL_GOAL_LETTERS.values())), 9)


if __name__ == "__main__":
    unittest.main()
