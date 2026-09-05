"""Ground-truth tests for the traced ``cent_override`` path in ds.stl_jax.

The traced-goals optimisation feeds goal coordinates into the STL evaluation as a
*dynamic* ``cent_override`` array instead of baking them into the (jit-static)
predicate leaves, so a new goal layout does not retrigger compilation. These tests
verify the evaluation stays numerically correct:

  (1) override with the SAME cents == baked eval        (backward compatibility)
  (2) override with NEW cents == rebuilding the form with those cents (substitution)
  (3) holds in both eval modes (train_mode False/True)
  (4) sign sanity on a ground-truth path (visit-all-goals > 0, sit-far < 0)

Covers the loop spec structure (G(F g0 & F g1 & F g2)) and a single reach predicate.
"""
import unittest

import numpy as np
import jax.numpy as jnp

from ds.stl_jax import STL, RectReachPredicate
from gcbfplus.stl.utils import ENV_CONFIG

GOAL_SIZE = np.array([1, 1]) * ENV_CONFIG["goal_size"]
TOL = 1e-5


def _pred(cent, name, shrink=1.0):
    return STL(RectReachPredicate(np.array(cent, dtype=float), np.array(GOAL_SIZE), name,
                                  shrink_factor=shrink))


def _build_loop(cents, time_int=15, num_loops=2, num_goals=3, shrink=1.0):
    """Mirror _load_loop_stl_form: G[0,(L-1)*per]( F[0,per] g0 & ... & F[0,per] g_{n-1} )."""
    preds = [_pred(c, i, shrink) for i, c in enumerate(cents)]
    per_loop = time_int // num_loops
    form = preds[0].eventually(0, per_loop)
    for g in preds[1:num_goals]:
        form = form & g.eventually(0, per_loop)
    form = form.always(0, max(1, (num_loops - 1) * per_loop))
    return form


def _cycle_path(cents, T=15):
    """Path that visits cents[0], cents[1], cents[2], 0,1,2, ... -> every window covers all goals."""
    pts = [cents[t % len(cents)] for t in range(T)]
    return jnp.asarray(np.array(pts, dtype=float))[None]  # (1, T, 2)


class TestCentOverride(unittest.TestCase):
    C1 = [[1.0, 1.0], [3.0, 1.0], [2.0, 3.0]]
    C2 = [[0.5, 3.5], [3.5, 0.5], [2.0, 2.0]]

    def _check_equiv(self, train_mode):
        form1 = _build_loop(self.C1)
        form2 = _build_loop(self.C2)
        ov1 = jnp.asarray(np.array(self.C1))
        ov2 = jnp.asarray(np.array(self.C2))
        # Use a non-trivial fixed path (cycle through C1 goals) for both forms.
        path = _cycle_path(self.C1)

        baked1 = np.asarray(form1.eval(path, train_mode=train_mode))
        baked2 = np.asarray(form2.eval(path, train_mode=train_mode))
        over_same = np.asarray(form1.eval(path, train_mode=train_mode, cent_override=ov1))
        over_c2 = np.asarray(form1.eval(path, train_mode=train_mode, cent_override=ov2))

        # (1) override with same cents reproduces baked eval exactly.
        np.testing.assert_allclose(over_same, baked1, atol=TOL, rtol=TOL,
                                   err_msg=f"override-same != baked (train_mode={train_mode})")
        # (2) override with C2 on form1's structure == baked form2 (pure substitution).
        np.testing.assert_allclose(over_c2, baked2, atol=TOL, rtol=TOL,
                                   err_msg=f"override-C2 != baked form2 (train_mode={train_mode})")

    def test_equivalence_eval(self):
        self._check_equiv(train_mode=False)

    def test_equivalence_eval_train(self):
        self._check_equiv(train_mode=True)

    def test_ground_truth_signs(self):
        form = _build_loop(self.C1)
        ov = jnp.asarray(np.array(self.C1))
        hit = _cycle_path(self.C1)               # visits all 3 goals every window
        miss = jnp.full((1, 15, 2), 10.0)        # sits far from every goal

        r_hit = float(np.asarray(form.eval(hit)))
        r_miss = float(np.asarray(form.eval(miss)))
        self.assertGreater(r_hit, 0.0, "visiting all goals should be satisfied (rho>0)")
        self.assertLess(r_miss, 0.0, "sitting far from goals should be violated (rho<0)")

        # Override must give the same verdict as baked on both ground-truth paths.
        np.testing.assert_allclose(np.asarray(form.eval(hit, cent_override=ov)), r_hit, atol=TOL, rtol=TOL)
        np.testing.assert_allclose(np.asarray(form.eval(miss, cent_override=ov)), r_miss, atol=TOL, rtol=TOL)

    # Signal-like structure (m2signal3): loop_form & (progress U final), 4 goals,
    # exercises until + always + and + eventually + or with cent_override.
    S1 = [[1.0, 1.0], [3.0, 1.0], [2.0, 3.0], [3.5, 3.5]]
    S2 = [[0.5, 3.5], [3.5, 0.5], [2.0, 2.0], [1.0, 1.0]]

    @staticmethod
    def _build_signal(cents, T=15):
        preds = [_pred(c, i) for i, c in enumerate(cents)]  # names 0..3
        per = 4
        cover = preds[0].eventually(0, per)
        for g in range(1, 3):
            cover = cover & preds[g].eventually(0, per)
        loop_form = cover.always(0, per)
        progress = preds[0]
        for g in range(1, 4):
            progress = progress | preds[g]
        until_form = progress.until(preds[3], 8, T)
        return loop_form & until_form

    def test_signal_until_substitution(self):
        f1 = self._build_signal(self.S1)
        f2 = self._build_signal(self.S2)
        path = _cycle_path(self.S1)
        baked1 = np.asarray(f1.eval(path))
        baked2 = np.asarray(f2.eval(path))
        np.testing.assert_allclose(np.asarray(f1.eval(path, cent_override=jnp.asarray(np.array(self.S1)))),
                                   baked1, atol=TOL, rtol=TOL, err_msg="signal override-same != baked")
        np.testing.assert_allclose(np.asarray(f1.eval(path, cent_override=jnp.asarray(np.array(self.S2)))),
                                   baked2, atol=TOL, rtol=TOL, err_msg="signal override-S2 != baked f2")

    def test_single_reach_substitution(self):
        # Simplest case: a single reach predicate, eventually within the horizon.
        f1 = _pred(self.C1[0], 0).eventually(0, 14)
        f2 = _pred(self.C2[0], 0).eventually(0, 14)
        path = _cycle_path(self.C1)
        np.testing.assert_allclose(
            np.asarray(f1.eval(path, cent_override=jnp.asarray(np.array([self.C2[0]])))),
            np.asarray(f2.eval(path)), atol=TOL, rtol=TOL,
            err_msg="single-reach override-C2 != baked f2")

    # (cents, sizes) tuple override: per-goal random SIZES flow with the centers.
    F2 = [0.6, 1.35, 0.9]  # per-goal size factors

    @staticmethod
    def _build_loop_sized(cents, sizes, shrink=1.0):
        preds = [STL(RectReachPredicate(np.array(c, dtype=float), np.array(s, dtype=float), i,
                                        shrink_factor=shrink)) for i, (c, s) in enumerate(zip(cents, sizes))]
        per_loop = 15 // 2
        form = preds[0].eventually(0, per_loop)
        for g in preds[1:3]:
            form = form & g.eventually(0, per_loop)
        return form.always(0, per_loop)

    def _check_size_equiv(self, train_mode):
        sizes2 = [GOAL_SIZE * f for f in self.F2]
        form1 = _build_loop(self.C1)                       # baked: C1 cents, default sizes
        form2 = self._build_loop_sized(self.C2, sizes2)    # baked: C2 cents, F2 sizes
        path = _cycle_path(self.C1)
        ov_same = (jnp.asarray(np.array(self.C1)), jnp.asarray(np.tile(GOAL_SIZE, (3, 1))))
        ov_new = (jnp.asarray(np.array(self.C2)), jnp.asarray(np.array(sizes2)))

        # (1) tuple override with baked cents+sizes == baked eval.
        np.testing.assert_allclose(
            np.asarray(form1.eval(path, train_mode=train_mode, cent_override=ov_same)),
            np.asarray(form1.eval(path, train_mode=train_mode)), atol=TOL, rtol=TOL,
            err_msg=f"size-override-same != baked (train_mode={train_mode})")
        # (2) tuple override with new cents+sizes on form1 == rebuilt form2 (pure substitution).
        np.testing.assert_allclose(
            np.asarray(form1.eval(path, train_mode=train_mode, cent_override=ov_new)),
            np.asarray(form2.eval(path, train_mode=train_mode)), atol=TOL, rtol=TOL,
            err_msg=f"size-override-new != rebuilt sized form (train_mode={train_mode})")

    def test_size_override_eval(self):
        self._check_size_equiv(train_mode=False)

    def test_size_override_eval_train(self):
        self._check_size_equiv(train_mode=True)

    def test_size_override_stlpy_form(self):
        # MILP path: tuple override must move AND resize the box exactly like rebuilding.
        sizes2 = [GOAL_SIZE * f for f in self.F2]
        form1 = _build_loop(self.C1)
        form2 = self._build_loop_sized(self.C2, sizes2)
        ov_new = (np.array(self.C2), np.array(sizes2))
        self.assertEqual(str(form1.get_stlpy_form(cent_override=ov_new)),
                         str(form2.get_stlpy_form()),
                         "stlpy tuple override != rebuilt sized form")


if __name__ == "__main__":
    unittest.main()
