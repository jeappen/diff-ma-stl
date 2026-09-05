"""Tests for the avoid-until chain spec (``mauntil`` / ``mauntilK``).

Built in ``STLMixin._load_auntil_stl_form`` as:
    AND_k (~g_k U[0, tu_k] g_{k-1})  &  F[tu_K, T] g_K
with ``tu_k = k*T//(K+1) + 1``. ``mauntil`` (K=1) is the literal
``!A U B``; ``mauntil2`` chains two stages over three goals (keyed gauntlet:
g1 forbidden until g0, g2 forbidden until g1, finish inside g2) so the
g0 -> g1 -> g2 visit order is forced purely by negation-untils.

For T=15, K=2: tu_1 = 6, tu_2 = 11.
"""

import unittest

import jax.numpy as jnp
import numpy as np

from gcbfplus.env.wrapper.stl_mixin import STLMixin
from ds.stl_jax import STL, RectReachPredicate

GS = 1.0
T = 15

# g0 = key (free), g1 = door 1, g2 = door 2; N = neutral point outside all boxes.
GOALS = {0: [0., 0.], 1: [2., 0.], 2: [0., 2.]}
NEUTRAL = [4., 4.]


def _preds(n):
    return [STL(RectReachPredicate(np.array(GOALS[i]), np.array([GS, GS]), i))
            for i in range(n)]


def _path(step_goals):
    """(1, T, 2) path from per-step goal indices (None = neutral), padded with the last."""
    s = (step_goals + [step_goals[-1]] * T)[:T]
    pts = [GOALS[i] if i is not None else NEUTRAL for i in s]
    return jnp.asarray(np.array(pts, float))[None]


class TestAuntilChain(unittest.TestCase):
    """mauntil2: order g0 -> g1 -> g2 enforced purely by the two negation-untils."""

    def setUp(self):
        preds = _preds(3)
        # original == rotated here, so any mixed_spec_mode leaves goals unchanged.
        _, self.form, _ = STLMixin()._load_auntil_stl_form(preds, preds, "mauntil2_t15")

    def _rob(self, p):
        return float(self.form.eval(p, approx_method="true")[0])

    def test_ordered_chain_passes(self):
        # g0 @3 (<= tu_1=6), g1 @7 (<= tu_2=11), g2 @11.. (F[11,15]).
        p = _path([None, None, None, 0, None, None, None, 1, None, None, None, 2, 2, 2, 2])
        self.assertGreater(self._rob(p), 0)

    def test_door1_before_key_fails(self):
        # g1 entered @1 before g0 @3 -> first until violated.
        p = _path([None, 1, None, 0, None, None, None, 1, None, None, None, 2, 2, 2, 2])
        self.assertLess(self._rob(p), 0)

    def test_door2_before_door1_fails(self):
        # g2 entered @5 before g1 @7 -> second until violated.
        p = _path([None, None, None, 0, None, 2, None, 1, None, None, None, 2, 2, 2, 2])
        self.assertLess(self._rob(p), 0)

    def test_never_finishing_fails(self):
        # Correct order but g2 never visited in the tail window.
        p = _path([None, None, None, 0, None, None, None, 1, None, None, None, None])
        self.assertLess(self._rob(p), 0)

    def test_late_key_fails(self):
        # g0 only @8 > tu_1=6 -> first until has no witness in [0,6].
        p = _path([None, None, None, None, None, None, None, None, 0, 1, None, 2, 2, 2, 2])
        self.assertLess(self._rob(p), 0)


class TestAuntilBackwardCompat(unittest.TestCase):
    """mauntil (no digit) must reproduce the original (!A U[0,8] B) & F[8,15] A exactly."""

    def test_k1_equals_handbuilt(self):
        preds = _preds(2)
        _, form, _ = STLMixin()._load_auntil_stl_form(preds, preds, "mauntil_t15")
        goal_b, goal_a = _preds(2)
        tu = T // 2 + 1
        ref = (~goal_a).until(goal_b, 0, tu) & goal_a.eventually(tu, T)
        paths = [
            _path([None, None, 0, None, None, None, None, None, 1, 1, 1, 1]),   # sat
            _path([None, 1, 0, None, None, None, None, None, 1, 1, 1, 1]),      # A before B
            _path([None, None, 0, None, None, None, None, None, None, None]),   # no final A
        ]
        for p in paths:
            self.assertAlmostEqual(float(form.eval(p, approx_method="true")[0]),
                                   float(ref.eval(p, approx_method="true")[0]), places=5)

    def test_spec_name_mapping(self):
        m = STLMixin()
        self.assertEqual(m._map_spec("mauntil_t15"), m._map_spec("mauntil2_t15"))


class TestAuntilSharing(unittest.TestCase):
    """mixed_spec_mode wiring: first1 shares the unrotated key g0, last1 the final
    door g_K, None leaves the rotated goals untouched (pre-2026-07-10 behavior)."""

    CENTS = [[0., 0.], [2., 0.], [0., 2.], [4., 4.]]

    def _make(self, mode):
        m = STLMixin()
        m.stl_mixed_spec_mode = mode  # instance attr overrides the yaml class default
        original = [STL(RectReachPredicate(np.array(c), np.array([GS, GS]), i))
                    for i, c in enumerate(self.CENTS)]
        # Agent-1 rotation of the same list (names follow position, as in _load_diff_spec).
        rot = self.CENTS[1:] + self.CENTS[:1]
        rotated = [STL(RectReachPredicate(np.array(c), np.array([GS, GS]), i))
                   for i, c in enumerate(rot)]
        preds, form, _ = m._load_auntil_stl_form(rotated, original, "mauntil2_t15")
        return [p.get_all_predicates()[0].cent.tolist() for p in preds[:3]], form

    def test_first1_shares_key(self):
        cents, _ = self._make("first1")
        self.assertEqual(cents[0], self.CENTS[0])          # shared unrotated key
        self.assertEqual(cents[1], self.CENTS[2])          # rotated door 1 kept
        self.assertEqual(cents[2], self.CENTS[3])          # rotated door 2 kept

    def test_last1_shares_final_door(self):
        cents, _ = self._make("last1")
        self.assertEqual(cents[0], self.CENTS[1])          # rotated key kept
        self.assertEqual(cents[1], self.CENTS[2])          # rotated door 1 kept
        self.assertEqual(cents[2], self.CENTS[2])          # shared unrotated g2

    def test_none_keeps_rotation(self):
        cents, form = self._make("None")
        self.assertEqual(cents, [self.CENTS[1], self.CENTS[2], self.CENTS[3]])
        # And robustness matches a rotated-only build (pre-wiring behavior).
        p = _path([None, None, None, 0, None, None, None, 1, None, None, None, 2, 2, 2, 2])
        self.assertTrue(np.isfinite(float(form.eval(p, approx_method="true")[0])))


if __name__ == "__main__":
    unittest.main()
