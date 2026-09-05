"""Equivalence gate for the O(N) traced fast path in DiffusionMAPlanner.forward.

For uniform per-agent specs the N rotated stl_forms share one AST (repr-identical;
only baked centers differ). The fast path evaluates the single structural form with
a dynamically indexed cent_override instead of lax.switch over N per-agent closures
(whose vmap evaluates all N branches per agent -> O(N^2)). These tests assert the
two constructions are numerically identical — values and gradients, eval and
eval_train, centers-only and (cents, sizes) tuple overrides.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ds.stl_jax import STL, RectReachPredicate
from gcbfplus.stl.utils import ENV_CONFIG

GOAL_SIZE = np.array([1, 1]) * ENV_CONFIG["goal_size"]
TOL = 1e-6
N = 4  # agents
T = 15


def _seq_form(cents, shrink=1.0):
    """mseq3-like: F[0,5]g0 & F[5,10]g1 & F[10,15]g2 (names 0..2)."""
    preds = [STL(RectReachPredicate(np.array(c, dtype=float), np.array(GOAL_SIZE), i,
                                    shrink_factor=shrink)) for i, c in enumerate(cents)]
    return preds[0].eventually(0, 5) & preds[1].eventually(5, 10) & preds[2].eventually(10, 15)


def _setup(with_sizes=False):
    base = np.array([[1.0, 1.0], [3.0, 1.0], [2.0, 3.0]])
    # Per-agent rotated forms, like _load_diff_spec's rotation (distinct baked cents,
    # identical structure/repr).
    forms = [_seq_form(np.roll(base, -i % 3, axis=0)) for i in range(N)]
    assert len({repr(f) for f in forms}) == 1, "rotated forms must share one repr/AST"
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    goal_centers = jax.random.uniform(k1, (N, 3, 2), minval=0.5, maxval=3.5)
    goal_sizes = (jax.random.uniform(k2, (N, 3, 1), minval=0.5, maxval=1.5)
                  * jnp.asarray(GOAL_SIZE)[None, None, :]) if with_sizes else None
    paths = jax.random.uniform(k3, (N, T, 2), minval=0.0, maxval=4.0)
    return forms, goal_centers, goal_sizes, paths


def _both_evals(forms, goal_centers, goal_sizes, train_mode):
    """Build the switch path and the shared-struct fast path exactly as loader.forward does."""
    n = len(forms)
    ovr = [goal_centers[i] if goal_sizes is None else (goal_centers[i], goal_sizes[i])
           for i in range(n)]
    method = 'eval_train' if train_mode else 'eval'
    fns = [(lambda x, i=i: getattr(forms[i], method)(x, cent_override=ovr[i])) for i in range(n)]
    switch_eval = lambda x, aid: jax.lax.switch(aid, fns, x)

    base = forms[0]
    if goal_sizes is None:
        _ovr_of = lambda aid: goal_centers[aid]
    else:
        _ovr_of = lambda aid: (goal_centers[aid], goal_sizes[aid])
    fast_eval = lambda x, aid: getattr(base, method)(x, cent_override=_ovr_of(aid))
    return switch_eval, fast_eval


class TestTracedVmapEquivalence(unittest.TestCase):

    def _check(self, with_sizes, train_mode):
        forms, gc, gs, paths = _setup(with_sizes)
        switch_eval, fast_eval = _both_evals(forms, gc, gs, train_mode)

        # Per-agent losses, mirroring edm_ma._calc_stl_loss's vmap over agent ids.
        def per_agent(fn):
            return jax.vmap(lambda x, aid: jnp.mean(fn(x[None], aid)))(paths, jnp.arange(N))

        np.testing.assert_allclose(np.asarray(per_agent(fast_eval)),
                                   np.asarray(per_agent(switch_eval)), atol=TOL, rtol=TOL,
                                   err_msg=f"values differ (sizes={with_sizes}, train={train_mode})")

        # Gradients w.r.t. the trajectories (the guidance signal).
        def loss(fn):
            return lambda ys: jax.vmap(lambda x, aid: jnp.mean(fn(x[None], aid)))(ys, jnp.arange(N)).mean()

        g_fast = jax.grad(loss(fast_eval))(paths)
        g_switch = jax.grad(loss(switch_eval))(paths)
        np.testing.assert_allclose(np.asarray(g_fast), np.asarray(g_switch), atol=TOL, rtol=TOL,
                                   err_msg=f"grads differ (sizes={with_sizes}, train={train_mode})")

    def test_centers_eval(self):
        self._check(with_sizes=False, train_mode=False)

    def test_centers_eval_train(self):
        self._check(with_sizes=False, train_mode=True)

    def test_sizes_eval(self):
        self._check(with_sizes=True, train_mode=False)

    def test_sizes_eval_train(self):
        self._check(with_sizes=True, train_mode=True)

    def test_mixed_structure_not_eligible(self):
        # A mixed-spec-like set (different ASTs) must NOT satisfy the fast-path detector.
        base = np.array([[1.0, 1.0], [3.0, 1.0], [2.0, 3.0]])
        f_seq = _seq_form(base)
        preds = [STL(RectReachPredicate(np.array(c, dtype=float), np.array(GOAL_SIZE), i))
                 for i, c in enumerate(base)]
        f_loop = (preds[0].eventually(0, 7) & preds[1].eventually(0, 7)).always(0, 7)
        self.assertGreater(len({repr(f) for f in [f_seq, f_loop, f_seq, f_loop]}), 1)


class TestBakedExtraction(unittest.TestCase):
    """Fixed-grid case: _shared_structural_override extracts baked reach centers and
    must match per-form (switch-style) evaluation exactly; anything not provably
    exchangeable must fall back (return None)."""

    def _rotated_forms(self):
        base = np.array([[1.0, 1.0], [3.0, 1.0], [2.0, 3.0]])
        return [_seq_form(np.roll(base, -i % 3, axis=0)) for i in range(N)]

    def test_baked_uniform_matches_per_form(self):
        from gcbfplus.diffusion.util.stl_fastpath import shared_structural_override as _shared_structural_override
        forms = self._rotated_forms()
        fast = _shared_structural_override(forms)
        self.assertIsNotNone(fast, "rotated fixed-grid forms should be eligible")
        fast_eval, fast_eval_train = fast
        paths = jax.random.uniform(jax.random.PRNGKey(1), (N, T, 2), minval=0.0, maxval=4.0)
        for train_mode, fn in ((False, fast_eval), (True, fast_eval_train)):
            method = 'eval_train' if train_mode else 'eval'
            ref = jnp.stack([jnp.mean(getattr(forms[i], method)(paths[i][None])) for i in range(N)])
            got = jax.vmap(lambda x, aid: jnp.mean(fn(x[None], aid)))(paths, jnp.arange(N))
            np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=TOL, rtol=TOL,
                                       err_msg=f"baked fast path != per-form eval (train={train_mode})")
        # Gradients too (the guidance signal).
        loss_fast = lambda ys: jax.vmap(lambda x, aid: jnp.mean(fast_eval(x[None], aid)))(ys, jnp.arange(N)).mean()
        loss_ref = lambda ys: jnp.stack([jnp.mean(forms[i].eval(ys[i][None])) for i in range(N)]).mean()
        np.testing.assert_allclose(np.asarray(jax.grad(loss_fast)(paths)),
                                   np.asarray(jax.grad(loss_ref)(paths)), atol=TOL, rtol=TOL)

    def test_per_agent_sizes_fall_back(self):
        from gcbfplus.diffusion.util.stl_fastpath import shared_structural_override as _shared_structural_override
        base = np.array([[1.0, 1.0], [3.0, 1.0], [2.0, 3.0]])
        forms = [_seq_form(base)]
        # Same AST, but agent 1's goal 0 has a different baked SIZE -> not exchangeable.
        preds = [STL(RectReachPredicate(np.array(c, dtype=float),
                                        np.array(GOAL_SIZE) * (1.3 if i == 0 else 1.0), i,
                                        shrink_factor=1.0)) for i, c in enumerate(base)]
        f_sized = preds[0].eventually(0, 5) & preds[1].eventually(5, 10) & preds[2].eventually(10, 15)
        forms.append(f_sized)
        self.assertEqual(len({repr(f) for f in forms}), 1, "sizes are not in repr")
        self.assertIsNone(_shared_structural_override(forms))

    def test_differing_avoid_leaf_falls_back(self):
        from gcbfplus.diffusion.util.stl_fastpath import shared_structural_override as _shared_structural_override
        from ds.stl_jax import RectAvoidPredicate
        base = np.array([[1.0, 1.0], [3.0, 1.0], [2.0, 3.0]])

        def form_with_avoid(avoid_cent):
            f = _seq_form(base)
            avoid = STL(RectAvoidPredicate(np.array(avoid_cent, dtype=float), np.array(GOAL_SIZE), 7))
            return f & avoid.always(0, 15)

        forms = [form_with_avoid([2.0, 2.0]), form_with_avoid([0.5, 3.0])]
        self.assertEqual(len({repr(f) for f in forms}), 1, "avoid cents are not in repr")
        self.assertIsNone(_shared_structural_override(forms),
                          "avoid leaves cannot be overridden -> differing cents must fall back")


if __name__ == "__main__":
    unittest.main()
