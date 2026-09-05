"""Unit tests for counting-aware repair assignment (edm_ma pure functions)."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from gcbfplus.diffusion.diffusion.edm_ma import (
    _assigned_loss, _repair_assignment, _task_R, TaskRepairCtx,
)

THRESH = 0.1
SINGLE_BRANCH = ((True, True, True, True),)
DNF_BRANCHES = ((True, True, False, False), (False, False, True, True))


def counts(assign, num_tasks):
    return [int((np.asarray(assign) == q).sum()) for q in range(num_tasks)]


class TestRepairAssignment(unittest.TestCase):

    def test_collapse_repair(self):
        # The actual failure mode: all 8 agents satisfy task 0 only.
        rng = np.random.RandomState(0)
        R = np.full((4, 8), -1.0) + rng.uniform(0, 0.01, (4, 8))
        R[0] = 0.5 + rng.uniform(0, 0.01, 8)
        assign_prev = jnp.full(8, 4, dtype=jnp.int32)
        assign, branch = _repair_assignment(
            jnp.asarray(R, jnp.float32), assign_prev, jnp.asarray(-1, jnp.int32),
            THRESH, (2, 2, 2, 2), SINGLE_BRANCH)
        self.assertEqual(counts(assign, 4), [2, 2, 2, 2])
        # 6 agents are assigned to tasks they don't yet satisfy -> redraw set.
        loss = _assigned_loss(assign, jnp.asarray(R, jnp.float32), jnp.zeros(8))
        self.assertEqual(int((loss < THRESH).sum()), 6)

    def test_stickiness_fixed_point(self):
        # Balanced satisfying assignment; a newcomer with higher raw rho on
        # task 0 must NOT displace a satisfying incumbent.
        R = np.full((2, 4), -1.0)
        assign_prev = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
        R[0, 0], R[0, 1] = 0.3, 0.3
        R[1, 2], R[1, 3] = 0.3, 0.3
        R[0, 2] = 0.9  # agent 2 looks better on task 0 than its incumbents
        assign, _ = _repair_assignment(
            jnp.asarray(R, jnp.float32), assign_prev, jnp.asarray(0, jnp.int32),
            THRESH, (2, 2), ((True, True),))
        np.testing.assert_array_equal(np.asarray(assign), [0, 0, 1, 1])

    def test_branch_choice_and_freeze(self):
        R = np.full((4, 8), -1.0)
        R[0], R[1] = 0.4, 0.4  # branch A clearly satisfiable
        ms = (6, 2, 6, 2)
        assign, branch = _repair_assignment(
            jnp.asarray(R, jnp.float32), jnp.full(8, 4, jnp.int32),
            jnp.asarray(-1, jnp.int32), THRESH, ms, DNF_BRANCHES)
        self.assertEqual(int(branch), 0)
        self.assertEqual(counts(assign, 4), [6, 2, 0, 0])
        # Frozen previous choice wins regardless of R.
        assign_b, branch_b = _repair_assignment(
            jnp.asarray(R, jnp.float32), jnp.full(8, 4, jnp.int32),
            jnp.asarray(1, jnp.int32), THRESH, ms, DNF_BRANCHES)
        self.assertEqual(int(branch_b), 1)
        self.assertEqual(counts(assign_b, 4), [0, 0, 6, 2])

    def test_oversubscription_short_fills(self):
        # N=4 agents, 4 tasks x m=2 -> later tasks short-fill, no crash.
        R = jnp.asarray(np.random.RandomState(1).uniform(-1, 1, (4, 4)), jnp.float32)
        assign, _ = _repair_assignment(
            R, jnp.full(4, 4, jnp.int32), jnp.asarray(-1, jnp.int32),
            THRESH, (2, 2, 2, 2), SINGLE_BRANCH)
        c = counts(assign, 4)
        self.assertEqual(c[:2], [2, 2])
        self.assertEqual(sum(c), 4)
        self.assertTrue(np.all((np.asarray(assign) >= 0) & (np.asarray(assign) <= 4)))

    def test_jit_vmap_match_eager(self):
        rng = np.random.RandomState(2)
        Rb = jnp.asarray(rng.uniform(-1, 1, (3, 4, 8)), jnp.float32)
        assign_prev = jnp.full(8, 4, jnp.int32)
        bprev = jnp.asarray(-1, jnp.int32)
        ms = (2, 2, 2, 2)

        def run(R):
            return _repair_assignment(R, assign_prev, bprev, THRESH, ms, SINGLE_BRANCH)

        eager = [run(Rb[i]) for i in range(3)]
        jitted = jax.jit(run)
        vmapped = jax.vmap(run)(Rb)
        for i in range(3):
            np.testing.assert_array_equal(np.asarray(eager[i][0]),
                                          np.asarray(jitted(Rb[i])[0]))
            np.testing.assert_array_equal(np.asarray(eager[i][0]),
                                          np.asarray(vmapped[0][i]))
        self.assertEqual(vmapped[0].dtype, jnp.int32)
        self.assertEqual(vmapped[1].dtype, jnp.int32)


class TestAssignedLoss(unittest.TestCase):

    def test_gather_and_free_fallback(self):
        R = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        or_vals = jnp.asarray([-7.0, -8.0, -9.0])
        assign = jnp.asarray([1, 2, 0], dtype=jnp.int32)  # 2 == free
        loss = _assigned_loss(assign, R, or_vals)
        np.testing.assert_allclose(np.asarray(loss), [4.0, -8.0, 3.0])


class TestTaskR(unittest.TestCase):

    def test_stacks_per_task_fns(self):
        ctx = TaskRepairCtx(
            inner_eval_fns=(lambda y: y[:, 0, 0], lambda y: -y[:, 0, 0]),
            ms=(1, 1), branch_masks=((True, True),))
        y = jnp.arange(6.0).reshape(3, 2, 1)
        R = _task_R(ctx, y)
        np.testing.assert_allclose(np.asarray(R), [[0, 2, 4], [0, -2, -4]])


if __name__ == '__main__':
    unittest.main()
