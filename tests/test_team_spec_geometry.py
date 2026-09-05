"""Geometry regression tests for branched team specs.

Guards against the 2026-07-11 goal-index bug: `_parse_choiceseq3_spec` used a
misdocumented goal_set layout (idx2/3/4 swapped), which
sent branch A's final sweep goal into branch B's cross-avoid zone — making branch A
unsatisfiable by construction. Any episode whose first kept draw froze onto branch A
then cap-ran the resample loop (surfaced as the "candidates>8 at N=16" failure).

Invariants: with cross-branch avoidance, no branch may contain a goal that belongs
to the other branch's goal set (that goal would be simultaneously reach-required and
always-avoided); choiceseq3's branches must respect their top/bottom-half design.
"""

import unittest

from gcbfplus.stl.team_spec import parse_team_spec

GOAL_SET = [[0, 0], [2, 2], [2, 0], [0, 2], [4, 4], [4, 0], [0, 4], [4, 2], [2, 4]]
GATHER = (2.0, 2.0)


def _branch_goals(spec, branch):
    pts = set()
    for tid in branch:
        for g in spec.tasks[tid].goal_centers:
            pt = (float(g[0]), float(g[1]))
            if pt != GATHER:  # shared gathering point is common to all branches
                pts.add(pt)
    return pts


class TestBranchedSpecGeometry(unittest.TestCase):

    def _assert_branches_disjoint(self, spec_string, n_agents=16):
        spec = parse_team_spec(spec_string, n_agents, GOAL_SET)
        self.assertIsNotNone(spec.branches)
        a, b = spec.branches
        ga, gb = _branch_goals(spec, a), _branch_goals(spec, b)
        overlap = ga & gb
        self.assertFalse(
            overlap,
            f"{spec_string}: goals {overlap} appear in BOTH branches — with "
            f"cross-branch avoidance the owning branch must reach a goal the other "
            f"branch forbids, making one branch unsatisfiable")
        return spec, ga, gb

    def test_choiceseq3_branches_disjoint(self):
        self._assert_branches_disjoint("team_choiceseq3_t15")

    def test_choiceseq3_top_bottom_halves(self):
        spec, ga, gb = self._assert_branches_disjoint("team_choiceseq3_t15")
        # Design: branch A sweeps the TOP half (y >= 2), branch B the BOTTOM (y <= 2).
        self.assertTrue(all(y >= 2 for _, y in ga), f"branch A goals not all top-half: {ga}")
        self.assertTrue(all(y <= 2 for _, y in gb), f"branch B goals not all bottom-half: {gb}")
        # The documented waypoints, exactly.
        self.assertEqual(ga, {(0., 4.), (4., 4.), (2., 4.)})
        self.assertEqual(gb, {(4., 0.), (0., 0.), (2., 0.)})


if __name__ == "__main__":
    unittest.main()
