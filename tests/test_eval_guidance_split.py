"""Tests for the eval-vs-guidance split introduced in commit f7a53a4 + the
OR-spec branch extraction merged from stash0_critical_waypoints.patch.

These exercise the contract the user asked for:
- `avoid_expansion` on the per-agent and team-level builders actually scales
  the avoid regions (so eval@1.0 produces strictly different regions than
  guidance@1.2).
- The OR-spec branch extraction filters predicates by `task_{tid}` name
  substring — this is the exact filter used to reduce ma_stl_form_guidance
  to the chosen branch while leaving ma_stl_form (eval) untouched.
"""
import unittest
import numpy as np


def _make_team_cover_spec(n_agents=4):
    from gcbfplus.stl.team_spec import TeamSpec, TeamTask
    tasks = [
        TeamTask(task_id=0, goal_centers=[[1.0, 1.0]]),
        TeamTask(task_id=1, goal_centers=[[5.0, 5.0]]),
    ]
    return TeamSpec(
        spec_type='team_cover', tasks=tasks,
        time_horizon=15, n_agents=n_agents, m_per_task=1, task_mode='cover',
    )


def _make_or_spec_with_branches(n_agents=4):
    """OR spec: branch [0,1] or branch [2,3]. Used to exercise branch extraction."""
    from gcbfplus.stl.team_spec import TeamSpec, TeamTask
    tasks = [
        TeamTask(task_id=0, goal_centers=[[1.0, 1.0]]),
        TeamTask(task_id=1, goal_centers=[[1.5, 1.5]]),
        TeamTask(task_id=2, goal_centers=[[5.0, 5.0]]),
        TeamTask(task_id=3, goal_centers=[[5.5, 5.5]]),
    ]
    return TeamSpec(
        spec_type='team_cover', tasks=tasks,
        time_horizon=15, n_agents=n_agents, m_per_task=1, task_mode='cover',
        branches=[[0, 1], [2, 3]],
    )


class TestAvoidExpansionScalesRegions(unittest.TestCase):
    """build_per_agent_stl_forms must return avoid_regions scaled by avoid_expansion.

    This is the foundation of the eval/guidance split: if expansion doesn't
    actually flow into the returned regions, the split is cosmetic.
    """

    def setUp(self):
        from gcbfplus.stl.team_allocator import allocate_tasks
        self.team_spec = _make_team_cover_spec(n_agents=4)
        self.goal_size = np.array([1.0, 1.0])
        self.positions = np.array([[0.5, 0.5], [1.5, 1.5], [4.5, 4.5], [5.5, 5.5]])
        self.allocation = allocate_tasks(self.team_spec, self.positions, strategy='greedy')

    def test_eval_vs_guidance_regions_differ(self):
        """expansion=1.0 and expansion=1.2 should produce region sizes differing by 1.2×."""
        from gcbfplus.stl.team_allocator import build_per_agent_stl_forms
        _, eval_regions = build_per_agent_stl_forms(
            self.team_spec, self.allocation, self.goal_size, 1.0, 4,
            add_avoidance=True, avoid_expansion=1.0,
        )
        _, guidance_regions = build_per_agent_stl_forms(
            self.team_spec, self.allocation, self.goal_size, 1.0, 4,
            add_avoidance=True, avoid_expansion=1.2,
        )
        # Find an assigned agent that has at least one avoid region.
        any_match = False
        for eval_agent, guid_agent in zip(eval_regions, guidance_regions):
            for (e_center, e_size), (g_center, g_size) in zip(eval_agent, guid_agent):
                any_match = True
                np.testing.assert_allclose(e_center, g_center, err_msg="centers must match")
                np.testing.assert_allclose(
                    g_size, e_size * 1.2,
                    err_msg="guidance region must be 1.2× the eval region",
                )
        self.assertTrue(any_match, "test_avoid needed at least one avoid region")

    def test_expansion_one_gives_goal_sized_regions(self):
        """expansion=1.0 means the avoid region equals the goal size itself."""
        from gcbfplus.stl.team_allocator import build_per_agent_stl_forms
        _, regions = build_per_agent_stl_forms(
            self.team_spec, self.allocation, self.goal_size, 1.0, 4,
            add_avoidance=True, avoid_expansion=1.0,
        )
        saw_region = False
        for agent_regions in regions:
            for _center, size in agent_regions:
                saw_region = True
                np.testing.assert_allclose(size, self.goal_size)
        self.assertTrue(saw_region)

    def test_no_avoidance_regions_are_empty_regardless_of_expansion(self):
        from gcbfplus.stl.team_allocator import build_per_agent_stl_forms
        _, regions = build_per_agent_stl_forms(
            self.team_spec, self.allocation, self.goal_size, 1.0, 4,
            add_avoidance=False, avoid_expansion=1.2,
        )
        for agent_regions in regions:
            self.assertEqual(agent_regions, [])


class TestBuildTeamCatlBranches(unittest.TestCase):
    """On an OR spec, build_team_catl tasks must expose `team_task_{id}` names —
    this is what the branch-extraction filter relies on.
    """

    def test_branch_extraction_keeps_only_chosen_branch(self):
        from gcbfplus.stl.team_allocator import build_team_catl
        team_spec = _make_or_spec_with_branches(n_agents=4)
        goal_size = np.array([1.0, 1.0])

        # Build at expansion=1.2 so this mirrors what ma_stl_form_guidance holds.
        catl_form = build_team_catl(team_spec, goal_size, 1.0,
                                    add_avoidance=True, avoid_expansion=1.2)
        all_tasks = catl_form.get_all_predicates()
        # Every OR branch task should appear by id in at least one predicate.name
        names = [t.name for t in all_tasks]
        for task_id in [0, 1, 2, 3]:
            self.assertTrue(
                any(f"task_{task_id}" in n for n in names),
                f"expected a predicate with task_{task_id} in its name (got {names})",
            )

        # Simulate the branch extraction: chosen_branch = [0, 1] (the "left" branch)
        chosen_branch = [0, 1]
        chosen_tasks = [t for t in all_tasks
                        if any(f"task_{tid}" in t.name for tid in chosen_branch)]
        other_tasks = [t for t in all_tasks
                       if any(f"task_{tid}" in t.name for tid in [2, 3])]

        self.assertGreater(len(chosen_tasks), 0, "chosen branch must yield predicates")
        self.assertGreater(len(other_tasks), 0, "other branch must also have predicates")
        # The filter must cleanly separate — no overlap by task_id substring.
        for t in chosen_tasks:
            self.assertFalse(any(f"task_{tid}" in t.name for tid in [2, 3]),
                             f"chosen-branch task {t.name} leaked into branch [2,3]")

    def test_non_or_spec_has_no_branches_so_extraction_is_skipped(self):
        """Non-OR spec must leave team_spec.branches as None so _reallocate_team_tasks
        never enters the branch-extraction block.
        """
        team_spec = _make_team_cover_spec(n_agents=4)
        self.assertIsNone(team_spec.branches)


class TestReallocateBranchExtractionTargetsGuidanceOnly(unittest.TestCase):
    """End-to-end on the real MASTLMixin: after _reallocate_team_tasks on an OR
    spec, ma_stl_form (eval) must be unchanged while ma_stl_form_guidance
    (guidance) must be reduced to the chosen branch.

    Uses a bare-bones subclass that sets only the attributes _load_team_spec /
    _reallocate_team_tasks need, avoiding the full env wrapper graph.
    """

    def _make_mixin_instance(self, avoid_expansion=1.2):
        from gcbfplus.env.wrapper.stl_mixin import MASTLMixin

        class _Bare(MASTLMixin):
            def __init__(self, avoid_expansion):
                import logging
                self.num_agents = 4
                self.GOAL_SET = np.array([[1.0, 1.0], [1.5, 1.5], [5.0, 5.0], [5.5, 5.5]])
                self.stl_shrink_factor = 1.0
                self.team_avoid = True
                self.avoid_expansion = avoid_expansion
                self.team_disjunctive = False
                self.allocation_strategy = 'greedy'
                self.ma_stl_exp_rob = False
                self.ma_stl_spec = None
                self.logger = logging.getLogger('test')

        return _Bare(avoid_expansion)

    def _build_or_spec_string(self):
        """We have to feed _load_team_spec a parseable spec string. Instead,
        call it via the team_spec_obj path directly for controllability.
        """
        # Bypass the spec parser — hand-construct the spec object and plug it in
        return _make_or_spec_with_branches(n_agents=4)

    def test_eval_form_unchanged_after_reallocate_on_or_spec(self):
        """After _reallocate_team_tasks on an OR spec, self.ma_stl_form
        (the eval form) must be identical to what _load_team_spec set.
        """
        from gcbfplus.stl.team_allocator import build_team_catl
        inst = self._make_mixin_instance(avoid_expansion=1.2)
        inst.team_spec_obj = self._build_or_spec_string()
        inst.allocation_strategy = 'greedy'

        goal_size = np.array([1.0, 1.0])
        # Manually populate the fields _load_team_spec would set, mirroring the
        # eval/guidance split in stl_mixin.py::_load_team_spec.
        inst.ma_stl_form = build_team_catl(
            inst.team_spec_obj, goal_size, inst.stl_shrink_factor,
            add_avoidance=True, avoid_expansion=1.0,
        )
        inst.ma_stl_form_guidance = build_team_catl(
            inst.team_spec_obj, goal_size, inst.stl_shrink_factor,
            add_avoidance=True, avoid_expansion=inst.avoid_expansion,
        )
        eval_form_before = inst.ma_stl_form
        eval_preds_before = [t.name for t in inst.ma_stl_form.get_all_predicates()]
        guidance_preds_before = [t.name for t in inst.ma_stl_form_guidance.get_all_predicates()]

        # Real agent positions that favour the left branch (tasks 0,1)
        agent_positions = np.array([[0.9, 0.9], [1.6, 1.6], [1.2, 1.2], [1.8, 1.8]])
        inst._reallocate_team_tasks(agent_positions)

        # Eval form must be the same object and carry the same predicate set.
        self.assertIs(inst.ma_stl_form, eval_form_before,
                      "eval form identity must not change — "
                      "branch extraction must target ma_stl_form_guidance only")
        eval_preds_after = [t.name for t in inst.ma_stl_form.get_all_predicates()]
        self.assertEqual(sorted(eval_preds_before), sorted(eval_preds_after))

        # Guidance form must now be reduced to the chosen branch.
        guidance_preds_after = [t.name for t in inst.ma_stl_form_guidance.get_all_predicates()]
        self.assertLess(len(guidance_preds_after), len(guidance_preds_before),
                        "guidance should shrink to chosen branch")

    def test_non_or_spec_leaves_both_forms_alone(self):
        """On a non-OR spec, _reallocate_team_tasks must not mutate either form."""
        from gcbfplus.stl.team_allocator import build_team_catl
        inst = self._make_mixin_instance(avoid_expansion=1.2)
        inst.team_spec_obj = _make_team_cover_spec(n_agents=4)

        goal_size = np.array([1.0, 1.0])
        inst.ma_stl_form = build_team_catl(
            inst.team_spec_obj, goal_size, inst.stl_shrink_factor,
            add_avoidance=True, avoid_expansion=1.0,
        )
        inst.ma_stl_form_guidance = build_team_catl(
            inst.team_spec_obj, goal_size, inst.stl_shrink_factor,
            add_avoidance=True, avoid_expansion=inst.avoid_expansion,
        )
        eval_before = inst.ma_stl_form
        guidance_before = inst.ma_stl_form_guidance

        positions = np.array([[0.5, 0.5], [1.5, 1.5], [4.5, 4.5], [5.5, 5.5]])
        inst._reallocate_team_tasks(positions)

        self.assertIs(inst.ma_stl_form, eval_before)
        self.assertIs(inst.ma_stl_form_guidance, guidance_before,
                      "non-OR spec must not touch guidance form")


if __name__ == '__main__':
    unittest.main()
