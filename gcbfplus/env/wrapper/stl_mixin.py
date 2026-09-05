"""Wrappers/Mixins for any environment."""
import jax.numpy as jnp
import jax.random as jrandom
import logging
import numpy as np
import os
import re

os.environ["DIFF_STL_BACKEND"] = "jax"
import ds.stl
import ds.stl_jax
from gcbfplus.stl.utils import TRAINING_CONFIG, ENV_CONFIG

import functools as ft


ds.stl.HARDNESS = TRAINING_CONFIG['ds_params']['HARDNESS']
ds.stl_jax.HARDNESS = TRAINING_CONFIG['ds_params']['HARDNESS']  # Set hardness for easier backpropagation

# Fixed plan length of the default diffusion model (qkmvppvt). The signal spec is
# always built at this horizon, so a bare `2signal3` (-> `_t30` by default) still
# evaluates correctly on the fixed-length plan.
SIGNAL_PLAN_LENGTH = 15

# Letter labels for the 9 in-distribution goal-grid points ({0,2,4}^2), row-major
# (y then x). Used to describe per-agent signal goal assignments readably.
SIGNAL_GOAL_LETTERS = {
    (0., 0.): 'A', (2., 0.): 'B', (4., 0.): 'C',
    (0., 2.): 'D', (2., 2.): 'E', (4., 2.): 'F',
    (0., 4.): 'G', (2., 4.): 'H', (4., 4.): 'I',
}


class STLMixin:
    """Contains functions for loading STL specs"""

    spec_regex = re.compile(r"([a-zA-Z0-9-]+)_t(\d+)")
    cover_regex = re.compile(r"m?cover(\d+)_t(\d+)")
    seq_regex = re.compile(r"m?seq(\d+)_t(\d+)")
    loop_regex = re.compile(r"m?(\d+)loop(\d+)_t(\d+)")
    branch_regex = re.compile(r"m?(\d+)branch(\d+)_t(\d+)")
    signal_regex = re.compile(r"m?(\d+)signal(\d+)_t(\d+)")
    reach_regex = re.compile(r"m?reach_t(\d+)")
    ereach_regex = re.compile(r"m?ereach_t(\d+)")
    # Expressive-fragment demos (persistence and until-type constraints):
    # stay<w>  = F[0,T-w]( G[0,w] A )      -- FG "reach and remain" (stabilization)
    # auntil   = (~A U[0,ceil(T/2)] B) & F A -- avoid-until-reach with a literal negation leaf
    # auntilK  = K chained stages over K+1 goals (mauntil2 = keyed gauntlet):
    #            AND_k (~g_k U g_{k-1}) & F g_K -- visit order forced purely by negation-untils
    stay_regex = re.compile(r"m?stay(\d+)_t(\d+)")
    auntil_regex = re.compile(r"m?auntil(\d*)_t(\d+)")
    # cuntil = cover-until: avoid ALL right-column goals until EVERY left-column goal
    # is visited at least once (a cover), then visit some right-column goal.
    # Optional digit = cover only the K bottom-most left goals (mcuntil2 -> 2 of 3).
    cuntil_regex = re.compile(r"m?cuntil(\d*)_t(\d+)")
    # seq for sequential with same goals for all agents, mseq for sequential with different goals for each agent
    # Now detect a combination of any of the above regexes with one time interval at the end
    # 1. Define the base patterns for each spec type without the time component.
    #    These are the building blocks of the mixed spec.
    _cover_base = r"m?cover\d+"
    _seq_base = r"m?seq\d+"
    _loop_base = r"m?\d+loop\d+"
    _branch_base = r"m?\d+branch\d+"
    _signal_base = r"m?\d+signal\d+"
    _reach_base = r"m?reach"
    _ereach_base = r"m?ereach"

    # 2. Combine the base patterns into a single non-capturing group (?:...)
    #    This group represents any single valid spec component.
    _spec_component = (
        f"(?:{_cover_base}|{_seq_base}|{_loop_base}|{_branch_base}|"
        f"{_signal_base}|{_reach_base}|{_ereach_base})"
    )

    # 3. Now, define the full regex for mixed specs.
    #    - It looks for one or more spec components separated by a hyphen.
    #    - It requires a single time interval `_t(\d+)` at the very end.
    #    - It captures two main groups:
    #        Group 1: The entire string of spec components (e.g., "mseq2-m2branch2")
    #        Group 2: The final time value (e.g., "30")
    mixed_spec_regex = re.compile(rf"^({_spec_component}(?:-{_spec_component})*)_t(\d+)$")
    mixed_spec_last_regex = re.compile(r"last(\d+)")  # for last X goals are common in a mixed spec
    mixed_spec_first_regex = re.compile(r"first(\d+)")  # for first X goals are common in a mixed spec
    # The fixed 3x3 goal grid ({0,2,4}^2) the bundled diffusion model was trained on. Spec builders
    # index into this list (see the team_choiceseq3 geometry); GCBF_GOAL_SCALE scales it uniformly.
    DEFAULT_GOALS = {'empty': [[0, 0], [2, 2], [2, 0], [0, 2], [4, 4], [4, 0], [0, 4], [4, 2], [2, 4]]}
    GOAL_SET = DEFAULT_GOALS['empty']
    stl_shrink_factor = TRAINING_CONFIG['ds_params']['shrink_factor']
    stl_mixed_spec_mode = TRAINING_CONFIG['ds_params']['mixed_spec_mode']
    stl_hardness = TRAINING_CONFIG['ds_params']['HARDNESS']
    stl_within_boundary = TRAINING_CONFIG['ds_params']['within_boundary']

    @property
    def mixed_spec_goals(self):
        """Return the mixed spec mode for the environment from the wrapper.

        This helps set common final goals for all agents to make the task harder to avoid collisions"""
        if self.stl_mixed_spec_mode is None or self.stl_mixed_spec_mode == "None":
            # If the mixed spec mode is not set, return 0 meaning no goals are common
            return 0
        elif self.mixed_spec_last_regex.match(self.stl_mixed_spec_mode):
            # If the mixed spec mode is set to lastX, return X meaning last X goals are common
            return int(self.mixed_spec_last_regex.match(self.stl_mixed_spec_mode).groups()[0])
        elif self.mixed_spec_first_regex.match(self.stl_mixed_spec_mode):
            # If the mixed spec mode is set to firstX, return X meaning first X goals are common
            return int(self.mixed_spec_first_regex.match(self.stl_mixed_spec_mode).groups()[0])
        else:
            raise ValueError(f"Mixed spec mode {self.stl_mixed_spec_mode} not recognized")

    @property
    def mixed_spec_mode(self):
        """Return the mixed spec mode for the environment from the wrapper."""
        if self.stl_mixed_spec_mode is None or self.stl_mixed_spec_mode == "None":
            # If the mixed spec mode is not set, return 0 meaning no goals are common
            return None
        elif self.mixed_spec_last_regex.match(self.stl_mixed_spec_mode):
            # If the mixed spec mode is set to lastX, return X meaning last X goals are common
            return "last"
        elif self.mixed_spec_first_regex.match(self.stl_mixed_spec_mode):
            # If the mixed spec mode is set to firstX, return X meaning first X goals are common
            return "first"
        else:
            raise ValueError(f"Mixed spec mode {self.stl_mixed_spec_mode} not recognized")

    def _set_shared_goal_predicates(self, spec, original_goal_predicates, goal_predicates, num_goals):
        """Set shared goal predicates for the environment"""

        if self.mixed_spec_goals > 0:
        # goal_predicates = agent i's rotated list; original_goal_predicates = the unrotated goal_list.
        # firstK: all agents share the first K (unrotated) goals -> a common start region.
        # lastK: all agents share the last K -> a common final region. Branch specs apply the rule
        # inside each branch's block of num_goals predicates.
            # If the mixed spec mode is set, share from the beginning of goal_set
            self.logger.info(f"Using mixed spec mode {self.mixed_spec_goals} for spec {spec}")
            assert original_goal_predicates is not None, "Original goal predicates should be set for mixed spec mode"
            unique_goals = num_goals - self.mixed_spec_goals
            if self.branch_regex.match(spec):
                # For branch specs, we need to set the goals in a specific order for each branch to match the shared spec
                shared_grouped_goals = [original_goal_predicates[i:i + num_goals] for i in
                                        range(0, len(goal_predicates), num_goals)]
                # Now share the last goals in the branch
                mixed_grouped_goals = [goal_predicates[i:i + num_goals] for i in
                                       range(0, len(goal_predicates), num_goals)]
                # Set the last goals in the branch to be the same as the shared spec
                final_goal_predicates = []
                for i, goals_in_branch in enumerate(shared_grouped_goals):
                    if self.mixed_spec_mode == "last":
                        # Set the last goals in the branch to be the same as the shared spec
                        final_goal_predicates += goals_in_branch[:unique_goals] + mixed_grouped_goals[i][unique_goals:]
                    elif self.mixed_spec_mode == "first":
                        final_goal_predicates += mixed_grouped_goals[i][:self.mixed_spec_goals] + goals_in_branch[
                                                                                                  self.mixed_spec_goals:]
                goal_predicates = final_goal_predicates
            else:
                if self.mixed_spec_mode == "last":
                    # Share the last N goals in the spec
                    goal_predicates = goal_predicates[:unique_goals] + original_goal_predicates[unique_goals:]
                elif self.mixed_spec_mode == "first":
                    # Share the first N goals in the spec
                    goal_predicates = original_goal_predicates[:self.mixed_spec_goals] + goal_predicates[
                                                                                         self.mixed_spec_goals:]

        return goal_predicates

    spec_name = None

    def _set_goal_set(self):
        """Set goal set for the environment"""
        self.GOAL_SET = self.DEFAULT_GOALS[self.env.map_name if hasattr(self,'env') else 'empty']
        # Optional goal-spread scaling for high-N density experiments (default 1.0 = unchanged).
        goal_scale = float(os.environ.get("GCBF_GOAL_SCALE", "1.0"))
        if goal_scale != 1.0:
            self.GOAL_SET = [[c * goal_scale for c in g] for g in self.GOAL_SET]
            print(f"[goal-scale] GOAL_SET x{goal_scale} -> {self.GOAL_SET}")

    def resample_goals(self, key, margin=None, traced=False, spacing=None, region=None,
                       size_range=None):
        """Resample goal-set positions uniformly within the area and rebuild the
        per-agent STL forms IN PLACE.

        Reuses the existing planner instance and diffusion model (no ``make_env``
        and no ``_setup_planner``/checkpoint restore), so the only work that
        repeats per call is rebuilding the lightweight ``stl_forms`` and the
        diffusion ``forward`` recompile that follows (the new predicate centers
        live inside ``stl_forms``, a jit-static arg in ``DiffusionMAPlanner.forward``).
        The number of goals is preserved so the spec/AST structure is identical
        across calls — only the goal coordinates change.

        :param key: JAX PRNG key. Vary per episode for a fresh predicate layout.
        :param margin: keep goal centers at least this far from the area border
            (default: half ``ENV_CONFIG['goal_size']`` so the goal rectangle stays
            inside the area while centers still span almost the full grid extent,
            e.g. ``[0.5, 3.5]`` for area 4 vs the fixed grid's ``{0, 2, 4}``).
        :param size_range: optional ``(lo, hi)`` multipliers of the default
            ``goal_size``. Each goal rectangle gets its own per-episode size
            ``goal_size * U(lo, hi)``. Sizes are baked into the RectReachPredicate
            AST (jit-static), so this is non-traced only — every episode pays the
            usual forward recompile, exactly like non-traced random centers.
        :returns: the new goal set (list of ``[x, y]``).
        """
        # Team/CaTL+ specs only partially support random goals: the CaTL+ team objective
        # (ma_stl_jax) is never threaded with cent_override, and the team formula builders
        # ignore goal_size_factors. Fail loudly for the unsupported combinations instead of
        # silently planning against a stale team objective.
        if getattr(self, 'team_spec_obj', None) is not None:
            if traced:
                raise ValueError("--traced-goals is not supported with team specs: the CaTL+ team "
                                 "objective does not receive cent_override (per-agent forms would use "
                                 "the resampled goals while the team score uses the original grid). "
                                 "Use non-traced --random-goals (centers only).")
            if size_range is not None:
                raise ValueError("--random-goals-size is not supported with team specs: the team "
                                 "formula builders (build_team_catl/build_per_agent_stl_forms) do not "
                                 "consult goal_size_factors.")
            if getattr(self.team_spec_obj, 'branches', None):
                raise ValueError("--random-goals is not supported with branched team specs "
                                 "(team_choiceseq3): their task geometry is defined on the fixed goal grid.")
        # Mark random-goals mode so spec handlers that otherwise hardcode the fixed grid
        # (e.g. the signal loop anchors) instead use the resampled goal_list -- needed so the
        # non-traced/stlpy path sees the SAME random signal goals as the traced cent_override roll.
        self.random_goals_active = True
        n_goals = len(self.GOAL_SET)
        if margin is None:
            margin = 0.5 * float(np.max(np.array([1, 1]) * ENV_CONFIG['goal_size']))
            if size_range is not None:
                # Keep even the LARGEST sampled rectangle fully inside the grid extent.
                margin *= float(size_range[1])
        # Sample within the GOAL-GRID extent the diffusion model trained on (DEFAULT_GOALS,
        # the {0,2,4} 4x4 grid in [0,4]) -- NOT the agent area_size. area_size (e.g. 6) only
        # sets where agents START; goals sampled beyond [0,4] (e.g. up to area-margin=5.5 at
        # area=6) are OUT-OF-DISTRIBUTION for the [0,4]-trained model and unreachable.
        _grid = np.asarray(self.DEFAULT_GOALS[self.env.map_name if hasattr(self, 'env') else 'empty'],
                           dtype=float)
        glo, ghi = float(_grid.min()), float(_grid.max())
        lo = glo + float(margin)
        hi = ghi - float(margin)
        pts = jrandom.uniform(key, shape=(n_goals, 2), minval=lo, maxval=hi)

        # Per-episode random predicate SIZES (region size, not just location).
        # Non-traced: stored as a {(x, y): factor} map keyed by goal center so the size follows
        # its goal through the per-agent rotation in _load_diff_spec (baked into the rebuilt AST).
        # Traced: rolled into self.goal_sizes below and fed with the centers as a dynamic
        # (cents, sizes) tuple override — zero recompile, same as centers alone.
        self.goal_size_factors = None
        self.goal_sizes = None
        fac = None
        if size_range is not None:
            if traced and spacing is not None and spacing > 0:
                raise ValueError("--random-goals-size with --traced-goals supports uniform sampling "
                                 "only (not --random-goals-spacing)")
            lo_f, hi_f = float(size_range[0]), float(size_range[1])
            fac = np.asarray(jrandom.uniform(jrandom.fold_in(key, 1), (n_goals,), minval=lo_f, maxval=hi_f))
            cents = np.asarray(pts)
            self.goal_size_factors = {(round(float(c[0]), 6), round(float(c[1]), 6)): float(f)
                                      for c, f in zip(cents, fac)}
            print(f"[random-goals] per-goal size factors U[{lo_f},{hi_f}]: "
                  f"{[round(float(f), 3) for f in fac]}")

        if traced:
            # Zero-recompile path: keep the existing structural stl_forms (do NOT rebuild)
            # and feed the new goal coordinates as a dynamic per-agent center array.
            n = n_goals
            if spacing is not None and spacing > 0:
                # Rigid placement matching the fixed signal demo's 2-unit sub-loops:
                # each agent's loop goals (names 0,1,2) are a randomly rotated+translated
                # copy of the canonical L `[[0,0],[s,0],[0,s]]`, so their pairwise
                # distances are EXACTLY {s, s, s*sqrt(2)} (= the demo's {2,2,2.83} for
                # s=2) regardless of orientation -- NOT grouped closer. The final goal
                # (name 3) is spread ~1.5s outward (the demo puts it on a distant grid
                # point). Remaining slots fill spaced points around the centre.
                s = float(spacing)
                L0 = np.array([[0., 0.], [s, 0.], [0., s]], dtype=np.float64)
                L0 = L0 - L0.mean(0)                      # centre the L on its centroid
                maxr = float(np.linalg.norm(L0, axis=1).max())
                cx = 0.5 * (glo + ghi)   # centre of the goal grid (NOT area centre)
                if region is not None and region > 0:
                    # Crowding mode: confine ALL agents' loop anchors to a tight central box of
                    # half-width `region`, so different agents' loops OVERLAP and they must
                    # coordinate (instead of each loop tiling a distinct part of the grid).
                    c_lo = max(maxr, cx - float(region)); c_hi = min(ghi - maxr, cx + float(region))
                else:
                    c_lo = max(lo, maxr); c_hi = min(hi, ghi - maxr)
                if c_hi < c_lo:
                    c_lo = c_hi = 0.5 * (lo + hi)
                kth, kc, kf, kfl = jrandom.split(key, 4)
                th = jrandom.uniform(kth, (self.num_agents,), minval=0.0, maxval=2 * np.pi)
                cs, sn = jnp.cos(th), jnp.sin(th)
                R = jnp.stack([jnp.stack([cs, -sn], -1), jnp.stack([sn, cs], -1)], -2)  # [A,2,2]
                centers = jrandom.uniform(kc, (self.num_agents, 2), minval=c_lo, maxval=c_hi)
                loop = centers[:, None, :] + jnp.einsum('pij,kj->pki', R, jnp.asarray(L0))  # [A,3,2]
                if region is not None and region > 0:
                    # Keep finals in the SAME tight box so agents can't fan out to widely-separated
                    # finals (which would decongest them); forces them to stay crowded.
                    f_lo = max(glo, cx - float(region)); f_hi = min(ghi, cx + float(region))
                    final = jrandom.uniform(kf, (self.num_agents, 2), minval=f_lo, maxval=f_hi)[:, None, :]
                else:
                    fang = jrandom.uniform(kf, (self.num_agents,), minval=0.0, maxval=2 * np.pi)
                    final = jnp.clip(centers + 1.5 * s * jnp.stack([jnp.cos(fang), jnp.sin(fang)], -1),
                                     glo, ghi)[:, None, :]  # [A,1,2]
                nfill = n - 4
                parts = [loop, final]
                if nfill > 0:
                    fill = jnp.clip(centers[:, None, :]
                                    + jrandom.uniform(kfl, (self.num_agents, nfill, 2), minval=-s, maxval=s),
                                    glo, ghi)
                    parts.append(fill)
                self.goal_centers = jnp.concatenate(parts, axis=1)[:, :n, :]
                self.GOAL_SET = [[float(x), float(y)] for x, y in np.asarray(self.goal_centers[0])]
                a0 = np.asarray(self.goal_centers[0, :3])
                d = [float(np.linalg.norm(a0[i] - a0[j])) for i, j in ((0, 1), (0, 2), (1, 2))]
                print(f"[traced-goals] rigid signal spacing={s}; agent0 loop dists "
                      f"={d[0]:.2f},{d[1]:.2f},{d[2]:.2f} (demo {s:.2f},{s:.2f},{s * np.sqrt(2):.2f}); "
                      f"goals={[[round(c, 2) for c in g] for g in self.GOAL_SET[:4]]} (no recompile)")
            else:
                # Per-agent order matches _load_diff_spec's rotation: agent i's predicate
                # name k references goals_to_use[k] = base[(i % n + k) % n].
                roll = (np.arange(n)[None, :] + (np.arange(self.num_agents)[:, None] % n)) % n  # [A, n] static
                self.goal_centers = pts[roll]  # [num_agents, n_goals, 2] (dynamic at forward)
                if fac is not None:
                    # Sizes roll with their goals (same roll as centers) and flow as the second
                    # element of the (cents, sizes) tuple override — dynamic, zero recompile.
                    default_size = np.array([1, 1]) * ENV_CONFIG['goal_size']
                    sizes_base = jnp.asarray(fac)[:, None] * jnp.asarray(default_size)[None, :]  # [n, 2]
                    self.goal_sizes = sizes_base[roll]  # [num_agents, n_goals, 2]
                self.GOAL_SET = [[float(x), float(y)] for x, y in np.asarray(pts)]
                print(f"[traced-goals] {n_goals} goals in [{lo:.2f},{hi:.2f}]^2: "
                      f"{[[round(x, 2) for x in g] for g in self.GOAL_SET]} (no recompile)")
            return self.GOAL_SET

        self.GOAL_SET = [[float(x), float(y)] for x, y in np.asarray(pts)]
        # Rebuild per-agent STL forms from the new goals (reuses self.planner).
        self._init_plan(spec=self.stl_string, agent_goals=self._set_agent_goals(),
                        ma_stl_spec=self.ma_stl_spec, key=key)
        self.goal_step = self.max_step // self.stl_forms[0].end_time()
        print(f"[random-goals] resampled {n_goals} goals in [{lo:.2f},{hi:.2f}]^2: {self.GOAL_SET}")
        return self.GOAL_SET

    def _map_spec(self, spec: str):
        """Map the spec to a full name"""
        if self.spec_regex.match(spec) is None:
            # Add a default time interval if not specified
            spec = f"{spec}_t30"
        if self.cover_regex.match(spec):
            return "Cover"
        elif self.seq_regex.match(spec):
            return "Sequence"
        elif self.loop_regex.match(spec):
            return "Loop"
        elif self.branch_regex.match(spec):
            return "Branch"
        elif self.signal_regex.match(spec):
            return "Signal"
        elif self.reach_regex.match(spec):
            return "Reach"
        elif self.ereach_regex.match(spec):
            return "EReach"  # Exact reach
        elif self.stay_regex.match(spec):
            return "Stay"  # F(G A) persistence
        elif self.auntil_regex.match(spec):
            return "AvoidUntil"  # (!A U B) & F A
        elif self.cuntil_regex.match(spec):
            return "CoverUntil"  # AND_i (!R U L_i) & F R  -- avoid right column until left column covered
        elif self.mixed_spec_regex.match(spec):
            return "Mixed"  # Exact reach
        else:
            raise ValueError(f"Spec {spec} not recognized")

    stl_forms = None  # For multiple agent
    logger = logging.getLogger(__name__)

    @property
    def extra_config(self):
        """Extra configuration for the environment from the wrapper."""
        return ENV_CONFIG

    @property
    def args_of_interest_all(self):
        """Return the arguments of interest for the planner. For logging results on wandb (incl.args_of_interest)."""
        return ['stl_shrink_factor', 'stl_hardness', 'stl_within_boundary'] + super().args_of_interest_all

    @property
    def args_of_interest(self):
        """Return the arguments of interest for the planner. This is used for logging results in a csv file."""
        return ['stl_mixed_spec_mode'] + super().args_of_interest

    def _load_diff_spec(self, spec: str, goal_list: list = None, agent_id: int = 0, key=None):
        """Load STL spec from the diffspec library

        :param spec: name of spec to load
        :type spec: str
        :param goal_list: list of goals to use for different agents
        :type goal_list: list or None
        :param agent_id: agent id (used to decide goal ordering for current spec)
        :type agent_id: int
        """
        from ds.stl_jax import STL, RectReachPredicate

        # STL spec loading
        # We have two options here:
        # 1. High level spec over time intervals
        # 2. Spec over all time

        # Simple sample spec for now
        goal_size = np.array([1, 1]) * ENV_CONFIG['goal_size']
        # Per-episode random predicate sizes (--random-goals-size): resample_goals stores a
        # {(x, y): factor} map keyed by goal center; goals not in the map keep the default size.
        _size_factors = getattr(self, 'goal_size_factors', None) or {}
        size_of = lambda cent: goal_size * _size_factors.get(
            (round(float(cent[0]), 6), round(float(cent[1]), 6)), 1.0)
        get_goal_pred = lambda cent, i: STL(RectReachPredicate(np.array(cent), np.array(size_of(cent)), i,
                                                               shrink_factor=self.stl_shrink_factor))

        time_int = 15  # Default time interval
        original_goal_predicates = None
        if goal_list is None:
            map_name = self.env.map_name if hasattr(self.env, 'map_name') else 'empty'
            goals_to_use = self.DEFAULT_GOALS[map_name]
        else:
            # Agent i uses goal_list rotated by i mod n: its predicate named k is goal_list[(i+k) % n].
            # resample_goals(traced=True) reproduces exactly this permutation as
            # goal_centers[i, k] = pts[(i+k) % n], so cent_override[k] replaces predicate k's centre.
            rotate_index = agent_id % len(goal_list)
            goals_to_use = goal_list[rotate_index:] + goal_list[:rotate_index]
            original_goal_predicates = list(map(get_goal_pred, goal_list, range(len(goal_list))))

        # Signal demo (fixed grid): every agent gets an equal-length 3-goal loop made of the
        # corners of a 2x2 sub-square of the training grid ({0,2,4}^2), anchored at one of the
        # nine grid points (round-robin over agent id), plus a distinct final "until" goal.
        if self.signal_regex.match(spec) and getattr(self, 'random_goals_active', False) and goal_list is not None:
            # Random-goals mode: use the SAME rotated goal_list as every other spec (and as the
            # traced cent_override roll) so the non-traced/stlpy path and traced diffusion get the
            # IDENTICAL random signal goals. Loop = first 3 (names 0,1,2), final = 4th (name 3).
            ri = agent_id % len(goal_list)
            goals_to_use = goal_list[ri:] + goal_list[:ri]
        elif self.signal_regex.match(spec):
            # Spread the coverage loops AND the final goal across the 9-grid (like
            # m2loop3) so agents don't all crowd in a few overlapping loops at
            # large N. Each agent gets a compact in-distribution 3-goal loop
            # anchored at a distinct grid point (9 loops -> ~N/9 agents per loop,
            # vs N/4 with the 4 sub-squares), plus a spread final (until) goal.
            signal_loops = [
                [[0., 0.], [2., 0.], [0., 2.]],   # anchor (0,0)
                [[2., 0.], [4., 0.], [2., 2.]],   # anchor (2,0)
                [[4., 0.], [2., 0.], [4., 2.]],   # anchor (4,0)
                [[0., 2.], [0., 4.], [2., 2.]],   # anchor (0,2)
                [[2., 2.], [4., 2.], [2., 4.]],   # anchor (2,2) center
                [[4., 2.], [4., 0.], [2., 2.]],   # anchor (4,2)
                [[0., 4.], [2., 4.], [0., 2.]],   # anchor (0,4)
                [[2., 4.], [0., 4.], [2., 2.]],   # anchor (2,4)
                [[4., 4.], [2., 4.], [4., 2.]],   # anchor (4,4)
            ]
            grid9 = [[0., 0.], [2., 0.], [4., 0.], [0., 2.], [2., 2.],
                     [4., 2.], [0., 4.], [2., 4.], [4., 4.]]
            loop = signal_loops[agent_id % len(signal_loops)]
            # The final (until) goal MUST be a distinct 4th goal NOT in the loop,
            # else "reach g4" is satisfied just by looping. Pick from the grid
            # points outside the loop, spread per agent.
            loop_pts = {tuple(p) for p in loop}
            remaining = [p for p in grid9 if tuple(p) not in loop_pts]
            final_goal = remaining[(agent_id * 5) % len(remaining)]
            goals_to_use = loop + [final_goal]
            _lp = ''.join(SIGNAL_GOAL_LETTERS[tuple(p)] for p in loop)
            self.logger.info(f"signal agent {agent_id}: loop {_lp} (x2) -> final "
                             f"{SIGNAL_GOAL_LETTERS[tuple(final_goal)]}")

        goal_predicates = list(map(get_goal_pred, goals_to_use, range(len(goals_to_use))))
        if _size_factors:
            _szs = [round(float(g.get_all_predicates()[0].size[0]), 3) for g in goal_predicates]
            print(f"[random-goals] agent {agent_id} predicate sizes: {_szs}")
        # Signal demo: each agent already has its own distinct goals, so make the
        # shared-goal step a per-agent no-op (don't force a common first goal).
        if self.signal_regex.match(spec):
            original_goal_predicates = goal_predicates
        get_goal_centers = lambda goal_pred: np.array([g.get_all_predicates()[0].cent for g in goal_pred]).tolist()

        if self.reach_regex.match(spec):
            # Single goal reach spec
            stl_form, time_int = self._load_reach_stl_form(goal_predicates, spec, time_int)
        elif self.ereach_regex.match(spec):
            # Single goal exact reach spec
            stl_form, time_int = self._load_ereach_stl_form(goal_predicates, spec, time_int)
        elif self.cover_regex.match(spec):
            # cover_spec=
            goal_predicates, stl_form, time_int = self._load_cover_stl_form(goal_predicates, original_goal_predicates,
                                                                            spec, time_int)
        elif self.seq_regex.match(spec):
            # seq_spec
            goal_predicates, stl_form, time_int = self._load_seq_stl_form(goal_predicates, original_goal_predicates,
                                                                          spec, time_int)

        elif self.loop_regex.match(spec):
            # loop_spec
            goal_predicates, stl_form, time_int = self._load_loop_stl_form(goal_predicates, original_goal_predicates,
                                                                           spec, time_int)

            # Old type of loop  # loop_spec = goal_predicates[0] & goal_predicates[1].eventually(0, 1)  # for i in range(1, num_loops):#len(goal_predicates)):  #     loop_spec |= goal_predicates[i] & goal_predicates[(i + 1) % len(goal_predicates)].eventually(0, 1)  # loop_spec = loop_spec.always(1, time_int - 1)  # stl_form = loop_spec

        elif self.signal_regex.match(spec):
            # signal_spec
            goal_predicates, stl_form, time_int = self._load_signal_stl_form(goal_predicates, original_goal_predicates,
                                                                             spec)

        elif self.branch_regex.match(spec):
            # branch_spec
            goal_predicates, stl_form, time_int = self._load_branch_stl_form(goal_predicates, original_goal_predicates,
                                                                             spec, time_int)
        elif self.stay_regex.match(spec):
            # persistence spec: F( G A )
            goal_predicates, stl_form, time_int = self._load_stay_stl_form(goal_predicates, spec)
        elif self.auntil_regex.match(spec):
            # avoid-until-reach spec: (!A U B) & F A
            goal_predicates, stl_form, time_int = self._load_auntil_stl_form(goal_predicates,
                                                                             original_goal_predicates, spec)
        elif self.cuntil_regex.match(spec):
            # cover-until spec: avoid right column until left column covered
            goal_predicates, stl_form, time_int = self._load_cuntil_stl_form(goal_predicates, spec)
        elif self.mixed_spec_regex.match(spec):
            # This is a mixed spec, so we parse it.
            if key is None:
                raise ValueError("A JAX random key must be provided to sample from a mixed spec.")

            match = self.mixed_spec_regex.match(spec)
            spec_parts_string, time_int_str = match.groups()
            time_int = int(time_int_str)

            # Split the string of specs into a list of individual components
            spec_components = spec_parts_string.split('-')

            # Use the JAX key to sample one of the spec components
            key, subkey = jrandom.split(key)
            # 1. Get the number of possible choices.
            n_components = len(spec_components)
            # 2. Use JAX to generate a random integer (index) in the valid range.
            random_index = jrandom.randint(subkey, shape=(), minval=0, maxval=n_components)
            # 3. Use the generated index to select the component from the Python list.
            sampled_component = spec_components[random_index]

            # The spec string must be reconstructed with the time interval for the helper functions to parse it
            sampled_spec_string = f"{sampled_component}_t{time_int}"

            self.logger.info(
                f"Mixed spec '{spec}' detected. Sampled component: '{sampled_spec_string}' for agent {agent_id}.")

            # Recursively call this same function with the new, specific spec string.
            # This will delegate the call to the appropriate helper (_load_cover_stl_form, etc.)
            return self._load_diff_spec(
                spec=sampled_spec_string,
                goal_list=goal_list,
                agent_id=agent_id,
                key=key  # Pass the updated key
            )
        else:
            # Default fixed loop spec
            # form is the formula goal_1 eventually in 0 to 5 and goal_2 eventually in 0 to 5
            # and that holds always in 0 to 8
            # In other words, the path will repeatedly visit goal_1 and goal_2 in 0 to 13
            stl_form = (goal_predicates[0].eventually(0, 5) & goal_predicates[1].eventually(0, 5)).always(0, 8)

        self.spec_name = self._map_spec(spec)

        if hasattr(self,'env') and self.stl_within_boundary:
            # Add boundary predicate
            boundary_predicate = self._create_boundary_predicate(
                goal_list, goal_size * (max(_size_factors.values()) if _size_factors else 1.0))
            stl_form = stl_form & boundary_predicate.always(0, time_int)

        self.logger.info(f"Loaded {self.spec_name} spec: {spec} "
                         f"STL spec: {stl_form}\n"
                         f"Goals: {get_goal_centers(goal_predicates)}")
        return stl_form, key

    def _load_reach_stl_form(self, goal_predicates, spec, time_int):
        """Reach: Reach the goal at any time within the time interval"""
        # Extract time interval
        time_int = int(self.reach_regex.match(spec).groups()[0])
        stl_form = goal_predicates[0].eventually(0, time_int)
        self.logger.info(f"Loaded reach spec {spec} with time interval {time_int}")
        return stl_form, time_int

    def _load_ereach_stl_form(self, goal_predicates, spec, time_int):
        """Exact reach: Reach the goal exactly at the end of the time interval"""
        # Extract time interval
        time_int = int(self.ereach_regex.match(spec).groups()[0])
        stl_form = goal_predicates[0].eventually(time_int-1, time_int)
        self.logger.info(f"Loaded exact reach spec {spec} at end of time interval {time_int}")
        return stl_form, time_int

    def _load_cover_stl_form(self, goal_predicates, original_goal_predicates, spec, time_int):
        # Extract number of goals and time interval
        num_goals, time_int = map(int, self.cover_regex.match(spec).groups())
        goal_predicates = self._set_shared_goal_predicates(spec, original_goal_predicates, goal_predicates,
                                                           num_goals)
        stl_form = goal_predicates[0].eventually(0, time_int)
        for i, goal in enumerate(goal_predicates[1:num_goals]):
            stl_form = stl_form & goal.eventually(0, time_int)
        self.logger.info(f"Loaded cover spec {spec} with {num_goals} goals, time interval {time_int}")
        return goal_predicates, stl_form, time_int

    def _load_seq_stl_form(self, goal_predicates, original_goal_predicates, spec, time_int):
        # Extract number of goals and time interval
        num_goals, time_int = map(int, self.seq_regex.match(spec).groups())
        interval_length = time_int // num_goals
        goal_predicates = self._set_shared_goal_predicates(spec, original_goal_predicates, goal_predicates,
                                                           num_goals)
        # Start from end to get sensible goal predicate order
        stl_form = goal_predicates[num_goals - 1].eventually(max(0, (num_goals - 1) * interval_length), time_int)
        for i, goal in enumerate(reversed(goal_predicates[:num_goals - 1])):
            time_ind = num_goals - 1 - i
            stl_form = goal.eventually(max(0, (time_ind - 1) * interval_length),  # try minus to fix weird bug
                                       time_ind * interval_length) & stl_form
        self.logger.info(f"Loaded seq spec {spec} with {num_goals} goals, time interval {time_int}")
        return goal_predicates, stl_form, time_int

    def _load_loop_stl_form(self, goal_predicates, original_goal_predicates, spec, time_int):
        """Loop spec: cover the ``num_goals`` goals repeatedly via ``always``.

        Matches the paper's loop formula (for ``num_loops=2``):

            G[0, T/2] ( F[0, T/2](X) & F[0, T/2](Y) & F[0, T/2](Z) )

        i.e. an *always* over a sliding window that must, at every shift, cover
        all goals -> the agent loops ~``num_loops`` times. The ``always`` and
        ``eventually`` windows are both ``per_loop = T / num_loops`` (symmetric
        T/2 for the 2-loop case), and the always range is ``(num_loops-1)*per_loop``
        so ``num_loops`` covers fit and the formula horizon stays <= T (important
        for the fixed-length diffusion plan). NOT an enumerated ordered sequence.
        """
        # Extract number of loops, goals and time interval
        num_loops, num_goals, time_int = map(int, self.loop_regex.match(spec).groups())
        per_loop = time_int // num_loops
        goal_predicates = self._set_shared_goal_predicates(spec, original_goal_predicates, goal_predicates,
                                                           num_goals)
        stl_form = goal_predicates[0].eventually(0, per_loop)
        for i, goal in enumerate(goal_predicates[1:num_goals]):
            stl_form = stl_form & goal.eventually(0, per_loop)
        # always window = (num_loops-1)*per_loop -> G and F windows both == per_loop
        # (== T/2 for num_loops=2), matching the paper's symmetric loop formula.
        stl_form = stl_form.always(0, max(1, (num_loops - 1) * per_loop))
        self.logger.info(f"Loaded loop spec {spec} with {num_loops} loops, {num_goals} goals,"
                         f" time interval {time_int}, per loop {per_loop}")
        return goal_predicates, stl_form, time_int

    def _load_stay_stl_form(self, goal_predicates, spec):
        """Persistence (stabilization) spec: ``F[0, T-w]( G[0, w] A )`` — the
        eventually-always fragment (FG). The agent must reach goal A and then
        REMAIN inside it for ``w`` consecutive plan steps, i.e. the plan tail
        must park on the region rather than pass through it. ``w`` comes from
        the spec name (``mstay5_t15`` -> w=5). Bounded-horizon counterpart of
        the classic F(G(A)); the dual GF (always-eventually / recurrent
        revisits) is the existing loop spec ``G[0,·](F[0,·] g_i)``.
        """
        dwell, time_int = map(int, self.stay_regex.match(spec).groups())
        dwell = max(1, min(dwell, time_int - 1))
        stl_form = goal_predicates[0].always(0, dwell).eventually(0, time_int - dwell)
        self.logger.info(f"Loaded stay spec {spec}: F[0,{time_int - dwell}] G[0,{dwell}] A, "
                         f"time interval {time_int}")
        return goal_predicates, stl_form, time_int

    def _load_auntil_stl_form(self, goal_predicates, original_goal_predicates, spec):
        """Avoid-until-reach chain: ``AND_k (¬g_k U[0, tu_k] g_{k-1}) & F[tu_K, T] g_K``.

        ``mauntil`` (K=1) is the literal ``!A U B``: goal A
        (= goals_to_use[1]) is FORBIDDEN until goal B (= goals_to_use[0]) has
        been visited — ``¬A`` is a literal negation leaf (``~`` operator) —
        and A must then be visited in the tail window, so the constraint is
        non-vacuous and the B-before-A order is enforced.

        ``mauntilK`` (K >= 2, e.g. ``mauntil2``) chains K such stages over
        K+1 goals into a keyed gauntlet: g_k is forbidden until g_{k-1} has
        been reached (each conjunct its own literal ¬-until), finishing
        inside g_K. The g_0 -> g_1 -> ... -> g_K visit order is forced
        purely by the negation-until conjunction — no F-sequencing of the
        intermediate goals. Until deadlines split the horizon evenly:
        ``tu_k = k*T//(K+1) + 1`` (K=1 reproduces the original
        ``tu = T//2 + 1``).

        Goal sharing (``mixed_spec_mode``) applies like the seq/cover/loop
        builders: ``first1`` makes every agent share the unrotated
        ``goal_list[0]`` as its key g_0 (all agents crowd the same key);
        ``last1`` shares the final door g_K (all agents finish in the same
        region).
        """
        k_str, t_str = self.auntil_regex.match(spec).groups()
        time_int = int(t_str)
        num_stages = int(k_str) if k_str else 1
        assert num_stages >= 1, f"auntil spec '{spec}' needs at least 1 stage"
        goal_predicates = self._set_shared_goal_predicates(spec, original_goal_predicates, goal_predicates,
                                                           num_stages + 1)
        assert len(goal_predicates) > num_stages, (
            f"auntil spec '{spec}' needs {num_stages + 1} goals "
            f"({num_stages} chained stages + the free key goal), got {len(goal_predicates)}")
        deadlines = [k * time_int // (num_stages + 1) + 1 for k in range(1, num_stages + 1)]
        stl_form = None
        for k, tu in zip(range(1, num_stages + 1), deadlines):
            stage = (~goal_predicates[k]).until(goal_predicates[k - 1], 0, tu)
            stl_form = stage if stl_form is None else stl_form & stage
        stl_form = stl_form & goal_predicates[num_stages].eventually(deadlines[-1], time_int)
        body = ' & '.join(f"(!g{k} U[0,{tu}] g{k - 1})"
                          for k, tu in zip(range(1, num_stages + 1), deadlines))
        self.logger.info(f"Loaded avoid-until spec {spec} ({num_stages} stage(s)): "
                         f"{body} & F[{deadlines[-1]},{time_int}] g{num_stages}")
        return goal_predicates, stl_form, time_int

    def _load_cuntil_stl_form(self, goal_predicates, spec):
        """Cover-until spec: avoid the RIGHT-column goals until EVERY LEFT-column
        goal has been visited at least once (a cover), then visit some right goal.

            AND_i [ (¬R1 ∧ ¬R2 ∧ ¬R3)  U[0, tu]  L_i ]   ∧   F[tu, T] (R1 ∨ R2 ∨ R3)

        Each conjunct's until picks its own witness time t_i for left goal L_i with
        the right column avoided on [0, t_i); the conjunction therefore keeps the
        right column forbidden until the LAST left goal is covered (order-free).
        The trailing eventually forces entry into the right column afterwards, so
        the avoidance actually binds. Column membership is classified by goal
        x-coordinate (min-x = left, max-x = right), which is invariant to the
        per-agent goal rotation — every agent gets the same column sets.
        ``tu = 2T//3`` leaves the tail third for the right-column visit.
        """
        k_str, time_int_str = self.cuntil_regex.match(spec).groups()
        time_int = int(time_int_str)
        cents = np.array([g.get_all_predicates()[0].cent for g in goal_predicates])
        xs = cents[:, 0]
        left_idx = [i for i in range(len(goal_predicates)) if xs[i] == xs.min()]
        right_idx = [i for i in range(len(goal_predicates)) if xs[i] == xs.max()]
        assert left_idx and right_idx, f"cuntil needs distinct min/max-x goal columns, got xs={xs}"
        # K-goal variant (mcuntil2): cover only the K bottom-most (lowest-y) left goals.
        left_idx = sorted(left_idx, key=lambda i: cents[i, 1])
        if k_str:
            left_idx = left_idx[:max(1, min(int(k_str), len(left_idx)))]

        avoid_right = ~goal_predicates[right_idx[0]]
        for j in right_idx[1:]:
            avoid_right = avoid_right & ~goal_predicates[j]
        reach_right = goal_predicates[right_idx[0]]
        for j in right_idx[1:]:
            reach_right = reach_right | goal_predicates[j]

        tu = 2 * time_int // 3
        stl_form = avoid_right.until(goal_predicates[left_idx[0]], 0, tu)
        for i in left_idx[1:]:
            stl_form = stl_form & avoid_right.until(goal_predicates[i], 0, tu)
        stl_form = stl_form & reach_right.eventually(tu, time_int)
        self.logger.info(f"Loaded cover-until spec {spec}: avoid right col {right_idx} until "
                         f"cover of left col {left_idx} (U[0,{tu}] each), then F[{tu},{time_int}] right")
        return goal_predicates, stl_form, time_int

    def _load_signal_stl_form(self, goal_predicates, original_goal_predicates, spec):
        """Signal spec: the ``2loop3`` cover loop, then reach the final goal
        (index ``num_goals``) via *until*.

            loop_form & (progress  U[0, loop_horizon+1]  final_goal)

        - ``loop_form`` is the same cover loop as the standalone loop spec
          (``_load_loop_stl_form``): ``G[0,(num_loops-1)*per_loop]( F[0,per_loop]X
          & F[0,per_loop]Y & F[0,per_loop]Z )`` -- a sliding ``always`` over a
          cover, so the agent loops ~``num_loops`` times. Here ``per_loop`` is
          sized over ``loop_horizon`` (< T) so the tail is left free for the final
          goal. (Order is not enforced -- this is a cover, matching 2loop3.)
        - The *until* keeps the agent making progress (reaching some goal within a
          short window, i.e. not stalling) **until** it reaches the final goal at
          the end. ``progress`` includes the final goal, so the invariant holds
          through the final leg and the until is robustly satisfiable.

        Fixed plan length: the diffusion model emits a fixed ``T``-step plan
        (``T = spec_len``, 15 for the default model). The until window therefore
        ends at ``loop_horizon + 1`` (NOT ``time_int + 1``) and the forward
        ``progress`` window is clamped so the formula never reads past index
        ``time_int``; otherwise ``until`` unrolls a forward window past the end of
        the fixed-length plan and eval slices out of bounds (size-0 gather).

        Example: ``2signal3_t15`` -> 2 cover loops over goals {0,1,2}, then reach
        goal 3. A satisfying trajectory visits g0,g1,g2, g0,g1,g2, g3.
        """
        # Extract number of loops and goals. The diffusion model emits a fixed
        # 15-step plan, so the signal spec is always evaluated at T=15 regardless
        # of the _t<n> in the spec string (which defaults to _t30 via _map_spec
        # when no interval is given). Force the time interval to 15.
        num_loops, num_goals, _t_parsed = map(int, self.signal_regex.match(spec).groups())
        time_int = SIGNAL_PLAN_LENGTH
        goal_predicates = self._set_shared_goal_predicates(spec, original_goal_predicates, goal_predicates,
                                                           num_goals)
        # One extra goal beyond the loop goals is the final ("fourth") goal.
        final_idx = num_goals
        assert len(goal_predicates) > final_idx, (
            f"signal spec '{spec}' needs at least {num_goals + 1} goals "
            f"({num_goals} loop goals + 1 final goal), got {len(goal_predicates)}")

        # Timing budget: num_loops loops over num_goals goals + 1 final visit.
        total_visits = num_loops * num_goals + 1
        per_visit = max(1, time_int // total_visits)
        per_loop = num_goals * per_visit      # steps for one full loop over the loop goals
        loop_horizon = num_loops * per_loop   # all loops finish by here

        # 1) Loop term: same cover loop as the standalone 2loop3 spec
        #    (G[0,(num_loops-1)*per_loop]( F[0,per_loop]X & F[0,per_loop]Y & ... )),
        #    but with per_loop sized over loop_horizon (< T) so the tail is free for
        #    the final goal. The sliding always forces ~num_loops covers.
        cover0 = goal_predicates[0].eventually(0, per_loop)
        for g in range(1, num_goals):
            cover0 = cover0 & goal_predicates[g].eventually(0, per_loop)
        loop_form = cover0.always(0, max(1, (num_loops - 1) * per_loop))

        # 2) Until term: keep progressing UNTIL the final goal, but the final goal
        #    may only be reached AFTER the loops -- the until window is
        #    [loop_horizon, time_int), so g4 cannot satisfy it during the loop
        #    phase. This enforces "reach g4 only after the num_loops loops are up".
        #    The forward progress window is clamped so eval stays within the fixed
        #    T-step plan (max read index <= time_int - 1).
        until_start = loop_horizon
        until_end = time_int
        # until unrolls phi1's forward window at every t' in [until_start, until_end);
        # the last t' = until_end-1 reads [t', t'+prog_w), so prog_w <= time_int-until_end+1.
        prog_w = max(1, min(per_visit + 1, time_int - until_end + 1))
        progress = goal_predicates[0]
        for g in range(1, num_goals + 1):
            progress = progress | goal_predicates[g]
        progress_within = progress.eventually(0, prog_w)
        final_reach = goal_predicates[final_idx]
        until_form = progress_within.until(final_reach, until_start, until_end)

        stl_form = loop_form & until_form
        self.logger.info(f"Loaded signal spec {spec} with {num_loops} loops, {num_goals} goals,"
                         f" time interval {time_int}, per loop {per_loop}, until_end {until_end} (until-based)")
        return goal_predicates, stl_form, time_int

    def _load_branch_stl_form(self, goal_predicates, original_goal_predicates, spec, time_int):
        # Extract number of branches, goals per branch and time interval
        num_branches, num_goals, time_int = map(int, self.branch_regex.match(spec).groups())
        stl_form_branches = []
        goal_predicates = self._set_shared_goal_predicates(spec, original_goal_predicates, goal_predicates,
                                                           num_goals)
        grouped_goals = [goal_predicates[i:i + num_goals] for i in range(0, len(goal_predicates), num_goals)]
        for goals_in_branch in grouped_goals[:num_branches]:
            stl_form = goals_in_branch[0].eventually(0, time_int)
            for i, goal in enumerate(goals_in_branch[1:num_goals]):
                stl_form = stl_form & goal.eventually(0, time_int)
            stl_form_branches.append(stl_form)
        stl_form = ft.reduce(lambda x, y: x | y, stl_form_branches)
        self.logger.info(f"Loaded branch spec {spec} with {num_branches} branches, {num_goals} goals,"
                         f" time interval {time_int}")
        return goal_predicates, stl_form, time_int

    def _create_boundary_predicate(self, goal_list, goal_size):
        """Create a boundary predicate for the environment that is satisfied if the agent is within the boundary"""
        from ds.stl_jax import STL, RectReachPredicate
        min_pred = np.array(goal_list).min(axis=0) - goal_size / 2
        max_pred = np.array(goal_list).max(axis=0) + goal_size / 2

        lbc = np.minimum(np.array(self.env._xy_min), min_pred)  # left bottom corner
        ruc = np.maximum(np.array(self.env._xy_max), max_pred)  # right upper corner

        map_center = (lbc + ruc) / 2
        map_size = ruc - lbc

        # Assume predicates with name < 0 are to be ignored during plotting
        boundary_predicate = STL(RectReachPredicate(map_center, map_size, -1))
        return boundary_predicate


from ds.ma_stl_jax import CaTLPlus
from gcbfplus.stl.team_spec import is_team_spec, parse_team_spec
from gcbfplus.stl.team_allocator import allocate_tasks, build_team_catl, build_per_agent_stl_forms, \
    build_diffusion_stl_forms


class MASTLMixin(STLMixin):
    """Contains functions for loading MA-STL specs like CaTL+ and other multi-agent STL specs"""

    spec_name = None

    stl_forms = None  # For multiple agent
    team_spec_obj = None  # TeamSpec (set when using team specs)
    team_allocation = None  # AllocationResult (set after allocation)
    allocation_strategy = 'greedy'  # Allocation strategy for team specs
    logger = logging.getLogger(__name__)

    # @property
    # def extra_config(self):
    #     """Extra configuration for the environment from the wrapper."""
    #     return super().extra_config

    def _is_team_spec(self, spec: str) -> bool:
        """Check if a spec string is a team specification."""
        return is_team_spec(spec)

    def _load_team_spec(self, spec: str, goal_list=None, allocation_strategy: str = 'greedy'):
        """Parse team spec, build CaTLPlus ma_stl_form, set per-agent stl_forms."""
        goal_set = goal_list if goal_list else self.GOAL_SET
        self.team_spec_obj = parse_team_spec(spec, self.num_agents, goal_set)
        self.allocation_strategy = allocation_strategy

        goal_size = np.array([1, 1]) * ENV_CONFIG['goal_size']
        # Evaluation CaTL+: avoid_expansion=1.0 (actual goal size for fair scoring)
        self.ma_stl_form = build_team_catl(self.team_spec_obj, goal_size, self.stl_shrink_factor,
                                          add_avoidance=self.team_avoid,
                                          avoid_expansion=1.0)
        # Guidance CaTL+: expanded avoidance for conservative planner margin
        if self.team_avoid and self.avoid_expansion != 1.0:
            self.ma_stl_form_guidance = build_team_catl(self.team_spec_obj, goal_size, self.stl_shrink_factor,
                                                        add_avoidance=True,
                                                        avoid_expansion=self.avoid_expansion)
        else:
            self.ma_stl_form_guidance = self.ma_stl_form

        # Joint acceptance GATE uses the exact (non-smoothed) robustness: hard top-k
        # m-th-best per task, true min/max across tasks/branches. The smoothed
        # eval_train stays gradient-only (ma_stl_spec_eval). Built once here so the
        # jit-static callable identity is stable across episodes (no recompile).
        import functools as _ft
        self.ma_stl_gate_eval = _ft.partial(self.ma_stl_form_guidance.eval,
                                            approx_method='true')

        # Per-task repair data for counting-aware resampling (disjunctive mode):
        # guidance-expansion Task list in task_id order. Built once here — never
        # rebuilt per episode, so the jit-static tuples stay identical across calls.
        _, _gtasks = build_team_catl(
            self.team_spec_obj, goal_size, self.stl_shrink_factor,
            add_avoidance=self.team_avoid,
            avoid_expansion=(self.avoid_expansion if self.team_avoid else 1.0),
            return_tasks=True)
        self.team_task_specs_guidance = tuple(t.spec for t in _gtasks)
        self.team_task_ms = tuple(int(t.num_satisfied_agents) for t in _gtasks)
        self.team_task_branches = (tuple(map(tuple, self.team_spec_obj.branches))
                                   if self.team_spec_obj.branches is not None else None)

        # Always build allocated forms for evaluation (consistent scoring)
        dummy_positions = np.zeros((self.num_agents, 2))
        allocation = allocate_tasks(self.team_spec_obj, dummy_positions, strategy=self.allocation_strategy)
        self.team_allocation = allocation
        # Eval forms use expansion=1.0 (actual goal size for fair scoring)
        self.stl_forms_eval, _ = build_per_agent_stl_forms(
            self.team_spec_obj, allocation, goal_size, self.stl_shrink_factor, self.num_agents,
            add_avoidance=self.team_avoid, avoid_expansion=1.0)
        # Guidance avoid regions use expansion factor (conservative margin for planner)
        _, self.agent_avoid_regions = build_per_agent_stl_forms(
            self.team_spec_obj, allocation, goal_size, self.stl_shrink_factor, self.num_agents,
            add_avoidance=self.team_avoid, avoid_expansion=self.avoid_expansion)

        if self.team_disjunctive:
            # Disjunctive forms for guidance only — CaTL+ gradient discovers allocation
            # With avoid: each disjunct includes avoidance of other tasks' goals
            # Uses expanded regions for conservative guidance
            self.stl_forms_guidance = build_diffusion_stl_forms(
                self.team_spec_obj, goal_size, self.stl_shrink_factor, self.num_agents,
                add_avoidance=self.team_avoid, avoid_expansion=self.avoid_expansion)
            print(f"  Using disjunctive per-agent spec for GUIDANCE (allocated spec for EVAL)"
                  f"{' + avoid' if self.team_avoid else ''}")
        else:
            # Non-disjunctive: build allocated guidance forms with expansion
            # (eval forms use 1.0, guidance forms use self.avoid_expansion)
            if self.team_avoid and self.avoid_expansion != 1.0:
                self.stl_forms_guidance, _ = build_per_agent_stl_forms(
                    self.team_spec_obj, allocation, goal_size, self.stl_shrink_factor, self.num_agents,
                    add_avoidance=True, avoid_expansion=self.avoid_expansion)
            else:
                self.stl_forms_guidance = None

        # stl_forms used by _score_set_of_histories for evaluation — always allocated
        self.stl_forms = self.stl_forms_eval
        self.stl_form = self.stl_forms[0]
        self.spec_name = f"Team{self.team_spec_obj.spec_type.replace('team_', '').title()}"

    def _reallocate_team_tasks(self, agent_positions: np.ndarray):
        """Re-allocate team tasks with real agent positions before planning."""
        goal_size = np.array([1, 1]) * ENV_CONFIG['goal_size']
        allocation = allocate_tasks(self.team_spec_obj, agent_positions, strategy=self.allocation_strategy)
        self.team_allocation = allocation
        # Eval forms: no expansion (fair scoring)
        self.stl_forms_eval, _ = build_per_agent_stl_forms(
            self.team_spec_obj, allocation, goal_size, self.stl_shrink_factor, self.num_agents,
            agent_positions=agent_positions, add_avoidance=self.team_avoid,
            avoid_expansion=1.0)
        # Guidance avoid regions: with expansion (conservative planner margin)
        _, self.agent_avoid_regions = build_per_agent_stl_forms(
            self.team_spec_obj, allocation, goal_size, self.stl_shrink_factor, self.num_agents,
            agent_positions=agent_positions, add_avoidance=self.team_avoid,
            avoid_expansion=self.avoid_expansion)
        # Evaluation always uses allocated forms
        self.stl_forms = self.stl_forms_eval
        self.stl_form = self.stl_forms[0]

        # For OR specs with pre-allocation: extract chosen branch from full CaTL+
        # formula so the diffusion gradient isn't pulled toward the unchosen branch.
        # Eval (ma_stl_form) stays at the full OR formula @ avoid_expansion=1.0;
        # only guidance (ma_stl_form_guidance) is reduced to the chosen branch.
        if (self.team_spec_obj.branches is not None
                and not self.team_disjunctive):
            assigned_task_ids = set(allocation.task_to_agents.keys())
            chosen_branch = None
            for branch_ids in self.team_spec_obj.branches:
                if assigned_task_ids & set(branch_ids):
                    chosen_branch = branch_ids
                    break
            if chosen_branch is not None:
                all_catl_tasks = self.ma_stl_form_guidance.get_all_predicates()
                chosen_catls = [CaTLPlus(t) for t in all_catl_tasks
                                if any(f"task_{tid}" in t.name for tid in chosen_branch)]
                if chosen_catls:
                    import functools as ft
                    self.ma_stl_form_guidance = ft.reduce(lambda x, y: x & y, chosen_catls)
                print(f"OR spec: CaTL+ guidance reduced to branch {chosen_branch} (eval uses full OR)")
                self.logger.info(f"OR spec: CaTL+ guidance reduced to branch {chosen_branch}")

