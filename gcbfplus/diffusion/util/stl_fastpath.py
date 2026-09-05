"""O(N) shared-structure fast path for per-agent STL evaluation.

``vmap`` over ``lax.switch`` evaluates ALL N per-agent formula branches for every
agent — O(N^2) formula evals per guidance step. When the N per-agent forms share
one AST, evaluating the single structural form with a per-agent, dynamically
indexed ``cent_override`` is exact and O(N). Kept free of planner/wrapper imports
so it stays independently importable and testable.
"""
import jax.numpy as jnp
import numpy as np


def shared_structural_override(stl_forms, goal_centers=None, goal_sizes=None):
    """Build O(N) per-agent STL eval fns from ONE shared structural form, or None.

    Two cases:

    - traced (``goal_centers`` given): the override replaces every reach-leaf center
      (and, with ``goal_sizes``, size), so repr-identical ASTs suffice — baked reach
      centers are ignored.
    - baked (fixed grid, ``goal_centers is None``): the per-agent centers are
      EXTRACTED from the forms' reach leaves and fed through the same override
      mechanism (override == baked equivalence is pinned by tests/test_cent_override).
      Exactness then additionally requires that the forms differ ONLY in reach-leaf
      centers: per-name reach sizes/shrink and every non-overridable leaf (RectAvoid
      obstacles, the name<0 boundary) must be identical across agents, and reach
      names must form a contiguous 0..M-1 index set.

    Returns ``(eval_fn, eval_train_fn)`` with signature ``(x, agent_id)``, or None —
    callers keep the ``lax.switch`` path (mixed specs, team-avoid forms, per-agent
    baked sizes, non-contiguous names).
    """
    if len({repr(f) for f in stl_forms}) != 1:
        return None
    base_form = stl_forms[0]

    if goal_centers is not None:
        if goal_sizes is None:
            _ovr_of = lambda aid: goal_centers[aid]
        else:
            _ovr_of = lambda aid: (goal_centers[aid], goal_sizes[aid])
        return (lambda x, aid: base_form.eval(x, cent_override=_ovr_of(aid)),
                lambda x, aid: base_form.eval_train(x, cent_override=_ovr_of(aid)))

    from ds.stl_jax import RectReachPredicate  # lazy: DIFF_STL_BACKEND is set by the wrapper import
    per_agent_cents = []
    ref_fixed = None
    for f in stl_forms:
        cents = {}
        fixed = []  # leaf params that the override CANNOT replace — must match across agents
        for p in f.get_all_predicates():
            if isinstance(p, RectReachPredicate) and p.name >= 0:
                cent = np.asarray(p.cent, dtype=np.float64)
                if p.name in cents and not np.array_equal(cents[p.name], cent):
                    return None  # one name, conflicting centers within a single form
                cents[p.name] = cent
                fixed.append(('reach', p.name, tuple(np.ravel(p.size)), float(p.shrink_factor)))
            else:
                fixed.append((type(p).__name__, p.name, tuple(np.ravel(p.cent)),
                              tuple(np.ravel(p.size)), float(getattr(p, 'shrink_factor', 1.0))))
        if sorted(cents) != list(range(len(cents))):
            return None  # reach names must index a dense [M, 2] override array
        fixed = sorted(fixed, key=str)
        if ref_fixed is None:
            ref_fixed = fixed
        elif fixed != ref_fixed:
            return None
        per_agent_cents.append(np.stack([cents[k] for k in range(len(cents))]))

    baked = jnp.asarray(np.stack(per_agent_cents))  # [N, M, 2], constant-folded under jit
    return (lambda x, aid: base_form.eval(x, cent_override=baked[aid]),
            lambda x, aid: base_form.eval_train(x, cent_override=baked[aid]))


def extract_baked_goal_centers(stl_forms):
    """Per-agent baked reach-leaf centers ``[N, M, 2]`` for VISUALIZATION, or None.

    Lenient, viz-only variant of the baked branch above: collects each form's
    named reach centers — i.e. only the goals the spec actually uses, including
    shared first1/last1 goals — without the cross-agent exactness gate (sizes/
    obstacles/boundary don't matter for drawing goal rectangles). Returns None
    when a form's reach names aren't a dense 0..M-1 set, forms disagree on M,
    or one name carries conflicting centers within a form.
    """
    from ds.stl_jax import RectReachPredicate  # lazy: DIFF_STL_BACKEND is set by the wrapper import
    per_agent = []
    for f in stl_forms:
        cents = {}
        for p in f.get_all_predicates():
            if isinstance(p, RectReachPredicate) and p.name >= 0:
                cent = np.asarray(p.cent, dtype=np.float64)
                if p.name in cents and not np.array_equal(cents[p.name], cent):
                    return None
                cents[p.name] = cent
        if not cents or sorted(cents) != list(range(len(cents))):
            return None
        per_agent.append(np.stack([cents[k] for k in range(len(cents))]))
    if len({a.shape for a in per_agent}) != 1:
        return None
    return np.stack(per_agent)
