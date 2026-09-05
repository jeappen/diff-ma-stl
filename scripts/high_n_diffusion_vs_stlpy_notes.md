# DIFF-MA (diffusion) vs STLPY-SA (MILP) — where each one actually loses

Companion to `scripts/high_n_scaling_notes.md` (which asks *why success falls at high N*; this
one asks *why diffusion beats the MILP, and why the N=128 tie happens*).

> **Panel sourcing (read first).** Success: native area-6 / epi=10 at N≤32, gs2.0 / area-10 /
> epi=3 at N≥64 — each column internally controlled. TtR + Plan Time: uniform gs2.0 at every N.
> The panels do not share runs. See `high_n_scaling_notes.md` §4.
>
> ⚠ **Supersedes an earlier draft of this file.** Its headline ("the advantage is *entirely*
> plan quality") was computed from the gs2.0 cells at N≤32, which we have since established
> break the STLPY-SA arm (`high_n_scaling_notes.md` §5). At the native setting the low-N story
> is different and much more modest. Do not reuse the old numbers.

**Short answer:** the diffusion advantage is **modest and mixed at low N** (+5.0 to +9.7, roughly
half safety and half plan quality), and becomes **plan-quality-dominated at high N** (+18.7 at
N=64) — where it is simultaneously cancelled by a safety deficit. The N=128 "tie" is not the
planners converging: diffusion still plans better by +11.5, and the shared controller takes
−10.4 straight back.

---

## 1. The gap decomposes exactly

Success is a per-agent AND (`test.py` (`success_matrix`)), so each planner's failure splits with no residual
into a **safety** term (satisfied the spec, then collided → controller) and a **spec** term
(never satisfied → planner). The *gap between planners* splits the same way:

```
success_D − success_S  =  (safety_S − safety_D)  +  (spec_S − spec_D)
                           ^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^
                           safety advantage         plan-quality advantage
                           (+ = diffusion safer)    (+ = diffusion plans better)
```

| N | setting | success gap | safety advantage | plan-quality advantage |
|---|---|---|---|---|
| 8 | native a6 | +5.0 | +1.3 | +3.7 |
| 16 | native a6 | +8.8 | **+6.9** | +1.9 |
| 32 | native a6 | +9.7 | +5.0 | +4.7 |
| 64 | gs2.0 a10 | +15.6 | **−3.1** | **+18.7** |
| 128 | gs2.0 a10 | +1.1 | **−10.4** | **+11.5** |

(Columns sum to the gap; ±0.1 from rounding summary values to 0.1.)

**Two regimes, and the boundary is also a setting change — be careful attributing it to N alone:**

1. **Low N (native):** the gap is small (+5.0 to +9.7) and split roughly evenly between safety
   and plan quality. Diffusion is both slightly safer and slightly better at satisfying the spec.
   Neither term dominates.
2. **High N (gs2.0):** plan quality dominates (+18.7, +11.5) and the safety advantage **goes
   negative** — diffusion becomes the *less* safe arm.

## 2. The N=128 tie, explained

| term | value |
|---|---|
| plan-quality advantage | **+11.5** |
| safety disadvantage | **−10.4** |
| **net** | **+1.1** (a tie at epi=3) |

A near-exact cancellation, not a convergence of planner ability. Diffusion still produces
spec-satisfying plans for 11.5 pts more agents; the controller then loses almost exactly that
much back in collisions. **Do not write "the planners are equivalent at N=128" — write that
diffusion's planning lead is cancelled by the shared controller.**

## 3. ⚠ STLPY-SA is not "safer" at N=128 — it just moves less

The raw safety numbers invite the wrong conclusion:

| N=128 | safe | finish | success |
|---|---|---|---|
| DIFF-MA | 41.9 | 77.6 | 34.4 |
| STLPY-SA | **64.1** | 66.1 | 33.3 |

STLPY-SA looks 22 pts safer. Normalise by exposure — the fraction of *spec-satisfying* agents
that collided, `P(collided | satisfied) = (finish − success) / finish`:

| N | DIFF-MA | STLPY-SA |
|---|---|---|
| 8 | **0.0%** | 1.3% |
| 16 | **0.0%** | 7.1% |
| 32 | **5.8%** | 11.5% |
| 64 | 20.0% | 20.9% |
| 128 | 55.7% | 49.6% |

At N≤32 diffusion is clearly the safer arm per unit of completed work. At N=64 the two are
**essentially identical (20.0% vs 20.9%)** — the controller loads both planners the same. At
N=128 diffusion is somewhat worse (55.7 vs 49.6), but nothing like the 22-pt raw gap: most of
that gap is an artifact of STLPY-SA completing 11.5 pts fewer tours and so spending less time in
transit where collisions happen.

**Never quote STLPY-SA's N=128 safety without its finish rate beside it.**

## 4. Plan time: a large constant-factor win, not a better exponent

Uniform gs2.0 at every N (the timing panels' source), so these are comparable end to end:

| N | DIFF-MA (s) | STLPY-SA (s) | speedup |
|---|---|---|---|
| 8 | 3.49 | 12.39 | 3.6× |
| 16 | 6.18 | 25.26 | 4.1× |
| 32 | 7.90 | 46.19 | **5.9×** |
| 64 | 20.94 | 90.96 | 4.3× |
| 128 | 49.89 | 195.15 | 3.9× |

Diffusion is **3.6–5.9× faster at every N**, and the margin is stable. That is the honest,
defensible claim from this figure.

**⚠ What this figure does NOT show: sub-linear scaling.** On `plan_time_mean`, both planners grow
~linearly over N=8→128 (16× agents):

| planner | 8 → 128 | scaling exponent (1.0 = linear) |
|---|---|---|
| DIFF-MA | 14.3× | **0.96** |
| STLPY-SA | 15.8× | **0.99** |

the internal experiment log claims *"edm-ma scales sub-linearly"* — that rests on a **different
quantity** (steady-state s/plan, compile excluded: 10.5 → 49.6 s = 4.7×, exponent 0.56). Both
claims can be true of their own metric; they are not interchangeable. If the text argues
sub-linear planning cost, cite the steady-state table, **not this figure**.

## 5. ⚠ Do not compare TtR across planners

| N | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| DIFF-MA TtR | 2424 | 2513 | 2699 | 3150 | 3662 |
| STLPY-SA TtR | 1441 | 1596 | 1808 | 1835 | 2265 |

STLPY-SA's TtR looks ~1.6× better. **It is not evidence that the MILP reaches goals faster.**
TtR is survivorship-biased (`AsyncPlannerMixin.process_finished_rollouts_info` sets TtR to NaN for every agent whose path score
fails; aggregation is `nanmean`, the internal replication log (§7.1)). Each planner's TtR averages over *only
the agents it satisfied* — and those are different, non-comparable populations, with STLPY-SA's
being the easier subset. Show the panel as a per-planner scaling trend; the cross-planner
comparison needs a matched-agent analysis that does not exist yet.

## 6. ⚠ The DIFF-MA arm excludes achievable guidance — deliberately, and for free

DIFF-MA is plain `edm-ma` with `achievable_guidance=False` at every N (`diffma.exclude_achievable_guidance`
in the config). Allowing the ach runs into the best-of pool **changes nothing** while costing
4–26× the plan time:

| N | ach=False best | ach=True best |
|---|---|---|
| 8 | **100.0** (plan 29.0 s) | 100.0 (plan **122.7 s**) |
| 16 | **98.8** (plan 25.2 s) | 98.1 (plan **355.2 s**) |
| 32 | **91.6** (plan 19.6 s) | 93.1 (plan **519.7 s**, epi=5) |

At N=32 achievable guidance buys **+1.5 pts for 26× the compute** — and since the timing panel is
sourced from the (cheap, ach=False) gs2.0 run, including it would advertise 93.1% at a plan cost
that run never paid. Also `edm-ach` stalls at N≥64 and does not exist there (internal experiment log),
so including it would make the arm inconsistent across the x-axis. Excluded on all three counts.

## 7. Safe wording for the paper

- ✅ "At the native setting the diffusion advantage is +5.0 to +9.7 pts, split roughly evenly
  between fewer collisions and better spec satisfaction."
- ✅ "At N=64 diffusion produces spec-satisfying plans for 18.7 pts more agents than the MILP."
- ✅ "At N=128 the two tie on success (34.4 vs 33.3) because diffusion's +11.5 pt planning
  advantage is cancelled by a −10.4 pt safety deficit under the shared controller — not because
  the planners become equivalent."
- ✅ "Diffusion plans 3.6–5.9× faster than the MILP at every N."
- ❌ Do not say STLPY-SA is safer at N=128 — normalised for exposure the controller loads both
  the same at N=64 (20.0% vs 20.9%).
- ❌ Do not cite this figure for sub-linear planning cost — on `plan_time_mean` both are ~linear.
- ❌ Do not compare TtR across planners — survivorship bias, different populations.
- ❌ Do not attribute the whole low→high-N change to N: the 32→64 boundary also changes goal
  scale (1.0 → 2.0) and epi (10 → 3).

## 8. Reproduce

```bash
conda activate gcbfplus
python scripts/plot_high_n_scaling.py --source csv \
    --data-csv plot_snapshots/high_n_mixed_DubinsCar_data.csv --out-dir out_final
```

Numbers derive from `success_mean`, `safe_mean`, `eval/finish_rate`, `plan_time_mean`,
`eval/TtR` on the cells the script prints (success run and timing run listed separately). The
`collide`/`timeout`/`neg_rho` channels cited in the companion file come from per-episode diagnostics
of separate N=64 runs (internal tooling, not part of this release), not from these runs.
