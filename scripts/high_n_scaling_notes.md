# Why success collapses at high N — figure `high_n_scaling_mixed.pdf`

Companion to `scripts/plot_high_n_scaling.py` / `scripts/high_n_plot_config.yaml`.
Mixed spec (`mseq3-m2branch2-mcover3-m2loop3-m2signal3`).

> **Panel sourcing (read first).** The panels do NOT share runs, by design.
> **Success**: native area-6 / no goal scaling / epi=10 at N≤32; goal-scale 2.0 / area-10 /
> epi=3 at N≥64 (decongestion is only needed there). Every *column* is internally controlled —
> both arms share the setting.
> **TtR + Plan Time**: uniform gs2.0 at every N, because `plan_time_mean` is not comparable
> across goal scales (§4). Numbers below are the **success-panel** runs unless stated.

**Short answer: yes — it is the GCBF+ tracking controller's safety breaking down, not the
planner.** Collisions cost DIFF-MA *nothing* at N≤16 and only 5.6 pts at N=32, then 17.7 at
N=64 and 43.2 at N=128. Across the full range, **66% of the lost success is collisions**.

---

## 1. The decomposition is exact, not an estimate

`test.py` (`success_matrix`) defines success per agent as a plain AND:

```python
success_matrix = (1 - is_unsafe) * is_finish
```

`is_unsafe` = collided at any point; `is_finish` = STL robustness > −tol (`gcbfplus/env/wrapper/wrapper.py` (`STL_EVAL_TOLERANCE`)).
So the success shortfall splits with no residual:

```
1 − success  =  (finish − success)  +  (1 − finish)
                 ^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^
                 satisfied the spec     never satisfied
                 but COLLIDED           the spec
                 → controller           → planner
```

Both terms are read straight from `run.summary` (`success_mean`, `safe_mean`,
`eval/finish_rate`). All values in %.

### DIFF-MA (edm-ma, achievable guidance OFF)

| N | setting | success | safe | finish | **collided-but-satisfied** | never-satisfied | total miss |
|---|---|---|---|---|---|---|---|
| 8 | native a6 | 100.0 | 100.0 | 100.0 | **0.0** | 0.0 | 0.0 |
| 16 | native a6 | 98.8 | 100.0 | 98.8 | **0.0** | 1.2 | 1.2 |
| 32 | native a6 | 91.6 | 94.4 | 97.2 | **5.6** | 2.8 | 8.4 |
| 64 | gs2.0 a10 | 70.8 | 80.2 | 88.5 | **17.7** | 11.5 | 29.2 |
| 128 | gs2.0 a10 | 34.4 | 41.9 | 77.6 | **43.2** | 22.4 | 65.6 |

**Read it:** at N≤16 the controller is free — a perfect 100.0 safe, zero success lost to
collisions. The safety term then grows 0.0 → 5.6 → 17.7 → **43.2**, while the planner's term
grows only 0.0 → 22.4.

Across N=8 → N=128, success falls 100.0 → 34.4 (**−65.6 pts**):
- **+43.2 pts (66%)** is the safety term — the controller colliding.
- +22.4 pts (34%) is the spec term — the planner.

At N=128 collisions account for 43.2 of the 65.6 total miss (66% of all failure).

### STLPY-SA

| N | setting | success | safe | finish | collided-but-satisfied | never-satisfied | total miss |
|---|---|---|---|---|---|---|---|
| 8 | native a6 | 95.0 | 98.8 | 96.3 | 1.3 | 3.7 | 5.0 |
| 16 | native a6 | 90.0 | 93.1 | 96.9 | 6.9 | 3.1 | 10.0 |
| 32 | native a6 | 81.9 | 89.4 | 92.5 | 10.6 | 7.5 | 18.1 |
| 64 | gs2.0 a10 | 55.2 | 83.3 | 69.8 | 14.6 | 30.2 | 44.8 |
| 128 | gs2.0 a10 | 33.3 | 64.1 | 66.1 | 32.8 | 33.9 | 66.7 |

The MILP is hit by the same controller-driven safety loss, and additionally carries a spec
deficit that jumps at the 32→64 boundary (7.5 → 30.2). Part of that jump is the goal-scale
change, not N — see §5.

**Do not read STLPY-SA's higher N=128 safety (64.1 vs 41.9) as a win.** Its finish rate is only
66.1% — it is partly "safe" because fewer of its agents ever complete their tours and thus spend
less time in transit. Never quote that safety number without the finish rate beside it.

---

## 2. Why the controller, and not the plan

The tracking controller is fixed across every cell: GCBF+ DubinsCar `pretrained/DubinsCar/gcbf+/`
step 1000, **trained at N=8, area-4**. It is the one component never re-fit as N grows 16×.

1. **It is planner-independent.** Safety degrades for *both* planners at N=128 (DIFF-MA 41.9,
   STLPY-SA 64.1) despite unrelated plan generators. A planner-side cause would not hit both.
2. **Collisions dominate the per-agent failure dump.** At N=64/gs2.5 (per-episode diagnostics
   from internal tooling, not part of this release): DIFF-MA collide **12.0%**, timeout 3.6%, neg_rho 8.3%.
3. **It is not the step budget.** max_step 9000 → 16000 at N=128/gs3.0 leaves finish unchanged
   (68.8 → 68.5) and success identical (31.5). Timeout at the published gs2.0 column is 2.6%,
   bounding any budget rescue at ~2.6 pts.

Planning itself scales fine: no OOM through N=128 (~6.7 GB peak), and edm-ma plan time is
3.6–5.9× below the MILP at every N. **The bottleneck at high N is downstream of planning.**

---

## 3. ⚠ The "start-density" explanation does not survive its own sweep

the internal replication log (§2) attributes the residual N=128 failure to *"start-density collisions
under the N=8-trained tracking controller"*. The density framing is **not supported** by the
goal-scale sweep in that same section, and should not be repeated as stated.

Starts are sampled uniformly over `area_size` (`dubins_car.py:83`), so `--area-size` genuinely
spreads starts — density is real and controllable. Training density is N=8 / area-4² =
**0.5 agents/unit²**. Matched-density pairs from §2's own table:

| cell | density | safe % |
|---|---|---|
| N=64, gs2.0, area-10 | 0.640 | **80.2** |
| N=128, gs3.0, area-14 | 0.653 | **43.0** |

**Same agent density; nearly half the safety.** N itself matters beyond density. The bounding
control: the N=32-vs-64 matched-density pair (0.320 → 99.0 vs 0.327 → 91.7) shares the same
gs3.0/area-14/max_step-9000 confound and loses only **7.3 pts** — nowhere near the 37-pt gap.

**Honest caveat:** not a clean control — the gs3.0 cell runs max_step 9000 vs 4800 (1.9× the
exposure) and goals 1.5× farther. A clean test (vary N at constant density, fixed budget and
goal distance) does not exist. So assert only the *negative* claim, which is solid:

> Decongesting goals cannot fix N=128 — safety saturates at ~43% across gs2.0/2.5/3.0 even as
> density falls from 1.28 to 0.65 agents/unit². The residual failure is **not** goal crowding,
> and **not** density alone.

## 4. ⚠ Why the panels don't share runs

`plan_time_mean` is **not comparable across goal scales**. Same N=8, same diffusion model
(`qkmvppvt`), same `num_candidates=8` / `spec_len=15`:

| N=8 DIFF-MA | plan_time_mean |
|---|---|
| native area-6 | **29.5 / 26.6 / 30.4 s** |
| area-10, gs2.0 | **3.49 s** |

Not `change_goal_immediately` (cgi on/off both give ~29 s at area-6). Likely replan frequency:
at area-6 goals are close, agents reach them constantly and replan far more often, so
`plan_time_mean` tracks goal spread. Valid *within* one goal scale, meaningless *across* one.
Hence the timing panels use uniform gs2.0 while the success panel uses the split. A third
number exists for the same cell — the internal experiment log reports N=8 edm-ma at **10.5 s**
(steady-state s/plan, compile excluded). **Three different quantities; never cite two of them
as the same number.**

## 5. ⚠ Open question: STLPY under goal scaling

The gs2.0 cells at N≤32 were dropped from the success panel because the STLPY-SA arm behaves
anomalously under goal scaling. At N=8, as goal scale rises 1.0 → 3.0, STLPY success collapses
**93.8 → 58.3 → 45.8 → 33.3** while its safety stays 100.0 and — the tell — **its TtR stays
flat**: 1539 / 1369 / 1441 / 1416 / 1363. A planner routing to genuinely farther goals cannot
have constant travel time. DIFF-MA's TtR over the same scales grows as expected: 1750 / 2359 /
2424 / 2537 / 2698.

Net effect: gs2.0 inflated the low-N gap **2–7×** (N=8 +5.0 → +33.4; N=16 +8.8 → +27.1;
N=32 +9.7 → +22.9). The native cells are the defensible ones.

**Not a max_step budget problem** (the first hypothesis, falsified): MORE budget makes STLPY
*worse* (gs2.5 = 1.5× steps → 45.8; gs3.0 = 1.875× → 33.3), safety is 100% at N=8, satisfying
agents finish at TtR 1441 = 30% of the 4800 cap, and the N=64/gs2.0 dump shows timeout 2.6% with
`wp_final` 14.6/15. `u_bound` (±20 on a unit-gain integrator) is far larger than the arena and
not binding.

Root cause not yet established — candidates: the MILP not receiving the scaled `GOAL_SET`, or a
plan-horizon limit at spec_len=15. **The N=64/128 STLPY cells still rely on gs2.0**, so this
should be settled before those bars are trusted for a camera-ready claim. Suggested next step:
a per-episode failure-channel dump at N=8/gs2.0/stlpy (internal tooling, not part of this release) for the direct collide/timeout/neg_rho split.

## 6. Safe wording for the paper

- ✅ "At N≤16 the tracking controller is not a limiting factor: collisions cost 0.0 success
  points and safety is 100%."
- ✅ "Beyond N=32 the fixed N=8-trained GCBF+ controller becomes the bottleneck: 66% of the
  success lost between N=8 and N=128 is agents that satisfied their spec and then collided."
- ✅ "Planning scales (3.6–5.9× faster than the MILP at every N, no OOM through N=128); the
  ceiling is the controller."
- ✅ "N=128 is a statistical tie (34.4 vs 33.3, epi=3, single seed) — the scaling limit, not a win."
- ❌ Do not write "start-density collisions" as the mechanism — §3.
- ❌ Do not quote STLPY-SA's N=128 safety (64.1%) without its finish rate (66.1%).
- ❌ Do not present the panels as one experiment — success and timing come from different runs
  (§4), and the success panel changes setting between N=32 and N=64 (goal scale AND epi).

## 7. Reproduce

```bash
conda activate gcbfplus
python scripts/plot_high_n_scaling.py --source wandb --out-dir out_final   # writes the snapshot
python scripts/plot_high_n_scaling.py --source csv \
    --data-csv plot_snapshots/high_n_mixed_DubinsCar_data.csv --out-dir out_final
```

The script prints every bar's source run id (success run and timing run separately) and asserts
the population shape (10 cells, 2 planners per N), so a silent data change fails loudly.
