# Reproducing the final paper figures

Everything needed to re-run the evaluations behind the three paper figures and redraw them,
without access to our W&B projects.

| file | role |
|---|---|
| `reproduce_final_plots.sh` | one `python test.py ...` line per plotted bar (exact original args), resumable runner. **Generated — do not hand-edit.** |
| `final_plots_manifest.csv` | one row per bar (or per panel-role of a bar): source W&B run, commit, date, host, env knobs, args, and the reference numbers the paper shows (`ref_*`) |
| `plot_reproduced.py` | reads the reproduced local logs and draws the figures with the **same renderers** the paper used |
| `_build_manifest.py` | maintainer tool: regenerates the two files above from W&B (`_wandb_metadata_cache.json` lets it re-run offline) |

## Quick start

```bash
conda activate gcbfplus            # GPU environment per the top-level README
bash scripts/release/reproduce_final_plots.sh                   # hours; resumable; FIGS=/ONLY=/DRY_RUN=1 knobs in the header
python scripts/release/plot_reproduced.py --out-dir out_repro   # -> out_repro/*.pdf + *_table.tex
```

To see the paper's figures from the recorded numbers without running anything:

```bash
python scripts/release/plot_reproduced.py --out-dir out_ref --fallback-manifest
```

That command is also the self-check: its three PDFs are pixel-identical (ImageMagick `compare -metric AE` = 0
at 150 dpi) to the figures `scripts/replot_all_final.sh` writes to `out_final/`, and so are the two LaTeX tables.

## What gets reproduced

| figure | paper file | bars | distinct runs | run commands | drawn from paper numbers |
|---|---|---|---|---|---|
| `random_loc` | `random_loc_DubinsCar_Success-Rate-↑_Planning-Time-(s)-↓_TtR-↓.pdf` (+ `_table.tex`, `_data.csv`, `_run_ids.csv`) | 72 | 72 | 54 | 18 (Gradient arm, see limits) |
| `team_spec` | `team_spec_comparison_merged_emergent.pdf` | 14 | 14 | 14 (6 best-effort) | 0 |
| `high_n` | `high_n_scaling_mixed.pdf` (+ `_table.tex`) | 10 | 16 | 16 | 0 |

The runs behind each bar were not guessed: `_build_manifest.py` executes each figure's own offline
selection pipeline on the committed snapshot and records which run id landed in each cell (for
`random_loc` the id is carried through the Plotter's `aggfunc="first"` pivot as an extra column).
Launch args, git commit and host come from each run's stored `wandb-metadata.json`; env knobs
(`GCBF_GOAL_SCALE`, `GCBF_TEAM_SELECT_OUTER`) from the `env_*` config keys `test.py` records.

### Code pin

All commands are checked against the argparse of **`dbg-ach-loss` @ `d1f2aa2` (2026-07-12)**. The
diffusion / STLPY runs were launched from nine commits that are all ancestors of it. That branch
also bundles the diffusion planner checkpoint (`diff_checkpoints/qkmvppvt`, loads without W&B).

### Per-figure provenance

**random_loc** — `plot_paper.py --figure random_loc_dubins`. Area 6, random predicate locations
(`--random-goals --traced-goals`), epi 10 (Gradient 5), `--seed 1234`, spec_len 15 (Gradient 30).
The figure's post-filter pool holds 94 runs; 22 cells had more than one candidate and the Plotter
keeps the best success (ties: lowest plan time, then run id). Selection rules that decide which
variant appears (all in `plot_config.yaml`): Signal column uses the crowded R=1.5 runs; Loop DIFF-MA
uses `achievable_guidance=False`; Cover and Mixed DIFF-SA use single-draw `edm` (other specs may use
the best-of-8 `edm-bo8`, i.e. `--use_batched_sampling --num_candidates 8`).
Runs: 54 from commits `8499d431` / `28eddb81` (2026-07-05 .. 07-11, workstation) plus the 18
Gradient runs from `0274dda0` (2026-07-06).

**team_spec** — `scripts/plot_team_spec_results.py --merge-edm --show-global`, scope `emergent`.
Area 6, epi 10, `--max-step 7200`, `--team-avoid`; DIFF-MA is `edm-ma --team-disjunctive` (no
pre-allocation). Choice N=32 DIFF-MA is the `GCBF_GOAL_SCALE=1.25` decongestion cell; the three
Choice DIFF-MA runs ran with `GCBF_TEAM_SELECT_OUTER=1`, the two STLPY-Global runs with `=0`
(both are passed as env prefixes in the script). Runs: 8 W&B runs from six commits dated
2026-07-03 .. 07-12, plus 6 STLPY-SA rows from an old local `test_log.csv` (next section).

**high_n** — `scripts/plot_high_n_scaling.py`. Mixed spec
`mseq3-m2branch2-mcover3-m2loop3-m2signal3`. The **success** panel uses the native area-6 / epi-10
runs at N ≤ 32 (docker sweep, commit `5227c2d0`, 2026-06-23) and the `GCBF_GOAL_SCALE=2.0` /
area-10 / epi-3 runs at N ≥ 64; the **TtR and plan-time** panels use the gs2.0 run at every N
(commit `2d6867f6`, 2026-06-25). So N ≤ 32 needs two runs per cell — the manifest marks them
`role=success` / `role=timing`, N ≥ 64 rows are `role=both`. `plan_time_mean` is not comparable
across goal scales, which is why the panels are sourced separately (`scripts/high_n_scaling_notes.md`).

## Known limits

* **Gradient (ce_nl) arm is not re-run.** Its 18 runs came from worktree commit `0274dda0`, which
  was never pushed (its nearest published ancestor lacks `--random-goals`), and the pinned
  code's `--planner` accepts only
  `stlpy | diffusion | stlpy_global`. The original commands are kept as comments in the script;
  `plot_reproduced.py` always draws those bars from the manifest's reference numbers and prints
  them as `ref`.
* **team_spec STLPY-SA bars are best-effort.** The paper takes them from an older local
  `test_log.csv` (complete TtR column; epi 5 for Redundant, 10 for Choice; budgets
  900/1200/1500 and 2400/2700/3600 steps; greedy allocation) written by pre-"spec-v2" code with
  no W&B run and no stored args. The commands are reconstructed from the logged settings plus the
  flag set every W&B STLPY team run of that code family used. On the pinned (spec-v2) code the
  MILP scores differ, most on Choice. The spec-v2 W&B STLPY-SA runs, for comparison:

  | cell | paper (old log) | spec-v2 W&B best | run |
  |---|---|---|---|
  | Choice N=8 | 92.5 | 47.5 | `7iuemluv` (oracle_hungarian) |
  | Choice N=16 | 68.8 | 35.3 | `rhmjauay` (oracle_hungarian) |
  | Choice N=32 | 41.7 | 36.1 | `71w7ebwy` (greedy) |
  | Redundant N=8 | 100.0 | 93.8 | `2clzb7yd` (oracle_hungarian) |
  | Redundant N=16 | 98.8 | 84.5 | `aajpk3e5` (oracle_hungarian) |
  | Redundant N=32 | 68.4 | 64.5 | `56rz6yf0` (oracle_hungarian) |

* **Single seed, few episodes.** Every cell is one seed (`--seed 1234`); high_n N ≥ 64 cells are
  epi=3. Expect a few points of variance on success and more on TtR; the N=128 success gap
  (34.4 vs 33.3) is inside noise.
* **Sidecar CSVs differ in shape.** The reproduced `random_loc_DubinsCar_data.csv` carries only
  the columns the Plotter needs; the paper's has the full W&B config dump (158 columns). Values
  in the shared columns are the same quantities.
* **Runtimes.** STLPY-SA rows take minutes to hours each (N=128: ~195 s per plan, 3 episodes);
  STLPY-Global ~800 s per episode × 10. Diffusion rows take seconds to minutes per episode.
  Gurobi is required for every STLPY row.

## How the log → figure path was validated

1. `plot_reproduced.py --fallback-manifest` on an empty log dir reproduces all three paper PDFs
   with AE = 0 and both `_table.tex` files byte-identical.
2. 84 synthetic `test_log_repro_<id>.csv` files were written in the exact header the pinned
   `test.py` emits, populated from the manifest reference numbers; `plot_reproduced.py` on those
   (no fallback) again gives AE = 0 on all three figures. This exercises the real parsing path.
3. `bash scripts/replot_all_final.sh` passes 4/4 (random predicates, team specs, high-N scaling, achievable-loss table) with every figure and table unchanged.

## Regenerating the bundle (maintainers)

```bash
python scripts/release/_build_manifest.py            # re-selects runs from the snapshots, refetches W&B metadata
python scripts/release/_build_manifest.py --no-fetch # same, from _wandb_metadata_cache.json
```

Run it whenever a snapshot, a plot config, or a selection rule changes, so the script and manifest
never drift from the figures in `out_final/`.
