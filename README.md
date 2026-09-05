# GCBF+ with STL Task Planning

Code release for [Generalizable Multi-Agent Planning from Signal Temporal Logic Specifications via Diffusion](https://www.jeappen.com/diff-ma-stl/) (project website with videos and results).

JAX implementation of GCBF+ ([S Zhang*](https://syzhang092218-source.github.io), [Oswin So*](https://oswinso.xyz/), [K Garg](https://kunalgarg.mit.edu/), [C Fan](https://chuchu.mit.edu): "[GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control](https://mit-realm.github.io/gcbfplus-website/)"), extended with **Signal Temporal Logic (STL) task planning**: a multi-agent diffusion planner generates waypoint plans that satisfy STL specifications, and the pretrained GCBF+ controller tracks them safely. A Gurobi MILP planner (STLPY) is included as the optimization baseline, per agent and as a joint centralized solve for team specifications.

**A major caveat of this approach is that execution is asynchronous.** Each agent advances to its next waypoint only when it reaches the current one, so the temporal structure of a specification is enforced over plan steps, not over a shared clock, and the tracking controller provides no timing guarantees between agents. Scalable *synchronous* multi-agent STL execution, where all agents meet deadlines on a common clock, requires further advancements and is not addressed by this code.

## Installation

```bash
conda create -n gcbfplus python=3.10
conda activate gcbfplus
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]>=0.5,<0.6"                                       # GPU build of JAX
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "git+https://github.com/jeappen/diff-spec.git@feature/jax"   # STL robustness library (JAX fork)
```

Smoke test (about a minute on a GPU, no Gurobi needed):

```bash
python test.py --path pretrained/DubinsCar/gcbf+/ --epi 2 --area-size 6 -n 8 --obs 0 --nojit-rollout --planner diffusion --spec mseq3 --spec-len 15 --goal-sample-interval 20 --async-planner --ignore-on-finish --no-video
```

`bash scripts/run_all_pipelines.sh` exercises every pipeline once at small scale and prints a summary.

## Running the planners

The pretrained GCBF+ controller (`pretrained/DubinsCar/gcbf+/`, released by the GCBF+ authors) tracks waypoint plans produced at episode start by a planner. The bundled diffusion planner `diff_checkpoints/qkmvppvt` loads locally.

**Specifications used in the paper** (`--spec`, horizon 15 plan steps via `--spec-len 15`):

| Spec | `--spec` | Meaning |
|---|---|---|
| Sequence | `mseq3` | visit three goals in order, each within its time window |
| Cover | `mcover3` | visit three goals in any order |
| Loop | `m2loop3` | repeat a three-goal loop twice (bounded recurrence) |
| Branch | `m2branch2` | complete one of two alternative two-goal tours |
| Signal | `m2signal3` | loop over three goals until a final goal is reached (until) |
| Mixed | `mseq3-m2branch2-mcover3-m2loop3-m2signal3` | each agent draws one of the five |
| Redundant (team, CaTL+) | `team_redun4x2`, `team_redun4x4`, `team_redun4x8` | four goals, each to be reached by N/4 agents |
| Choice (team, CaTL+) | `team_choiceseq3` | the team picks one of two branches, each a counted main sweep plus a support task |

**DIFF-MA** (the paper's method), with per-episode random predicate locations and best-of-8 joint candidates:

```bash
python test.py --path pretrained/DubinsCar/gcbf+/ --epi 10 -n 8 --area-size 6 --obs 0 --nojit-rollout --planner diffusion --diffusion-method edm-ma --wandb-run-id qkmvppvt --use_batched_sampling --num_candidates 8 --spec mseq3 --spec-len 15 --goal-sample-interval 20 --async-planner --ignore-on-finish --change_goal_immediately --random-goals --traced-goals --no-video --log
```

**STLPY-SA** (MILP per agent, needs Gurobi): replace the planner block with `--planner stlpy`. Both planners see identical episodes and goals for the same seed.

**Team specifications**: add `--team-avoid --max-step 7200` and drop the `--random-goals --traced-goals` pair. DIFF-MA discovers the allocation with `--team-disjunctive`; `--planner stlpy_global` is the joint MILP over the whole team.

## Reproducing the paper figures

```bash
bash scripts/replot_all_final.sh   # -> out_final/*.pdf and *_table.tex, offline
```

The figures and LaTeX tables regenerate from the committed W&B snapshots in `plot_snapshots/`: `random_loc_DubinsCar_data.csv.gz` (random predicate locations), `team_spec_DubinsCar_data.csv` plus `stlpy_sa_old_override.csv` (team specifications), and `high_n_mixed_DubinsCar_data.csv` (scaling). The fixed-predicate-center snapshot `single_hetero_DubinsCar_data.csv.gz` is kept for reference but not plotted. To re-run the evaluation behind every plotted bar (84 `test.py` commands with their original arguments) and redraw the figures from the new logs, see [`scripts/release/README.md`](scripts/release/README.md).

## Docker

```bash
bash scripts/rebuild_docker_image.sh   # build gcbfplus:jax2404 on nvcr.io/nvidia/jax:24.04-py3
bash scripts/test_docker_image.sh      # smoke test: imports + STLPY rollout
```

Run against a checkout by mounting the repository at `/workspace`; `pretrained/` and `diff_checkpoints/` are not baked into the image.

## Training the diffusion planner

`create_dataset.py` rolls the controller out under STLPY plans (same flags as `test.py`, single-agent rollouts: `-n 1`) and writes `datasets/<Env>_*.h5`; `train_diffusion.py` trains on such a file with the bundled model's settings:

```bash
python train_diffusion.py --dataset_name DubinsCar_<name>.h5 --diffusion_trajectory_mode sg --trajectory_length 15 --stl_train_traj_len 15 --num_features 32 --num_blocks 3 --batch_size 32
```

## Tests

```bash
python -m pytest tests/
```

## Notes

**Execution model.** Always pass `--async-planner`: goals advance per agent on reach, which is the execution model of the paper. `--change_goal_immediately` refreshes the tracked goal as soon as an agent advances (the paper's evaluation uses it). The episode budget is `goal_sample_interval x spec_len x` a per-spec factor; `--max-step` overrides it.

**Other planner options.** `--diffusion-method edm` is the per-agent DIFF-SA baseline (add `--use_batched_sampling --num_candidates 8` for per-agent best-of-8); `--achievable_guidance` adds dynamics-feasibility guidance through the controller; `--random-goals-size 0.5 1.5` randomizes predicate sizes; `--random-goals-spacing 2.0` and `--random-goals-region 1.5` set the rigid loop geometry and crowding used for `m2signal3`; `--team-alloc {greedy,random,greedy_last,oracle_hungarian}` pre-allocates team tasks for either planner. `--wandb-run-id` selects the diffusion checkpoint; checkpoints not present under `diff_checkpoints/` are fetched from wandb using the team/project in `gcbfplus/utils/configs/default_config.yaml`.

**Other specifications.** The `m` prefix gives every agent its own rotated goal order. The spec builders in `gcbfplus/env/wrapper/stl_mixin.py` and `gcbfplus/stl/team_spec.py` accept other sizes and a few further fragments (reach-and-stay, avoid-until, cover-until, `team_cover`, `team_<M>of<N>`); those are not part of the paper's evaluation. The signal spec is always built at horizon 15 to match the bundled planner.

**Sampler ablations** (environment variables; defaults reproduce the paper): `GCBF_KEEP_BEST` (best-so-far resample merge, default on), `GCBF_MA_ACCEPT` (joint CaTL+ acceptance gate, default on) and `GCBF_MA_ACCEPT_THRESH`, `GCBF_TASK_REPAIR` / `GCBF_TASK_REPAIR_GRAD` (counting-aware repair for disjunctive team specs, default on), `GCBF_TEAM_SELECT_OUTER` (team-aware outer candidate selection, default off), `GCBF_GOAL_SCALE` (scale the goal grid, used with larger areas at N >= 64). `GCBF_DIFFUSION_VERBOSE=1` prints the sampler configuration at each compile.

**Evaluation outputs.** `--log` appends one row per run to `<path>/test_log.csv` (the schema the plotting scripts read); videos go to `<path>/videos` unless `--no-video`; `--highlight-agent 0,3` marks agents and their goals in videos. Success is safe AND finished per agent; for team specs it is safe x (tasks completed / tasks total).

**Environment and versions.** Python 3.10; jax 0.5.x with the CUDA wheels resolved by `jax[cuda12]` (do not pin `nvidia-*` packages by hand: mixing wheel generations breaks the denoiser's cuDNN convolution). Verified from a fresh environment with jax 0.5.3 on an RTX 3070. jax 0.6.x runs the per-agent specs but its XLA aborts while compiling the `team_choiceseq3` guidance; `export XLA_FLAGS=--xla_disable_hlo_passes=priority-fusion` avoids that crash. jax 0.4.38 works only with the CUDA 12.8 wheel family. `torch` is only a transitive dependency of diff-spec; install the CPU build first so it cannot shadow JAX's CUDA libraries. The STLPY planners need a [Gurobi](https://www.gurobi.com/) license reachable through `GRB_LICENSE_FILE` (a free academic license works).

**Docker details.** The Dockerfile also builds on the NGC 25.01 base (JAX 0.4.38); the 25.08 base (JAX 0.6.x) needs the XLA flag above for the Choice team spec. The VCS dependencies are public; a GitHub token (`GITHUB_PAT`, `.secrets/read_gcbfplus_PAT.txt`, or `gh auth token`) is only needed for private forks and is passed as a BuildKit secret. Drop a Gurobi WLS `gurobi.lic` at `.secrets/gurobi.lic` for the STLPY smoke test.

**Controllers and environments.** Environments: `SingleIntegrator`, `DoubleIntegrator`, `DubinsCar` (2D); `LinearDrone`, `CrazyFlie` (3D); all STL planning results use `DubinsCar`. Controllers (`--algo`): `gcbf+`, `gcbf` (pretrained in `pretrained/`, per-environment hyper-parameters in `settings.yaml`); `centralized_cbf`, `dec_share_cbf` (CBF-QP baselines, no checkpoint). Controller-only evaluation: `python test.py --path pretrained/DubinsCar/gcbf+/ --epi 5 --area-size 4 -n 16 --obs 0`; CBF-QP baseline: `python test.py --env SingleIntegrator -n 16 --algo dec_share_cbf --epi 1 --area-size 4 --obs 0 --alpha 1`. `train.py` remains for retraining a controller (`python train.py -h`).

**Training details.** `--stl_train_traj_len` must equal `--trajectory_length`: every episode is subsampled to that many waypoints before windows of that length are cut. The bundled model used 5000 epochs on a 10k-episode dataset; `scripts/diffusion/train_diff.sh` is a sweep template and `python train_diffusion.py -h` lists the EDM and guidance options.

**Sample pipeline runner.** `scripts/run_all_pipelines.sh` runs the controller, the CBF-QP baseline, all planner variants, team specs, high-N scaling and the figure replot at N=8 with one episode; `WITH_GLOBAL=1` adds the joint MILP and `WITH_TRAIN=1` the training pipelines (see the script header for `EPI`, `N` and `ONLY`).

**Performance.** When all agents share one formula structure (uniform `m*` specs, disjunctive team specs), per-agent STL guidance evaluates a single structural formula with per-agent center overrides, O(N) formula evaluations per guidance step (`gcbfplus/diffusion/util/stl_fastpath.py`). Structurally heterogeneous cases (mixed specs, pre-allocated team specs with `--team-avoid`) fall back to per-agent dispatch via `jax.lax.switch`, whose `vmap` evaluates all N branches per agent (O(N^2)). The EDM scaffolding shared between `edm.py` and `edm_ma.py` is a known deduplication follow-up.
