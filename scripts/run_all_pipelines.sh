#!/usr/bin/env bash
# run_all_pipelines.sh -- exercise every pipeline in the release once, at small scale.
#
# A smoke test for a fresh install and a worked example of each command family:
# controller evaluation, the CBF-QP baseline, the diffusion planner in all its variants,
# the STLPY MILP baseline, team (CaTL+) specifications, high-N goal scaling, random predicate
# sizes, an expressive STL fragment, and the offline figure replot. Optional groups add the
# joint MILP (slow) and the three training pipelines (controller, dataset, diffusion model).
#
# Usage:
#   bash scripts/run_all_pipelines.sh                 # evaluation pipelines + figures (~15 min on an 8 GB GPU)
#   EPI=2 N=16 bash scripts/run_all_pipelines.sh      # more episodes / agents
#   WITH_GLOBAL=1 bash scripts/run_all_pipelines.sh   # + STLPY-Global joint MILP (minutes per episode)
#   WITH_TRAIN=1 bash scripts/run_all_pipelines.sh    # + 1-step controller training (~20 min, compile-bound), dataset creation, 1-epoch diffusion training
#   ONLY="diffma stlpy" bash scripts/run_all_pipelines.sh   # subset of step labels
#
# Requirements: the conda environment from the README; a Gurobi license for the STLPY steps
# (they are reported as failed, not skipped, when it is missing). Logs go to
# logs/run_all_pipelines/<timestamp>/<step>.log; the summary prints at the end.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EPI="${EPI:-1}"
N="${N:-8}"
PY="${PY:-python}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="logs/run_all_pipelines/${STAMP}"
mkdir -p "$LOG_DIR"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"

CTRL="--path pretrained/DubinsCar/gcbf+/ --obs 0 --no-video"
PLAN="$CTRL --nojit-rollout --spec-len 15 --goal-sample-interval 20 --async-planner --ignore-on-finish --log"
RAND="--random-goals --traced-goals --change_goal_immediately --area-size 6"
DIFFMA="--planner diffusion --diffusion-method edm-ma --wandb-run-id qkmvppvt --use_batched_sampling --num_candidates 8"
TEAM="--area-size 6 --max-step 7200 --change_goal_immediately --team-avoid"

PASS=(); FAIL=(); SKIP=()
want() { [ -z "${ONLY:-}" ] || case " $ONLY " in *" $1 "*) return 0;; *) return 1;; esac; }

# step <label> [VAR=VAL ...] -- <command...>
step() {
  local label="$1"; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  want "$label" || { SKIP+=("$label"); return 0; }
  local log="$LOG_DIR/$label.log"
  echo; echo "==> $label"; echo "+ ${envs[*]} $*"
  local t0=$SECONDS
  if env "${envs[@]}" "$@" > "$log" 2>&1; then
    local rates; rates=$(grep -E '^reward:' "$log" | grep -oE 'safe_rate: [0-9.]+%, finish_rate: [0-9.]+%, success_rate: [0-9.]+%' | tail -1)
    local plan; plan=$(grep -oE "plan_time': [0-9.]+" "$log" | tail -1 | grep -oE '[0-9.]+')
    echo "   ok  ($((SECONDS - t0)) s)  ${rates:-}${plan:+  plan ${plan:0:6} s}"
    PASS+=("$label")
  else
    echo "   FAILED ($((SECONDS - t0)) s) -- see $log"; grep -E "Error|error:" "$log" | tail -2 | sed 's/^/   /'
    FAIL+=("$label")
  fi
}

# --- Controller and CBF-QP baseline (no planner) --------------------------------------------------
step controller -- $PY test.py $CTRL --epi $EPI --area-size 4 -n $N
step cbfqp      -- $PY test.py --env SingleIntegrator -n $N --algo dec_share_cbf --epi $EPI --area-size 4 --obs 0 --alpha 1 --no-video

# --- Diffusion planner (DIFF-MA), fixed grid and paper-default random predicates -------------------
step diffma_fixed -- $PY test.py $PLAN --area-size 6 --epi $EPI -n $N --spec mseq3 --planner diffusion
step diffma       -- $PY test.py $PLAN $RAND --epi $EPI -n $N --spec mseq3 $DIFFMA
step diffma_mixed -- $PY test.py $PLAN $RAND --epi $EPI -n $N --spec mseq3-m2branch2-mcover3-m2loop3-m2signal3 $DIFFMA
step diffma_ach   -- $PY test.py $PLAN $RAND --epi $EPI -n $N --spec mseq3 --planner diffusion --diffusion-method edm-ma --wandb-run-id qkmvppvt --num_candidates 8 --achievable_guidance
step diffsa       -- $PY test.py $PLAN $RAND --epi $EPI -n $N --spec m2branch2 --planner diffusion --diffusion-method edm --wandb-run-id qkmvppvt
step diffsa_bo8   -- $PY test.py $PLAN $RAND --epi $EPI -n $N --spec m2branch2 --planner diffusion --diffusion-method edm --wandb-run-id qkmvppvt --use_batched_sampling --num_candidates 8

# --- STLPY MILP baseline (Gurobi) ------------------------------------------------------------------
step stlpy -- $PY test.py $PLAN $RAND --epi $EPI -n $N --spec mseq3 --planner stlpy

# --- Predicate variations and an expressive fragment ------------------------------------------------
step signal_crowded -- $PY test.py $PLAN --area-size 6 --change_goal_immediately --random-goals --traced-goals --random-goals-spacing 2.0 --random-goals-region 1.5 --stl_mixed_spec_mode None --epi $EPI -n $N --spec m2signal3 $DIFFMA
step random_sizes   -- $PY test.py $PLAN $RAND --epi $EPI -n $N --spec mcover3 --random-goals-size 0.5 1.5 $DIFFMA
step avoid_until    -- $PY test.py $PLAN --area-size 6 --change_goal_immediately --epi $EPI -n $N --spec mauntil2 --stl_mixed_spec_mode last1 $DIFFMA

# --- Team specifications (CaTL+) --------------------------------------------------------------------
step team_redundant_emergent -- $PY test.py $PLAN $TEAM --epi $EPI -n $N --spec team_redun4x2 $DIFFMA --team-disjunctive
step team_choice_emergent GCBF_TEAM_SELECT_OUTER=1 -- $PY test.py $PLAN $TEAM --epi $EPI -n $N --spec team_choiceseq3 $DIFFMA --team-disjunctive
step team_choice_hungarian -- $PY test.py $PLAN $TEAM --epi $EPI -n $N --spec team_choiceseq3 $DIFFMA --team-alloc oracle_hungarian
step team_stlpy            -- $PY test.py $PLAN $TEAM --epi $EPI -n $N --spec team_redun4x2 --planner stlpy --team-alloc greedy
if [ -n "${WITH_GLOBAL:-}" ]; then
  step team_stlpy_global -- $PY test.py $PLAN $TEAM --epi $EPI -n $N --spec team_redun4x2 --planner stlpy_global
fi

# --- High-N goal-spread scaling (goal grid x2, area 10) -----------------------------------------------
step high_n_goal_scale GCBF_GOAL_SCALE=2 -- $PY test.py $PLAN --area-size 10 --epi $EPI -n $N --spec mseq3-m2branch2-mcover3-m2loop3-m2signal3 --planner diffusion --diffusion-method edm-ma --wandb-run-id qkmvppvt --num_candidates 8

# --- Training pipelines (optional) --------------------------------------------------------------------
if [ -n "${WITH_TRAIN:-}" ]; then
  step train_controller -- $PY train.py --algo gcbf+ --env DubinsCar -n $N --area-size 4 --steps 1 --n-env-train 2 --n-env-test 2 --debug
  # The diffusion planner is trained on single-agent trajectories (-n 1); a few episodes are
  # needed so the loader can hold out a validation split.
  DS_EPI=$(( EPI > 4 ? EPI : 4 ))
  step create_dataset   -- $PY create_dataset.py $PLAN --area-size 4 --epi $DS_EPI -n 1 --spec mseq3 --planner stlpy
  DATASET="$(ls -t datasets/DubinsCar_*.h5 2>/dev/null | head -1 | xargs -r basename)"
  if [ -n "$DATASET" ]; then
    step train_diffusion -- $PY train_diffusion.py --dataset_name "$DATASET" --num_epochs 1 --val_ratio 0.25 --diffusion_trajectory_mode sg --trajectory_length 15 --stl_train_traj_len 15 --num_features 32 --num_blocks 3 --batch_size 1
  else
    echo "==> train_diffusion: no datasets/DubinsCar_*.h5 found, skipping"; SKIP+=(train_diffusion)
  fi
fi

# --- Figures from the committed snapshots (offline) --------------------------------------------------
step replot_figures -- env JAX_PLATFORMS=cpu MPLBACKEND=Agg OUT_DIR="$LOG_DIR/figures" bash scripts/replot_all_final.sh

echo; echo "==================== summary ===================="
echo "PASS (${#PASS[@]}): ${PASS[*]:-none}"
echo "FAIL (${#FAIL[@]}): ${FAIL[*]:-none}"
[ ${#SKIP[@]} -gt 0 ] && echo "SKIP (${#SKIP[@]}): ${SKIP[*]}"
echo "logs: $LOG_DIR"
[ ${#FAIL[@]} -eq 0 ]
