#!/usr/bin/env bash
#
# replot_all_final.sh — regenerate every final paper figure/table in one shot.
#
# Each result is one `run_step` line that calls a plotting script. Add new results by
# writing a small `<name> () { ... }` function and a `run_step` line in the marked section
# below. Failures are collected (not fatal) so one missing result does not block the rest;
# the script exits non-zero if any step failed.
#
# Prereqs: the conda environment from the top-level README
#   csv mode (default) is fully offline from plot_snapshots/ — no GPU / W&B needed.
#   wandb mode needs W&B auth (and, for figures, the usual matplotlib stack).
#
# Env knobs:
#   SOURCE=csv|wandb   (default csv)   OUT_DIR=<dir> (default out_final)
# Examples:
#   bash scripts/replot_all_final.sh
#   SOURCE=wandb OUT_DIR=out_live bash scripts/replot_all_final.sh

set -uo pipefail

SOURCE="${SOURCE:-csv}"
OUT_DIR="${OUT_DIR:-out_final}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$OUT_DIR"

PASS=(); FAIL=()
run_step () {  # run_step "label" cmd...
  local label="$1"; shift
  echo; echo "==================== ${label} ===================="
  echo "+ $*"
  if "$@"; then PASS+=("$label"); else FAIL+=("$label"); echo "!! FAILED: ${label}"; fi
}

# ---------------------------------------------------------------------------
# Figures (plot_paper.py). --data-csv is added only in csv (offline) mode.
# ---------------------------------------------------------------------------
fig_single_hetero () {  # figure + matching LaTeX results table (--table)
  local a=(--figure single_hetero_dubins --source "$SOURCE" --out-dir "$OUT_DIR" --table)
  [ "$SOURCE" = csv ] && a+=(--data-csv plot_snapshots/single_hetero_DubinsCar_data.csv.gz)
  python plot_paper.py "${a[@]}"
}
fig_random_loc () {  # figure + matching LaTeX results table (--table)
  local a=(--figure random_loc_dubins --source "$SOURCE" --out-dir "$OUT_DIR" --table)
  [ "$SOURCE" = csv ] && a+=(--data-csv plot_snapshots/random_loc_DubinsCar_data.csv.gz)
  python plot_paper.py "${a[@]}"
}
fig_team_spec () {  # CaTL team-spec figure (redun4x + choiceseq); emergent scope only
  # local a=(--source "$SOURCE" --merge-edm --show-global --diffma-scope both --out-dir "$OUT_DIR")  # also emits the all-schemes variant
  local a=(--source "$SOURCE" --merge-edm --show-global --diffma-scope emergent --out-dir "$OUT_DIR")
  [ "$SOURCE" = csv ] && a+=(--data-csv plot_snapshots/team_spec_DubinsCar_data.csv)
  python scripts/plot_team_spec_results.py "${a[@]}"
}
fig_high_n () {  # high-N scaling curve + Safe/Finish/Success table explaining the success bars
  local a=(--source "$SOURCE" --out-dir "$OUT_DIR" --table)
  [ "$SOURCE" = csv ] && a+=(--data-csv plot_snapshots/high_n_mixed_DubinsCar_data.csv)
  python scripts/plot_high_n_scaling.py "${a[@]}"
}

# ---------------------------------------------------------------------------
# Tables.
# ---------------------------------------------------------------------------
tbl_ach_mixed4 () {  # achievable-loss ablation, 4-component Mixed (swap FLAG per overrides yaml)
  python scripts/make_ach_ablation_table.py --source "$SOURCE" \
    --out "$OUT_DIR/ach_ablation_mixed4.tex"
}

# run_step "fig: single_hetero (fixed centers)" fig_single_hetero   # fixed predicate centers: not plotted
run_step "fig: random_loc (random pred.)" fig_random_loc
run_step "fig: team_spec (CaTL)"          fig_team_spec
run_step "fig: high_n scaling (N=8..128)" fig_high_n
run_step "tbl: achievable ablation Mixed" tbl_ach_mixed4

# ---------------------------------------------------------------------------
# ADD YOUR RESULTS BELOW
#   1) define   my_result () { python scripts/<your_script>.py --source "$SOURCE" ... ; }
#   2) call     run_step "label: my result" my_result
# ---------------------------------------------------------------------------


# ---- summary ----
echo; echo "==================== summary ===================="
echo "PASS (${#PASS[@]}): ${PASS[*]:-none}"
echo "FAIL (${#FAIL[@]}): ${FAIL[*]:-none}"
echo "outputs in: $OUT_DIR"
[ "${#FAIL[@]}" -eq 0 ]
