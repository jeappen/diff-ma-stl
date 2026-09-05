#!/usr/bin/env python
"""MAINTAINER TOOL -- regenerate the release reproduction bundle from W&B.

Writes, next to this file:
  final_plots_manifest.csv    one row per plotted bar (or per panel-role of a bar): which W&B
                              run it came from, the exact launch args, env knobs, commit, and the
                              reference metrics the paper figure shows.
  reproduce_final_plots.sh    the exact `python test.py ...` command for every row, wrapped in a
                              resumable runner. GENERATED -- do not hand-edit.

Release users never run this (it needs read access to the csbric W&B projects). They run
reproduce_final_plots.sh and then plot_reproduced.py, which only need the manifest.

How the plotted runs are identified (no guessing): each figure's own offline pipeline is
executed on its committed snapshot and asked which run id ended up in each cell --
  random_loc : plot_paper.py       (Plotter pipeline; the run id is carried through the
                                    aggfunc="first" pivot as an extra column)
  team_spec  : plot_team_spec_results.prepare(scope="emergent")
  high_n     : plot_high_n_scaling.prepare()   (success run and timing run per cell)
so the manifest is by construction the set of runs behind out_final/. Launch args, git commit
and env knobs come from each run's stored wandb-metadata (run.metadata["args"], ["git"]) and
config (env_gcbf_goal_scale etc. are recorded by test.py from the GCBF_* environment).

Usage:  python scripts/release/_build_manifest.py [--no-fetch]   (--no-fetch reuses a cached
        metadata json next to this file, for iterating on the bash template offline)
"""
import argparse
import json
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
HERE = os.path.dirname(os.path.abspath(__file__))

import pandas as pd
import yaml

ENTITY = "csbric"
# Code the commands are verified against (every flag below exists in its argparse). All the
# W&B runs behind the diffusion / STLPY arms were launched from commits that are ancestors of it.
CODE_PIN = {"branch": "dbg-ach-loss", "commit": "d1f2aa2", "date": "2026-07-12"}
PATH_NORMAL = "./pretrained/DubinsCar/gcbf+/"
LOG_DIR = "pretrained/DubinsCar/gcbf+"

# Old-local-test_log rows (team-spec STLPY-SA override) carry no launch args; these are the
# flags every W&B STLPY team run of the same code family used, so the reconstruction is the
# closest command the pinned code accepts. Budget/alloc/epi come from the row itself.
OVERRIDE_COMMON = ("--obs 0 --nojit-rollout --spec-len 15 --goal-sample-interval 20 "
                   "--async-planner --ignore-on-finish --no-video --log --change_goal_immediately")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Which run is behind each bar -- ask each figure's own pipeline.
# ─────────────────────────────────────────────────────────────────────────────
def select_random_loc():
    import plot_paper as pp
    spec = pp.FIGURES["random_loc_dubins"]
    df = pp.load_df(spec, "csv", os.path.join(ROOT, "plot_snapshots/random_loc_DubinsCar_data.csv.gz"),
                    "out_final")
    pk = pp.apply_signal_toggle(dict(spec.plot_kwargs), True)
    pk.pop("homo_mode", None); pk.pop("plot_grouped", None); pk.pop("debug", None)
    # Appending "id" makes it another pivot VALUE column, so the pivot's aggfunc="first" carries
    # the run id of the very same best row it picked for the metrics -- no re-derivation, no guess.
    pk["columns_to_print"] = list(pk["columns_to_print"]) + ["id"]
    plotter = pp.Plotter(df, env_name=spec.env_name, **spec.plotter_kwargs)
    piv, _ = plotter.plot_all(homo_specs=False, make_bar=False, return_table=False,
                              use_stddev=True, **pk)
    sel = piv["id"].stack().rename("id").reset_index()          # Spec, N, Planner, id
    sel = sel.rename(columns={"level_2": "Planner"}) if "level_2" in sel.columns else sel
    raw = df.set_index("id")
    rows = []
    for _, r in sel.iterrows():
        rr = raw.loc[r["id"]]
        rows.append(dict(fig="random_loc", spec=rr["spec"], spec_label=r["Spec"], num_agents=int(r["N"]),
                         planner_label=r["Planner"], role="bar", run_id=r["id"],
                         project=str(rr["url"]).split("/")[-3], source="random_loc",
                         stl_mixed_spec_mode=rr.get("stl_mixed_spec_mode")))
    return rows


def select_team_spec():
    import plot_team_spec_results as ts
    cfg = yaml.safe_load(open(os.path.join(ROOT, "scripts/team_spec_plot_config.yaml")))
    cli = types.SimpleNamespace(source="csv", data_csv=os.path.join(ROOT, cfg["snapshot"]))
    best = ts.prepare(ts.load_raw(cfg, cli), "emergent", cfg)
    rows = []
    for _, r in best.iterrows():
        rid = r.get("id")
        row = dict(fig="team_spec", spec=r["spec"], spec_label=r["Spec"], num_agents=int(r["N"]),
                   planner_label=r["Planner"], role="bar", source="team_spec",
                   project="gcbfplus-stl-test", gs=r.get("gs"))
        if isinstance(rid, str) and rid:
            row["run_id"] = rid
        else:   # STLPY-SA override: an old local test_log.csv row, no W&B run
            row.update(run_id=f"local_{re.sub('[^a-z0-9]', '', r['Spec'].lower())[:6]}_N{int(r['N'])}",
                       project="", override=True,
                       ref=dict(success_mean=r["success"], success_std=r["success_s"],
                                task_rate=r["task_pct"] / 100, TtR=r["ttr"], TtR_std=r["ttr_s"],
                                plan_time_mean=r["pt"], plan_time_std=r["pt_s"], epi=r.get("epi"),
                                max_step=r.get("max_step"), team_alloc=r.get("team_alloc"),
                                safe_mean=r.get("safe_mean"), safe_std=r.get("safe_std"),
                                finish_rate=r.get("finish_rate"), finish_rate_std=r.get("finish_rate_std")))
        rows.append(row)
    return rows


def select_high_n():
    import plot_high_n_scaling as hn
    cfg = yaml.safe_load(open(os.path.join(ROOT, "scripts/high_n_plot_config.yaml")))
    cli = types.SimpleNamespace(source="csv", data_csv=os.path.join(ROOT, cfg["snapshot"]))
    agg = hn.prepare(hn.load_raw(cfg, cli), cfg)
    rows = []
    for _, r in agg.iterrows():
        base = dict(fig="high_n", spec=r["spec"], spec_label=r["Spec"], num_agents=int(r["N"]),
                    planner_label=r["Planner"], project="diff-mastl-test")
        if r["id"] == r["_timing_id"]:
            rows.append(dict(base, role="both", run_id=r["id"], source=r["_source"]))
        else:
            rows.append(dict(base, role="success", run_id=r["id"], source=r["_source"]))
            rows.append(dict(base, role="timing", run_id=r["_timing_id"], source=r["_timing_source"]))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 2. Launch metadata from W&B.
# ─────────────────────────────────────────────────────────────────────────────
SUMMARY = {"success_mean": ("success_mean", 100), "success_std": ("success_std", 100),
           "safe_mean": ("safe_mean", 100), "safe_std": ("safe_std", 100),
           "finish_rate": ("eval/finish_rate", 1), "finish_rate_std": ("eval/finish_rate_std", 1),
           "TtR": ("eval/TtR", 1), "TtR_std": ("eval/TtR_std", 1),
           "plan_time_mean": ("plan_time_mean", 1), "plan_time_std": ("plan_time_std", 1),
           "task_rate": ("eval/task_rate", 1), "task_rate_std": ("eval/task_rate_std", 1)}
CONFIG = ["epi", "max_step", "area_size", "stl_mixed_spec_mode", "diffusion_method",
          "achievable_guidance", "use_batched_sampling", "random_goals_region", "team_alloc",
          "spec_len", "env_gcbf_goal_scale", "env_gcbf_team_select_outer"]
ENV_KNOBS = ["env_gcbf_goal_scale", "env_gcbf_team_select_outer", "env_gcbf_ma_accept_thresh",
             "env_gcbf_task_repair", "env_gcbf_task_repair_grad", "env_gcbf_keep_best",
             "env_gcbf_ma_accept", "env_gcbf_ach_select", "env_gcbf_ach_async_select"]


def fetch(project_ids, cache):
    if os.path.exists(cache):
        got = json.load(open(cache))
    else:
        got = {}
    todo = [(p, i) for p, i in project_ids if i not in got]
    if todo:
        import wandb
        api = wandb.Api(timeout=120)
        for k, (proj, rid) in enumerate(todo):
            run = api.run(f"{ENTITY}/{proj}/{rid}")
            md = run.metadata or {}
            cfg = run.config
            got[rid] = dict(
                url=run.url, state=run.state, created=str(run.created_at), name=run.name,
                program=md.get("program"), args=md.get("args") or [],
                commit=(md.get("git") or {}).get("commit"), host=md.get("host"),
                config={k: cfg.get(k) for k in CONFIG},
                env={k[4:].upper(): cfg[k] for k in ENV_KNOBS if cfg.get(k) is not None},
                summary={k: run.summary.get(sk) for k, (sk, _) in SUMMARY.items()})
            print(f"  fetched {k + 1}/{len(todo)} {rid}", flush=True)
        json.dump(got, open(cache, "w"), indent=1)
    return got


# ─────────────────────────────────────────────────────────────────────────────
# 3. Normalise args into a reproduction command.
# ─────────────────────────────────────────────────────────────────────────────
def normalise(args):
    """Verbatim launch args minus logging-to-the-original-W&B-project bits.

    Drops --wandb-log, --wandb-tags <t...> and --test-log-suffix <s> (the runner re-adds a
    per-run suffix), normalises the two spellings of --path. Everything else -- including
    explicit defaults like --seed 1234 -- is kept in the original order.
    """
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a == "--wandb-log":
            i += 1; continue
        if a == "--wandb-tags":
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                i += 1
            continue
        if a == "--test-log-suffix":
            i += 2; continue
        if a == "--path":
            out += ["--path", PATH_NORMAL]; i += 2; continue
        out.append(a); i += 1
    return out


def override_args(row):
    """Closest command for an old local test_log row (no W&B run). See OVERRIDE_COMMON."""
    ref = row["ref"]
    alloc = ref.get("team_alloc") or "greedy"
    return (f"--path {PATH_NORMAL} --epi {int(ref['epi'])} --area-size 6 --max-step {int(ref['max_step'])} "
            f"{OVERRIDE_COMMON} --team-avoid -n {row['num_agents']} --spec {row['spec']} "
            f"--planner stlpy --team-alloc {alloc} --stl_mixed_spec_mode first1").split()


def env_str(env):
    return " ".join(f"{k}={v}" for k, v in sorted(env.items()))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Emit.
# ─────────────────────────────────────────────────────────────────────────────
BASH_HEADER = r'''#!/usr/bin/env bash
# reproduce_final_plots.sh -- re-run every evaluation behind the three paper figures.
#
# GENERATED by scripts/release/_build_manifest.py from the W&B runs that were actually plotted
# (one `run` line per bar; provenance in each comment and in final_plots_manifest.csv).
# Do not hand-edit commands -- regenerate.
#
# Figures reproduced (then drawn by scripts/release/plot_reproduced.py):
#   random_loc : random-predicate setting, 6 specs x N=8/16/32 x {DIFF-MA, DIFF-SA, STLPY-SA, Gradient}
#   team_spec  : CaTL team specs (Choice / Redundant) x N=8/16/32 x {DIFF-MA, STLPY-SA, STLPY-Global}
#   high_n     : Mixed spec, N=8..128, DIFF-MA vs STLPY-SA (success from the native/gs2.0 split,
#                TtR + plan time from uniform gs2.0 -- so N<=32 needs TWO runs per cell)
#
# CODE PIN: __CODE_PIN__
#   Every command below was checked against that commit's argparse. The runs themselves were
#   launched from ancestors of it (git hashes in the comments).
#
# PREREQS
#   conda activate gcbfplus   (GPU environment per the top-level README)
#   pretrained/DubinsCar/gcbf+/            the GCBF+ tracking controller (bundled)
#   diff_checkpoints/qkmvppvt/             the diffusion planner (bundled; loads without W&B)
#   Gurobi licence reachable               STLPY-SA / STLPY-Global rows (MILP)
#   GPU with >= 8 GB                       diffusion rows; N=128 peaks ~6.7 GB
#
# USAGE
#   bash scripts/release/reproduce_final_plots.sh                  # everything
#   FIGS="high_n" bash scripts/release/reproduce_final_plots.sh    # one figure (space-separated list)
#   ONLY=f0td1sgv bash scripts/release/reproduce_final_plots.sh    # one original run id
#   DRY_RUN=1 bash ...                                             # print the commands, run nothing
#   FORCE=1 bash ...                                               # re-run rows whose log exists
#   WANDB_ARGS="--wandb-log --wandb-tags repro" bash ...           # also log to YOUR W&B
#
# OUTPUT
#   Each row appends one line to pretrained/DubinsCar/gcbf+/test_log_repro_<id>.csv (test.py --log
#   with --test-log-suffix). A row whose log already exists is skipped, so the script is resumable.
#   Then:  python scripts/release/plot_reproduced.py --out-dir out_repro
#
# KNOWN LIMITS (read before trusting a diff against the paper)
#   * Gradient (ce_nl) arm of random_loc is NOT re-run by this script. Those 18 runs came from an
#     unpublished development commit (0274dda0) and the pinned code's
#     --planner accepts only {stlpy, diffusion, stlpy_global}. Their original commands are listed
#     as comments for the record; plot_reproduced.py draws those bars from the paper's recorded
#     numbers (manifest ref_* columns) and labels them "ref" in its provenance print.
#   * STLPY-SA arm of team_spec: the paper's bars come from older local test_log.csv rows (no W&B
#     run, pre "spec-v2" code, epi 5/10). The commands are reconstructed from the logged settings
#     and marked "best-effort"; on the pinned (spec-v2) code STLPY-SA scores differ, especially
#     on Choice (the W&B spec-v2 STLPY-SA runs are tabulated in scripts/release/README.md).
#   * Every cell is a single seed (--seed 1234 default); N>=64 high_n cells are epi=3. Expect
#     run-to-run variance of a few points on success, more on TtR.
#   * Runtime: STLPY rows are minutes-to-hours each (N=128 ~195 s/plan x 3 episodes x many replans);
#     STLPY-Global rows ~800 s/episode x 10. Diffusion rows are seconds-to-minutes per episode.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."          # repo root
FIGS="${FIGS:-random_loc team_spec high_n}"
LOG_DIR="__LOG_DIR__"
WANDB_ARGS="${WANDB_ARGS:-}"
PASS=(); FAIL=(); SKIP=()

want_fig () { case " $FIGS " in *" $1 "*) return 0;; *) return 1;; esac; }

# run <fig> <id> [KEY=VAL ...] -- <test.py args...>
run () {
  local fig="$1" id="$2"; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  want_fig "$fig" || return 0
  [ -n "${ONLY:-}" ] && [ "$ONLY" != "$id" ] && return 0
  local log="$LOG_DIR/test_log_repro_${id}.csv"
  if [ -s "$log" ] && [ -z "${FORCE:-}" ]; then echo "[skip] $id -> $log exists"; SKIP+=("$id"); return 0; fi
  echo; echo "==> [$fig] $id"; echo "+ ${envs[*]} python test.py $* --test-log-suffix _repro_${id} ${WANDB_ARGS}"
  [ -n "${DRY_RUN:-}" ] && return 0
  # shellcheck disable=SC2086
  if env "${envs[@]}" python test.py "$@" --test-log-suffix "_repro_${id}" ${WANDB_ARGS}; then
    PASS+=("$id"); else FAIL+=("$id"); echo "!! FAILED $id"; fi
}
'''

BASH_FOOTER = r'''
echo; echo "==================== summary ===================="
echo "PASS (${#PASS[@]}): ${PASS[*]:-none}"
echo "SKIP (${#SKIP[@]}): ${SKIP[*]:-none}"
echo "FAIL (${#FAIL[@]}): ${FAIL[*]:-none}"
echo "logs in: $LOG_DIR/test_log_repro_<id>.csv   ->   python scripts/release/plot_reproduced.py --out-dir out_repro"
[ "${#FAIL[@]}" -eq 0 ]
'''

SECTION = {
    "random_loc": "FIGURE random_loc  (plot_paper.py --figure random_loc_dubins)\n"
                  "# area-6, random predicate locations (--random-goals), epi 10 (Gradient: 5); Signal uses the\n"
                  "# crowded R=1.5 region runs; DIFF-SA = single-draw edm except the edm-bo8 cells; Loop DIFF-MA\n"
                  "# = no achievable guidance. Selection rules: plot_config.yaml -> random_loc_dubins.",
    "team_spec": "FIGURE team_spec  (scripts/plot_team_spec_results.py --merge-edm --show-global, scope=emergent)\n"
                 "# area-6, epi 10, max-step 7200, team-avoid; DIFF-MA = emergent (team-disjunctive) edm-ma; Choice N=32\n"
                 "# DIFF-MA uses GCBF_GOAL_SCALE=1.25 decongestion. STLPY-SA rows: see KNOWN LIMITS.",
    "high_n": "FIGURE high_n  (scripts/plot_high_n_scaling.py)\n"
              "# Mixed spec mseq3-m2branch2-mcover3-m2loop3-m2signal3. success panel: native area-6/epi-10 at N<=32,\n"
              "# GCBF_GOAL_SCALE=2.0/area-10/epi-3 at N>=64. TtR + plan-time panels: the gs2.0 run at EVERY N.",
}


def build(no_fetch):
    rows = select_random_loc() + select_team_spec() + select_high_n()
    ids = [(r["project"], r["run_id"]) for r in rows if not r.get("override")]
    cache = os.path.join(HERE, "_wandb_metadata_cache.json")
    if no_fetch and not os.path.exists(cache):
        sys.exit("--no-fetch but no cache present")
    meta = fetch(sorted(set(ids)), cache)

    manifest, bash = [], [BASH_HEADER.replace("__CODE_PIN__", f"{CODE_PIN['branch']} @ {CODE_PIN['commit']} "
                                                              f"({CODE_PIN['date']})")
                             .replace("__LOG_DIR__", LOG_DIR)]
    order = {"random_loc": 0, "team_spec": 1, "high_n": 2}
    label_order = {"DIFF-MA": 0, "DIFF-MA (Ours)": 0, "DIFF-SA": 1, "STLPY-SA": 2, "STLPY-Global": 3, "Gradient": 4}
    rows.sort(key=lambda r: (order[r["fig"]], r["spec_label"], r["num_agents"],
                             label_order.get(r["planner_label"], 9), r.get("role", "")))
    cur = None
    for r in rows:
        if r["fig"] != cur:
            cur = r["fig"]
            bash.append(f"\n# {'=' * 96}\n# {SECTION[cur]}\n# {'=' * 96}")
        rid = r["run_id"]
        if r.get("override"):
            m = dict(url="", created="", commit="", host="", program="", args=[], env={},
                     config=dict(epi=r["ref"]["epi"], max_step=r["ref"]["max_step"], area_size=6,
                                 team_alloc=r["ref"].get("team_alloc"), spec_len=15),
                     summary=r["ref"])
            rargs = override_args(r)
            repro = "best-effort"
            note = ("old local test_log.csv row (pre spec-v2 code, no W&B run); flags reconstructed from the "
                    "logged settings")
            src = "old-local-testlog"
        else:
            m = meta[rid]
            rargs = normalise(m["args"])
            if r["planner_label"] == "Gradient":
                repro, note = "no-code", ("not re-run: ce_nl planner is not in the pinned code (unpublished commit "
                                          "0274dda0); plot_reproduced.py uses the ref_* values")
            else:
                repro, note = "yes", ""
            src = r["source"]
        envs = m["env"]
        cfg = m["config"]
        s = m["summary"]
        if not r.get("override"):
            # W&B summary stores success/safe as 0-1; the paper loaders and test_log.csv use
            # percent. Apply the per-key scale so every ref_* column is on the test_log scale.
            s = {k: (None if s.get(k) is None else s[k] * SUMMARY[k][1]) for k in SUMMARY}
        manifest.append(dict(
            fig=r["fig"], spec=r["spec"], spec_label=r["spec_label"], num_agents=r["num_agents"],
            planner_label=r["planner_label"], role=r["role"], source=src, reproducible=repro,
            run_id=rid, project=r["project"], url=m["url"], commit=(m["commit"] or "")[:8],
            created=m["created"][:10], host=m["host"], env=env_str(envs),
            repro_args=" ".join(rargs), orig_args=" ".join(m["args"]), log_suffix=f"_repro_{rid}",
            epi=cfg.get("epi"), max_step=cfg.get("max_step"), area_size=cfg.get("area_size"),
            spec_len=cfg.get("spec_len"), stl_mixed_spec_mode=r.get("stl_mixed_spec_mode", cfg.get("stl_mixed_spec_mode")),
            diffusion_method=cfg.get("diffusion_method"), achievable_guidance=cfg.get("achievable_guidance"),
            use_batched_sampling=cfg.get("use_batched_sampling"), random_goals_region=cfg.get("random_goals_region"),
            team_alloc=cfg.get("team_alloc"), gs=r.get("gs"),
            **{f"ref_{k}": s.get(k) for k in SUMMARY},
            note=note))
        # bash line
        succ = s.get("success_mean"); pt = s.get("plan_time_mean"); ttr = s.get("TtR")
        prov = (f"{r['project']}/{rid} commit {(m['commit'] or '')[:8]} {m['created'][:10]}" if m["url"]
                else "old local test_log.csv (no W&B run) -- BEST-EFFORT reconstruction")
        metr = " ".join(x for x in [f"success={succ:.1f}" if succ is not None and succ == succ else "",
                                    f"plan={pt:.2f}s" if pt is not None and pt == pt else "",
                                    f"TtR={ttr:.0f}" if ttr is not None and ttr == ttr else ""] if x)
        role = "" if r["role"] in ("bar", "both") else f" [{r['role']} panel]"
        bash.append(f"# {r['spec_label']:16s} N={r['num_agents']:<3d} {r['planner_label']:12s}{role}  <- {prov}  ({metr})")
        env_tokens = " ".join(f"{k}={v}" for k, v in sorted(envs.items()))
        line = f"run {r['fig']} {rid} {env_tokens + ' ' if env_tokens else ''}-- {' '.join(rargs)}"
        if r["planner_label"] == "Gradient":     # not re-run (see KNOWN LIMITS); record only
            line = f"#   NOT RUN (ce_nl not in pinned code; bar drawn from manifest ref values):\n#   {line}"
        bash.append(line)
    bash.append(BASH_FOOTER)

    mpath = os.path.join(HERE, "final_plots_manifest.csv")
    pd.DataFrame(manifest).to_csv(mpath, index=False)
    spath = os.path.join(HERE, "reproduce_final_plots.sh")
    with open(spath, "w") as f:
        f.write("\n".join(bash))
    os.chmod(spath, 0o755)
    df = pd.DataFrame(manifest)
    print(f"\n[manifest] {mpath}: {len(df)} rows, {df['run_id'].nunique()} distinct runs")
    print(df.groupby(["fig", "reproducible"]).size().to_string())
    print(f"[script]   {spath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fetch", action="store_true", help="reuse _wandb_metadata_cache.json, no W&B calls")
    build(ap.parse_args().no_fetch)
