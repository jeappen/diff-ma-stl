"""Draw the three paper figures from the runs produced by reproduce_final_plots.sh.

For every row of final_plots_manifest.csv (one row per plotted bar, or per panel-role of a bar)
this reads the one-line local log that `test.py --log --test-log-suffix` wrote:

    {log_dir}/test_log_repro_<run_id>.csv

and hands the numbers to the SAME renderers the paper figures were drawn with -- no W&B access:

    random_loc -> plot_paper.render / emit_table         (Plotter pipeline)
                  {out}/random_loc_DubinsCar_<metrics>.pdf, _table.tex, _data.csv, _run_ids.csv
    team_spec  -> plot_team_spec_results.render           {out}/team_spec_comparison_merged_emergent.pdf
    high_n     -> plot_high_n_scaling.render / emit_table {out}/high_n_scaling_mixed.pdf, _table.tex

Missing logs stop the script with a list, unless --fallback-manifest: then those bars are filled
from the reference numbers recorded in the manifest (the paper's W&B values) and each such bar
is labelled "ref" in the provenance print. This is also the self-check: with an EMPTY log dir and
--fallback-manifest the output must be pixel-identical to the paper's out_final/ figures.

Usage:
    python scripts/release/plot_reproduced.py --out-dir out_repro
    python scripts/release/plot_reproduced.py --figs high_n,team_spec --fallback-manifest
"""
import argparse
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import yaml

# Metric columns read from a test_log row. Scales match the paper loaders: success/safe are
# percent, finish_rate/task_rate are 0-1, TtR is steps, plan time is seconds.
METRICS = ["success_mean", "success_std", "safe_mean", "safe_std", "finish_rate", "finish_rate_std",
           "TtR", "TtR_std", "plan_time_mean", "plan_time_std", "task_rate", "task_rate_std"]
PATH = "./pretrained/DubinsCar/gcbf+/"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(HERE, "final_plots_manifest.csv"))
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "pretrained/DubinsCar/gcbf+"),
                    help="where test.py --log wrote test_log_repro_<id>.csv (its --path dir)")
    ap.add_argument("--out-dir", default="out_repro")
    ap.add_argument("--figs", default="random_loc,team_spec,high_n",
                    help="comma-separated subset of random_loc,team_spec,high_n")
    ap.add_argument("--fallback-manifest", action="store_true",
                    help="use the manifest's reference (paper) numbers for bars whose log is missing")
    return ap.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Logs -> one metrics dict per manifest row
# ─────────────────────────────────────────────────────────────────────────────
def read_log(path):
    df = pd.read_csv(path, keep_default_na=False, na_values=["", "NaN", "nan"])
    if df.empty:
        raise ValueError(f"{path}: header only, no result row")
    if len(df) > 1:
        print(f"[warn] {os.path.basename(path)}: {len(df)} rows, using the last")
    row = df.iloc[-1]
    return {k: (pd.to_numeric(row[k], errors="coerce") if k in row else np.nan) for k in METRICS}


def load(manifest, log_dir, figs, fallback):
    m = pd.read_csv(manifest, keep_default_na=False, na_values=[""])
    m = m[m["fig"].isin(figs)].copy()
    vals, prov, missing = [], [], []
    for _, r in m.iterrows():
        # log_suffix is "_repro_<run_id>" -- exactly the --test-log-suffix that
        # reproduce_final_plots.sh passes to test.py, so each manifest row maps to one local log.
        path = os.path.join(log_dir, f"test_log{r['log_suffix']}.csv")
        ref = {k: pd.to_numeric(r.get(f"ref_{k}"), errors="coerce") for k in METRICS}
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            vals.append(read_log(path)); prov.append("log")
        elif r.get("reproducible") == "no-code":
            # Gradient (ce_nl) rows are never re-run (planner absent from the pinned code); the
            # paper's recorded numbers are the only source, so they are used without a flag.
            vals.append(ref); prov.append("ref")
        elif fallback:
            # No reproduced log for this bar: fill it from the paper's recorded numbers
            # (manifest ref_* columns) and label the bar "ref" in the provenance print.
            vals.append(ref); prov.append("ref")
        else:
            vals.append({k: np.nan for k in METRICS}); prov.append("MISSING"); missing.append(path)
    if missing:
        print("\n".join(f"  missing: {p}" for p in missing))
        sys.exit(f"{len(missing)} reproduced log(s) missing (run reproduce_final_plots.sh, or pass "
                 f"--fallback-manifest to fill them with the paper's reference numbers)")
    for k in METRICS:
        m[k] = [v[k] for v in vals]
    m["prov"] = prov
    n_ref = prov.count("ref")
    print(f"[load] {len(m)} bars: {len(m) - n_ref} from reproduced logs, {n_ref} from manifest "
          f"reference values (Gradient/ce_nl bars are always reference -- not re-run)")
    return m


def show(m, fig):
    d = m[m["fig"] == fig].sort_values(["spec_label", "num_agents", "planner_label", "role"])
    print(f"\n[{fig}] bars (provenance -> run):")
    for _, r in d.iterrows():
        role = "" if r["role"] in ("bar", "both") else f" [{r['role']}]"
        print(f"  {r['spec_label']:16s} N={int(r['num_agents']):<4d} {r['planner_label']:12s}{role:10s} "
              f"{r['prov']:4s} {r['run_id']:22s} success={r['success_mean']:5.1f} "
              f"plan={r['plan_time_mean']:7.2f}s TtR={r['TtR']:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Frames in the exact shape each paper renderer consumes
# ─────────────────────────────────────────────────────────────────────────────
def _env(env_str, key, default=np.nan):
    for tok in str(env_str).split():
        k, _, v = tok.partition("=")
        if k == key:
            return pd.to_numeric(v, errors="coerce")
    return default


def draw_high_n(m, out_dir):
    import plot_high_n_scaling as hn
    cfg = yaml.safe_load(open(os.path.join(ROOT, "scripts/high_n_plot_config.yaml")))
    d = m[m["fig"] == "high_n"]
    succ = d[d["role"].isin(["success", "both"])]
    tim = d[d["role"].isin(["timing", "both"])]
    agg = pd.DataFrame({
        "Spec": succ["spec_label"].values, "N": succ["num_agents"].astype(int).values,
        "Planner": succ["planner_label"].values,
        "success": succ["success_mean"].values, "success_s": succ["success_std"].fillna(0).values,
        "safe": succ["safe_mean"].values, "finish": succ["finish_rate"].values * 100,
        "epi": succ["epi"].values, "area_size": succ["area_size"].values,
        "env_gcbf_goal_scale": [_env(e, "GCBF_GOAL_SCALE") for e in succ["env"]],
        "id": succ["run_id"].values, "_source": succ["source"].values, "spec": succ["spec"].values})
    t = pd.DataFrame({
        "N": tim["num_agents"].astype(int).values, "Planner": tim["planner_label"].values,
        "ttr": tim["TtR"].values, "ttr_s": tim["TtR_std"].fillna(0).values,
        "pt": tim["plan_time_mean"].values, "pt_s": tim["plan_time_std"].fillna(0).values,
        "_timing_id": tim["run_id"].values, "_timing_source": tim["source"].values})
    agg = agg.merge(t, on=["N", "Planner"], how="left")
    exp = cfg["expect"]
    assert len(agg) == exp["total_rows"], f"high_n: expected {exp['total_rows']} cells, got {len(agg)}"
    hn.emit_table(agg, cfg, out_dir)
    hn.render(agg, cfg, out_dir)


def draw_team_spec(m, out_dir):
    import plot_team_spec_results as ts
    cfg = yaml.safe_load(open(os.path.join(ROOT, "scripts/team_spec_plot_config.yaml")))
    d = m[m["fig"] == "team_spec"]
    agg = pd.DataFrame({
        "Spec": d["spec_label"].values, "N": d["num_agents"].astype(int).values,
        "Planner": d["planner_label"].values,
        "success": d["success_mean"].values, "success_s": d["success_std"].fillna(0).values,
        "task_pct": d["task_rate"].values * 100,
        "ttr": d["TtR"].values, "ttr_s": d["TtR_std"].fillna(0).values,
        "pt": d["plan_time_mean"].values, "pt_s": d["plan_time_std"].fillna(0).values})
    cli = types.SimpleNamespace(merge_edm=True, one_row=False, show_global=True, out_dir=out_dir)
    ts.render(agg, "emergent", cfg, cli)


_PLANNER_RAW = {"DIFF-MA": ("diffusion", "edm-ma"), "DIFF-SA": ("diffusion", "edm"),
                "STLPY-SA": ("stlpy", None), "Gradient": ("ce_nl", None)}


def draw_random_loc(m, out_dir):
    import plot_paper as pp
    spec_obj = pp.FIGURES["random_loc_dubins"]
    d = m[m["fig"] == "random_loc"]
    raw = [_PLANNER_RAW[p] for p in d["planner_label"]]
    df = pd.DataFrame({
        # identity / tiebreak columns the Plotter sorts and exports on
        "id": d["run_id"].values, "url": d["url"].values, "name": d["run_id"].values, "path": PATH,
        "wandb_run_id": ["qkmvppvt" if p == "diffusion" else np.nan for p, _ in raw],
        # arm-defining config
        "planner": [p for p, _ in raw], "diffusion_method": [dm for _, dm in raw],
        "achievable_guidance": d["achievable_guidance"].astype(str).values,
        "use_batched_sampling": d["use_batched_sampling"].astype(str).values,
        "random_goals": True, "random_goals_region": d["random_goals_region"].values,
        "spec": d["spec"].values, "num_agents": d["num_agents"].astype(int).values,
        "epi": d["epi"].values, "spec_len": d["spec_len"].values,
        "stl_mixed_spec_mode": d["stl_mixed_spec_mode"].astype(str).values,
        "n_obs": 0, "obs": 0, "async_planner": True,
        # metrics
        **{k: d[k].values for k in METRICS}})
    # Same post-load steps as plot_paper.load_df (env/obs/epi/Signal-R1.5 filters + the
    # per-spec variant choices). With one run per cell they are no-ops, kept for fidelity.
    df = pp.prepare_df(df, spec_obj)
    df = pp.drop_achievable_for_specs(df, spec_obj.no_ach_specs)
    df = pp.drop_batched_for_specs(df, spec_obj.single_draw_specs)
    pk = pp.apply_signal_toggle(dict(spec_obj.plot_kwargs), include_signal=True)
    pp.render(spec_obj, df, out_dir, plot_kwargs=pk)
    pp.emit_table(spec_obj, df, out_dir, plot_kwargs=pk)


DRAW = {"random_loc": draw_random_loc, "team_spec": draw_team_spec, "high_n": draw_high_n}


def main(argv=None):
    cli = parse_args(argv)
    figs = [f.strip() for f in cli.figs.split(",") if f.strip()]
    bad = [f for f in figs if f not in DRAW]
    if bad:
        sys.exit(f"unknown figure(s) {bad}; choose from {list(DRAW)}")
    os.makedirs(cli.out_dir, exist_ok=True)
    m = load(cli.manifest, cli.log_dir, figs, cli.fallback_manifest)
    # Each renderer draws under ITS OWN matplotlib style, exactly as when run as its own script:
    # the Plotter (random_loc) under plot.py's paper style, the bar-grid figures under the
    # single-column style. Applying one style globally shifts the other's grid lines.
    from gcbfplus.utils.plot import apply_paper_style
    from gcbfplus.utils.spec_bar_figure import apply_column_style
    STYLE = {"random_loc": apply_paper_style, "team_spec": apply_column_style, "high_n": apply_column_style}
    for fig in figs:
        show(m, fig)
        STYLE[fig]()
        DRAW[fig](m, cli.out_dir)
    print(f"\n[done] figures in {cli.out_dir}/")


if __name__ == "__main__":
    main()
