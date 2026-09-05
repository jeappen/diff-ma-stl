"""High-N scaling figure: Mixed spec, DIFF-MA vs STLPY-SA, N = 8..128.

Answers "what happens at N=64 / N=128?". One spec (the 5-component Mixed
spec), three panels -- Success / TtR / Plan Time -- with N on the x-axis. Shares its drawing
core with the CaTL team-spec figure, so the visual scheme (colors, error bars, log plan-time
panel, red 100% line) is identical by construction.

The panels are sourced SEPARATELY (scripts/high_n_plot_config.yaml `panels`):
  success  -- native area-6 / epi=10 runs at N<=32, goal-scale 2.0 / area-10 / epi=3 at N>=64
              (decongestion is only needed at N>=64, and forcing it onto N<=32 breaks the
              STLPY-SA arm), so every COLUMN is an internally controlled two-arm comparison.
  ttr/pt   -- the uniform gs2.0 runs at EVERY N, because plan_time_mean and TtR are not
              comparable across goal scales.
Both arms of a given panel always share the same setting within a column; the success panel's
x-axis changes setting once, between N=32 and N=64.

Data source (--source):
  wandb  (default) : pull the authoritative runs per scripts/high_n_plot_config.yaml
  csv              : re-render offline from a snapshot written by a previous wandb pull

All non-obvious choices (the env_gcbf_goal_scale-not-tag filter, epi=3, the absent task rate,
the plan-time definition) live in scripts/high_n_plot_config.yaml -- edit there, not here.

Output: {out_dir}/high_n_scaling_mixed.{pdf,png}
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd

from gcbfplus.utils.spec_bar_figure import (
    SCALING_METRICS, apply_column_style, fetch_summary_runs, label_planner, planner_colors,
    planner_sort_key, read_snapshot, render_metric_grid)

# Config columns kept per run: the setting fields the caption/provenance need.
_CONFIG_KEYS = ["spec", "num_agents", "epi", "planner", "diffusion_method", "area_size",
                "env_gcbf_goal_scale", "achievable_guidance", "stl_mixed_spec_mode",
                "num_candidates", "max_step", "spec_len"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["wandb", "csv"], default="wandb",
                        help="Pull runs from W&B (default) or re-render from a saved snapshot")
    parser.add_argument("--data-csv", default=None,
                        help="Snapshot path for --source csv (default: config 'snapshot')")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__),
                                                         "high_n_plot_config.yaml"))
    parser.add_argument("--out-dir", default="barplots")
    parser.add_argument("--table", action="store_true",
                        help="Also emit the Safe/Finish/Success LaTeX table that explains the "
                             "success bars ({out_dir}/{plot_name}_table.tex)")
    return parser.parse_args(argv)


def load_raw(cfg, cli):
    """Concatenate the per-N sources. Each source supplies BOTH arms for its N values, so a
    column is never a mixed-setting comparison (see the config's WHY block)."""
    if cli.source == "csv":
        return read_snapshot(cli.data_csv or cfg.get("snapshot"))
    wb = cfg["wandb"]
    frames = []
    for src in wb["sources"]:
        rows = fetch_summary_runs(wb["entity"], wb["project"], src["preload"], _CONFIG_KEYS)
        keep_n = set(src["num_agents"])
        rows = [d for d in rows if d.get("num_agents") in keep_n]
        for d in rows:
            d["_source"] = src["name"]
        print(f"[wandb] source {src['name']:12s} N={src['num_agents']}: {len(rows)} runs")
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _best_cells(df, sources):
    """Best-success run per (N, Planner), restricted to the named sources."""
    sub = df[df["_source"].isin(sources)]
    return sub.loc[sub.groupby(["N", "Planner"])["success"].idxmax()].copy() if len(sub) else sub


def prepare(df_raw, cfg):
    """One row per (N, Planner), with the SUCCESS panel and the TIMING panels sourced
    independently (see the config's PER-PANEL SOURCING block for why)."""
    df = df_raw.copy()
    df["N"] = df["num_agents"].astype(int)

    # The DIFF-MA arm must be the same variant across the whole x-axis. Without the method
    # filter, label_planner folds DIFF-SA (edm / edm-bo8) rows into DIFF-MA; the native area-6
    # pool also carries achievable_guidance=True runs the gs2.0 pool does not have.
    dc = cfg["diffma"]
    is_diff = df["planner"].astype(str).str.contains("diffusion")
    bad_method = is_diff & ~df["diffusion_method"].astype(str).str.contains(
        dc["require_method_contains"], na=False)
    bad_ach = is_diff & df["achievable_guidance"].astype(str).str.lower().isin(["true", "1", "1.0"]) \
        if dc.get("exclude_achievable_guidance") else pd.Series(False, index=df.index)
    dropped = int((bad_method | bad_ach).sum())
    df = df[~(bad_method | bad_ach)].copy()
    print(f"[diffma] kept {dc['require_method_contains']}, ach=False: dropped {dropped} diffusion rows")

    df["Planner"] = df.apply(label_planner, axis=1)
    # Two-arm figure: drop anything else the pool happens to contain (e.g. a ce_nl/Gradient run
    # at area-6, which has no N>=64 counterpart).
    keep = cfg["planners"]
    before = len(df)
    df = df[df["Planner"].isin(keep)].copy()
    if before != len(df):
        print(f"[planners] keep {keep}: dropped {before - len(df)} rows")
    df["Spec"] = cfg["spec_label"]
    # success_mean/_std arrive already x100 (fetch_summary_runs); TtR/plan_time are raw.
    df["success"] = pd.to_numeric(df["success_mean"], errors="coerce")
    df["success_s"] = pd.to_numeric(df.get("success_std"), errors="coerce").fillna(0)
    # Not plotted, but they EXPLAIN the plotted success: success = safe AND finish, per agent
    # (test.py:1008). safe_mean arrives x100 from the loader; eval/finish_rate is raw 0-1.
    df["safe"] = pd.to_numeric(df["safe_mean"], errors="coerce")
    df["finish"] = pd.to_numeric(df["finish_rate"], errors="coerce") * 100
    df["ttr"] = pd.to_numeric(df["TtR"], errors="coerce")
    df["ttr_s"] = pd.to_numeric(df.get("TtR_std"), errors="coerce").fillna(0)
    df["pt"] = pd.to_numeric(df["plan_time_mean"], errors="coerce")
    df["pt_s"] = pd.to_numeric(df.get("plan_time_std"), errors="coerce").fillna(0)

    assert cfg.get("select") == "best_success", f"unsupported select: {cfg.get('select')}"
    pan = cfg["panels"]

    # --- SUCCESS panel: prefer the native source where it covers an N, else gs2.0. ---
    # Both sources cover N<=32, so a cell can have two candidate rows; sorting on _rank puts
    # the preferred (native) row first and groupby(...).first() keeps it. N>=64 exists only in
    # the gs2.0 source, so those cells fall through to it unchanged.
    succ = _best_cells(df, pan["success"]["sources"])
    prefer = pan["success"]["prefer"]
    succ["_rank"] = (succ["_source"] != prefer).astype(int)   # 0 = preferred source
    succ = succ.sort_values("_rank").groupby(["N", "Planner"], as_index=False).first()

    # --- TIMING panels (ttr + plan time): one goal scale end to end. ---
    tim = _best_cells(df, pan["timing"]["sources"])[["N", "Planner", "ttr", "ttr_s", "pt", "pt_s",
                                                     "id", "_source"]]
    tim = tim.rename(columns={"id": "_timing_id", "_source": "_timing_source"})

    agg = succ.drop(columns=["ttr", "ttr_s", "pt", "pt_s"]).merge(tim, on=["N", "Planner"],
                                                                  how="left")

    # Fail loudly if the run population ever changes shape (a silently dropped cell would
    # otherwise just render as a missing bar).
    exp = cfg["expect"]
    assert len(agg) == exp["total_rows"], f"expected {exp['total_rows']} cells, got {len(agg)}"
    assert sorted(agg["N"].unique()) == exp["num_agents"], \
        f"expected N={exp['num_agents']}, got {sorted(agg['N'].unique())}"
    per_n = agg.groupby("N")["Planner"].nunique()
    assert (per_n == exp["planners_per_n"]).all(), f"expected 2 planners per N, got:\n{per_n}"
    assert agg[["ttr", "pt"]].notna().all().all(), "a timing cell is missing a gs2.0 run"

    # Provenance: each bar -> its run(s). The success and timing columns are DIFFERENT runs by
    # design, so both ids are printed.
    print("\nselected bars (success run | timing run):")
    for _, r in agg.sort_values(["N", "Planner"]).iterrows():
        print(f"  N={int(r['N']):<4d} {r['Planner']:10s} "
              f"success={r['success']:5.1f}% [{r['_source']:11s} epi={r.get('epi')} "
              f"gs={r.get('env_gcbf_goal_scale')} area={r.get('area_size')} {r.get('id', '?')}]"
              f"  |  TtR={r['ttr']:.0f} plan={r['pt']:6.2f}s [{r['_timing_source']:11s} "
              f"{r['_timing_id']}]")
    return agg


def emit_table(agg, cfg, out_dir):
    """Safety / Finish / Success per (N, planner) as LaTeX -- the decomposition that EXPLAINS
    the plotted success bars. Success is the per-agent conjunction of the other two
    (test.py:1008: success_matrix = (1 - is_unsafe) * is_finish), so the shortfall splits with
    no residual: (1 - success) = (finish - success) + (1 - finish).
    All three come from the SUCCESS-panel run of each cell, so the rows are self-consistent.
    """
    arms = cfg["planners"]
    rows = []
    for n in cfg["expect"]["num_agents"]:
        cells = {a: agg[(agg["N"] == n) & (agg["Planner"].str.startswith(a.split(" ")[0]))]
                 for a in arms}
        r = agg[agg["N"] == n].iloc[0]
        setting = "native" if r["_source"] == "native-a6" else "gs2.0"
        vals = []
        for a in arms:
            c = cells[a]
            vals += ["--"] * 3 if c.empty else [f"{c.iloc[0][k]:.1f}" for k in ("safe", "finish", "success")]
        rows.append(f"{n} & {setting} & " + " & ".join(vals) + r" \\")

    body = "\n".join(rows)
    latex = rf"""% Auto-generated by scripts/plot_high_n_scaling.py --table -- do not hand-edit.
% Success = Safe AND Finish (per agent), so these three columns explain the success bars.
\begin{{table}}[t]
\centering
\caption{{Safety, finish and success rate (\%) on the Mixed specification as the team scales.
Success is the per-agent conjunction of collision-freedom (Safe) and specification satisfaction
(Finish), so the success shortfall decomposes exactly into agents that collided and agents that
never satisfied the spec. At $N\ge64$ the shortfall is dominated by Safe: the GCBF+ tracking
controller is fixed and trained at $N=8$, while Finish degrades far more gently.
$N\le32$ use the native fixed-grid setting (area-6, epi=10); $N\ge64$ use goal-scale 2.0
(area-10, epi=3), where goal-spread decongestion is required.
STLPY-SA's higher Safe at $N=128$ is not an advantage --- its Finish is only 66.1\%, i.e. fewer
of its agents ever complete their tours.}}
\label{{tab:high-n-safe-finish}}
\begin{{tabular}}{{c|l|ccc|ccc}}
\toprule
& & \multicolumn{{3}}{{c|}}{{DIFF-MA (Ours)}} & \multicolumn{{3}}{{c}}{{STLPY-SA}} \\
$N$ & setting & Safe $\uparrow$ & Finish $\uparrow$ & Success $\uparrow$ & Safe $\uparrow$ & Finish $\uparrow$ & Success $\uparrow$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cfg['plot_name']}_table.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"[table] {path}")
    return latex


def render(agg, cfg, out_dir):
    """Draw {out_dir}/{plot_name}.pdf from a prepared `agg` (one row per N x planner).
    Shared with scripts/release/plot_reproduced.py so reproduced runs get the identical figure."""
    agg = agg.copy()
    # Legend wording only -- applied after the asserts/provenance print so those stay keyed to
    # the canonical "DIFF-MA" label. planner_colors/planner_sort_key treat both spellings alike.
    diffma_label = cfg.get("diffma_label")
    if diffma_label:
        agg.loc[agg["Planner"] == "DIFF-MA", "Planner"] = diffma_label

    planners = sorted(agg["Planner"].unique(), key=planner_sort_key)

    # One spec -> one row. extra_task=False: eval/task_rate is not logged for this spec.
    # A 1-row grid needs the legend lifted clear of the panel titles, and 5 N values per panel
    # need smaller x ticks than the team-spec figure's 3.
    render_metric_grid(agg, [cfg["spec_label"]], planners, planner_colors(planners),
                       SCALING_METRICS, f"{out_dir}/{cfg['plot_name']}.pdf",
                       figsize=(4.2, 1.35), bar_w_frac=0.7, extra_task=False,
                       legend_anchor=(0.5, 1.20), xtick_fontsize=5.5, n_label_x=-0.22)


def main(argv=None):
    cli = parse_args(argv)
    cfg = yaml.safe_load(open(cli.config))
    apply_column_style()

    df_raw = load_raw(cfg, cli)

    # On a wandb pull, drop an offline snapshot so the figure can be re-rendered with --source csv.
    if cli.source == "wandb":
        snap = cli.data_csv or cfg.get("snapshot")
        if snap:
            os.makedirs(os.path.dirname(snap) or ".", exist_ok=True)
            df_raw.to_csv(snap, index=False)
            print(f"[snapshot] wrote {snap} ({len(df_raw)} rows)")

    agg = prepare(df_raw, cfg)

    if cli.table:
        emit_table(agg, cfg, cli.out_dir)

    render(agg, cfg, cli.out_dir)


if __name__ == "__main__":
    main()
