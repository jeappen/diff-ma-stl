"""
CaTL team-spec comparison figure (single-column, two-column-paper friendly).

Compares three planners on two team specifications:
  - redun4x + avoid  (AND spec) -> "Redundant (∧)"
  - choiceseq3       (DNF OR spec) -> "Choice (∨)"
Planners: STLPY-Global (joint MILP, N=8 only), STLPY-SA (per-agent MILP, best over
allocation schemes), DIFF-MA (diffusion + CaTL+ gradient).

Each Success/Task panel draws the task-satisfaction rate as a DASHED box with the success
rate nested inside; TtR and Plan Time are side panels.

Data source (--source):
  wandb  (default) : pull the authoritative runs from W&B per scripts/team_spec_plot_config.yaml
  csv              : re-render offline from a snapshot written by a previous wandb pull

All non-obvious choices (sources, gs=1.25 for choiceseq N=32, best-of scopes, excludes) live
in scripts/team_spec_plot_config.yaml -- edit there, not here.

STLPY-SA override: the STLPY-SA arm is not taken from W&B but from the committed CSV
plot_snapshots/stlpy_sa_old_override.csv (config key `stlpy_sa_override_csv`). Its rows are
local test_log.csv lines written by the pre-spec-v2 code; only rows with planner == "stlpy"
are used, and one row per (Spec, N) is kept -- best epi, then best success. The reason is the
TtR column: those runs log a complete, sensible TtR everywhere, while the newer W&B STLPY
runs under/over-log it. STLPY-Global and DIFF-MA still come from W&B.

Modes:
  --merge-edm         Best-of DIFF-MA merged, 2 rows (specs) x 3 cols (metrics)   [main figure]
  --merge-edm --1row  Same, squeezed into 1 row with both specs on the x-axis
  (default)           All planners separately (appendix/ablation)
  --show-global       Include the STLPY-Global bar (N=8 only)
  --diffma-scope      emergent | all | both  (default both -> one figure per scope)

Output: {out_dir}/team_spec_comparison_{layout}_{scope}.{pdf,png}
"""
import sys, os, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Drawing core + targeted W&B reader are shared with scripts/plot_high_n_scaling.py.
from gcbfplus.utils.spec_bar_figure import (
    TEAM_METRICS, add_legend, apply_column_style, draw_metric_ax, fetch_summary_runs,
    label_planner, planner_colors, planner_sort_key, read_snapshot, render_metric_grid,
    save_figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-edm", action="store_true",
                        help="Merge all DIFF-MA variants into best-of 'DIFF-MA (Ours)'")
    parser.add_argument("--1row", dest="one_row", action="store_true",
                        help="Force single-row layout (only with --merge-edm)")
    parser.add_argument("--show-global", action="store_true",
                        help="Include STLPY-Global as a separate planner (N=8 only)")
    parser.add_argument("--source", choices=["wandb", "csv"], default="wandb",
                        help="Pull runs from W&B (default) or re-render from a saved snapshot")
    parser.add_argument("--data-csv", default=None,
                        help="Snapshot path for --source csv (default: config 'snapshot')")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__),
                                                         "team_spec_plot_config.yaml"))
    parser.add_argument("--diffma-scope", choices=["emergent", "all", "both"], default="both",
                        help="DIFF-MA best-of scope: emergent (paper method), all schemes, or both")
    parser.add_argument("--out-dir", default="barplots")
    return parser.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════
#  Data loading  (W&B via config, or offline snapshot)
# ═══════════════════════════════════════════════════════════════════════
# Config values are read either from a real dataframe column (W&B pull) or from the packed
# ``comments`` string (local test_log.csv). One accessor handles both -> no branchy duplication.
def _cfg(row, key, default=None):
    if key in row and pd.notna(row[key]) and str(row[key]) != "":
        return row[key]
    m = re.search(rf"(?:^|;){re.escape(key)}:([^;]+)", str(row.get("comments", "")))
    return m.group(1) if m else default


def _is_true(v):
    return str(v).strip().lower() in ("true", "1", "1.0")


def _as_float(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# Exactly the config columns this figure needs, read straight off each run. (The summary keys
# and the reason we avoid WandbLoader live in gcbfplus/utils/spec_bar_figure.py.)
_CONFIG_KEYS = ["spec", "num_agents", "epi", "planner", "team_alloc", "team_disjunctive",
                "team_avoid", "global_stl_only", "ma_stl_loss_coeff", "achievable_loss_coeff",
                "achievable_guidance", "env_gcbf_goal_scale", "area_size"]


def build_wandb_df(cfg):
    wb = cfg["wandb"]
    exclude = set(wb.get("exclude_tags", []))
    rows = []
    for src in wb["sources"]:
        gs = float(src["gs"])
        # gs1.5 runs are decongestion probes that are never plotted, so every source drops them.
        # A gs1.0 source additionally drops gs1.25-tagged runs, so the choiceseq N=32 cell is
        # supplied only by the dedicated gs1.25 source rather than by both.
        gs_drop = {"gs1.25", "gs1.5"} if gs == 1.0 else {"gs1.5"}
        src_rows = fetch_summary_runs(wb["entity"], wb["project"], src["preload"], _CONFIG_KEYS,
                                      exclude_tags=exclude, drop_tags=gs_drop)
        for d in src_rows:
            d.update(gs=gs, _source=src["name"])
        rows.extend(src_rows)
        print(f"[wandb] source {src['name']:16s}: {len(src_rows)} runs")
    df = pd.DataFrame(rows)
    # Safety dedup: a run pulled by >1 source keeps its highest-gs stamp (gs1.25 wins).
    if "id" in df.columns:
        df = df.sort_values("gs", ascending=False).drop_duplicates("id", keep="first")
    return df


def load_raw(cfg, cli):
    if cli.source == "wandb":
        return build_wandb_df(cfg)
    return read_snapshot(cli.data_csv or cfg.get("snapshot"))


# ═══════════════════════════════════════════════════════════════════════
#  Selection: spec labels, DIFF-MA best-of scope, goal-scale, best-per-cell
# ═══════════════════════════════════════════════════════════════════════
def spec_display(spec, cfg):
    for m in cfg["spec_display"]:
        if m["contains"] in str(spec):
            return m["label"]
    return str(spec)


def keep_diffma(row, scope, cfg):
    """Keep-mask for the DIFF-MA best-of pool. Non-diffusion rows always pass; diffusion rows
    are filtered by the scope's requirements from the config."""
    planner = str(row.get("planner", ""))
    if "diffusion" not in planner:
        return True
    spec = str(row.get("spec", ""))
    fam = "choiceseq" if "choiceseq" in spec else "redun" if "redun" in spec else None
    if fam is None:
        return True
    dc = cfg["diffma"]
    # global excludes (both scopes)
    if dc["exclude"].get("achievable_guidance") and _is_true(_cfg(row, "achievable_guidance")):
        return False
    if fam == "choiceseq" and _as_float(_cfg(row, "achievable_loss_coeff")) == \
            float(dc["exclude"].get("choiceseq_achievable_loss_coeff", -1)):
        return False
    req = cfg["diffma"]["emergent_requires" if scope == "emergent" else "all_requires"].get(fam, {})
    for key, want in req.items():
        if key == "ma_stl_loss_coeff_min":
            if _as_float(_cfg(row, "ma_stl_loss_coeff"), 0) < float(want):
                return False
        elif want is True and not _is_true(_cfg(row, key)):
            return False
    return True


def select_goal_scale(df, cfg):
    """Keep only rows whose stamped gs matches each cell's target goal-scale."""
    gcfg = cfg["goal_scale"]
    default = float(gcfg.get("default", 1.0))

    def target(row):
        for ov in gcfg.get("overrides", []):
            if ov["spec_contains"] in str(row["spec"]) and int(ov["num_agents"]) == int(row["N"]):
                return float(ov["gs"])
        return default

    df = df.copy()
    tgt = df.apply(target, axis=1)
    keep = np.isclose(df["gs"].astype(float), tgt.astype(float))
    if (~keep).sum():
        print(f"[gs] dropped {int((~keep).sum())} rows off target goal-scale "
              f"(choiceseq N=32 -> {gcfg['overrides']}, else {default})")
    return df[keep]


def pick_best_epi(group):
    """Best single row for one (Spec, N, Planner): highest epi, then highest success."""
    max_epi = group["epi"].astype(float).max()
    best_epi = group[group["epi"].astype(float) == max_epi]
    return best_epi.loc[best_epi["success_mean"].astype(float).idxmax()]


def stlpy_sa_override(best, cfg):
    """Replace the STLPY-SA arm (all metrics, per cell) with the old plotting code's local
    test_log.csv, which has a complete TtR column. STLPY-Global and DIFF-MA are untouched."""
    path = cfg.get("stlpy_sa_override_csv")
    if not path:
        return best
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    old = pd.read_csv(path, keep_default_na=False, na_values=[""])
    old = old[old["spec"].astype(str).apply(
        lambda s: ("redun" in s and "redunseq" not in s) or "choiceseq3" in s)]
    old = old[old["planner"] == "stlpy"].copy()          # STLPY-SA only (not stlpy_global)
    old["Planner"] = "STLPY-SA"
    old["N"] = old["num_agents"].astype(int)
    old["Spec"] = old["spec"].apply(lambda s: spec_display(s, cfg))
    old["success"] = pd.to_numeric(old["success_mean"], errors="coerce")
    old["success_s"] = pd.to_numeric(old.get("success_std"), errors="coerce").fillna(0)
    old["task_pct"] = pd.to_numeric(old["task_rate"], errors="coerce") * 100
    old["ttr"] = pd.to_numeric(old["TtR"], errors="coerce")
    old["ttr_s"] = pd.to_numeric(old.get("TtR_std"), errors="coerce").fillna(0)
    old["pt"] = pd.to_numeric(old["plan_time_mean"], errors="coerce")
    old["pt_s"] = pd.to_numeric(old.get("plan_time_std"), errors="coerce").fillna(0)
    old["gs"] = 1.0                                      # exempt from the goal-scale cell rule
    old_best = pd.DataFrame([pick_best_epi(g) for _, g in old.groupby(["Spec", "N"])])
    out = pd.concat([best[best["Planner"] != "STLPY-SA"], old_best], ignore_index=True)
    print(f"[override] STLPY-SA <- old CSV {path}")
    for _, r in old_best.sort_values(["Spec", "N"]).iterrows():
        print(f"   {r['Spec']:16s} N={int(r['N']):<3d} STLPY-SA success={r['success']:.1f}% "
              f"task={r['task_pct']:.0f}% TtR={r['ttr']:.0f} epi={r.get('epi')}")
    return out


def prepare(df_raw, scope, cfg):
    """Full data prep for one DIFF-MA scope -> one row per (Spec, N, Planner)."""
    # Restrict to the two target specs (robust to the _t15 suffix / regex forms).
    target = df_raw["spec"].astype(str).apply(
        lambda s: ("redun" in s and "redunseq" not in s) or "choiceseq3" in s)
    df = df_raw[target].copy()
    assert not df.empty, f"No team rows. specs={sorted(df_raw['spec'].astype(str).unique())}"

    df["Planner"] = df.apply(label_planner, axis=1)
    df["N"] = df["num_agents"].astype(int)
    df["Spec"] = df["spec"].apply(lambda s: spec_display(s, cfg))
    if "gs" not in df.columns:            # offline snapshot from a plain test_log.csv
        df["gs"] = 1.0

    # DIFF-MA best-of pool (scope) + goal-scale cell selection.
    df = df[df.apply(lambda r: keep_diffma(r, scope, cfg), axis=1)].copy()
    df = select_goal_scale(df, cfg)

    # STLPY-Global only where it has data (N=8).
    gmax = cfg["planners"]["stlpy_global_max_n"]
    df = df[~((df["Planner"] == "STLPY-Global") & (df["N"] > gmax))].copy()

    # Numeric columns (W&B rescales *_mean/_std for success/safe by 100; task_rate is 0-1).
    df["success"] = pd.to_numeric(df["success_mean"], errors="coerce")
    df["success_s"] = pd.to_numeric(df.get("success_std"), errors="coerce").fillna(0)
    df["task_pct"] = pd.to_numeric(df["task_rate"], errors="coerce") * 100
    df["ttr"] = pd.to_numeric(df["TtR"], errors="coerce")
    df["ttr_s"] = pd.to_numeric(df.get("TtR_std"), errors="coerce").fillna(0)
    df["pt"] = pd.to_numeric(df["plan_time_mean"], errors="coerce")
    df["pt_s"] = pd.to_numeric(df.get("plan_time_std"), errors="coerce").fillna(0)

    rows_best = [pick_best_epi(g) for _, g in df.groupby(["Spec", "N", "Planner"])]
    best = pd.DataFrame(rows_best)

    # NOTE: logged TtR values are genuine reach times (finish_rate>0, TtR << max_step=7200),
    # NOT capped placeholders -- even when task_success_rate==0 (a strict full-spec DNF metric
    # that stays ~0 for the Choice (OR) spec while agents still reach their goals). So TtR is
    # shown wherever a run logged it; a cell is n/a only when eval/TtR was never written
    # (older code under-logged it for a few STLPY runs).
    best = stlpy_sa_override(best, cfg)

    # Provenance: each plotted bar -> one W&B run (id/url) or CSV row.
    print(f"\n[scope={scope}] selected bars (spec, N, planner -> best run):")
    for _, r in best.sort_values(["Spec", "N", "Planner"]).iterrows():
        prov = str(r.get("id", "old-csv"))   # W&B run id, or the STLPY-SA override CSV
        print(f"  {r['Spec']:16s} N={int(r['N']):<3d} {r['Planner']:13s} epi={r.get('epi')} "
              f"gs={r.get('gs')} success={r['success']:.1f}% task={r['task_pct']:.0f}% "
              f"plan={r['pt']:.1f}s  <{prov}>")
    return best


def render(agg, scope, cfg, cli):
    specs = [m["label"] for m in cfg["spec_display"]]           # Choice (∨) first, Redundant (∧)
    specs = [s for s in specs if s in set(agg["Spec"])]
    n_vals = sorted(agg["N"].unique())
    # Optional per-spec fixed TtR-panel top (else auto).
    ttr_ymax = {m["label"]: m.get("ttr_ymax") for m in cfg["spec_display"]}

    planners = sorted(agg["Planner"].unique(), key=planner_sort_key)
    if not cli.show_global:
        planners = [p for p in planners if "Global" not in p]
    n_p = len(planners)
    colors = planner_colors(planners)

    if cli.merge_edm:
        agg = agg.copy()
        agg.loc[agg["Planner"] == "DIFF-MA", "Planner"] = "DIFF-MA (Ours)"
        planners = ["DIFF-MA (Ours)" if p == "DIFF-MA" else p for p in planners]
        colors = planner_colors(planners)

    def out_for(suffix):
        return f"{cli.out_dir}/team_spec_comparison_{suffix}_{scope}.pdf"

    # ── Layout A: --merge-edm --1row (team-spec specific: both specs on one x-axis) ──
    if cli.merge_edm and cli.one_row:
        fig, axes = plt.subplots(1, len(TEAM_METRICS), figsize=(3.5, 1.55),
                                 gridspec_kw={"wspace": 0.50})
        n_per = len(n_vals); gap = 0.8
        x_left = np.arange(n_per); x_right = np.arange(n_per) + n_per + gap
        x_base = np.concatenate([x_left, x_right]); bar_w = 0.7 / n_p
        for col_j, (title, mcol, scol, log_sc, ylim, is_pct, show_task) in enumerate(TEAM_METRICS):
            ax = axes[col_j]
            spec_list = [(s, agg[agg["Spec"] == s]) for s in specs]
            draw_metric_ax(ax, planners, colors, n_p, spec_list, x_base, bar_w, mcol, scol,
                           log_sc, ylim, is_pct, show_task, col_j == 0, n_vals)
            ax.set_xticks(x_base)
            ax.set_xticklabels([str(n) for n in n_vals] * len(specs), fontsize=6)
            ax.set_title(title, pad=3)
            ax.axvline((x_left[-1] + x_right[0]) / 2, color="grey", ls="-", lw=0.6, alpha=0.4)
            for xl, spec in zip([x_left, x_right], specs):
                ax.text(xl.mean(), -0.25, spec, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=6, fontweight="semibold")
            ax.set_xlabel("")
        axes[0].text(-0.02, -0.18, "N", transform=axes[0].get_xaxis_transform(),
                     ha="right", va="top", fontsize=6.5, fontweight="semibold")
        add_legend(fig, axes)
        fig.subplots_adjust(left=0.14)
        save_figure(out_for("merged_1row"))

    # ── Layout B: --merge-edm (main paper layout) / C: default (all planners) ──
    # Identical grids; they differ only in figure height and bar width.
    elif cli.merge_edm:
        render_metric_grid(agg, specs, planners, colors, TEAM_METRICS, out_for("merged"),
                           figsize=(3.8, 2.5), bar_w_frac=0.7, ttr_ymax=ttr_ymax)
    else:
        render_metric_grid(agg, specs, planners, colors, TEAM_METRICS, out_for("full"),
                           figsize=(3.8, 2.6), bar_w_frac=0.65, ttr_ymax=ttr_ymax)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
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

    scopes = ["emergent", "all"] if cli.diffma_scope == "both" else [cli.diffma_scope]
    for scope in scopes:
        agg = prepare(df_raw, scope, cfg)
        render(agg, scope, cfg, cli)


if __name__ == "__main__":
    main()
