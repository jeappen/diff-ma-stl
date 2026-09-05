"""Shared bar-grid figure core: rows = spec, cols = metric, x = N, hue = planner.

Import-safe (no argparse / no rcParam side effects at import). Used by:

  scripts/plot_team_spec_results.py  -- CaTL team-spec comparison (2 specs, task-rate box)
  scripts/plot_high_n_scaling.py     -- high-N scaling curve   (1 spec, no task rate)

Data contract: renderers take an ``agg`` frame with ONE row per (Spec, N, Planner) and
columns ``Spec``, ``N``, ``Planner`` plus, for each metric, its ``mean_col``/``std_col``
pair (and ``task_pct`` when a metric enables the dashed task box).

Metric tuples are positional::

    (title, mean_col, std_col, log_scale, ylim, is_percentage, show_task_rate_outline)
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from gcbfplus.utils.plot import PAPER_RC

# ── Style / labels ─────────────────────────────────────────────────────
def apply_column_style():
    """Inherit PAPER_RC, override sizes for a single-column paper figure."""
    col_rc = {**PAPER_RC,
              "font.size": 7, "axes.titlesize": 8, "axes.titleweight": "bold",
              "axes.labelsize": 7, "axes.labelweight": "semibold",
              "xtick.labelsize": 6.5, "ytick.labelsize": 6,
              "legend.fontsize": 6, "axes.grid": True, "grid.alpha": 0.25}
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(col_rc)
    sns.set_theme(style="whitegrid", context="paper", rc={})


def planner_colors(planners):
    """Stable colorblind palette keyed by display label (DIFF-MA blue, STLPY-SA red)."""
    cb = sns.color_palette("colorblind", 10)
    color_map = {"STLPY-SA": cb[3], "STLPY-Global": cb[2],
                 "DIFF-MA": cb[0], "DIFF-MA (Ours)": cb[0]}
    return {p: color_map.get(p, cb[1]) for p in planners}


def planner_sort_key(p):
    if "Global" in p: return (0, p)
    if "STLPY" in p: return (1, p)
    if "DIFF-MA" in p: return (2, p)
    return (3, p)


def label_planner(row):
    planner = str(row.get("planner", ""))
    if "stlpy_global" in planner:
        return "STLPY-Global"
    elif "stlpy" in planner:
        return "STLPY-SA"           # single-agent MILP; alloc schemes merged, best via dedup
    elif "diffusion" in planner:
        return "DIFF-MA"
    return planner.upper()


# ── W&B: targeted summary reader ───────────────────────────────────────
# We do NOT use WandbLoader here: it prunes any column with <5 non-NaN values and discards run
# tags, which corrupts small, tag-scoped pulls (the 3-run gs1.25 cell; the 10-run high-N sweep,
# where diffusion_method has exactly 5 non-NaN values). This reader keeps every field intact.
# summary key -> output column (eval/ prefix stripped, matching the local test_log.csv schema)
SUMMARY_KEYS = {"success_mean": "success_mean", "success_std": "success_std",
                "safe_mean": "safe_mean", "eval/task_rate": "task_rate",
                "eval/task_rate_std": "task_rate_std", "eval/TtR": "TtR",
                "eval/TtR_std": "TtR_std", "plan_time_mean": "plan_time_mean",
                "plan_time_std": "plan_time_std", "eval/finish_rate": "finish_rate"}


def fetch_summary_runs(entity, project, preload, config_keys, summary_keys=None,
                       exclude_tags=(), drop_tags=(), timeout=29):
    """Pull finished runs matching ``preload`` -> list of flat dicts (config + summary + id/url/tags).

    Reads run.summary aggregates only. NOTE: summary success_rate/safe_rate/finish_rate are
    LAST-EPISODE (or suppressed); the aggregates are success_mean / safe_mean / eval/*.

    ``exclude_tags`` and ``drop_tags`` behave identically -- a run carrying any of those tags
    is skipped -- and are two arguments only because they come from different places: the
    caller's config supplies ``exclude_tags`` once for globally bad runs (buggy / probe /
    duplicate), while ``drop_tags`` is passed per source to keep goal-scale variants out of
    the source that must not own them.
    """
    import wandb
    api = wandb.Api(timeout=timeout)
    summary_keys = summary_keys or SUMMARY_KEYS
    exclude, drop = set(exclude_tags), set(drop_tags)
    rows = []
    for r in api.runs(f"{entity}/{project}", filters=preload):
        if r.state != "finished":
            continue
        tags = set(r.tags)
        if tags & exclude or tags & drop:
            continue
        d = {k: r.config.get(k) for k in config_keys}
        for skey, col in summary_keys.items():
            d[col] = r.summary.get(skey)
        # Match the WandbLoader/test_log convention: success in %, task_rate/TtR/plan raw.
        for c in ("success_mean", "success_std", "safe_mean"):
            if c in d:
                d[c] = d[c] * 100 if d[c] is not None else d[c]
        d.update(id=r.id, url=r.url, tags=",".join(sorted(tags)))
        rows.append(d)
    return rows


def read_snapshot(path):
    """Offline snapshot read. keep_default_na=False keeps the literal string "None"
    (stl_mixed_spec_mode etc.) a real category rather than NaN."""
    print(f"[csv] reading snapshot {path}")
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


# ── Metric presets ─────────────────────────────────────────────────────
# (title, mean_col, std_col, log_scale, y_limits, is_percentage, show_task_rate_outline)
TEAM_METRICS = [
    ("Success / Task (%)", "success", "success_s", False, (0, 119), True, True),
    ("TtR (steps)",        "ttr",     "ttr_s",     False, None,     False, False),
    ("Plan Time (s)",      "pt",      "pt_s",      True,  None,     False, False),
]

# Same panels, minus the dashed task box: eval/task_rate is not logged for the non-team specs.
SCALING_METRICS = [
    ("Success (%)",  "success", "success_s", False, (0, 119), True, False),
    ("TtR (steps)",  "ttr",     "ttr_s",     False, None,     False, False),
    ("Plan Time (s)", "pt",     "pt_s",      True,  None,     False, False),
]


# ── Drawing ────────────────────────────────────────────────────────────
def draw_metric_ax(ax, planners, colors, n_p, spec_data_list, x_base, bar_w, mcol, scol,
                   log_sc, ylim, is_pct, show_task, is_first_legend_ax, n_list):
    na_x = []   # (x) of cells that HAVE a run but whose metric is undefined (e.g. TtR never reached)
    for p_i, planner in enumerate(planners):
        means, stds, tasks, missing = [], [], [], []
        for spec_name, spec_df in spec_data_list:
            for n in n_list:
                r = spec_df[(spec_df["N"] == n) & (spec_df["Planner"] == planner)]
                has_run = not r.empty
                val = r[mcol].iloc[0] if has_run and pd.notna(r[mcol].iloc[0]) else np.nan
                # A run whose metric is NaN = genuinely undefined -> draw NO bar (never a 0).
                means.append(val)
                stds.append(r[scol].iloc[0] if has_run and pd.notna(r[scol].iloc[0]) else np.nan)
                missing.append(has_run and np.isnan(val))
                if show_task:
                    tasks.append(r["task_pct"].iloc[0] if has_run and pd.notna(r["task_pct"].iloc[0]) else np.nan)
        x = x_base + (p_i - n_p / 2 + 0.5) * bar_w
        means, stds = np.array(means, float), np.array(stds, float)
        # matplotlib skips NaN heights/errorbars, so undefined cells leave a gap, not a 0-bar.
        if show_task and tasks:
            ax.bar(x, np.array(tasks, float), bar_w, facecolor="none", edgecolor=colors[planner],
                   linewidth=0.8, linestyle="--", zorder=3)
        # Solid bar = success rate (safe_rate x task_rate), nested inside the task box.
        ax.bar(x, means, bar_w, color=colors[planner], edgecolor="white", linewidth=0.3,
               alpha=0.85, label=planner if is_first_legend_ax else "", zorder=4)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black",
                    capsize=1.2, elinewidth=0.4, capthick=0.3, zorder=5)
        na_x.extend(xi for xi, m in zip(x, missing) if m)
    # Mark "n/a" for undefined-metric cells (only meaningful on TtR: spec ~never satisfied).
    if mcol == "ttr" and na_x:
        for xi in na_x:
            ax.annotate("n/a", (xi, 0.02), xycoords=ax.get_xaxis_transform(), rotation=90,
                        ha="center", va="bottom", fontsize=4.5, color="0.5", zorder=6)
    if log_sc:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    if is_pct:
        ax.axhline(100, color="red", ls="--", lw=0.6, alpha=0.6, zorder=1)
    if mcol == "ttr":
        top = ax.get_ylim()[1]
        # Ladder must keep labels covering the bars: high-N TtR reaches ~3.7k (N=128 DIFF-MA),
        # so step in 1k above 3.2k rather than stopping at a hardcoded 3k ceiling.
        ticks = [0, 500, 1000, 1500] if top <= 1600 else \
                [0, 1000, 2000] if top <= 2200 else \
                [0, 1000, 2000, 3000] if top <= 3200 else \
                list(range(0, int(top) + 1000, 1000))
        ax.set_yticks(ticks)
        ax.set_yticklabels(["0" if t == 0 else (f"{t // 1000}k" if t % 1000 == 0
                            else f"{t / 1000:.1f}k") for t in ticks])
    ax.tick_params(axis="y", pad=1); ax.tick_params(axis="x", pad=1)
    ax.grid(axis="y", alpha=0.3, linewidth=0.3)


def add_legend(fig, ref_ax, extra_task=True, bbox_to_anchor=(0.5, 1.04)):
    """bbox_to_anchor: a 1-row grid needs more headroom than the 2-row default, else the
    legend sits on top of the panel titles."""
    if isinstance(ref_ax, np.ndarray):
        h, l = ref_ax.flat[0].get_legend_handles_labels()
    else:
        h, l = ref_ax.get_legend_handles_labels()
    if extra_task:
        h.append(Patch(facecolor="none", edgecolor="gray", linewidth=0.8, linestyle="--"))
        l.append("Task Rate")
    fig.legend(h, l, loc="upper center", ncol=min(len(h), 5), frameon=True, fontsize=5.5,
               bbox_to_anchor=bbox_to_anchor, handlelength=1.0, handletextpad=0.2,
               columnspacing=0.6, borderpad=0.3)


def save_figure(out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path} (+ .png)")


def render_metric_grid(agg, specs, planners, colors, metrics, out_path,
                       figsize=(3.8, 2.5), bar_w_frac=0.7, ttr_ymax=None, extra_task=True,
                       legend_anchor=(0.5, 1.04), xtick_fontsize=None, n_label_x=-0.32):
    """specs x metrics grid: one row per spec, one column per metric, N on the x-axis.

    squeeze=False keeps ``axes`` 2-D so a SINGLE spec renders (one row) rather than raising
    IndexError on axes[row, col].

    legend_anchor / xtick_fontsize / n_label_x default to the 2-spec x 3-N team-spec figure;
    a 1-row grid with more N values needs a higher legend and smaller x ticks to avoid overlap.
    """
    n_vals = sorted(agg["N"].unique())
    n_p = len(planners)
    ttr_ymax = ttr_ymax or {}
    fig, axes = plt.subplots(len(specs), len(metrics), figsize=figsize, squeeze=False,
                             gridspec_kw={"hspace": 0.40, "wspace": 0.55})
    bar_w = bar_w_frac / n_p
    x_base = np.arange(len(n_vals))
    for row_i, spec in enumerate(specs):
        spec_list = [(spec, agg[agg["Spec"] == spec])]
        for col_j, (title, mcol, scol, log_sc, ylim, is_pct, show_task) in enumerate(metrics):
            ax = axes[row_i, col_j]
            # Optional per-spec fixed TtR-panel top (else auto).
            eff_ylim = (0, ttr_ymax[spec]) if mcol == "ttr" and ttr_ymax.get(spec) else ylim
            draw_metric_ax(ax, planners, colors, n_p, spec_list, x_base, bar_w, mcol, scol,
                           log_sc, eff_ylim, is_pct, show_task,
                           (row_i == 0 and col_j == 0), n_vals)
            ax.set_xticks(x_base)
            tick_kw = {"fontsize": xtick_fontsize} if xtick_fontsize else {}
            ax.set_xticklabels([str(n) for n in n_vals], **tick_kw)
            if row_i == 0:
                ax.set_title(title, pad=3)
            ax.set_ylabel(spec if col_j == 0 else "", fontsize=7, fontweight="bold", labelpad=2)
            ax.set_xlabel("")
            if is_pct:
                ax.set_yticks([0, 25, 50, 75, 100])
    axes[-1, 0].text(n_label_x, -0.06, "N", transform=axes[-1, 0].get_xaxis_transform(),
                     ha="right", va="top", fontsize=8, fontweight="bold")
    add_legend(fig, axes, extra_task=extra_task, bbox_to_anchor=legend_anchor)
    fig.subplots_adjust(left=0.14)
    save_figure(out_path)
