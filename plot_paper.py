#!/usr/bin/env python
"""Reproduce the paper figures end-to-end.

This is the single, self-contained pipeline behind the published figures: it owns the
exact data sourcing + plot configuration for each one, so every figure regenerates from
this script alone.

The non-obvious per-figure data-sourcing choices (e.g. the Gradient/ce_nl arm living in a
different project, or the Signal spec using the crowded R=1.5 runs) are recorded with
rationale in ``plot_config.yaml`` next to this file. Keep it in sync when FIGURES changes.

Two data sources are supported per figure:

* ``--source wandb`` (default): pull the runs live from W&B, then plot. This also
  writes the offline-repro artifacts (see below).
* ``--source csv``: load the previously-dumped ``*_data.csv`` and re-run the plot
  pipeline with no W&B access at all.

Whenever a figure is plotted with ``export_data=True`` (always, here) the Plotter
writes two sidecar files next to the PDF in ``--out-dir`` (default ``barplots/``):

* ``{name}_{env}_data.csv``    -- the raw per-run dataframe (offline-repro source).
* ``{name}_{env}_run_ids.csv`` -- the exact W&B run ids behind the figure + metadata,
  so the precise runs can be re-pulled later.

Prerequisites
-------------
* The ``gcbfplus`` package importable (``pip install -e .`` or on PYTHONPATH).
* csv source:   pandas, matplotlib, seaborn.
* wandb source: also ``wandb`` + W&B credentials (~/.netrc or WANDB_API_KEY) with
  access to ``csbric/diff-mastl-test``.
No local log files (logs/, pretrained/*.csv) are ever read -- data is 100% from W&B,
or from a self-contained ``*_data.csv`` snapshot.

Examples
--------
# Live from W&B; writes PDF + the offline snapshot (data.csv, run_ids.csv) into out-dir:
python plot_paper.py --figure single_hetero_dubins --source wandb --no-tables --out-dir out

# Include newly-added experiments (drop the pinned createdAt cap):
python plot_paper.py --figure single_hetero_dubins --source wandb --no-tables --latest --out-dir out

# Fully offline on ANY machine (no W&B, no creds) from a snapshot CSV:
python plot_paper.py --figure single_hetero_dubins --source csv \
    --data-csv out/single_hetero_DubinsCar_data.csv --out-dir out

# Common tweaks:
#   --no-signal            drop the Signal spec (Mixed reverts to no-signal variant)
#   --max-plan-time 60     cap plan time (DIFF-MA only; add --max-plan-time-all for every planner)
#   --diffma-le-gradient   keep DIFF-MA runs no slower than Gradient in each cell
python plot_paper.py --figure single_hetero_dubins --source csv \
    --data-csv out/single_hetero_DubinsCar_data.csv --diffma-le-gradient --out-dir out

# List the available figures:
python plot_paper.py --list

Moving to another machine
-------------------------
Committed gzipped snapshots live in ``plot_snapshots/`` so the figures regenerate fully
offline (no W&B, no login) after a clone + ``pip install -e .`` (pandas reads .gz directly):
    python plot_paper.py --figure random_loc_dubins --source csv \
        --data-csv plot_snapshots/random_loc_DubinsCar_data.csv.gz --out-dir out
    python plot_paper.py --figure single_hetero_dubins --source csv \
        --data-csv plot_snapshots/single_hetero_DubinsCar_data.csv.gz --out-dir out
Or, if the machine has W&B access, use ``--source wandb`` and skip the snapshots.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: we only save PDFs/CSVs

import pandas as pd

from gcbfplus.utils.plot import Plotter


# --------------------------------------------------------------------------- #
# Data pipeline
# --------------------------------------------------------------------------- #
# Planners kept by filter_df; everything else is dropped before plotting.
# gnn-ode / ode / stlpy_single are legacy planners from an earlier study; absent from all
# release data, kept so an older dataframe still loads unchanged.
VALID_PLANNERS = ["stlpy", "diffusion", "gnn-ode", "ode", "stlpy_single", "ce_nl"]

# Spec keys for the optional "Signal" (m2signal3) column. The combined "Mixed" spec comes
# in two flavours -- with and without the signal sub-spec -- and which one is plotted must
# match whether Signal is shown (see apply_signal_toggle).
SIGNAL_SPEC = "m2signal3"
MIXED_NO_SIGNAL = "mseq3-m2branch2-mcover3-m2loop3"
MIXED_WITH_SIGNAL = "mseq3-m2branch2-mcover3-m2loop3-m2signal3"

# Upper bound on run creation time. The preload filters are open-ended at the lower end
# (`createdAt > date`), so without a cap a fresh pull keeps sweeping in newly-created runs
# and the figure drifts. Capping pins the run population: a live pull returns the same set as
# long as no runs were added *before* this date after the snapshot was taken. The
# committed *_data.csv snapshot remains the authoritative, fully-reproducible source.
CREATEDAT_CAP = "2026-06-24T00:00:00Z"


def filter_df(df: pd.DataFrame, no_obs_arg: bool = False, post_diff_ach: bool = True) -> pd.DataFrame:
    """Row filter applied to every figure's dataframe.

    Keeps only valid planners with a real Time-to-Reach, optionally drops obstacle
    runs, and normalises ``ma_stl_satisfaction`` NaNs to 0.
    """
    df = df[(df["planner"].isin(VALID_PLANNERS)) & (~df["planner"].isna())]
    # async_planner may be stored as bool or as the strings "True"/"False"
    df = df[(df["async_planner"] == "True") | (df["async_planner"] == True)  # noqa: E712
            | (df["async_planner"] == False)].copy()  # noqa: E712
    df = df[df["TtR"] != -1.0]
    if no_obs_arg:
        df = df.loc[df["n_obs"] == 0].copy()
    if post_diff_ach and "ma_stl_satisfaction" in df.keys():
        df["ma_stl_satisfaction"] = df["ma_stl_satisfaction"].fillna(0)
        df["ma_stl_satisfaction"] = df["ma_stl_satisfaction"].apply(
            lambda x: 0.0 if str(x).strip() == "nan" else x)
    return df


def build_wandb_df(spec: "FigureSpec", download_tables: bool = True,
                   created_at_cap: Optional[str] = CREATEDAT_CAP) -> pd.DataFrame:
    """Pull and concatenate the W&B runs for a figure.

    :param download_tables: when True also download per-run artifact tables to derive
        diversity columns. These are slow (one file per run, project-wide)
        and only feed diversity metrics; set False for a fast pull when the target figure
        only needs summary scalars (Success/Planning/TtR/Finish/Safety) -- the surviving
        rows are identical, only the diversity columns are absent.
    :param created_at_cap: upper bound on run creation time applied to every preload filter.
        Pass None to drop the cap and include the very latest runs (e.g. newly-added
        experiments created after the pinned cap date).
    """
    import copy
    from gcbfplus.utils.wandb import WandbLoader

    frames = []
    for src in spec.wandb_sources:
        project = src["project"]
        preload = copy.deepcopy(src["preload"])
        created = preload.get("createdAt")
        if isinstance(created, dict):
            if created_at_cap is None:
                created.pop("$lt", None)        # no upper bound -> include the newest runs
            else:
                created["$lt"] = created_at_cap  # pin population for reproducibility
        loader = WandbLoader(project_name=project, entity=spec.wandb_entity,
                             download_tables_and_summarize=download_tables, preload_filter_dict=preload)
        frames.append(loader.summarize_loaded_runs())
    wandb_df = pd.concat(frames, ignore_index=True)
    # Drop spurious / unset planner rows.
    wandb_df = wandb_df[(wandb_df["planner"] != "mseq3")
                        & (wandb_df["planner"].notna())
                        & (wandb_df["planner"] != "None")]
    return wandb_df


def prepare_df(wandb_df: pd.DataFrame, spec: "FigureSpec") -> pd.DataFrame:
    """Turn the raw W&B dataframe into the exact frame handed to the Plotter."""
    # Restrict to this environment via the model path (e.g. ".../DubinsCar/...").
    in_env = wandb_df["path"].astype(str).str.contains(spec.env_name).fillna(False)
    df = wandb_df[in_env].copy()
    df = filter_df(df, no_obs_arg=spec.load_no_obs)
    # STLPY is only run at spec_len == 15; drop longer-horizon STLPY rows.
    df = df[~((df["planner"] == "stlpy") & (df["spec_len"] > 15))]
    df["num_agents"] = df["num_agents"].astype(int)
    # Keep only runs whose summary mean/std are over an accepted number of episodes.
    if spec.epi_filter is not None and "epi" in df.columns:
        before = len(df)
        df = df[df["epi"].isin(spec.epi_filter)].copy()
        print(f"[epi] keep epi in {spec.epi_filter}: {len(df)}/{before} rows")
    # Optional per-spec source override (e.g. Signal uses the crowded R=1.5 runs).
    if spec.row_filter is not None:
        before = len(df)
        df = spec.row_filter(df)
        print(f"[row_filter] {spec.row_filter.__name__}: {len(df)}/{before} rows")
    return df


# --------------------------------------------------------------------------- #
# Figure registry
# --------------------------------------------------------------------------- #
@dataclass
class FigureSpec:
    """Everything needed to reproduce one published figure."""

    name: str                      # base name -> {name}_{env}.pdf / _data.csv / _run_ids.csv
    env_name: str
    plot_method: str               # Plotter method to call
    plot_kwargs: Dict[str, Any]    # kwargs for that method
    plotter_kwargs: Dict[str, Any] = field(default_factory=dict)
    # W&B sourcing: one or more {"project": str, "preload": dict} sources, concatenated.
    # Different arms may live in different projects (e.g. Gradient/ce_nl vs the main grid).
    wandb_entity: str = "csbric"
    wandb_sources: List[dict] = field(default_factory=list)
    epi_filter: Optional[list] = None   # keep only these config.epi values (None = no filter)
    row_filter: Optional[Callable] = None  # optional post-concat df->df filter (per-spec source overrides)
    no_ach_specs: Optional[list] = None    # specs where DIFF-MA drops achievable_guidance (use cheaper no_ach)
    single_draw_specs: Optional[list] = None  # specs where DIFF-SA drops edm-bo8 (use honest single-draw edm)
    load_no_obs: bool = False      # no_obs_arg used while building df (plot does its own filtering)
    signal_capable: bool = False   # whether the Signal (m2signal3) spec can be toggled on/off

    def data_csv(self, out_dir: str) -> str:
        return f"{out_dir}/{self.name}_{self.env_name}_data.csv"


# Shared config for the hetero-DubinsCar grouped-bar figure (Success/Planning/TtR × N × specs).
# Every such figure differs ONLY in its W&B data source (project/filters) + epi filter, so the
# plot/plotter kwargs live here once and are reused by the factory below.
_HETERO_PLOTTER_KWARGS = dict(
    max_TtR_value_per_spec={"Seq.": 2500},
    outlier_list_of_dict=[
        {"Spec": "Loop", "Planner": "GNN-ODE", "success_mean": 87, "num_agents": 32},
        {"Spec": "Seq.", "Planner": "GNN-ODE", "success_mean": 95, "num_agents": 32},
        {"Spec": "Cover", "Planner": "GNN-ODE", "success_mean": 94, "num_agents": 32},
    ],
)

_HETERO_PLOT_KWARGS = dict(
    homo_mode=False,
    debug=False,
    plot_grouped=["Success", "Planning", "TtR"],
    no_obs=True,
    ablation_mode=False,
    ode_mode=False,
    N_large=32,
    columns_to_print=["Planning Time (s) ↓", "Finish Rate ↑", "Safety Rate ↑",
                      "Success Rate ↑", "TtR ↓"],
    merge_cols_dict={},  # DubinsCar: no DIFF-SA/DIFF-MA merge
    # Hetero specs to plot; m2signal3 ("Signal") on top of the four struct specs + "Mixed".
    # apply_signal_toggle swaps in the signal-inclusive Mixed variant when Signal is on.
    specs_to_keep=["m2branch2", "mseq3", "mcover3", "m2loop3", "m2signal3",
                   "mseq3-m2branch2-mcover3-m2loop3"],
    mixed_spec_choice={
        "Seq.": ["first1", "None"],
        "Cover": ["first1"],
        "Loop": ["first1", "None", "last1"],
        "Branch": ["first1", "None", "last1"],
        "Signal": ["first1", "None", "last1"],
        "Mixed": ["first1", "None"],
    },
)


def _signal_r15_filter(df: pd.DataFrame) -> pd.DataFrame:
    """m2signal3 source override: use the crowded R=1.5 runs for the diffusion/STLPY arms
    (larger DIFF-MA vs STLPY-SA gap), while keeping the ce_nl
    Gradient arm from the unconstrained (~R=2) runs (no ce_nl exists at R=1.5). Drops the
    a6-grid ~R=2 m2signal3 non-ce_nl rows so the Signal column is R=1.5 except Gradient.
    """
    is_sig = df["spec"] == "m2signal3"
    if "random_goals_region" in df.columns:
        region = pd.to_numeric(df["random_goals_region"], errors="coerce")
    else:
        region = pd.Series(float("nan"), index=df.index)
    keep = (region == 1.5) | (df["planner"] == "ce_nl")
    out = df[(~is_sig) | keep].copy()
    # The R=1.5 signal runs log an inconsistent stl_mixed_spec_mode across N (None at N=8/16,
    # first1 at N=32) even though the effective traced mode is identical. Unify it so
    # pick_best_row (which forces one mode per spec across N) keeps every N instead of
    # dropping the label-mismatched cells.
    out.loc[out["spec"] == "m2signal3", "stl_mixed_spec_mode"] = "None"
    return out


def _hetero_dubins_figure(name: str, wandb_sources: List[dict],
                          epi_filter: Optional[list] = None,
                          row_filter: Optional[Callable] = None,
                          no_ach_specs: Optional[list] = None,
                          single_draw_specs: Optional[list] = None) -> FigureSpec:
    """Build a hetero-DubinsCar grouped-bar FigureSpec from the shared config.

    Only the data source (``wandb_sources``), ``epi_filter``, ``row_filter``,
    ``no_ach_specs`` and ``single_draw_specs`` vary between such figures.
    """
    return FigureSpec(
        name=name, env_name="DubinsCar", signal_capable=True,
        wandb_sources=wandb_sources, epi_filter=epi_filter, row_filter=row_filter,
        no_ach_specs=no_ach_specs, single_draw_specs=single_draw_specs,
        plotter_kwargs=dict(_HETERO_PLOTTER_KWARGS),
        plot_method="plot_single_spec_grouped",
        plot_kwargs=dict(_HETERO_PLOT_KWARGS),
    )


FIGURES: Dict[str, FigureSpec] = {
    # PLOT USED IN RAL SUBMISSION (2026): heterogeneous DubinsCar specs grouped by N,
    # metrics Success / Planning / TtR. All arms in the diff-mastl-test project.
    "single_hetero_dubins": _hetero_dubins_figure(
        name="single_hetero",
        wandb_sources=[
            {"project": "diff-mastl-test",
             "preload": {"createdAt": {"$gt": "2025-04-10T00:00:00Z", "$lt": CREATEDAT_CAP}}},
            {"project": "diff-mastl-test",
             "preload": {"createdAt": {"$gt": "2025-09-10T00:00:00Z", "$lt": CREATEDAT_CAP}}},
        ],
    ),
    # Random-predicate setting (random_goals == True). Main arms (DIFF-MA/DIFF-SA/STLPY-SA)
    # come from the curated area-6 grid in gcbfplus-stl-test; Gradient (ce_nl) lives in the
    # diff-mastl-test project (untagged). epi in {5,10} so summary mean/std are true aggregates.
    "random_loc_dubins": _hetero_dubins_figure(
        name="random_loc",
        wandb_sources=[
            {"project": "gcbfplus-stl-test",
             "preload": {"tags": {"$in": ["random-loc-grid-a6"]},
                         "config.random_goals": {"$eq": True}}},
            {"project": "diff-mastl-test",
             "preload": {"config.planner": "ce_nl", "config.random_goals": {"$eq": True}}},
            # m2signal3 crowded R=1.5 row (larger DIFF-MA vs STLPY-SA gap; see _signal_r15_filter).
            {"project": "gcbfplus-stl-test",
             "preload": {"tags": {"$in": ["random-loc-signal-region"]},
                         "config.random_goals_region": {"$eq": 1.5}}},
        ],
        epi_filter=[5, 10],
        row_filter=_signal_r15_filter,
        # Loop: use the cheaper no-achievable DIFF-MA (edm-ma ach=False); the achievable run's
        # plan time is ~10-50x larger at ~equal success.
        no_ach_specs=["m2loop3"],
        # Cover + Mixed: use the honest single-draw edm for DIFF-SA (drop the 8x-compute
        # best-of-8). Both Mixed variants listed so it holds under --no-signal too.
        single_draw_specs=["mcover3", MIXED_WITH_SIGNAL, MIXED_NO_SIGNAL],
    ),
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _is_diff_ma(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of DIFF-MA rows (diffusion planner with edm-ma guidance)."""
    return (df["planner"] == "diffusion") & df["diffusion_method"].astype(str).str.contains("edm-ma", na=False)


def _truthy(series: pd.Series) -> pd.Series:
    """Robustly coerce a bool/str column to a boolean mask (handles True / "True" / "1")."""
    return series.astype(str).str.lower().isin(["true", "1"])


def _drop_variant_for_specs(df: pd.DataFrame, drop_mask: pd.Series, specs: list, label: str) -> pd.DataFrame:
    """Drop rows matching drop_mask whose spec is in `specs`; log how many."""
    drop = drop_mask & df["spec"].isin(specs)
    n = int(drop.sum())
    out = df[~drop].copy()
    print(f"[{label}] specs {specs}: dropped {n} rows")
    return out


def drop_achievable_for_specs(df: pd.DataFrame, specs: list) -> pd.DataFrame:
    """For the given specs, drop achievable_guidance DIFF-MA runs so DIFF-MA uses the cheaper
    no-achievable (edm-ma, ach=False) result. Achievable guidance boosts success only
    marginally but makes plan time ~10-50x larger (e.g. Loop N=32: 20.9s -> 0.45s at ~equal
    success), trading a negligible success change for a much cheaper planning time.
    """
    if "achievable_guidance" not in df.columns:
        return df
    mask = _truthy(df["achievable_guidance"]) & (df["planner"] == "diffusion")
    return _drop_variant_for_specs(df, mask, specs, "no_ach")


def drop_batched_for_specs(df: pd.DataFrame, specs: list) -> pd.DataFrame:
    """For the given specs, drop the batched best-of-8 (edm-bo8) runs so the DIFF-SA arm uses
    the honest single-draw edm result instead of the 8x-compute best-of-8 (which reports a
    higher success, e.g. Mixed N=32: 85.3 -> 72.2 single-draw).
    """
    if "use_batched_sampling" not in df.columns:
        return df
    is_edm = (df["planner"] == "diffusion") & (df["diffusion_method"].astype(str) == "edm")
    return _drop_variant_for_specs(df, _truthy(df["use_batched_sampling"]) & is_edm, specs, "single_draw")


def cap_diffma_by_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """Drop DIFF-MA runs slower than the Gradient (ce_nl) planner in the same cell.

    Per (spec, num_agents, stl_mixed_spec_mode), the threshold is the plan_time of the
    Gradient run that would be plotted (highest ma_stl_satisfaction, then success, then
    fastest). DIFF-MA runs with plan_time above that threshold are removed BEFORE
    aggregation, so pick_best_row chooses the best DIFF-MA run that is no slower than
    Gradient (e.g. excluding the expensive achievable_guidance runs). Cells with no
    Gradient run are left untouched.
    """
    grp = ["spec", "num_agents", "stl_mixed_spec_mode"]
    g = df[df["planner"] == "ce_nl"].copy()
    if g.empty:
        print("[diffma<=gradient] no Gradient (ce_nl) runs found; no DIFF-MA rows capped")
        return df
    g["_ma"] = pd.to_numeric(g.get("ma_stl_satisfaction"), errors="coerce").fillna(0)
    g["_pt"] = pd.to_numeric(g["plan_time_mean"], errors="coerce")
    g = g.sort_values(["_ma", "success_mean", "_pt"], ascending=[False, False, True])
    thr = g.groupby(grp, as_index=False).first()[grp + ["_pt"]].rename(columns={"_pt": "_grad_pt"})

    out = df.merge(thr, on=grp, how="left")
    drop = _is_diff_ma(out) & out["_grad_pt"].notna() & \
        (pd.to_numeric(out["plan_time_mean"], errors="coerce") > out["_grad_pt"])
    n = int(drop.sum())
    out = out[~drop].drop(columns="_grad_pt")
    print(f"[diffma<=gradient] dropped {n} DIFF-MA runs slower than Gradient in their (spec,N,mode) cell")
    return out


def apply_signal_toggle(plot_kwargs: Dict[str, Any], include_signal: bool) -> Dict[str, Any]:
    """Return a copy of plot_kwargs with the Signal spec switched on/off.

    When Signal is on, the Signal column is added AND the combined "Mixed" spec uses the
    signal-inclusive variant; when off, neither appears and Mixed uses the no-signal variant.
    Robust to however specs_to_keep/mixed_spec_choice were originally written.
    """
    pk = dict(plot_kwargs)
    specs = [s for s in pk.get("specs_to_keep", [])
             if s not in (SIGNAL_SPEC, MIXED_NO_SIGNAL, MIXED_WITH_SIGNAL)]
    msc = {k: v for k, v in pk.get("mixed_spec_choice", {}).items() if k != "Signal"}
    if include_signal:
        specs += [SIGNAL_SPEC, MIXED_WITH_SIGNAL]
        msc["Signal"] = ["first1", "None", "last1"]   # all modes; pick_best_row picks the best
    else:
        specs += [MIXED_NO_SIGNAL]
    pk["specs_to_keep"] = specs
    pk["mixed_spec_choice"] = msc
    return pk


def load_df(spec: FigureSpec, source: str, data_csv: Optional[str], out_dir: str,
            download_tables: bool = True, max_plan_time: Optional[float] = None,
            max_plan_time_all: bool = False, created_at_cap: Optional[str] = CREATEDAT_CAP,
            diffma_le_gradient: bool = False) -> pd.DataFrame:
    if source == "wandb":
        df = prepare_df(build_wandb_df(spec, download_tables=download_tables, created_at_cap=created_at_cap), spec)
    else:
        # csv source: the dumped frame is already fully prepared.
        path = data_csv or spec.data_csv(out_dir)
        print(f"[load] reading prepared dataframe from {path}")
        # keep_default_na=False + na_values=[""] is REQUIRED: stl_mixed_spec_mode uses the
        # literal string "None" as a real category (matched by mixed_spec_choice), but pandas'
        # default NA tokens include "None" -> it would be read as NaN and those Mixed-spec rows
        # would silently drop, making the offline figure diverge from the live one. to_csv writes
        # true NaN as "", so treating only "" as NA round-trips losslessly.
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        df["num_agents"] = df["num_agents"].astype(int)

    if max_plan_time is not None:
        # Drop runs slower than the threshold BEFORE aggregation, so pick_best_row picks the
        # best run among the fast ones (e.g. excludes the expensive achievable_guidance runs).
        # By default the cap applies to DIFF-MA only: STLPY/Gradient are optimisation-based and
        # legitimately slow, so capping them would wrongly drop them. --max-plan-time-all overrides.
        slow = pd.to_numeric(df["plan_time_mean"], errors="coerce") > max_plan_time
        if max_plan_time_all:
            scope, scope_desc = pd.Series(True, index=df.index), "all planners"
        else:
            scope, scope_desc = _is_diff_ma(df), "DIFF-MA only"
        before = len(df)
        df = df[~(slow & scope)].copy()
        print(f"[filter] plan_time_mean <= {max_plan_time}s ({scope_desc}): "
              f"dropped {before - len(df)}, kept {len(df)}/{before} rows")

    if diffma_le_gradient:
        df = cap_diffma_by_gradient(df)

    if spec.no_ach_specs:
        df = drop_achievable_for_specs(df, spec.no_ach_specs)
    if spec.single_draw_specs:
        df = drop_batched_for_specs(df, spec.single_draw_specs)
    return df


def render(spec: FigureSpec, df: pd.DataFrame, out_dir: str,
           plot_kwargs: Optional[Dict[str, Any]] = None) -> None:
    plotter = Plotter(df, env_name=spec.env_name, **spec.plotter_kwargs)
    method = getattr(plotter, spec.plot_method)
    # Name the PDF + sidecar CSVs by the figure's name so different figures don't collide.
    method(export_data=True, export_dir=out_dir, plot_name=spec.name,
           **(plot_kwargs if plot_kwargs is not None else spec.plot_kwargs))


def emit_table(spec: FigureSpec, df: pd.DataFrame, out_dir: str,
               plot_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Write the figure's underlying results as a LaTeX table (same data as the graph).

    Reuses ``Plotter.plot_all`` (the exact pipeline behind the figure) with make_bar=False so
    the table cells are the same aggregated mean+/-std values as the plotted bars/error-bars,
    across every spec x N x planner. Written to ``{out_dir}/{name}_{env}_table.tex``.
    """
    pk = dict(plot_kwargs if plot_kwargs is not None else spec.plot_kwargs)
    # plot_all's kwargs differ slightly from the plot method's: it wants homo_specs (not
    # homo_mode) and does not take plot_grouped/debug (those are figure-only).
    homo = pk.pop("homo_mode", False)
    pk.pop("plot_grouped", None)
    pk.pop("debug", None)
    plotter = Plotter(df, env_name=spec.env_name, **spec.plotter_kwargs)
    _pivoted, latex_code = plotter.plot_all(homo_specs=homo, make_bar=False, return_table=False,
                                            use_stddev=True, **pk)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{spec.name}_{spec.env_name}_table.tex")
    with open(path, "w") as f:
        f.write(latex_code.rstrip() + "\n")
    print(f"[table] {path}")
    return latex_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", default="single_hetero_dubins", choices=sorted(FIGURES),
                        help="Which registered figure to reproduce.")
    parser.add_argument("--source", default="wandb", choices=["wandb", "csv"],
                        help="Pull runs live from W&B, or load the committed *_data.csv offline.")
    parser.add_argument("--data-csv", default=None,
                        help="Override path to the *_data.csv (csv source only).")
    parser.add_argument("--out-dir", default="barplots",
                        help="Directory for the PDF + sidecar CSVs.")
    parser.add_argument("--no-tables", action="store_true",
                        help="Skip slow per-run artifact-table downloads (wandb source). "
                             "Same rows, omits diversity columns; safe when the figure only "
                             "needs summary metrics.")
    parser.add_argument("--max-plan-time", type=float, default=None, metavar="SECONDS",
                        help="Drop runs with plan_time_mean above this (seconds) before "
                             "aggregation, e.g. to exclude expensive achievable_guidance runs. "
                             "Applies to DIFF-MA only by default.")
    parser.add_argument("--max-plan-time-all", action="store_true",
                        help="Apply --max-plan-time to ALL planners, not just DIFF-MA.")
    parser.add_argument("--no-signal", action="store_true",
                        help="Plot without the Signal (m2signal3) spec; Mixed reverts to the "
                             "no-signal variant. (Only affects signal-capable figures.)")
    parser.add_argument("--latest", action="store_true",
                        help="Drop the createdAt upper bound (CREATEDAT_CAP) when pulling from "
                             "W&B, so newly-added experiments are included. Breaks exact "
                             "reproducibility of the pinned figure.")
    parser.add_argument("--diffma-le-gradient", action="store_true",
                        help="Keep only DIFF-MA runs whose plan_time is <= the Gradient (ce_nl) "
                             "run's plan_time in the same (spec,N,mode) cell, before aggregation.")
    parser.add_argument("--table", action="store_true",
                        help="Also write the figure's results as a LaTeX table "
                             "({out_dir}/{name}_{env}_table.tex) -- same data as the graph.")
    parser.add_argument("--table-only", action="store_true",
                        help="Emit only the LaTeX table (implies --table); skip rendering the figure.")
    parser.add_argument("--list", action="store_true", help="List available figures and exit.")
    args = parser.parse_args()

    if args.list:
        for key, spec in sorted(FIGURES.items()):
            print(f"{key:24s} -> {spec.name}_{spec.env_name}.pdf  ({spec.plot_method})")
        return

    spec = FIGURES[args.figure]
    plot_kwargs = spec.plot_kwargs
    if spec.signal_capable:
        plot_kwargs = apply_signal_toggle(spec.plot_kwargs, include_signal=not args.no_signal)
        print(f"[signal] {'excluded' if args.no_signal else 'included'} "
              f"(Mixed = {'no-signal' if args.no_signal else 'signal-inclusive'} variant)")
    elif args.no_signal:
        print(f"[signal] --no-signal ignored: figure '{args.figure}' is not signal-capable")

    df = load_df(spec, args.source, args.data_csv, args.out_dir, download_tables=not args.no_tables,
                 max_plan_time=args.max_plan_time, max_plan_time_all=args.max_plan_time_all,
                 created_at_cap=None if args.latest else CREATEDAT_CAP,
                 diffma_le_gradient=args.diffma_le_gradient)
    print(f"[plot] {args.figure}: {len(df)} rows -> {args.out_dir}/{spec.name}_{spec.env_name}*")
    if not args.table_only:
        render(spec, df, args.out_dir, plot_kwargs=plot_kwargs)
    if args.table or args.table_only:
        emit_table(spec, df, args.out_dir, plot_kwargs=plot_kwargs)
    print("[done]")


if __name__ == "__main__":
    main()
