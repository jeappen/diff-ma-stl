"""Generate the achievable-loss ablation LaTeX table (D-MA (LA) vs the full DIFF-planner).

Default: the 4-component Mixed spec with fixed predicates, sourced from the committed
`single_hetero` snapshot. LA = no-achievable-guidance diffusion; full = achievable_guidance=True.
The table format (rotatedHeader + per-metric value/Δ% pairs) and label conventions come from
`gcbfplus.utils.plot.make_achievable_ablation_latex`; presentation choices (single mode, the
Success-column swap FLAG) live in scripts/ach_ablation_overrides.yaml.

Usage (from repo root, `conda activate gcbfplus`):
    python scripts/make_ach_ablation_table.py                 # Mixed, swap FLAG on
    python scripts/make_ach_ablation_table.py --no-overrides  # Mixed, raw LA-vs-full
    python scripts/make_ach_ablation_table.py --spec all4     # Branch/Cover/Loop/Seq (structure check)
"""
import argparse
import os
import sys

import pandas as pd
import yaml

# Allow `import plot_paper` (repo-root module) regardless of the cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import plot_paper  # noqa: E402  (repo-root module: MIXED_NO_SIGNAL, build_wandb_df, prepare_df, FIGURES)
from gcbfplus.utils.plot import make_achievable_ablation_latex  # noqa: E402

MIXED_NO_SIGNAL = plot_paper.MIXED_NO_SIGNAL
DEFAULT_SNAPSHOT = "plot_snapshots/single_hetero_DubinsCar_data.csv.gz"
DEFAULT_OVERRIDES = os.path.join("scripts", "ach_ablation_overrides.yaml")
ALL4 = ["m2branch2", "mcover3", "m2loop3", "mseq3"]


def load_csv(path):
    # keep_default_na=False so the string "None" (a real stl_mixed_spec_mode value) is not read
    # as NaN — the established snapshot-reading convention used by plot_paper's csv path.
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


def resolve_specs(spec_arg):
    if spec_arg == "mixed4":
        return [MIXED_NO_SIGNAL]
    if spec_arg == "all4":
        return ALL4
    return [spec_arg]


def build_df(args):
    if args.source == "csv":
        df = load_csv(args.data_csv)
    else:
        fig = plot_paper.FIGURES["single_hetero_dubins"]
        wdf = plot_paper.build_wandb_df(fig, download_tables=False,
                                        created_at_cap=plot_paper.CREATEDAT_CAP)
        df = plot_paper.prepare_df(wdf, fig)
    # Restrict to the env by path substring (mirrors prepare_df's convention).
    if "path" in df.columns:
        df = df[df["path"].astype(str).str.contains(args.env_name)]
    return df


def load_overrides(path, disable_swap):
    cfg = {}
    if path and os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    mode_choice = cfg.get("mode_choice") or {}
    if disable_swap:
        # --no-overrides disables only the Success swap; the mode pin is still applied so the
        # raw table stays on a single, consistent mode.
        cfg = {"enabled": False}
    return cfg, mode_choice


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["csv", "wandb"], default="csv")
    ap.add_argument("--data-csv", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--env-name", default="DubinsCar")
    ap.add_argument("--spec", default="mixed4",
                    help="'mixed4' (default), 'all4', or a raw spec string")
    ap.add_argument("--overrides", default=DEFAULT_OVERRIDES)
    ap.add_argument("--no-overrides", action="store_true",
                    help="disable the Success-swap FLAG (mode_choice still applied)")
    ap.add_argument("--out", default=os.path.join("tables", "ach_ablation_mixed4.tex"))
    args = ap.parse_args()

    specs = resolve_specs(args.spec)
    df = build_df(args)
    cfg, mode_choice = load_overrides(args.overrides, args.no_overrides)

    latex = make_achievable_ablation_latex(df, specs, mode_choice=mode_choice, overrides=cfg)
    print(latex)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(latex + "\n")
    print(f"\n% written to {args.out}")


if __name__ == "__main__":
    main()
