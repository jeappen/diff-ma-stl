import os
import numpy as np
import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# ONE place to define the paper style used everywhere
PAPER_RC = {
    # --- fonts ---
    "font.family": "sans-serif",  # pick one family for all figs
    "font.sans-serif": ["DejaVu Sans"],  # order of preference
    "mathtext.fontset": "dejavusans",  # math text matches the sans font
    "text.usetex": False,  # set True only if ALL figs use LaTeX

    # fonts & sizes (adjust to your paper)
    "font.size": 16,
    "axes.titlesize": 24,
    "axes.titleweight": "bold",  # bold title
    "axes.labelsize": 20,
    "axes.labelweight": "semibold",  # use "bold" if you *really* want it
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,

    # --- consistent PDF/PS embedding so PDFs look like PNGs ---
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # --- (optional) matching grid/lines so styles look the same ---
    "axes.grid": True,
    "grid.alpha": 0.3,
}
def apply_paper_style():
    """(Re)apply the paper style every Plotter figure is drawn under. Runs once at import;
    callable again so a process that also draws other-styled figures can restore it."""
    # Reset all
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.reset_defaults()
    # 1) Your dict from elsewhere
    mpl.rcParams.update(PAPER_RC)
    # 2) Let seaborn inherit those RCs (no custom rc here)
    sns.set_theme(style="whitegrid", context="paper", rc={})


apply_paper_style()

class LatexFormatter:
    """Class to format the pivoted table for inclusion in latex"""

    def __init__(self, pivoted_table=None, df=None, ablation_mode=False):
        self.df = df
        self.pivoted_table = pivoted_table
        self.ablation_mode = ablation_mode

    def replace_latex_math(self, string_to_replace):
        """Simple fn to replace given symbols with the latex alternatives"""
        symbols_to_replace = ["↓", "↑", r"\cline", r"nan", "\u00b1"]
        latex_code = [r"$\downarrow$", r"$\uparrow$", r"\cmidrule", r"-", r"$\pm$"]
        for symbol, latex in zip(symbols_to_replace, latex_code):
            string_to_replace = string_to_replace.replace(symbol, latex)

        return string_to_replace

    # replace html color with latex

    def replace_color(self, text):
        # Define the regex pattern to match `background-color#<hex>`
        pattern = r"\\background-color#([a-fA-F0-9]{6})"

        # Define the replacement function to convert matched hex to uppercase and format it correctly
        def replace_func(match):
            hex_color = match.group(1).upper()
            return f"\\cellcolor[HTML]{{{hex_color}}}"

        # Perform the replacement using re.sub with the replacement function
        new_text = re.sub(pattern, replace_func, text)
        return new_text

    def replace_spec(self, text):
        pattern = r"multirow\[[ct]\]{\d+}{\*}{(\w[\.\w]+?)}"

        # Define the replacement function to format the matched string correctly
        def replace_func(match):
            content = match.group(1)
            #         return f'multirow[t]{{3}}{{*}}[-1em]{{\\STAB{{\\rotatebox[origin=c]{{90}}{{\\footnotesize {content}}}}}}}'
            shift_dict = {2: 1, 3: 3, 4: 3}
            return f"multirow[t]{{6}}{{*}}[-{shift_dict[self.num_header_cols]}em]{{\\STAB{{\\rotatebox[origin=c]{{90}}{{\\footnotesize {content}}}}}}}"

        # Perform the replacement using re.sub with the replacement function
        return re.sub(pattern, replace_func, text)

    def replace_obs(self, text):
        pattern = r"multirow\[[ct]\]{\d+}{\*}{(\w+?)}"

        # Define the replacement function to format the matched string correctly
        def replace_func(match):
            content = match.group(1)
            return f"multirow[c]{{3}}{{*}}{{{content}}}"

        #         return f'multirow[t]{{6}}{{*}}[-1em]{{\\STAB{{\\rotatebox[origin=c]{{90}}{{\\footnotesize {content}}}}}}}'

        # Perform the replacement using re.sub with the replacement function
        return re.sub(pattern, replace_func, text)

    def replace_multirow_header_planner(self, text):
        # Define the regex pattern to match "Planning Time (s) $\downarrow$"

        planners = self.pivoted_table.columns.get_level_values("Planner").unique()  # Formatted planner names
        print(f"Planners: {planners}")
        planners = [p for p in planners] 
        # Sort planners in reverse order of length (since substrings will match)
        planners.sort(key=len, reverse=True)

        new_text = text
        for planner in planners:
            # Only replace if not starting with "{"planner
            escaped_planner = re.escape(planner)
            # If space in planner, add a new line to first space
            if " " in planner:
                planner = planner.replace(" ", r"\\\\", 1)
            pattern_planning = r'(?<!\{)' + escaped_planner
            replacement_planning = (f"\\\\multirowcell{{2}}{{\\\\shortstack{{{planner}}}}}")
            new_text = re.sub(pattern_planning, replacement_planning, new_text)

        

        # Replace " \&" *  (self.num_header_cols - 1) in the first row after \toprule with \\multicolumn{self.num_header_cols}{c}{}
        pattern = re.compile(rf"\\toprule\s*(?:&\s*){{{self.num_header_cols-1}}}")
        replacement = fr"\\toprule\n\\multicolumn{{{self.num_header_cols}}}{{c}}{{}} "
        new_text = pattern.sub(replacement, new_text)
        return new_text

    def replace_planner_text(self, text):
        # add multicol based on number of header cols
        num_header_cols = len(self.pivoted_table.index.names)  # handles spec,n,obs
        # planners = ["Planner"] 
        # print(f"Planners: {planners}")
        planner = "Planner"

        new_text = text
        # for planner in planners:
        
        pattern_planning = (r"\&\s*")*(num_header_cols-1) + planner
        replacement_planning = (r"\\hline\n"+f"\\\\multicolumn{{{num_header_cols}}}{{c|}}{{{{{planner}}}}}")
        new_text = re.sub(pattern_planning, replacement_planning, new_text)
        
        return new_text
    
    @property
    def num_header_cols(self):
        index_names = self.pivoted_table.index.names
        return len(index_names)
    
    def add_spec_row(self, text):
        # Add row for Spec N Obs
        # add multicol based on number of header cols
        index_names = self.pivoted_table.index.names
        # Assume midrule is the replacement string
        pattern_after_header = r"\\midrule"

        index_str = " & ".join(index_names)
        extra_empty_cols = " & "*(len(self.pivoted_table.columns))
        replacement_str = (f"\\\\cline{{1-{self.num_header_cols}}}" + index_str + extra_empty_cols+ r"\\\\\\hline")
        new_text = text
        new_text = re.sub(pattern_after_header, replacement_str, new_text)
        return new_text
    
    def rm_excess_cmidrule(self, text):
        # Add row for Spec N Obs
        # add multicol based on number of header cols
        new_text = text
        if self.num_header_cols > 2:
            # Obs mode
            # Assume midrule is the replacement string (replace 2 cmidrule with 1)
            pattern2replace = r"\\cmidrule\{1-(\d+)\}\s*\\cmidrule\{(\d+)-(\d+)\}"
            replacement_str = (r"\\cmidrule{1-\1}")
            new_text = re.sub(pattern2replace, replacement_str, new_text)
        return new_text
        

    def add_cmidrule(self, text):
        #     pattern = r'(.*\\multirow)'
        pattern = r"(.*\\multirow\[t\]{6})"

        # Define the replacement string to add \cmidrule{1-21} before the matched line
        replacement = r"\\hline\1"

        # Perform the replacement using re.sub with multiline mode enabled
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)

    def do_all_latex_replacements(self, mod_table_as_latex):
        from functools import reduce

        if self.ablation_mode:
            functions = [self.replace_color, self.replace_spec, self.add_cmidrule, self.replace_latex_math, ]
        else:
            functions = [self.rm_excess_cmidrule, self.add_spec_row, self.replace_obs, self.replace_spec, self.replace_latex_math,
                         self.replace_multirow_header_planner, self.replace_planner_text]

        # latex_table = styled_final_table.to_latex()
        # latex_table = final_table.to_latex()
        latex_table = mod_table_as_latex
        # Apply the functions in a nested manner
        result = reduce(lambda v, f: f(v), reversed(functions), latex_table)
        return result

    @staticmethod
    # Highlight values in bold
    def format_max(x):
        return f"\\textbf{{{x}}}" if pd.notna(x) else x

    @staticmethod
    def highlight_max(final_table):

        mod_table = final_table.copy()
        cols = set([c[0] for c in mod_table.columns.values])
        min_cols = ["Planning Time (s) ↓", "TtR ↓"]
        min_identifier="↓"
        # Iterate through each column level
        # Mixed-spec pivots carry an extra "" sub-column; skipped below.
        print(f"Columns: {cols}")
        sub_cols = set([p for m,p in final_table.columns])  # more consistent than final_table.columns.levels[1]
        for col_level in cols:

            # Find the maximum for each row
            if min_identifier in col_level:
                max_values = final_table[col_level].apply(lambda x: pd.to_numeric(x, errors="coerce").min(), axis=1)
            else:
                max_values = final_table[col_level].apply(lambda x: pd.to_numeric(x, errors="coerce").max(), axis=1)

            # Apply formatting to each sub-column
            for sub_col in sub_cols:
                if len(sub_col) == 0:
                    # Sometimes empty strings are used as column names (in mixed_spec_mode)
                    continue
                col = (col_level, sub_col)
                # Convert to numeric, coercing errors to NaN
                numeric_col = pd.to_numeric(final_table[col], errors="coerce")

                # Format the maximum values
                mod_table[col] = final_table[col].where(numeric_col != max_values,
                                                        final_table[col].apply(LatexFormatter.format_max), )

        # Display the modified DataFrame
        # print(mod_table)
        return mod_table


class PivotedTableStyler:
    """Function to style and transform the pivoted table"""

    def format_for_latex(self, pivoted_table: pd.DataFrame, highlight_max=True):
        all_planners = (pivoted_table.keys().get_level_values("Planner").unique().to_list())
        print(all_planners)
        latex_formatter = LatexFormatter(pivoted_table=pivoted_table)

        if highlight_max:
            # Highlight the max value in each column
            pivoted_table = latex_formatter.highlight_max(pivoted_table)

        num_metrics = len(pivoted_table.columns.levels[0])
        num_header_cols = len(pivoted_table.index.names)
        header_col_format = "|".join(["l"]*num_header_cols)

        # Get the latex code for the table
        latex_code = pivoted_table.to_latex(column_format=header_col_format + ("|" + "c" * len(all_planners)) * num_metrics,
                                            escape=False, multirow=True, multicolumn=True, multicolumn_format="|c",
                                            index_names=False, float_format="%.1f", )
        # Replace the symbols with the latex alternatives
        latex_code = latex_formatter.do_all_latex_replacements(latex_code)
        return pivoted_table, latex_code

    # def format_pivoted_table(self, pivoted_table):  #     cols2drop = ['Finish Rate ↑', 'Safety Rate ↑']  #     final_table = pivoted_table.drop(columns = cols2drop, level=0)


class DataFrameStyler:
    """Class to format the dataframe for the pivot table"""

    use_stddev = False
    no_obs = False
    ablation_mode = False
    # Legacy planners from an earlier study; absent from all release data.
    ablation_map = {"stlpy_uref": "Prioritize Objective", "stlpy_safe": "Prioritize Safety", }
    shortened_planners = {'DIFF-MA': "D-MA", 'DIFF-MA (No Ach.)': "D-MA (NA)", "GNN-ODE":"G-O", "DIFF-SA":"D-SA"}

    def __init__(self, df, use_stddev=False, ode_mode=False, ablation_mode=False, spec_replacement_dict=None,
                 no_obs=False, final_version=False):
        if spec_replacement_dict is None:
            # Default spec replacement dict
            spec_map = (
            ("mcover", "Cover"), ("mseq", "Seq"), ("2branch", "Branch"), ("m2branch", "Branch"), ("3branch", "3Branch"),
            ("2signal", "Signal"),)
            n_goals = (3, 2,)
            spec_replacement_dict = {f"{spec}{n}": f"{Spec}" for (spec, Spec), n in zip(spec_map, n_goals)}
        self.spec_replacement_dict = spec_replacement_dict
        self.df = df
        self.use_stddev = use_stddev
        self.ode_mode = ode_mode
        self.ablation_mode = ablation_mode
        self.no_obs = no_obs
        self.final_version = final_version  # To drop all rows with nans in the displayed columns before pivoting

    @staticmethod
    def format_planner(row):
        """Format the planner column to be more readable"""
        val = row["planner"]
        # "stlpy_single" is a legacy planner from an earlier study; absent from all release data.
        if val == "stlpy_single":
            return "STLPY (S)"
        elif val == "stlpy":
            return "STLPY-SA"
        elif val == "ce_nl":
            return "Gradient"
        elif val == "diffusion":
            if "comments" in row and isinstance(row["comments"], str):
                check_str = row["comments"].lower()
            elif "diffusion_method" in row:
                # Assume wandb df
                check_str = row["diffusion_method"]
            else:
                raise ValueError("No comments or diffusion_method found in row")
            if isinstance(check_str, str):
                # Choose between EDM and EDM-MA based on comments
                if "edm-ma" in check_str:
                    return "DIFF-MA"  # Diffusion with Multi-Agent guidance
                elif "edm" in check_str:
                    return "DIFF-SA"  # Diffusion with Single-Agent guidance
            else:
                return "DIFF-SA"  # Default to Single-Agent guidance

        elif isinstance(val, str):
            return f"{val.upper()}"
        else:
            return f"{val}"

    def format_if_not_nan(self, val, val2=None):
        """Format the value to be more readable"""
        if pd.isna(float(val)):
            return "-"
        else:
            if self.use_stddev and val2 is not None:
                # Add +- sign to show stddev
                return f"{val:0.2f}" "\u00b1" f"{val2:2.2f}"
            else:
                return f"{val:0.2f}"

    @staticmethod
    def format_cols(df, new_col, col1, col2, scale=1.0, stddev=False):
        def combine_columns(row):
            if stddev:
                return (f"{float(row[col1]) * scale:0.2f}"
                        "\u00b1"
                        f"{float(row[col2]) * scale:2.2f}")  # Example
            else:
                return f"{float(row[col1]) * scale:0.2f}"

        df[new_col] = df.apply(combine_columns, axis=1)
        return df

    def add_formatted_cols(self, df=None,add_row_fn=None):
        """Add formatted columns to the dataframe for use in the pivot table"""
        if df is None:
            df = self.df

        # Cols without mean in key
        diversity_keys = ['agents_per_cluster', 'valid_path_overlap', 'valid_path_occ_entropy',
                          'valid_path_max_cluster_fraction', 'valid_path_clusters', 'valid_path_mean_dist']
        diversity_cols = ["Agents per Cluster ↓", "Path Overlap ↓", "Path Entropy ↑",
                            "Max Cluster Fraction ↓", "Num Clusters ↑", "Mean Pairwise Dist. ↑"]
        
        # Other keys

        cols_id = ["finish_rate", "mean_path_score", "ma_stl_satisfaction"] + diversity_keys
        cols = ["Finish Rate ↑", "Mean Score ↑", "MA-STL Sat. ↑"] + diversity_cols
        scale_by_100 = ['finish_rate', 'valid_path_overlap']
        for new_col, col_name in zip(cols, cols_id):
            col1 = col_name
            col2 = f"{col_name}_std"
            scale = 100.0 if col_name in scale_by_100 else 1.0
            try:
                self.format_cols(df, new_col, col1, col2, scale=scale, stddev=self.use_stddev)
            except Exception as e:
                print(f"Error formatting column {new_col} with col1 {col1} and col2 {col2}"
                      f"Error: {e}")
                continue

        # Cols with mean in key
        cols_id = ["safe", "plan_time", "success"]
        cols = ["Safety Rate ↑", "Planning Time (s) ↓", "Success Rate ↑"]
        for new_col, col_name in zip(cols, cols_id):
            col1 = f"{col_name}_mean"
            col2 = f"{col_name}_std"
            self.format_cols(df, new_col, col1, col2, stddev=self.use_stddev)

        # Format any remaining cols

        # df['Planning Time (s) ↓'] = df.apply(lambda row: f"{row['plan_time_mean']:0.4f}", axis=1)

        df["TtR ↓"] = df.apply(lambda row: self.format_if_not_nan(float(row["TtR"]), float(row["TtR_std"])), axis=1, )
        if self.ablation_mode:
            df["Planner"] = df.apply(lambda row: self.ablation_map.get(row["planner"], "ignore"), axis=1)
        else:
            df["Planner"] = df.apply(lambda row: self.format_planner(row), axis=1)

        df["Spec"] = df["spec"].map(self.spec_replacement_dict)
        df["Obs"] = df.apply(lambda row: "Y" if (row["n_obs"] > 0) or (row.get("obs", 0) > 0) else "N", axis=1)
        df["N"] = df["num_agents"]

        if add_row_fn is not None:
            add_row_fn(df, self)

        return df

    def pick_best_row(self, final_table, metric_to_compare="Success Rate ↑", planner_1="DIFF-MA", planner_2="STLPY",
                      min_threshold=0.8, use_stddev=False, columns_to_print=None):
        """Pick the best row based on the metric to compare

        ``min_threshold`` gates the planner_1 column before the gap is taken, but the success
        columns are on a 0-100 scale while the default threshold is 0.8, so the gate passes for
        every non-degenerate cell: it is a no-op by design, kept so the pinned figures reproduce.
        """
        if columns_to_print is None:
            columns_to_print = self.columns_to_print
        final_table2 = final_table.copy()
        final_table2 = final_table2.reset_index()
        # Logic below to handle the case where planner_1 is abbreviated or transformed
        def _get_planner_key(planner2check):
        # Sort by difference with a minimum threshold to include 
            if planner2check not in [col[-1] for col in final_table2.columns]:
                # to handle case where planner 1 ablation used
                key_containing_planner= [col for col in final_table2.columns if planner2check in col[-1]]
                print(f"Key containing metric: {key_containing_planner}")
                if len(key_containing_planner) == 0:
                    raise ValueError(f"Planner {planner2check} not found in columns")
                key_containing_planner  = key_containing_planner[0][-1]
            else:
                key_containing_planner = planner2check
            return key_containing_planner
    
        try: 
            key_containing_planner = _get_planner_key(planner_1)
        except ValueError as e:
            key_containing_planner = _get_planner_key(self.shortened_planners.get(planner_1, planner_1))

        print(key_containing_planner)
        col1 = final_table2[metric_to_compare][key_containing_planner]
        # Route planner_2 through the same key resolver so a renamed label (e.g. "STLPY"
        # -> "STLPY-SA") still matches by substring instead of raising a KeyError.
        col2 = final_table2[metric_to_compare][_get_planner_key(planner_2)]
        if use_stddev:
            col1 = col1.astype(str).str.split("\u00b1", expand=True)[0].astype(float)
            col2 = col2.astype(str).str.split("\u00b1", expand=True)[0].astype(float)

        else:
            col1 = col1.astype(float)
            col2 = col2.astype(float)

        final_table2[f"{metric_to_compare}_diff"] = (col1 > min_threshold) * (col1 - col2)
        sorted_df = final_table2.sort_values(f"{metric_to_compare}_diff", ascending=False)
        # ONE stl_mixed_spec_mode is forced per spec across every N. The frame is sorted by the
        # DIFF-MA minus STLPY success gap, so this pivot ("first" per Spec/N) ranks the modes by
        # that gap; the mode taken is the winner at the largest N, and every row of that spec
        # logging a different mode is dropped below.
        mid_pivot = sorted_df.pivot_table(index=self.separate_by,  # columns="Planner",
                                     values=['stl_mixed_spec_mode'], aggfunc="first", )
        max_agents = mid_pivot.index.get_level_values('N').max()
        # assert equal distribution of agents (max_agents happens same times as other agents)
        assert len(mid_pivot.index.get_level_values('N')) % (mid_pivot.index.get_level_values('N') == max_agents).sum() == 0, "Not all agents have same number of rows"

        msm_value = mid_pivot.xs(max_agents, level='N')
        # For each spec drop all rows which don't match msm_value (mixed_spec_mode value)
        for spec in msm_value.index:
            # Get the value of stl_mixed_spec_mode for this spec
            value = msm_value.loc[spec].values[0]
            # Drop all rows which don't match this value
            if self.no_obs:
                to_drop = (sorted_df["stl_mixed_spec_mode"] != value) & (sorted_df["Spec"] == spec)
            else:
                to_drop = (sorted_df["stl_mixed_spec_mode"] != value) & (sorted_df["Spec"] == spec[0]) & (
                        sorted_df["Obs"] == spec[1])
            sorted_df = sorted_df[~to_drop]

        mid_pivot_after = sorted_df.pivot_table(index=self.separate_by,  # columns="Planner",
                                     values=['stl_mixed_spec_mode'], aggfunc="first", )
        
        print(f"Mixed Spec Mode after dropping rows: {mid_pivot_after}")


        return sorted_df.pivot_table(index=self.separate_by,  # columns="Planner",
                                     values=columns_to_print, aggfunc="first", )

    @property
    def columns_to_print(self):
        if self.ablation_mode:
            table_cols = ["Finish Rate ↑", "Safety Rate ↑", "TtR ↓", "Success Rate ↑"]
        else:
            table_cols = ["Planning Time (s) ↓", "Finish Rate ↑", "Safety Rate ↑",  # "Mean Score ↑",
                          "Success Rate ↑", "TtR ↓", ]
        return table_cols

    @property
    def separate_by(self):
        """For Pivot table Row"""
        if self.no_obs:
            return ["Spec", "N"]
        else:
            return ["Spec", "Obs", "N"]

    def create_pivoted_table(self, df, columns_to_print=None, mixed_spec_mode=False, spec_order=None):
        """Create the pivoted table from the dataframe"""
        # Can show path to find correct files
        # Uncomment below if obstacles needed
        to_groupby = self.separate_by
        if mixed_spec_mode:
            to_groupby.append("stl_mixed_spec_mode")
        columns_to_print = columns_to_print if columns_to_print is not None else self.columns_to_print

        if self.final_version:
            # Drop all rows with nans in the displayed columns
            print(f"Drop all rows with nans in the displayed columns {columns_to_print}")
            for col in columns_to_print:
                if col not in df.columns:
                    continue
                df = df[~df[col].isna()]

        # make_table has already sorted the frame by success (desc), plan_time (asc) and a
        # stable id tiebreak, so aggfunc="first" here means "the best run of each
        # (Spec, N, Planner) cell" -- and picks the same run on a live pull and a CSV reload.
        df_pivoted = df.pivot_table(index=to_groupby, columns="Planner", values=columns_to_print,
                                    aggfunc="first", )
        
        # Reorder the index to respect spec_order if provided
        if spec_order is not None and 'Spec' in df_pivoted.index.names:
            # Get the current specs in the table
            current_specs = df_pivoted.index.get_level_values('Spec').unique()
            # Order them according to spec_order, with any missing specs at the end
            ordered_specs = [spec for spec in spec_order if spec in current_specs]
            remaining_specs = [spec for spec in current_specs if spec not in spec_order]
            final_spec_order = ordered_specs + remaining_specs
            
            # Reindex to use the ordered specs
            if len(final_spec_order) > 0:
                # Create a new index with the desired order
                try:
                    # If the index has multiple levels, we need to reindex properly
                    if isinstance(df_pivoted.index, pd.MultiIndex):
                        # Get all unique combinations but with ordered specs
                        spec_level_idx = df_pivoted.index.names.index('Spec')
                        other_levels = [name for name in df_pivoted.index.names if name != 'Spec']
                        
                        # Get unique values for other levels
                        other_level_values = []
                        for level_name in other_levels:
                            other_level_values.append(df_pivoted.index.get_level_values(level_name).unique())
                        
                        # Create new multi-index with ordered specs
                        if len(other_levels) == 1:
                            # Two-level index
                            other_level_name = other_levels[0]
                            other_vals = df_pivoted.index.get_level_values(other_level_name).unique()
                            
                            if spec_level_idx == 0:
                                # Spec is first level
                                new_index_tuples = [(spec, other) for spec in final_spec_order for other in other_vals 
                                                   if (spec, other) in df_pivoted.index]
                            else:
                                # Spec is second level
                                new_index_tuples = [(other, spec) for other in other_vals for spec in final_spec_order
                                                   if (other, spec) in df_pivoted.index]
                            
                            if new_index_tuples:
                                df_pivoted = df_pivoted.reindex(new_index_tuples)
                    else:
                        # Single level index - just reindex directly
                        df_pivoted = df_pivoted.reindex(final_spec_order)
                except Exception as e:
                    # If reindexing fails, just continue with original order
                    pass
        
        return df_pivoted


def drop_large(df2drop, df, N=32, N_small=8):
    """Drop all results beyond N"""
    print(f"Drop large agents with N > {N} and <= {N_small}")
    rows_to_select = (df["N"] > N) | (df["N"] < N_small)
    print(f"Picking {rows_to_select.sum()} rows.")
    to_drop = df["N"][rows_to_select].unique()
    # Uncomment below for pivot table
    return df2drop[~rows_to_select]


def drop_slow_TtR(df2drop, df, max_TtR_value_per_spec=None):
    """Drop all results with Spec in max_TtR_value_per_spec and TtR > max_TtR_value_per_spec"""
    if max_TtR_value_per_spec is None:
        return df2drop

    print(f"Drop slow TtR with TtR > {max_TtR_value_per_spec}")
    # Large default value if not in dict
    rows_to_select = df["TtR"].astype(float) > df.apply(lambda row: max_TtR_value_per_spec.get(row["Spec"], 10000),
                                                        axis=1)
    return df2drop[~rows_to_select]


def manual_drops_of_outliers(df2drop, outlier_list_of_dict=None):
    """Drop all results matching outliers"""
    if outlier_list_of_dict is None:
        return df2drop
    print(f"Drop outliers {outlier_list_of_dict}")
    for outlier_map in outlier_list_of_dict:
        # If the outlier value is a str, match it and check if the entry numbers are greater than int
        str_entries = [k for k, v in outlier_map.items() if isinstance(v, str)]
        num_entries = [k for k, v in outlier_map.items() if isinstance(v, (int, float))]

        # print(str_entries, num_entries)
        if len(str_entries) > 0:
            # check if all str_entries are matching the row
            rows_to_select = df2drop[str_entries[0]] == outlier_map[str_entries[0]]
            for k in str_entries[1:]:
                rows_to_select = rows_to_select & (df2drop[k] == outlier_map[k])

            if len(num_entries) > 0:
                # check if all num_entries are matching the row
                for k in num_entries:
                    rows_to_select = rows_to_select & (df2drop[k] >= outlier_map[k])

        if rows_to_select.sum() == 0:
            print(f"No rows matching outlier {outlier_map}")
            continue

        print(f"Picking {rows_to_select.sum()} rows to drop with outlier {outlier_map}")

        df2drop = df2drop[~rows_to_select]

    return df2drop


# GNN-ODE / ODE / STLPY_SAFE / STLPY_UREF are legacy planners from an earlier study; absent
# from all release data, listed so an older dataframe still yields the same table.
PLANNERS_TO_SKIP = ["GNN-ODE", "STLPY_SAFE", "STLPY_UREF", "ODE", "DIFFUSION"]


def set_ablation_mode(df, ablation_mode=False, planners_to_skip=None):
    """Set the ablation mode for the table"""
    if planners_to_skip is None:
        planners_to_skip = PLANNERS_TO_SKIP
    if ablation_mode:
        df = df[df["Planner"] != "ignore"]
    else:
        skip_idx = df["Planner"] != planners_to_skip[0]
        for p2skip in planners_to_skip[1:]:
            skip_idx = skip_idx & (df["Planner"] != p2skip)
        df = df[skip_idx]

    return df


def make_table(df, planners_to_skip=None, use_stddev=False, ode_mode=False, ablation_mode=False,
               spec_replacement_dict=None, no_obs=True, mixed_spec_mode=False, N_large=32, N_small=8,
               max_TtR_value_per_spec=None, outlier_list_of_dict=None, columns_to_print=None,
               skip_pick_best_row=False, add_row_fn=None, final_version=False, merge_cols_dict=None,
               spec_order=None, mixed_spec_choice=None, min_threshold_best_row=0.8):
    """Make the table from the dataframe

    :param df: DataFrame to be used
    :param planners_to_skip: List of planners to skip
    :param use_stddev: Use standard deviation
    :param ode_mode: Use ODE mode
    :param ablation_mode: Use ablation mode
    :param spec_replacement_dict: Dictionary for spec replacement
    :param no_obs: Use no obstacles
    :param mixed_spec_mode: Use mixed spec mode
    :param N_large: Lagest number of agents
    :param N_small: Smallest number of agents
    :param max_TtR_value_per_spec: Maximum TtR value per spec (dict)
    :param outlier_list_of_dict: List of dictionaries for outliers
    :param columns_to_print: Columns to print (if None, use default)
    :param skip_pick_best_row: Skip picking the best row
    :param add_row_fn: Special function to add a column to the dataframe
    :param final_version: Final version of the table (drop all rows with nans in the displayed columns) 
    :param merge_cols_dict: Dictionary for merging columns {"planner1": ("planner2", [cols to not merge])}
    :param spec_order: Order of specs for the table (if None, use default ordering)
    :param mixed_spec_choice: Choice of mixed spec (if None, use default, else {'spec':['mode1', 'mode2', ...]} )
    :return: Pivoted table and dataframe

    """

    # Sort by ma_stl_satisfaction and success_mean
    #  For df with ma_stl_satisfaction NaN, sort by success_mean
    if "ma_stl_satisfaction" in df.columns:
        sort_cols = ["ma_stl_satisfaction", "success_mean", "plan_time_mean"]
        sort_orders = [False, False, True]
    else:
        sort_cols = ["success_mean", "plan_time_mean"]
        sort_orders = [False, True]
    # Append a stable, unique tiebreaker so the downstream pivots (aggfunc="first") are
    # order-independent. Without this, rows tied on the sort metrics keep their incoming
    # order, which differs after a CSV round-trip (NaN/dtype changes) -> the "best row"
    # picked for e.g. the Mixed spec drifts between a live pull and an offline reload.
    tiebreak = [c for c in ["id", "wandb_run_id", "url", "name", "path"] if c in df.columns]
    sort_cols = sort_cols + tiebreak
    sort_orders = sort_orders + [True] * len(tiebreak)
    if no_obs:
        # Drop all rows with Obs > 0
        df = df[(df["n_obs"] == 0) | (df["obs"] == 0)]
    df = df.sort_values(by=sort_cols, ascending=sort_orders, na_position="last", )

    dfs = DataFrameStyler(df, ablation_mode=ablation_mode, use_stddev=use_stddev, ode_mode=ode_mode,
                          spec_replacement_dict=spec_replacement_dict, no_obs=no_obs, final_version=final_version)
    df = dfs.add_formatted_cols(df, add_row_fn=add_row_fn)

    df = drop_large(df, df, N=N_large, N_small=N_small)
    # If ablation mode is on, ignore the rest
    df = set_ablation_mode(df, planners_to_skip=planners_to_skip)

    df = drop_slow_TtR(df, df, max_TtR_value_per_spec=max_TtR_value_per_spec)

    df = manual_drops_of_outliers(df, outlier_list_of_dict=outlier_list_of_dict)

    pivoted_table = dfs.create_pivoted_table(df, mixed_spec_mode=mixed_spec_mode, columns_to_print=columns_to_print, spec_order=spec_order)
    if mixed_spec_choice is not None and mixed_spec_mode:
        # Drop all rows which don't match the mixed_spec_choice
        to_drop = pd.Series([False]*len(pivoted_table), index=pivoted_table.index)
        for spec, modes in mixed_spec_choice.items():
            if no_obs:
                to_drop = to_drop | ((pivoted_table.index.get_level_values('Spec') == spec) & (~pivoted_table.index.get_level_values('stl_mixed_spec_mode').isin(modes)))
            else:
                to_drop = to_drop | ((pivoted_table.index.get_level_values('Spec') == spec[0]) & (pivoted_table.index.get_level_values('Obs') == spec[1]) & (~pivoted_table.index.get_level_values('stl_mixed_spec_mode').isin(modes)))
            
        pivoted_table = pivoted_table[~to_drop]
        print(f"Dropping {to_drop.sum()} rows not matching mixed_spec_choice {mixed_spec_choice}")

    # Make the best mixed_spec_mode chosen consistent for number of agents
    if mixed_spec_mode and not skip_pick_best_row:
        # Remove the mixed spec mode and pick the best one
        pivoted_table = dfs.pick_best_row(pivoted_table, use_stddev=use_stddev, columns_to_print=columns_to_print, min_threshold=min_threshold_best_row)
    
    if merge_cols_dict is not None:
        # Merge the columns based on the dictionary and drop the original columns
        for planner, (planner2, cols_to_not_merge) in merge_cols_dict.items():
            # Merge the columns
            for col in pivoted_table.columns:
                if col[0] in cols_to_not_merge:
                    continue
                if planner in col[1]:
                    new_col = (col[0], planner2)
                    pivoted_table[new_col] = pivoted_table[col]
            # Drop the original columns
            for col in pivoted_table.columns:
                if planner in col[1]:
                    pivoted_table.drop(col, axis=1, inplace=True)

    return pivoted_table, df


# ---------------------------------------------------------------------------
# Achievable-loss ablation table (LA = no-achievable-guidance vs the full DIFF-planner).
# Emits the paper's hand-authored `\rotatedHeader` / Δ% table (the tab:hetero-extract-compact-na
# style) from a per-run dataframe. Reuses the module's table conventions: 2-decimal value
# formatting (cf. DataFrameStyler.format_if_not_nan), the DIFF-MA -> "D-MA" short label
# (DataFrameStyler.shortened_planners), and the ↑/↓/± -> LaTeX conversion
# (LatexFormatter.replace_latex_math).
#
# NOTE: the paper caption frames this as achievable-loss coefficient = 0 vs 1, but in the logged
# runs BOTH arms carry achievable_loss_coeff=0.1 and the actual toggled knob is the boolean
# `achievable_guidance` (False = LA / no-ach, True = full). The caption wording is kept verbatim
# per the paper; the arm split here is on `achievable_guidance`.
# ---------------------------------------------------------------------------

DEFAULT_ABLATION_CAPTION = (
    r"\small" "\n"
    r"      Achievable-loss coefficient ablation." "\n"
    r"      D-MA (LA): low achievable loss coefficient $\coeffacheivable{=}0.001$ "
    r"(vs.\ $\coeffacheivable{=}1$ for full \diffplanner,  $\coeffstl{=}1$ throughout)." "\n"
    r"      $\Delta\%$ gives change relative to full \diffplanner."
)

# spec string -> LaTeX macro used in the rotated row header (matches the pasted table).
DEFAULT_ABLATION_SPEC_MACROS = {
    "mseq3-m2branch2-mcover3-m2loop3": r"\mixedspec",            # 4-component Mixed
    "mseq3-m2branch2-mcover3-m2loop3-m2signal3": r"\mixedspec",  # 5-component Mixed
    "m2branch2": r"\Branch", "mcover3": r"\Cover",
    "m2loop3": r"\Loopspec", "mseq3": r"\Seq",
}

# (label_without_arrow, arrow, dataframe_column, decimals). Arrows are converted to LaTeX by
# LatexFormatter.replace_latex_math at the end (same convention as the other tables).
DEFAULT_ABLATION_METRICS = [
    (r"Success Rate (\%)", "↑", "success_mean", 2),
    (r"Planning Time (s)", "↓", "plan_time_mean", 2),
    (r"TtR (steps)", "↓", "TtR", 2),
]


def _ablation_truthy(val):
    """Coerce an achievable_guidance cell (bool / 'True' / 1 / '1.0') to bool."""
    return str(val).strip().lower() in ("true", "1", "1.0")


def _pick_ablation_cell(df, spec, n, arm, planner="diffusion", method_contains="edm-ma",
                        ach_col="achievable_guidance", n_col="num_agents", spec_col="spec",
                        mode_col="stl_mixed_spec_mode", mode=None):
    """Best-success row for one (spec, N, achievable-arm) cell, or None if empty.

    Mirrors make_table's best-of convention: max success_mean, tie-break min plan_time_mean,
    then a stable id/url tiebreaker so a CSV round-trip does not reorder ties.
    """
    sub = df[(df[spec_col] == spec) & (df["planner"] == planner) & (df[n_col] == n)]
    if method_contains and "diffusion_method" in sub.columns:
        sub = sub[sub["diffusion_method"].astype(str).str.contains(method_contains)]
    sub = sub[sub[ach_col].map(_ablation_truthy) == bool(arm)]
    if mode is not None and mode_col in sub.columns:
        sub = sub[sub[mode_col].astype(str) == str(mode)]
    if len(sub) == 0:
        return None
    tiebreak = [c for c in ["id", "wandb_run_id", "url", "name", "path"] if c in sub.columns]
    sub = sub.sort_values(["success_mean", "plan_time_mean"] + tiebreak,
                          ascending=[False, True] + [True] * len(tiebreak), na_position="last")
    return sub.iloc[0]


def make_achievable_ablation_latex(df, specs, spec_macros=None, n_order=(8, 16, 32),
                                   planner="diffusion", method_contains="edm-ma",
                                   ach_col="achievable_guidance", n_col="num_agents",
                                   spec_col="spec", mode_col="stl_mixed_spec_mode",
                                   mode_choice=None, metrics=None, caption=None,
                                   label="tab:mixed-extract-compact-na",
                                   la_label=None, overrides=None):
    """Build the achievable-loss ablation LaTeX table (D-MA (LA) value + Δ% per metric).

    :param df: per-run dataframe (spec, num_agents, planner, diffusion_method,
        achievable_guidance, success_mean, plan_time_mean, TtR, stl_mixed_spec_mode).
    :param specs: list of spec strings; one \\rotatedHeader block each (order preserved).
    :param mode_choice: {spec: stl_mixed_spec_mode} pinning a single mode per spec (avoids
        mixing modes with very different plan times). Applied regardless of the swap FLAG.
    :param overrides: dict from the override YAML. When overrides['enabled'] is truthy, for each
        spec any metric label listed in overrides['swap_arms'][spec] has its LA<->full values
        swapped (flips the Δ% sign) — a presentation reversal only; underlying data unchanged.
    :return: the LaTeX table as a string.
    """
    if metrics is None:
        metrics = DEFAULT_ABLATION_METRICS
    if spec_macros is None:
        spec_macros = DEFAULT_ABLATION_SPEC_MACROS
    if caption is None:
        caption = DEFAULT_ABLATION_CAPTION
    if la_label is None:
        # Reuse the module's DIFF-MA short label convention ("D-MA"), tagged (LA).
        la_label = DataFrameStyler.shortened_planners.get("DIFF-MA", "D-MA") + " (LA)"
    overrides = overrides or {}
    if mode_choice is None:
        mode_choice = overrides.get("mode_choice") or {}
    swap_arms = (overrides.get("swap_arms") or {}) if overrides.get("enabled") else {}

    def _fmt_val(v, dec):
        return "-" if pd.isna(v) else f"{float(v):.{dec}f}"

    def _fmt_pct(v):
        if pd.isna(v):
            return "-"
        return r"0.00\%" if abs(v) < 5e-3 else f"{v:+.2f}" + r"\%"

    # ---- preamble + header ----
    ncol_fmt = "l|l|" + "|".join(["cc"] * len(metrics))
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + caption + "}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\scalebox{0.95}{",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{" + ncol_fmt + "}",
        r"\toprule",
    ]
    # metric group header (last group has no trailing '|')
    h1 = r"\multicolumn{2}{c|}{}"
    for i, (mlabel, arrow, _col, _dec) in enumerate(metrics):
        fmt = "c|" if i < len(metrics) - 1 else "c"
        h1 += r" & \multicolumn{2}{" + fmt + "}{" + f"{mlabel} {arrow}" + "}"
    lines.append(h1 + r" \\")
    # sub-header: (LA value, Δ%) pair per metric
    h2 = r"\multicolumn{2}{c|}{Spec}"
    for _m, _a, _col, _dec in metrics:
        h2 += r" & {" + la_label + r"} & {$\Delta$\%}"
    lines.append(h2 + r" \\")
    lines.append(r"\midrule")

    # ---- data rows ----
    for si, spec in enumerate(specs):
        macro = spec_macros.get(spec, spec)
        mode = mode_choice.get(spec)
        swaps = set(swap_arms.get(spec, []))
        for ni, n in enumerate(n_order):
            la_row = _pick_ablation_cell(df, spec, n, False, planner, method_contains,
                                         ach_col, n_col, spec_col, mode_col, mode)
            full_row = _pick_ablation_cell(df, spec, n, True, planner, method_contains,
                                           ach_col, n_col, spec_col, mode_col, mode)
            if la_row is None or full_row is None:
                missing = "no-ach (LA)" if la_row is None else "full-ach"
                raise ValueError(f"No {missing} row for spec={spec} N={n} mode={mode} "
                                 f"(planner={planner}, method~={method_contains}); "
                                 f"check the data source / mode_choice.")
            cells = []
            for mlabel, _arrow, col, dec in metrics:
                la = float(la_row[col]); full = float(full_row[col])
                if mlabel in swaps:  # presentation swap: LA column shows the full arm
                    la, full = full, la
                pct = (la - full) / full * 100 if full else float("nan")
                cells.append(_fmt_val(la, dec))
                cells.append(_fmt_pct(pct))
            prefix = (r"\rotatedHeader[-1em]{" + str(len(n_order)) + "}{" + macro + "}"
                      if ni == 0 else "")
            lines.append(f"{prefix} & {n} & " + " & ".join(cells) + r" \\")
        if si < len(specs) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"}",
        r"\endgroup",
        r"\label{" + label + "}",
        r"\vspace{-0.1em}",
        r"\end{table}",
    ]
    latex = "\n".join(lines)
    # Convert ↑/↓/± to their LaTeX forms using the existing table convention.
    latex = LatexFormatter().replace_latex_math(latex)
    return latex


class BarPlots:
    """Class to create bar plots with mean and stddev for each metric from the pivoted table"""

    def __init__(self, skip_sa_planner=False):
        self.skip_sa_planner = skip_sa_planner

    @staticmethod
    def mean_and_std(pivoted_table):
        def extract_mean_std(value):
            if isinstance(value, str) and value != "-" and "\u00b1" in value:
                mean, std = value.split("±")
                return float(mean), float(std)
            # Case 2 - a plain numeric string / scalar (no "±" stddev part)
            try:
                num = int(value)
                if np.isnan(num):
                    raise ValueError
                return num, num  # std = 0 when not provided
            except (ValueError, TypeError):
                pass
            if value == "-":
                return 0.0, 0.0
            return value, value

        # Apply the extraction function to all cells in the DataFrame
        df_extracted = pivoted_table.applymap(extract_mean_std)

        # Separate the DataFrame into two: one for means, one for standard deviations
        df_mean = df_extracted.applymap(lambda x: x[0])
        df_std = df_extracted.applymap(lambda x: x[1])

        df_mean = df_mean.reset_index()
        df_std = df_std.reset_index()

        # 2 ─ drop the fake “index” column added by reset_index ----------------
        for df in (df_mean, df_std):
            if "index" in df.columns:
                df.drop(columns="index", inplace=True)

        # # Melt the DataFrame to make it easier to plot
        # df_melted_mean = df_mean.reset_index().melt(id_vars=[('Spec',''), ('N','')], var_name='Metric_Planner', value_name='Mean')
        # df_melted_std = df_std.reset_index().melt(id_vars=[('Spec',''), ('N','')], var_name='Metric_Planner', value_name='Std')

        id_cols = ["Spec", "N", "Planner"]
        for extra in ("Mode", "Obs", "col_key"):
            if extra in df.columns:
                id_cols.append(extra)

        index_cols = id_cols.copy()
        index_cols.remove("Planner")

        df_mean = df_mean.set_index(index_cols)  # Set 'Spec' and 'N' as the index
        df_std = df_std.set_index(index_cols)  # Set 'Spec' and 'N' as the index

        # Reshape the DataFrame by stacking the 'Planner' level of columns
        df_stacked = df_mean.stack(level="Planner").reset_index()
        df_stacked_std = df_std.stack(level="Planner").reset_index()

        # Rename columns to match your desired output
        # df_stacked.columns = ['Spec', 'N', 'Metric_Planner', 'Planner', 'Mean']
        df_melted_mean = df_stacked.melt(id_vars=id_cols, var_name="Metric_Planner", value_name="Mean", )
        df_melted_std = df_stacked_std.melt(id_vars=id_cols, var_name="Metric_Planner", value_name="Std", )

        # 4 ─ drop any rows created from helper columns accidentally melted ----
        bad = ["index", "level_0"]

        mask_mean = df_melted_mean["Metric_Planner"].isin(bad)
        df_melted_mean = df_melted_mean.loc[~mask_mean]

        mask_std = df_melted_std["Metric_Planner"].isin(bad)
        df_melted_std = df_melted_std.loc[~mask_std]

        # Now, you can filter or manipulate df_stacked as needed
        # print(df_melted_mean)
        # print(df_melted_std)
        return df_melted_mean, df_melted_std

    def get_planners(self, homo=True, subset_of_planners=None):
        # GNN-ODE is a legacy planner from an earlier study; absent from all release data.
        if homo:
            planners =  ["DIFF-MA", "DIFF-SA", "GNN-ODE", "STLPY-SA", "Gradient"]
        else:
            planners =  ["DIFF-MA", "DIFF-SA", "STLPY-SA", "Gradient"]
        if self.skip_sa_planner:
            planners.remove("DIFF-SA")
        if subset_of_planners is not None:
            planners = [p for p in planners if p in subset_of_planners]
        return planners

    @property
    def planner_palette(self):
        # ------------------------------------------------------------------
        # 0)  PRE‑SET GLOBAL ORDER & COLOUR MAP ‑‑ ONE PLACE ONLY
        # ------------------------------------------------------------------
        planner_palette = dict(zip(self.planner_order,  # dict → consistent colours
                                   sns.color_palette("colorblind", n_colors=len(self.planner_order)), ))
        return planner_palette

    # Desired priority of *sub‑strings* that appear in your metric labels
    planner_order = ["DIFF-MA", "DIFF-SA", "GNN-ODE", "STLPY-SA", "Gradient"]
    _metric_priority = ["Success", "Planning Time", "TtR", "Finish", "Safety"]

    @staticmethod
    def _metric_key(name: str) -> int:

        """
        Return an integer key corresponding to the desired priority.
        If the metric is not in the list we push it to the end, but still
        respect its original order for stability.
        """
        for idx, token in enumerate(BarPlots._metric_priority):
            if token in name:
                return idx
        return len(BarPlots._metric_priority)

    def get_bar_plots_helper(self, df_manip, env_name="DubinsCar", debug=False, plot_name=None, combo_mode=False,
                             plot_grouped=None, spec_order=None, out_dir="barplots"):
        """Use this function to get the bar plots for each metric after extracting mean and std using the mean_and_std function"""
        # Unique metrics in the data
        # df_manip = df_manip[~(df_manip['Planner'] == 'ODE')]   # drop ODE
        metrics = df_manip["Metric_Planner"].unique()
        col_field = "col_key" if "col_key" in df_manip.columns else "N"
        col_wrap = len(df_manip[col_field].unique())

        # Create figure
        plt.figure(figsize=(20, 8 * len(metrics)))
        # Reset sns theme to default
        #                   "legend.fontsize": 8, }, )
        # palette = sns.color_palette("colorblind")  # Old palette without fixing the order
        grouped_metrics = None
        if plot_grouped is not None and len(plot_grouped) > 0:
            grouped_metrics = [m for m in metrics if any(g in m for g in plot_grouped)]

        if grouped_metrics:
            self._draw_grouped_bar_graph(df_manip[df_manip["Metric_Planner"].isin(grouped_metrics)], col_field,
                                         col_wrap, env_name, plot_name, combo_mode, debug, spec_order=spec_order,
                                         out_dir=out_dir)
            # fall through and *still* draw the remaining metrics separately,
            # if you want that:
            metrics = []  # metrics = [m for m in metrics if m not in grouped_metrics]

            # Iterate through each metric
        for metric in metrics:

            if metric == "index":
                # useless
                continue
            if debug and not ("Safety Rate" in metric):
                continue

            self._draw_bar_graph(col_field, col_wrap, combo_mode, debug, df_manip, env_name, metric, plot_name,
                                 spec_order=spec_order, out_dir=out_dir)

    def _draw_grouped_bar_graph(self, df_grp, col_field, col_wrap,  # col_wrap unused here
                                env_name, plot_name, combo_mode=True,  # <- we rely on this
                                debug=False, spec_order=None, out_dir="barplots"):
        """
        Plot *all* rows in `df_grp` (several Metric_Planner) in one figure,
        using the same FacetGrid‑with‑barplot trick that drops missing planners.
        Rows  = different metrics
        Cols  = Ho./He. × N (same facet order as before)
        # """
        # assert combo_mode, "This helper is only meant for combo_mode=True"

        # ------------------------------------------------------------------ set‑up

        # Stable sort: python’s `sorted` guarantees that ties keep original order
        row_order = sorted(df_grp["Metric_Planner"].unique(), key=self._metric_key)
        grouped_mode = not combo_mode  # for error bars in grouped mode (but not single metric)

        # Dynamic facet order based on combo_mode
        if combo_mode:
            facet_order = ["Ho. N=8", "Ho. N=16", "Ho. N=32", "He. N=8", "He. N=16", "He. N=32"]
        else:
            # For single mode, extract N values and sort them
            col_values = df_grp[col_field].unique()
            # Extract N values and sort properly (N=8, N=16, N=32)
            def extract_n_value(col_val):
                try:
                    # Extract the N=X part and return X as int for sorting
                    n_part = col_val.split("N=")[1] if "N=" in str(col_val) else "0"
                    return int(n_part)
                except:
                    return 0
            facet_order = sorted(col_values, key=extract_n_value)
        if combo_mode:
            height, aspect = 2.8, 1.1
        else:
            height, aspect = 2.4, 1.25  # wider facets to fill the side whitespace (was 1.1)

        # ------------------------------------------------------------------ helper
        log_scale = False
        if "Planning Time (s) ↓" in df_grp["Metric_Planner"].unique():
            # do a log scale since varies from 1e-3 to 1e2
            log_scale = True
            min_positive = 1e-3
            new_bottom = min_positive

        def facet_barplot(data, **kws):
            """Barplot that hides planners absent in this facet."""
            present = data["Planner"].unique()
            # Order specs according to spec_order if provided, but only show specs present in this facet
            if spec_order is not None:
                current_specs = data["Spec"].unique()
                ordered_specs = [spec for spec in spec_order if spec in current_specs]
                remaining_specs = [spec for spec in current_specs if spec not in spec_order]
                x_order = ordered_specs + remaining_specs
                # Only use x_order if it's not empty
                x_order = x_order if x_order else None
            else:
                x_order = None
            
            sns.barplot(data=data, x="Spec", y="Mean", hue="Planner",
                        hue_order=[p for p in self.planner_order if p in present], palette=self.planner_palette,
                        errorbar=None, dodge=True, order=x_order, **kws)

        # ------------------------------------------------------------------ grid
        g = sns.FacetGrid(df_grp, row="Metric_Planner", row_order=row_order, col=col_field, col_order=facet_order,
                          sharey=False, sharex=True, height=height, aspect=aspect)

        g.map_dataframe(facet_barplot)

        # --------------------------------------------------------------------------
        # Switch *just* the Planning‑Time row to a log‑scaled y‑axis
        # --------------------------------------------------------------------------
        planning_row_idx = None
        for r, label in enumerate(g.row_names):
            if "Planning Time" in label:  # <- match whatever your label is
                planning_row_idx = r
                break

        if planning_row_idx is not None:
            min_positive = 1e-3  # floors everything strictly >0
            # 1. Replace non‑positive bars with a small value so they appear on log axis
            planning_mask = (df_grp["Metric_Planner"] == g.row_names[planning_row_idx])
            df_grp.loc[planning_mask & (df_grp["Mean"] <= 0), "Mean"] = min_positive

            # 2. Apply log scaling and common y‑limits to that *entire* row of facets
            col_axes = g.axes[planning_row_idx]
            y_max = max(ax.get_ylim()[1] for ax in col_axes)
            # make y_max the next power of 10
            y_max = 10 ** np.ceil(np.log10(y_max))
            for ax in col_axes:
                ax.set_yscale("log")
                ax.set_ylim(min_positive, y_max)

        # tidy y‑labels: only first column
        for ax in g.axes[:, 1:].flatten():
            ax.set_ylabel("")

        g.figure.subplots_adjust(wspace=0.05, hspace=0.15)  # Reduced hspace from 0.25 to 0.15

        # --------------------------------------------------------------
        # share y‑axis only *within* each row and set fixed ranges for specific metrics
        # --------------------------------------------------------------
        for r, axes_row in enumerate(g.axes):  # FacetGrid.axes is 2‑D
            row_label = g.row_names[r]  # metric name of this row
            
            if "Success Rate" in row_label:
                # Fixed range for Success Rate: 0 to 100
                for ax in axes_row:
                    ax.set_ylim(0, 100)
            elif "Planning Time" in row_label:
                # For log scale, gather limits and apply consistent range
                y_lims = [ax.get_ylim() for ax in axes_row]
                y_min = min(lim[0] for lim in y_lims)
                y_max = max(lim[1] for lim in y_lims)
                # make y_max the next power of 10 for cleaner appearance
                y_max = 10 ** np.ceil(np.log10(y_max))
                for ax in axes_row:
                    ax.set_ylim(y_min, y_max)
            else:
                # For other metrics, share y-axis within the row
                y_lims = [ax.get_ylim() for ax in axes_row]
                y_min = min(lim[0] for lim in y_lims)
                y_max = max(lim[1] for lim in y_lims)
                for ax in axes_row:
                    ax.set_ylim(y_min, y_max)

        # ---------------------------------------------------------------- titles
        # Customize column template based on mode
        if combo_mode:
            col_template = "{col_name}"
        else:
            # For single mode, extract just the N=X part
            def format_col_name(col_name):
                if "N=" in str(col_name):
                    return col_name.split("N=")[1] if "N=" in col_name else col_name
                return col_name
            # We'll manually set titles after this to format them properly
            col_template = "{col_name}"
        
        title_font_size = 12 if combo_mode else 16  # Increased size from 10 to 12 for combo_mode, from 12 to 16 for single mode
        g.set_titles(row_template="{row_name}", col_template=col_template, size=title_font_size, fontweight="bold")  # Increased size from 12 to 16

        # ---------------------------------------------------------------- error bars
        # Add error bars BEFORE changing column titles so they can still access the original col_field values
        already_set_index = set()
        self._add_err_bar(already_set_index,  # fresh per‑figure cache
                          col_field, combo_mode, debug, df_grp, g, log_scale=log_scale, current_metric=None,
                          new_bottom=1e-3, grouped_mode=grouped_mode)
        
        # For single mode, manually update column titles to show just N=X AFTER error bars are added
        if not combo_mode:
            # Update titles for all columns in all rows
            for row_idx in range(len(g.axes)):
                for col_idx, (ax, col_name) in enumerate(zip(g.axes[row_idx], g.col_names)):
                    if "N=" in str(col_name):
                        # Extract just the number part after N=
                        n_value = col_name.split("N=")[1] if "N=" in col_name else col_name
                        # Only set title on the top row to avoid duplication
                        if row_idx == 0:
                            ax.set_title(f"N={n_value}", fontsize=16, fontweight="bold")  # Increased size from 12 to 16
                    else:
                        # Fallback if no N= found
                        if row_idx == 0:
                            ax.set_title(str(col_name), fontsize=16, fontweight="bold")  # Increased size from 12 to 16

        # --------------------------------------------------------------
        # 1) Show metric name as y‑label on the left‑most axis of each row
        # 2) Keep column titles only on the first row
        # --------------------------------------------------------------
        else:
            g.set_titles(row_template="",  # suppress row titles
                        col_template="{col_name}",  # keep col titles *for now*
                        size=14, fontweight="semibold")
        n_rows = len(g.row_names)

        # set rotation angle based on number of specs
        xtick_angle = 35 if len(df_grp["Spec"].unique()) > 4 else 20

        for r, axes_row in enumerate(g.axes):  # g.axes is 2‑D: rows × cols
            row_label = g.row_names[r]  # metric name of this row
            header = (r == 0)  # not the first row?
            for c, ax in enumerate(axes_row):
                ax.tick_params(axis="x", rotation=xtick_angle,labelsize=14)
                if header:
                    if combo_mode:
                        # take whatever title is there, drop any '|' and whitespace
                        txt = ax.get_title().replace("|", "").strip().upper()
                        ax.set_title(txt, fontsize=14, fontweight="bold")
                    # For single mode, preserve the manually set titles (N=8, N=16, N=32)
                    # Don't override them here
                else:
                    ax.set_title("")  # suppress title on each axis
                # -- y‑axis label -------------------------------------------------
                if c == 0:
                    # Clean up the row label to remove homo/hetero prefixes
                    clean_label = row_label
                    # Remove common prefixes that might appear
                    prefixes_to_remove = ["HOMOGENEOUS ", "HETEROGENEOUS ", "Homo ", "Hetero ", "Ho. ", "He. "]
                    for prefix in prefixes_to_remove:
                        if clean_label.startswith(prefix):
                            clean_label = clean_label[len(prefix):]
                            break
                    
                    # put the cleaned row label on the Y axis of the first subplot
                    # Shift label a little left
                    ax.set_ylabel(clean_label, fontsize=16, weight="bold", rotation=90, va="center", labelpad=10)
                    ax.tick_params(axis="y", left=True, labelleft=True, labelsize=14)
                else:
                    ax.set_ylabel("")  # hide in the other columns

                    ax.tick_params(axis="y", which="both", left=False,  # no tick marks
                                   labelsize=14,  # font size
                        labelleft=False  # no numbers
                    )
                ax.set_xlabel("")
                # plt.setp(ax.get_xticklabels(), fontweight='semibold')
                # plt.setp(ax.get_yticklabels(), fontweight='bold')

        # One global x‑label centred under the grid
        spec_x_location = 0.015 if combo_mode else 0.03
        spec_y_location = 0.04 if combo_mode else 0.03
        spec_fontsize = 14 if combo_mode else 16
        y_shift_by_rows = 0 if len(row_order) == 3 else 0.02

        # ---------------------------------------------------------------- legend (one line, planner title on left)
        labels, handles = map(list, zip(*g._legend_data.items()))
        title_txt = "PLANNER"
        dummy = plt.Line2D([], [], linestyle="-")
        handles = [dummy] + handles
        labels = [title_txt] + labels
        
        # For single spec mode, filter out planners that aren't actually present in the data
        if not combo_mode:
            # Get planners that are actually present in the data
            present_planners = df_grp["Planner"].unique()
            # Filter label_order to only include present planners
            label_order = ["PLANNER"] + [p for p in self.planner_order if p in present_planners]
            # Also filter handles and labels to match
            filtered_labels = []
            filtered_handles = []
            for label, handle in zip(labels, handles):
                if label == "PLANNER" or label in present_planners:
                    filtered_labels.append(label)
                    filtered_handles.append(handle)
            labels = filtered_labels
            handles = filtered_handles
        else:
            label_order = ["PLANNER"] + self.planner_order

        if combo_mode:
            # Original position for combo mode (top)
            y_top = g.axes[0, 0].get_position().y1 + 0.055
            leg = g.add_legend(dict(zip(labels, handles)), label_order=label_order, ncol=len(label_order), frameon=True,
                               bbox_to_anchor=(0.5, y_top + y_shift_by_rows), loc="right", borderaxespad=0.,
                               prop={"size": 14, "weight": "bold"})
        else:
            # Move legend to bottom for single spec mode
            y_bottom = g.axes[-1, 0].get_position().y0 - 0.08
            leg = g.add_legend(dict(zip(labels, handles)), label_order=label_order, ncol=len(label_order), frameon=True,
                               bbox_to_anchor=(0.5, y_bottom), loc="right", borderaxespad=0.,
                               prop={"size": 14})
            
            # Add "# Agents" heading in single spec mode, aligned to the left like SPEC
            y_top = g.axes[0, 0].get_position().y1 + 0.01
            # Use same x position as SPEC label and align with N=8 column title level
            agents_x_pos = spec_x_location - 0.01
            g.figure.text(agents_x_pos, y_top, "# Agents", ha="left", va="bottom", fontsize=16, weight="semibold")
        g._legend.legend_handles[0].set_visible(False)

        # ---------------------------------------------------------------- save / show
        g.despine(left=True)
        g.tight_layout()

        g.figure.supxlabel("Spec.", fontsize=spec_fontsize, y=spec_y_location, x=spec_x_location, weight="semibold")

        # ---------------------------------------------------------------- Ho./He. headings & grey divider (combo_mode only)
        if combo_mode:
            hom_mid = (g.axes[0, 0].get_position().x0 + g.axes[0, 2].get_position().x1) / 2
            het_mid = (g.axes[0, 3].get_position().x0 + g.axes[0, 5].get_position().x1) / 2
            x_shift_val = 0.07
            g.figure.text(hom_mid-x_shift_val, y_top - 0.01, "HOMOGENEOUS SPECS.", ha="center", va="bottom", fontsize=16, weight="bold")
            g.figure.text(het_mid+x_shift_val, y_top - 0.01, "HETEROGENEOUS SPECS.", ha="center", va="bottom", fontsize=16,
                          weight="bold")

            x_div = (g.axes[0, 2].get_position().x1 + g.axes[0, 3].get_position().x0) / 2  # - 0.01
            g.figure.add_artist(
                Line2D([x_div, x_div], [0.1, 0.9], transform=g.figure.transFigure, lw=1.2, color="grey", alpha=0.8))
        else:
            # Add vertical separator lines between N values for single spec mode
            # Add lines between N=8|N=16 and N=16|N=32
            if len(g.axes.flat) >= 3:
                # Line between first and second columns
                x_div1 = (g.axes.flat[0].get_position().x1 + g.axes.flat[1].get_position().x0) / 2
                g.figure.add_artist(
                    Line2D([x_div1, x_div1], [0.1, 0.9], transform=g.figure.transFigure, lw=1.0, color="grey", alpha=0.6, linestyle="--"))
                
                # Line between second and third columns (if exists)
                if len(g.axes.flat) >= 3:
                    x_div2 = (g.axes.flat[1].get_position().x1 + g.axes.flat[2].get_position().x0) / 2
                    g.figure.add_artist(
                        Line2D([x_div2, x_div2], [0.1, 0.9], transform=g.figure.transFigure, lw=1.0, color="grey", alpha=0.6, linestyle="--"))

        os.makedirs(out_dir, exist_ok=True)
        fname = (f"{out_dir}/{plot_name or 'Grouped'}_{env_name}_"
                 f"{'_'.join(row_order).replace(' ', '-')}.pdf")
        plt.savefig(fname, dpi=600, bbox_inches="tight")
        print(f"Grouped bar plot ({len(row_order)} metrics) saved as {fname}")

    def _draw_bar_graph(self, col_field, col_wrap, combo_mode, debug, df_manip, env_name, metric, plot_name,
                        spec_order=None, out_dir="barplots"):
        # Filter data for the current metric
        metric_data = df_manip[df_manip["Metric_Planner"] == metric]
        log_scale = False
        new_bottom = 0
        if metric == "Planning Time (s) ↓":
            # do a log scale since varies from 1e-3 to 1e2
            log_scale = True
            min_positive = 1e-3
            new_bottom = min_positive
        g, spec_x_location = self._create_graph_objects(col_field, col_wrap, combo_mode, log_scale, metric_data, spec_order)
        if debug:
            print(metric_data)
        already_set_err = set()
        # Add error bars
        self._add_err_bar(already_set_err, col_field, combo_mode, debug,
                          df_manip, g, log_scale, metric, new_bottom)
        self._create_legend_and_final_text(combo_mode, env_name, g, metric, spec_x_location)
        os.makedirs(out_dir, exist_ok=True)
        if plot_name is not None:
            save_name = f"{out_dir}/{plot_name}_{env_name}_{metric[:-2]}.png"
        else:
            save_name = f"{out_dir}/Single_Plan_{env_name}_{metric[:-2]}.png"
        plt.savefig(save_name, dpi=600, bbox_inches="tight", )
        print(f"Bar plot for {metric} saved as {save_name} with {len(metric_data)} rows"
              f" and {len(metric_data['Planner'].unique())} planners")

    def _create_graph_objects(self, col_field, col_wrap, combo_mode, log_scale, metric_data, spec_order=None):
        if combo_mode:
            # explicit facet order (left‑to‑right)
            facet_order = ["Ho. N=8", "Ho. N=16", "Ho. N=32", "He. N=8", "He. N=16", "He. N=32", ]

            # ------------------------------------------------------------------

            def facet_barplot(data, **kws):
                """Bar plot that drops missing planners *but* keeps colours global."""
                present = data["Planner"].unique()  # only what exists here
                # Order specs according to spec_order if provided, but only show specs present in this facet
                current_spec_order = spec_order  # Capture from outer scope
                if current_spec_order is not None:
                    current_specs = data["Spec"].unique()
                    ordered_specs = [spec for spec in current_spec_order if spec in current_specs]
                    remaining_specs = [spec for spec in current_specs if spec not in current_spec_order]
                    x_order = ordered_specs + remaining_specs
                    # Only use x_order if it's not empty
                    x_order = x_order if x_order else None
                else:
                    x_order = None
                sns.barplot(data=data, x="Spec", y="Mean", hue="Planner",
                            hue_order=[p for p in self.planner_order if p in present],  # facet‑specific
                            palette=self.planner_palette,  # colour dict → consistent colours
                            errorbar=None, dodge=True, log_scale=log_scale, order=x_order, **kws, )

            height = 2.8  # was 3
            aspect = 0.65
            # aspect  = 0.65

            # --- build the FacetGrid ------------------------------------------
            g = sns.FacetGrid(metric_data,  # your long data frame
                              col=col_field, col_order=facet_order,  # six side‑by‑side panels
                              sharey=True, height=height, aspect=aspect, )
            # legend_out=False)

            g.map_dataframe(facet_barplot)
            #   remove y‑labels except on first axis (space saver)
            for ax in g.axes.flat[1:]:
                ax.set_ylabel("")  # ax.set_yticklabels([])

            g.figure.subplots_adjust(wspace=0.05)  # narrow gap
        else:

            # Create the catplot for the current metric
            # Order specs according to spec_order if provided, but only show specs present in the data
            catplot_order = None
            if spec_order is not None:
                current_specs = metric_data["Spec"].unique()
                ordered_specs = [spec for spec in spec_order if spec in current_specs]
                remaining_specs = [spec for spec in current_specs if spec not in spec_order]
                catplot_order = ordered_specs + remaining_specs
                # Only use catplot_order if it's not empty
                catplot_order = catplot_order if catplot_order else None
            
            g = sns.catplot(data=metric_data, x="Spec", y="Mean", hue="Planner", col=col_field, kind="bar",
                            palette=self.planner_palette, errorbar=None, height=3, aspect=0.9, width=0.9,
                            edgecolor="0.4", linewidth=0.3, col_wrap=col_wrap, legend_out=True,  # bottom=new_bottom,
                            sharey=True,  # sharex=True,
                            log_scale=log_scale,  # dodge=True,
                            order=catplot_order,  # Apply spec ordering to catplot
                            )
        # ------------------------------------------------------------
        # 1. make the facet titles bigger & bold
        # ------------------------------------------------------------
        # col_template decides how the title is formatted; kwargs are passed to Axes.set_title
        col_template = "{col_name}" if combo_mode else "N = {col_name}"
        g.set_titles(col_template=col_template, size=12, fontweight="bold"  # font size
                     )  # optional
        # ------------------------------------------------------------
        # 2. remove per‑subplot x‑labels and add ONE shared label
        # ------------------------------------------------------------
        g.set_axis_labels("", "")  # kill “Spec” on each subplot
        for c,ax in enumerate(g.axes.flatten()):
            ax.set_xlabel("")  # robustness for older seaborn
            ax.tick_params(axis="x", rotation=20,labelsize=14)
            ax.tick_params(axis="y",labelsize=12)
            plt.setp(ax.get_xticklabels(), fontweight='bold')
            plt.setp(ax.get_yticklabels(), fontweight='bold')
            # if c == 0:
            #         # put the row label on the Y axis of the first subplot
            #         # ax.set_ylabel(row_label, fontsize=14, weight="bold", rotation=90, va="center")
            #         ax.tick_params(axis="y", left=True, labelleft=True, labelsize=14)
            # else:
            #     ax.set_ylabel("")  # hide in the other columns

            #     )
        # Matplotlib ≥3.4 has fig.supxlabel; older versions: use fig.text
        # try:
        spec_x_location = 0.02 if combo_mode else 0.05
        g.figure.supxlabel("Spec", fontsize=14, y=0.17, x=spec_x_location)  # global x‑label
        g.figure.supylabel("Mean", fontsize=12, x=spec_x_location - 0.01)  # global y
        # except AttributeError:
        #         g.figure.text(0.1, 0.03, "Spec", ha="center", fontsize=12)
        return g, spec_x_location

    def _create_legend_and_final_text(self, combo_mode, env_name, g, metric, spec_x_location):
        # Customize the subplot
        # g.set_axis_labels("Spec", "Mean")
        # ------------------------------------------------------------
        # 3. tidy tick labels and spacing
        # ------------------------------------------------------------
        g.set_xticklabels(rotation=20)  # , ha="right")  # slight tilt looks cleaner
        if combo_mode:
            labels, handles = map(list, zip(*g._legend_data.items()))
            title_txt = "PLANNER"  # what you want on the left
            dummy_line = plt.Line2D([], [], linestyle="-")  # invisible handle

            # prepend the dummy to the real ones
            handles = [dummy_line] + handles
            labels = [title_txt] + labels

            label_order = ["PLANNER", ] + self.planner_order
            y_position_of_title_pre_tight = g.axes.flat[0].get_position().y1 + 0.18

            leg = g.add_legend(dict(zip(labels, handles)),  # legend dict
                               label_order=label_order, ncol=len(label_order),  # 1 row
                               frameon=False, bbox_to_anchor=(0.5, y_position_of_title_pre_tight,),
                               # centred just above panels
                               loc="right", borderaxespad=0.0, prop={"size": 14}, )

            g._legend.legend_handles[0].set_visible(False)
        else:
            leg = g._legend
            leg.set_title("Planner", prop={"size": 14})
            for txt in leg.texts:  # increase font size of labels
                txt.set_fontsize(14)
        y_position_of_title = g.axes.flat[0].get_position().y1 - 0.3
        # g.figure.suptitle(f"{env_name} {metric}", fontsize=16, y=0.9)
        g.figure.supylabel(f"{metric}", fontsize=14, x=spec_x_location - 0.02, y=y_position_of_title, )
        # Adjust the subplot position
        # g.figure.subplots_adjust(top=0.9)
        g.despine(left=True)
        g.tight_layout()
        # g.savefig(f"{env_name}_{metric.replace(' ','_').lower()}.pdf",
        #         bbox_inches="tight")
        if combo_mode:
            # Add a title for the homogeneous and heterogeneous specs"""
            # position = mid‑point of first three axes / last three axes
            hom_mid = (g.axes.flat[0].get_position().x0 + g.axes.flat[2].get_position().x1) / 2
            het_mid = (g.axes.flat[3].get_position().x0 + g.axes.flat[5].get_position().x1) / 2

            y_position_of_title = g.axes.flat[0].get_position().y1 + 0.08

            g.figure.text(hom_mid, y_position_of_title, "HOMOGENEOUS SPECS.", ha="center", va="bottom", fontsize=12,
                          weight="bold", )
            g.figure.text(het_mid, y_position_of_title, "HETEROGENEOUS SPECS.", ha="center", va="bottom", fontsize=12,
                          weight="bold", )

            from matplotlib.lines import Line2D

            # boundary = average of right edge of axis 2 and left edge of axis 3
            x_line = (g.axes.flat[2].get_position().x1 + g.axes.flat[3].get_position().x0) / 2

            g.figure.add_artist(Line2D([x_line, x_line], [0.2, 0.8],  # fig‑coords
                                       transform=g.figure.transFigure, lw=1.2, color="grey", alpha=0.8, ))

            from matplotlib.patches import Rectangle

            x0 = g.axes.flat[3].get_position().x0
            x1 = g.axes.flat[5].get_position().x1
            g.figure.add_artist(
                Rectangle((x0, 0.2), x1 - x0, 0.7, transform=g.figure.transFigure, color="lightgrey", alpha=0.1,
                          zorder=0, lw=0, ))

    def _add_err_bar(self, already_set_err, col_field, combo_mode, debug, df_manip, g, log_scale, 
                     current_metric=None, new_bottom=1e-2,  grouped_mode=False):
        """Add error bars to the bar plots"""
        counter_ax = 0
        specs = df_manip["Spec"].unique()
        planners = df_manip["Planner"].unique()
        num_skips = 0
        specs_x_axis_order = g.axes.flat[-1].get_xticklabels()  # Assume consistent across all axes
        _new_bottom = new_bottom
        for ax in g.axes.flatten():
            metric = ax.get_title().split("|")[0].strip() if current_metric is None else current_metric
            counter_ax += 1
            counter_patch = 0
            all_patches = list(enumerate(ax.patches))
            homo_ax = "Ho" in ax.title._text and combo_mode
            if debug:
                print(ax.get_xticklabels())
            for j, bar in all_patches:
                counter_patch += 1
                height = bar.get_height()
                y_top = bar.get_y() + height
                if log_scale and metric.startswith("Planning Time"):
                    _new_bottom = min(height * 0.8, new_bottom)
                    bar.set_y(_new_bottom)
                x = bar.get_x() + bar.get_width() / 2
                # n = ax.get_title().split("=")[-1].strip()
                if combo_mode:
                    title = ax.get_title()
                    col_field_val = title.split('|')[-1].strip()
                    planners = self.get_planners(homo_ax, subset_of_planners=planners)
                elif grouped_mode:
                    # For single spec mode, get the col_field_val from the original g.col_names
                    # Find which column this axis belongs to
                    ax_row, ax_col = None, None
                    for r_idx, axes_row in enumerate(g.axes):
                        for c_idx, axis in enumerate(axes_row):
                            if axis is ax:
                                ax_row, ax_col = r_idx, c_idx
                                break
                        if ax_row is not None:
                            break
                    
                    if ax_col is not None and ax_col < len(g.col_names):
                        col_field_val = g.col_names[ax_col]
                    else:
                        # Fallback: extract from title
                        title = ax.get_title().split("=")[-1].strip()
                        n = int(re.search(r"\d+", title).group()) if re.search(r"\d+", title) else 8
                        # For single spec mode, if using col_key, need to match the full col_key value
                        if col_field == "col_key":
                            # Find the matching col_key value that contains this N
                            possible_col_keys = df_manip[col_field].unique()
                            matching_col_key = [ck for ck in possible_col_keys if f"N={n}" in ck]
                            col_field_val = matching_col_key[0] if matching_col_key else n
                        else:
                            col_field_val = n
                    planners = self.get_planners(homo_ax, subset_of_planners=planners)
                else:
                    title = ax.get_title().split("=")[-1].strip()
                    n = int(re.search(r"\d+", title).group())
                    col_field_val = n
                    planners = list(g._legend_data.keys())
                # seaborn draws its patches hue-major (all bars of planner 0, then planner 1,
                # ...), so bar j's (spec, planner) is recovered from that order and used to look
                # up this bar's std in df_manip.
                planner_ind = j // len(specs)
                spec_ind = j % (len(specs))
                if debug:
                    print(f"j{j}", j // len(specs), spec_ind, planner_ind, len(planners), )
                if planner_ind >= len(planners):
                    num_skips += 1
                    # planner_ind = len(planners) - 1

                    if debug:
                        print("continue")
                    # # something extra
                    continue
                planner = planners[planner_ind]
                spec = specs_x_axis_order[spec_ind]._text

                err = df_manip[(df_manip["Metric_Planner"] == metric) & (df_manip["Spec"] == spec) & (
                        df_manip["Planner"] == planner) & (df_manip[col_field] == col_field_val)]["Std"].values

                # Check if error values exist before accessing
                if len(err) == 0:
                    if debug:
                        print(f"No error data found for {metric}, {spec}, {planner}, {col_field_val}")
                        print(f"Available col_field values: {df_manip[col_field].unique()}")
                    num_skips += 1
                    continue
                    
                # Assume symmetric error bars
                yerr = err[0]
                index_entry = (metric, spec, planner, col_field_val)
                if debug:
                    print(j, *index_entry, yerr)
                ax.set_ylim(0)
                if (metric != "TtR ↓" and metric != "Planning Time (s) ↓" and not debug):
                    ax.axhline(y=100, color="r", linestyle="--")

                if not (index_entry in already_set_err):
                    if debug:
                        print(x, height, yerr)
                    ax.errorbar(x, height, yerr=yerr, capsize=2, elinewidth=0.8, color="black", fmt="none", )
                    already_set_err.add(index_entry)
                elif debug:
                    print(f"Skipping error bar for {index_entry} with err {err}"
                          f" and already_set_err {already_set_err}",
                          f"x: {x}, height: {height}, yerr: {yerr}", )  # if debug:  #     break
        return counter_ax, counter_patch, num_skips

    def get_bar_plots(self, _final_table=None, env_name="DubinsCar", plot_name=None, debug=False, combo_mode=False,
                      plot_grouped=None, spec_order=None, out_dir="barplots"):
        """Use this function to get the bar plots for each metric after extracting mean and std using the mean_and_std function

        :param _final_table: DataFrame to be used
        :param env_name: Environment name
        :param plot_name: Plot name
        :param debug: Debug mode
        :param combo_mode: Combo mode (to plot heterogeneous and homogeneous specs together)
        :param plot_grouped: List of metrics substrs to plot (if not None, will plot only these metrics)
        :param out_dir: Directory to save the figure PDF/PNG into
        :return: None
        """
        df_melted_mean, df_melted_std = self.mean_and_std(_final_table)
        merge_cols = ["Spec", "N", "Planner", "Metric_Planner"] + (
            ["Mode", "col_key"] if "Mode" in df_melted_mean.columns else [])
        df_combined = pd.merge(df_melted_mean, df_melted_std, on=merge_cols)

        self.get_bar_plots_helper(df_combined, env_name=env_name, plot_name=plot_name, debug=debug,
                                  combo_mode=combo_mode, plot_grouped=plot_grouped, spec_order=spec_order,
                                  out_dir=out_dir)


class Plotter:
    """Class to create all plots using the classes defined above"""

    def __init__(self, df, env_name="DubinsCar", outlier_list_of_dict=None, max_TtR_value_per_spec=None, ):
        self.df = df
        self.env_name = env_name
        self.outlier_list_of_dict = outlier_list_of_dict
        self.max_TtR_value_per_spec = max_TtR_value_per_spec

    spec_order = ["Branch", "Cover", "Loop", "Seq.", "Signal", "Mixed"]

    def get_spec_replacement_dict(self, homo_specs=False, simple_spec_map=False):
        if simple_spec_map:
            # Simple spec list below
            spec_list_unique = self.df["spec"].unique()
            # Capitalize the first letter if it is not 'm' and second letter if it is a number
            spec_list = [(spec.capitalize() if spec[0] != "m" else spec[:1] + spec[1:].capitalize()) for spec in
                         spec_list_unique]
            spec_replacement_dict = dict(zip(spec_list_unique, spec_list))
            return spec_replacement_dict

        # spec_map = (('seq', 'Sequence'), ('cover', 'Cover'),('2loop', 'Loop'), ('2branch', 'Branch'), ('2signal', 'Signal'))
        hetero_spec_map = (
            ("mseq", "Seq."), ("mcover", "Cover"), ("m2loop", "Loop"), ("m2branch", "Branch"), ("m2signal", "Signal"),
            ("mseq3-m2branch2-mcover3-m2loop3", "Mixed"),  # mixed struct spec, no signal (2025/07/01)
            ("mseq3-m2branch2-mcover3-m2loop3-m2signal3", "Mixed"))  # mixed struct spec incl. signal
        # Both Mixed variants map to "Mixed"; specs_to_keep picks which one is plotted so they
        # never collide in a single figure.
        hetero_n_goals = (3, 3, 3, 2, 3, '', '')
        # hetero_n_goals = (5, 5, 3, 2, 3)
        homo_spec_map = (
            ("seq", "Seq."), ("cover", "Cover"), ("2loop", "Loop"), ("2branch", "Branch"), ("2signal", "Signal"),)
        homo_n_goals = (3, 3, 3, 2, 3)

        if homo_specs:
            spec_map = homo_spec_map
            n_goals = homo_n_goals
        else:
            spec_map = hetero_spec_map
            n_goals = hetero_n_goals

        spec_replacement_dict = {f"{spec}{n}": f"{Spec}" for (spec, Spec), n in zip(spec_map, n_goals)}
        # spec_replacement_dict = {f"{spec}{n}":f"{Spec}{n}" for (spec,Spec),n in itertools.product(spec_map,n_goals)}
        return spec_replacement_dict

    # @property
    # def max_TtR_value_per_spec(self):
    #     return {"Seq.": 2500}

    # @property
    # def outlier_list_of_dict(self):
    #     return [{"Spec": "Loop", "Planner": "GNN-ODE", "success_mean": 87}]

    # Columns used to identify the exact W&B runs behind a figure (those present are kept).
    # ``id``/``url`` are the eval run's W&B identifiers (use these to re-pull a run);
    # ``wandb_run_id`` is the loaded diffusion-model id (e.g. qkmvppvt), kept as metadata.
    RUN_ID_COLS = ["id", "url", "wandb_run_id", "name", "path", "Planner", "planner",
                   "Spec", "spec", "N", "num_agents", "n_obs", "spec_len",
                   "stl_mixed_spec_mode", "success_mean", "plan_time_mean", "TtR"]

    def _export_repro_artifacts(self, plot_name, new_df, out_dir="barplots"):
        """Dump the artifacts needed to reproduce a figure offline.

        Writes two files next to the figure PDF (``{out_dir}/{plot_name}_{env}.pdf``):

        * ``{plot_name}_{env}_data.csv`` -- the raw per-run dataframe handed to this
          Plotter. ``plot_paper.py --source csv`` reloads it and re-runs the full
          pipeline to regenerate the figure without any W&B access.
        * ``{plot_name}_{env}_run_ids.csv`` -- the exact W&B runs that survive the
          figure's filters (taken from ``make_table``'s post-filter rows), projected to
          the run ``id``/``url`` + key metadata so the precise runs can be re-pulled.

        :param plot_name: figure base name (matches the PDF, e.g. ``single_hetero``)
        :param new_df: post-filter candidate rows returned by :func:`make_table`
        :param out_dir: directory to write into (created if missing)
        :return: (data_csv_path, run_ids_csv_path)
        """
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"{plot_name}_{self.env_name}")

        # {name}_{env}_data.csv -- the full frame that `plot_paper.py --source csv` reloads.
        data_path = f"{base}_data.csv"
        self.df.to_csv(data_path, index=False)

        id_cols = [c for c in self.RUN_ID_COLS if c in new_df.columns]
        mapping = new_df[id_cols].copy()
        sort_cols = [c for c in ["Spec", "Planner", "N"] if c in mapping.columns]
        if sort_cols:
            mapping = mapping.sort_values(sort_cols)
        # {name}_{env}_run_ids.csv -- the W&B run id behind each bar of the figure.
        run_ids_path = f"{base}_run_ids.csv"
        mapping.to_csv(run_ids_path, index=False)

        id_key = next((c for c in ["id", "url", "wandb_run_id"] if c in mapping.columns), None)
        n_runs = mapping[id_key].nunique() if id_key else len(mapping)
        print(f"[repro] {data_path}  ({len(self.df)} rows)")
        print(f"[repro] {run_ids_path}  ({n_runs} unique runs in figure)")
        return data_path, run_ids_path

    def plot_all(self, homo_specs=True, make_bar=False, ablation_mode=False, ode_mode=False, N_large=32, debug=False,
                 return_table=False, columns_to_print=None, N_small=8, no_obs=True, add_row_fn=None,
                 use_stddev=False, additional_planners_to_skip=None, no_highlight_max=False,
                 skip_pick_best_row=False, final_version=False, merge_cols_dict=None,
                 specs_to_keep=None, mixed_spec_choice=None, min_threshold_best_row=0.8,
                 export_data=False, export_plot_name=None, export_dir="barplots"):
        if specs_to_keep is None:
            specs_to_keep = ["m2branch2", "mseq3", "mcover3", "m2loop3"]  # hetero
            specs_to_keep += ["mseq3-m2branch2-mcover3-m2loop3"]  # Add mixed struct spec
            if homo_specs:
                specs_to_keep = ["2branch2", "seq3", "cover3", "2loop3"]  # homo
        spec_replacement_dict = self.get_spec_replacement_dict(homo_specs=homo_specs)
        spec_replacement_dict = {k: v for k, v in spec_replacement_dict.items() if k in specs_to_keep}

        if homo_specs: 
            PLANNERS_TO_SKIP = ["STLPY_SAFE", "STLPY_UREF", "ODE", "DIFFUSION", ]
        else:
            PLANNERS_TO_SKIP = ["GNN-ODE", "STLPY_SAFE", "STLPY_UREF", "ODE", "DIFFUSION", ]
        if additional_planners_to_skip is not None and isinstance(additional_planners_to_skip,list):
            PLANNERS_TO_SKIP += additional_planners_to_skip

        df_to_summarize = self.df.copy()

        use_stddev = make_bar if not use_stddev else use_stddev

        final_table, new_df = make_table(df_to_summarize, spec_replacement_dict=spec_replacement_dict,
                                         use_stddev=use_stddev, ablation_mode=ablation_mode,
                                         add_row_fn=add_row_fn, planners_to_skip=PLANNERS_TO_SKIP, ode_mode=ode_mode,
                                         mixed_spec_mode=not homo_specs, N_large=N_large,
                                         N_small=N_small, no_obs=no_obs,
                                         max_TtR_value_per_spec=self.max_TtR_value_per_spec,
                                         outlier_list_of_dict=self.outlier_list_of_dict, 
                                         columns_to_print=columns_to_print,
                                         final_version=final_version, merge_cols_dict=merge_cols_dict,
                                         skip_pick_best_row=skip_pick_best_row, spec_order=self.spec_order,
                                         mixed_spec_choice=mixed_spec_choice,
                                         min_threshold_best_row=min_threshold_best_row)

        if export_data and export_plot_name is not None:
            # new_df holds the post-filter rows feeding the figure (with wandb_run_id).
            self._export_repro_artifacts(export_plot_name, new_df, out_dir=export_dir)

        if make_bar:
            plot_filename = "homo" if homo_specs else "hetero"
            BarPlots().get_bar_plots(final_table, env_name=self.env_name, plot_name=plot_filename, debug=debug,
                                     spec_order=self.spec_order, out_dir=export_dir)
            # Get tables w/o stddev
            final_table, new_df = make_table(df_to_summarize, spec_replacement_dict=spec_replacement_dict,
                                             use_stddev=False, ablation_mode=ablation_mode,
                                             planners_to_skip=PLANNERS_TO_SKIP, ode_mode=ode_mode,
                                             mixed_spec_mode=not homo_specs, N_large=N_large,
                                             max_TtR_value_per_spec=self.max_TtR_value_per_spec,
                                             outlier_list_of_dict=self.outlier_list_of_dict, 
                                            columns_to_print=columns_to_print, final_version=final_version,
                                            no_obs=no_obs, merge_cols_dict=merge_cols_dict,
                                            skip_pick_best_row=skip_pick_best_row, spec_order=self.spec_order,
                                            mixed_spec_choice=mixed_spec_choice,
                                            min_threshold_best_row=min_threshold_best_row)
            use_stddev = False

        if return_table:
            return final_table

        pivoted_table, latex_code = PivotedTableStyler().format_for_latex(final_table, highlight_max=not use_stddev and not no_highlight_max)
        return pivoted_table, latex_code

    def plot_single_spec_grouped(self, homo_mode=True, debug=False, plot_grouped=None,
                                 export_data=False, export_dir="barplots", plot_name=None, **kwargs):
        """Plot either homogeneous or heterogeneous specs grouped by N values

        :param homo_mode: If True, plot homogeneous specs; if False, plot heterogeneous specs
        :param debug: Debug Mode
        :param plot_grouped: List of metrics substrs to plot (if not None, will plot only these metrics)
        :param export_data: If True, also dump the offline-repro CSV + wandb run-id mapping
            (see :meth:`_export_repro_artifacts`) keyed to this figure's name
        :param export_dir: Directory for the exported artifacts (defaults next to the PDF)
        :param plot_name: Output basename for the PDF + sidecar CSVs; defaults to
            ``single_{homo|hetero}`` when not given (so distinct figures don't collide)
        """
        # A DIFF-SA entry in merge_cols_dict means DIFF-SA was folded into DIFF-MA; hide its bar.
        merge_cols_dict = kwargs.get('merge_cols_dict', {})
        skip_sa_planner = 'DIFF-SA' in merge_cols_dict

        if plot_name is None:
            plot_name = f"single_{'homo' if homo_mode else 'hetero'}"

        # 1 / collect the table for the selected mode
        tbl_single = self.plot_all(homo_specs=homo_mode, make_bar=False, return_table=True, use_stddev=True,
                                   export_data=export_data, export_plot_name=plot_name, export_dir=export_dir,
                                   **kwargs).reset_index()

        # 2 / add the "Mode" tag and build the facet key
        mode_label = "Homo" if homo_mode else "Hetero"
        tbl_single = tbl_single.assign(Mode=mode_label)
        tbl_single = tbl_single.assign(col_key=lambda d: d["Mode"].str[:2] + ". N=" + d["N"].astype(str))

        # 3 / pass straight into the existing BarPlots pipeline
        BarPlots(skip_sa_planner).get_bar_plots(tbl_single, env_name=self.env_name, plot_name=plot_name, debug=debug,
                                 combo_mode=False, plot_grouped=plot_grouped, spec_order=self.spec_order,
                                 out_dir=export_dir)
