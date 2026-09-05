"""For loading results from wandb."""

import json
import os
import tempfile

import pandas as pd
import wandb


class WandbLoader:
    """Class to load wandb results."""

    # Scratch space for downloaded artifact tables and the per-project dump written by
    # summarize_loaded_runs. Under the platform temp dir so nothing lands in the cwd.
    TMP_FOLDER = os.path.join(tempfile.gettempdir(), "wandb-artifacts-dl")

    def __init__(
        self,
        # Placeholder defaults: every caller in this repo passes its own entity/project.
        project_name: str = "my-project",
        entity: str = "my-team",
        filter_dict: dict = None,
        preload_filter_dict: dict = None,  # To filter runs before loading
        download_tables_and_summarize: bool = False,
        tables_to_check: dict = None,
        debug: bool = False,
    ):
        """Initialize the loader.

        Args:
            project_name: Name of the wandb project.
            entity: Entity name for wandb.
            filter_dict: Dictionary to filter runs.
                Example: {<config entry>: (<value>, <comparison operator>)}
                The comparison operator can be one of the following:
                - '==': Equal to
                - '!=': Not equal to
                - '<', '<=': Less than, less than or equal to
                - '>', '>=': Greater than, greater than or equal to
                - 'in': In a list of values
            preload_filter_dict: Dictionary to filter runs before loading (following MongoDB-like syntax).
                Example: {"config.<config entry>": {"$eq": <value>}}
            download_tables_and_summarize: Whether to download tables and summarize them.
        """
        self.project_name = project_name
        self.entity = entity
        self.runs = None
        self.filter_dict = filter_dict
        self.filtered_runs = None
        self.download_tables_and_summarize = download_tables_and_summarize
        if tables_to_check is None:
            self.tables_to_check = {('diffusion_method', 'edm'): 'num_resampling_iters',
                                ('diffusion_method', 'edm-ma'): 'cum_agent_mask',
                            }

        # Load the runs
        self._load(debug=debug, preload_filter_dict=preload_filter_dict)

    def _load(self, debug=False, preload_filter_dict=None):
        """Load wandb results."""
        self.api = wandb.Api(timeout=29)  # Increased timeout since some runs take longer to load

        # Project is specified by <entity/project-name>
        self.runs = self.api.runs(f"{self.entity}/{self.project_name}",
                                  filters=preload_filter_dict)
        # Filter out runs that are not finished
        self.runs = [run for run in self.runs if run.state == "finished"]
        if self.download_tables_and_summarize:
            # Download tables in preparation for summarizing
            self._download_tables(debug=debug)

    def _download_tables(self,debug=False):
        """Download tables from wandb runs."""
        # if TMP_FOLDER does not exist, create it
        if not os.path.exists(self.TMP_FOLDER):
            os.makedirs(self.TMP_FOLDER)
        for run in self.runs:
            # Download the tables
            for (config_key, config_val), artifact_name in self.tables_to_check.items():
                # Check if the table exists
                if config_key in run.config and run.config[config_key] == config_val and run.summary.get(artifact_name) is not None:
                    # Check if the artifact exists and download the table
                    artifact_name = f"{self.entity}/{self.project_name}/run-{run.id}-{artifact_name}:latest"
                    artifact = self.api.artifact(artifact_name)
                    # Check if the artifact is already downloaded
                    download_path = os.path.join(self.TMP_FOLDER, run.id)
                    if not os.path.exists(download_path):
                        os.makedirs(download_path)
                    # if download path is empty, download the artifact
                    if len(os.listdir(download_path)) == 0:
                        artifact.download(download_path)
                        if debug:
                            print(f"Downloaded {artifact_name} for run {run.id}")
                else:
                    if debug:
                        print(f"Skipping {artifact_name} for run {run.id} as config {config_key} is not {config_val}")
    
    @staticmethod
    def load_json(file_path):
        # Load the JSON file
        with open(file_path, 'r') as f:
            table_json = json.load(f)

        # Extract columns and data
        columns = table_json['columns']
        data = table_json['data']

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        return df
        
    @staticmethod
    def summarize_artifact(artifact_df, table_name):
        """Summarize the artifact."""
        # Get max, min, mean across agents
        max_df = artifact_df.max(axis=0)
        min_df = artifact_df.min(axis=0)
        mean_df = artifact_df.mean(axis=0)
        dfs = {"max": max_df, "min": min_df, "mean": mean_df}
        # Now get max, min, mean and std across all rows
        summary_dict = {}
        for df_name, df in dfs.items():
            key_root = f"{table_name}_{df_name}"
            update_dict = {f"{key_root}_max": df.max(), f"{key_root}_min": df.min(), f"{key_root}_mean": df.mean(), f"{key_root}_std": df.std()}
            summary_dict.update(update_dict)
        return summary_dict


    def  filter_runs(self, filter_dict=None):
        """Filter runs based on the filter_dict."""
        if filter_dict is None:
            filter_dict = self.filter_dict
        if filter_dict is None:
            return self.runs

        filtered_runs = []
        for run in self.runs:
            # Check if all filters are satisfied
            if all(
                self._check_filter(run, key, value)
                for key, value in filter_dict.items()
            ):
                filtered_runs.append(run)
        return filtered_runs

    def _check_filter(self, run, key, value):
        """See if the run satisfies the filter with key and value (operator, value)."""
        # Get the value from the run
        run_value = run.config.get(key) or run.summary.get(key)
        if run_value is None:
            return False

        # Check the operator
        operator, filter_value = value
        if operator == "==":
            return run_value == filter_value
        elif operator == "!=":
            return run_value != filter_value
        elif operator == "<":
            return run_value < filter_value
        elif operator == "<=":
            return run_value <= filter_value
        elif operator == ">":
            return run_value > filter_value
        elif operator == ">=":
            return run_value >= filter_value
        elif operator == "in":
            return run_value in filter_value
        else:
            raise ValueError(f"Invalid operator: {operator}")

    def _load_filtered(self, runs=None, filter_dict=None):
        if runs is None:
            runs = self.filter_runs(filter_dict=filter_dict)

        # Create a DataFrame to store the results
        if len(runs) == 0:
            print("No runs found.")
            return None

        # Create a DataFrame with the summary, config, and name of each run
        summary_list, config_list, name_list = [], [], []
        for run in runs:
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)

            # Add table summaries to the summary
            if self.download_tables_and_summarize:
                download_path = os.path.join(self.TMP_FOLDER, run.id)
                if os.path.exists(download_path):
                    # check files in download_path
                    for file in os.listdir(download_path):
                        if file.endswith(".table.json"):
                            # Load the JSON file
                            table_df = self.load_json(os.path.join(download_path, file))
                            table_name = file.split(".")[0]
                            # Add the summary to the run
                            table_summary_dict = self.summarize_artifact(table_df, table_name)
                            summary_list[-1].update(table_summary_dict)

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )

            # .name is the human-readable name of the run.
            name_list.append(run.name)

            # Add id and url to the config
            config_list[-1]["id"] = run.id
            config_list[-1]["url"] = run.url


        # Create a DataFrame with the summary, config, and name of each run with the columns
        #  'summary/<summary_key>', 'config/<config_key>', and 'name'
        runs_df = pd.DataFrame(summary_list)
        config_df = pd.DataFrame(config_list)
        runs_df = pd.concat([runs_df, config_df], axis=1)
        # Strip the 'eval/' prefix so the columns match the local test_log.csv schema.
        runs_df.columns = [col.replace("eval/", "") for col in runs_df.columns]
        runs_df["name"] = name_list
        runs_df = self.rm_columns(runs_df, keep_always=self.COLUMNS_TO_KEEP_ALWAYS)

        # Remove columns with more than 90% NaN values
        runs_df, mask = self.drop_nan_col(
            runs_df, threshold=0.90, keep_always=self.COLUMNS_TO_KEEP_ALWAYS
        )

        # Rescale the columns by 100
        runs_df = self.rescale_columns(runs_df)

        return runs_df
    
    def rescale_columns(self, df, columns=None):
        """Rescale the columns by 100."""
        if columns is None:
            # finish_rate stays 0-1 (only safe/success are rescaled to %).
            column_names = ['safe', 'success']
            columns = list(
                filter(lambda x: any(col in x for col in column_names), df.columns)
            )
        
        # Rescale the columns by 100
        for col in columns:
            if col.endswith("_mean"):
                df[col] = df[col] * 100
            elif col.endswith("_std"):
                df[col] = df[col] * 100
            else:
                print(f"Skipping column {col} as it does not end with _mean or _std")

        return df

    COLUMNS_TO_KEEP_ALWAYS = ["ma_stl_satisfaction", "ma_stl_satisfaction_std", "stl_mixed_spec_mode"]

    @staticmethod
    def drop_nan_col(df, threshold=0.90, keep_always=None):
        # 1. Compute the fraction of NaNs in each column (by position).
        na_fraction = df.isna().mean(
            axis=0
        )  # Returns a Series indexed by column labels

        # 2. Construct a boolean mask: True for columns to *keep* (NaN fraction ≤ threshold)
        mask = (na_fraction <= threshold) | (df.columns.isin(keep_always))

        # 3. Slice the DataFrame by column index positions matching 'mask'.
        #    Using .values converts the mask to a NumPy array, so columns are sliced purely by index.
        df = df.iloc[:, mask.values]

        # 4. Lastly merge columns with the same name
        df = df.groupby(df.columns, axis=1).first()

        return df, mask

    @staticmethod
    def rm_columns(df, k_not_nan=5, keep_always=None):
        """Remove columns with less than k_not_nan non-NaN values."""
        # Get the columns with at least k_not_nan non-NaN values
        cols = df.columns[
            (df.notna().sum() >= k_not_nan) | (df.columns.isin(keep_always))
        ]
        return df[cols]

    def summarize_loaded_runs(self):
        """Summarize the loaded runs."""
        # Filter the runs based on the filter_dict
        if self.filtered_runs is None:
            print("Loading filtered runs...")
            self.filtered_runs = self._load_filtered()
        else:
            print("Filtered runs already loaded with len:", len(self.filtered_runs))

        runs_df = self.filtered_runs

        # Dump of what was loaded, for inspection only -- nothing reads it back. Written under
        # TMP_FOLDER so a pull never drops a stray CSV into the caller's working directory.
        os.makedirs(self.TMP_FOLDER, exist_ok=True)
        filtered_runs_filename = os.path.join(self.TMP_FOLDER, f"filtered_runs_{self.project_name}.csv")
        runs_df.to_csv(filtered_runs_filename)

        return runs_df
