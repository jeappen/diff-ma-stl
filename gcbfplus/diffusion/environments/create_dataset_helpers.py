"""Functions for loading and processing datasets from the GCBF+ code."""

import datetime
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('agg')  # Use this before importing pyplot (non-interactive)
from matplotlib import pyplot as plt

from gcbfplus.env.plot import get_obs_collection
from gcbfplus.utils.utils import tree_index, jtu

DATASET_DIR = "datasets"


class RolloutToDataset:
    """Class to convert rollouts to a dataset compatible with D4RL.

    Converts a list of jax rollouts to a d4rl dataset to be used by the diffusion models.
    The dataset is saved to a file in the specified directory. Some helper functions
    to visualize the dataset coverage are also provided.
    """

    def __init__(self, rollouts, env_id=None, predicate_centers=None, env=None):
        self.rollouts = rollouts
        self.env_id = env_id
        self.predicate_centers = predicate_centers
        self.env = env

    @staticmethod
    def cat_rollout_entries(*_rollouts):
        """Stack entries in the list of rollouts"""
        return jnp.concatenate(_rollouts)

    @staticmethod
    def stack_rollout_entries(*_rollouts):
        """Stack entries in the list of rollouts"""
        # Requires equal-length rollouts (no padding).
        return jnp.stack(_rollouts)

    @property
    def key_dict_map(self):
        return dict(zip(["Tp1_graph", "T_action", "T_reward", "T_done", "T_info"],
                        ["observations", "actions", "rewards", "terminals", "timeouts"], ))

    def save_dataset(self, named_tuple, file_name):
        """Save a dataset to a file"""
        hdf5_file_path = f"{file_name}.h5"
        os.makedirs(os.path.dirname(hdf5_file_path) or ".", exist_ok=True)
        with h5py.File(hdf5_file_path, 'w') as hf:
            for i, array in enumerate(named_tuple):
                # Convert JAX array to NumPy array
                if not (isinstance(array, jnp.ndarray) or isinstance(array, np.ndarray)) or \
                        not (named_tuple._fields[i] in self.key_dict_map):
                    continue
                    # skip if not a jax array
                numpy_array = np.array(array)
                # Create a dataset for each array in the tuple
                hf.create_dataset(f'{self.key_dict_map[named_tuple._fields[i]]}'
                                  , data=numpy_array)

        print(f"Dataset saved to {hdf5_file_path}")

    @staticmethod
    def _get_dataset_coverage_plot(all_traj, predicate_centers, rollouts=None, plot_name="",
                                   number_of_trajectories=100, env=None):
        from matplotlib.colors import to_rgba
        plt.clf()
        colors = plt.cm.tab10(np.linspace(0, 1, len(predicate_centers)))
        goal_color = "GREEN"
        import string

        # randomly sample 1000 trajectories
        if len(all_traj) > number_of_trajectories:
            all_traj = np.random.choice(all_traj, number_of_trajectories, replace=False)

        for traj in all_traj:
            traj = traj[:, :2]
            plt.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5)
            # Also mark the starting point with a red dot
            plt.plot(traj[0][0], traj[0][1], 'rx', zorder=3)
        plt.title('Trajectories Over Time')

        ax = plt.gca()
        pred_size = np.array([1, 1])
        shifted_cents = [x - (pred_size / 2) for x in predicate_centers]

        # For consistency with other plots we want to sort the centers similarly
        def sort_by_center(centers, y_tol=1e-8):
            """Sort centers by y-coordinate and then by x-coordinate"""
            return sorted(centers, key=lambda c: (round(c[1] / y_tol) * y_tol, c[0]))

        sorted_shifted_cents = sort_by_center(shifted_cents)
        for i, center in enumerate(sorted_shifted_cents):
            rect = plt.Rectangle(center, pred_size[0], pred_size[1], edgecolor='gray',
                                 facecolor=to_rgba(goal_color, 0.4),
                                 linewidth=2, label=f'Predicate {i}')
            ax.add_patch(rect)
            center_x = rect.get_x() + rect.get_width() / 2
            center_y = rect.get_y() + rect.get_height() / 2
            pred_txt = string.ascii_uppercase[i]  # Use letters for predicates
            # Add a text box in the center of the circle
            ax.text(center_x, center_y, pred_txt, ha='center', va='center', fontsize=48, color='gray', alpha=0.9,
                    zorder=2)

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obs_color = "#8a0000"
        edge_goal_color = goal_color

        if rollouts is not None:
            # Plot obstacles in this fig now (copied from render_video)
            # plot the first frame
            T_graph = rollouts[0].Tp1_graph
            graph0 = tree_index(T_graph, 0)

            # plot obstacles
            obs = graph0.env_states.obstacle
            ax.add_collection(get_obs_collection(obs, obs_color, alpha=0.8))

        plt.title('Trajectory Distribution with Predicates')
        plt.axis('equal')
        # Turn off axis
        plt.axis('off')
        ax.set_aspect('equal', adjustable='box')
        # Frame the workspace with a half-cell margin. Falls back to the 4x4 arena
        # the original datasets were generated in when no env is supplied.
        area_size = env.area_size if env is not None else 4.0
        plt.xlim(-0.5, area_size + 0.5)
        plt.ylim(-0.5, area_size + 0.5)
        plt.tight_layout()
        plt.savefig(f"{plot_name}dataset_coverage.png", dpi=300, bbox_inches="tight")

    @staticmethod
    def _get_dataset_coverage_bar(predicate_centers, predicate_percentages, plot_name=""):
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(predicate_centers)), predicate_percentages)
        plt.title('Percentage of Trajectories Reaching Each Predicate')
        plt.xlabel('Predicate Index')
        plt.ylabel('Percentage of Trajectories (%)')
        plt.ylim(0, 100)
        plt.xticks(range(len(predicate_centers)), [f'Predicate {i}' for i in range(len(predicate_centers))])
        plt.grid(True)
        plt.savefig(f"{plot_name}dataset_percent_coverage_bar.png", dpi=300)

    @staticmethod
    def _get_dataset_coverage(all_traj, predicate_centers, plot_name=""):
        """Get the coverage of the dataset based on distance to the predicates"""
        # Speedy calculation
        predicate_counts = jnp.zeros(len(predicate_centers))
        dist_threshold = 0.5

        def calc_pred(predicate_counts, ep):
            traj = ep[:, :2]
            pairwise_diff = traj[:, np.newaxis, :] - jnp.array(predicate_centers)[np.newaxis, :, :]
            predicate_counts += (jnp.linalg.norm(pairwise_diff, axis=2, ord=1) < dist_threshold).any(axis=0)
            return predicate_counts, len(ep)

        scan_output = jax.lax.scan(calc_pred, jnp.zeros(len(predicate_centers)), jnp.array(all_traj))
        predicate_counts = scan_output[0]
        predicate_percentages = (predicate_counts / len(all_traj)) * 100

        for i, percent in enumerate(predicate_percentages):
            print(f"Predicate {i}: {percent:.2f}% of trajectories reach this goal.")
        RolloutToDataset._get_dataset_coverage_bar(predicate_centers, predicate_percentages, plot_name=plot_name)

    @staticmethod
    def prepare_rollouts_for_dataset(rollouts):
        """Prepare rollouts for saving to a dataset"""

        # Add entry to the rollout named tuple called timeout in place of infos
        # Replace last entry with True
        def set_timeout_and_done(rollout):
            # NOTE: all done flags are cleared and one timeout is set at the last step,
            # so each rollout is exactly one episode for _assemble_dataset.
            rollout = rollout._replace(T_done=jnp.zeros_like(rollout.T_done))
            # Also set one done and truncate the
            timeout = jax.lax.cond(rollout.T_done[-1], lambda done: jnp.zeros_like(done),
                                   lambda done: jnp.zeros_like(done).at[-1].set(True), rollout.T_done)
            rollout = rollout._replace(T_done=timeout)  # NOTE: Ignoring the done for now

            return rollout._replace(T_info=rollout.Tp1_graph.env_states.goal.squeeze(1))  # Add goals to info

        rollouts = [set_timeout_and_done(rollout) for rollout in rollouts]
        # Stack entries in the list of rollouts
        # Replace Tp1_graph with the agent state
        rollouts = [rollout._replace(Tp1_graph=rollout.Tp1_graph.env_states.agent.squeeze(1)) for
                    rollout in rollouts]
        # Squeeze actions to remove the agent dimension
        rollouts = [rollout._replace(T_action=rollout.T_action.squeeze(1)) for rollout in
                    rollouts]
        return rollouts

    def rollouts_to_d4rl_dataset(self, rollouts=None, env_id=None, predicate_centers=None, env=None):
        """Converts a list of jax rollouts to a d4rl dataset to be used by the diffusion models"""

        if rollouts is None:
            rollouts = self.rollouts
        assert rollouts[0].T_action.shape[1] == 1, "single-agent rollouts only"
        if env_id is None:
            env_id = self.env_id
        if predicate_centers is None:
            predicate_centers = self.predicate_centers
        if env is None:
            env = self.env

        # join dataset dir with env_id
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if env is not None and env.map_name is not None and env.map_name != "empty":
            env_id = f"{env_id}_{env.map_name}"
        filename = f"{env_id}_N{len(rollouts)}_{current_time}"
        full_path_to_file = f"{DATASET_DIR}/{filename}"

        formatted_rollouts = self.prepare_rollouts_for_dataset(rollouts)

        # Stack entries in the list of rollouts
        rollouts_cat = jtu.tree_map(self.cat_rollout_entries, *formatted_rollouts)
        self.save_dataset(rollouts_cat, full_path_to_file)

        # If observations, calculate coverage (after saving the dataset)
        if predicate_centers is not None:
            print(f"Calculating coverage for dataset {full_path_to_file}")
            rollouts_stack = jtu.tree_map(self.stack_rollout_entries, *formatted_rollouts)
            self._get_dataset_coverage(rollouts_stack.Tp1_graph, predicate_centers, plot_name=full_path_to_file)
            self._get_dataset_coverage_plot(rollouts_stack.Tp1_graph, predicate_centers, rollouts=rollouts,
                                            plot_name=full_path_to_file, env=env)

        print("Dataset saved successfully")
