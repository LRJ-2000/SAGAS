import os
import time
import ogbenchTL
import gymnasium
import numpy as np

from O_utils.datasets import Dataset


def dataset_directory_options():
    path = os.environ.get("SAGAS_DATA_DIR")
    return {"dataset_dir": path} if path else {}


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()

    def _reset_stats(self):
        """Reset episode statistics."""
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def set_max_episode_steps(self, max_episode_steps):
        self.unwrapped.max_episode_steps = max_episode_steps

    def step(self, action):
        """Step and log episodic metrics."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.reward_sum += reward
        self.episode_length += 1

        if terminated or truncated:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            if hasattr(self.unwrapped, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.unwrapped.get_normalized_score(info["episode"]["return"])
                    * 100.0
                )
            else:
                info["episode"]["normalized_return"] = info["episode"]["return"] * 100.0
            info["episode"]["success"] = float(
                np.isclose(info["episode"]["normalized_return"], 100.0, rtol=1e-10)
            )
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        """Reset environment and episode stats."""
        seed = kwargs.pop("seed", None)
        if seed is not None:
            np.random.seed(seed)
            self.env.np_random = np.random.RandomState(seed)
            self.env._np_random_seed = seed
        self._reset_stats()
        return self.env.reset(*args, **kwargs)

    def get_maze_map(self):
        return self.unwrapped.maze_map

    def ij_to_xy(self, i, j):
        x, y = self.unwrapped.ij_to_xy((i, j))
        return x, y

    def xy_to_ij(self, x, y):
        i, j = self.unwrapped.xy_to_ij((x, y))
        return i, j

    def get_xy(self):
        x, y = self.unwrapped.get_xy()
        return x, y

    def get_maze_params(self):
        return {
            "offset_x": self.unwrapped._offset_x,
            "offset_y": self.unwrapped._offset_y,
            "maze_unit": self.unwrapped._maze_unit,
        }


def make_env_and_datasets(env_name, seed, size=480):
    """Make OGBench environment and datasets."""
    env, train_dataset, val_dataset = ogbenchTL.make_env_and_datasets(
        env_name, compact_dataset=False, width=size, height=size,
        **dataset_directory_options(),
    )
    env = EpisodeMonitor(env)
    observation, info = env.reset(seed=seed)

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    return env, train_dataset, val_dataset


def make_ltl_env_and_datasets(
    env_name, seed, size=480, region_type="circle", max_regions=5
):
    """Make LTL OGBench environment and datasets."""
    from ogbenchTL.locomaze.maze import MAZE_MAPS

    env_map_type = env_name.split("-")[1]
    assert (
        env_map_type in MAZE_MAPS.keys()
    ), f"Unsupported maze map type: {env_map_type}"
    maze_map = np.array(MAZE_MAPS[env_map_type])
    height = size
    ratio = maze_map.shape[1] / maze_map.shape[0]
    width = int(height * ratio)

    env, train_dataset, val_dataset = ogbenchTL.make_env_and_datasets(
        env_name,
        compact_dataset=False,
        height=height,
        width=width,
        region_type=region_type,
        max_regions=max_regions,
        **dataset_directory_options(),
    )
    env = EpisodeMonitor(env)
    env.reset(seed=seed)

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    return env, train_dataset, val_dataset
