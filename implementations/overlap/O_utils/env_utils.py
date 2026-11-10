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
        max_episode_steps = int(max_episode_steps)
        self.unwrapped.max_episode_steps = max_episode_steps
        if hasattr(self.env, "_max_episode_steps"):
            self.env._max_episode_steps = max_episode_steps
        if getattr(self.env, "spec", None) is not None:
            self.env.spec.max_episode_steps = max_episode_steps

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


def make_env_and_datasets(env_name, seed, size=480, add_info=False):
    """Make OGBench environment and datasets."""
    env, train_dataset, val_dataset = ogbenchTL.make_env_and_datasets(
        env_name,
        compact_dataset=False,
        width=size,
        height=size,
        add_info=add_info,
        **dataset_directory_options(),
    )
    env = EpisodeMonitor(env)
    observation, info = env.reset(seed=seed)

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    return env, train_dataset, val_dataset


def maze_map_type_from_env_name(env_name):
    """Extract the OGBench locomaze map-size token from state or visual env names."""
    parts = str(env_name).split("-")
    if parts and parts[0] == "visual":
        if len(parts) < 3:
            raise ValueError(f"Cannot infer maze map type from env_name={env_name!r}")
        return parts[2]
    if len(parts) < 2:
        raise ValueError(f"Cannot infer maze map type from env_name={env_name!r}")
    return parts[1]


def ltl_env_render_dimensions(env_name, maze_map_shape, size):
    """Return width/height for LTL env creation without changing visual checkpoint input shape."""
    height = int(size)
    if str(env_name).startswith("visual-"):
        return height, height
    ratio = float(maze_map_shape[1]) / float(maze_map_shape[0])
    width = int(height * ratio)
    return width, height


def make_ltl_env_and_datasets(
    env_name,
    seed,
    size=480,
    region_type="circle",
    max_regions=5,
    add_info=False,
):
    """Make LTL OGBench environment and datasets."""
    from ogbenchTL.locomaze.maze import MAZE_MAPS

    env_map_type = maze_map_type_from_env_name(env_name)
    assert (
        env_map_type in MAZE_MAPS.keys()
    ), f"Unsupported maze map type: {env_map_type}"
    maze_map = np.array(MAZE_MAPS[env_map_type])
    width, height = ltl_env_render_dimensions(env_name, maze_map.shape, size)

    env, train_dataset, val_dataset = ogbenchTL.make_env_and_datasets(
        env_name,
        compact_dataset=False,
        height=height,
        width=width,
        region_type=region_type,
        max_regions=max_regions,
        add_info=add_info,
        **dataset_directory_options(),
    )
    env = EpisodeMonitor(env)
    env.reset(seed=seed)

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    return env, train_dataset, val_dataset
