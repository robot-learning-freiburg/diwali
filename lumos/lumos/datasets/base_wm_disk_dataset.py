from bisect import bisect_right
from itertools import chain
import logging
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from lumos.datasets.base_dataset import BaseDataset
from lumos.datasets.utils.episode_utils import lookup_naming_pattern

logger = logging.getLogger(__name__)

def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class BaseWMDiskDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in dataset_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        mask_all_actions: bool = False,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        use_cached_data: bool = False,
        index_file: str = "ep_start_end_ids.npy",
        cache_file: str = "cached_data.pkl",
        action_chunk_size: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            action_chunk_size=action_chunk_size,
            **kwargs,
        )
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.mask_all_actions = mask_all_actions
        self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.action_chunk_size = action_chunk_size

        self.episode_lookup, self.episode_length, self.start_ids = self._build_file_indices(self.abs_dataset_dir, index_file)
        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_dataset_dir, self.save_format)
        self.naming_pattern_action_chunk, self.n_digits_action_chunk = lookup_naming_pattern(self.abs_action_chunk_data_dir, self.save_format)
        if self.points_displacement:
            self.naming_pattern_p_disp, self.n_digits_p_disp = lookup_naming_pattern(self.abs_points_displacement_data_dir, self.save_format)
        
        self.use_cached_data = use_cached_data
        if self.use_cached_data:
            self.preloaded_data = {}
            self.preloaded_data_action_chunk = {}
            self.preloaded_data_p_disp = {}
            self.preload_dataset(index_file, cache_file)

        self.INF_ACTION = np.full(7 * action_chunk_size, np.inf, dtype=np.float64)

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.

        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")
    
    def _get_episode_name_action_chunk(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.

        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern_action_chunk[0]}{file_idx:0{self.n_digits_action_chunk}d}{self.naming_pattern_action_chunk[1]}")

    def _get_episode_name_p_disp(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.


        Returns:
            Path to file.
        """
        assert self.abs_points_displacement_data_dir is not None, "Points displacement data directory is not provided."
        return Path(f"{self.naming_pattern_p_disp[0]}{file_idx:0{self.n_digits_p_disp}d}{self.naming_pattern_p_disp[1]}")

    def _load_episode(self, start_idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            start_idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("rel_actions")
        keys = list(dict.fromkeys(keys))

        episode = {
            'rgb_static': None,
            'rgb_gripper': None,
            'optical_flow_static': None,
            'optical_flow_gripper': None,
            'robot_obs': None,
            'scene_obs': None,
            'pre_actions': None,
            'reset': None,
            'frame': np.arange(start_idx, end_idx, self.action_chunk_size, dtype=np.int32)[:, np.newaxis],
            'p_disp': None,
        }
        
        is_episode_start = False
        if (start_idx - self.action_chunk_size) < self.episode_lookup.min():
            is_episode_start = True
        for idx in range(start_idx, start_idx - self.action_chunk_size, -1):
            is_episode_start |= idx in self.start_ids

        ep_start_idx, ep_end_idx = self._get_episode_bounds(start_idx)
        if is_episode_start:
            episodes, pre_action_chunks, points_disp = self.zip_sequence(start_idx, end_idx, self.action_chunk_size, ep_start_idx, ep_end_idx)
            
            # reorder episodes by labels
            episode.update({key: np.stack([ep[key] for ep in episodes]) for key in keys})
            # repeat the first frame as previous frame
            for key in keys:
                if "rgb" in key or "obs" in key:
                    episode[key] = np.concatenate([episode[key][0:1], episode[key]], axis=0)

            # random masking the GT action
            if not self.mask_all_actions:
                # "rel_actions" here are pre-action chunks so no need to roll them.
                episode["pre_actions"] = np.stack([ac["rel_actions"].reshape(-1) for ac in pre_action_chunks])
            else:
                episode["pre_actions"] = np.stack([self.INF_ACTION.copy() for _ in range(window_size // self.action_chunk_size)], axis=0)

            if self.points_displacement:
                episode["p_disp"] = {key: np.stack([pd[key] for pd in points_disp]) for key in self.observation_space["p_disp_obs"]}
        else:
            episodes, pre_action_chunks, points_disp = self.zip_sequence(start_idx - self.action_chunk_size, end_idx, self.action_chunk_size, ep_start_idx, ep_end_idx)

            for key in keys:
                if "rgb" in key or "obs" in key:
                    episode[key] = np.stack([ep[key] for ep in episodes])
                else:
                    episode[key] = np.stack([ep[key] for ep in episodes[1:]])

            if not self.mask_all_actions:
                episode["pre_actions"] = np.stack([ac["rel_actions"].reshape(-1) for ac in pre_action_chunks[1:]])
            else:
                episode["pre_actions"] = np.stack([self.INF_ACTION.copy() for _ in range(window_size // self.action_chunk_size)], axis=0)

            if self.points_displacement:
                episode["p_disp"] = {key: np.stack([pd[key] for pd in points_disp[1:]]) for key in self.observation_space["p_disp_obs"]}

        resets = np.zeros((episode["pre_actions"].shape[0], 1), dtype=bool)
        if self.reset_prob != 0:
            resets = np.random.rand(episode["pre_actions"].shape[0], 1) <= self.reset_prob
        if is_episode_start:
            resets[0] = True
        episode["reset"] = resets

        return episode

    def _build_file_indices(self, abs_dataset_dir: Path, index_file: str) -> tuple[np.ndarray, np.ndarray, List[int]]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_dataset_dir: Absolute path of the directory containing the dataset.
            index_file: Name of the index file containing start and end indices of episodes.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            start_ids: List of starting indices of episodes.
        """
        assert abs_dataset_dir.is_dir()

        episode_lookup = []
        remain_length = []
        start_ids = []
        end_ids = []
        max_episode_len = 0
        n_padded_episodes = 0

        ep_start_end_ids = np.load(abs_dataset_dir / index_file)
        logger.info(f'Found {index_file} with {len(ep_start_end_ids)} episodes.')
        assert len(ep_start_end_ids) > 0, "No episodes found in the dataset directory."

        for start_idx, end_idx in ep_start_end_ids:
            if end_idx < start_idx:
                logger.info(f"Episode from {start_idx} to {end_idx} is empty. Skipping.")
                continue

            end_idx_sample = end_idx + self.max_pad_size - self.max_window_size + 1
            if end_idx_sample < start_idx:
                logger.info(
                    f"Episode from {start_idx} to {end_idx} is shorter than the window. "
                    "Padding with first/last-frame hold (zero actions, gripper held)."
                )
                end_idx_sample = start_idx
                n_padded_episodes += 1

            start_ids.append(start_idx)
            end_ids.append(end_idx)
            episode_lookup.extend(range(start_idx, end_idx_sample + 1))
            # Report at least max_window_size remaining so the loader requests a full
            # window; zip_sequence holds the boundary frame for indices past the end.
            remain_length.extend(
                max(remain, self.max_window_size)
                for remain in range(end_idx - start_idx + 1, end_idx - end_idx_sample, -1)
            )
            max_episode_len = max(max_episode_len, end_idx - start_idx + 1)

        self.end_ids = end_ids
        self.max_episode_len = max_episode_len
        logger.info(f"Max episode length: {max_episode_len}")
        logger.info(f"Padded {n_padded_episodes} short episodes with first/last-frame hold.")
        logger.info(f"Total number of frames: {len(episode_lookup)}")

        return np.array(episode_lookup), np.array(remain_length), start_ids 

    def preload_dataset(self, index_file: str, cache_file: str) -> None:
        """Preloads the entire dataset into memory."""
        if (self.abs_dataset_dir / cache_file).exists():
            logger.info(f"Loading preloaded dataset from {cache_file}")
            self.preloaded_data = load_pkl(self.abs_dataset_dir / cache_file)
        else:
            ep_start_end_ids = np.load(self.abs_dataset_dir / index_file)
            for start_idx, end_idx in tqdm(ep_start_end_ids, desc="Preloading dataset"):
                for idx in range(start_idx, end_idx):
                    file_name = self._get_episode_name(idx)
                    np_obj = load_npz(file_name)
                    data = {key: np.stack([np_obj[key]]) for key, _ in np_obj.items()}

                    value = {
                        "robot_obs": np.squeeze(data["robot_obs"]),
                        "rgb_static": np.squeeze(data["rgb_static"]),
                        "rgb_gripper": np.squeeze(data["rgb_gripper"]),
                    }
                    self.preloaded_data[idx] = value
            logger.info("Preloaded the dataset into cache.")

            with open(self.abs_dataset_dir / cache_file, "wb") as f:
                pickle.dump(self.preloaded_data, f)
            logger.info(f"Saved preloaded dataset to {cache_file}")

        if (self.abs_action_chunk_data_dir / cache_file).exists():
            logger.info(f"Loading preloaded action chunk dataset from {cache_file}")
            self.preloaded_data_action_chunk = load_pkl(self.abs_action_chunk_data_dir / cache_file)
        else:
            ep_start_end_ids = np.load(self.abs_action_chunk_data_dir / index_file)
            for start_idx, end_idx in tqdm(ep_start_end_ids, desc="Preloading action chunk dataset"):
                for idx in range(start_idx, end_idx):
                    file_name = self._get_episode_name_action_chunk(idx)
                    np_obj = load_npz(file_name)

                    value = {
                        "rel_actions": np.squeeze(np.stack([np_obj["rel_actions"]])),
                    }
                    self.preloaded_data_action_chunk[idx] = value
            logger.info("Preloaded the action chunk dataset into cache.")

            with open(self.abs_action_chunk_data_dir / cache_file, "wb") as f:
                pickle.dump(self.preloaded_data_action_chunk, f)
            logger.info(f"Saved preloaded action chunk dataset to {cache_file}")

        if not self.points_displacement:
            return

        if (self.abs_points_displacement_data_dir / cache_file).exists():
            logger.info(f"Loading preloaded points displacement data from {cache_file}")
            self.preloaded_data_p_disp = load_pkl(self.abs_points_displacement_data_dir / cache_file)
        else:
            ep_start_end_ids = np.load(self.abs_points_displacement_data_dir / index_file)
            for start_idx, end_idx in tqdm(ep_start_end_ids, desc="Preloading points displacement dataset"):
                for idx in range(start_idx, end_idx):
                    file_name = self._get_episode_name_p_disp(idx)
                    np_obj = load_npz(file_name)
                    data = {key: np.stack([np_obj[key]]) for key, _ in np_obj.items()}

                    value = {
                        "rgb_static": np.squeeze(data["rgb_static"]),
                        "rgb_gripper": np.squeeze(data["rgb_gripper"]),
                    }
                    self.preloaded_data_p_disp[idx] = value
            logger.info("Preloaded the points displacement dataset into cache.")

            with open(self.abs_points_displacement_data_dir / cache_file, "wb") as f:
                pickle.dump(self.preloaded_data_p_disp, f)
            logger.info(f"Saved preloaded points displacement dataset to {cache_file}")


    def _get_episode_bounds(self, frame_idx: int) -> tuple[int, int]:
        """Return inclusive (start, end) frame indices of the episode containing ``frame_idx``."""
        i = bisect_right(self.start_ids, frame_idx) - 1
        if i < 0:
            return int(self.start_ids[0]), int(self.end_ids[0])
        return int(self.start_ids[i]), int(self.end_ids[i])

    def zip_sequence(
        self,
        start_idx: int,
        end_idx: int,
        action_chunk_size: int,
        ep_start_idx: Optional[int] = None,
        ep_end_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if ep_start_idx is None or ep_end_idx is None:
            ep_start_idx, ep_end_idx = self._get_episode_bounds(start_idx)
        requested = list(range(start_idx, end_idx, action_chunk_size))
        # Hold the first/last real frame when the requested window is outside the episode.
        file_indices = [min(max(file_idx, ep_start_idx), ep_end_idx) for file_idx in requested]

        points_disp = None
        if not self.use_cached_data:
            episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in file_indices]
            pre_action_chunks = [self.load_file(self._get_episode_name_action_chunk(file_idx)) for file_idx in file_indices]
            if self.points_displacement:
                points_disp = [self.load_file(self._get_episode_name_p_disp(file_idx)) for file_idx in file_indices]
        else:
            episodes = [self._cached(self.preloaded_data, file_idx, self._get_episode_name) for file_idx in file_indices]
            pre_action_chunks = [self._cached(self.preloaded_data_action_chunk, file_idx, self._get_episode_name_action_chunk) for file_idx in file_indices]
            if self.points_displacement:
                points_disp = [self._cached(self.preloaded_data_p_disp, file_idx, self._get_episode_name_p_disp) for file_idx in file_indices]

        for i, file_idx in enumerate(requested):
            if file_idx == file_indices[i]:
                continue
            src = np.array(pre_action_chunks[i]["rel_actions"], copy=True).reshape(self.action_chunk_size, -1)
            held = np.zeros_like(src)
            held[:, -1] = 1.0 if src[-1, -1] >= 0 else -1.0
            pre_action_chunks[i] = {"rel_actions": held.reshape(-1)}

        return episodes, pre_action_chunks, points_disp

    def _cached(self, cache: Dict[int, Dict[str, np.ndarray]], file_idx: int, name_fn) -> Dict[str, np.ndarray]:
        item = cache.get(file_idx)
        if item is not None:
            return item
        np_obj = self.load_file(name_fn(file_idx))
        item = {k: np.squeeze(np.stack([v])) for k, v in np_obj.items()}
        cache[file_idx] = item
        return item
