import copy
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import mmh3
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset

from lumos.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_points_disp,
    process_depth,
    process_flow,
    process_language,
    process_rgb,
    process_state,
    optional_path,
    has_obs_keys,
)
from lumos.utils.transforms import UnnormalizeVectorMinMax

logger = logging.getLogger(__name__)


def get_validation_window_size(idx: int, min_window_size: int, max_window_size: int) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1

    hash_val = mmh3.hash(str(idx), seed=0, signed=False)  # unsigned 32-bit
    return min_window_size + (hash_val % window_range)


class BaseDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        dataset_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        dataset_dir: Path,
        action_chunk_data_dir: Path,
        obs_space: DictConfig,
        proprio_state: DictConfig,
        key: str,
        lang_folder: str,
        num_workers: int,
        transforms: Dict = {},
        batch_size: int = 32,
        min_window_size: int = 16,
        max_window_size: int = 32,
        action_chunk_size: int = 1,
        max_pad_size: int = 0,
        aux_lang_loss_window: int = 1,
        for_wm: bool = False,
        reset_prob: float = 0.05,
        robot_id: int = 0,
        points_displacement_data_dir: Optional[Path] = None,
    ):
        self.episode_lookup: np.ndarray
        self.episode_length: np.ndarray
        self.start_ids: List[int]

        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.optical_flow = has_obs_keys(self.observation_space["flow_obs"])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.action_chunk_size = action_chunk_size
        self.min_window_size = min_window_size * action_chunk_size
        self.max_window_size = max_window_size * action_chunk_size
        self.max_pad_size = max_pad_size
        if self.min_window_size > self.max_window_size:
            logger.error(f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}")
            raise ValueError
        
        self.abs_dataset_dir = Path(dataset_dir)
        self.abs_action_chunk_data_dir = Path(action_chunk_data_dir)
        self.abs_points_displacement_data_dir = optional_path(points_displacement_data_dir)
        self.points_displacement = has_obs_keys(self.observation_space["p_disp_obs"]) and self.abs_points_displacement_data_dir is not None
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        self.for_wm = for_wm
        self.robot_id = robot_id
        self.reset_prob = reset_prob

        assert "validation" in self.abs_dataset_dir.as_posix() or "training" in self.abs_dataset_dir.as_posix()
        self.validation = "validation" in self.abs_dataset_dir.as_posix()
        assert self.abs_dataset_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_dataset_dir}")
        logger.info("finished loading dataset")
        assert "validation" in self.abs_action_chunk_data_dir.as_posix() or "training" in self.abs_action_chunk_data_dir.as_posix()
        self.validation_action_chunk = "validation" in self.abs_action_chunk_data_dir.as_posix()
        assert self.abs_action_chunk_data_dir.is_dir()
        logger.info(f"loading action chunk dataset at {self.abs_action_chunk_data_dir}")
        logger.info("finished loading action chunk dataset")
        if self.points_displacement:
            assert self.abs_points_displacement_data_dir is not None, "points displacement data directory must be provided if points displacement observation space is specified"
            assert "validation" in self.abs_points_displacement_data_dir.as_posix() or "training" in self.abs_points_displacement_data_dir.as_posix()
            self.validation_points_disp = "validation" in self.abs_points_displacement_data_dir.as_posix()
            assert self.abs_points_displacement_data_dir.is_dir()
            logger.info(f"loading points displacement dataset at {self.abs_points_displacement_data_dir}")
            logger.info("finished loading points displacement dataset")
        if self.for_wm:
            assert self.min_window_size == self.max_window_size
            self.with_lang = False

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            window_size = self._get_window_size(idx)
        else:
            idx, window_size = idx
        start_id = self.episode_lookup[idx]
        sequence = self._get_sequences(start_id, window_size)
        sequence["idx"] = idx
        # pad_size = self._get_pad_size(sequence) # I think this function is incorrect,different pad size for beginning and middle of episode
        pad_size = self.max_window_size - window_size
        sequence = self._pad_sequence(sequence, pad_size)
        sequence["pad_size"] = pad_size
        return sequence

    def _get_sequences(self, start_id: int, window_size: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            start_id: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """
        episode = self._load_episode(start_id, window_size)
        if self.for_wm:
            # seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
            seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
            seq_flow_obs = process_flow(episode, self.observation_space, self.transforms)
            seq_acts = process_actions(episode, ["pre_actions"], self.transforms)
            if self.points_displacement:
                seq_p_disp = process_points_disp(episode['p_disp'], self.observation_space["p_disp_obs"])
            else:
                seq_p_disp = {"p_disp": {}}
            # info = {}

            seq_reset = {"reset": torch.from_numpy(episode["reset"]).bool()}
            seq_frames = {"frame": torch.from_numpy(episode["frame"])}
            seq_robot_obs = {}
            if episode.get("robot_obs") is not None:
                seq_robot_obs = {"robot_obs": torch.from_numpy(episode["robot_obs"]).float()}

            seq_dict = {
                # **seq_state_obs,
                **seq_rgb_obs,
                **seq_flow_obs,
                **seq_acts,
                **seq_p_disp,
                **seq_robot_obs,
                # **info,
                **seq_reset,
                **seq_frames,
            }  # type:ignore

        else:
            seq_features = {"feature": torch.from_numpy(episode["features"])}
            seq_zero_features = {"zero_feature": torch.from_numpy(episode["zero_features"])}
            seq_acts = process_actions(episode, self.observation_space["actions"], self.transforms)
            if self.points_displacement:
                seq_p_disp = process_points_disp(episode['p_disp'], self.observation_space["p_disp_obs"])
            else:
                seq_p_disp = {"p_disp": {}}
            seq_reset = {"reset": torch.from_numpy(episode["reset"]).bool()}
            # info = get_state_info_dict(episode, self.for_wm)
            seq_lang = process_language(episode, self.transforms, self.with_lang)
            # info = self._add_language_info(info, idx, window_size)
            seq_dict = {
                **seq_features,
                **seq_zero_features,
                **seq_acts,
                **seq_reset,
                # **info,
                **seq_lang,
            }  # type:ignore

        seq_dict["robot_id"] = torch.tensor([self.robot_id], dtype=torch.int32)

        return seq_dict

    def _load_episode(self, start_id: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        max_window = self._get_max_window_size(idx)
        if self.min_window_size == self.max_window_size:
            return max_window
        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    # def _get_max_window_size(self, idx: int) -> int:
    #     """
    #     Get maximum window size for a given starting index.

    #     Args:
    #         idx: Starting index.
    #         N.B. th index refers to the order in the episode lookup list, not the actual frame index.
        
    #     Returns:
    #         Maximum window size.
    #     """
    #     window_diff = self.max_window_size - self.min_window_size

    #     # final_end_id = self.episode_lookup[-1] + min
    #     # idx > final_end_id - max = self.episode_lookup[-1] + min - max
    #     if len(self.episode_lookup) < idx + window_diff:
    #         # last episode
    #         max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
    #     # episode_end_id - max < idx < episode_end_id - min
    #     # all these frames + max - min will land in next episode
    #     elif self.episode_lookup[idx + window_diff] != self.episode_lookup[idx] + window_diff:
    #         steps_to_next_episode = int(np.nonzero(self.episode_lookup[idx : idx + window_diff + 1] - (self.episode_lookup[idx] + np.arange(window_diff + 1)))[0][0])
    #         max_window = min(self.max_window_size, (self.min_window_size + steps_to_next_episode - 1))
    #     else:
    #         max_window = self.max_window_size
    #     return max_window
    
    def _get_max_window_size(self, idx: int) -> int:
        return min(self.max_window_size, self.episode_length[idx])
    
    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"]["rel_actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        if pad_size == 0:
            return seq
        
        if self.for_wm:
            for key in ['rgb_obs', 'depth_obs']:
                seq[key] = {k: self._pad_with_repetition(v, pad_size) for k, v in seq[key].items()}
        else:
            seq.update({"feature": self._pad_with_repetition(seq["feature"], pad_size)})
            seq.update({"zero_feature": self._pad_with_repetition(seq["zero_feature"], pad_size)})

        seq.update({"reset": self._pad_with_random(seq["reset"], pad_size, self.reset_prob)})
        seq.update({"frame": torch.cat([seq["frame"], torch.from_numpy(np.arange(seq["frame"][-1] + 1, seq["frame"][-1] + pad_size + 1, dtype=np.int32)[:, np.newaxis])], dim=0)})
        
        #  todo: find better way of distinguishing rk and play action spaces
        if self.relative_actions:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            # rel actions store original loaded actions
            # pre action is the action label
            seq_acts = torch.cat([
                    self._pad_with_zeros(seq["actions"]["rel_actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"]["rel_actions"][..., -1:], pad_size),
                ], dim=-1,
            )
            seq["actions"].update({"rel_actions": seq_acts})
            if torch.any(torch.isinf(seq["actions"]["pre_actions"])):
                seq["actions"].update({"pre_actions": self._pad_with_infinity(seq["actions"]["pre_actions"], pad_size)})
            else:
                seq_acts = torch.cat([
                        self._pad_with_zeros(seq["actions"]["pre_actions"][..., :-1], pad_size),
                        self._pad_with_repetition(seq["actions"]["pre_actions"][..., -1:], pad_size),
                    ], dim=-1,
                )
                seq["actions"].update({"pre_actions": seq_acts})
        else:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})

        if self.optical_flow:
            seq.update({"flow_obs": {key: self._pad_with_zeros(obs, pad_size) for key, obs in seq["flow_obs"].items()}})

        if self.points_displacement:
            seq.update({"p_disp": {key: self._pad_with_zeros(obs, pad_size) for key, obs in seq["p_disp"].items()}})

        if "robot_obs" in seq:
            seq["robot_obs"] = self._pad_with_repetition(seq["robot_obs"], pad_size)

        # seq.update({"state_info": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["state_info"].items()}})
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    @staticmethod
    def _pad_with_infinity(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a Tensor with INFs.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        inf_vec = torch.full((1, input_tensor.shape[-1]), float('inf'))
        inf_repeated = torch.repeat_interleave(inf_vec, repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, inf_repeated))
        return padded

    def _pad_with_random(self, input_tensor: torch.Tensor, pad_size: int, prob: float) -> torch.Tensor:
        """
        Pad a Tensor with random boolean values based on reset probability.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.
            reset_prob: Probability of a reset (True value).

        Returns:
            Padded Tensor.
        """
        return torch.vstack((input_tensor, torch.rand(pad_size, 1) <= prob))

    def _add_language_info(self, info: Dict, idx: int, window_size: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        # use_for_aux_lang_loss = (
        #     idx + window_size + self.aux_lang_loss_window >= len(self.lang_lookup)
        #     or self.lang_lookup[idx + window_size] < self.lang_lookup[idx + window_size + self.aux_lang_loss_window]
        # )
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize action from [-1, 1] to original action space.
        """
        if "action" not in self.transforms:
            return action
        transform = UnnormalizeVectorMinMax(min=self.transforms["action"]["min"], max=self.transforms["action"]["max"])
        return transform(action)