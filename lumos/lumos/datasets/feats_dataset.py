
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FeatsDataset(Dataset):
    """Dataset that loads precomputed features from a features.pkl cache.

    Expects features.pkl to be a dict keyed by frame index with entries:
        {frame_id: {"lat_actions": ndarray, "features": ndarray,
                    "pre_actions": ndarray}}

    Args:
        feats_path: Path to features.pkl.
        index_file: Path to ep_start_end_ids.npy (episode boundary file).
        seq_len: Number of timesteps per sample.
        with_features: If False, "features" key is omitted (lat-only variant).
    """

    def __init__(self, feats_path: str, index_file: str, seq_len: int, with_features: bool = True):
        feats_path = Path(feats_path)
        logger.info(f"Loading features from {feats_path}")
        with open(feats_path, "rb") as f:
            raw: Dict = pickle.load(f)

        frame_ids: List[int] = sorted(raw.keys())
        logger.info(f"Found features.pkl with {len(frame_ids)} frames.")

        self.lat_actions = np.stack([raw[f]["lat_actions"] for f in frame_ids])
        self.pre_actions = np.stack([raw[f]["pre_actions"] for f in frame_ids])

        self.with_features = with_features
        if with_features:
            self.features = np.stack([raw[f]["features"] for f in frame_ids])

        self.robot_obs = None
        if "robot_obs" in raw[frame_ids[0]]:
            self.robot_obs = np.stack([raw[f]["robot_obs"] for f in frame_ids])

        self.rel_act = None
        if "rel_actions_chunk" in raw[frame_ids[0]]:
            zero = np.zeros_like(raw[frame_ids[0]]["rel_actions_chunk"])
            self.rel_act = np.stack([raw[f].get("rel_actions_chunk", zero) for f in frame_ids])

        frame_to_pos = {frame_id: pos for pos, frame_id in enumerate(frame_ids)}

        self.seq_len = seq_len
        self.episode_lookup, self.start_ids = self._build_file_indices(index_file, frame_ids, frame_to_pos)
        logger.info(f"Total number of windows: {len(self.episode_lookup)}")

    def _build_file_indices(self, index_file: str, frame_ids: List[int], frame_to_pos: Dict[int, int]) -> Tuple[np.ndarray, List[int]]:
        import bisect
        ep_start_end_ids = np.load(index_file)
        logger.info(f"Found {Path(index_file).name} with {len(ep_start_end_ids)} episodes.")
        assert len(ep_start_end_ids) > 0, "No episodes found in the index file."

        episode_lookup = []
        start_ids = []

        index_total = sum(int(e) - int(s) + 1 for s, e in ep_start_end_ids)
        logger.info(f"Total frames in index file: {index_total}")

        for start_frame, end_frame in ep_start_end_ids:
            if start_frame not in frame_to_pos:
                continue
            # clamp end_frame to the last available frame in features.pkl within this episode
            end_pos_in_list = bisect.bisect_right(frame_ids, int(end_frame)) - 1
            end_frame = frame_ids[end_pos_in_list]
            if end_frame < start_frame:
                continue
            start_pos = frame_to_pos[start_frame]
            end_pos = frame_to_pos[end_frame]
            end_pos_sample = end_pos - self.seq_len
            if end_pos_sample < start_pos:
                logger.info(f"Episode from {start_frame} to {end_frame} is too short. Skipping.")
                continue
            start_ids.append(start_pos)
            episode_lookup.extend(range(start_pos, end_pos_sample + 1))

        return np.array(episode_lookup), start_ids

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.episode_lookup[idx]
        end = start + self.seq_len

        item = {
            "lat_actions": torch.tensor(self.lat_actions[start:end], dtype=torch.float32),
            "pre_actions": torch.tensor(self.pre_actions[start:end], dtype=torch.float32),
        }
        if self.with_features:
            item["features"] = torch.tensor(self.features[start:end], dtype=torch.float32)
        if self.robot_obs is not None:
            item["robot_obs"] = torch.tensor(self.robot_obs[start:end], dtype=torch.float32)
        if self.rel_act is not None:
            item["rel_actions_chunk"] = torch.tensor(self.rel_act[start:end], dtype=torch.float32)
        return item
