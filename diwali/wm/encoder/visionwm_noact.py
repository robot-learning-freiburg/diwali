import copy

import numpy as np
import torch

from diwali.wm.encoder.base import BaseWMObsEncoder
from diwali.wm.utils import (
    get_rgb_normalizer,
    get_robot_obs_normalizer,
    get_rgb_unnormalizer,
    transpose_tensor,
)


class VisionWMObsEncoder(BaseWMObsEncoder):
    """VisionWMObsEncoder for dreamer_v2_pd checkpoints (separate static + gripper encoders).

    dreamer_v2_pd ignores the `actions` arg and computes lat_act internally from
    consecutive frame pairs via inv_dyna(embed_t, embed_{t+1}). For online eval we
    therefore keep the previous frame and feed 2-frame sequences to infer_features.
    """

    def __init__(self, ckpt_path, device, stats_path, action_space: int = 56, wm_type: str = "diwali"):
        if wm_type in ["LANOF", "OF", "PF", "OFPF"]:
            wm_type = "diwali"
        else:
            wm_type = wm_type.lower()
        super().__init__(ckpt_path, device, stats_path, wm_type)
        self._prev_rgb_static = None
        self._prev_rgb_gripper = None
        self._feat_dim = self.wm.decoder.in_dim

    def get_normalizers(self, device):
        r_obs_norm = get_robot_obs_normalizer(self.stats_path, device)
        rgb_obs_norm = get_rgb_normalizer(device)
        return r_obs_norm, rgb_obs_norm

    def get_unnormalizers(self, device):
        rgb_obs_unnorm = get_rgb_unnormalizer(device)
        return None, rgb_obs_unnorm

    def _normalize_rgb(self, obs, device, rgb_norm):
        """Return (rgb_static, rgb_gripper) as (1, B, C, H, W) tensors, normalized."""
        B = obs["robot_obs"].shape[0]
        rgb_static = torch.from_numpy(obs["rgb_static"]).float().to(device).squeeze()   
        rgb_gripper = torch.from_numpy(obs["rgb_gripper"]).float().to(device).squeeze()   
        if B == 1:
            rgb_static = rgb_static.unsqueeze(0)
            rgb_gripper = rgb_gripper.unsqueeze(0)
        rgb_static = rgb_norm(rgb_static).unsqueeze(1)   # (B, 1, C, H, W)
        rgb_gripper = rgb_norm(rgb_gripper).unsqueeze(1)
        rgb_static = transpose_tensor(rgb_static)         # (1, B, C, H, W)
        rgb_gripper = transpose_tensor(rgb_gripper)
        return rgb_static, rgb_gripper

    def get_zero_wm_features(self, obs, device):
        """Run WM on (frame_0, frame_0) pair to get a properly-formatted out_state."""
        B = obs["robot_obs"].shape[0]
        _, rgb_norm = self.get_normalizers(device)
        self._prev_rgb_static, self._prev_rgb_gripper = self._normalize_rgb(obs, device, rgb_norm)

        # Duplicate frame so inv_dyna can run (no motion → lat_act ≈ 0)
        rgb_static_pair = torch.cat([self._prev_rgb_static, self._prev_rgb_static], dim=0)
        rgb_gripper_pair = torch.cat([self._prev_rgb_gripper, self._prev_rgb_gripper], dim=0)

        reset = torch.ones(1, B, 1).bool().to(device)

        features, out_state = self.wm.infer_features_noact(
            rgb_static_pair,
            reset,
            self.wm.rssm_core.init_state(B),
            rgb_gripper_pair,
        )
        return transpose_tensor(features), out_state

    def get_hist_wm_features(self, obs, action, prev_done, in_state, device):
        """Feed (prev_obs, curr_obs) pair so dreamer_v2_pd can compute lat_act."""
        B = obs["robot_obs"].shape[0]
        _, rgb_norm = self.get_normalizers(device)

        if np.sum(prev_done) > 0:
            in_state_copy = copy.deepcopy(in_state)
            for j in range(B):
                if prev_done[j]:
                    h, w = self.wm.rssm_core.init_state(1)
                    in_state_copy[0][j] = h
                    in_state_copy[1][j] = w
            in_state = in_state_copy

        curr_rgb_static, curr_rgb_gripper = self._normalize_rgb(obs, device, rgb_norm)

        # Concatenate (prev, curr) along the sequence dimension → (2, B, C, H, W)
        rgb_static_pair = torch.cat([self._prev_rgb_static, curr_rgb_static], dim=0)
        rgb_gripper_pair = torch.cat([self._prev_rgb_gripper, curr_rgb_gripper], dim=0)

        reset = torch.from_numpy(prev_done).reshape(B, 1, 1).bool().to(device)
        reset = transpose_tensor(reset)  # (1, B, 1) — applies to the transition

        # infer_features returns features for frames [1:], i.e. the current frame only
        features, out_state = self.wm.infer_features_noact(
            rgb_static_pair,
            reset,
            in_state,
            rgb_gripper_pair,
        )

        self._prev_rgb_static = curr_rgb_static
        self._prev_rgb_gripper = curr_rgb_gripper

        return transpose_tensor(features), out_state
