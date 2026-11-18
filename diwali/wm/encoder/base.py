from pathlib import Path

import torch


class BaseWMObsEncoder:
    def __init__(self, ckpt_path, device, stats_path, wm_type):
        assert wm_type in ["vis", "diwali", "clam", "clamx"], "Invalid world model type specified."
        assert Path(ckpt_path).exists(), f"Checkpoint path {ckpt_path} does not exist."
        assert Path(stats_path).exists(), f"Stats path {stats_path} does not exist."
        print(f"wm_type: {wm_type}")
        if wm_type == "vis":
            from diwali.wm.models.dreamer_v2 import DreamerV2
        elif wm_type == "diwali":
            from diwali.wm.models.dreamer_v2_ofpf import DreamerV2
        elif wm_type == "clam" or wm_type == "clamx":
            from diwali.wm.models.dreamer_v2_clam import DreamerV2

        self.wm = DreamerV2.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
        self.wm.requires_grad_(False)
        self.wm.eval()
        self.wm.to(device)

        self.stats_path = stats_path
        self.device = device

    def decode_latent(self, latent):
        """Decode RSSM features to uint8 HWC images (static, gripper)."""
        dcd_img_s, dcd_img_g = self.wm.decoder(latent)
        _, rgb_unnorm = self.get_unnormalizers(latent.device)
        rgb_static = rgb_unnorm(dcd_img_s.permute(0, 2, 3, 1))
        rgb_gripper = None
        if dcd_img_g is not None:
            rgb_gripper = rgb_unnorm(dcd_img_g.permute(0, 2, 3, 1))
        return rgb_static, rgb_gripper

    def wm_step(self, latent, action, temperature=1.0):
        """Imagine one RSSM step. `latent` is concatenated (h, z); `action` is (B, action_dim)."""
        deter_dim = self.wm.rssm_core.cell.deter_dim
        h, z = latent.split([deter_dim, latent.shape[-1] - deter_dim], -1)
        _, (h, z) = self.wm.dream(action, (h, z), temperature=temperature)
        return torch.cat((h, z), dim=-1)

    def get_normalizers(self, device):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_unnormalizers(self, device):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_zero_wm_features(self, obs, device):
        """
        Given environment's initial observations, get features from the world model.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_hist_wm_features(self, obs, action, prev_done, in_state, device):
        """
        Given environment's history of observations, actions taken and done flags,
        get features from the world model.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")
