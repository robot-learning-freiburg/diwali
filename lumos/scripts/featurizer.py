
import logging
import os
from pathlib import Path
import pickle
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_only
import torch
from tqdm import tqdm

from lumos.utils.info_utils import print_system_env_info

# To use the mul resolver in the config file
OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="featurizer/featurizer")
def featurizer(cfg: DictConfig) -> None:
    """
    This is called to calculate features for a dataset under a loaded world model
    Args:
        cfg: hydra config
    """
    if cfg.exp_dir is None:
        raise ValueError("The path to the world model is not given.")
    chk = Path(cfg.exp_dir)
    log_rank_0(f"checkpoint:{chk.as_posix()}")
    log_rank_0(f"Extracting features with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())

    seed_everything(cfg.seed, workers=True)

    if cfg.world_model.name == "dreamer_v2":
        from lumos.world_models.dreamer_v2 import DreamerV2
    elif cfg.world_model.name == "dreamer_v2_ofpf":
        from lumos.world_models.dreamer_v2_ofpf import DreamerV2
    else:
        raise NotImplementedError(f"Unknown model: {cfg.world_model.name}")

    world_model = DreamerV2.load_from_checkpoint(chk.as_posix(), map_location="cpu", weights_only=False).to(cfg.device)
    world_model.eval()
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    logger.info("Extracting features from the training dataset.")
    save_dir = Path(cfg.output_dir) / "training"
    save_dir.mkdir(parents=True, exist_ok=True)
    extract_features(world_model, datamodule.train_dataloader(), datamodule.train_datasets["vis"], cfg, save_dir)
    logger.info("Extracting features from the validation dataset.")
    save_dir = Path(cfg.output_dir) / "validation"
    save_dir.mkdir(parents=True, exist_ok=True)
    extract_features(world_model, datamodule.val_dataloader(), datamodule.val_datasets["vis"], cfg, save_dir)
    logger.info("Finished extracting features.")

def extract_features(world_model, data_loader, dataset, cfg, save_dir: str):
    """
    Extracts features from the given model and data_loader
    Args:
        world_model: The world model
        data_loader: The data loader for the dataset
        dataset: The dataset to extract features from
        cfg: The configuration for the extraction
    """
    model_has_latent_action_head = "la" in cfg.world_model.name or "of" in cfg.world_model.name or "pf" in cfg.world_model.name or "ofpf" in cfg.world_model.name or "laf" in cfg.world_model.name
    # rel_action ablation mode has no latent action space
    use_latact = model_has_latent_action_head and cfg.world_model.get("mode", None) != "rel_action"

    #### set up empty arrays #####
    dim_features = world_model.decoder.in_dim
    feats = np.zeros((len(dataset), dim_features), dtype=np.float32)
    pre_acts = np.zeros((len(dataset), cfg.datamodule.action_space), dtype=np.float32)
    resets = np.zeros((len(dataset), 1), dtype=bool)
    frames = np.zeros((len(dataset), 1), dtype=int)
    if use_latact:
        lat_acts = np.zeros((len(dataset), cfg.world_model.latent_action_dim), dtype=np.float32)
    robot_obs_arr = None
    data_dict = {}

    #### loop to ensure every episode is featurized from the beginning.")
    max_episode_len = max([data_loader["vis"].dataset.datasets[i].max_episode_len for i in range(len(data_loader["vis"].dataset.datasets))])
    EPOCH = np.ceil(max_episode_len / data_loader["vis"].batch_sampler.chunk_size).astype(int)
    # EPOCH = 4
    for epoch in range(EPOCH):
        #### first epoch start with zeros. It will be replaced later by the correct values.
        if epoch == 0:
            in_state = world_model.rssm_core.init_state(data_loader["vis"].batch_sampler.batch_size)
        else:
            in_state = [torch.roll(x, 1, 0) for x in in_state]
            in_state[0][0] = torch.zeros_like(in_state[0][0])
            in_state[1][0] = torch.zeros_like(in_state[1][0])

        loader = tqdm(data_loader["vis"], desc=f"Epoch {epoch+1}/{EPOCH}")

        for i, batch in enumerate(loader):
            # Reset at starts of chunked episodes (indices < action_chunk_size)
            batch["reset"][0][batch["idx"] < cfg.datamodule.action_chunk_size] = True

            features, out_state = world_model.infer_features_noact(
                batch["rgb_obs"]["rgb_static"],
                batch["reset"],
                in_state,
                batch["rgb_obs"]["rgb_gripper"],
            )

            in_state = out_state
            idxs = batch["idx"].cpu().numpy()
            feats[idxs] = features.cpu().numpy().squeeze(0)
            pre_acts[idxs] = batch["actions"]["pre_actions"].cpu().numpy().squeeze(0)
            frames[idxs] = batch["frame"].cpu().numpy().squeeze(0)
            if use_latact:
                lat_act = world_model.infer_action(batch["rgb_obs"]["rgb_static"], batch["rgb_obs"]["rgb_gripper"])
                lat_acts[idxs] = lat_act.cpu().numpy().squeeze(0)


            for idx in idxs:
                data_dict[int(frames[idx])] = {
                    "features": feats[idx],
                    "pre_actions": pre_acts[idx],
                }
                if use_latact:
                    data_dict[int(frames[idx])]["lat_actions"] = lat_acts[idx]

    # Shift rel_act and lat_actions by +chunk so they encode POST actions instead of PRE actions.
    if model_has_latent_action_head:
        action_chunk_size = cfg.datamodule.action_chunk_size
        ep_start_end_ids = np.load(dataset.datasets[0].abs_dataset_dir / "ep_start_end_ids.npy")
        frame_set = set(data_dict.keys())
        for ep_start, ep_end in ep_start_end_ids:
            if use_latact:
                for f in range(int(ep_start), int(ep_end) + 1):
                    if f not in frame_set:
                        continue
                    f_next = f + action_chunk_size
                    if f_next <= int(ep_end) and f_next in frame_set:
                        data_dict[f]["lat_actions"] = data_dict[f_next]["lat_actions"].copy()
                    else:
                        data_dict[f]["lat_actions"] = np.zeros_like(data_dict[f]["lat_actions"])

            action_chunk = np.zeros((action_chunk_size, 7))
            if int(ep_end) in frame_set:
                data_dict[int(ep_end)]["rel_actions_chunk"] = action_chunk.reshape(-1).copy()
            for ep_idx in range(ep_end - 1, ep_start - 1, -1):
                if ep_idx not in frame_set:
                    continue
                action_chunk = np.roll(action_chunk, shift=1, axis=0)
                if ep_idx + 1 in frame_set:
                    action_chunk[0] = data_dict[ep_idx + 1]["pre_actions"][-7:].copy()
                else:
                    action_chunk[0] = np.zeros(7)
                data_dict[ep_idx]["rel_actions_chunk"] = action_chunk.reshape(-1).copy()

    cached_feats_path = save_dir / cfg.output_file
    with open(str(cached_feats_path), "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def obs_to_zero_feature(wm, rgb_s, rgb_g, action_space, proprio=None):
    bz = rgb_s.size(1)
    zero_action = torch.zeros((1, bz, action_space)).to(wm.device)
    zero_action[:, :, -1] = 1.0
    true_reset = torch.ones((1, bz, 1), dtype=torch.bool).to(wm.device)
    features, _ = wm.infer_features(rgb_s, zero_action, true_reset, wm.rssm_core.init_state(bz), rgb_g, proprio)
    return features


@rank_zero_only
def log_rank_0(*args, **kwargs):
    """
    Log the information using the logger at rank 0.
    """
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    featurizer()
