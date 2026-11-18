"""
Modified from https://github.com/irom-princeton/dppo
"""

"""
Parent pre-training agent class.

"""

from copy import deepcopy
import logging
import os
import random

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

import wandb

log = logging.getLogger(__name__)
from diwali.env.utils import make_calvin_env
from diwali.utils.scheduler import CosineAnnealingWarmupRestarts

DEVICE = "cuda:0"


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch, device="cuda:0"):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)


def build_dataset(dataset_cfg):
    """
    Combine datasets as in lumos
    """
    if "_target_" in dataset_cfg:
        return hydra.utils.instantiate(dataset_cfg)

    datasets = []
    names = []
    for name, sub_cfg in dataset_cfg.items():
        ds = hydra.utils.instantiate(sub_cfg)
        log.info(f"Dataset '{name}' ({sub_cfg.dataset_path}): {len(ds)} samples")
        datasets.append(ds)
        names.append(name)
    concat = torch.utils.data.ConcatDataset(datasets)
    concat.device = datasets[0].device
    concat.names = names
    log.info(f"Combined {len(datasets)} datasets into {len(concat)} total samples")
    return concat


def _maybe_instantiate_wm(cfg):
    if "wm" not in cfg:
        return None
    ckpt = cfg.wm.get("ckpt_path", None)
    if ckpt in {None, "", "null", "None"}:
        log.warning("wm.ckpt_path is not set; skipping world model.")
        return None
    if not os.path.isfile(str(ckpt)):
        log.warning("wm checkpoint %s does not exist; skipping world model.", ckpt)
        return None
    return hydra.utils.instantiate(cfg.wm)


class EMA:
    """
    Empirical moving average

    """

    def __init__(self, cfg):
        super().__init__()
        self.beta = cfg.decay

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class PreTrainAgent:
    def __init__(self, cfg):
        super().__init__()
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        DEVICE = cfg.get("device", "cuda:0")
        # Wandb
        self.use_wandb = cfg.wandb is not None
        if cfg.wandb is not None:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=cfg.wandb.dir,
            )

        # WM wrapper
        self.wme = _maybe_instantiate_wm(cfg)

        if "action_decoder" in cfg and cfg.action_decoder is not None: 
            from diwali.wm.models.action_decoder_from_feats import ActionDecoderFromFeats
            decoder_path = cfg.action_decoder.get("ckpt_path", None)
            if decoder_path in {None, "", "null", "None"} or not os.path.isfile(str(decoder_path)):
                self.action_decoder = None
                print("NO ACTION DECODER")
            else:
                self.action_decoder = ActionDecoderFromFeats.load_from_checkpoint(decoder_path, map_location="cpu", weights_only=False)
                self.action_decoder.to(DEVICE).eval()
                print("ACTION DECODER WAS LOADED")
        else:
            self.action_decoder = None
            print("NO ACTION DECODER")

        self.env_name = cfg.get("env_name", None)
        self.env_type = cfg.get("env_type", None)

        # Make vectorized env
        if self.env_type == "calvin":
            self.venv, self.env = make_calvin_env(
                cfg.env.name,
                env_type=self.env_type,
                num_envs=cfg.env.n_envs,
                max_episode_steps=cfg.env.max_episode_steps,
                calvin_env_cfg=cfg.get("env_cfg", None),
                normalization_path=cfg.get("normalization_path", None),
                device=cfg.get("device", "cuda:0"),
                n_obs_steps=cfg.env.get("n_obs_steps", 1),
                n_action_steps=cfg.env.get("n_action_steps", 4),
                load_scene_from_dataset=cfg.env.get("load_scene_from_dataset", False),
                eval_scene_from_dataset=cfg.env.get("eval_scene_from_dataset", False),
                offline_method=cfg.get("offline_method", False),
                rgb_obs=cfg.get("rgb_obs", False),
                hybrid_obs=cfg.get("hybrid_obs", False),
                stacked_obs=cfg.get("stacked_obs", False),
            )
        elif self.env_type == "real":
            # Real-world eval uses the vision WM as a simulator (see env_runner).
            self.venv = None
            if "env_runner" in cfg:
                self.env = cfg.env
            else:
                self.env = None

        self.n_envs = cfg.env.n_envs
        self.n_render = cfg.env.n_render
        self.render_video = cfg.env.save_video
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.reset_at_iteration = cfg.env.get("reset_at_iteration", True)

        # Build model
        self.model = hydra.utils.instantiate(cfg.model)
        self.ema = EMA(cfg.ema)
        self.ema_model = deepcopy(self.model)

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.update_ema_freq = cfg.train.update_ema_freq
        self.epoch_start_ema = cfg.train.epoch_start_ema
        self.val_freq = cfg.train.get("val_freq", 100)

        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        self.render_dir = os.path.join(self.logdir, "render")
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq

        if self.env_type != "real":
            assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
            assert not (self.n_render <= 0 and self.render_video), "Need to set n_render > 0 if saving video"

        self.dataset_train = build_dataset(cfg.train_dataset_group if "train_dataset_group" in cfg else cfg.train_dataset)
        if self.use_wandb and isinstance(self.dataset_train, torch.utils.data.ConcatDataset):
            wandb.run.summary["dataset/num_sub_datasets"] = len(self.dataset_train.datasets)
            wandb.run.summary["dataset/total_samples"] = len(self.dataset_train)
            for name, sub_ds in zip(self.dataset_train.names, self.dataset_train.datasets):
                wandb.run.summary[f"dataset/{name}_samples"] = len(sub_ds)
        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=4 if self.dataset_train.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset_train.device == "cpu" else False,
        )
        self.dataloader_val = None
        if "train_split" in cfg.train and cfg.train.train_split < 1:
            assert hasattr(self.dataset_train, "set_train_val_split"), (
                "train_split auto-splitting is not supported when train_dataset combines multiple "
                "datasets; use validation_dataset instead."
            )
            val_indices = self.dataset_train.set_train_val_split(cfg.train.train_split)
            self.dataset_val = deepcopy(self.dataset_train)
            self.dataset_val.set_indices(val_indices)
            self.dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=4 if self.dataset_val.device == "cpu" else 0,
                shuffle=True,
                pin_memory=True if self.dataset_val.device == "cpu" else False,
            )
        elif "validation_dataset_group" in cfg or "validation_dataset" in cfg:
            self.dataset_val = build_dataset(cfg.validation_dataset_group if "validation_dataset_group" in cfg else cfg.validation_dataset)
            self.dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=4 if self.dataset_val.device == "cpu" else 0,
                shuffle=True,
                pin_memory=True if self.dataset_val.device == "cpu" else False,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.learning_rate,
            min_lr=cfg.train.lr_scheduler.min_lr,
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.reset_parameters()

    def run(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save_model(self):
        """
        saves model and ema to disk;
        """
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")
        if self.use_wandb and self.epoch == self.n_epochs:
            wandb.run.summary["checkpoint/path"] = savepath
            artifact = wandb.Artifact(
                f"{self.env_name}-pretrained-policy", type="model",
                metadata={"epoch": self.epoch, "seed": self.seed},
            )
            artifact.add_file(savepath)
            wandb.log_artifact(artifact, aliases=["latest", f"epoch-{self.epoch}"])

    def load(self, epoch):
        """
        loads model and ema from disk
        """
        loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
        data = torch.load(loadpath, weights_only=True)

        self.epoch = data["epoch"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
