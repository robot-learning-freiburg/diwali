"""


---
Train an action decoder directly from precomputed features.pkl (no world model needed).

Supports two conditioning variants:
  - lat_actions + features  (ActionDecoderFromFeats)
  - lat_actions only        (ActionDecoderFromFeatsLatOnly)

"""
import logging
import os
from pathlib import Path

import hydra
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer

from lumos.utils.info_utils import (
    print_system_env_info,
    setup_callbacks,
    setup_logger,
    setup_tensor_cores,
)

os.environ["HYDRA_FULL_ERROR"] = "1"

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="train_action_decoder_from_feats")
def train(cfg: DictConfig) -> None:
    cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    model_dir = Path(cfg.exp_dir) / "model_weights/"
    cfg.callbacks.checkpoint.dirpath = model_dir
    os.makedirs(model_dir, exist_ok=True)

    log_rank_0(f"Training action decoder from feats with config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())

    seed_everything(cfg.seed, workers=True)

    if setup_tensor_cores(enabled=cfg.tc_enabled, precision=cfg.tc_precision):
        log_rank_0("Tensor Cores detected and configured for optimal performance")
        log_rank_0(f"Using matmul precision: {cfg.tc_precision}")

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.world_model)

    trainer_args = {**cfg.trainer, "logger": setup_logger(cfg), "callbacks": setup_callbacks(cfg.callbacks)}
    trainer = Trainer(**trainer_args)

    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    train()
