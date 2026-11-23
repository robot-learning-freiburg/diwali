import logging
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Set
from copy import deepcopy

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
import torchvision

from lumos.datasets.utils.episode_utils import load_dataset_statistics
from lumos.utils.nn_utils import (
    transpose_collate_ag,
    transpose_collate_hybrid_wm,
    transpose_collate_state_wm,
    transpose_collate_wm,
)

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})


class GroupDataModule(pl.LightningDataModule):
    """Group datamodule takes data from a list of folders with the same structure."""
    def __init__(
        self,
        datasets: DictConfig,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        batch_sampler: DictConfig = None,
        observation_space: DictConfig = None,
        shuffle_val: bool = False,
        load_feats: bool = False,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        num_workers: int = 8,
        action_chunk_size: int = 1,
        **kwargs: Dict,
    ):
        """Initialize GroupDataModule.

        datasets:
            Configuration for the dataset. Not the real dataset. The real dataset depends on the provided root_data_dir.
        training_repo_root:
            Path to the root of the training repository. Left none since we are using root_data_dir.
        transforms:
            Transformations to apply to the data.
        batch_sampler:
            Configuration for the batch sampler.
        """
        super().__init__()
        self.robot_name_ids = dict(enumerate(datasets.keys()))
        self.robot_nums = len(self.robot_name_ids)
        self.single_robot = self.robot_nums == 1
        self.datasets_cfg = [val for val in datasets.values()]
        self.train_datasets = None
        self.val_datasets = None

        self.shuffle_val = shuffle_val
        self.modalities: Set[str] = set()
        self.transforms = [deepcopy(transforms) for _ in range(self.robot_nums)]
        self.batch_sampler = batch_sampler
        self.load_feats = load_feats
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.action_chunk_size = action_chunk_size

        key_name = [cfg.train.key for cfg in self.datasets_cfg]
        if all(k == "vis" for k in key_name):
            self.collate_fn = transpose_collate_wm
        elif all(k == "hybrid" for k in key_name):
            self.collate_fn = transpose_collate_hybrid_wm
        elif all(k == "state" for k in key_name):
            self.collate_fn = transpose_collate_state_wm
        else:
            raise NotImplementedError(f"All dataset keys must be same and from ['vis', 'hybrid', 'state']. Got {key_name}")

    def _compute_batch_params(self):
        """Return (num_replicas, global_batch_size)."""
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size, self.train_batch_size * world_size, self.val_batch_size * world_size

    def setup(self, stage=None):
        # instantiate single transforms and compose into one
        # # update dataset mean and var. Only keys in the transform will be updated. New terms in the loaded yaml will not be added.
        # transforms = [load_dataset_statistics(self.train_dir[id], self.val_dir[id], self.transforms[id]) for id in range(self.robot_nums)]
        transforms = self.transforms # temporarily disabled since no need to merge
        train_transforms = [{k: [hydra.utils.instantiate(tf) for tf in tfs] for k, tfs in rob.train.items()} for rob in transforms]
        val_transforms = [{k: [hydra.utils.instantiate(tf) for tf in tfs] for k, tfs in rob.val.items()} for rob in transforms]
        self.train_transforms = [{k: torchvision.transforms.Compose(v) for k, v in tf.items()} for tf in train_transforms]
        self.val_transforms = [{k: torchvision.transforms.Compose(v) for k, v in tf.items()} for tf in val_transforms]

        self.train_datasets, self.val_datasets = {}, {}

        # # load feature datasets if specified
        # if self.load_feats:
        #     train_features = [self.load_latent_dataset(dir / "cached_feats.pkl") for dir in self.train_dir]
        #     val_features = [self.load_latent_dataset(dir / "cached_feats.pkl") for dir in self.val_dir]

        # consider vis only for now
        train_dataset = [
            hydra.utils.instantiate(
                self.datasets_cfg[idx].train,
                robot_id=idx,
                transforms=self.train_transforms[idx],
                action_chunk_size=self.action_chunk_size,
            ) 
            for idx in range(self.robot_nums)
        ]
        val_dataset = [
            hydra.utils.instantiate(
                self.datasets_cfg[idx].val, 
                robot_id=idx, 
                transforms=self.val_transforms[idx],
                action_chunk_size=self.action_chunk_size,
            ) 
            for idx in range(self.robot_nums)
        ]
        self.pseudo_masked_val_act = [cfg.pseudo_masked_val_act for cfg in self.datasets_cfg]

        # if self.load_feats:
        #     for idx, sub_dataset in enumerate(train_dataset):
        #         sub_dataset.setup_features(train_features[idx])
        #     for idx, sub_dataset in enumerate(val_dataset):
        #         sub_dataset.setup_features(val_features[idx])
        
        self.train_datasets["vis"] = ConcatDataset(train_dataset) if len(train_dataset) > 1 else train_dataset[0]
        self.val_datasets["vis"] = ConcatDataset(val_dataset) if len(val_dataset) > 1 else val_dataset[0]
        self.modalities.add("vis")

        # for _, dataset in self.datasets_cfg.items():
        #     train_dataset = hydra.utils.instantiate(dataset, datasets_dir=self.train_dir, transforms=self.train_transforms)
        #     val_dataset = hydra.utils.instantiate(dataset, datasets_dir=self.val_dir, transforms=self.val_transforms)

        #     if self.load_feats:
        #         train_dataset.setup_features(train_features)
        #         val_dataset.setup_features(val_features)

        #     key = dataset.key
        #     self.train_datasets[key] = train_dataset
        #     self.val_datasets[key] = val_dataset
        #     self.modalities.append(key)

    def train_dataloader(self):
        num_replicas, global_train_batch_size, global_val_batch_size = self._compute_batch_params()

        if self.load_feats:
            collate = transpose_collate_ag
        else:
            collate = self.collate_fn
        if self.batch_sampler:
            dataloaders = {}
            for key, dataset in self.train_datasets.items():
                dataset_length = len(dataset)
                print(f"Dataset '{key}' has {dataset_length} items.")  # Or use another form of logging if preferred

                if num_replicas > 1:
                    self.batch_sampler.data_size = dataset_length
                    self.batch_sampler.global_batch_size = global_train_batch_size
                    self.batch_sampler.num_replicas = num_replicas
                else:
                    self.batch_sampler.data_size = dataset_length
                    self.batch_sampler.batch_size = self.train_batch_size
                batch_sampler = hydra.utils.instantiate(self.batch_sampler)

                dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=collate,
                    persistent_workers=True,
                )
                dataloaders[key] = dataloader
            return dataloaders
        else:
            return {
                key: DataLoader(
                    dataset,
                    batch_size=self.train_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=False,
                    collate_fn=collate,
                    persistent_workers=False,
                )
                for key, dataset in self.train_datasets.items()
            }

    def val_dataloader(self):
        num_replicas, global_train_batch_size, global_val_batch_size = self._compute_batch_params()

        if self.load_feats:
            collate = transpose_collate_ag
        else:
            collate = self.collate_fn
        if self.batch_sampler:
            val_dataloaders = {}
            for key, dataset in self.val_datasets.items():
                dataset_length = len(dataset)
                print(f"Dataset '{key}' has {dataset_length} items.")  # Or use another form of logging if preferred

                if num_replicas > 1:
                    self.batch_sampler.data_size = dataset_length
                    self.batch_sampler.global_batch_size = global_val_batch_size
                    self.batch_sampler.num_replicas = num_replicas
                else:
                    self.batch_sampler.data_size = dataset_length
                    self.batch_sampler.batch_size = self.val_batch_size
                batch_sampler = hydra.utils.instantiate(self.batch_sampler)

                dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=self.num_workers,
                    pin_memory=False,
                    shuffle=self.shuffle_val,
                    collate_fn=collate,
                    persistent_workers=False,
                )
                val_dataloaders[key] = dataloader
            return val_dataloaders
        else:
            return {
                key: DataLoader(
                    dataset,
                    batch_size=self.val_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    shuffle=self.shuffle_val,
                    collate_fn=collate,
                    persistent_workers=True,
                )
                for key, dataset in self.val_datasets.items()
            }

    def load_latent_dataset(self, features_path):
        if features_path.is_file():
            logger.info("Loading preloaded features from cache...")
            with open(str(features_path), "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("Cached features not found. Generate features via featurizer.py")
